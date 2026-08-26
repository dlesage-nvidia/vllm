# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
import atexit
import contextlib
import json
import os
import threading
import time
from collections import Counter, deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, Literal

from vllm.logger import init_logger
from vllm.multimodal.image_decoders.nvimagecodec import (
    NVIMAGECODEC_MAX_ENCODED_BYTES,
    validate_nvimagecodec_batch_size,
    validate_nvimagecodec_decoders,
    validate_nvimagecodec_pipeline_depth,
)

from .base import MediaWithBytes

if TYPE_CHECKING:
    from PIL import Image

    from .image import ImageMediaIO

logger = init_logger(__name__)

NVIMAGECODEC_DEFAULT_COALESCE_TIMEOUT_MS = 0.0
NVIMAGECODEC_MAX_COALESCE_TIMEOUT_MS = 1000.0
_QUEUE_DEPTH_MULTIPLIER = 4


def validate_nvimagecodec_coalesce_timeout_ms(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("coalesce_timeout_ms must be a non-negative number")
    timeout_ms = float(value)
    if not 0 <= timeout_ms <= NVIMAGECODEC_MAX_COALESCE_TIMEOUT_MS:
        raise ValueError(
            "coalesce_timeout_ms must be between 0 and "
            f"{NVIMAGECODEC_MAX_COALESCE_TIMEOUT_MS:g}"
        )
    return timeout_ms


class NvImageCodecQueueFullError(RuntimeError):
    """The bounded nvImageCodec decode service cannot accept more work."""


@dataclass(frozen=True)
class ImageDecodeServiceConfig:
    decoders: int
    batch_size: int
    pipeline_depth: int
    coalesce_timeout_ms: float
    max_pending_items: int
    max_pending_encoded_bytes: int

    @classmethod
    def from_image_io(cls, image_io: ImageMediaIO) -> ImageDecodeServiceConfig:
        decoders = validate_nvimagecodec_decoders(image_io.decoders)
        batch_size = validate_nvimagecodec_batch_size(image_io.batch_size)
        pipeline_depth = validate_nvimagecodec_pipeline_depth(image_io.pipeline_depth)
        timeout_ms = validate_nvimagecodec_coalesce_timeout_ms(
            image_io.coalesce_timeout_ms
        )
        queue_depth = max(_QUEUE_DEPTH_MULTIPLIER, pipeline_depth)
        return cls(
            decoders=decoders,
            batch_size=batch_size,
            pipeline_depth=pipeline_depth,
            coalesce_timeout_ms=timeout_ms,
            max_pending_items=queue_depth * decoders * batch_size,
            max_pending_encoded_bytes=(
                queue_depth * decoders * NVIMAGECODEC_MAX_ENCODED_BYTES
            ),
        )


@dataclass(frozen=True)
class ImageDecodeSpec:
    image_mode: str | None
    rgba_background_color: tuple[int, int, int]
    semantics_version: int = 1


@dataclass(frozen=True)
class ImageDecodeServiceStats:
    """Cross-request scheduler statistics.

    ``batch_widths`` counts service claim widths. A pipelined claim can contain
    multiple native batches and therefore exceed the configured batch size.
    """

    submitted_images: int
    direct_jobs: int
    batch_widths: dict[int, int]
    queue_wait_seconds: float


@dataclass
class _DecodeJob:
    image_io: ImageMediaIO
    encoded_images: tuple[bytes, ...]
    future: Future[list[MediaWithBytes[Image.Image]]]
    sequence: int
    deadline: float
    spec: ImageDecodeSpec | None
    kind: Literal["coalesced", "direct"]
    accounting: Literal["waiting", "owned", "released"]
    enqueued_at: float

    @property
    def item_count(self) -> int:
        return len(self.encoded_images)

    @property
    def encoded_bytes(self) -> int:
        return sum(map(len, self.encoded_images))


@dataclass(frozen=True)
class _Claim:
    jobs: tuple[_DecodeJob, ...]
    kind: Literal["coalesced", "direct"]


def _close_loaded_images(items: list[MediaWithBytes[Image.Image]]) -> None:
    for item in items:
        close = getattr(item.media, "close", None)
        if close is not None:
            close()


def _dispose_future_result(
    future: Future[list[MediaWithBytes[Image.Image]]],
) -> None:
    try:
        items = future.result()
    except BaseException:
        return
    _close_loaded_images(items)


def _indexed_error(error: BaseException) -> tuple[int, BaseException] | None:
    index = getattr(error, "index", None)
    cause = getattr(error, "error", None)
    if isinstance(index, int) and isinstance(cause, BaseException):
        return index, cause
    return None


def _is_jpeg(data: bytes) -> bool:
    return data.startswith(b"\xff\xd8")


def _decode_spec(image_io: ImageMediaIO) -> ImageDecodeSpec | None:
    unknown_kwargs = set(image_io.kwargs).difference({"rgba_background_color"})
    if unknown_kwargs or (
        image_io.image_mode is not None and not isinstance(image_io.image_mode, str)
    ):
        return None
    spec = ImageDecodeSpec(
        image_mode=image_io.image_mode,
        rgba_background_color=tuple(image_io.rgba_background_color),
    )
    # Admission accounting must not be changed before the compatibility key is
    # known to be usable as a queue key.
    try:
        hash(spec)
    except TypeError:
        return None
    return spec


class NvImageCodecDecodeService:
    """Process-local scheduler for direct and cross-request image batches."""

    def __init__(self, config: ImageDecodeServiceConfig) -> None:
        self.config = config
        self.owner_pid = os.getpid()
        self._cond = threading.Condition()
        self._batch_queues: dict[ImageDecodeSpec, deque[_DecodeJob]] = {}
        self._direct_queue: deque[_DecodeJob] = deque()
        self._admission_waiters: deque[_DecodeJob] = deque()
        self._owned_items = 0
        self._owned_bytes = 0
        self._waiting_items = 0
        self._waiting_bytes = 0
        self._in_flight = 0
        self._sequence = 0
        self._closing = False
        self._shutdown_complete = False
        self._started = False
        self._dispatcher: threading.Thread | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._submitted_images = 0
        self._direct_jobs = 0
        self._batch_widths: Counter[int] = Counter()
        self._queue_wait_seconds = 0.0

    @property
    def started(self) -> bool:
        return self._started

    def _check_pid(self) -> None:
        if self.owner_pid != os.getpid():
            raise RuntimeError(
                "nvImageCodec decode state cannot be reused after fork; "
                "start API workers with the spawn multiprocessing method."
            )

    def _ensure_started_locked(self) -> None:
        if self._started:
            return
        self._executor = ThreadPoolExecutor(
            max_workers=self.config.decoders,
            thread_name_prefix="vllm-nvimagecodec",
        )
        self._dispatcher = threading.Thread(
            target=self._dispatch_loop,
            name="vllm-nvimagecodec-dispatch",
            daemon=True,
        )
        self._started = True
        self._dispatcher.start()

    def _can_own_locked(self, job: _DecodeJob) -> bool:
        items_fit = self._owned_items + job.item_count <= self.config.max_pending_items
        bytes_fit = (
            self._owned_bytes + job.encoded_bytes
            <= self.config.max_pending_encoded_bytes
        )
        return items_fit and bytes_fit

    def _can_wait_locked(self, job: _DecodeJob) -> bool:
        items_fit = (
            self._waiting_items + job.item_count <= self.config.max_pending_items
        )
        bytes_fit = (
            self._waiting_bytes + job.encoded_bytes
            <= self.config.max_pending_encoded_bytes
        )
        return items_fit and bytes_fit

    def _enqueue_owned_locked(self, job: _DecodeJob) -> None:
        if job.kind == "direct":
            self._direct_queue.append(job)
        else:
            assert job.spec is not None
            self._batch_queues.setdefault(job.spec, deque()).append(job)
        job.accounting = "owned"
        self._owned_items += job.item_count
        self._owned_bytes += job.encoded_bytes

    def _enqueue_waiter_locked(self, job: _DecodeJob) -> None:
        self._admission_waiters.append(job)
        job.accounting = "waiting"
        self._waiting_items += job.item_count
        self._waiting_bytes += job.encoded_bytes

    def _release_accounting_locked(self, job: _DecodeJob) -> None:
        if job.accounting == "owned":
            self._owned_items -= job.item_count
            self._owned_bytes -= job.encoded_bytes
        elif job.accounting == "waiting":
            self._waiting_items -= job.item_count
            self._waiting_bytes -= job.encoded_bytes
        job.accounting = "released"

    def _notify_if_cancelled(
        self,
        job: _DecodeJob,
        future: Future[list[MediaWithBytes[Image.Image]]],
    ) -> None:
        if future.cancelled():
            with self._cond:
                self._release_accounting_locked(job)
                self._promote_waiters_locked()
                self._cond.notify_all()

    def submit(
        self,
        image_io: ImageMediaIO,
        encoded_images: list[bytes] | tuple[bytes, ...],
    ) -> Future[list[MediaWithBytes[Image.Image]]]:
        self._check_pid()
        data = tuple(encoded_images)
        future: Future[list[MediaWithBytes[Image.Image]]] = Future()
        if not data:
            future.set_result([])
            return future

        spec = _decode_spec(image_io)
        coalesce = (
            len(data) == 1
            and self.config.batch_size > 1
            and _is_jpeg(data[0])
            and spec is not None
        )
        now = time.monotonic()
        with self._cond:
            if self._closing:
                raise RuntimeError("nvImageCodec decode service is shutting down")
            self._ensure_started_locked()
            self._sequence += 1
            job = _DecodeJob(
                image_io=image_io,
                encoded_images=data,
                future=future,
                sequence=self._sequence,
                deadline=now + self.config.coalesce_timeout_ms / 1000,
                spec=spec if coalesce else None,
                kind="coalesced" if coalesce else "direct",
                accounting="released",
                enqueued_at=now,
            )
            if self._can_own_locked(job):
                self._enqueue_owned_locked(job)
            elif self._can_wait_locked(job):
                self._enqueue_waiter_locked(job)
            else:
                raise NvImageCodecQueueFullError(
                    "nvImageCodec decode queue and admission backlog are full"
                )
            self._submitted_images += len(data)
            if not coalesce:
                self._direct_jobs += 1
            future.add_done_callback(partial(self._notify_if_cancelled, job))
            self._cond.notify()
        return future

    def _prune_cancelled_locked(self) -> None:
        waiting: deque[_DecodeJob] = deque()
        while self._admission_waiters:
            job = self._admission_waiters.popleft()
            if job.future.cancelled():
                self._release_accounting_locked(job)
            else:
                waiting.append(job)
        self._admission_waiters = waiting

        direct: deque[_DecodeJob] = deque()
        while self._direct_queue:
            job = self._direct_queue.popleft()
            if job.future.cancelled():
                self._release_accounting_locked(job)
            else:
                direct.append(job)
        self._direct_queue = direct

        for spec, queue in list(self._batch_queues.items()):
            retained: deque[_DecodeJob] = deque()
            while queue:
                job = queue.popleft()
                if job.future.cancelled():
                    self._release_accounting_locked(job)
                else:
                    retained.append(job)
            if retained:
                self._batch_queues[spec] = retained
            else:
                self._batch_queues.pop(spec, None)

    def _promote_waiters_locked(self) -> None:
        while self._admission_waiters:
            job = self._admission_waiters[0]
            if job.future.cancelled():
                self._admission_waiters.popleft()
                self._release_accounting_locked(job)
                continue
            if not self._can_own_locked(job):
                return
            self._admission_waiters.popleft()
            self._waiting_items -= job.item_count
            self._waiting_bytes -= job.encoded_bytes
            job.accounting = "released"
            self._enqueue_owned_locked(job)

    def _ready_batch_candidates_locked(
        self, now: float
    ) -> list[tuple[int, ImageDecodeSpec]]:
        candidates = []
        for spec, queue in self._batch_queues.items():
            if len(queue) >= self.config.batch_size or queue[0].deadline <= now:
                candidates.append((queue[0].sequence, spec))
        return candidates

    def _next_deadline_locked(self) -> float | None:
        if not self._batch_queues:
            return None
        return min(queue[0].deadline for queue in self._batch_queues.values())

    def _claim_locked(self, now: float) -> _Claim | None:
        batch_candidates = self._ready_batch_candidates_locked(now)
        direct_sequence = self._direct_queue[0].sequence if self._direct_queue else None
        batch_candidate = min(batch_candidates) if batch_candidates else None
        if direct_sequence is None and batch_candidate is None:
            return None

        use_direct = batch_candidate is None or (
            direct_sequence is not None and direct_sequence < batch_candidate[0]
        )
        if use_direct:
            jobs = [self._direct_queue.popleft()]
            kind: Literal["coalesced", "direct"] = "direct"
        else:
            assert batch_candidate is not None
            spec = batch_candidate[1]
            queue = self._batch_queues[spec]
            max_claim_size = self.config.batch_size * self.config.pipeline_depth
            jobs = [queue.popleft() for _ in range(min(len(queue), max_claim_size))]
            if not queue:
                self._batch_queues.pop(spec)
            kind = "coalesced"

        claimed = []
        for job in jobs:
            if job.future.set_running_or_notify_cancel():
                claimed.append(job)
            else:
                self._release_accounting_locked(job)
        if not claimed:
            self._promote_waiters_locked()
            return None
        self._in_flight += 1
        self._queue_wait_seconds += sum(now - job.enqueued_at for job in claimed)
        if kind == "coalesced":
            self._batch_widths[len(claimed)] += 1
        return _Claim(tuple(claimed), kind)

    def _dispatch_loop(self) -> None:
        while True:
            with self._cond:
                self._prune_cancelled_locked()
                self._promote_waiters_locked()
                if self._closing:
                    return
                now = time.monotonic()
                claim = None
                if self._in_flight < self.config.decoders:
                    claim = self._claim_locked(now)
                if claim is None:
                    deadline = (
                        self._next_deadline_locked()
                        if self._in_flight < self.config.decoders
                        else None
                    )
                    timeout = None if deadline is None else max(0.0, deadline - now)
                    self._cond.wait(timeout=timeout)
                    continue
                executor = self._executor
                assert executor is not None

            try:
                worker = executor.submit(self._execute_claim, claim)
            except BaseException as error:
                self._fail_claim(claim, error)
                self._claim_finished(claim)
            else:
                worker.add_done_callback(partial(self._worker_finished, claim))

    def _worker_finished(self, claim: _Claim, _worker: Future[None]) -> None:
        self._claim_finished(claim)

    def _release_before_publish(self, job: _DecodeJob) -> None:
        with self._cond:
            self._release_accounting_locked(job)
            self._promote_waiters_locked()
            self._cond.notify_all()

    def _set_exception(self, job: _DecodeJob, error: BaseException) -> None:
        self._release_before_publish(job)
        job.future.set_exception(error)

    def _set_result(
        self, job: _DecodeJob, result: list[MediaWithBytes[Image.Image]]
    ) -> None:
        self._release_before_publish(job)
        job.future.set_result(result)

    def _fail_claim(self, claim: _Claim, error: BaseException) -> None:
        for job in claim.jobs:
            self._set_exception(job, error)

    def _execute_direct(self, job: _DecodeJob) -> None:
        try:
            result = job.image_io.load_bytes_many(job.encoded_images)
            if len(result) != job.item_count:
                _close_loaded_images(result)
                raise RuntimeError(
                    "nvImageCodec image service returned an unexpected "
                    f"result count: expected {job.item_count}, got {len(result)}"
                )
        except BaseException as error:
            self._set_exception(job, error)
            return
        self._set_result(job, result)

    def _execute_coalesced(self, jobs: tuple[_DecodeJob, ...]) -> None:
        remaining = list(jobs)
        while remaining:
            image_io = remaining[0].image_io
            encoded = [job.encoded_images[0] for job in remaining]
            try:
                result = image_io.load_bytes_many(encoded)
                if len(result) != len(remaining):
                    _close_loaded_images(result)
                    raise RuntimeError(
                        "nvImageCodec image service returned an unexpected "
                        f"result count: expected {len(remaining)}, got {len(result)}"
                    )
            except BaseException as error:
                indexed = _indexed_error(error)
                if indexed is None or not 0 <= indexed[0] < len(remaining):
                    for job in remaining:
                        self._set_exception(job, error)
                    return
                failed = remaining.pop(indexed[0])
                self._set_exception(failed, indexed[1])
                continue

            for job, item in zip(remaining, result):
                self._set_result(job, [item])
            return

    def _execute_claim(self, claim: _Claim) -> None:
        try:
            if claim.kind == "direct":
                self._execute_direct(claim.jobs[0])
            else:
                self._execute_coalesced(claim.jobs)
        except BaseException as error:
            for job in claim.jobs:
                if not job.future.done():
                    self._set_exception(job, error)

    def _claim_finished(self, claim: _Claim) -> None:
        with self._cond:
            for job in claim.jobs:
                self._release_accounting_locked(job)
            self._in_flight -= 1
            self._promote_waiters_locked()
            self._cond.notify_all()

    def snapshot_stats(self) -> ImageDecodeServiceStats:
        with self._cond:
            return ImageDecodeServiceStats(
                submitted_images=self._submitted_images,
                direct_jobs=self._direct_jobs,
                batch_widths=dict(self._batch_widths),
                queue_wait_seconds=self._queue_wait_seconds,
            )

    def shutdown(self, *, log_stats: bool = True) -> None:
        self._check_pid()
        dispatcher = None
        executor = None
        with self._cond:
            if self._shutdown_complete:
                return
            if self._closing:
                raise RuntimeError(
                    "nvImageCodec decode service shutdown did not complete"
                )
            self._closing = True
            shutdown_error = RuntimeError("nvImageCodec decode service shut down")
            queued = list(self._direct_queue) + list(self._admission_waiters)
            for queue in self._batch_queues.values():
                queued.extend(queue)
            self._direct_queue.clear()
            self._admission_waiters.clear()
            self._batch_queues.clear()
            for job in queued:
                self._release_accounting_locked(job)
                if not job.future.done():
                    job.future.set_exception(shutdown_error)
            self._cond.notify_all()
            dispatcher = self._dispatcher
            executor = self._executor
        if dispatcher is not None:
            dispatcher.join()
        if executor is not None:
            executor.shutdown(wait=True)
        from vllm.multimodal.image_decoders.nvimagecodec import (
            shutdown_nvimagecodec_decoder_pool,
        )

        shutdown_nvimagecodec_decoder_pool()
        with self._cond:
            self._shutdown_complete = True
            stats_payload = json.dumps(
                {
                    "submitted_images": self._submitted_images,
                    "direct_jobs": self._direct_jobs,
                    "batch_widths": dict(sorted(self._batch_widths.items())),
                },
                sort_keys=True,
            )
            self._cond.notify_all()
        if log_stats:
            logger.info("nvImageCodec decode service stats: %s", stats_payload)


_manager_pid = os.getpid()
_manager_lock = threading.Lock()
_decode_service: NvImageCodecDecodeService | None = None
_service_leases = 0


def _reset_after_pristine_fork() -> None:
    global _decode_service, _manager_lock, _manager_pid, _service_leases
    pid = os.getpid()
    if _manager_pid == pid:
        return
    service = _decode_service
    if service is not None and service.started:
        raise RuntimeError(
            "nvImageCodec decode state cannot be reused after fork; start API "
            "workers with the spawn multiprocessing method."
        )
    _manager_lock = threading.Lock()
    _decode_service = None
    _service_leases = 0
    _manager_pid = pid


def get_nvimagecodec_decode_service(
    image_io: ImageMediaIO,
) -> NvImageCodecDecodeService:
    global _decode_service
    _reset_after_pristine_fork()
    config = ImageDecodeServiceConfig.from_image_io(image_io)
    with _manager_lock:
        if _decode_service is None:
            _decode_service = NvImageCodecDecodeService(config)
        elif _decode_service.config != config:
            raise RuntimeError(
                "nvImageCodec decode service is already configured as "
                f"{_decode_service.config}; got {config}"
            )
        return _decode_service


def load_images_with_service(
    image_io: ImageMediaIO,
    encoded_images: list[bytes] | tuple[bytes, ...],
) -> list[MediaWithBytes[Image.Image]]:
    service = get_nvimagecodec_decode_service(image_io)
    return service.submit(image_io, encoded_images).result()


async def load_images_with_service_async(
    image_io: ImageMediaIO,
    encoded_images: list[bytes] | tuple[bytes, ...],
) -> list[MediaWithBytes[Image.Image]]:
    service = get_nvimagecodec_decode_service(image_io)
    future = service.submit(image_io, encoded_images)
    wrapped = asyncio.wrap_future(future)
    try:
        return await asyncio.shield(wrapped)
    except asyncio.CancelledError:
        if not future.cancel():
            future.add_done_callback(_dispose_future_result)
        raise


def get_nvimagecodec_decode_service_stats() -> ImageDecodeServiceStats:
    _reset_after_pristine_fork()
    with _manager_lock:
        service = _decode_service
    if service is None:
        return ImageDecodeServiceStats(0, 0, {}, 0.0)
    return service.snapshot_stats()


def acquire_nvimagecodec_decode_service_lease() -> None:
    global _service_leases
    _reset_after_pristine_fork()
    with _manager_lock:
        _service_leases += 1


def release_nvimagecodec_decode_service_lease() -> None:
    global _decode_service, _service_leases
    _reset_after_pristine_fork()
    with _manager_lock:
        if _service_leases == 0:
            return
        _service_leases -= 1
        if _service_leases == 0:
            service = _decode_service
            if service is not None:
                service.shutdown()
                _decode_service = None
            else:
                from vllm.multimodal.image_decoders.nvimagecodec import (
                    shutdown_nvimagecodec_decoder_pool,
                )

                shutdown_nvimagecodec_decoder_pool()


def shutdown_nvimagecodec_decode_service(*, log_stats: bool = True) -> None:
    global _decode_service, _service_leases
    _reset_after_pristine_fork()
    with _manager_lock:
        service = _decode_service
        _service_leases = 0
        if service is not None:
            service.shutdown(log_stats=log_stats)
            _decode_service = None
        else:
            from vllm.multimodal.image_decoders.nvimagecodec import (
                shutdown_nvimagecodec_decoder_pool,
            )

            shutdown_nvimagecodec_decoder_pool()


def _shutdown_nvimagecodec_decode_service_at_exit() -> None:
    # A forked child cannot safely touch parent-owned threads or locks. Normal
    # lifecycle shutdown reports errors earlier; atexit is only best effort.
    if _manager_pid != os.getpid():
        return
    with contextlib.suppress(Exception):
        shutdown_nvimagecodec_decode_service(log_stats=False)


atexit.register(_shutdown_nvimagecodec_decode_service_at_exit)


__all__ = [
    "ImageDecodeServiceConfig",
    "ImageDecodeServiceStats",
    "NVIMAGECODEC_DEFAULT_COALESCE_TIMEOUT_MS",
    "NvImageCodecDecodeService",
    "NvImageCodecQueueFullError",
    "acquire_nvimagecodec_decode_service_lease",
    "get_nvimagecodec_decode_service",
    "get_nvimagecodec_decode_service_stats",
    "load_images_with_service",
    "load_images_with_service_async",
    "release_nvimagecodec_decode_service_lease",
    "shutdown_nvimagecodec_decode_service",
    "validate_nvimagecodec_coalesce_timeout_ms",
]
