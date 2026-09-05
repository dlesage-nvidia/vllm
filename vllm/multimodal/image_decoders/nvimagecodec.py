# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import os
import threading
import time
import weakref
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, TypeAlias

from PIL import Image

import vllm.envs as envs
from vllm.utils.mem_constants import MiB_bytes

_JPEG_EXIF_SIGNATURE = b"Exif\x00\x00"
_MAX_NATIVE_PIXELS = 3840 * 2160
_SINGLE_PROCESS_BATCH_CAP = 20
_MULTI_PROCESS_AGGREGATE_BATCH_CAP = 16
_DEVICE_ARENAS = 2
_DECODER_MAX_NUM_CPU_THREADS = 4
_PINNED_OUTPUT_WAIT_TIMEOUT_S = 30.0
_MISSING_PACKAGE_ERROR = "nvImageCodec requires CUDA-matched nvidia-nvimgcodec."

# The decoder retains two batch-wide device arenas and plugin workspace. CUDA
# context memory is shared with the video decoder when both backends are used.
NVIMAGECODEC_BYTES_PER_BATCH_SLOT = _DEVICE_ARENAS * _MAX_NATIVE_PIXELS * 3
NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES = 1664 * MiB_bytes
NVIMAGECODEC_CUDA_CONTEXT_BYTES = 640 * MiB_bytes

PILLOW_IMAGE_BACKEND = "pillow"
NVIMAGECODEC_IMAGE_BACKEND = "nvimagecodec"


class _PinnedOutputBudget:
    """Bound borrowed pinned outputs without splitting a decode wave."""

    def __init__(
        self,
        cap: int = 0,
        wait_timeout: float = _PINNED_OUTPUT_WAIT_TIMEOUT_S,
    ) -> None:
        self._pid = os.getpid()
        self._condition = threading.Condition()
        self._cap = cap
        self._in_use = 0
        self._wait_timeout = wait_timeout

    def configure(self, cap: int) -> None:
        with self._condition:
            self._cap = cap
            self._condition.notify_all()

    def try_acquire(self, count: int) -> tuple["_PinnedPermit", ...]:
        with self._condition:
            if self._in_use + count > self._cap:
                return ()
            self._in_use += count
            return tuple(_PinnedPermit(self) for _ in range(count))

    def wait_for(self, count: int, stop: threading.Event) -> bool:
        with self._condition:
            deadline = time.monotonic() + self._wait_timeout
            while not stop.is_set() and self._in_use + count > self._cap:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        "nvImageCodec pinned-output budget exhausted; a decoded "
                        "image was retained beyond its request."
                    )
                self._condition.wait(remaining)
            return not stop.is_set()

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _release(self, permit: "_PinnedPermit") -> None:
        if self._pid != os.getpid():
            return
        with self._condition:
            if permit._released:
                return
            permit._released = True
            self._in_use -= 1
            self._condition.notify_all()


class _PinnedPermit:
    def __init__(self, budget: _PinnedOutputBudget) -> None:
        self._budget = budget
        self._attached = False
        self._released = False

    def attach(self, host: object) -> None:
        if self._attached or self._released:
            raise RuntimeError("invalid pinned output reservation")
        weakref.finalize(host, self._budget._release, self)
        self._attached = True

    def release(self) -> None:
        if not self._attached:
            self._budget._release(self)


_pinned_output_budget = _PinnedOutputBudget()


def _reset_pinned_output_budget_after_fork() -> None:
    global _pinned_output_budget
    _pinned_output_budget = _PinnedOutputBudget()


os.register_at_fork(after_in_child=_reset_pinned_output_budget_after_fork)


class _PinnedImageLease:
    """Processor-scoped borrow of one pinned RGB/CHW tensor."""

    _is_vllm_nvimagecodec_pinned_lease = True

    __slots__ = ("_host", "_pid", "height", "width")

    def __init__(self, host: object, *, width: int, height: int) -> None:
        self._host: object | None = host
        self._pid = os.getpid()
        self.width = width
        self.height = height

    def borrow_tensor(self) -> object:
        if self._pid != os.getpid():
            raise RuntimeError("cannot borrow an nvImageCodec output after fork")
        if self._host is None:
            raise RuntimeError("nvImageCodec pinned output lease has expired")
        return self._host

    def release(self) -> None:
        self._host = None

    def __reduce__(self):
        raise TypeError("nvImageCodec pinned output leases cannot be serialized")


def get_nvimagecodec_batch_cap(api_process_count: int = 1) -> int:
    """Reduce each process's arena capacity as the API process count grows."""
    if api_process_count <= 0:
        raise ValueError("api_process_count must be a positive integer")
    if api_process_count == 1:
        return _SINGLE_PROCESS_BATCH_CAP
    return max(1, _MULTI_PROCESS_AGGREGATE_BATCH_CAP // api_process_count)


def get_nvimagecodec_non_context_bytes(api_process_count: int = 1) -> int:
    batch_cap = get_nvimagecodec_batch_cap(api_process_count)
    return (
        batch_cap * NVIMAGECODEC_BYTES_PER_BATCH_SLOT
        + NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES
    )


@dataclass(frozen=True)
class NvImageCodecInput:
    """An RGB JPEG admitted to the borrowed CHW decode path."""

    code_stream: object
    data: bytes = field(repr=False)
    width: int
    height: int


def validate_image_backend(backend: object) -> str:
    if backend not in (PILLOW_IMAGE_BACKEND, NVIMAGECODEC_IMAGE_BACKEND):
        raise ValueError(
            f"Unknown image backend {backend!r}; expected "
            f"{PILLOW_IMAGE_BACKEND!r} or {NVIMAGECODEC_IMAGE_BACKEND!r}."
        )
    return str(backend)


@dataclass(eq=False)
class _DeviceArena:
    decode_done: Any
    copy_done: Any
    images: list[Any | None]
    items: tuple[NvImageCodecInput, ...] = ()
    copies: list[tuple[int, Any, Any]] = field(default_factory=list)
    errors: dict[int, BaseException] = field(default_factory=dict)

    def reset(self) -> None:
        self.items = ()
        self.copies.clear()
        self.errors.clear()


NvImageCodecResult: TypeAlias = _PinnedImageLease
_DecodeJob: TypeAlias = tuple[NvImageCodecInput, Future[NvImageCodecResult]]
_DecodeOutcome: TypeAlias = NvImageCodecResult | BaseException


def _load_nvimgcodec():
    try:
        from nvidia import nvimgcodec
    except ImportError as exc:
        raise RuntimeError(_MISSING_PACKAGE_ERROR) from exc
    return nvimgcodec


class _NvImageCodecDecoder:
    """One owner-thread decoder with two reusable device arenas."""

    def __init__(
        self, batch_cap: int = _SINGLE_PROCESS_BATCH_CAP, device_index: int = 0
    ) -> None:
        import torch

        self._owner_thread_id = threading.get_ident()
        self._torch = torch
        self._device_index = device_index
        nvimgcodec = _load_nvimgcodec()
        torch.accelerator.set_device_index(device_index)
        device = torch.device("cuda", device_index)
        self._decode_stream = torch.Stream(device=device)
        self._copy_stream = torch.Stream(device=device)
        self._params = nvimgcodec.DecodeParams(
            allow_any_depth=False,
            apply_exif_orientation=False,
            color_spec=nvimgcodec.ColorSpec.SRGB,
            sample_format=nvimgcodec.SampleFormat.P_RGB,
        )
        self._decoder = nvimgcodec.Decoder(
            device_id=device_index,
            max_num_cpu_threads=_DECODER_MAX_NUM_CPU_THREADS,
            backends=[
                nvimgcodec.Backend(kind)
                for kind in (
                    nvimgcodec.BackendKind.HW_GPU_ONLY,
                    nvimgcodec.BackendKind.GPU_ONLY,
                    nvimgcodec.BackendKind.HYBRID_CPU_GPU,
                )
            ],
        )
        self._arenas = [
            _DeviceArena(torch.Event(), torch.Event(), [])
            for _ in range(_DEVICE_ARENAS)
        ]
        self._available = deque(self._arenas)
        try:
            self._prime_jpeg_hardware_route(nvimgcodec, batch_cap)
        except BaseException:
            with contextlib.suppress(BaseException):
                self.close()
            raise

    def _prime_jpeg_hardware_route(self, nvimgcodec: Any, batch_cap: int) -> None:
        buffer = BytesIO()
        with Image.new("RGB", (64, 64), (73, 131, 197)) as image:
            image.save(buffer, "JPEG", quality=90, subsampling=2)
        streams = [nvimgcodec.CodeStream(buffer.getvalue()) for _ in range(batch_cap)]
        with self._decode_stream:
            outputs = self._decoder.decode(
                streams,
                params=self._params,
                cuda_stream=self._decode_stream.native_handle,
            )
        self._decode_stream.synchronize()
        if len(outputs) != batch_cap:
            raise RuntimeError("nvImageCodec returned the wrong JPEG primer width")
        for output in outputs:
            if output is None:
                raise RuntimeError("nvImageCodec failed the JPEG hardware-route primer")
            self._validate_device_output(output, (3, 64, 64))

    def submit(
        self,
        items: tuple[NvImageCodecInput, ...],
        permits: tuple[_PinnedPermit, ...],
    ) -> _DeviceArena:
        self._check_owner()
        if len(permits) != len(items):
            raise RuntimeError("invalid nvImageCodec pinned output reservation")
        arena = self._available.popleft()
        arena.items = items
        try:
            self._submit_on_arena(arena, permits)
        except BaseException:
            for stream in (self._decode_stream, self._copy_stream):
                with contextlib.suppress(BaseException):
                    stream.synchronize()
            arena.reset()
            for permit in permits:
                permit.release()
            self._available.appendleft(arena)
            raise
        return arena

    def collect(self, arena: _DeviceArena) -> list[_DecodeOutcome]:
        self._check_owner()
        try:
            arena.copy_done.synchronize()
            results: list[_DecodeOutcome | None] = [None] * len(arena.items)
            for index, error in arena.errors.items():
                results[index] = error
            for index, _, host in arena.copies:
                results[index] = _PinnedImageLease(
                    host,
                    width=arena.items[index].width,
                    height=arena.items[index].height,
                )
            if any(result is None for result in results):
                raise RuntimeError("nvImageCodec returned an incomplete decode wave")
            return [result for result in results if result is not None]
        except BaseException:
            with contextlib.suppress(BaseException):
                self._copy_stream.synchronize()
            raise
        finally:
            arena.reset()
            self._available.append(arena)

    def close(self) -> None:
        self._check_owner()
        failures = []
        for cleanup in (self._decode_stream.synchronize, self._copy_stream.synchronize):
            try:
                cleanup()
            except BaseException as error:
                failures.append(error)
        for arena in self._arenas:
            arena.reset()
            arena.images.clear()
        self._available.clear()
        self._arenas.clear()
        self._decoder = None
        self._copy_stream = None
        self._decode_stream = None
        if failures:
            raise RuntimeError("nvImageCodec decoder cleanup failed") from failures[0]

    def _submit_on_arena(
        self,
        arena: _DeviceArena,
        permits: tuple[_PinnedPermit, ...],
    ) -> None:
        items = arena.items
        reusable = arena.images[: len(items)]
        reuse_images = len(reusable) == len(items) and all(
            image is not None
            and tuple(int(value) for value in image.shape)
            == (3, item.height, item.width)
            for item, image in zip(items, reusable, strict=True)
        )
        if not reuse_images:
            arena.images.clear()
            reusable.clear()

        kwargs = {
            "params": self._params,
            "cuda_stream": self._decode_stream.native_handle,
        }
        if reuse_images:
            kwargs["images"] = reusable
        with self._decode_stream:
            outputs = self._decoder.decode(
                [item.code_stream for item in items],
                **kwargs,
            )
            arena.decode_done.record(self._decode_stream)
        if len(outputs) != len(items):
            raise RuntimeError("nvImageCodec returned the wrong wave width")

        arena.images.extend([None] * (len(items) - len(arena.images)))
        for index, (item, output) in enumerate(zip(items, outputs, strict=True)):
            if output is None:
                permits[index].release()
                arena.errors[index] = ValueError(
                    "nvImageCodec could not decode the JPEG image."
                )
                continue
            self._validate_device_output(output, (3, item.height, item.width))
            arena.images[index] = output
            view = self._torch.from_dlpack(output)
            host = self._torch.empty(
                tuple(int(value) for value in view.shape),
                dtype=view.dtype,
                device="cpu",
                pin_memory=True,
            )
            permits[index].attach(host)
            arena.copies.append((index, view, host))
        with self._copy_stream:
            self._copy_stream.wait_event(arena.decode_done)
            for _, view, host in arena.copies:
                host.copy_(view, non_blocking=True)
            arena.copy_done.record(self._copy_stream)

    def _validate_device_output(
        self,
        output: Any,
        expected_shape: tuple[int, int, int],
    ) -> None:
        if (
            tuple(int(value) for value in output.shape) != expected_shape
            or str(output.dtype) != "uint8"
            or output.device_id != self._device_index
        ):
            raise RuntimeError("nvImageCodec violated the RGB/CHW device contract")

    def _check_owner(self) -> None:
        if threading.get_ident() != self._owner_thread_id:
            raise RuntimeError("nvImageCodec decoder used outside its owner thread")


class _NvImageCodecService:
    """Batch scalar decode jobs on one process-local decoder."""

    def __init__(self, batch_cap: int, device_index: int = 0) -> None:
        self._batch_cap = batch_cap
        self._device_index = device_index
        _pinned_output_budget.configure(_DEVICE_ARENAS * batch_cap)
        self._budget = _pinned_output_budget
        self._jobs: deque[_DecodeJob] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._stop_event = threading.Event()
        self._failure: BaseException | None = None
        self._outstanding: set[Future[NvImageCodecResult]] = set()
        self._ready: Future[None] = Future()
        self._thread = threading.Thread(
            target=self._owner,
            name="vllm-nvimagecodec-owner",
            daemon=True,
        )
        self._thread.start()

    def wait_until_ready(self) -> None:
        self._ready.result()

    def submit(self, item: NvImageCodecInput) -> Future[NvImageCodecResult]:
        future: Future[NvImageCodecResult] = Future()
        with self._condition:
            if self._failure is not None:
                raise RuntimeError("nvImageCodec decoder failed") from self._failure
            if self._closed:
                raise RuntimeError("nvImageCodec decoder is closed")
            self._outstanding.add(future)
            self._jobs.append((item, future))
            self._condition.notify()
        return future

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._stop_event.set()
            rejected = tuple(self._jobs)
            self._jobs.clear()
            self._outstanding.difference_update(job[1] for job in rejected)
            self._condition.notify()
        self._budget.wake_all()
        for _, future in rejected:
            if future.set_running_or_notify_cancel():
                future.set_exception(RuntimeError("nvImageCodec decoder is closed"))
        self._thread.join()

    def _owner(self) -> None:
        decoder = None
        try:
            decoder = _NvImageCodecDecoder(self._batch_cap, self._device_index)
            self._ready.set_result(None)
            self._run(decoder)
        except BaseException as error:
            self._fail(error)
        finally:
            if decoder is not None:
                try:
                    decoder.close()
                except BaseException as error:
                    self._fail(error)

    def _run(self, decoder: _NvImageCodecDecoder) -> None:
        pending: deque[tuple[tuple[_DecodeJob, ...], _DeviceArena]] = deque()
        held: tuple[_DecodeJob, ...] = ()
        stopping = False
        while pending or held or not stopping:
            while len(pending) < _DEVICE_ARENAS and not stopping:
                if not held:
                    jobs = self._claim(wait=not pending)
                    if jobs is None:
                        stopping = True
                        break
                    if not jobs:
                        break
                    held = jobs

                if self._stop_event.is_set():
                    self._reject_claimed(held)
                    held = ()
                    stopping = True
                    break

                permits = self._budget.try_acquire(len(held))
                if not permits:
                    if pending:
                        break
                    try:
                        budget_available = self._budget.wait_for(
                            len(held), self._stop_event
                        )
                    except TimeoutError as error:
                        self._reject_claimed(held, error)
                        held = ()
                        continue
                    if not budget_available:
                        self._reject_claimed(held)
                        held = ()
                        stopping = True
                    else:
                        held = self._claim(wait=False, claimed=held) or ()
                    continue

                try:
                    token = decoder.submit(tuple(job[0] for job in held), permits)
                except BaseException:
                    for permit in permits:
                        permit.release()
                    raise
                pending.append((held, token))
                held = ()

            if not pending:
                continue

            jobs, token = pending.popleft()
            results = decoder.collect(token)
            for (_, future), result in zip(jobs, results, strict=True):
                if isinstance(result, BaseException):
                    future.set_exception(result)
                else:
                    future.set_result(result)
            with self._condition:
                self._outstanding.difference_update(job[1] for job in jobs)

    def _claim(
        self,
        *,
        wait: bool,
        claimed: tuple[_DecodeJob, ...] = (),
    ) -> tuple[_DecodeJob, ...] | None:
        with self._condition:
            while wait and not self._jobs and not self._closed:
                self._condition.wait()
            if self._closed and not self._jobs and not claimed:
                return None
            jobs = list(claimed)
            while self._jobs and len(jobs) < self._batch_cap:
                job = self._jobs.popleft()
                if job[1].set_running_or_notify_cancel():
                    jobs.append(job)
                else:
                    self._outstanding.discard(job[1])
            return tuple(jobs)

    def _reject_claimed(
        self,
        jobs: tuple[_DecodeJob, ...],
        error: BaseException | None = None,
    ) -> None:
        if error is None:
            error = RuntimeError("nvImageCodec decoder is closed")
        with self._condition:
            self._outstanding.difference_update(job[1] for job in jobs)
        for _, future in jobs:
            future.set_exception(error)

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                error.__traceback__ = None
                self._failure = error
            failure = self._failure
            self._closed = True
            futures = tuple(self._outstanding)
            self._outstanding.clear()
            self._jobs.clear()
        if not self._ready.done():
            self._ready.set_exception(failure)
        for future in futures:
            with contextlib.suppress(InvalidStateError):
                future.set_exception(failure)


def create_nvimagecodec_decode_service(
    api_process_count: int = 1,
    *,
    device_index: int = 0,
) -> _NvImageCodecService:
    """Create the process-local nvImageCodec owner."""
    return _NvImageCodecService(
        get_nvimagecodec_batch_cap(api_process_count), device_index
    )


def preflight_image_nvimagecodec(
    data: bytes,
    *,
    image_mode: str | None = "RGB",
) -> NvImageCodecInput:
    """Validate and prepare an image for the native-only JPEG backend."""
    if image_mode != "RGB":
        raise ValueError("The nvimagecodec backend requires image_mode='RGB'.")
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("The nvimagecodec backend supports JPEG images only.")
    if not data.endswith(b"\xff\xd9"):
        raise ValueError("The nvimagecodec backend requires a complete JPEG image.")
    if _JPEG_EXIF_SIGNATURE in data:
        raise ValueError("The nvimagecodec backend does not support EXIF metadata.")

    nvimgcodec = _load_nvimgcodec()
    try:
        code_stream = nvimgcodec.CodeStream(data)
        codec_name = str(code_stream.codec_name).lower()
        width = int(code_stream.width)
        height = int(code_stream.height)
        precision = int(code_stream.precision)
        num_channels = int(code_stream.num_channels)
    except MemoryError:
        raise
    except Exception as error:
        raise ValueError("nvImageCodec could not parse the JPEG image.") from error

    if codec_name != "jpeg":
        raise ValueError("The nvimagecodec backend supports JPEG images only.")
    if width <= 4:
        raise ValueError(
            "The nvimagecodec backend requires a width of at least 5 pixels."
        )
    if height <= 0:
        raise ValueError("nvImageCodec reported invalid JPEG dimensions.")
    pixels = width * height
    max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
    if max_pixels > 0 and pixels > max_pixels:
        raise ValueError(
            f"Image dimensions {width}x{height} ({pixels} pixels) exceed "
            f"the maximum of {max_pixels} pixels. Set VLLM_MAX_IMAGE_PIXELS "
            "to increase this limit."
        )
    if pixels > _MAX_NATIVE_PIXELS:
        raise ValueError(
            "The nvimagecodec backend supports images with at most "
            "8,294,400 total pixels."
        )
    if precision != 8 or num_channels != 3:
        raise ValueError(
            "The nvimagecodec backend supports 8-bit, three-channel JPEG images only."
        )
    return NvImageCodecInput(code_stream, data, width, height)
