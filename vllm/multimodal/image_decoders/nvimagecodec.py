# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib.util
import os
import threading
import weakref
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field
from io import BytesIO
from typing import TYPE_CHECKING, Any, TypeAlias

import regex as re
from PIL import Image

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes

_JPEG_EXIF_SIGNATURE = b"Exif\x00\x00"
_JPEG_SCAN_MARKER = re.compile(rb"\xff[^\x00\xff\xd0-\xd7]")
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_MAX_NATIVE_PIXELS = 3840 * 2160
_SINGLE_PROCESS_BATCH_CAP = 20
_MULTI_PROCESS_AGGREGATE_BATCH_CAP = 16
_DEVICE_ARENAS = 2
_DECODER_MAX_NUM_CPU_THREADS = 4
_MISSING_PACKAGE_ERROR = "nvImageCodec requires CUDA-matched nvidia-nvimgcodec."

# The decoder retains two batch-wide device arenas and plugin workspace. Its
# output gate also bounds live pinned payload to two batches (949 MiB at one
# API process and 4K), though allocator caching can retain more physical RAM.
# CUDA context memory is shared with the video decoder when both backends run.
_NVIMAGECODEC_DEVICE_BYTES_PER_BATCH_SLOT = _DEVICE_ARENAS * _MAX_NATIVE_PIXELS * 3
NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES = 1664 * MiB_bytes
NVIMAGECODEC_CUDA_CONTEXT_BYTES = 640 * MiB_bytes

PILLOW_IMAGE_BACKEND = "pillow"
NVIMAGECODEC_IMAGE_BACKEND = "nvimagecodec"


class _OwnedOutputBudget:
    """Bound retained pinned tensors to the decoder's two full batches.

    The public backend admits one image per prompt, so each consumer can finish
    and release its output without waiting for another budget slot itself.
    """

    def __init__(self, cap: int) -> None:
        self._pid = os.getpid()
        self._condition = threading.Condition()
        self._cap = cap
        self._in_use = 0

    def try_reserve(self, count: int) -> bool:
        with self._condition:
            if self._in_use + count > self._cap:
                return False
            self._in_use += count
            return True

    def configure(self, cap: int) -> None:
        with self._condition:
            self._cap = cap
            self._condition.notify_all()

    def wait_for(self, count: int, stop: threading.Event) -> bool:
        with self._condition:
            while not stop.is_set() and self._in_use + count > self._cap:
                self._condition.wait()
            return not stop.is_set()

    def wake_all(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def track(self, outputs: list[object | None]) -> None:
        retained = [output for output in outputs if output is not None]
        for output in retained:
            weakref.finalize(output, self.release, 1)
        self.release(len(outputs) - len(retained))

    def release(self, count: int) -> None:
        if count == 0 or self._pid != os.getpid():
            return
        with self._condition:
            self._in_use -= count
            self._condition.notify_all()


_owned_output_budget = _OwnedOutputBudget(0)


def _reset_owned_output_budget_after_fork() -> None:
    global _owned_output_budget
    _owned_output_budget = _OwnedOutputBudget(0)


os.register_at_fork(after_in_child=_reset_owned_output_budget_after_fork)


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
        batch_cap * _NVIMAGECODEC_DEVICE_BYTES_PER_BATCH_SLOT
        + NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES
    )


@dataclass(frozen=True)
class NvImageCodecInput:
    """An RGB JPEG admitted to the native CHW decode path."""

    data: bytes
    code_stream: object
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

    def reset(self) -> None:
        self.items = ()
        self.copies.clear()


if TYPE_CHECKING:
    import torch

    NvImageCodecResult: TypeAlias = torch.Tensor | None
else:
    NvImageCodecResult: TypeAlias = object | None
_DecodeJob: TypeAlias = tuple[NvImageCodecInput, Future[NvImageCodecResult]]

logger = init_logger(__name__)


def _load_nvimgcodec():
    try:
        from nvidia import nvimgcodec
    except ImportError as exc:
        raise RuntimeError(_MISSING_PACKAGE_ERROR) from exc
    return nvimgcodec


def ensure_nvimagecodec_available() -> None:
    """Check package availability without importing CUDA code before forking."""
    try:
        available = importlib.util.find_spec("nvidia.nvimgcodec") is not None
    except (ImportError, ModuleNotFoundError):
        available = False
    if not available:
        raise RuntimeError(_MISSING_PACKAGE_ERROR)


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
    ) -> _DeviceArena:
        self._check_owner()
        arena = self._available.popleft()
        arena.items = items
        try:
            self._submit_on_arena(arena)
        except BaseException:
            for stream in (self._decode_stream, self._copy_stream):
                with contextlib.suppress(BaseException):
                    stream.synchronize()
            arena.reset()
            self._available.appendleft(arena)
            raise
        return arena

    def collect(self, arena: _DeviceArena) -> list[NvImageCodecResult]:
        self._check_owner()
        try:
            arena.copy_done.synchronize()
            results: list[NvImageCodecResult] = [None] * len(arena.items)
            for index, _, host in arena.copies:
                item = arena.items[index]
                if (
                    tuple(int(value) for value in host.shape)
                    != (3, item.height, item.width)
                    or host.device.type != "cpu"
                    or not host.is_pinned()
                ):
                    raise RuntimeError(
                        "nvImageCodec violated the pinned RGB/CHW host contract"
                    )
                results[index] = host
            return results
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

    def _submit_on_arena(self, arena: _DeviceArena) -> None:
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
        _owned_output_budget.configure(_DEVICE_ARENAS * batch_cap)
        self._budget = _owned_output_budget
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
        try:
            self._run_pending(decoder, pending)
        except BaseException:
            self._budget.release(sum(len(jobs) for jobs, _ in pending))
            raise

    def _run_pending(
        self,
        decoder: _NvImageCodecDecoder,
        pending: deque[tuple[tuple[_DecodeJob, ...], _DeviceArena]],
    ) -> None:
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

                if not self._budget.try_reserve(len(held)):
                    if pending:
                        break
                    if not self._budget.wait_for(len(held), self._stop_event):
                        self._reject_claimed(held)
                        held = ()
                        stopping = True
                    continue

                try:
                    token = decoder.submit(tuple(job[0] for job in held))
                    pending.append((held, token))
                except BaseException:
                    self._budget.release(len(held))
                    raise
                held = ()

            if not pending:
                continue

            jobs, token = pending.popleft()
            try:
                results = decoder.collect(token)
            except BaseException:
                self._budget.release(len(jobs))
                raise
            self._budget.track(results)
            for (_, future), result in zip(jobs, results, strict=True):
                if result is None:
                    future.set_exception(
                        ValueError("nvImageCodec failed to decode the JPEG.")
                    )
                else:
                    future.set_result(result)
            with self._condition:
                self._outstanding.difference_update(job[1] for job in jobs)

    def _claim(
        self,
        *,
        wait: bool,
    ) -> tuple[_DecodeJob, ...] | None:
        with self._condition:
            while wait and not self._jobs and not self._closed:
                self._condition.wait()
            if self._closed and not self._jobs:
                return None
            jobs: list[_DecodeJob] = []
            while self._jobs and len(jobs) < self._batch_cap:
                job = self._jobs.popleft()
                if job[1].set_running_or_notify_cancel():
                    jobs.append(job)
                else:
                    self._outstanding.discard(job[1])
            return tuple(jobs)

    def _reject_claimed(self, jobs: tuple[_DecodeJob, ...]) -> None:
        error = RuntimeError("nvImageCodec decoder is closed")
        with self._condition:
            self._outstanding.difference_update(job[1] for job in jobs)
        for _, future in jobs:
            future.set_exception(error)

    def _fail(self, error: BaseException) -> None:
        with self._condition:
            if self._failure is None:
                logger.exception("nvImageCodec decoder failed")
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


def _inspect_jpeg(data: bytes) -> tuple[bool, bool]:
    if not data.startswith(b"\xff\xd8"):
        return False, False

    position = 2
    in_scan = False
    saw_sof = False
    saw_scan = False
    has_exif = False
    while position < len(data):
        from_scan = in_scan
        if from_scan:
            match = _JPEG_SCAN_MARKER.search(data, position)
            if match is None:
                return False, has_exif
            marker = data[match.start() + 1]
            position = match.end()
            in_scan = False
        else:
            if data[position] != 0xFF:
                return False, has_exif
            while position < len(data) and data[position] == 0xFF:
                position += 1
            if position >= len(data):
                return False, has_exif
            marker = data[position]
            position += 1

        if marker == 0xD9:
            return saw_sof and saw_scan, has_exif
        if marker == 0x01:
            in_scan = from_scan
            continue
        if marker < 0xC0 or 0xD0 <= marker <= 0xD8:
            return False, has_exif
        if marker == 0xDC and not from_scan:
            return False, has_exif
        if position + 2 > len(data):
            return False, has_exif
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            return False, has_exif
        if marker == 0xE1 and data[position + 2 : position + 8] == _JPEG_EXIF_SIGNATURE:
            has_exif = True
        if marker in _JPEG_SOF_MARKERS:
            if saw_sof:
                return False, has_exif
            saw_sof = True
        if marker == 0xDA:
            if not saw_sof:
                return False, has_exif
            saw_scan = True
            in_scan = True
        elif marker == 0xDC:
            in_scan = True
        position += segment_length
    return False, has_exif


def preflight_image_nvimagecodec(
    data: bytes,
    *,
    image_mode: str | None = "RGB",
) -> NvImageCodecInput:
    """Validate the native RGB JPEG contract without decoding pixels."""
    if image_mode != "RGB":
        raise ValueError("nvImageCodec requires image_mode='RGB'.")
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("nvImageCodec currently supports only JPEG images.")
    complete, has_exif = _inspect_jpeg(data)
    if not complete:
        raise ValueError("nvImageCodec requires a structurally complete JPEG.")
    if has_exif:
        raise ValueError("nvImageCodec does not support JPEG EXIF metadata.")

    nvimgcodec = _load_nvimgcodec()
    try:
        code_stream = nvimgcodec.CodeStream(data)
        codec = str(code_stream.codec_name).lower()
        width = int(code_stream.width)
        height = int(code_stream.height)
        precision = int(code_stream.precision)
        num_channels = int(code_stream.num_channels)
    except MemoryError:
        raise
    except Exception as exc:
        raise ValueError(f"nvImageCodec rejected the JPEG metadata: {exc}") from exc

    if codec != "jpeg":
        raise ValueError("nvImageCodec currently supports only JPEG images.")
    if width <= 0 or height <= 0:
        raise ValueError("nvImageCodec reported invalid JPEG dimensions.")
    if width <= 4:
        raise ValueError("nvImageCodec requires JPEG width greater than 4 pixels.")
    pixels = width * height
    max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
    if max_pixels > 0 and pixels > max_pixels:
        raise ValueError(
            f"Image dimensions {width}x{height} ({pixels} pixels) exceed "
            f"the maximum of {max_pixels} pixels. Set VLLM_MAX_IMAGE_PIXELS "
            "to increase this limit."
        )
    if pixels > _MAX_NATIVE_PIXELS:
        raise ValueError("nvImageCodec JPEGs are limited to 8,294,400 pixels.")
    if precision != 8 or num_channels != 3:
        raise ValueError("nvImageCodec requires 8-bit, three-channel JPEG input.")
    return NvImageCodecInput(data, code_stream, width, height)
