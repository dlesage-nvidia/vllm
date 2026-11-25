# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import ctypes
import importlib.util
import multiprocessing as mp
import os
import queue
import shutil
import socket
import tempfile
import threading
import traceback
import weakref
from collections import deque
from concurrent.futures import Future, InvalidStateError
from dataclasses import dataclass, field, replace
from io import BytesIO
from multiprocessing.connection import (
    Client,
    Connection,
    Listener,
    answer_challenge,
    deliver_challenge,
)
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias, cast

import regex as re
from PIL import Image

import vllm.envs as envs
from vllm.exceptions import RetryableRequestError
from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes

_JPEG_EXIF_SIGNATURE = b"Exif\x00\x00"
_JPEG_SCAN_MARKER = re.compile(rb"\xff[^\x00\xff\xd0-\xd7]")
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_MAX_NATIVE_PIXELS = 3840 * 2160
_SINGLE_PROCESS_BATCH_CAP = 20
_DEVICE_ARENAS = 2
_DECODER_MAX_NUM_CPU_THREADS = 4
_MISSING_PACKAGE_ERROR = "nvImageCodec requires CUDA-matched nvidia-nvimgcodec."

# The decoder retains two batch-wide device arenas and plugin workspace. Its
# output gate also bounds live pinned payload to two batches (949 MiB at one
# API process and 4K), though allocator caching can retain more physical RAM.
_NVIMAGECODEC_DEVICE_BYTES_PER_BATCH_SLOT = _DEVICE_ARENAS * _MAX_NATIVE_PIXELS * 3
NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES = 1664 * MiB_bytes
NVIMAGECODEC_CUDA_CONTEXT_BYTES = 640 * MiB_bytes

PILLOW_IMAGE_BACKEND = "pillow"
NVIMAGECODEC_IMAGE_BACKEND = "nvimagecodec"


def get_nvimagecodec_batch_cap(
    *,
    batch_cap: int | None = None,
) -> int:
    """Return the single GPU owner's startup batch cap."""
    topology_cap = _SINGLE_PROCESS_BATCH_CAP
    if batch_cap is None:
        return topology_cap
    if isinstance(batch_cap, bool) or not isinstance(batch_cap, int) or batch_cap <= 0:
        raise ValueError("batch_cap must be a positive integer or None")
    if batch_cap > topology_cap:
        raise ValueError(f"batch_cap must be <= {topology_cap}")
    return batch_cap


def get_nvimagecodec_workspace_bytes(
    workspace_reservation_mib: int | None = None,
) -> int:
    """Return the startup workspace allowance, not an allocator limit."""
    if workspace_reservation_mib is None:
        return NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES
    if (
        isinstance(workspace_reservation_mib, bool)
        or not isinstance(workspace_reservation_mib, int)
        or workspace_reservation_mib <= 0
    ):
        raise ValueError("workspace_reservation_mib must be a positive integer or None")
    return workspace_reservation_mib * MiB_bytes


def get_nvimagecodec_non_context_bytes(
    *,
    batch_cap: int | None = None,
    workspace_reservation_mib: int | None = None,
) -> int:
    batch_cap = get_nvimagecodec_batch_cap(batch_cap=batch_cap)
    return (
        batch_cap * _NVIMAGECODEC_DEVICE_BYTES_PER_BATCH_SLOT
        + get_nvimagecodec_workspace_bytes(workspace_reservation_mib)
    )


@dataclass(frozen=True)
class NvImageCodecInput:
    """An RGB JPEG admitted to the native CHW decode path."""

    data: bytes
    code_stream: object
    width: int
    height: int
    destination: Any = field(default=None, repr=False, compare=False)


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

    NvImageCodecResult: TypeAlias = torch.Tensor
else:
    NvImageCodecResult: TypeAlias = object
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
        self._closed = False
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
            self._warmup_jpeg_decode(nvimgcodec, batch_cap)
        except BaseException:
            with contextlib.suppress(BaseException):
                self.close()
            raise

    def _warmup_jpeg_decode(self, nvimgcodec: Any, batch_cap: int) -> None:
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
            raise RuntimeError("nvImageCodec returned the wrong JPEG warmup batch size")
        for output in outputs:
            if output is None:
                raise RuntimeError("nvImageCodec failed the JPEG decode warmup")
            self._validate_device_output(output, (3, 64, 64))

    def submit(
        self,
        items: tuple[NvImageCodecInput, ...],
    ) -> _DeviceArena:
        self._check_owner()
        if any(item.destination is None for item in items):
            raise ValueError("nvImageCodec requires a final pinned RGB destination")
        arena = self._available.popleft()
        arena.items = items
        try:
            self._submit_on_arena(arena)
        except BaseException:
            for stream in (self._decode_stream, self._copy_stream):
                with contextlib.suppress(BaseException):
                    stream.synchronize()
            raise
        return arena

    def collect(self, arena: _DeviceArena) -> list[NvImageCodecResult | None]:
        self._check_owner()
        arena.copy_done.synchronize()
        results: list[NvImageCodecResult | None] = [None] * len(arena.items)
        for index, _, host in arena.copies:
            results[index] = host
        arena.reset()
        self._available.append(arena)
        return results

    def close(self) -> None:
        self._check_owner()
        if self._closed:
            return
        self._closed = True
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
        for cleanup in (self._decode_stream.synchronize, self._copy_stream.synchronize):
            try:
                cleanup()
            except BaseException as error:
                failures.append(error)
        self._copy_stream = None
        self._decode_stream = None
        if failures:
            raise RuntimeError("nvImageCodec decoder cleanup failed") from failures[0]

    def _submit_on_arena(self, arena: _DeviceArena) -> None:
        items = arena.items
        reusable = arena.images[: len(items)]
        reuse_images = len(reusable) == len(items) and all(
            image is not None for image in reusable
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
            host = item.destination
            if (
                host.device.type != "cpu"
                or not host.is_pinned()
                or not host.is_contiguous()
                or host.dtype != view.dtype
                or tuple(host.shape) != tuple(view.shape)
            ):
                raise RuntimeError("Invalid final pinned RGB destination")
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
        self._jobs: deque[_DecodeJob] = deque()
        self._condition = threading.Condition()
        self._closed = False
        self._stop_event = threading.Event()
        self._failure: BaseException | None = None
        self._cleanup_failure: BaseException | None = None
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

    def is_healthy(self) -> bool:
        with self._condition:
            return self._ready.done() and self._failure is None and not self._closed

    def submit(self, item: NvImageCodecInput) -> Future[NvImageCodecResult]:
        if item.destination is None:
            raise ValueError("nvImageCodec requires a final pinned RGB destination")
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
        for _, future in rejected:
            if future.set_running_or_notify_cancel():
                future.set_exception(RuntimeError("nvImageCodec decoder is closed"))
        self._thread.join()
        if self._cleanup_failure is not None:
            raise RuntimeError(
                "nvImageCodec decoder cleanup failed"
            ) from self._cleanup_failure

    def _owner(self) -> None:
        decoder = None
        try:
            decoder = _NvImageCodecDecoder(self._batch_cap, self._device_index)
            logger.info(
                "nvImageCodec JPEG warmup succeeded (batch cap %d). "
                "Configured HW_GPU_ONLY, GPU_ONLY, HYBRID_CPU_GPU backends; "
                "HW JPEG availability is unverified, actual backend selection "
                "is unobservable through the supported decode API, and "
                "GPU/hybrid fallback is permitted, not confirmed selected.",
                self._batch_cap,
            )
            self._ready.set_result(None)
            self._run(decoder)
        except BaseException as error:
            self._fail(error)
        finally:
            if decoder is not None:
                try:
                    decoder.close()
                except BaseException as error:
                    self._cleanup_failure = error
                    self._fail(error)

    def _run(self, decoder: _NvImageCodecDecoder) -> None:
        pending: deque[tuple[tuple[_DecodeJob, ...], _DeviceArena]] = deque()
        stopping = False
        while pending or not stopping:
            while len(pending) < _DEVICE_ARENAS and not stopping:
                jobs = self._claim(wait=not pending)
                if jobs is None:
                    stopping = True
                    break
                if not jobs:
                    break
                if self._stop_event.is_set():
                    self._reject_claimed(jobs)
                    stopping = True
                    break
                token = decoder.submit(tuple(job[0] for job in jobs))
                pending.append((jobs, token))
                del jobs, token
            if pending:
                self._complete(decoder, *pending.popleft())

    def _complete(
        self,
        decoder: _NvImageCodecDecoder,
        jobs: tuple[_DecodeJob, ...],
        token: _DeviceArena,
    ) -> None:
        results = decoder.collect(token)
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
                logger.error(
                    "nvImageCodec decoder failed:\n%s",
                    "".join(traceback.format_exception(error)),
                )
                error.__traceback__ = None
                error.__cause__ = None
                error.__context__ = None
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


def _inspect_jpeg(data: bytes) -> tuple[bool, bool]:
    if not data.startswith(b"\xff\xd8"):
        return False, False

    position = 2
    in_scan = False
    saw_sof = False
    saw_scan = False
    has_exif = False
    frame_components: set[int] = set()
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
            if saw_sof or segment_length < 8:
                return False, has_exif
            num_components = data[position + 7]
            if num_components == 0 or segment_length != 8 + 3 * num_components:
                return False, has_exif
            frame_components = set(data[position + 8 : position + segment_length : 3])
            if len(frame_components) != num_components:
                return False, has_exif
            saw_sof = True
        if marker == 0xDA:
            if not saw_sof or segment_length < 6:
                return False, has_exif
            num_components = data[position + 2]
            if not 1 <= num_components <= 4 or segment_length != 6 + 2 * num_components:
                return False, has_exif
            scan_components = set(
                data[position + 3 : position + 3 + 2 * num_components : 2]
            )
            if (
                len(scan_components) != num_components
                or not scan_components <= frame_components
            ):
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


_SLOT_BYTES = 3 * _MAX_NATIVE_PIXELS
_CLOSE_TIMEOUT = 30


def _disconnect(connection: Connection) -> None:
    with (
        contextlib.suppress(OSError),
        socket.socket(fileno=os.dup(connection.fileno())) as transport,
    ):
        transport.shutdown(socket.SHUT_RDWR)


def _join_thread(thread: threading.Thread) -> None:
    thread.join(_CLOSE_TIMEOUT)
    if thread.is_alive():
        raise RuntimeError(f"Shared decoder thread {thread.name} did not stop")


class _MappedOutputs:
    def __init__(self, memory: SharedMemory) -> None:
        self.memory = memory
        self.condition = threading.Condition()
        self.views = 0

    def tensor(self, slot: int, width: int, height: int, release=None):
        import torch

        offset = slot * _SLOT_BYTES
        size = width * height * 3
        if (
            offset < 0
            or size <= 0
            or size > _SLOT_BYTES
            or offset + size > self.memory.size
        ):
            raise ValueError("Invalid shared RGB output extent")
        buffer = cast(memoryview, self.memory.buf)[offset : offset + size]
        with self.condition:
            self.views += 1
        weakref.finalize(buffer, self._release, release)
        return torch.frombuffer(buffer, dtype=torch.uint8).reshape(3, height, width)

    def _release(self, callback) -> None:
        with self.condition:
            self.views -= 1
            self.condition.notify_all()
        if callback is not None:
            callback()

    def close(self) -> None:
        with self.condition:
            if not self.condition.wait_for(lambda: self.views == 0, _CLOSE_TIMEOUT):
                raise RuntimeError("Shared RGB outputs still have device-owner views")
        self.memory.close()


class _PinnedOutputs(_MappedOutputs):
    def __init__(self, slots: int, device_index: int) -> None:
        import torch
        from cuda.bindings import runtime

        torch.accelerator.set_device_index(device_index)
        memory = SharedMemory(create=True, size=slots * _SLOT_BYTES)
        super().__init__(memory)
        self.runtime = runtime
        self.pointer_view: ctypes.Array[ctypes.c_ubyte] | None = (
            ctypes.c_ubyte * memory.size
        ).from_buffer(cast(memoryview, memory.buf))
        self.pointer = ctypes.addressof(self.pointer_view)
        (status,) = runtime.cudaHostRegister(self.pointer, memory.size, 0)
        if status != runtime.cudaError_t.cudaSuccess:
            self.pointer_view = None
            memory.close()
            memory.unlink()
            raise RuntimeError(f"cudaHostRegister failed: {status}")

    def close(self) -> None:
        with self.condition:
            if not self.condition.wait_for(lambda: self.views == 0, _CLOSE_TIMEOUT):
                raise RuntimeError("Pinned output views survived decoder shutdown")
        (status,) = self.runtime.cudaHostUnregister(self.pointer)
        if status != self.runtime.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaHostUnregister failed: {status}")
        try:
            self.pointer_view = None
            self.memory.close()
        finally:
            self.memory.unlink()


@dataclass(frozen=True)
class SharedDecoderEndpoint:
    address: str
    authkey: bytes
    batch_cap: int
    owner_pid: int


class _Peer:
    def __init__(self, connection, service, slots, device_index):
        self.connection, self.service = connection, service
        self.outputs = _PinnedOutputs(slots, device_index)
        self.condition = threading.Condition(threading.RLock())
        self.free = deque(range(slots))
        self.pending: deque[tuple[int, bytes]] = deque()
        self.running = {}
        self.leases = {}
        self.closed = False
        self.failure = None
        self.responses: queue.Queue[tuple[Any, ...] | None] = queue.Queue(slots + 1)
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.writer.start()

    def _write(self):
        try:
            while (message := self.responses.get()) is not None:
                self.connection.send(message)
        except (EOFError, OSError):
            _disconnect(self.connection)

    def _reply(self, message):
        try:
            self.responses.put_nowait(message)
        except queue.Full:
            _disconnect(self.connection)

    def run(self):
        try:
            self._reply(("ready", self.outputs.memory.name, self.outputs.memory.size))
            while True:
                message = self.connection.recv()
                with self.condition:
                    if message[0] == "decode":
                        request_id, data = message[1:]
                        if request_id in self.leases or any(
                            request_id == item[0] for item in self.pending
                        ):
                            raise ValueError("Duplicate shared decode request")
                        if (
                            len(self.pending) + len(self.leases)
                            >= self.outputs.memory.size // _SLOT_BYTES
                        ):
                            raise ValueError("Shared decoder admission limit exceeded")
                        self.pending.append((request_id, data))
                    elif message[0] == "release":
                        request_id = message[1]
                        if request_id in self.running or request_id not in self.leases:
                            raise ValueError("Invalid shared output release")
                        self.free.append(self.leases.pop(request_id))
                    else:
                        raise ValueError("Invalid shared decoder command")
                    self._dispatch()
        except (EOFError, OSError):
            pass
        finally:
            with self.condition:
                self.closed = True
                self.pending.clear()
                running = tuple(self.running.values())
            for future in running:
                future.cancel()
            del running
            future = None
            with self.condition:
                if not self.condition.wait_for(
                    lambda: not self.running, _CLOSE_TIMEOUT
                ):
                    raise RuntimeError("Shared decoder still has in-flight copies")
            with contextlib.ExitStack() as cleanup:
                cleanup.callback(self.outputs.close)
                cleanup.callback(self.connection.close)
                cleanup.callback(_join_thread, self.writer)
                _disconnect(self.connection)
                with contextlib.suppress(queue.Full):
                    self.responses.put_nowait(None)
            if self.failure is not None:
                raise RuntimeError("Shared decoder peer failed") from self.failure

    def _dispatch(self):
        while self.pending and self.free and not self.closed:
            request_id, data = self.pending.popleft()
            slot = self.free.popleft()
            self.leases[request_id] = slot
            try:
                item = preflight_image_nvimagecodec(data)
                target = self.outputs.tensor(slot, item.width, item.height)
                item = replace(item, destination=target)
                future = self.service.submit(item)
                self.running[request_id] = future
                shape = item.width, item.height
                future.add_done_callback(
                    lambda done, request_id=request_id, shape=shape: self._complete(
                        request_id, shape, done
                    )
                )
            except ValueError as error:
                self._reply(("error", request_id, str(error)))
            except BaseException as error:
                self.failure = RuntimeError(str(error))
                self.closed = True
                _disconnect(self.connection)
                return

    def _complete(self, request_id, shape, future):
        try:
            future.result()
            error = None
        except ValueError as failure:
            error = str(failure)
            failure.__traceback__ = None
        except BaseException as failure:
            with self.condition:
                if not self.closed:
                    failure.__traceback__ = None
                    self.failure = RuntimeError(str(failure))
                    self.closed = True
                    _disconnect(self.connection)
            error = None
        with self.condition:
            self.running.pop(request_id)
            if not self.closed:
                if error is None:
                    self._reply(("result", request_id, self.leases[request_id], *shape))
                else:
                    self._reply(("error", request_id, error))
                self._dispatch()
            self.condition.notify_all()


def _owner(endpoint, control, slots_per_client, device_index, api_process_count):
    listener = service = None
    active: dict[Connection, threading.Thread] = {}
    active_lock = threading.Lock()
    failures: list[BaseException] = []
    acceptor = None
    stopping = threading.Event()

    def serve(connection):
        peer = None
        authentication_timeout = threading.Timer(
            _CLOSE_TIMEOUT, _disconnect, (connection,)
        )
        try:
            authentication_timeout.start()
            try:
                deliver_challenge(connection, endpoint.authkey)
                answer_challenge(connection, endpoint.authkey)
            except (mp.AuthenticationError, EOFError, OSError):
                return
            finally:
                authentication_timeout.cancel()
            peer = _Peer(connection, service, slots_per_client, device_index)
            peer.run()
        except BaseException as error:
            error.__traceback__ = None
            with active_lock:
                if not failures:
                    failures.append(error)
            _disconnect(connection)
            connection.close()
        finally:
            if peer is None:
                connection.close()
            with active_lock:
                active.pop(connection)

    def accept():
        while not stopping.is_set():
            try:
                connection = cast(Listener, listener).accept()
            except (EOFError, OSError):
                return
            if stopping.is_set():
                connection.close()
                return
            thread = threading.Thread(target=serve, args=(connection,), daemon=True)
            with active_lock:
                if len(active) >= api_process_count:
                    connection.close()
                else:
                    active[connection] = thread
                    thread.start()

    def stop_accepting():
        if acceptor is not None and acceptor.is_alive():
            with (
                contextlib.suppress(OSError),
                socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as wakeup,
            ):
                wakeup.settimeout(_CLOSE_TIMEOUT)
                wakeup.connect(endpoint.address)
            _join_thread(acceptor)

    def disconnect_peers():
        with active_lock:
            for connection in active:
                _disconnect(connection)

    def join_peers():
        with active_lock:
            threads = tuple(active.values())
        with contextlib.ExitStack() as cleanup:
            for thread in threads:
                cleanup.callback(_join_thread, thread)
        if failures:
            raise RuntimeError("Shared decoder peer cleanup failed") from failures[0]

    try:
        listener = Listener(endpoint.address, family="AF_UNIX")
        service = _NvImageCodecService(endpoint.batch_cap, device_index)
        service.wait_until_ready()
        control.send(("ready", os.getpid()))
        acceptor = threading.Thread(target=accept, daemon=True)
        acceptor.start()
        while not control.poll(0.25):
            if failures or not service.is_healthy() or not acceptor.is_alive():
                raise RuntimeError("Shared nvImageCodec owner failed") from (
                    failures[0] if failures else service._failure
                )
        if control.recv() != "stop":
            raise ValueError("Invalid shared decoder lifecycle command")
    finally:
        stopping.set()
        with contextlib.ExitStack() as cleanup:
            cleanup.callback(control.close)
            cleanup.callback(join_peers)
            if service is not None:
                cleanup.callback(service.close)
            cleanup.callback(disconnect_peers)
            if listener is not None:
                cleanup.callback(listener.close)
            cleanup.callback(stop_accepting)


class SharedNvImageCodecManager:
    def __init__(self, api_process_count: int, batch_cap: int, device_index: int = 0):
        if (
            isinstance(api_process_count, bool)
            or not isinstance(api_process_count, int)
            or api_process_count <= 0
        ):
            raise ValueError("api_process_count must be a positive integer")
        batch_cap = get_nvimagecodec_batch_cap(batch_cap=batch_cap)
        self.directory = tempfile.mkdtemp(prefix="vllm-image-decoder-")
        context = mp.get_context("spawn")
        self.control, child = context.Pipe()
        endpoint = SharedDecoderEndpoint(
            str(Path(self.directory) / "owner.sock"), os.urandom(32), batch_cap, 0
        )
        slots = max(1, _DEVICE_ARENAS * batch_cap // api_process_count)
        self.process = context.Process(
            target=_owner,
            name="NvImageCodecOwner",
            args=(endpoint, child, slots, device_index, api_process_count),
        )
        self.closed = False
        try:
            self.process.start()
            child.close()
            if not self.control.poll(120):
                raise RuntimeError("Shared nvImageCodec owner startup timed out")
            kind, pid = self.control.recv()
            if kind != "ready":
                raise RuntimeError("Shared nvImageCodec owner did not initialize")
            self.endpoint = replace(endpoint, owner_pid=pid)
            logger.info(
                "Shared nvImageCodec owner PID %d: global batch cap %d, "
                "%d pinned shared RGB slots per API process",
                pid,
                batch_cap,
                slots,
            )
        except BaseException:
            self.close()
            raise

    def close(self, timeout: float | None = None):
        if self.closed:
            return
        self.closed = True
        if self.process.pid is not None:
            if self.process.is_alive():
                with contextlib.suppress(EOFError, OSError):
                    self.control.send("stop")
                self.process.join(_CLOSE_TIMEOUT if timeout is None else timeout)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(5)
            exitcode = self.process.exitcode
        else:
            exitcode = 0
        self.control.close()
        shutil.rmtree(self.directory)
        if exitcode != 0:
            raise RuntimeError(f"Shared nvImageCodec owner exited with {exitcode}")
        logger.info("Shared nvImageCodec owner PID %s exited cleanly", self.process.pid)


class SharedNvImageCodecClient:
    def __init__(self, endpoint: SharedDecoderEndpoint):
        self.owner_pid = endpoint.owner_pid
        self.connection = Client(
            endpoint.address, family="AF_UNIX", authkey=endpoint.authkey
        )
        if not self.connection.poll(120):
            self.connection.close()
            raise RuntimeError("Shared pinned output registration timed out")
        kind, name, size = self.connection.recv()
        if kind != "ready":
            self.connection.close()
            raise RuntimeError("Shared decoder did not register its final outputs")
        self.mapping = _MappedOutputs(SharedMemory(name=name))
        if self.mapping.memory.size != size:
            self.connection.close()
            self.mapping.close()
            raise RuntimeError("Shared output region size changed")
        self.condition = threading.Condition()
        self.pending: dict[int, Future[NvImageCodecResult]] = {}
        self.admitted: set[int] = set()
        self.capacity = self.mapping.memory.size // _SLOT_BYTES
        self.next_id = 0
        self.closed = False
        self.failure: RuntimeError | None = None
        self.messages: queue.SimpleQueue[tuple[Any, ...]] = queue.SimpleQueue()
        self.writer = threading.Thread(target=self._write, daemon=True)
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.writer.start()
        self.reader.start()

    def wait_until_ready(self):
        if not self.is_healthy():
            raise RuntimeError("Shared nvImageCodec client is not healthy")

    def is_healthy(self):
        with self.condition:
            return not self.closed and self.failure is None and self.reader.is_alive()

    def submit(self, item: NvImageCodecInput):
        with self.condition:
            if self.closed or self.failure is not None:
                raise RuntimeError(
                    "Shared nvImageCodec decoder is closed"
                ) from self.failure
            if len(self.admitted) >= self.capacity:
                raise RetryableRequestError(
                    "Shared image decoder is busy; release retained images or retry."
                )
            request_id = self.next_id
            self.next_id += 1
            future: Future[NvImageCodecResult] = Future()
            self.pending[request_id] = future
            self.admitted.add(request_id)
            self.messages.put(("decode", request_id, item.data))
            return future

    def _release(self, request_id):
        self.messages.put(("release", request_id))

    def _write(self):
        while True:
            message = self.messages.get()
            if not self.closed and self.failure is None:
                try:
                    self.connection.send(message)
                except (EOFError, OSError) as error:
                    self._fail(error)
            if message[0] == "release":
                with self.condition:
                    self.admitted.discard(message[1])
                    self.condition.notify_all()
            if self.closed:
                self.reader.join(_CLOSE_TIMEOUT)
                if self.reader.is_alive():
                    raise RuntimeError("Shared decoder response reader did not stop")
                if self.mapping.views == 0:
                    self.mapping.close()
                    return
            del message

    def _read(self):
        try:
            while True:
                message = self.connection.recv()
                kind, request_id = message[:2]
                with self.condition:
                    future = self.pending.pop(request_id)
                if kind == "result":
                    if future.cancelled():
                        self._release(request_id)
                    else:
                        slot, width, height = message[2:]
                        host = self.mapping.tensor(
                            slot,
                            width,
                            height,
                            lambda request_id=request_id: self._release(request_id),
                        )
                        with contextlib.suppress(InvalidStateError):
                            future.set_result(host)
                        del host
                elif kind == "error":
                    with contextlib.suppress(InvalidStateError):
                        future.set_exception(ValueError(message[2]))
                    self._release(request_id)
                else:
                    raise ValueError("Invalid shared decoder response")
                del future
        except BaseException as error:
            self._fail(error)

    def _fail(self, error):
        with self.condition:
            if self.failure is None:
                self.failure = RuntimeError(str(error))
            pending = tuple(self.pending.values())
            self.pending.clear()
            self.admitted.clear()
        for future in pending:
            with contextlib.suppress(InvalidStateError):
                future.set_exception(
                    RuntimeError("Shared nvImageCodec owner disconnected")
                )
        _disconnect(self.connection)

    def close(self):
        with self.condition:
            if self.closed:
                return
            self.closed = True
        self._fail(RuntimeError("Shared nvImageCodec client closed"))
        self.messages.put(("close",))
        if threading.current_thread() is not self.reader:
            self.reader.join(_CLOSE_TIMEOUT)
        if self.mapping.views == 0 and threading.current_thread() not in (
            self.writer,
            self.reader,
        ):
            self.writer.join(_CLOSE_TIMEOUT)
        if self.reader.is_alive() and threading.current_thread() is not self.reader:
            raise RuntimeError("Shared decoder response reader did not stop")
        self.connection.close()
        try:
            os.kill(self.owner_pid, 0)
        except ProcessLookupError:
            with contextlib.suppress(FileNotFoundError):
                self.mapping.memory.unlink()
