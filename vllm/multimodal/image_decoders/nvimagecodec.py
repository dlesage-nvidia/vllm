# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib.util
import threading
from collections import deque
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Literal, TypeAlias

import numpy as np
import regex as re
from PIL import ExifTags, Image, ImageOps

import vllm.envs as envs
from vllm.utils.mem_constants import MiB_bytes

_NATIVE_CODECS = frozenset({"bmp", "jpeg", "jpeg2k", "png", "pnm", "tiff", "webp"})
_JPEG_EXIF_SIGNATURE = b"Exif\x00\x00"
_JPEG_SCAN_MARKER = re.compile(rb"\xff[^\x00\xff\xd0-\xd7]")
_JPEG_SOF_MARKERS = frozenset(range(0xC0, 0xD0)) - {0xC4, 0xC8, 0xCC}
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG2000_SIGNATURE = b"\x00\x00\x00\x0cjP  \r\n\x87\n"
_TIFF_SIGNATURES = (b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+")
_BMP_DIB_HEADER_SIZES = frozenset({12, 40, 52, 56, 64, 108, 124})
_JP2_SUPERBOX_TYPES = frozenset({b"jp2h", b"res "})
_MAX_NATIVE_PIXELS = 3840 * 2160
_SINGLE_PROCESS_BATCH_CAP = 20
_MULTI_PROCESS_AGGREGATE_BATCH_CAP = 16
_DEVICE_ARENAS = 2
_DECODER_MAX_NUM_CPU_THREADS = 4
_NATIVE_TIFF_COMPRESSIONS = frozenset({"jpeg", "raw", "tiff_adobe_deflate", "tiff_lzw"})
_MISSING_PACKAGE_ERROR = "nvImageCodec requires CUDA-matched nvidia-nvimgcodec."

# nvJPEG, nvJPEG 2000, and nvTIFF retain raster-dependent workspace for the
# Decoder lifetime. CUDA context memory stays separate so image and video share
# one max(context) reservation instead of double-counting it.
NVIMAGECODEC_BYTES_PER_BATCH_SLOT = _DEVICE_ARENAS * _MAX_NATIVE_PIXELS * 3
NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES = 1664 * MiB_bytes
NVIMAGECODEC_CUDA_CONTEXT_BYTES = 640 * MiB_bytes

PILLOW_IMAGE_BACKEND = "pillow"
NVIMAGECODEC_IMAGE_BACKEND = "nvimagecodec"
NvImageCodecOutputLayout: TypeAlias = Literal["hwc_rgb", "chw_rgb"]


def get_nvimagecodec_batch_cap(api_process_count: int = 1) -> int:
    """Share a bounded aggregate arena budget across API processes."""
    if api_process_count <= 0:
        raise ValueError("api_process_count must be a positive integer")
    if api_process_count == 1:
        return _SINGLE_PROCESS_BATCH_CAP
    return max(1, _MULTI_PROCESS_AGGREGATE_BATCH_CAP // api_process_count)


def get_nvimagecodec_non_context_bytes(api_process_count: int = 1) -> int:
    batch_cap = get_nvimagecodec_batch_cap(api_process_count)
    arena_bytes = batch_cap * NVIMAGECODEC_BYTES_PER_BATCH_SLOT
    return arena_bytes + NVIMAGECODEC_PLUGIN_WORKSPACE_BYTES


@dataclass(frozen=True)
class NvImageCodecInput:
    """An image admitted to the fixed RGB uint8 decode contract."""

    data: bytes
    code_stream: object
    width: int
    height: int
    orientation: int
    output_layout: NvImageCodecOutputLayout = "hwc_rgb"


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
    device_layout: NvImageCodecOutputLayout = "hwc_rgb"
    items: tuple[NvImageCodecInput, ...] = ()
    copies: list[tuple[int, Any, Any]] = field(default_factory=list)

    def reset(self) -> None:
        self.items = ()
        self.copies.clear()


_DecodeJob: TypeAlias = tuple[NvImageCodecInput, Future[np.ndarray]]


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
    """One owner-thread Decoder with the fixed two-arena host-delivery path."""

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
        params = dict(
            allow_any_depth=False,
            apply_exif_orientation=False,
            color_spec=nvimgcodec.ColorSpec.SRGB,
        )
        self._params = {
            "hwc_rgb": nvimgcodec.DecodeParams(
                **params, sample_format=nvimgcodec.SampleFormat.I_RGB
            ),
            "chw_rgb": nvimgcodec.DecodeParams(
                **params, sample_format=nvimgcodec.SampleFormat.P_RGB
            ),
        }
        self._decoder = nvimgcodec.Decoder(
            device_id=device_index,
            max_num_cpu_threads=_DECODER_MAX_NUM_CPU_THREADS,
            backends=[
                nvimgcodec.Backend(kind)
                for kind in (
                    nvimgcodec.BackendKind.HW_GPU_ONLY,
                    nvimgcodec.BackendKind.GPU_ONLY,
                    nvimgcodec.BackendKind.HYBRID_CPU_GPU,
                    nvimgcodec.BackendKind.CPU_ONLY,
                )
            ],
        )
        self._arenas = [
            _DeviceArena(torch.Event(), torch.Event(), [])
            for _ in range(_DEVICE_ARENAS)
        ]
        self._available = deque(self._arenas)
        self._materializer = ThreadPoolExecutor(
            max_workers=_DECODER_MAX_NUM_CPU_THREADS,
            thread_name_prefix="vllm-nvimagecodec-materializer",
        )
        # nvImageCodec fixes nvJPEG's maximum batch from its first JPEG call.
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
        data = buffer.getvalue()
        code_streams = [nvimgcodec.CodeStream(data) for _ in range(batch_cap)]
        with self._decode_stream:
            outputs = self._decoder.decode(
                code_streams,
                params=self._params["chw_rgb"],
                cuda_stream=self._decode_stream.native_handle,
            )
        self._decode_stream.synchronize()
        if len(outputs) != batch_cap:
            raise RuntimeError("nvImageCodec returned the wrong JPEG primer width")

        for output in outputs:
            if output is None:
                raise RuntimeError("nvImageCodec failed the JPEG hardware-route primer")
            self._validate_device_output(output, (3, 64, 64))

    def submit(self, items: tuple[NvImageCodecInput, ...]) -> _DeviceArena:
        self._check_owner()
        arena = self._available.popleft()
        all_chw = all(item.output_layout == "chw_rgb" for item in items)
        device_layout: NvImageCodecOutputLayout = "chw_rgb" if all_chw else "hwc_rgb"
        arena.items = items
        try:
            self._submit_on_arena(arena, device_layout)
        except BaseException:
            for stream in (self._decode_stream, self._copy_stream):
                with contextlib.suppress(BaseException):
                    stream.synchronize()
            arena.reset()
            self._available.appendleft(arena)
            raise
        return arena

    def collect(self, token: _DeviceArena) -> list[np.ndarray | Exception]:
        self._check_owner()
        try:
            token.copy_done.synchronize()
            arrays = list(
                self._materializer.map(
                    self._materialize,
                    (token.items[index] for index, _, _ in token.copies),
                    [token.device_layout] * len(token.copies),
                    (host for _, _, host in token.copies),
                )
            )
            results: list[np.ndarray | Exception] = [
                ValueError("nvImageCodec failed to decode image") for _ in token.items
            ]
            for (index, _, _), array in zip(token.copies, arrays, strict=True):
                results[index] = array
            return results
        except BaseException:
            with contextlib.suppress(BaseException):
                self._copy_stream.synchronize()
            raise
        finally:
            token.reset()
            self._available.append(token)

    def close(self) -> None:
        self._check_owner()
        failures = []
        for cleanup in (
            self._decode_stream.synchronize,
            self._copy_stream.synchronize,
            self._materializer.shutdown,
        ):
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
        self._params.clear()
        self._copy_stream = None
        self._decode_stream = None
        if failures:
            raise RuntimeError("nvImageCodec decoder cleanup failed") from failures[0]

    def _submit_on_arena(
        self,
        arena: _DeviceArena,
        device_layout: NvImageCodecOutputLayout,
    ) -> None:
        items = arena.items
        reusable = arena.images[: len(items)]
        reuse_images = (
            arena.device_layout == device_layout
            and len(reusable) == len(items)
            and all(
                image is not None
                and tuple(int(value) for value in image.shape)
                == self._device_shape(item, device_layout)
                for item, image in zip(items, reusable, strict=True)
            )
        )
        if not reuse_images:
            arena.images.clear()
            reusable.clear()
            arena.device_layout = device_layout
        kwargs = {
            "params": self._params[device_layout],
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
            if reuse_images and output is not reusable[index]:
                raise RuntimeError("nvImageCodec did not reuse its device arena")
            shape = self._device_shape(item, device_layout)
            self._validate_device_output(output, shape)
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
            raise RuntimeError("nvImageCodec violated the RGB device contract")

    @staticmethod
    def _device_shape(
        item: NvImageCodecInput,
        device_layout: NvImageCodecOutputLayout,
    ) -> tuple[int, int, int]:
        shape = (item.height, item.width)
        return (3, *shape) if device_layout == "chw_rgb" else (*shape, 3)

    @staticmethod
    def _materialize(
        item: NvImageCodecInput,
        device_layout: NvImageCodecOutputLayout,
        host: Any,
    ) -> np.ndarray:
        staging = host.numpy()
        hwc = staging if device_layout == "hwc_rgb" else np.moveaxis(staging, 0, -1)
        source = _apply_exif_orientation_view(hwc, item.orientation)
        if item.output_layout == "chw_rgb":
            source = np.moveaxis(source, -1, 0)
        return np.array(source, dtype=np.uint8, copy=True, order="C")

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
        self._failure: BaseException | None = None
        self._outstanding: set[Future[np.ndarray]] = set()
        self._ready: Future[None] = Future()
        self._thread = threading.Thread(
            target=self._owner,
            name="vllm-nvimagecodec-owner",
            daemon=True,
        )
        self._thread.start()

    def wait_until_ready(self) -> None:
        self._ready.result()

    def submit(self, item: NvImageCodecInput) -> Future[np.ndarray]:
        future: Future[np.ndarray] = Future()
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
            rejected = tuple(self._jobs)
            self._jobs.clear()
            self._outstanding.difference_update(job[1] for job in rejected)
            self._condition.notify()
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
        stopping = False
        while pending or not stopping:
            while len(pending) < _DEVICE_ARENAS and not stopping:
                jobs = self._claim(wait=not pending)
                if jobs is None:
                    stopping = True
                    break
                if not jobs:
                    break
                token = decoder.submit(tuple(job[0] for job in jobs))
                pending.append((jobs, token))

            if not pending:
                continue

            jobs, token = pending.popleft()
            results = decoder.collect(token)
            for (_, future), result in zip(jobs, results, strict=True):
                if isinstance(result, Exception):
                    future.set_exception(result)
                else:
                    future.set_result(result)
            with self._condition:
                self._outstanding.difference_update(job[1] for job in jobs)

    def _claim(self, *, wait: bool) -> tuple[_DecodeJob, ...] | None:
        jobs = []
        with self._condition:
            while wait and not self._jobs and not self._closed:
                self._condition.wait()
            if self._closed and not self._jobs:
                return None
            while self._jobs:
                job = self._jobs.popleft()
                if job[1].set_running_or_notify_cancel():
                    jobs.append(job)
                else:
                    self._outstanding.discard(job[1])
                if len(jobs) == self._batch_cap:
                    break
        return tuple(jobs)

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
            self._condition.notify_all()
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


def jpeg_has_complete_scan_and_eoi(data: bytes) -> bool:
    """Return whether a JPEG has a structurally complete frame and scan."""
    if not data.startswith(b"\xff\xd8"):
        return False

    position = 2
    in_scan = False
    saw_sof = False
    saw_scan = False
    while position < len(data):
        from_scan = in_scan
        if from_scan:
            match = _JPEG_SCAN_MARKER.search(data, position)
            if match is None:
                return False
            marker = data[match.start() + 1]
            position = match.end()
            in_scan = False
        else:
            if data[position] != 0xFF:
                return False
            while position < len(data) and data[position] == 0xFF:
                position += 1
            if position >= len(data):
                return False
            marker = data[position]
            position += 1

        if marker == 0xD9:
            return saw_sof and saw_scan
        if marker == 0x01:
            in_scan = from_scan
            continue
        if marker < 0xC0 or 0xD0 <= marker <= 0xD8:
            return False
        if marker == 0xDC and not from_scan:
            return False
        if position + 2 > len(data):
            return False
        segment_length = int.from_bytes(data[position : position + 2], "big")
        if segment_length < 2 or position + segment_length > len(data):
            return False
        if marker in _JPEG_SOF_MARKERS:
            if saw_sof:
                return False
            saw_sof = True
        if marker == 0xDA:
            if not saw_sof:
                return False
            saw_scan = True
            in_scan = True
        elif marker == 0xDC:
            in_scan = True
        position += segment_length
    return False


def _png_has_complete_iend(data: bytes) -> bool:
    if not data.startswith(_PNG_SIGNATURE):
        return False

    position = len(_PNG_SIGNATURE)
    while position + 12 <= len(data):
        chunk_length = int.from_bytes(data[position : position + 4], "big")
        chunk_end = position + 12 + chunk_length
        if chunk_end > len(data):
            return False
        if data[position + 4 : position + 8] == b"IEND":
            return chunk_length == 0
        position = chunk_end
    return False


def _parse_bmp_header(data: bytes) -> tuple[int, int] | None:
    if len(data) < 18:
        return None
    declared_size = int.from_bytes(data[2:6], "little")
    pixel_offset = int.from_bytes(data[10:14], "little")
    dib_size = int.from_bytes(data[14:18], "little")
    header_end = 14 + dib_size
    if (
        dib_size not in _BMP_DIB_HEADER_SIZES
        or header_end > len(data)
        or (declared_size and declared_size > len(data))
    ):
        return None

    if dib_size == 12:
        width = int.from_bytes(data[18:20], "little")
        height = int.from_bytes(data[20:22], "little")
        planes = int.from_bytes(data[22:24], "little")
        bits = int.from_bytes(data[24:26], "little")
        compression = image_size = 0
    else:
        width = int.from_bytes(data[18:22], "little", signed=True)
        height = int.from_bytes(data[22:26], "little", signed=True)
        planes = int.from_bytes(data[26:28], "little")
        bits = int.from_bytes(data[28:30], "little")
        compression = int.from_bytes(data[30:34], "little")
        image_size = int.from_bytes(data[34:38], "little")

    valid_compression = compression == 0 or (
        (compression, bits) in {(1, 8), (2, 4)}
        or (compression == 3 and bits in {16, 24, 32})
    )
    file_end = declared_size or len(data)
    if (
        width <= 0
        or height == 0
        or planes != 1
        or bits not in {1, 4, 8, 16, 24, 32}
        or not valid_compression
        or pixel_offset < header_end
        or pixel_offset >= file_end
        or (compression in {1, 2} and height < 0)
        or (image_size and pixel_offset + image_size > file_end)
    ):
        return None
    if compression in {0, 3}:
        row_bytes = ((width * bits + 31) // 32) * 4
        if pixel_offset + row_bytes * abs(height) > file_end:
            return None

    return bits, compression


def _jp2_has_valid_box_structure(data: bytes) -> bool:
    ranges = [(0, len(data))]
    while ranges:
        position, end = ranges.pop()
        while position < end:
            if end - position < 8:
                return False
            box_length = int.from_bytes(data[position : position + 4], "big")
            box_type = data[position + 4 : position + 8]
            header_length = 8
            if box_length == 1:
                if end - position < 16:
                    return False
                box_length = int.from_bytes(data[position + 8 : position + 16], "big")
                header_length = 16
            elif box_length == 0:
                box_length = end - position
            if box_length < header_length or box_length > end - position:
                return False
            box_end = position + box_length
            if box_type in _JP2_SUPERBOX_TYPES:
                ranges.append((position + header_length, box_end))
            position = box_end
    return True


def _signed_native_codec(data: bytes) -> tuple[str, str] | None:
    if data.startswith(b"\xff\xd8"):
        return "jpeg", "JPEG"
    if data.startswith(_PNG_SIGNATURE):
        return "png", "PNG"
    if data.startswith(b"BM"):
        return "bmp", "BMP"
    if data.startswith(_TIFF_SIGNATURES):
        return "tiff", "TIFF"
    if data.startswith(_JPEG2000_SIGNATURE) or data.startswith(b"\xff\x4f"):
        return "jpeg2k", "JPEG 2000"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp", "WebP"
    if (
        len(data) >= 3
        and data[0:1] == b"P"
        and data[1:2] in (b"1", b"2", b"3", b"4", b"5", b"6", b"7")
        and data[2:3].isspace()
    ):
        return "pnm", "PNM"
    return None


def _pnm_maxval(data: bytes) -> int | None:
    if data[:2] not in (b"P2", b"P3", b"P5", b"P6"):
        return None

    position = 2
    tokens: list[bytes] = []
    while len(tokens) < 3:
        while position < len(data) and data[position : position + 1].isspace():
            position += 1
        if position < len(data) and data[position] == ord("#"):
            position += 1
            while position < len(data) and data[position] not in (ord("\r"), ord("\n")):
                position += 1
            if position == len(data):
                return None
            continue
        start = position
        while (
            position < len(data)
            and not data[position : position + 1].isspace()
            and data[position] != ord("#")
        ):
            position += 1
        if start == position:
            return None
        tokens.append(data[start:position])
    try:
        return int(tokens[2])
    except ValueError:
        return None


def _apply_exif_orientation_view(
    array: np.ndarray,
    orientation: int | None,
) -> np.ndarray:
    if orientation in (None, 1):
        return array
    if orientation == 2:
        oriented = np.flip(array, axis=1)
    elif orientation == 3:
        oriented = np.flip(array, axis=(0, 1))
    elif orientation == 4:
        oriented = np.flip(array, axis=0)
    elif orientation == 5:
        oriented = np.swapaxes(array, 0, 1)
    elif orientation == 6:
        oriented = np.rot90(array, k=3)
    elif orientation == 7:
        oriented = np.flip(np.swapaxes(array, 0, 1), axis=(0, 1))
    elif orientation == 8:
        oriented = np.rot90(array, k=1)
    else:
        raise ValueError(f"invalid EXIF orientation: {orientation}")
    return oriented


def _opened_pillow_semantics(
    image: Image.Image, codec: str, preserve_mode: bool = False
) -> tuple[int, bool]:
    frame_count = int(getattr(image, "n_frames", 1))
    if bool(getattr(image, "is_animated", False)) or frame_count > 1:
        return 1, True
    has_transparency = image.mode in {"RGBA", "LA", "PA", "RGBa", "La"}
    has_transparency = has_transparency or "transparency" in image.info
    if preserve_mode and (image.mode != "RGB" or has_transparency):
        return 1, True
    if has_transparency and image.mode != "RGB":
        return 1, True
    if (
        codec == "tiff"
        and image.info.get("compression") not in _NATIVE_TIFF_COMPRESSIONS
    ):
        return 1, True
    if codec == "tiff" and image.mode not in {"1", "L", "P", "RGB"}:
        return 1, True
    if codec == "tiff" and (
        image.tag_v2.get(262) == 0 or image.tag_v2.get(266, 1) == 2
    ):
        return 1, True

    try:
        exif = image.getexif()
        orientation = exif.get(ExifTags.Base.Orientation, 1)
        if ExifTags.Base.ImageID in exif:
            return 1, True
    except Exception:
        orientation = 1

    if not isinstance(orientation, int) or orientation not in range(1, 9):
        orientation = 1
    return orientation, False


def _pillow_can_decode(data: bytes) -> bool:
    try:
        with Image.open(BytesIO(data)) as image:
            width, height = image.size
            max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
            if width * height > _MAX_NATIVE_PIXELS or (
                max_pixels > 0 and width * height > max_pixels
            ):
                return False
            with contextlib.suppress(Exception):
                image = ImageOps.exif_transpose(image)
            image.load()
            return True
    except MemoryError:
        raise
    except Exception:
        return False


def _palette_tiff_bit_depth(data: bytes) -> int | None:
    try:
        with Image.open(BytesIO(data)) as image:
            if image.mode != "P":
                return None
            bits = image.tag_v2.get(258)
            if isinstance(bits, tuple) and len(bits) == 1:
                bits = bits[0]
            return bits if isinstance(bits, int) else None
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to inspect TIFF metadata: {exc}") from exc


def _pillow_semantics(
    data: bytes, codec: str, num_channels: int, preserve_mode: bool = False
) -> tuple[int, bool]:
    needs_metadata = (
        preserve_mode
        or codec in {"png", "tiff", "webp"}
        or (codec in {"bmp", "jpeg2k", "pnm"} and num_channels in {2, 4})
        or (codec == "jpeg" and _JPEG_EXIF_SIGNATURE in data)
    )
    if not needs_metadata:
        return 1, False

    try:
        with Image.open(BytesIO(data)) as image:
            return _opened_pillow_semantics(image, codec, preserve_mode)
    except (OSError, ValueError) as exc:
        raise ValueError(f"Failed to inspect image metadata: {exc}") from exc


def _validate_metadata(
    codec: str, width: int, height: int, precision: int, num_channels: int
) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("nvImageCodec reported invalid image dimensions")

    max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
    pixels = width * height
    if max_pixels > 0 and pixels > max_pixels:
        raise ValueError(
            f"Image dimensions {width}x{height} ({pixels} pixels) exceed "
            f"the maximum of {max_pixels} pixels. Set VLLM_MAX_IMAGE_PIXELS "
            "to increase this limit."
        )
    if not 1 <= precision <= 8:
        raise ValueError(
            f"nvImageCodec {codec} precision {precision} is not supported by "
            "the RGB uint8 output contract"
        )
    if not 1 <= num_channels <= 4:
        raise ValueError("nvImageCodec reported an invalid image channel count")


def preflight_image_nvimagecodec(
    data: bytes,
    *,
    image_mode: str | None = "RGB",
    output_layout: NvImageCodecOutputLayout = "hwc_rgb",
) -> NvImageCodecInput | None:
    """Classify encoded bytes for native decode without decoding pixels.

    Returns ``None`` when Pillow owns the image semantics. Invalid images raise
    ``ValueError`` and never enter a native decode wave.
    """
    if output_layout not in ("hwc_rgb", "chw_rgb"):
        raise ValueError(f"Unknown nvImageCodec output layout: {output_layout!r}")
    if image_mode not in (None, "RGB"):
        return None

    signed_codec = _signed_native_codec(data)
    bmp_header = None
    if signed_codec is not None:
        codec, _ = signed_codec
        if codec == "jpeg" and not jpeg_has_complete_scan_and_eoi(data):
            raise ValueError(
                "Invalid JPEG image: malformed or incomplete marker stream"
            )
        if codec == "png" and not _png_has_complete_iend(data):
            raise ValueError("Invalid PNG image: incomplete or missing IEND chunk")
        if (
            codec == "jpeg2k"
            and data.startswith(_JPEG2000_SIGNATURE)
            and not _jp2_has_valid_box_structure(data)
        ):
            raise ValueError("Invalid JPEG 2000 image: malformed JP2 box structure")
        if codec == "bmp":
            bmp_header = _parse_bmp_header(data)
            if bmp_header is None:
                raise ValueError(
                    "Invalid BMP image: truncated data or unsupported DIB header"
                )

    nvimgcodec = _load_nvimgcodec()
    try:
        code_stream = nvimgcodec.CodeStream(data)
        codec = str(code_stream.codec_name).lower()
    except MemoryError:
        raise
    except Exception as exc:
        if signed_codec is not None:
            codec, label = signed_codec
            if _pillow_can_decode(data):
                return None
            raise ValueError(f"Invalid {label} image: {exc}") from exc
        return None
    if codec not in _NATIVE_CODECS:
        return None

    try:
        if codec == "tiff":
            code_stream = code_stream.get_sub_code_stream(0)
        width = int(code_stream.width)
        height = int(code_stream.height)
        precision = int(code_stream.precision)
        num_channels = int(code_stream.num_channels)
    except MemoryError:
        raise
    except Exception as exc:
        if _pillow_can_decode(data):
            return None
        raise ValueError(f"Invalid {codec} image metadata: {exc}") from exc

    if min(width, height, precision, num_channels) <= 0 and _pillow_can_decode(data):
        return None
    if codec == "tiff" and precision == 16:
        palette_depth = _palette_tiff_bit_depth(data)
        if palette_depth == 8:
            # TIFF ColorMap entries are 16-bit even when the palette indices
            # and decoded RGB samples are 8-bit; nvImageCodec reports the former.
            precision = 8
        elif palette_depth is not None:
            return None
    _validate_metadata(codec, width, height, precision, num_channels)
    if width * height > _MAX_NATIVE_PIXELS:
        raise ValueError(
            f"Image dimensions {width}x{height} exceed the nvImageCodec "
            f"frontend's fixed raster ceiling of {_MAX_NATIVE_PIXELS} pixels"
        )

    if codec == "jpeg2k" and num_channels == 4:
        return None
    if codec == "bmp" and bmp_header != (24, 0):
        return None
    if codec == "pnm" and _pnm_maxval(data) not in (None, 255):
        return None
    orientation, use_pillow = _pillow_semantics(
        data, codec, num_channels, preserve_mode=image_mode is None
    )
    if use_pillow:
        return None

    return NvImageCodecInput(
        data, code_stream, width, height, orientation, output_layout
    )
