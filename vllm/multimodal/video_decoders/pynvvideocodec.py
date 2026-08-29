# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
import threading
from collections import Counter
from contextlib import contextmanager, suppress
from typing import ClassVar, NamedTuple, cast

import numpy as np
import numpy.typing as npt

from vllm.logger import init_logger
from vllm.utils.mem_constants import MiB_bytes

from .base import (
    PYNVVIDEOCODEC_DEFAULT_HW_DECODERS,
    PYNVVIDEOCODEC_DEFAULT_OUTPUT_LAYOUT,
    PyNvVideoCodecOutputLayout,
    VideoSourceMetadata,
    VideoTargetMetadata,
    check_frame_pixel_limit,
)
from .capability import VideoResizeTarget

logger = init_logger(__name__)


def decode_pynvvideocodec(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    hw_decoders: int = PYNVVIDEOCODEC_DEFAULT_HW_DECODERS,
    output_layout: PyNvVideoCodecOutputLayout = PYNVVIDEOCODEC_DEFAULT_OUTPUT_LAYOUT,
    gpu_resize: bool = False,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    PyNvVideoCodecVideoBackendMixin._configure_decoder_slots(
        hw_decoders,
        output_layout,
        gpu_resize=gpu_resize,
    )
    return PyNvVideoCodecVideoBackendMixin.decode_frames_pynvvideocodec(
        loader_cls,
        data,
        target,
        **sampling_kwargs,
    )


class PyNvVideoCodecSourceMetadata(NamedTuple):
    """Metadata needed before GPU video decode."""

    source: VideoSourceMetadata
    width: int
    height: int


# Per-decoder upper bound reserved for persistent PyNvVideoCodec surfaces.
PYNVVIDEOCODEC_DECODER_GPU_MEMORY_BYTES = 128 * MiB_bytes
PYNVVIDEOCODEC_DECODER_CACHE_SIZE = 2
# Per-API-server CUDA context and driver allocation, measured with
# PyNvVideoCodec 2.0.4 on H100.
PYNVVIDEOCODEC_CUDA_CONTEXT_BYTES = int(1.8 * 1024 * MiB_bytes)


def _load_cvcuda():
    if os.environ.get("VLLM_PYNVVIDEOCODEC_NO_CVCUDA") == "1":
        return None
    try:
        import cvcuda
    except Exception:
        return None
    if not hasattr(cvcuda, "hq_resize"):
        logger.info(
            "PyNvVideoCodec: cvcuda is installed without hq_resize; using "
            "the torch GPU resize path."
        )
        return None
    return cvcuda


_CVCUDA = None
_CVCUDA_RESOLVED = False
_RESIZE_COUNTERS: Counter[str] = Counter()
_RESIZE_COUNTERS_LOCK = threading.Lock()


def _cvcuda():
    global _CVCUDA, _CVCUDA_RESOLVED
    if not _CVCUDA_RESOLVED:
        _CVCUDA = _load_cvcuda()
        _CVCUDA_RESOLVED = True
    return _CVCUDA


def _count_resize(name: str, amount: int = 1) -> None:
    with _RESIZE_COUNTERS_LOCK:
        _RESIZE_COUNTERS[name] += amount


def get_pynvvideocodec_resize_stats() -> dict[str, int]:
    with _RESIZE_COUNTERS_LOCK:
        return dict(_RESIZE_COUNTERS)


def validate_pynvvideocodec_hw_decoders(hw_decoders: object) -> int:
    if (
        isinstance(hw_decoders, bool)
        or not isinstance(hw_decoders, int)
        or hw_decoders < 1
    ):
        raise ValueError("hw_decoders must be a positive integer")
    return hw_decoders


def validate_pynvvideocodec_output_layout(
    output_layout: object,
) -> PyNvVideoCodecOutputLayout:
    if output_layout not in ("thwc", "tchw"):
        raise ValueError("output_layout must be either 'thwc' or 'tchw'")
    return cast(PyNvVideoCodecOutputLayout, output_layout)


def configure_pynvvideocodec_decoder_pool(
    hw_decoders: object = PYNVVIDEOCODEC_DEFAULT_HW_DECODERS,
    output_layout: object = PYNVVIDEOCODEC_DEFAULT_OUTPUT_LAYOUT,
    *,
    gpu_resize: bool = False,
    resize_target: VideoResizeTarget | None = None,
) -> None:
    """Freeze process-wide decoder capacity, layout, and resize behavior."""
    validated_hw_decoders = validate_pynvvideocodec_hw_decoders(hw_decoders)
    validated_output_layout = validate_pynvvideocodec_output_layout(output_layout)
    if not isinstance(gpu_resize, bool):
        raise ValueError("gpu_resize must be a boolean")
    if gpu_resize and resize_target is None and _pynv_decoder_pool.max_slots is None:
        raise RuntimeError(
            "PyNvVideoCodec gpu_resize must be configured at server startup"
        )
    if gpu_resize and _pynv_decoder_pool.max_slots is None:
        if _cvcuda() is None:
            logger.warning(
                "PyNvVideoCodec: gpu_resize is enabled without CV-CUDA "
                "HQResize; using the slower torch GPU resize fallback."
            )
        else:
            logger.info("PyNvVideoCodec: gpu_resize is using CV-CUDA HQResize.")
    _pynv_decoder_pool.configure(
        validated_hw_decoders,
        validated_output_layout,
        gpu_resize,
        resize_target,
    )


def _pynvvideocodec_exception_types(nvc) -> tuple[type[Exception], ...]:
    return tuple(
        exception_type
        for name in dir(nvc)
        if name.startswith("PyNvVCException")
        and isinstance((exception_type := getattr(nvc, name)), type)
        and issubclass(exception_type, Exception)
    )


def _pynvvc_frames_to_pinned_host(
    decoded_frames,
    output_layout: PyNvVideoCodecOutputLayout,
    stream,
    *,
    resize_target: VideoResizeTarget | None = None,
    cvstream=None,
) -> npt.NDArray:
    """Copy individual PyNvVideoCodec frames directly to one host batch."""
    import torch

    torch_frames = []
    synchronized = False
    try:
        for frame in decoded_frames:
            torch_frames.append(torch.from_dlpack(frame))
        if not torch_frames:
            stream.synchronize()
            synchronized = True
            return np.empty((0,), dtype=np.uint8)

        expected_device = torch.device(stream.device)
        expected_shape = tuple(torch_frames[0].shape)
        channel_dim = 0 if output_layout == "tchw" else 2
        expected_frame_layout = "CHW" if output_layout == "tchw" else "HWC"

        for index, frame in enumerate(torch_frames):
            shape = tuple(frame.shape)
            if frame.ndim != 3:
                raise ValueError(
                    "PyNvVideoCodec returned frame "
                    f"{index} with unexpected shape {shape}; expected "
                    f"{expected_frame_layout} with three channels"
                )
            if frame.dtype != torch.uint8:
                raise ValueError(
                    "PyNvVideoCodec returned frame "
                    f"{index} with unexpected dtype {frame.dtype}; "
                    "expected torch.uint8"
                )
            if frame.device != expected_device:
                raise ValueError(
                    "PyNvVideoCodec returned frame "
                    f"{index} on {frame.device}; expected {expected_device}"
                )
            if not frame.is_contiguous():
                raise ValueError(
                    "PyNvVideoCodec returned non-contiguous frame "
                    f"{index} with stride {tuple(frame.stride())}"
                )
            if shape != expected_shape:
                raise ValueError(
                    "PyNvVideoCodec returned frames with inconsistent shapes: "
                    f"frame 0 is {expected_shape}, frame {index} is {shape}"
                )
            if any(dimension <= 0 for dimension in shape) or shape[channel_dim] != 3:
                raise ValueError(
                    "PyNvVideoCodec returned frame "
                    f"{index} with unexpected shape {shape}; expected "
                    f"{expected_frame_layout} with three channels"
                )

        if output_layout == "tchw":
            height, width = expected_shape[1:]
        else:
            height, width = expected_shape[:2]

        if resize_target is not None:
            target = resize_target(width, height, len(torch_frames))
            if target is not None:
                target_width, target_height = target
                if target_width * target_height < width * height:
                    torch_frames = _resize_pynvvc_frames(
                        torch_frames,
                        output_layout,
                        (target_width, target_height),
                        cvstream=cvstream,
                    )
                    expected_shape = (
                        (3, target_height, target_width)
                        if output_layout == "tchw"
                        else (target_height, target_width, 3)
                    )
                    _count_resize("gpu_resized", len(torch_frames))
                else:
                    _count_resize("resize_not_downscale")
            else:
                _count_resize("resize_declined")

        for index, frame in enumerate(torch_frames):
            if (
                tuple(frame.shape) != expected_shape
                or frame.dtype != torch.uint8
                or frame.device != expected_device
                or not frame.is_contiguous()
            ):
                raise ValueError(
                    "PyNvVideoCodec GPU resize returned frame "
                    f"{index} with shape {tuple(frame.shape)}, dtype "
                    f"{frame.dtype}, device {frame.device}, and stride "
                    f"{tuple(frame.stride())}; expected contiguous "
                    f"{expected_shape} torch.uint8 on {expected_device}"
                )

        host_frames = torch.empty(
            (len(torch_frames), *expected_shape),
            dtype=torch.uint8,
            device="cpu",
            pin_memory=True,
        )
        for index, frame in enumerate(torch_frames):
            host_frames[index].copy_(frame, non_blocking=True)

        stream.synchronize()
        synchronized = True
        return host_frames.numpy()
    finally:
        # A failed DLPack conversion or copy may still leave decoder work in
        # flight. Keep both PyNv and Torch wrappers alive until the stream is
        # safe before the decoder slot can be reused or invalidated.
        if not synchronized:
            with suppress(BaseException):
                stream.synchronize()


def _resize_pynvvc_frames(
    frames,
    output_layout: PyNvVideoCodecOutputLayout,
    target: tuple[int, int],
    *,
    cvstream=None,
):
    """Resize decoded frames without materializing a full device batch."""
    import torch

    target_width, target_height = target
    cv = _cvcuda()
    if cv is not None and cvstream is not None:
        layout = "CHW" if output_layout == "tchw" else "HWC"
        resized = [
            torch.as_tensor(
                cv.hq_resize(
                    cv.as_tensor(frame, layout),
                    (target_height, target_width),
                    antialias=True,
                    interpolation=cv.Interp.CUBIC,
                    stream=cvstream,
                ).cuda(),
                device=frame.device,
            )
            for frame in frames
        ]
        _count_resize("resize_cvcuda", len(resized))
        return resized

    import torch.nn.functional as F

    resized = []
    for frame in frames:
        chw = frame if output_layout == "tchw" else frame.permute(2, 0, 1)
        output = F.interpolate(
            chw.unsqueeze(0).to(torch.float16),
            size=(target_height, target_width),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        output = output.clamp_(0, 255).to(torch.uint8).squeeze(0)
        if output_layout == "thwc":
            output = output.permute(1, 2, 0).contiguous()
        resized.append(output)
    _count_resize("resize_torch", len(resized))
    return resized


class PyNvVideoCodecDecoderSlot:
    """A retained PyNv decoder slot and its CUDA stream.

    The decoder is reused across requests: ``reconfigure_decoder`` repoints the
    existing decoder at each new source instead of paying a fresh
    ``SimpleDecoder`` construction per request. Construction (CUVID parser +
    decoder + surface-pool allocation) is the dominant per-request cost, so
    reconfiguring is far cheaper. A single decoder serves both metadata
    (``len``/``get_stream_metadata``) and frame decode -- no separate
    metadata decoder.
    """

    def __init__(
        self,
        stream,
        output_layout: PyNvVideoCodecOutputLayout = (
            PYNVVIDEOCODEC_DEFAULT_OUTPUT_LAYOUT
        ),
        cvstream=None,
    ) -> None:
        self.stream = stream
        self.cvstream = cvstream
        self.output_layout = validate_pynvvideocodec_output_layout(output_layout)
        self.decoder = None
        self.source_path: str | None = None

    def invalidate(self) -> None:
        self.decoder = None
        self.source_path = None

    def close(self) -> None:
        with suppress(BaseException):
            if self.cvstream is not None:
                self.cvstream.sync()
            elif self.stream is not None:
                self.stream.synchronize()
        self.invalidate()
        self.stream = None
        self.cvstream = None

    def _construct(self, file_path: str, nvc, device_index: int) -> None:
        self.invalidate()
        color_type_name = "RGBP" if self.output_layout == "tchw" else "RGB"
        try:
            output_color_type = getattr(nvc.OutputColorType, color_type_name)
        except AttributeError:
            raise RuntimeError(
                "The installed PyNvVideoCodec does not support "
                f"OutputColorType.{color_type_name}, required for "
                f"output_layout={self.output_layout!r}."
            ) from None
        decoder = nvc.SimpleDecoder(
            file_path,
            output_color_type=output_color_type,
            use_device_memory=True,
            need_scanned_stream_metadata=True,
            gpu_id=device_index,
            cuda_stream=self.stream.cuda_stream,
            decoder_cache_size=PYNVVIDEOCODEC_DECODER_CACHE_SIZE,
        )
        self.decoder = decoder
        self.source_path = file_path

    def get_decoder(self, file_path: str, nvc, device_index: int):
        if self.decoder is None:
            self._construct(file_path, nvc, device_index)
        elif self.source_path != file_path:
            try:
                self.decoder.reconfigure_decoder(file_path)
                self.source_path = file_path
            except Exception:
                # reconfigure unsupported/unsafe for this source -> rebuild.
                self._construct(file_path, nvc, device_index)
        return self.decoder


class _PyNvDecoderPool:
    """Process-wide singleton managing PyNvVideoCodec decoder slot state.

    Prevents subclass counter shadowing (GHSA-j682-9xp5-rrf3) by storing
    all mutable pool state in a single module-level instance rather than
    in ClassVar attributes that get shadowed by Python's augmented
    assignment semantics on subclasses.
    """

    def __init__(self) -> None:
        self.slots: list[PyNvVideoCodecDecoderSlot] = []
        self.active: int = 0
        self.cond: threading.Condition = threading.Condition()
        self.max_slots: int | None = None
        self.output_layout: PyNvVideoCodecOutputLayout | None = None
        self.gpu_resize: bool | None = None
        self.resize_target: VideoResizeTarget | None = None
        self.generation: int = 0

    def configure(
        self,
        hw_decoders: int,
        output_layout: PyNvVideoCodecOutputLayout,
        gpu_resize: bool,
        resize_target: VideoResizeTarget | None,
    ) -> None:
        with self.cond:
            if self.max_slots is None:
                self.max_slots = hw_decoders
                self.output_layout = output_layout
                self.gpu_resize = gpu_resize
                self.resize_target = resize_target
            elif self.max_slots != hw_decoders:
                raise RuntimeError(
                    "PyNvVideoCodec decoder count is already configured as "
                    f"{self.max_slots}, got {hw_decoders}"
                )
            elif self.output_layout != output_layout:
                raise RuntimeError(
                    "PyNvVideoCodec output layout is already configured as "
                    f"{self.output_layout!r}, got {output_layout!r}"
                )
            elif self.gpu_resize != gpu_resize:
                raise RuntimeError(
                    "PyNvVideoCodec gpu_resize is already configured as "
                    f"{self.gpu_resize}, got {gpu_resize}"
                )
            elif resize_target is not None and self.resize_target is not resize_target:
                raise RuntimeError("PyNvVideoCodec resize target is already configured")

    def shutdown(self) -> None:
        with self.cond:
            slots = self.slots
            self.slots = []
            self.active = 0
            self.max_slots = None
            self.output_layout = None
            self.gpu_resize = None
            self.resize_target = None
            self.generation += 1
            self.cond.notify_all()
        for slot in slots:
            slot.close()


_pynv_decoder_pool = _PyNvDecoderPool()


def shutdown_pynvvideocodec_decoder_pool() -> None:
    _pynv_decoder_pool.shutdown()
    with _RESIZE_COUNTERS_LOCK:
        stats = dict(_RESIZE_COUNTERS)
        _RESIZE_COUNTERS.clear()
    if stats:
        logger.info("PyNvVideoCodec GPU resize outcomes: %s", stats)


class PyNvVideoCodecVideoBackendMixin:
    """PyNvVideoCodec utilities for GPU-backed frame decode."""

    _DEVICE_INDEX: ClassVar[int] = 0

    @classmethod
    def _create_decoder_slot(cls) -> PyNvVideoCodecDecoderSlot:
        import torch

        output_layout = _pynv_decoder_pool.output_layout
        if output_layout is None:
            raise RuntimeError("PyNvVideoCodec output layout is not configured")
        if _pynv_decoder_pool.resize_target is not None and (cv := _cvcuda()):
            torch.accelerator.set_device_index(cls._DEVICE_INDEX)
            cvstream = cv.Stream()
            stream = torch.cuda.ExternalStream(
                cvstream.handle,
                device=cls._DEVICE_INDEX,
            )
            return PyNvVideoCodecDecoderSlot(
                stream,
                output_layout,
                cvstream=cvstream,
            )
        return PyNvVideoCodecDecoderSlot(
            torch.cuda.Stream(device=cls._DEVICE_INDEX),
            output_layout,
        )

    @classmethod
    def _configure_decoder_slots(
        cls,
        hw_decoders: object,
        output_layout: object = PYNVVIDEOCODEC_DEFAULT_OUTPUT_LAYOUT,
        *,
        gpu_resize: bool = False,
        resize_target: VideoResizeTarget | None = None,
    ) -> None:
        configure_pynvvideocodec_decoder_pool(
            hw_decoders,
            output_layout,
            gpu_resize=gpu_resize,
            resize_target=resize_target,
        )

    @staticmethod
    @contextmanager
    def _torch_stream_context(stream):
        import torch

        torch.accelerator.set_device_index(stream.device.index)
        previous_stream = torch.accelerator.current_stream()
        torch.accelerator.set_stream(stream)
        try:
            yield
        finally:
            torch.accelerator.set_stream(previous_stream)

    @classmethod
    @contextmanager
    def _borrow_decoder_slot(cls):
        pool = _pynv_decoder_pool
        create_slot = False
        with pool.cond:
            if pool.max_slots is None:
                raise RuntimeError("PyNvVideoCodec decoder slots are not configured")
            while True:
                if pool.max_slots is None:
                    raise RuntimeError(
                        "PyNvVideoCodec decoder slots are not configured"
                    )
                if pool.slots:
                    slot = pool.slots.pop()
                    break
                if pool.active < pool.max_slots:
                    pool.active += 1
                    create_slot = True
                    break
                pool.cond.wait()
            generation = pool.generation

        if create_slot:
            try:
                slot = cls._create_decoder_slot()
            except Exception:
                with pool.cond:
                    if generation == pool.generation:
                        pool.active -= 1
                    pool.cond.notify()
                raise

        borrow_succeeded = False
        try:
            yield slot
            borrow_succeeded = True
        finally:
            if not borrow_succeeded:
                slot.invalidate()
            close_slot = False
            with pool.cond:
                if generation == pool.generation and pool.max_slots is not None:
                    pool.slots.append(slot)
                else:
                    close_slot = True
                pool.cond.notify()
            if close_slot:
                slot.close()

    @staticmethod
    def _metadata_value(metadata, *names: str, default=None):
        for name in names:
            value = getattr(metadata, name, None)
            if value is not None:
                return value
        return default

    @classmethod
    def _read_source_metadata(
        cls,
        file_path: str,
        nvc,
    ) -> PyNvVideoCodecSourceMetadata:
        with cls._borrow_decoder_slot() as decoder_slot:
            with cls._torch_stream_context(decoder_slot.stream):
                decoder = decoder_slot.get_decoder(
                    file_path, nvc, device_index=cls._DEVICE_INDEX
                )
                metadata = decoder.get_stream_metadata()
                total_frames_num = len(decoder)
            width = int(cls._metadata_value(metadata, "width", default=0))
            height = int(cls._metadata_value(metadata, "height", default=0))
            original_fps = float(
                cls._metadata_value(
                    metadata,
                    "average_fps",
                    "avg_frame_rate",
                    "frame_rate",
                    "frameRate",
                    default=0.0,
                )
            )
            duration = float(
                cls._metadata_value(metadata, "duration", default=0.0)
                or (total_frames_num / original_fps if original_fps > 0 else 0.0)
            )
            if total_frames_num <= 0:
                raise ValueError("Could not determine video frame count")
            if width <= 0 or height <= 0:
                raise ValueError("Could not determine video dimensions")
            return PyNvVideoCodecSourceMetadata(
                source=VideoSourceMetadata(total_frames_num, original_fps, duration),
                width=width,
                height=height,
            )

    @classmethod
    def _decode_to_pinned_host(
        cls,
        file_path: str,
        frame_idx: list[int],
        nvc,
    ) -> npt.NDArray:
        if not frame_idx:
            return np.empty((0,), dtype=np.uint8)

        with cls._borrow_decoder_slot() as decoder_slot:
            stream = decoder_slot.stream
            with cls._torch_stream_context(stream):
                try:
                    decode_submitted = True
                    try:
                        decoder = decoder_slot.get_decoder(
                            file_path, nvc, device_index=cls._DEVICE_INDEX
                        )
                        decoded_frames = decoder.get_batch_frames_by_index(frame_idx)
                    except Exception as exc:
                        if not isinstance(
                            exc,
                            _pynvvideocodec_exception_types(nvc) + (IndexError,),
                        ):
                            raise
                        raise ValueError("Invalid or unsupported video file.") from exc
                    if len(decoded_frames) < len(frame_idx):
                        logger.warning(
                            "pynvvideocodec video loading: expected %d frames "
                            "but got %d.",
                            len(frame_idx),
                            len(decoded_frames),
                        )
                    # The helper now owns synchronization for both success and
                    # failure while retaining the decoded frame references.
                    decode_submitted = False
                    return _pynvvc_frames_to_pinned_host(
                        decoded_frames,
                        decoder_slot.output_layout,
                        stream,
                        resize_target=_pynv_decoder_pool.resize_target,
                        cvstream=decoder_slot.cvstream,
                    )
                finally:
                    if decode_submitted:
                        with suppress(BaseException):
                            stream.synchronize()

    @classmethod
    def decode_frames_pynvvideocodec(
        cls,
        loader_cls,
        data: bytes,
        target: VideoTargetMetadata,
        **kwargs,
    ) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
        import PyNvVideoCodec as nvc

        from vllm.multimodal.gpu_ipc_memory import get_mm_gpu_ipc_pool

        temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
        try:
            with os.fdopen(temp_fd, "wb") as temp_file:
                temp_file.write(data)

            try:
                gpu_source = cls._read_source_metadata(temp_path, nvc)
            except Exception as exc:
                if not isinstance(exc, _pynvvideocodec_exception_types(nvc)):
                    raise
                raise ValueError("Invalid or unsupported video file.") from exc
            check_frame_pixel_limit(gpu_source.width, gpu_source.height)
            source = loader_cls._prepare_source(gpu_source.source)
            frame_idx = loader_cls.compute_frames_index_to_sample(
                source=source, target=target, **kwargs
            )
            raw_frame_bytes = len(frame_idx) * gpu_source.height * gpu_source.width * 3
            leased_frame_bytes = raw_frame_bytes
            if _pynv_decoder_pool.resize_target is not None:
                # The decoder surfaces and resized outputs overlap until the
                # asynchronous host copies complete. A second raw-raster budget
                # is a conservative upper bound because only downscales run.
                leased_frame_bytes *= 2
            pool = get_mm_gpu_ipc_pool()
            if pool is None or leased_frame_bytes == 0:
                frames = cls._decode_to_pinned_host(temp_path, frame_idx, nvc)
            else:
                with pool.acquire(leased_frame_bytes):
                    frames = cls._decode_to_pinned_host(temp_path, frame_idx, nvc)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)

        valid_frame_indices = frame_idx[: int(frames.shape[0])]
        return frames, source, frame_idx, valid_frame_indices
