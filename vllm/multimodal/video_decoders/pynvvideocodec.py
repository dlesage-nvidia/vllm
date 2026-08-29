# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import tempfile
import threading
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

logger = init_logger(__name__)


def decode_pynvvideocodec(
    loader_cls,
    data: bytes,
    target: VideoTargetMetadata,
    sampling_kwargs: dict,
    *,
    hw_decoders: int = PYNVVIDEOCODEC_DEFAULT_HW_DECODERS,
    output_layout: PyNvVideoCodecOutputLayout = PYNVVIDEOCODEC_DEFAULT_OUTPUT_LAYOUT,
) -> tuple[npt.NDArray, VideoSourceMetadata, list[int], list[int]]:
    PyNvVideoCodecVideoBackendMixin._configure_decoder_slots(hw_decoders, output_layout)
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
) -> None:
    """Freeze the process-wide decoder capacity and output layout."""
    validated_hw_decoders = validate_pynvvideocodec_hw_decoders(hw_decoders)
    validated_output_layout = validate_pynvvideocodec_output_layout(output_layout)
    _pynv_decoder_pool.configure(validated_hw_decoders, validated_output_layout)


def _pynvvideocodec_exception_types(nvc) -> tuple[type[Exception], ...]:
    return tuple(
        exception_type
        for name in dir(nvc)
        if name.startswith("PyNvVCException")
        and isinstance((exception_type := getattr(nvc, name)), type)
        and issubclass(exception_type, Exception)
    )


def _pynvvc_frames_to_nhwc(frames):
    """Return a stacked PyNvVideoCodec RGB frame batch as contiguous THWC."""
    if frames.shape[-1] == 3:
        return frames.contiguous()
    if frames.shape[1] == 3:
        return frames.permute(0, 2, 3, 1).contiguous()
    raise ValueError(
        "PyNvVideoCodec RGB returned frames with unexpected shape "
        f"{tuple(frames.shape)}; expected THWC or TCHW with three channels"
    )


def _pynvvc_frames_to_tchw(frames):
    """Validate a stacked PyNvVideoCodec RGBP frame batch as contiguous TCHW."""
    if frames.shape[1] != 3:
        raise ValueError(
            "PyNvVideoCodec RGBP returned frames with unexpected shape "
            f"{tuple(frames.shape)}; expected TCHW with three channels"
        )
    return frames.contiguous()


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
    ) -> None:
        self.stream = stream
        self.output_layout = validate_pynvvideocodec_output_layout(output_layout)
        self.decoder = None
        self.source_path: str | None = None

    def invalidate(self) -> None:
        self.decoder = None
        self.source_path = None

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

    def configure(
        self,
        hw_decoders: int,
        output_layout: PyNvVideoCodecOutputLayout,
    ) -> None:
        with self.cond:
            if self.max_slots is None:
                self.max_slots = hw_decoders
                self.output_layout = output_layout
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


_pynv_decoder_pool = _PyNvDecoderPool()


class PyNvVideoCodecVideoBackendMixin:
    """PyNvVideoCodec utilities for GPU-backed frame decode."""

    _DEVICE_INDEX: ClassVar[int] = 0

    @classmethod
    def _create_decoder_slot(cls) -> PyNvVideoCodecDecoderSlot:
        import torch

        output_layout = _pynv_decoder_pool.output_layout
        if output_layout is None:
            raise RuntimeError("PyNvVideoCodec output layout is not configured")
        return PyNvVideoCodecDecoderSlot(
            torch.cuda.Stream(device=cls._DEVICE_INDEX), output_layout
        )

    @classmethod
    def _configure_decoder_slots(
        cls,
        hw_decoders: object,
        output_layout: object = PYNVVIDEOCODEC_DEFAULT_OUTPUT_LAYOUT,
    ) -> None:
        configure_pynvvideocodec_decoder_pool(hw_decoders, output_layout)

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
                if pool.slots:
                    slot = pool.slots.pop()
                    break
                if pool.active < pool.max_slots:
                    pool.active += 1
                    create_slot = True
                    break
                pool.cond.wait()

        if create_slot:
            try:
                slot = cls._create_decoder_slot()
            except Exception:
                with pool.cond:
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
            with pool.cond:
                pool.slots.append(slot)
                pool.cond.notify()

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
        import torch

        if not frame_idx:
            return np.empty((0,), dtype=np.uint8)

        with cls._borrow_decoder_slot() as decoder_slot:
            stream = decoder_slot.stream
            with cls._torch_stream_context(stream):
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
                        "pynvvideocodec video loading: expected %d frames but got %d.",
                        len(frame_idx),
                        len(decoded_frames),
                    )
                torch_frames = [torch.from_dlpack(frame) for frame in decoded_frames]
                if not torch_frames:
                    return np.empty((0,), dtype=np.uint8)
                device_frames = torch.stack(torch_frames)
                if device_frames.ndim != 4:
                    raise ValueError(
                        "PyNvVideoCodec returned frames with unexpected shape "
                        f"{tuple(device_frames.shape)}"
                    )
                if device_frames.dtype != torch.uint8:
                    raise ValueError(
                        "PyNvVideoCodec returned frames with unexpected dtype "
                        f"{device_frames.dtype}; expected torch.uint8"
                    )
                if decoder_slot.output_layout == "tchw":
                    device_frames = _pynvvc_frames_to_tchw(device_frames)
                else:
                    device_frames = _pynvvc_frames_to_nhwc(device_frames)
                host_frames = torch.empty(
                    device_frames.shape,
                    dtype=device_frames.dtype,
                    device="cpu",
                    pin_memory=True,
                )
                host_frames.copy_(device_frames, non_blocking=True)
                stream.synchronize()
                host_array = host_frames.numpy()
                del decoded_frames, torch_frames, device_frames
                return host_array

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
            pool = get_mm_gpu_ipc_pool()
            if pool is None or raw_frame_bytes == 0:
                frames = cls._decode_to_pinned_host(temp_path, frame_idx, nvc)
            else:
                with pool.acquire(raw_frame_bytes):
                    frames = cls._decode_to_pinned_host(temp_path, frame_idx, nvc)
        finally:
            with suppress(FileNotFoundError):
                os.unlink(temp_path)

        valid_frame_indices = frame_idx[: int(frames.shape[0])]
        return frames, source, frame_idx, valid_frame_indices
