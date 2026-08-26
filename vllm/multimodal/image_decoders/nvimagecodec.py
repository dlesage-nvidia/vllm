# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
from collections import deque
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from typing import Any, ClassVar, Literal

import numpy as np
from PIL import Image

from vllm.logger import init_logger
from vllm.multimodal.image import check_image_pixel_limit
from vllm.utils.mem_constants import MiB_bytes

logger = init_logger(__name__)

PILLOW_IMAGE_BACKEND = "pillow"
NVIMAGECODEC_IMAGE_BACKEND = "nvimagecodec"
NVIMAGECODEC_DEFAULT_DECODERS = 2
NVIMAGECODEC_DEFAULT_BATCH_SIZE = 5
NVIMAGECODEC_DEFAULT_PIPELINE_DEPTH = 2
NVIMAGECODEC_MAX_BATCH_SIZE = 64
NVIMAGECODEC_MAX_PIPELINE_DEPTH = 8
NVIMAGECODEC_MAX_CHANNELS = 4

# Largest raster covered by the retained-memory calibration below. Larger
# inputs do not enter the nvImageCodec path.
NVIMAGECODEC_MAX_PIXELS = 178_956_970
# Bound both individual streams and the aggregate payload in one native batch.
NVIMAGECODEC_MAX_ENCODED_BYTES = 64 * MiB_bytes

# Conservative allowance beyond one retained decoded batch. The aggregate
# decoded size is independently capped at four bytes per calibrated pixel.
NVIMAGECODEC_DECODER_WORKSPACE_BYTES = 512 * MiB_bytes
# Per-API-server CUDA context and library-state estimate. gpu_ipc_memory.py
# keeps it additive with other GPU media backends.
NVIMAGECODEC_CUDA_CONTEXT_BYTES = int(1.8 * 1024 * MiB_bytes)

_NVIMAGECODEC_GPU_CODECS = frozenset({"jpeg", "jpeg2k", "tiff"})
_NVIMAGECODEC_CPU_CODECS = frozenset({"bmp", "png", "pnm", "webp"})
_NVIMAGECODEC_CODECS = _NVIMAGECODEC_GPU_CODECS | _NVIMAGECODEC_CPU_CODECS

_OutputMode = Literal["RGB", "RGBA"]


class NvImageCodecBatchItemError(ValueError):
    """A native-batch failure owned by one input position."""

    def __init__(self, index: int, error: Exception) -> None:
        super().__init__(str(error))
        self.index = index
        self.error = error


def _close_decoded_results(results: list[Image.Image | None]) -> None:
    for index, result in enumerate(results):
        try:
            if result is not None:
                result.close()
        except Exception:
            logger.warning("Failed to close an abandoned decoded image", exc_info=True)
        finally:
            results[index] = None


def _get_effective_max_pixels() -> int:
    from vllm import envs

    configured_limit = envs.VLLM_MAX_IMAGE_PIXELS
    return (
        NVIMAGECODEC_MAX_PIXELS
        if configured_limit <= 0
        else min(configured_limit, NVIMAGECODEC_MAX_PIXELS)
    )


def get_nvimagecodec_decoder_gpu_memory_bytes() -> int:
    """Return the per-slot reservation for retained decoder allocations."""
    return (
        _get_effective_max_pixels() * NVIMAGECODEC_MAX_CHANNELS
        + NVIMAGECODEC_DECODER_WORKSPACE_BYTES
    )


def validate_nvimagecodec_decoders(decoders: object) -> int:
    if isinstance(decoders, bool) or not isinstance(decoders, int) or decoders < 1:
        raise ValueError("decoders must be a positive integer")
    return decoders


def validate_nvimagecodec_batch_size(batch_size: object) -> int:
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or not 1 <= batch_size <= NVIMAGECODEC_MAX_BATCH_SIZE
    ):
        raise ValueError(
            f"batch_size must be an integer between 1 and {NVIMAGECODEC_MAX_BATCH_SIZE}"
        )
    return batch_size


def validate_nvimagecodec_pipeline_depth(pipeline_depth: object) -> int:
    if (
        isinstance(pipeline_depth, bool)
        or not isinstance(pipeline_depth, int)
        or not 1 <= pipeline_depth <= NVIMAGECODEC_MAX_PIPELINE_DEPTH
    ):
        raise ValueError(
            "pipeline_depth must be an integer between 1 and "
            f"{NVIMAGECODEC_MAX_PIPELINE_DEPTH}"
        )
    return pipeline_depth


def _validate_output_modes(
    data: Sequence[bytes],
    output_modes: Sequence[_OutputMode] | None,
) -> list[_OutputMode]:
    if output_modes is None:
        return ["RGB"] * len(data)
    if len(output_modes) != len(data):
        raise ValueError("output_modes must have the same length as data")

    modes = list(output_modes)
    if any(mode not in ("RGB", "RGBA") for mode in modes):
        raise ValueError("output_modes entries must be 'RGB' or 'RGBA'")
    return modes


def _load_nvimgcodec():
    try:
        from nvidia import nvimgcodec
    except ImportError as exc:
        raise RuntimeError(
            "The nvImageCodec image backend requires the CUDA-major-matched "
            "nvidia-nvimgcodec package. On x86-64, install vLLM's "
            "'nvimagecodec' extra; on Arm, install the platform-appropriate "
            "NVIDIA package."
        ) from exc
    return nvimgcodec


class NvImageCodecDecoderSlot:
    """Retained GPU and CPU nvImageCodec decoders for one pool slot."""

    def __init__(self, stream=None) -> None:
        self.stream = stream
        self.extra_streams: list[object] = []
        self.gpu_decoder = None
        self.cpu_decoder = None
        self.decode_params: dict[_OutputMode, object] = {}

    def invalidate(self) -> None:
        self.stream = None
        self.extra_streams.clear()
        self.gpu_decoder = None
        self.cpu_decoder = None
        self.decode_params.clear()

    def get_stream(self, index: int = 0):
        if index == 0 and self.stream is None:
            import torch

            self.stream = torch.cuda.Stream(device=NvImageCodecBackend._DEVICE_INDEX)
        if index == 0:
            return self.stream

        import torch

        while len(self.extra_streams) < index:
            self.extra_streams.append(
                torch.cuda.Stream(device=NvImageCodecBackend._DEVICE_INDEX)
            )
        return self.extra_streams[index - 1]

    def get_decode_params(self, nvimgcodec, output_mode: _OutputMode):
        params = self.decode_params.get(output_mode)
        if params is None:
            sample_format = (
                nvimgcodec.SampleFormat.I_RGB
                if output_mode == "RGB"
                else nvimgcodec.SampleFormat.I_RGBA
            )
            params = nvimgcodec.DecodeParams(
                # ImageMediaIO applies the original EXIF orientation once on CPU.
                apply_exif_orientation=False,
                color_spec=nvimgcodec.ColorSpec.SRGB,
                # vLLM's image contract is uint8. Higher-precision inputs are
                # rejected before decoding rather than implicitly rescaled.
                allow_any_depth=False,
                sample_format=sample_format,
            )
            self.decode_params[output_mode] = params
        return params

    def get_gpu_decoder(self, nvimgcodec):
        if self.gpu_decoder is None:
            self.gpu_decoder = nvimgcodec.Decoder(
                device_id=NvImageCodecBackend._DEVICE_INDEX,
                max_num_cpu_threads=1,
                options=":num_cuda_streams=1",
                backends=[
                    nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY),
                    nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY),
                    nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU),
                ],
            )
        return self.gpu_decoder

    def get_cpu_decoder(self, nvimgcodec):
        if self.cpu_decoder is None:
            # Keep CPU_ONLY separate from the GPU decoder. Combining them in one
            # Decoder can deadlock in heterogeneous native batches.
            self.cpu_decoder = nvimgcodec.Decoder(
                device_id=-1,
                max_num_cpu_threads=1,
                backends=[nvimgcodec.Backend(nvimgcodec.BackendKind.CPU_ONLY)],
            )
        return self.cpu_decoder


class _NvImageCodecDecoderPool:
    """Process-wide bounded pool of nvImageCodec decoder slots."""

    def __init__(self) -> None:
        self.owner_pid = os.getpid()
        self.slots: list[NvImageCodecDecoderSlot] = []
        self.active = 0
        self.cond = threading.Condition()
        self.max_slots: int | None = None
        self.batch_size: int | None = None
        self.pipeline_depth: int | None = None
        self.closing = False
        self.generation = 0

    def check_pid(self) -> None:
        pid = os.getpid()
        if self.owner_pid == pid:
            return
        # Do not acquire a condition inherited from the parent. Pristine state
        # has never configured the pool or constructed a native resource and can
        # therefore be replaced safely in the child.
        if (
            self.max_slots is None
            and self.active == 0
            and not self.slots
            and not self.closing
        ):
            self.owner_pid = pid
            self.slots = []
            self.active = 0
            self.cond = threading.Condition()
            self.batch_size = None
            self.pipeline_depth = None
            self.generation = 0
            return
        raise RuntimeError(
            "nvImageCodec decoder state cannot be reused after fork; start API "
            "workers with the spawn multiprocessing method."
        )

    def configure(
        self,
        decoders: int,
        batch_size: int,
        pipeline_depth: int = NVIMAGECODEC_DEFAULT_PIPELINE_DEPTH,
    ) -> None:
        self.check_pid()
        with self.cond:
            if self.closing:
                raise RuntimeError("nvImageCodec decoder pool is shutting down")
            if self.max_slots is None:
                self.max_slots = decoders
                self.batch_size = batch_size
                self.pipeline_depth = pipeline_depth
            elif (
                self.max_slots != decoders
                or self.batch_size != batch_size
                or self.pipeline_depth != pipeline_depth
            ):
                raise RuntimeError(
                    "nvImageCodec decoder pool is already configured as "
                    f"decoders={self.max_slots}, batch_size={self.batch_size}, "
                    f"pipeline_depth={self.pipeline_depth}; got "
                    f"decoders={decoders}, batch_size={batch_size}, "
                    f"pipeline_depth={pipeline_depth}"
                )

    def shutdown(self) -> None:
        self.check_pid()
        with self.cond:
            if self.closing:
                raise RuntimeError(
                    "nvImageCodec decoder pool shutdown did not complete"
                )
            self.closing = True
            self.generation += 1
            self.cond.notify_all()
            while len(self.slots) < self.active:
                self.cond.wait()
            slots = self.slots
            self.slots = []
            self.active = 0
            self.max_slots = None
            self.batch_size = None
            self.pipeline_depth = None
        for slot in slots:
            slot.invalidate()
        with self.cond:
            self.closing = False
            self.cond.notify_all()


_nvimagecodec_decoder_pool = _NvImageCodecDecoderPool()


@dataclass(frozen=True)
class _DecodeItem:
    index: int
    data: bytes
    code_stream: object
    codec_name: str
    width: int
    height: int
    output_mode: _OutputMode

    @property
    def raw_bytes(self) -> int:
        channels = 3 if self.output_mode == "RGB" else 4
        return self.width * self.height * channels


@dataclass
class _PendingGPUChunk:
    items: list[_DecodeItem]
    decoded: list[object | None]
    device_views: list[object | None]
    host_buffers: list[object | None]
    event: Any
    lease: Any | None
    stream: Any
    stream_index: int


class NvImageCodecBackend:
    """nvImageCodec utilities for bounded native-batch image decoding."""

    _DEVICE_INDEX: ClassVar[int] = 0

    @classmethod
    def _create_decoder_slot(cls) -> NvImageCodecDecoderSlot:
        return NvImageCodecDecoderSlot()

    @classmethod
    def _configure_decoder_slots(
        cls,
        decoders: object,
        batch_size: object,
        pipeline_depth: object = NVIMAGECODEC_DEFAULT_PIPELINE_DEPTH,
    ) -> None:
        _nvimagecodec_decoder_pool.configure(
            validate_nvimagecodec_decoders(decoders),
            validate_nvimagecodec_batch_size(batch_size),
            validate_nvimagecodec_pipeline_depth(pipeline_depth),
        )

    @staticmethod
    @contextmanager
    def _torch_stream_context(stream):
        import torch

        previous_device_index = torch.accelerator.current_device_index()
        previous_stream = torch.accelerator.current_stream()
        torch.accelerator.set_device_index(stream.device.index)
        previous_target_stream = torch.accelerator.current_stream()
        torch.accelerator.set_stream(stream)
        try:
            yield
        finally:
            torch.accelerator.set_stream(previous_target_stream)
            torch.accelerator.set_device_index(previous_device_index)
            torch.accelerator.set_stream(previous_stream)

    @classmethod
    @contextmanager
    def _borrow_decoder_slot(cls):
        pool = _nvimagecodec_decoder_pool
        pool.check_pid()
        create_slot = False
        with pool.cond:
            generation = pool.generation
            if pool.closing:
                raise RuntimeError("nvImageCodec decoder pool is shutting down")
            if pool.max_slots is None:
                raise RuntimeError("nvImageCodec decoder slots are not configured")
            while True:
                if pool.closing or pool.generation != generation:
                    raise RuntimeError("nvImageCodec decoder pool is shutting down")
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
                    pool.cond.notify_all()
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
                pool.cond.notify_all()

    @staticmethod
    def _inspect_item(
        data: bytes,
        output_mode: _OutputMode,
        index: int,
        nvimgcodec,
    ) -> _DecodeItem | None:
        if len(data) > NVIMAGECODEC_MAX_ENCODED_BYTES:
            logger.warning_once(
                "Image input exceeds the nvImageCodec encoded-size limit; "
                "falling back to Pillow."
            )
            return None

        try:
            code_stream = nvimgcodec.CodeStream(data)
            codec_name = str(code_stream.codec_name).lower()
            if codec_name not in _NVIMAGECODEC_CODECS:
                return None
            # Pillow exposes the first image in a TIFF by default. Selecting its
            # substream also avoids eagerly walking the complete IFD chain.
            if codec_name == "tiff":
                code_stream = code_stream.get_sub_code_stream(0)
            width = int(code_stream.width)
            height = int(code_stream.height)
            precision = int(code_stream.precision)
            num_channels = int(code_stream.num_channels)
        except Exception:
            logger.warning_once(
                "nvImageCodec could not read image metadata; falling back to Pillow."
            )
            return None

        if width <= 0 or height <= 0:
            return None
        check_image_pixel_limit(width, height)
        if width * height > NVIMAGECODEC_MAX_PIXELS:
            logger.warning_once(
                "Image input exceeds the calibrated nvImageCodec pixel limit; "
                "falling back to Pillow."
            )
            return None
        if not 0 < precision <= 8 or not 1 <= num_channels <= NVIMAGECODEC_MAX_CHANNELS:
            return None

        return _DecodeItem(
            index=index,
            data=data,
            code_stream=code_stream,
            codec_name=codec_name,
            width=width,
            height=height,
            output_mode=output_mode,
        )

    @staticmethod
    def _iter_chunks(
        items: Sequence[_DecodeItem],
        *,
        batch_size: int,
        max_raw_bytes: int,
    ) -> Iterator[list[_DecodeItem]]:
        chunk: list[_DecodeItem] = []
        encoded_bytes = 0
        raw_bytes = 0
        for item in items:
            item_encoded_bytes = len(item.data)
            item_raw_bytes = item.raw_bytes
            if chunk and (
                len(chunk) == batch_size
                or encoded_bytes + item_encoded_bytes > NVIMAGECODEC_MAX_ENCODED_BYTES
                or raw_bytes + item_raw_bytes > max_raw_bytes
            ):
                yield chunk
                chunk = []
                encoded_bytes = 0
                raw_bytes = 0
            chunk.append(item)
            encoded_bytes += item_encoded_bytes
            raw_bytes += item_raw_bytes
        if chunk:
            yield chunk

    @staticmethod
    def _decode_native(decoder, items, params, *, cuda_stream: int):
        decoded = list(
            decoder.decode(
                [item.code_stream for item in items],
                params=params,
                cuda_stream=cuda_stream,
            )
        )
        if len(decoded) != len(items):
            actual_count = len(decoded)
            decoded.clear()
            raise RuntimeError(
                "nvImageCodec returned an unexpected native-batch result count: "
                f"expected {len(items)}, got {actual_count}"
            )
        return decoded

    @staticmethod
    def _decoded_to_pillow(decoded, item: _DecodeItem) -> Image.Image:
        channels = 3 if item.output_mode == "RGB" else 4
        expected_shape = (item.height, item.width, channels)
        host_image = None
        host_array = None
        try:
            if (
                decoded.ndim != 3
                or tuple(decoded.shape) != expected_shape
                or decoded.dtype != np.uint8
            ):
                raise ValueError(
                    "nvImageCodec returned an image with unexpected shape or dtype: "
                    f"shape={tuple(decoded.shape)}, dtype={decoded.dtype}"
                )

            # Image.cpu() synchronizes the image's recorded CUDA stream. It uses a
            # pageable host buffer rather than retaining a pinned allocation.
            host_image = decoded.cpu()
            if host_image is None:
                raise RuntimeError("nvImageCodec failed to copy an image to host")
            host_array = np.asarray(host_image)
            if (
                tuple(host_array.shape) != expected_shape
                or host_array.dtype != np.uint8
            ):
                raise ValueError(
                    "nvImageCodec returned a host image with unexpected "
                    "shape or dtype: "
                    f"shape={tuple(host_array.shape)}, dtype={host_array.dtype}"
                )
            if not host_array.flags.c_contiguous:
                host_array = np.ascontiguousarray(host_array)

            return Image.frombytes(
                item.output_mode,
                (item.width, item.height),
                host_array,
            )
        finally:
            # Exception tracebacks retain frame locals. Drop the source and host
            # images before the caller returns its GPU lease.
            del host_image, host_array, decoded

    @staticmethod
    def _decoded_to_pinned(decoded, item: _DecodeItem):
        """Queue one device-to-pinned-host copy on the current CUDA stream."""
        import torch

        channels = 3 if item.output_mode == "RGB" else 4
        expected_shape = (item.height, item.width, channels)
        device_view = None
        host_buffer = None
        try:
            if (
                decoded.ndim != 3
                or tuple(decoded.shape) != expected_shape
                or decoded.dtype != np.uint8
            ):
                raise ValueError(
                    "nvImageCodec returned an image with unexpected shape or dtype: "
                    f"shape={tuple(decoded.shape)}, dtype={decoded.dtype}"
                )

            device_view = torch.from_dlpack(decoded)
            if (
                tuple(device_view.shape) != expected_shape
                or device_view.dtype != torch.uint8
            ):
                raise ValueError(
                    "nvImageCodec returned a DLPack image with unexpected shape or "
                    f"dtype: shape={tuple(device_view.shape)}, "
                    f"dtype={device_view.dtype}"
                )
            host_buffer = torch.empty(
                expected_shape,
                dtype=torch.uint8,
                device="cpu",
                pin_memory=True,
            )
            host_buffer.copy_(device_view, non_blocking=True)
            return device_view, host_buffer
        except BaseException:
            del host_buffer, device_view, decoded
            raise

    @staticmethod
    def _record_stream_event(stream):
        return stream.record_event()

    @staticmethod
    def _pinned_to_pillow(host_buffer, item: _DecodeItem) -> Image.Image:
        channels = 3 if item.output_mode == "RGB" else 4
        expected_shape = (item.height, item.width, channels)
        host_array = None
        try:
            to_numpy = getattr(host_buffer, "numpy", None)
            host_array = to_numpy() if callable(to_numpy) else np.asarray(host_buffer)
            if (
                tuple(host_array.shape) != expected_shape
                or host_array.dtype != np.uint8
            ):
                raise ValueError(
                    "nvImageCodec returned a pinned host image with unexpected "
                    "shape or dtype: "
                    f"shape={tuple(host_array.shape)}, dtype={host_array.dtype}"
                )
            if not host_array.flags.c_contiguous:
                host_array = np.ascontiguousarray(host_array)
            return Image.frombytes(
                item.output_mode,
                (item.width, item.height),
                host_array,
            )
        finally:
            del host_array, host_buffer

    @classmethod
    def _submit_gpu_chunk(
        cls,
        slot: NvImageCodecDecoderSlot,
        items: Sequence[_DecodeItem],
        nvimgcodec,
        lease,
        stream_index: int,
    ) -> _PendingGPUChunk:
        stream: Any = None
        decoded: list[object | None] = []
        device_views: list[object | None] = []
        host_buffers: list[object | None] = []
        output = None
        device_view = None
        host_buffer = None
        may_have_submitted = False
        try:
            params = slot.get_decode_params(nvimgcodec, items[0].output_mode)
            stream = slot.get_stream(stream_index)
            with cls._torch_stream_context(stream):
                may_have_submitted = True
                decoded = cls._decode_native(
                    slot.get_gpu_decoder(nvimgcodec),
                    items,
                    params,
                    cuda_stream=stream.cuda_stream,
                )
                for output, item in zip(decoded, items):
                    if output is None:
                        device_views.append(None)
                        host_buffers.append(None)
                    else:
                        try:
                            device_view, host_buffer = cls._decoded_to_pinned(
                                output, item
                            )
                        except Exception as error:
                            raise NvImageCodecBatchItemError(
                                item.index, error
                            ) from error
                        device_views.append(device_view)
                        host_buffers.append(host_buffer)
                event = cls._record_stream_event(stream)
            return _PendingGPUChunk(
                items=list(items),
                decoded=decoded,
                device_views=device_views,
                host_buffers=host_buffers,
                event=event,
                lease=lease,
                stream=stream,
                stream_index=stream_index,
            )
        except BaseException:
            try:
                if may_have_submitted and stream is not None:
                    with suppress(BaseException):
                        stream.synchronize()
            finally:
                host_buffers.clear()
                device_views.clear()
                decoded.clear()
                del host_buffer, device_view, output
                lease.release()
            raise

    @staticmethod
    def _clear_pending_device_state(pending: _PendingGPUChunk) -> None:
        pending.device_views.clear()
        pending.decoded.clear()
        if pending.lease is not None:
            pending.lease.release()
            pending.lease = None

    @classmethod
    def _drain_gpu_chunk(
        cls, pending: _PendingGPUChunk
    ) -> tuple[list[_DecodeItem], list[object | None], int]:
        pending.event.synchronize()
        items = pending.items
        host_buffers = pending.host_buffers
        pending.host_buffers = []
        cls._clear_pending_device_state(pending)
        return items, host_buffers, pending.stream_index

    @classmethod
    def _discard_pending_gpu_chunk(cls, pending: _PendingGPUChunk) -> None:
        try:
            pending.event.synchronize()
        except BaseException:
            with suppress(BaseException):
                pending.stream.synchronize()
        finally:
            pending.host_buffers.clear()
            cls._clear_pending_device_state(pending)

    @classmethod
    def _materialize_pinned_chunk(
        cls,
        items: Sequence[_DecodeItem],
        host_buffers: list[object | None],
        results: list[Image.Image | None],
    ) -> list[_DecodeItem]:
        misses: list[_DecodeItem] = []
        host_buffer = None
        try:
            for item, host_buffer in zip(items, host_buffers):
                if host_buffer is None:
                    misses.append(item)
                else:
                    try:
                        results[item.index] = cls._pinned_to_pillow(host_buffer, item)
                    except Exception as error:
                        raise NvImageCodecBatchItemError(item.index, error) from error
        finally:
            host_buffers.clear()
            del host_buffer
        return misses

    @classmethod
    def _decode_cpu_chunk(
        cls,
        slot: NvImageCodecDecoderSlot,
        items: Sequence[_DecodeItem],
        nvimgcodec,
        results: list[Image.Image | None],
    ) -> None:
        if not items:
            return
        params = slot.get_decode_params(nvimgcodec, items[0].output_mode)
        decoded = cls._decode_native(
            slot.get_cpu_decoder(nvimgcodec),
            items,
            params,
            cuda_stream=0,
        )
        output = None
        try:
            for item, output in zip(items, decoded):
                if output is not None:
                    try:
                        results[item.index] = cls._decoded_to_pillow(output, item)
                    except Exception as error:
                        raise NvImageCodecBatchItemError(item.index, error) from error
        finally:
            decoded.clear()
            del output, decoded

    @classmethod
    def _decode_gpu_chunk(
        cls,
        slot: NvImageCodecDecoderSlot,
        items: Sequence[_DecodeItem],
        nvimgcodec,
        memory_pool,
        results: list[Image.Image | None],
    ) -> None:
        params = slot.get_decode_params(nvimgcodec, items[0].output_mode)
        stream = slot.get_stream()
        raw_bytes = sum(item.raw_bytes for item in items)
        misses: list[_DecodeItem] = []
        with (
            memory_pool.acquire(raw_bytes),
            cls._torch_stream_context(stream),
        ):
            decoded = cls._decode_native(
                slot.get_gpu_decoder(nvimgcodec),
                items,
                params,
                cuda_stream=stream.cuda_stream,
            )
            output = None
            try:
                for item, output in zip(items, decoded):
                    if output is None:
                        misses.append(item)
                    else:
                        try:
                            results[item.index] = cls._decoded_to_pillow(output, item)
                        except Exception as error:
                            raise NvImageCodecBatchItemError(
                                item.index, error
                            ) from error
            finally:
                # Drop all device images before returning the raw-byte lease,
                # including when conversion raises partway through the batch.
                decoded.clear()
                del output, decoded

        # A plugin/capability miss is expected and remains positional. Unexpected
        # Python/CUDA exceptions above propagate and invalidate the whole slot.
        cls._decode_cpu_chunk(slot, misses, nvimgcodec, results)

    @classmethod
    def _decode_gpu_chunks_pipelined(
        cls,
        slot: NvImageCodecDecoderSlot,
        chunks: Sequence[Sequence[_DecodeItem]],
        nvimgcodec,
        memory_pool,
        results: list[Image.Image | None],
        *,
        pipeline_depth: int,
    ) -> None:
        """Overlap native decode, pinned D2H, and Pillow materialization."""
        chunk_list = [list(chunk) for chunk in chunks]
        pending: deque[_PendingGPUChunk] = deque()
        free_stream_indices = deque(range(pipeline_depth))
        miss_batches: list[list[_DecodeItem]] = []
        next_chunk = 0

        try:
            while next_chunk < len(chunk_list) or pending:
                # Fill the ring. Once this worker owns a lease, acquisition must
                # stay nonblocking so a pool that fits only one chunk cannot
                # deadlock waiting for memory that only this worker can release.
                while next_chunk < len(chunk_list) and free_stream_indices:
                    chunk = chunk_list[next_chunk]
                    raw_bytes = sum(item.raw_bytes for item in chunk)
                    lease = (
                        memory_pool.try_acquire(raw_bytes)
                        if pending
                        else memory_pool.acquire(raw_bytes)
                    )
                    if lease is None:
                        break
                    stream_index = free_stream_indices.popleft()
                    try:
                        submitted = cls._submit_gpu_chunk(
                            slot,
                            chunk,
                            nvimgcodec,
                            lease,
                            stream_index,
                        )
                    except BaseException:
                        free_stream_indices.appendleft(stream_index)
                        raise
                    pending.append(submitted)
                    next_chunk += 1

                if not pending:
                    continue

                oldest = pending.popleft()
                try:
                    items, host_buffers, stream_index = cls._drain_gpu_chunk(oldest)
                except BaseException:
                    cls._discard_pending_gpu_chunk(oldest)
                    raise
                free_stream_indices.append(stream_index)

                # Refill the freed GPU entry before doing CPU work. This attempt
                # is deliberately nonblocking even when the ring became empty;
                # materializing the ready host batch is more useful than waiting
                # for another decoder worker to release global GPU capacity.
                while next_chunk < len(chunk_list) and free_stream_indices:
                    chunk = chunk_list[next_chunk]
                    raw_bytes = sum(item.raw_bytes for item in chunk)
                    lease = memory_pool.try_acquire(raw_bytes)
                    if lease is None:
                        break
                    refill_stream_index = free_stream_indices.popleft()
                    try:
                        submitted = cls._submit_gpu_chunk(
                            slot,
                            chunk,
                            nvimgcodec,
                            lease,
                            refill_stream_index,
                        )
                    except BaseException:
                        free_stream_indices.appendleft(refill_stream_index)
                        raise
                    pending.append(submitted)
                    next_chunk += 1

                misses = cls._materialize_pinned_chunk(
                    items,
                    host_buffers,
                    results,
                )
                if misses:
                    miss_batches.append(misses)
        finally:
            while pending:
                cls._discard_pending_gpu_chunk(pending.popleft())

        for misses in miss_batches:
            cls._decode_cpu_chunk(slot, misses, nvimgcodec, results)

    @classmethod
    def decode_many(
        cls,
        data: Sequence[bytes],
        *,
        output_modes: Sequence[_OutputMode] | None = None,
        decoders: int = NVIMAGECODEC_DEFAULT_DECODERS,
        batch_size: int = NVIMAGECODEC_DEFAULT_BATCH_SIZE,
        pipeline_depth: int = NVIMAGECODEC_DEFAULT_PIPELINE_DEPTH,
    ) -> list[Image.Image | None]:
        """Decode supported images in native batches with positional fallback."""
        decoders = validate_nvimagecodec_decoders(decoders)
        batch_size = validate_nvimagecodec_batch_size(batch_size)
        pipeline_depth = validate_nvimagecodec_pipeline_depth(pipeline_depth)
        encoded_images = list(data)
        modes = _validate_output_modes(encoded_images, output_modes)
        if not encoded_images:
            return []

        _nvimagecodec_decoder_pool.check_pid()
        nvimgcodec = _load_nvimgcodec()
        inspected = [
            cls._inspect_item(encoded, mode, index, nvimgcodec)
            for index, (encoded, mode) in enumerate(zip(encoded_images, modes))
        ]
        eligible = [item for item in inspected if item is not None]
        results: list[Image.Image | None] = [None] * len(encoded_images)
        if not eligible:
            return results

        try:
            cls._configure_decoder_slots(decoders, batch_size, pipeline_depth)
            cpu_items = [
                item for item in eligible if item.codec_name in _NVIMAGECODEC_CPU_CODECS
            ]
            gpu_items = [
                item for item in eligible if item.codec_name in _NVIMAGECODEC_GPU_CODECS
            ]
            effective_max_pixels = _get_effective_max_pixels()

            if gpu_items:
                from vllm.multimodal.gpu_ipc_memory import get_mm_gpu_ipc_pool

                memory_pool = get_mm_gpu_ipc_pool()
                if memory_pool is None:
                    raise RuntimeError(
                        "The nvImageCodec image backend requires a positive "
                        "--mm-ipc-gpu-memory-gb value."
                    )
                for item in gpu_items:
                    if item.raw_bytes > memory_pool.total_bytes:
                        # Preserve MultiModalGPUMemoryPool's actionable capacity error.
                        try:
                            memory_pool.acquire(item.raw_bytes)
                        except ValueError as error:
                            raise NvImageCodecBatchItemError(
                                item.index, error
                            ) from error
                max_gpu_raw_bytes = min(
                    memory_pool.total_bytes,
                    effective_max_pixels * NVIMAGECODEC_MAX_CHANNELS,
                )
                for mode in ("RGB", "RGBA"):
                    mode_items = [
                        item for item in gpu_items if item.output_mode == mode
                    ]
                    chunks = list(
                        cls._iter_chunks(
                            mode_items,
                            batch_size=batch_size,
                            max_raw_bytes=max_gpu_raw_bytes,
                        )
                    )
                    if pipeline_depth > 1 and len(chunks) > 1:
                        with cls._borrow_decoder_slot() as slot:
                            cls._decode_gpu_chunks_pipelined(
                                slot,
                                chunks,
                                nvimgcodec,
                                memory_pool,
                                results,
                                pipeline_depth=pipeline_depth,
                            )
                    else:
                        for chunk in chunks:
                            with cls._borrow_decoder_slot() as slot:
                                cls._decode_gpu_chunk(
                                    slot,
                                    chunk,
                                    nvimgcodec,
                                    memory_pool,
                                    results,
                                )

            max_cpu_raw_bytes = effective_max_pixels * NVIMAGECODEC_MAX_CHANNELS
            for mode in ("RGB", "RGBA"):
                mode_items = [item for item in cpu_items if item.output_mode == mode]
                for chunk in cls._iter_chunks(
                    mode_items,
                    batch_size=batch_size,
                    max_raw_bytes=max_cpu_raw_bytes,
                ):
                    with cls._borrow_decoder_slot() as slot:
                        cls._decode_cpu_chunk(slot, chunk, nvimgcodec, results)
        except BaseException:
            _close_decoded_results(results)
            raise

        return results

    @classmethod
    def decode(
        cls,
        data: bytes,
        *,
        output_mode: _OutputMode = "RGB",
        decoders: int = NVIMAGECODEC_DEFAULT_DECODERS,
        batch_size: int = NVIMAGECODEC_DEFAULT_BATCH_SIZE,
        pipeline_depth: int = NVIMAGECODEC_DEFAULT_PIPELINE_DEPTH,
    ) -> Image.Image | None:
        """Decode one image through the same native-batch implementation."""
        return cls.decode_many(
            [data],
            output_modes=[output_mode],
            decoders=decoders,
            batch_size=batch_size,
            pipeline_depth=pipeline_depth,
        )[0]


def shutdown_nvimagecodec_decoder_pool() -> None:
    """Release retained native decoders and CUDA streams in this process."""
    _nvimagecodec_decoder_pool.shutdown()


def decode_images_nvimagecodec(
    data: Sequence[bytes],
    *,
    output_modes: Sequence[_OutputMode] | None = None,
    decoders: int = NVIMAGECODEC_DEFAULT_DECODERS,
    batch_size: int = NVIMAGECODEC_DEFAULT_BATCH_SIZE,
    pipeline_depth: int = NVIMAGECODEC_DEFAULT_PIPELINE_DEPTH,
) -> list[Image.Image | None]:
    return NvImageCodecBackend.decode_many(
        data,
        output_modes=output_modes,
        decoders=decoders,
        batch_size=batch_size,
        pipeline_depth=pipeline_depth,
    )


def decode_image_nvimagecodec(
    data: bytes,
    *,
    output_mode: _OutputMode = "RGB",
    decoders: int = NVIMAGECODEC_DEFAULT_DECODERS,
    batch_size: int = NVIMAGECODEC_DEFAULT_BATCH_SIZE,
    pipeline_depth: int = NVIMAGECODEC_DEFAULT_PIPELINE_DEPTH,
) -> Image.Image | None:
    return NvImageCodecBackend.decode(
        data,
        output_mode=output_mode,
        decoders=decoders,
        batch_size=batch_size,
        pipeline_depth=pipeline_depth,
    )
