# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata

import torch

NATIVE_SCRATCH_BYTES = 512 * 1024**2
ALLOCATION_OVERHEAD_BYTES = 8192
DEFAULT_TEMPORARY_BYTES = 384 * 1024**2
_scratch_reserved: set[int] = set()
_upload_streams: dict[int, torch.cuda.Stream] = {}
_resize_streams: dict[int, torch.cuda.Stream] = {}


def _cvcuda_distribution(cuda_version: str | None) -> str:
    if cuda_version is None or cuda_version.split(".")[0] not in ("12", "13"):
        raise RuntimeError("CV-CUDA resize requires a CUDA 12 or 13 Torch runtime")
    return f"cvcuda-cu{cuda_version.split('.')[0]}"


def _validate_cvcuda_release_version(version: str) -> None:
    if version not in ("0.19.0", "0.20.0"):
        raise RuntimeError("Cosmos resize requires CV-CUDA 0.19.0 or 0.20.0")


def _validate_cvcuda_api(version: str, exports: Mapping[str, object]) -> None:
    _validate_cvcuda_release_version(version)
    if exports.get("__version__") != version:
        raise RuntimeError("CV-CUDA module and distribution versions must match")
    for name in ("tvresize_into", "reformat_into", "as_tensor", "as_stream"):
        if not callable(exports.get(name)):
            raise RuntimeError(f"Required public CV-CUDA API is unavailable: {name}")
    if getattr(exports.get("Interp"), "CUBIC", None) is None:
        raise RuntimeError("Public CV-CUDA cubic interpolation is unavailable")


def require_cvcuda_release() -> str:
    distribution = _cvcuda_distribution(torch.version.cuda)
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(f"Cosmos resize requires {distribution}") from error
    _validate_cvcuda_release_version(version)
    return version


def _cvcuda_workspace_bytes(
    source_size: tuple[int, int], size: tuple[int, int], num_frames: int
) -> int:
    if any(
        type(extent) is not int or not 1 <= extent <= 2**31 - 1
        for extent in (*source_size, *size)
    ):
        raise ValueError("CV-CUDA spatial extents must fit positive signed int32")
    if type(num_frames) is not int or not 1 <= num_frames <= 65535:
        raise ValueError("CV-CUDA batch size must be in [1, 65535]")
    height, width = source_size
    if source_size == size:
        return 0
    required = 12 * num_frames * height * size[1]
    if (
        height >= 8
        and width >= 8
        and min(size) >= 16
        and max(size) <= 4096
        and height != size[0]
        and width != size[1]
        and width <= 7 * size[1]
        and height <= 7 * size[0]
    ):
        required += 136 * sum(size)
    if required > 2**63 - 1:
        raise ValueError("CV-CUDA workspace must fit signed int64")
    return required


def _cvcuda_warmup_rows(max_scratch_bytes: int) -> int:
    upper = min(max_scratch_bytes // (12 * 4096), 2**31 - 1)
    if upper >= 113:
        return upper
    for rows in range(upper, 0, -1):
        if _cvcuda_workspace_bytes((rows, 16), (16, 4096), 1) <= max_scratch_bytes:
            return rows
    raise ValueError("CV-CUDA resize warmup exceeds the scratch budget")


@dataclass(frozen=True)
class ResizePlan:
    frame_step: int
    upload_capacity: int
    temporary_bytes: int
    output_bytes: int
    native_bytes: int


def resize_plan(
    source_size: tuple[int, int],
    size: tuple[int, int],
    num_frames: int,
    max_temporary_bytes: int = DEFAULT_TEMPORARY_BYTES,
    *,
    cpu_source: bool = True,
    preferred_frame_step: int | None = None,
) -> ResizePlan:
    _cvcuda_workspace_bytes(source_size, size, 1)
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError("Frame count must be a positive integer")
    if type(max_temporary_bytes) is not int or max_temporary_bytes < 1:
        raise ValueError("Temporary budget must be a positive integer")
    if preferred_frame_step is not None and (
        type(preferred_frame_step) is not int or preferred_frame_step < 1
    ):
        raise ValueError("Preferred frame step must be a positive integer")
    source_pixels = source_size[0] * source_size[1]
    target_pixels = size[0] * size[1]
    output_bytes = 3 * num_frames * target_pixels
    if output_bytes > 2**63 - 1:
        raise ValueError("Output size must fit signed int64")
    frame_bytes = (6 if cpu_source else 3) * source_pixels
    if source_size != size:
        frame_bytes += 3 * target_pixels
    step = min(
        num_frames,
        65535,
        max(0, (max_temporary_bytes - ALLOCATION_OVERHEAD_BYTES) // frame_bytes),
    )
    if preferred_frame_step is not None:
        step = min(step, preferred_frame_step)
    if source_size != size:
        native_per_frame = 12 * source_size[0] * size[1]
        filter_bytes = _cvcuda_workspace_bytes(source_size, size, 1) - native_per_frame
        step = min(step, (NATIVE_SCRATCH_BYTES - filter_bytes) // native_per_frame)
    if step < 1:
        raise ValueError("A full frame exceeds the temporary or native scratch budget")
    return ResizePlan(
        step,
        3 * step * source_pixels if cpu_source else 0,
        ALLOCATION_OVERHEAD_BYTES + step * frame_bytes,
        output_bytes,
        _cvcuda_workspace_bytes(source_size, size, step),
    )


class _CudaArrayOnStream:
    def __init__(self, tensor: torch.Tensor, stream: int) -> None:
        self.__cuda_array_interface__ = {
            **tensor.__cuda_array_interface__,
            "version": 3,
            "stream": stream,
        }


def _frame_parts(
    inputs: Sequence[torch.Tensor], start: int, count: int
) -> list[torch.Tensor]:
    parts = []
    for frames in inputs:
        if start >= len(frames):
            start -= len(frames)
            continue
        part = frames[start : start + count]
        parts.append(part)
        count -= len(part)
        start = 0
        if count == 0:
            break
    return parts


def _resize_inputs(
    inputs: Sequence[torch.Tensor],
    size: tuple[int, int],
    device: torch.device,
    max_temporary_bytes: int,
    preferred_frame_step: int | None,
) -> torch.Tensor:
    import cvcuda

    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    _validate_cvcuda_api(require_cvcuda_release(), vars(cvcuda))
    consumer = torch.cuda.current_stream(device)
    device_index = consumer.device_index
    device = torch.device("cuda", device_index)
    cpu_source = inputs[0].device.type == "cpu"
    if not cpu_source and inputs[0].device != device:
        raise ValueError("Source must be CPU or the target CUDA device")
    if cpu_source or consumer.cuda_stream == 0:
        if device_index not in _resize_streams:
            _resize_streams[device_index] = torch.cuda.Stream(device_index)
        independent = _resize_streams[device_index]
        if independent != consumer:
            if not cpu_source:
                independent.wait_stream(consumer)
                for frames in inputs:
                    frames.record_stream(independent)
            with torch.cuda.stream(independent):
                output = _resize_inputs(
                    inputs, size, device, max_temporary_bytes, preferred_frame_step
                )
            consumer.wait_stream(independent)
            output.record_stream(consumer)
            return output
    stream_id = consumer.cuda_stream or 1
    stream = cvcuda.as_stream(stream_id)

    def tvresize(destination: torch.Tensor, source: torch.Tensor) -> None:
        source.record_stream(consumer)
        destination.record_stream(consumer)
        cvcuda.tvresize_into(
            cvcuda.as_tensor(_CudaArrayOnStream(destination, stream_id), "NHWC"),
            cvcuda.as_tensor(_CudaArrayOnStream(source, stream_id), "NHWC"),
            interp=cvcuda.Interp.CUBIC,
            antialias=True,
            stream=stream,
        )

    source_size = (inputs[0].shape[1], inputs[0].shape[2])
    frame_count = sum(len(frames) for frames in inputs)
    plan = resize_plan(
        source_size,
        size,
        frame_count,
        max_temporary_bytes,
        cpu_source=cpu_source,
        preferred_frame_step=preferred_frame_step,
    )
    if device_index not in _scratch_reserved:
        rows = _cvcuda_warmup_rows(NATIVE_SCRATCH_BYTES)
        tvresize(
            torch.empty((1, 16, 4096, 3), dtype=torch.uint8, device=device),
            torch.zeros((1, rows, 16, 3), dtype=torch.uint8, device=device),
        )
        _scratch_reserved.add(device_index)
    output = torch.empty((frame_count, 3, *size), dtype=torch.uint8, device=device)
    slots = (
        [
            torch.empty(plan.upload_capacity, dtype=torch.uint8, device=device)
            for _ in range(2)
        ]
        if cpu_source
        else []
    )
    free = [torch.cuda.Event() for _ in slots]
    ready = [torch.cuda.Event() for _ in slots]
    for event in free:
        event.record(consumer)
    if device_index not in _upload_streams:
        _upload_streams[device_index] = torch.cuda.Stream(device_index)
    upload = _upload_streams[device_index]
    for chunk_index, start in enumerate(range(0, frame_count, plan.frame_step)):
        count = min(plan.frame_step, frame_count - start)
        parts = _frame_parts(inputs, start, count)
        slot_index = chunk_index % 2
        if cpu_source:
            source = slots[slot_index][: count * 3 * source_size[0] * source_size[1]]
            source = source.view(count, *source_size, 3)
            with torch.cuda.stream(upload), gpu_sync_allowed():
                upload.wait_event(free[slot_index])
                offset = 0
                for part in parts:
                    source[offset : offset + len(part)].copy_(
                        part.contiguous(), non_blocking=True
                    )
                    offset += len(part)
                ready[slot_index].record(upload)
            consumer.wait_event(ready[slot_index])
        else:
            source = parts[0].contiguous() if len(parts) == 1 else torch.cat(parts)
        if source_size != size:
            resized = torch.empty((count, *size, 3), dtype=torch.uint8, device=device)
            tvresize(resized, source)
        else:
            resized = source
        destination = output[start : start + count]
        resized.record_stream(consumer)
        cvcuda.reformat_into(
            cvcuda.as_tensor(_CudaArrayOnStream(destination, stream_id), "NCHW"),
            cvcuda.as_tensor(_CudaArrayOnStream(resized, stream_id), "NHWC"),
            stream=stream,
        )
        if cpu_source:
            free[slot_index].record(consumer)
        del source, resized, destination, parts
    return output


def cvcuda_bicubic_resize_many(
    pixels: Sequence[torch.Tensor],
    sizes: Sequence[tuple[int, int]],
    device: torch.device,
    max_temporary_bytes: int = DEFAULT_TEMPORARY_BYTES,
) -> list[torch.Tensor]:
    if len(pixels) != len(sizes) or not pixels:
        raise ValueError("Each raw pixel tensor requires a target size")
    if device.type != "cuda":
        raise ValueError("Cosmos resize requires a CUDA device")
    groups: dict[tuple[tuple[int, ...], torch.device, tuple[int, int]], list[int]] = {}
    for index, (frames, size) in enumerate(zip(pixels, sizes)):
        if frames.ndim != 4 or frames.dtype != torch.uint8 or frames.shape[-1] != 3:
            raise ValueError("Raw pixels must be uint8 NHWC RGB tensors")
        if len(frames) < 1 or frames.device.type not in ("cpu", "cuda"):
            raise ValueError("Raw pixels must be nonempty CPU or CUDA tensors")
        resize_plan(
            (frames.shape[1], frames.shape[2]),
            size,
            len(frames),
            max_temporary_bytes,
            cpu_source=frames.device.type == "cpu",
        )
        groups.setdefault((tuple(frames.shape[1:]), frames.device, size), []).append(
            index
        )
    outputs: dict[int, torch.Tensor] = {}
    for indices in groups.values():
        sources = [pixels[index] for index in indices]
        resized = _resize_inputs(
            sources, sizes[indices[0]], device, max_temporary_bytes, None
        )
        offset = 0
        for index, source in zip(indices, sources):
            outputs[index] = resized[offset : offset + len(source)]
            offset += len(source)
    return [outputs[index] for index in range(len(pixels))]


def cvcuda_bicubic_resize(
    frames: torch.Tensor,
    size: tuple[int, int],
    device: torch.device,
    max_temporary_bytes: int = DEFAULT_TEMPORARY_BYTES,
    preferred_frame_step: int | None = None,
) -> torch.Tensor:
    if frames.ndim != 4 or frames.dtype != torch.uint8 or frames.shape[-1] != 3:
        raise ValueError("Raw pixels must be uint8 NHWC RGB tensors")
    if len(frames) < 1 or frames.device.type not in ("cpu", "cuda"):
        raise ValueError("Raw pixels must be nonempty CPU or CUDA tensors")
    resize_plan(
        (frames.shape[1], frames.shape[2]),
        size,
        len(frames),
        max_temporary_bytes,
        cpu_source=frames.device.type == "cpu",
        preferred_frame_step=preferred_frame_step,
    )
    if device.type != "cuda":
        raise ValueError("Cosmos resize requires a CUDA device")
    return _resize_inputs(
        [frames], size, device, max_temporary_bytes, preferred_frame_step
    )


def reserve_cvcuda_workspace(device: torch.device) -> torch.Tensor:
    require_cvcuda_release()
    workspace = torch.empty(DEFAULT_TEMPORARY_BYTES, dtype=torch.uint8, device=device)
    cvcuda_bicubic_resize(
        torch.zeros((1, 17, 33, 3), dtype=torch.uint8), (16, 32), device
    )
    return workspace
