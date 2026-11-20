# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from importlib import metadata

import torch

_CVCUDA_SCRATCH_BYTES = 512 * 1024 * 1024
_cvcuda_scratch_reserved: set[int] = set()
_upload_streams: dict[int, torch.cuda.Stream] = {}
_default_resize_streams: dict[int, torch.cuda.Stream] = {}
_cpu_resize_streams: dict[int, torch.cuda.Stream] = {}


class _CudaArrayOnStream:
    """A non-owning CAI v3 view avoids CV-CUDA's CAI v2 device synchronization."""

    def __init__(self, tensor: torch.Tensor, stream: int) -> None:
        self.__cuda_array_interface__ = {
            **tensor.__cuda_array_interface__,
            "version": 3,
            "stream": stream,
        }


class _UploadSlots:
    """Two consumer-owned staging buffers, recycled by explicit CUDA events."""

    def __init__(self, capacity, device, consumer, upload):
        self.consumer = consumer
        self.upload = upload
        self.slots = [
            torch.empty(capacity, dtype=torch.uint8, device=device) for _ in range(2)
        ]
        self.free = [torch.cuda.Event() for _ in self.slots]
        self.ready = [torch.cuda.Event() for _ in self.slots]
        self.pending = [False, False]
        self.index = 0
        for event in self.free:
            event.record(consumer)

    def copy(self, pixels):
        from vllm.utils.gpu_sync_debug import gpu_sync_allowed

        index = self.index
        self.index = (index + 1) % len(self.slots)
        destination = self.slots[index][: pixels.numel()].view(pixels.shape)
        with torch.cuda.stream(self.upload), gpu_sync_allowed():
            self.upload.wait_event(self.free[index])
            self.pending[index] = True
            destination.copy_(pixels.contiguous(), non_blocking=True)
            self.ready[index].record(self.upload)
        self.consumer.wait_event(self.ready[index])
        return destination, index

    def release(self, index):
        self.free[index].record(self.consumer)

    def close(self):
        if any(self.pending):
            self.consumer.wait_stream(self.upload)
        self.slots.clear()


def _cvcuda_distribution(cuda_version: str | None) -> str:
    if cuda_version is None or cuda_version.split(".")[0] not in ("12", "13"):
        raise RuntimeError("CV-CUDA resize requires a CUDA 12 or 13 Torch runtime")
    return f"cvcuda-cu{cuda_version.split('.')[0]}"


def _validate_cvcuda_release_version(version: str) -> None:
    if version != "0.19.0":
        raise RuntimeError("CPU uint8 resize requires public CV-CUDA 0.19.0")


def require_cvcuda_release() -> None:
    """Require the public release matching Torch's CUDA major, without CUDA calls."""
    distribution = _cvcuda_distribution(torch.version.cuda)
    try:
        version = metadata.version(distribution)
    except metadata.PackageNotFoundError as error:
        raise RuntimeError(
            f"CPU uint8 resize requires {distribution}==0.19.0"
        ) from error
    _validate_cvcuda_release_version(version)


def _cvcuda_workspace_bytes(
    source_size: tuple[int, int], size: tuple[int, int], num_frames: int
) -> int:
    """Mirror public v0.19 antialiased cubic TvResize workspace requirements."""
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
    """Choose a maximal bounded startup shape, including coefficient storage."""
    upper = min(max_scratch_bytes // (12 * 4096), 2**31 - 1)
    if upper >= 113:
        return upper
    for rows in range(upper, 0, -1):
        if _cvcuda_workspace_bytes((rows, 16), (16, 4096), 1) <= max_scratch_bytes:
            return rows
    raise ValueError("CV-CUDA resize warmup exceeds the scratch budget")


def cvcuda_bicubic_resize(
    frames: torch.Tensor,
    size: tuple[int, int],
    device: torch.device,
    max_temporary_bytes: int = 256 * 1024 * 1024,
    *,
    preferred_frame_step: int | None = None,
) -> torch.Tensor:
    """Upload NHWC RGB and resize with public TV FP32 plus sparse exact correction.

    CV-CUDA performs each one-axis floating resize. Only ambiguous half-integer
    outputs use integer MACs; all other pixels consume the CV result. Horizontal
    quantization/clipping precedes the vertical stage. Both use the independently
    derived CPU uint8 coefficient table and conservative released-CV enclosure.
    CPU inputs use an independent resize-consumer stream; bounded upload slots
    never wait for unrelated model work. CUDA input producers must be ordered
    before the caller's stream. Full output storage is additional to the bounded
    temporaries and the separately reserved native workspace.
    """
    if (
        not isinstance(frames, torch.Tensor)
        or frames.ndim != 4
        or frames.shape[-1] != 3
        or frames.dtype != torch.uint8
    ):
        raise ValueError("CV-CUDA resize requires NHWC uint8 RGB")
    if min(frames.shape[:3]) < 1 or len(size) != 2 or max_temporary_bytes < 1:
        raise ValueError("CV-CUDA resize requires nonempty frames and positive budgets")
    source_size = tuple(frames.shape[1:3])
    size = tuple(size)
    _cvcuda_workspace_bytes(source_size, size, 1)
    if device.type != "cuda":
        raise ValueError("CV-CUDA resize requires a CUDA device")
    require_cvcuda_release()
    import cvcuda

    from vllm.transformers_utils.processors.cvcuda_resize_plan import resize_plan
    from vllm.transformers_utils.processors.cvcuda_uint8_coefficients import (
        resize_axis_table,
    )
    from vllm.transformers_utils.processors.cvcuda_uint8_correction import (
        correct_axis_uint8,
        validate_layout,
    )
    from vllm.utils.gpu_sync_debug import gpu_sync_allowed

    if not callable(getattr(cvcuda, "tvresize_into", None)):
        raise RuntimeError("Public CV-CUDA tvresize_into is unavailable")
    torch_stream = torch.cuda.current_stream(device)
    device_index = torch_stream.device_index
    device = torch.device("cuda", device_index)
    if frames.device.type not in ("cpu", "cuda") or (
        frames.is_cuda and frames.device != device
    ):
        raise ValueError("Source must be CPU or the target CUDA device")
    validate_layout(tuple(frames.shape), tuple(frames.stride()))
    if frames.device.type == "cpu":
        if device_index not in _cpu_resize_streams:
            _cpu_resize_streams[device_index] = torch.cuda.Stream(device_index)
        resize_stream = _cpu_resize_streams[device_index]
        if torch_stream != resize_stream:
            with torch.cuda.stream(resize_stream):
                output = cvcuda_bicubic_resize(
                    frames,
                    size,
                    device,
                    max_temporary_bytes,
                    preferred_frame_step=preferred_frame_step,
                )
            torch_stream.wait_stream(resize_stream)
            output.record_stream(torch_stream)
            return output
    if torch_stream.cuda_stream == 0:
        if device_index not in _default_resize_streams:
            _default_resize_streams[device_index] = torch.cuda.Stream(device_index)
        resize_stream = _default_resize_streams[device_index]
        resize_stream.wait_stream(torch_stream)
        if frames.is_cuda:
            frames.record_stream(resize_stream)
        with torch.cuda.stream(resize_stream):
            output = cvcuda_bicubic_resize(
                frames,
                size,
                device,
                max_temporary_bytes,
                preferred_frame_step=preferred_frame_step,
            )
        torch_stream.wait_stream(resize_stream)
        output.record_stream(torch_stream)
        return output
    stream_id = torch_stream.cuda_stream
    stream = cvcuda.as_stream(stream_id)

    def tvresize(destination: torch.Tensor, source: torch.Tensor) -> None:
        source.record_stream(torch_stream)
        destination.record_stream(torch_stream)
        cvcuda.tvresize_into(
            cvcuda.as_tensor(_CudaArrayOnStream(destination, stream_id), "NHWC"),
            cvcuda.as_tensor(_CudaArrayOnStream(source, stream_id), "NHWC"),
            interp=cvcuda.Interp.CUBIC,
            antialias=True,
            stream=stream,
        )

    height, width = source_size
    tables = {}
    for axis, source_extent, target_extent in (
        (2, width, size[1]),
        (1, height, size[0]),
    ):
        if source_extent != target_extent:
            tables[axis] = resize_axis_table(source_extent, target_extent)
    table_bytes = sum(
        table.starts.nbytes + table.coefficients.nbytes + table.error_bounds.nbytes
        for table in tables.values()
    )
    plan = resize_plan(
        source_size,
        size,
        len(frames),
        max_temporary_bytes,
        table_bytes,
        cpu_source=frames.device.type == "cpu",
        preferred_frame_step=preferred_frame_step,
    )
    if device_index not in _cvcuda_scratch_reserved:
        rows = _cvcuda_warmup_rows(_CVCUDA_SCRATCH_BYTES)
        tvresize(
            torch.empty((1, 16, 4096, 3), dtype=torch.float32, device=device),
            torch.zeros((1, rows, 16, 3), dtype=torch.float32, device=device),
        )
        _cvcuda_scratch_reserved.add(device_index)
    if device_index not in _upload_streams:
        _upload_streams[device_index] = torch.cuda.Stream(device_index)
    upload = _upload_streams[device_index]
    with torch.cuda.stream(upload), gpu_sync_allowed():
        device_tables = {
            axis: (
                torch.tensor(table.starts, dtype=torch.int64, device=device),
                torch.tensor(table.coefficients, dtype=torch.int16, device=device),
                torch.tensor(table.error_bounds, dtype=torch.float32, device=device),
                table.precision,
            )
            for axis, table in tables.items()
        }
    torch_stream.wait_stream(upload)
    for starts, coefficients, bounds, _ in device_tables.values():
        for tensor in (starts, coefficients, bounds):
            tensor.record_stream(torch_stream)
    output = torch.empty((len(frames), 3, *size), dtype=torch.uint8, device=device)
    cpu_source = frames.device.type == "cpu"

    def axis_resize(source, axis):
        if axis not in device_tables:
            return source
        starts, coefficients, bounds, precision = device_tables[axis]
        axis_shape = list(source.shape)
        axis_shape[axis] = size[axis - 1]
        approximate = torch.empty(axis_shape, dtype=torch.float32, device=device)
        tvresize(approximate, source.to(dtype=torch.float32))
        result, count = correct_axis_uint8(
            source, approximate, bounds, starts, coefficients, precision, axis
        )
        del approximate, count
        return result

    def reformat(destination, source):
        destination.record_stream(torch_stream)
        source.record_stream(torch_stream)
        cvcuda.reformat_into(
            cvcuda.as_tensor(_CudaArrayOnStream(destination, stream_id), "NCHW"),
            cvcuda.as_tensor(_CudaArrayOnStream(source, stream_id), "NHWC"),
            stream=stream,
        )

    def slots():
        return _UploadSlots(plan.upload_capacity, device, torch_stream, upload)

    def load(pixels, staging):
        if staging is not None:
            return staging.copy(pixels)
        source = pixels.contiguous()
        source.record_stream(torch_stream)
        return source, None

    if plan.mode == "frames":
        staging = slots() if cpu_source else None
        try:
            for start in range(0, len(frames), plan.frame_step):
                source, index = load(frames[start : start + plan.frame_step], staging)
                result = axis_resize(axis_resize(source, 2), 1)
                reformat(output[start : start + plan.frame_step], result)
                if staging is not None:
                    staging.release(index)
                del source, result
        finally:
            if staging is not None:
                staging.close()
        return output

    def horizontal(pixels, destination, first_column=0, to_output=False):
        staging = slots() if cpu_source else None
        try:
            for first_row in range(0, height, plan.row_step):
                last_row = min(first_row + plan.row_step, height)
                source, index = load(pixels[:, first_row:last_row], staging)
                result = axis_resize(source, 2)
                if to_output:
                    reformat(destination[:, :, first_row:last_row], result)
                else:
                    destination[:, first_row:last_row].copy_(
                        result[:, :, first_column : first_column + destination.shape[2]]
                    )
                if staging is not None:
                    staging.release(index)
                del source, result
        finally:
            if staging is not None:
                staging.close()

    for frame in range(len(frames)):
        pixels = frames[frame : frame + 1]
        destination = output[frame : frame + 1]
        if plan.mode in ("rows", "identity"):
            horizontal(pixels, destination, to_output=True)
        elif plan.mode == "columns":
            staging = slots() if cpu_source else None
            try:
                for first in range(0, size[1], plan.column_step):
                    last = min(first + plan.column_step, size[1])
                    source, index = load(pixels[:, :, first:last], staging)
                    result = axis_resize(source, 1)
                    reformat(destination[:, :, :, first:last], result)
                    if staging is not None:
                        staging.release(index)
                    del source, result
            finally:
                if staging is not None:
                    staging.close()
        elif plan.mode == "middle":
            middle = torch.empty(
                (1, height, size[1], 3), dtype=torch.uint8, device=device
            )
            horizontal(pixels, middle)
            for first in range(0, size[1], plan.column_step):
                last = min(first + plan.column_step, size[1])
                result = axis_resize(middle[:, :, first:last], 1)
                reformat(destination[:, :, :, first:last], result)
                del result
            del middle
        else:
            for first in range(0, size[1], plan.column_step):
                last = min(first + plan.column_step, size[1])
                middle = torch.empty(
                    (1, height, last - first, 3), dtype=torch.uint8, device=device
                )
                horizontal(pixels, middle, first)
                result = axis_resize(middle, 1)
                reformat(destination[:, :, :, first:last], result)
                del middle, result
    return output


def reserve_cvcuda_workspace(device: torch.device) -> torch.Tensor:
    """Charge bounded tensor temporaries and persistent CV-CUDA scratch at startup."""
    require_cvcuda_release()
    workspace = torch.empty(384 * 1024 * 1024, dtype=torch.uint8, device=device)
    cvcuda_bicubic_resize(
        torch.zeros((1, 17, 33, 3), dtype=torch.uint8, device="cpu"),
        (16, 32),
        device,
    )
    return workspace
