# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Sparse exact correction of one released CV-CUDA floating resize axis.

This module does not generate coefficients, prove bounds, import CV-CUDA, or
initialize a framework at import time. GPU dependencies load only on invocation.
"""

from functools import lru_cache
from math import prod


def validate_layout(shape, strides):
    """Validate a positive, non-overlapping strided layout without reading data."""
    if not shape or len(shape) != len(strides):
        raise ValueError("Shape/stride ranks must agree and be nonzero")
    if any(type(extent) is not int or extent <= 0 for extent in shape):
        raise ValueError("All extents must be positive integers")
    if any(type(stride) is not int or stride <= 0 for stride in strides):
        raise ValueError("All strides must be positive integers")
    span = 1
    for stride, extent in sorted(zip(strides, shape)):
        if extent == 1:
            continue
        if stride < span:
            raise ValueError("Overlapping tensor layouts are not supported")
        span += (extent - 1) * stride
    offset = sum((extent - 1) * stride for extent, stride in zip(shape, strides))
    if offset > 2**63 - 1 or prod(shape) > 2**63 - 1:
        raise ValueError("Tensor indexing must fit signed int64")


def validate_geometry(source_shape, approx_shape, table_shape, precision, axis):
    """Check one-axis geometry and a conservative signed-int64 MAC bound."""
    if type(axis) is not int or axis not in (1, 2):
        raise ValueError("axis must be NHWC height=1 or width=2")
    if len(source_shape) != 4 or len(approx_shape) != 4:
        raise ValueError("Source and approximation must both be NHWC")
    if any(
        type(extent) is not int or not 1 <= extent <= 2**31 - 1
        for extent in (*source_shape, *approx_shape)
    ):
        raise ValueError("NHWC extents must fit positive signed int32")
    for dimension in range(4):
        if dimension != axis and source_shape[dimension] != approx_shape[dimension]:
            raise ValueError("Only the selected axis may change extent")
    if len(table_shape) != 2 or table_shape[0] != approx_shape[axis]:
        raise ValueError("Coefficient rows must match the target axis")
    if type(precision) is not int or not 1 <= precision <= 22:
        raise ValueError("Torch uint8 reference precision must be in [1, 22]")
    if not 1 <= table_shape[1] <= (2**63 - 1 - (1 << (precision - 1))) // (255 * 32768):
        raise ValueError("Conservative integer MAC bound exceeds signed int64")


@lru_cache(maxsize=1)
def _kernels():
    import triton
    import triton.language as tl

    @triton.jit(
        do_not_specialize=("TARGET", "SOURCE", "TAPS", "BS", "SS", "CS0", "CS1"),
        do_not_specialize_on_alignment=(
            "TARGET",
            "SOURCE",
            "TAPS",
            "BS",
            "SS",
            "CS0",
            "CS1",
        ),
    )
    def validate_tables(
        Bounds,
        Starts,
        Coeffs,
        Invalid,
        TARGET,
        SOURCE,
        TAPS,
        BS,
        SS,
        CS0,
        CS1,
        BLOCK: tl.constexpr,
    ):
        coordinate = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = coordinate < TARGET
        bound = tl.load(Bounds + coordinate * BS, valid, other=0)
        start = tl.load(Starts + coordinate * SS, valid, other=0)
        bad = valid & ((bound != bound) | (bound < 0) | (start < 0) | (start > SOURCE))
        safe_start = tl.where((start >= 0) & (start <= SOURCE), start, 0)
        for tap in range(TAPS):
            coefficient = tl.load(Coeffs + coordinate * CS0 + tap * CS1, valid, other=0)
            bad |= valid & (coefficient != 0) & (safe_start + tap >= SOURCE)
        if tl.sum(bad.to(tl.int32), 0) != 0:
            tl.atomic_or(Invalid, 1)

    @triton.jit(
        do_not_specialize=(
            "HEIGHT",
            "WIDTH",
            "ELEMENTS",
            "SOURCE_AXIS",
            "TAPS",
            "PRECISION",
            "SRC0",
            "SRC1",
            "SRC2",
            "SRC3",
            "AP0",
            "AP1",
            "AP2",
            "AP3",
            "BS",
            "SS",
            "CS0",
            "CS1",
        ),
        do_not_specialize_on_alignment=(
            "HEIGHT",
            "WIDTH",
            "ELEMENTS",
            "SOURCE_AXIS",
            "TAPS",
            "PRECISION",
            "SRC0",
            "SRC1",
            "SRC2",
            "SRC3",
            "AP0",
            "AP1",
            "AP2",
            "AP3",
            "BS",
            "SS",
            "CS0",
            "CS1",
        ),
    )
    def correct(
        Source,
        Approx,
        Bounds,
        Starts,
        Coeffs,
        Output,
        Invalid,
        AmbiguousCount,
        HEIGHT,
        WIDTH,
        CHANNELS: tl.constexpr,
        ELEMENTS,
        SOURCE_AXIS,
        AXIS: tl.constexpr,
        TAPS,
        PRECISION,
        SRC0,
        SRC1,
        SRC2,
        SRC3,
        AP0,
        AP1,
        AP2,
        AP3,
        BS,
        SS,
        CS0,
        CS1,
        BLOCK: tl.constexpr,
    ):
        offset = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        valid = offset < ELEMENTS
        height = HEIGHT.to(tl.int64)
        width = WIDTH.to(tl.int64)
        channel = offset % CHANNELS
        column = offset // CHANNELS % width
        row = offset // (CHANNELS * width) % height
        batch = offset // (CHANNELS * width * height)
        coordinate = row if AXIS == 1 else column
        value = tl.load(
            Approx + batch * AP0 + row * AP1 + column * AP2 + channel * AP3,
            valid,
            other=0,
        )
        bound = tl.load(Bounds + coordinate * BS, valid, other=0)
        finite = (value == value) & (tl.abs(value) != float("inf"))
        if tl.sum((valid & ~finite).to(tl.int32), 0) != 0:
            tl.atomic_or(Invalid, 2)
        value = tl.where(finite, value, 0)
        floored = tl.floor(value)
        fractional = value - floored
        allowance = (tl.abs(value) + 1.0) * (2.0**-20)
        ambiguous = valid & (tl.abs(fractional - 0.5) <= bound + allowance)
        rounded = floored + (fractional >= 0.5).to(tl.float32)
        result = tl.minimum(tl.maximum(rounded, 0), 255).to(tl.int64)
        count = tl.sum(ambiguous.to(tl.int32), 0)
        tl.atomic_add(AmbiguousCount, count.to(tl.int64))
        if count != 0:
            start = tl.load(Starts + coordinate * SS, ambiguous, other=0)
            start = tl.where((start >= 0) & (start <= SOURCE_AXIS), start, 0)
            accumulator = tl.full((BLOCK,), 1 << (PRECISION - 1), tl.int64)
            for tap in range(TAPS):
                coefficient = tl.load(
                    Coeffs + coordinate * CS0 + tap * CS1,
                    ambiguous,
                    other=0,
                ).to(tl.int64)
                source_coordinate = start + tap
                load_mask = ambiguous & (coefficient != 0)
                load_mask &= source_coordinate < SOURCE_AXIS
                source_coordinate = tl.where(load_mask, source_coordinate, 0)
                source_row = source_coordinate if AXIS == 1 else row
                source_column = source_coordinate if AXIS == 2 else column
                source_offset = (
                    batch * SRC0
                    + source_row * SRC1
                    + source_column * SRC2
                    + channel * SRC3
                )
                pixel = tl.load(Source + source_offset, load_mask, other=0)
                accumulator += pixel.to(tl.int64) * coefficient
            exact = tl.minimum(tl.maximum(accumulator >> PRECISION, 0), 255)
            result = tl.where(ambiguous, exact, result)
        tl.store(Output + offset, result.to(tl.uint8), valid)

    return triton, validate_tables, correct


def correct_axis_uint8(source, approx, error_bounds, starts, coeffs, precision, axis):
    """Quantize a CV one-axis result, correcting only ambiguous scalar outputs.

    Args:
        source: Positive non-overlapping CUDA NHWC uint8 tensor.
        approx: CUDA NHWC float32 CV result for exactly one axis. Its producer
            must already be ordered before this call on the current Torch stream.
        error_bounds: CUDA float32 vector, one upward-rounded absolute bound per
            target coordinate, relative to the unrounded fixed-point dot product.
            Positive infinity requests exact evaluation everywhere. NaN/negative
            bounds are asynchronously rejected; this helper does not prove bounds.
        starts: CUDA int64 vector of nonnegative source-axis tap starts.
        coeffs: CUDA int16 `[target, max_taps]` table, zero padded at truncated
            boundaries. All nonzero taps must reference valid source coordinates.
        precision: Reference shift in `[1, 22]`. No coefficient renormalization.
        axis: Height=1 or width=2. The other NHWC extents must agree.

    Returns:
        Contiguous NHWC uint8 output and a CUDA int64 scalar ambiguous-output
        count. No tensor-dependent host read or synchronization is requested by
        this helper. Invalid values cause a stream-ordered asynchronous assertion.
        Such an assert can poison the CUDA context: never use this helper with
        untrusted tables or values. Apply width first, then height using the uint8
        intermediate for the original two-axis contract.

    """
    import torch

    tensors = (source, approx, error_bounds, starts, coeffs)
    expected = (torch.uint8, torch.float32, torch.float32, torch.int64, torch.int16)
    for tensor, dtype in zip(tensors, expected):
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype:
            raise TypeError("Expected uint8/float32/float32/int64/int16 tensors")
        if (
            tensor.layout != torch.strided
            or not tensor.is_cuda
            or tensor.is_conj()
            or tensor.is_neg()
        ):
            raise ValueError("Only CUDA strided tensors are supported")
        if tensor.device != source.device:
            raise ValueError("All tensors must be on the same CUDA device")
        if tensor.requires_grad:
            raise ValueError("Resize quantization is inference-only")
        validate_layout(tuple(tensor.shape), tuple(tensor.stride()))
    validate_geometry(source.shape, approx.shape, coeffs.shape, precision, axis)
    if (
        error_bounds.shape != (approx.shape[axis],)
        or starts.shape != error_bounds.shape
    ):
        raise ValueError("Bounds/starts must be vectors matching the target axis")
    if not callable(getattr(torch, "_assert_async", None)):
        raise RuntimeError("Stream-ordered input validation requires _assert_async")
    triton, validate_tables, correct = _kernels()
    with torch.accelerator.device_index(source.device.index):
        stream = torch.cuda.current_stream(source.device)
        for tensor in tensors:
            tensor.record_stream(stream)
        output = torch.empty(
            tuple(approx.shape), device=source.device, dtype=torch.uint8
        )
        invalid = torch.zeros((), device=source.device, dtype=torch.int32)
        count = torch.zeros((), device=source.device, dtype=torch.int64)
        validate_tables[(triton.cdiv(approx.shape[axis], 128),)](
            error_bounds,
            starts,
            coeffs,
            invalid,
            approx.shape[axis],
            source.shape[axis],
            coeffs.shape[1],
            error_bounds.stride(0),
            starts.stride(0),
            *coeffs.stride(),
            128,
        )
        correct[(triton.cdiv(output.numel(), 128),)](
            source,
            approx,
            error_bounds,
            starts,
            coeffs,
            output,
            invalid,
            count,
            approx.shape[1],
            approx.shape[2],
            approx.shape[3],
            output.numel(),
            source.shape[axis],
            axis,
            coeffs.shape[1],
            precision,
            *source.stride(),
            *approx.stride(),
            error_bounds.stride(0),
            starts.stride(0),
            *coeffs.stride(),
            128,
        )
        torch._assert_async(
            invalid == 0, "Invalid sparse-axis input or coefficient table"
        )
    return output, count
