# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Host-only tensor/workspace accounting for orthogonal-axis resize tiles."""

from dataclasses import dataclass

NATIVE_SCRATCH_BYTES = 512 * 1024 * 1024
ALLOCATION_OVERHEAD_BYTES = 8192


@dataclass(frozen=True)
class ResizePlan:
    mode: str
    frame_step: int
    row_step: int
    column_step: int
    upload_capacity: int
    temporary_bytes: int
    output_bytes: int
    native_bytes: int


def resize_plan(
    source_size: tuple[int, int],
    size: tuple[int, int],
    num_frames: int,
    max_temporary_bytes: int,
    table_bytes: int = 0,
    cpu_source: bool = True,
    preferred_frame_step: int | None = None,
) -> ResizePlan:
    """Keep filtered axes whole, even when an entire U8 middle cannot fit.

    The returned output is separately charged for all frames. Two CPU upload
    slots, FP32 input/approximation, U8 correction and middle storage are charged
    as temporaries. Native operator storage has its own fixed ceiling.
    """
    if (
        len(source_size) != 2
        or len(size) != 2
        or any(
            type(extent) is not int or not 1 <= extent <= 2**31 - 1
            for extent in (*source_size, *size)
        )
    ):
        raise ValueError("CV-CUDA spatial extents must fit positive signed int32")
    if type(num_frames) is not int or num_frames < 1:
        raise ValueError("Frame count must be a positive integer")
    if type(max_temporary_bytes) is not int or max_temporary_bytes < 1:
        raise ValueError("Temporary budget must be a positive integer")
    if type(table_bytes) is not int or table_bytes < 0:
        raise ValueError("Table byte count must be a nonnegative integer")
    if preferred_frame_step is not None and (
        type(preferred_frame_step) is not int or preferred_frame_step < 1
    ):
        raise ValueError("Preferred frame step must be a positive integer")
    height, width = source_size
    target_height, target_width = size
    source = height * width
    middle = height * target_width
    target = target_height * target_width
    output_bytes = 3 * num_frames * target
    if output_bytes > 2**63 - 1:
        raise ValueError("Output size must fit signed int64")
    fixed = table_bytes + ALLOCATION_OVERHEAD_BYTES
    available = max_temporary_bytes - fixed
    upload_factor = 6 if cpu_source else 3
    axis_factor = 18 if cpu_source else 15
    width_changed = width != target_width
    height_changed = height != target_height
    if not width_changed and not height_changed:
        frame_bytes = upload_factor * source
        frame_native = 0
    elif not width_changed or not height_changed:
        frame_bytes = axis_factor * source + 15 * target
        frame_native = 12 * height * target_width
    else:
        frame_bytes = max(
            axis_factor * source + 15 * middle,
            upload_factor * source + 15 * middle + 15 * target,
        )
        frame_native = 12 * height * target_width
    step = min(num_frames, 65535, max(0, available // frame_bytes))
    if preferred_frame_step is not None:
        step = min(step, preferred_frame_step)
    if frame_native:
        step = min(step, NATIVE_SCRATCH_BYTES // frame_native)
    if step:
        return ResizePlan(
            "frames",
            step,
            height,
            target_width,
            3 * step * source if cpu_source else 0,
            fixed + step * frame_bytes,
            output_bytes,
            step * frame_native,
        )

    def finish(mode, rows, columns, temporary, native, upload):
        if rows < 1 or columns < 1 or temporary > available:
            raise ValueError(
                "A full filtered-axis operand exceeds the temporary budget"
            )
        if native > NATIVE_SCRATCH_BYTES:
            raise ValueError("A full filtered-axis operand exceeds the scratch budget")
        return ResizePlan(
            mode,
            1,
            rows,
            columns,
            upload if cpu_source else 0,
            fixed + temporary,
            output_bytes,
            native,
        )

    if not width_changed and not height_changed:
        rows = min(height, max(0, available // (upload_factor * width)))
        return finish(
            "identity",
            rows,
            target_width,
            upload_factor * rows * width,
            0,
            3 * rows * width,
        )
    row_cost = axis_factor * width + 15 * target_width
    column_cost = axis_factor * height + 15 * target_height
    row_native = 12 * target_width
    column_native = 12 * height
    if not height_changed:
        rows = min(
            height, max(0, available // row_cost), NATIVE_SCRATCH_BYTES // row_native
        )
        return finish(
            "rows",
            rows,
            target_width,
            rows * row_cost,
            rows * row_native,
            3 * rows * width,
        )
    if not width_changed:
        columns = min(
            target_width,
            max(0, available // column_cost),
            NATIVE_SCRATCH_BYTES // column_native,
        )
        return finish(
            "columns",
            height,
            columns,
            columns * column_cost,
            columns * column_native,
            3 * height * columns,
        )
    middle_bytes = 3 * middle
    remaining = available - middle_bytes
    rows = min(
        height, max(0, remaining // row_cost), NATIVE_SCRATCH_BYTES // row_native
    )
    vertical_cost = 15 * (height + target_height)
    columns = min(
        target_width,
        max(0, remaining // vertical_cost),
        NATIVE_SCRATCH_BYTES // column_native,
    )
    if rows and columns:
        return finish(
            "middle",
            rows,
            columns,
            middle_bytes + max(rows * row_cost, columns * vertical_cost),
            max(rows * row_native, columns * column_native),
            3 * rows * width,
        )
    columns = min(
        target_width,
        max(0, available // vertical_cost),
        max(0, (available - row_cost) // (3 * height)),
        NATIVE_SCRATCH_BYTES // column_native,
    )
    rows = min(
        height,
        max(0, (available - 3 * height * columns) // row_cost),
        NATIVE_SCRATCH_BYTES // row_native,
    )
    return finish(
        "strips",
        rows,
        columns,
        max(3 * height * columns + rows * row_cost, columns * vertical_cost),
        max(rows * row_native, columns * column_native),
        3 * rows * width,
    )
