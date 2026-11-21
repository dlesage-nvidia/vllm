# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses

import pytest
import torch

from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
    ALLOCATION_OVERHEAD_BYTES,
    NATIVE_SCRATCH_BYTES,
    cvcuda_bicubic_resize,
    cvcuda_bicubic_resize_many,
    reserve_cvcuda_workspace,
    resize_plan,
)


@pytest.mark.parametrize("cpu_source", [False, True])
@pytest.mark.parametrize("target", [(48, 64), (67, 64), (48, 97), (67, 97)])
def test_plan_bounds_whole_frames_without_per_axis_correction(target, cpu_source):
    budget = 512 * 1024
    plan = resize_plan((67, 97), target, 3, budget, cpu_source=cpu_source)
    assert plan.temporary_bytes <= budget
    assert plan.native_bytes <= NATIVE_SCRATCH_BYTES
    assert plan.output_bytes == 3 * 3 * target[0] * target[1]
    assert 1 <= plan.frame_step <= 3
    if cpu_source:
        assert (
            2 * plan.upload_capacity + ALLOCATION_OVERHEAD_BYTES <= plan.temporary_bytes
        )
    else:
        assert plan.upload_capacity == 0
    with pytest.raises(dataclasses.FrozenInstanceError):
        plan.__setattr__("frame_step", 7)


@pytest.mark.parametrize("cpu_source", [False, True])
def test_long_video_charges_full_output_separately_without_growing_temporary(
    cpu_source,
):
    short = resize_plan(
        (1080, 1920), (672, 896), 128, 32 * 1024**2, cpu_source=cpu_source
    )
    long = resize_plan(
        (1080, 1920), (672, 896), 1024, 32 * 1024**2, cpu_source=cpu_source
    )
    assert long.temporary_bytes == short.temporary_bytes
    assert long.output_bytes == 8 * short.output_bytes
    assert long.output_bytes > 32 * 1024**2


def test_full_frame_cannot_be_silently_split_or_exceed_native_scratch():
    with pytest.raises(ValueError, match="full frame"):
        resize_plan((67, 97), (48, 64), 1, ALLOCATION_OVERHEAD_BYTES + 1024)
    with pytest.raises(ValueError, match="full frame"):
        resize_plan((32768, 16), (16, 4096), 1)
    with pytest.raises(ValueError, match="signed int64"):
        resize_plan((67, 97), (48, 64), 2**63, 64 * 1024)


def test_preferred_single_frame_preserves_real_temporary_budget():
    plan = resize_plan((67, 97), (48, 64), 8, preferred_frame_step=1)
    assert plan.frame_step == 1
    assert plan.temporary_bytes > 1


def test_native_batch32_fits_both_real_budgets():
    plan = resize_plan((117, 173), (128, 160), 32)
    assert plan.frame_step == 32
    assert plan.upload_capacity == 32 * 117 * 173 * 3
    assert plan.native_bytes <= NATIVE_SCRATCH_BYTES


@pytest.mark.parametrize("cpu_source", [False, True])
@pytest.mark.parametrize("default_stream", [False, True])
@pytest.mark.parametrize("chunk_frames", [1, 2])
def test_gpu_bounded_chunks_preserve_fp32_oracle_and_lifetime(
    cpu_source, default_stream, chunk_frames
):
    pixels = torch.randint(
        0,
        256,
        (3, 67, 97, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(1234),
    )
    target = (48, 64)
    expected = (
        torch.nn.functional.interpolate(
            pixels.permute(0, 3, 1, 2).float(),
            target,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    budget = resize_plan(
        (67, 97), target, chunk_frames, cpu_source=cpu_source
    ).temporary_bytes
    assert (
        resize_plan((67, 97), target, 3, budget, cpu_source=cpu_source).frame_step
        == chunk_frames
    )
    device = torch.device("cuda", 0)
    reserve = reserve_cvcuda_workspace(device)
    torch.accelerator.synchronize(device)
    del reserve
    stream = (
        torch.cuda.default_stream(device)
        if default_stream
        else torch.cuda.Stream(device)
    )
    with torch.cuda.stream(stream):
        source = pixels if cpu_source else pixels.to(device)
        baseline = torch.accelerator.memory_allocated(device)
        torch.accelerator.reset_peak_memory_stats(device)
        outputs = [
            cvcuda_bicubic_resize(source, target, device, max_temporary_bytes=budget)
            for _ in range(2)
        ]
        retained = outputs[-1].clone()
        del outputs[-1]
        torch.empty_like(retained).fill_(99)
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    for actual in (*outputs, retained):
        difference = (actual.cpu().int() - expected.int()).abs()
        assert difference.max() <= 1
        assert (difference > 0).float().mean() < 1e-3
    torch.testing.assert_close(outputs[0], retained, rtol=0, atol=0)
    assert (
        torch.accelerator.max_memory_allocated(device)
        <= baseline + 3 * expected.numel() + budget
    )


@pytest.mark.parametrize("cpu_source", [False, True])
@pytest.mark.parametrize("default_stream", [False, True])
def test_gpu_native_batch32_equals_chunked_and_grouped_public_resize(
    cpu_source, default_stream
):
    pixels = torch.randint(
        0,
        256,
        (32, 117, 173, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(42),
    )
    target = (128, 160)
    assert resize_plan((117, 173), target, 32, cpu_source=cpu_source).frame_step == 32
    expected = (
        torch.nn.functional.interpolate(
            pixels.permute(0, 3, 1, 2).float(),
            target,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    device = torch.device("cuda", 0)
    stream = (
        torch.cuda.default_stream(device)
        if default_stream
        else torch.cuda.Stream(device)
    )
    with torch.cuda.stream(stream):
        source = pixels if cpu_source else pixels.to(device)
        batched = cvcuda_bicubic_resize(source, target, device)
        chunked = cvcuda_bicubic_resize(source, target, device, preferred_frame_step=1)
        grouped = cvcuda_bicubic_resize_many(
            list(source.split(1)), [target] * 32, device
        )
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    assert batched.shape == (32, 3, *target)
    difference = (batched.cpu().int() - expected.int()).abs()
    assert difference.max() <= 1
    assert (difference > 0).float().mean() < 1e-3
    torch.testing.assert_close(batched, chunked, rtol=0, atol=0)
    torch.testing.assert_close(batched, torch.cat(grouped), rtol=0, atol=0)
    assert len({item.untyped_storage().data_ptr() for item in grouped}) == 1
