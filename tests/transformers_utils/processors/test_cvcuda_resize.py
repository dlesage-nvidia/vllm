# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as functional

from vllm.transformers_utils.processors.cvcuda_utils import cvcuda_bicubic_resize


@pytest.mark.parametrize("cpu_source", [False, True])
@pytest.mark.parametrize("default_stream", [False, True])
@pytest.mark.parametrize("chunk_frames", [1, 32])
def test_direct_resize_batch32_pixel_stream_and_chunk_parity(
    cpu_source, default_stream, chunk_frames
):
    pixels = torch.randint(
        0,
        256,
        (32, 67, 97, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(1234),
    )
    target = (48, 64)
    expected = (
        functional.interpolate(
            pixels.permute(0, 3, 1, 2).float(),
            size=target,
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    device = torch.device("cuda", 0)
    reserve = cvcuda_bicubic_resize(pixels[:1], target, device)
    torch.accelerator.synchronize(device)
    del reserve
    stream = (
        torch.cuda.default_stream(device)
        if default_stream
        else torch.cuda.Stream(device)
    )
    frame_bytes = pixels[0].numel()
    budget = chunk_frames * frame_bytes
    with torch.cuda.stream(stream):
        source = pixels if cpu_source else pixels.to(device)
        baseline = torch.accelerator.memory_allocated(device)
        torch.accelerator.reset_peak_memory_stats(device)
        outputs = [
            cvcuda_bicubic_resize(source, target, device, budget) for _ in range(2)
        ]
        retained = outputs.pop().clone()
        torch.empty_like(retained).fill_(99)
        complete = torch.cuda.Event()
        complete.record(stream)
    complete.synchronize()
    actual = retained.cpu()
    assert actual.shape == (32, 3, *target)
    assert actual.dtype == torch.uint8
    difference = (actual.to(torch.int16) - expected.to(torch.int16)).abs()
    assert difference.max().item() <= 1
    assert (difference != 0).float().mean().item() < 1e-3
    torch.testing.assert_close(outputs[0].cpu(), actual, rtol=0, atol=0)
    reference = cvcuda_bicubic_resize(pixels, target, device, frame_bytes).cpu()
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    output_bytes = expected.numel()
    temporary_bytes = budget + chunk_frames * 3 * target[0] * target[1]
    assert (
        torch.accelerator.max_memory_allocated(device)
        <= baseline + 4 * output_bytes + 2 * temporary_bytes
    )
