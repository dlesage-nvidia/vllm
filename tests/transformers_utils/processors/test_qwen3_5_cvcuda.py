# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util

import pytest
import torch
from transformers import Qwen2VLImageProcessor, Qwen3VLVideoProcessor

from vllm.transformers_utils.processors.qwen3_5_cvcuda import (
    Qwen3_5CVCUDAImageProcessor,
    Qwen3_5CVCUDAVideoProcessor,
    cvcuda_qwen3_5_patchify,
)
from vllm.utils.mem_utils import device_memory_info


def _processors(is_video: bool):
    kwargs = dict(patch_size=16, temporal_patch_size=2, merge_size=2)
    if is_video:
        return Qwen3VLVideoProcessor(**kwargs), Qwen3_5CVCUDAVideoProcessor(**kwargs)
    return Qwen2VLImageProcessor(**kwargs), Qwen3_5CVCUDAImageProcessor(**kwargs)


def _inputs(is_video: bool, height: int, width: int, num_frames: int = 3):
    generator = torch.Generator().manual_seed(1234)
    shape = (num_frames, 3, height, width) if is_video else (3, height, width)
    return [torch.randint(0, 256, shape, generator=generator, dtype=torch.uint8)]


def _process(processor, pixels, is_video: bool, **kwargs):
    if is_video:
        kwargs["do_sample_frames"] = False
    return processor(
        pixels,
        do_rescale=False,
        do_normalize=False,
        return_tensors="pt",
        **kwargs,
    )


@pytest.mark.parametrize(
    "source,target",
    [(1, 32), (7, 3), (31, 32), (32, 31), (64, 1), (97, 96), (225, 224)],
)
def test_cvcuda_uint8_coefficients_preserve_torchvision_cpu_rounding(source, target):
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2.functional import resize

    from vllm.transformers_utils.processors.cvcuda_uint8_coefficients import (
        resize_axis_table,
    )

    pixels = _inputs(False, 16, source)[0]
    expected = resize(pixels, [16, target], InterpolationMode.BICUBIC, antialias=True)
    table = resize_axis_table(source, target)
    actual = torch.empty((3, 16, target), dtype=torch.uint8)
    for coordinate in range(target):
        start, count = int(table.starts[coordinate]), int(table.counts[coordinate])
        coefficients = torch.from_numpy(
            table.coefficients[coordinate, :count].copy()
        ).to(dtype=torch.int64)
        values = pixels[:, :, start : start + count].to(dtype=torch.int64)
        accumulator = (values * coefficients).sum(dim=-1)
        accumulator += 1 << (table.precision - 1)
        actual[:, :, coordinate] = (accumulator >> table.precision).clamp(0, 255)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert resize_axis_table(source, target) is table
    assert all(
        not values.flags.writeable
        for values in (
            table.starts,
            table.counts,
            table.coefficients,
            table.error_bounds,
        )
    )


@pytest.mark.parametrize("is_video", [False, True])
@pytest.mark.parametrize("do_resize", [False, True])
def test_raw_pixels_and_grid_match_stock_processor(is_video, do_resize):
    stock, device = _processors(is_video)
    pixels = _inputs(is_video, 160, 224)
    expected = _process(stock, pixels, is_video, do_resize=do_resize)
    actual = _process(device, pixels, is_video, do_resize=do_resize)
    grid_key = "video_grid_thw" if is_video else "image_grid_thw"
    pixel_key = "pixel_values_videos" if is_video else "pixel_values"
    torch.testing.assert_close(actual[grid_key], expected[grid_key], rtol=0, atol=0)
    assert actual[pixel_key].numel() == 0
    raw = actual[f"{pixel_key}_raw"][0]
    expected_raw = (
        pixels[0].permute(0, 2, 3, 1)
        if is_video
        else pixels[0].permute(1, 2, 0).unsqueeze(0)
    )
    torch.testing.assert_close(raw, expected_raw, rtol=0, atol=0)


@pytest.mark.parametrize("numpy_input", [False, True])
@pytest.mark.parametrize("is_video", [False, True])
def test_rgb_source_layout_avoids_full_resolution_cpu_copies(numpy_input, is_video):
    _, processor = _processors(is_video)
    pixels = _inputs(is_video, 160, 224)[0]
    source = (
        pixels.permute(0, 2, 3, 1) if is_video else pixels.permute(1, 2, 0)
    ).contiguous()
    actual = _process(processor, [source.numpy() if numpy_input else source], is_video)
    pixel_key = "pixel_values_videos" if is_video else "pixel_values"
    raw = actual[f"{pixel_key}_raw"][0]
    if not is_video:
        source = source.unsqueeze(0)
    torch.testing.assert_close(raw, source, rtol=0, atol=0)
    assert raw.data_ptr() == source.data_ptr()


@pytest.mark.parametrize("is_video", [False, True])
def test_device_resize_false_is_exactly_the_stock_processor(is_video):
    stock, device = _processors(is_video)
    pixels = _inputs(is_video, 161, 225)
    expected = _process(stock, pixels, is_video)
    actual = _process(device, pixels, is_video, device_resize=False)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("is_video", [False, True])
@pytest.mark.parametrize("invalid", ["float", "channels", "interpolation", "normalize"])
def test_device_resize_rejects_unsupported_pixels(is_video, invalid):
    _, device = _processors(is_video)
    pixels = _inputs(is_video, 161, 225)
    kwargs = {}
    if invalid == "float":
        pixels = [pixels[0].float()]
    elif invalid == "channels":
        pixels = [pixels[0][..., :1, :, :]]
        kwargs["do_convert_rgb"] = False
    elif invalid == "interpolation":
        kwargs["resample"] = 2
    else:
        kwargs["do_normalize"] = True
    if is_video:
        kwargs["do_sample_frames"] = False
    with pytest.raises(ValueError, match="device_resize requires"):
        device(pixels, do_rescale=False, **kwargs)


@pytest.mark.parametrize("is_video", [False, True])
def test_mixed_source_sizes_preserve_item_order_and_target_grids(is_video):
    stock, device = _processors(is_video)
    pixels = _inputs(is_video, 161, 225) + _inputs(is_video, 320, 160)
    expected = _process(stock, pixels, is_video)
    actual = _process(device, pixels, is_video)
    grid_key = "video_grid_thw" if is_video else "image_grid_thw"
    pixel_key = "pixel_values_videos" if is_video else "pixel_values"
    torch.testing.assert_close(actual[grid_key], expected[grid_key], rtol=0, atol=0)
    assert [tuple(raw.shape[1:3]) for raw in actual[f"{pixel_key}_raw"]] == [
        (161, 225),
        (320, 160),
    ]


@pytest.mark.parametrize("cap", [False, True])
def test_video_per_frame_cap_keeps_stock_grids(cap):
    stock, device = _processors(True)
    pixels = _inputs(True, 161, 225, num_frames=5)
    kwargs = dict(
        cap_pixels_per_frame=cap,
        size={"shortest_edge": 4 * 32**2, "longest_edge": 100 * 32**2},
    )
    expected = _process(stock, pixels, True, **kwargs)
    actual = _process(device, pixels, True, **kwargs)
    torch.testing.assert_close(
        actual["video_grid_thw"], expected["video_grid_thw"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "shape,expected",
    [
        ((2, 3, 3, 5, 1), 648),
        ((2, 3, 2, 5, 1), 384),
        ((2, 3, 3, 3, 1), 232),
        ((2, 3, 2, 3, 1), 0),
        ((2, 3, 3, 5, 0), 0),
    ],
)
def test_cpu_uint8_workspace_accounts_for_aligned_coefficients_and_axis_passes(
    shape, expected
):
    from vllm.transformers_utils.processors.cvcuda_utils import (
        cvcuda_cpu_uint8_scratch_bytes,
    )

    assert cvcuda_cpu_uint8_scratch_bytes(*shape) == expected


def test_cpu_uint8_warmup_reserves_the_largest_shape_within_the_memory_cap():
    from vllm.transformers_utils.processors.cvcuda_utils import (
        _CVCUDA_SCRATCH_BYTES,
        _cpu_uint8_reservation_shape,
        cvcuda_cpu_uint8_scratch_bytes,
    )

    rows, reserved = _cpu_uint8_reservation_shape()
    assert reserved == cvcuda_cpu_uint8_scratch_bytes(rows, 16, 16, 4096)
    assert reserved <= _CVCUDA_SCRATCH_BYTES
    assert (
        cvcuda_cpu_uint8_scratch_bytes(rows + 1, 16, 16, 4096) > _CVCUDA_SCRATCH_BYTES
    )


requires_cvcuda = pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("cvcuda") is None,
    reason="Requires CUDA and the optional vllm[cvcuda] dependency",
)


@requires_cvcuda
@pytest.mark.parametrize("is_video", [False, True])
@pytest.mark.parametrize(
    "size", [(160, 224), (161, 225), (321, 97), (33, 1025), (678, 900), (960, 1280)]
)
@pytest.mark.parametrize("max_chunk_bytes", [1, 256 * 1024 * 1024])
def test_device_patch_packing_matches_stock_processor(is_video, size, max_chunk_bytes):
    stock, device = _processors(is_video)
    pixels = _inputs(is_video, *size)
    processor_size = {"shortest_edge": 4096, "longest_edge": 1048576}
    expected = _process(stock, pixels, is_video, size=processor_size)
    raw = _process(device, pixels, is_video, size=processor_size)
    pixel_key = "pixel_values_videos" if is_video else "pixel_values"
    grid_key = "video_grid_thw" if is_video else "image_grid_thw"
    torch.testing.assert_close(raw[grid_key], expected[grid_key], rtol=0, atol=0)
    actual = cvcuda_qwen3_5_patchify(
        raw[f"{pixel_key}_raw"],
        raw[grid_key],
        torch.device("cuda"),
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        is_video=is_video,
        max_chunk_bytes=max_chunk_bytes,
    ).cpu()
    torch.testing.assert_close(actual, expected[pixel_key], rtol=0, atol=0)


@requires_cvcuda
def test_resize_scratch_stays_flat_across_growing_videos():
    from vllm.transformers_utils.processors.cvcuda_utils import cvcuda_bicubic_resize

    cvcuda_bicubic_resize(
        torch.zeros((1, 641, 641, 3), dtype=torch.uint8),
        (512, 512),
        torch.device("cuda"),
    )
    torch.accelerator.synchronize()
    torch.accelerator.empty_cache()
    baseline_free = device_memory_info()[0]
    for size in (686, 960, 1301):
        output = cvcuda_bicubic_resize(
            torch.zeros((16, size, size, 3), dtype=torch.uint8),
            (512, 512),
            torch.device("cuda"),
        )
        del output
        torch.accelerator.synchronize()
        torch.accelerator.empty_cache()
        assert baseline_free - device_memory_info()[0] < 16 * 1024 * 1024


@requires_cvcuda
@pytest.mark.parametrize("host_input", [False, True])
def test_resize_does_not_wait_for_unrelated_stream_work(host_input):
    from vllm.transformers_utils.processors.cvcuda_utils import cvcuda_bicubic_resize

    pixels = _inputs(True, 160, 224)[0].permute(0, 2, 3, 1).contiguous()
    if not host_input:
        pixels = pixels.cuda()
    cvcuda_bicubic_resize(pixels, (128, 192), torch.device("cuda"))
    torch.accelerator.synchronize()
    unrelated = torch.cuda.Stream()
    pending = torch.cuda.Event()
    with torch.cuda.stream(unrelated):
        torch.cuda._sleep(2_000_000_000)
        pending.record()
    output = cvcuda_bicubic_resize(
        pixels, (128, 192), torch.device("cuda"), quantize_intermediate=True
    )
    assert not pending.query()
    assert output.shape == (3, 3, 128, 192)
    pending.synchronize()


@requires_cvcuda
def test_host_upload_is_not_reused_before_current_stream_reads_it():
    from vllm.transformers_utils.processors.cvcuda_utils import cvcuda_bicubic_resize

    pixels = _inputs(True, 160, 224)[0].permute(0, 2, 3, 1).contiguous()
    next_pixels = 255 - pixels
    expected = cvcuda_bicubic_resize(pixels, (128, 192), torch.device("cuda"))
    next_expected = cvcuda_bicubic_resize(next_pixels, (128, 192), torch.device("cuda"))
    torch.accelerator.synchronize()
    torch.cuda._sleep(2_000_000_000)
    pending = torch.cuda.Event()
    pending.record()
    actual = cvcuda_bicubic_resize(pixels, (128, 192), torch.device("cuda"))
    next_actual = cvcuda_bicubic_resize(next_pixels, (128, 192), torch.device("cuda"))
    assert not pending.query()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(next_actual, next_expected, rtol=0, atol=0)


@requires_cvcuda
def test_startup_workspace_profile_covers_upload_and_intermediate_buffers():
    from vllm.transformers_utils.processors.qwen3_5_cvcuda import (
        reserve_qwen3_5_cvcuda_workspace,
    )

    torch.accelerator.reset_peak_memory_stats()
    allocated_before = torch.accelerator.memory_allocated()
    workspace = reserve_qwen3_5_cvcuda_workspace(torch.device("cuda"))
    assert torch.accelerator.memory_allocated() >= allocated_before + 384 * 1024 * 1024
    assert torch.accelerator.max_memory_allocated() >= 384 * 1024 * 1024
    del workspace
