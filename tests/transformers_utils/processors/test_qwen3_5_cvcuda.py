# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.util
import sys
from enum import Enum

import pytest
import torch
from transformers import Qwen2VLImageProcessor, Qwen3VLVideoProcessor

from vllm.transformers_utils.processors.qwen3_5_cvcuda import (
    Qwen3_5CVCUDAImageProcessor,
    Qwen3_5CVCUDAVideoProcessor,
    _image_resize_groups,
    _validate_qwen3_5_cvcuda_api,
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


def _assert_fp32_resize_pixels(actual: torch.Tensor, expected: torch.Tensor) -> None:
    assert actual.dtype == expected.dtype == torch.uint8
    assert actual.shape == expected.shape
    diff = (actual.int() - expected.int()).abs()
    assert diff.max() <= 1
    assert (diff > 0).float().mean() < 1e-3


def _fp32_resize_and_pack(pixels, grid_thw, is_video):
    expected = []
    for frames, grid in zip(pixels, grid_thw.tolist(), strict=True):
        grid_t, grid_h, grid_w = grid
        target = (grid_h * 16, grid_w * 16)
        resized = (
            torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2).float(),
                size=target,
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        if not is_video:
            resized = resized.expand(2, -1, -1, -1)
        elif len(resized) % 2:
            resized = torch.cat((resized, resized[-1:]))
        patches = resized.unfold(2, 16, 16).unfold(3, 16, 16)
        patches = patches.reshape(grid_t, 2, 3, grid_h // 2, 2, grid_w // 2, 2, 16, 16)
        expected.append(
            patches.permute(0, 3, 5, 4, 6, 2, 1, 7, 8).reshape(
                grid_t * grid_h * grid_w, 3 * 2 * 16 * 16
            )
        )
    return torch.cat(expected)


@pytest.mark.parametrize(
    "changed,num_pixels,level,accepted",
    [
        (0, 1000, 0, True),
        (1, 1001, 1, True),
        (1, 1000, 1, False),
        (2, 1000, 1, False),
        (1, 2000, 2, False),
    ],
)
def test_fp32_resize_contract_rejects_excess_error_or_fraction(
    changed, num_pixels, level, accepted
):
    expected = torch.zeros(num_pixels, dtype=torch.uint8)
    actual = expected.clone()
    actual[:changed] = level
    if accepted:
        _assert_fp32_resize_pixels(actual, expected)
    else:
        with pytest.raises(AssertionError):
            _assert_fp32_resize_pixels(actual, expected)


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
def test_qwen_direct_resize_public_versions(version):
    exports: dict[str, object] = {
        name: len
        for name in ("tvresize_into", "reformat_into", "as_tensor", "as_stream")
    }
    exports.update(__version__=version, Interp=Enum("Interpolation", {"CUBIC": 3}))
    _validate_qwen3_5_cvcuda_api(version, exports)


@pytest.mark.parametrize(
    "version",
    [
        "0.18.0",
        "0.19.0b0",
        "0.19.0+custom",
        "0.20.0rc1",
        "0.20.0.dev1",
        "0.20.0+custom",
        "0.20.1",
        "0.21.0",
    ],
)
def test_qwen_direct_resize_rejects_unreviewed_versions(version):
    with pytest.raises(RuntimeError, match="requires CV-CUDA"):
        _validate_qwen3_5_cvcuda_api(version, {})


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
@pytest.mark.parametrize("noncallable", [False, True])
@pytest.mark.parametrize(
    "missing",
    [
        "__version__",
        "tvresize_into",
        "reformat_into",
        "as_tensor",
        "as_stream",
        "Interp",
    ],
)
def test_qwen_direct_resize_rejects_missing_public_api(version, missing, noncallable):
    exports: dict[str, object] = {
        name: len
        for name in ("tvresize_into", "reformat_into", "as_tensor", "as_stream")
    }
    exports.update(__version__=version, Interp=Enum("Interpolation", {"CUBIC": 3}))
    if noncallable:
        exports[missing] = None
    else:
        exports.pop(missing)
    with pytest.raises(RuntimeError):
        _validate_qwen3_5_cvcuda_api(version, exports)


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


@pytest.mark.parametrize(
    "max_chunk_bytes,grid,step",
    [(1, (3, 7), 1), (2016, (3, 7), 2), (65536, (3, 7), 32), (462, (1, 1), 2)],
)
def test_image_resize_groups_bound_source_and_target_buffers(
    max_chunk_bytes, grid, step
):
    pixels = [
        torch.full((1, 7, 11, 3), index, dtype=torch.uint8) for index in range(32)
    ]
    grids = [[1, *grid] for _ in pixels]
    groups = list(
        _image_resize_groups(
            pixels, grids, patch_size=4, max_chunk_bytes=max_chunk_bytes
        )
    )
    assert groups == [(start, min(start + step, 32)) for start in range(0, 32, step)]
    for start, end in groups:
        batch = torch.cat(pixels[start:end])
        assert torch.equal(
            batch[:, 0, 0, 0], torch.arange(start, end, dtype=torch.uint8)
        )
        if end - start > 1:
            assert batch.numel() <= max_chunk_bytes
            assert (end - start) * grid[0] * grid[1] * 16 * 3 <= max_chunk_bytes


def test_image_resize_groups_preserve_contiguous_shape_grid_device_boundaries():
    pixels = [torch.empty((1, 7, 11, 3), dtype=torch.uint8) for _ in range(5)]
    pixels += [torch.empty((1, 11, 7, 3), dtype=torch.uint8)]
    pixels += [torch.empty((1, 7, 11, 3), dtype=torch.uint8, device="meta")]
    pixels += [torch.empty((1, 7, 11, 3), dtype=torch.float32)]
    grids = [[1, 3, 7], [1, 3, 7], [1, 7, 3], [1, 3, 7], [1, 3, 7]]
    grids += [[1, 3, 7] for _ in range(3)]
    groups = list(
        _image_resize_groups(pixels, grids, patch_size=4, max_chunk_bytes=65536)
    )
    assert groups == [(0, 2), (2, 3), (3, 5), (5, 6), (6, 7), (7, 8)]


@pytest.mark.parametrize("mixed_shapes", [False, True])
def test_32_image_processor_raw_inputs_form_compatible_resize_groups(mixed_shapes):
    stock, processor = _processors(False)
    generator = torch.Generator().manual_seed(1234)
    sizes = [(161, 225)] * 32
    if mixed_shapes:
        sizes[16:24] = [(321, 97)] * 8
    pixels = [
        torch.randint(0, 256, (3, *size), generator=generator, dtype=torch.uint8)
        for size in sizes
    ]
    kwargs = {"size": {"shortest_edge": 4096, "longest_edge": 1048576}}
    expected = _process(stock, pixels, False, **kwargs)
    raw = _process(processor, pixels, False, **kwargs)
    assert torch.equal(raw["image_grid_thw"], expected["image_grid_thw"])
    groups = list(
        _image_resize_groups(
            raw["pixel_values_raw"],
            raw["image_grid_thw"].tolist(),
            patch_size=16,
            max_chunk_bytes=256 * 1024 * 1024,
        )
    )
    assert groups == ([(0, 16), (16, 24), (24, 32)] if mixed_shapes else [(0, 32)])
    for start, end in groups:
        batch = torch.cat(raw["pixel_values_raw"][start:end])
        assert torch.equal(batch, torch.stack(pixels[start:end]).permute(0, 2, 3, 1))


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
def test_device_patch_packing_matches_fp32_resize_oracle(
    is_video, size, max_chunk_bytes
):
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
    reference = _fp32_resize_and_pack(raw[f"{pixel_key}_raw"], raw[grid_key], is_video)
    _assert_fp32_resize_pixels(actual, reference)


@requires_cvcuda
@pytest.mark.parametrize("is_video", [False, True])
def test_device_patch_packing_without_resize_is_bit_exact(is_video):
    stock, device = _processors(is_video)
    pixels = _inputs(is_video, 160, 224, num_frames=32)
    expected = _process(stock, pixels, is_video, do_resize=False)
    raw = _process(device, pixels, is_video, do_resize=False)
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
    ).cpu()
    torch.testing.assert_close(actual, expected[pixel_key], rtol=0, atol=0)


@requires_cvcuda
@pytest.mark.parametrize("mixed_shapes", [False, True])
@pytest.mark.parametrize("host_input", [False, True])
def test_genuine_32_image_processor_batch_preserves_grids_and_packing(
    mixed_shapes, host_input
):
    from vllm.transformers_utils.processors.cvcuda_utils import cvcuda_bicubic_resize

    stock, device = _processors(False)
    generator = torch.Generator().manual_seed(1234)
    sizes = [(161, 225)] * 32
    if mixed_shapes:
        sizes[16:24] = [(321, 97)] * 8
    pixels = [
        torch.randint(
            0,
            256,
            (3, *sizes[index]),
            generator=generator,
            dtype=torch.uint8,
        )
        for index in range(32)
    ]
    kwargs = {"size": {"shortest_edge": 4096, "longest_edge": 1048576}}
    expected = _process(stock, pixels, False, **kwargs)
    raw = _process(device, pixels, False, **kwargs)
    assert len(raw["pixel_values_raw"]) == len(raw["image_grid_thw"]) == 32
    torch.testing.assert_close(
        raw["image_grid_thw"], expected["image_grid_thw"], rtol=0, atol=0
    )
    for source, retained in zip(pixels, raw["pixel_values_raw"], strict=True):
        torch.testing.assert_close(
            retained, source.permute(1, 2, 0).unsqueeze(0), rtol=0, atol=0
        )
    frames = raw["pixel_values_raw"]
    if not host_input:
        frames = [source.cuda() for source in frames]
    observed_batches = []

    def observe_resize(frame, event, arg):
        if event == "call" and frame.f_code is cvcuda_bicubic_resize.__code__:
            observed_batches.append(frame.f_locals["frames"].shape[0])

    previous_profile = sys.getprofile()
    try:
        sys.setprofile(observe_resize)
        actual = cvcuda_qwen3_5_patchify(
            frames,
            raw["image_grid_thw"],
            torch.device("cuda"),
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            is_video=False,
        )
    finally:
        sys.setprofile(previous_profile)
    assert set(observed_batches) == ({16, 8} if mixed_shapes else {32})
    chunked = cvcuda_qwen3_5_patchify(
        frames,
        raw["image_grid_thw"],
        torch.device("cuda"),
        patch_size=16,
        temporal_patch_size=2,
        merge_size=2,
        is_video=False,
        max_chunk_bytes=1,
    )
    torch.testing.assert_close(actual, chunked, rtol=0, atol=0)
    _assert_fp32_resize_pixels(
        actual.cpu(),
        _fp32_resize_and_pack(raw["pixel_values_raw"], raw["image_grid_thw"], False),
    )


@requires_cvcuda
@pytest.mark.parametrize("source_size", [(360, 640), (224, 384), (73, 129)])
@pytest.mark.parametrize("host_input", [False, True])
def test_public_resize_genuine_batch32_matches_fp32_oracle(source_size, host_input):
    from vllm.transformers_utils.processors.cvcuda_utils import cvcuda_bicubic_resize

    source_height, source_width = source_size
    pixels = (
        _inputs(True, source_height, source_width, num_frames=32)[0]
        .permute(0, 2, 3, 1)
        .contiguous()
    )
    expected = (
        torch.nn.functional.interpolate(
            pixels.permute(0, 3, 1, 2).float(),
            size=(224, 384),
            mode="bicubic",
            align_corners=False,
            antialias=True,
        )
        .round()
        .clamp(0, 255)
        .to(torch.uint8)
    )
    if not host_input:
        pixels = pixels.cuda()
    actual = cvcuda_bicubic_resize(pixels, (224, 384), torch.device("cuda"))
    chunked = cvcuda_bicubic_resize(
        pixels, (224, 384), torch.device("cuda"), max_chunk_bytes=1
    )
    assert actual.shape == (32, 3, 224, 384)
    torch.testing.assert_close(actual, chunked, rtol=0, atol=0)
    _assert_fp32_resize_pixels(actual.cpu(), expected)


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
    output = cvcuda_bicubic_resize(pixels, (128, 192), torch.device("cuda"))
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
