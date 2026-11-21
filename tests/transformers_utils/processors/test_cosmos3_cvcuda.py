# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.video_utils import VideoMetadata

from vllm.config import ModelConfig
from vllm.model_executor.models.cosmos3 import (
    Cosmos3ProcessingInfo,
    Cosmos3RawMultiModalProcessor,
)
from vllm.model_executor.models.qwen3_vl import Qwen3VLDummyInputsBuilder
from vllm.multimodal.processing.context import InputProcessingContext, TimingContext
from vllm.multimodal.processing.inputs import ProcessorInputs
from vllm.transformers_utils.processors.cosmos3_cvcuda import (
    Cosmos3CVCUDAProcessor,
    cvcuda_cosmos3_patchify,
    pack_cosmos3_frames,
)

MODELS = {
    "Nano": "45666784803b3a7ad577e0def405f9afcc3bddb2",
    "Nano-FP8": "9d45b0910219989a76c56aae82a08007bc1b597e",
    "Super": "73d5746645961a0ec9607123a003bd9c1f8a0282",
    "Super-FP8": "3f62010edec964e581cd5752c011d4e3b5d2d059",
}


@pytest.fixture(scope="module", params=MODELS)
def processors(request):
    name = request.param
    stock = AutoProcessor.from_pretrained(
        f"nvidia/Cosmos3-{name}", revision=MODELS[name]
    )
    adapter = Cosmos3CVCUDAProcessor(
        image_processor=stock.image_processor,
        video_processor=stock.video_processor,
        tokenizer=stock.tokenizer,
        chat_template=stock.chat_template,
    )
    return stock, adapter


@pytest.mark.parametrize("shape", [(64, 96), (193, 319)])
@pytest.mark.parametrize("is_video", [False, True], ids=["image", "video"])
@pytest.mark.parametrize("batch_size", [1, 32])
def test_gpu_patchify_matches_fp32_round_once(processors, shape, is_video, batch_size):
    """Keep stock geometry; compare pixels to the merged Nemotron FP32 oracle."""
    stock, adapter = processors
    processor = stock.video_processor if is_video else stock.image_processor
    pixel_key = "pixel_values_videos" if is_video else "pixel_values"
    grid_key = "video_grid_thw" if is_video else "image_grid_thw"
    pixels = np.random.default_rng(71).integers(
        0, 256, (3 if is_video else 1, *shape, 3), dtype=np.uint8
    )
    inputs = dict(
        text=[
            "<|vision_start|>"
            + ("<|video_pad|>" if is_video else "<|image_pad|>")
            + "<|vision_end|>"
        ]
        * batch_size,
        return_tensors="pt",
        do_rescale=False,
        do_normalize=False,
    )
    if is_video:
        inputs.update(
            videos=[pixels] * batch_size,
            do_sample_frames=False,
            video_metadata=[
                VideoMetadata(total_num_frames=3, fps=7, frames_indices=[0, 1, 2])
            ]
            * batch_size,
        )
    else:
        inputs["images"] = [pixels[0]] * batch_size
    expected = stock(**copy.deepcopy(inputs))
    raw = adapter(**copy.deepcopy(inputs))
    torch.testing.assert_close(raw["input_ids"], expected["input_ids"], rtol=0, atol=0)
    torch.testing.assert_close(raw[grid_key], expected[grid_key], rtol=0, atol=0)
    assert expected[pixel_key].dtype == torch.uint8
    device = torch.device("cuda", 0)
    actual = cvcuda_cosmos3_patchify(
        raw[f"{pixel_key}_raw"],
        raw[grid_key],
        device,
        image_processor=processor,
        is_video=is_video,
    )
    assert actual.device == device
    assert actual.dtype == torch.uint8
    assert actual.shape == expected[pixel_key].shape
    reference = []
    for frames, grid in zip(raw[f"{pixel_key}_raw"], raw[grid_key].tolist()):
        resized = (
            torch.nn.functional.interpolate(
                frames.permute(0, 3, 1, 2).float(),
                size=(grid[1] * processor.patch_size, grid[2] * processor.patch_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
            .round()
            .clamp(0, 255)
            .to(torch.uint8)
        )
        reference.append(
            pack_cosmos3_frames(
                resized,
                grid,
                patch_size=processor.patch_size,
                temporal_patch_size=processor.temporal_patch_size,
                merge_size=processor.merge_size,
                is_video=is_video,
            )
        )
    difference = (actual.cpu().int() - torch.cat(reference).int()).abs()
    assert difference.max() <= 1
    assert (difference > 0).float().mean() < 1e-3


@pytest.mark.parametrize("shape", [(31, 65), (67, 97), (320, 160)])
def test_images_transport_original_rgb_and_keep_stock_tokens(processors, shape):
    stock, adapter = processors
    height, width = shape
    pixels = np.random.default_rng(11).integers(
        0, 256, (height, width, 3), dtype=np.uint8
    )
    images = [Image.fromarray(pixels), Image.fromarray(pixels.transpose(1, 0, 2))]
    kwargs = dict(
        text=["<|vision_start|><|image_pad|><|vision_end|>" * 2],
        images=images,
        return_tensors="pt",
    )
    expected = stock(**copy.deepcopy(kwargs))
    actual = adapter(**copy.deepcopy(kwargs))
    torch.testing.assert_close(
        actual["input_ids"], expected["input_ids"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["image_grid_thw"], expected["image_grid_thw"], rtol=0, atol=0
    )
    assert actual["pixel_values"].numel() == 0
    for raw, image in zip(actual["pixel_values_raw"], images):
        assert raw.dtype == torch.uint8 and raw.device.type == "cpu"
        np.testing.assert_array_equal(raw[0].numpy(), np.array(image))


@pytest.mark.parametrize("num_frames", [1, 3, 8])
def test_video_sampling_timestamps_and_temporal_padding_match_stock(
    processors, num_frames
):
    stock, adapter = processors
    video = np.random.default_rng(17).integers(
        0, 256, (num_frames, 65, 97, 3), dtype=np.uint8
    )
    metadata = VideoMetadata(
        total_num_frames=num_frames,
        fps=7,
        frames_indices=list(range(num_frames)),
    )
    kwargs = dict(
        text=["<|vision_start|><|video_pad|><|vision_end|>"],
        videos=[video],
        video_metadata=[metadata],
        do_sample_frames=False,
        return_tensors="pt",
    )
    if num_frames < stock.video_processor.temporal_patch_size:
        with pytest.raises(ValueError, match="must be larger than temporal_factor"):
            stock(**copy.deepcopy(kwargs))
        with pytest.raises(ValueError, match="must be larger than temporal_factor"):
            adapter(**copy.deepcopy(kwargs))
        return
    expected = stock(**copy.deepcopy(kwargs))
    actual = adapter(**copy.deepcopy(kwargs))
    torch.testing.assert_close(
        actual["input_ids"], expected["input_ids"], rtol=0, atol=0
    )
    torch.testing.assert_close(
        actual["video_grid_thw"], expected["video_grid_thw"], rtol=0, atol=0
    )
    assert actual["pixel_values_videos"].numel() == 0
    np.testing.assert_array_equal(actual["pixel_values_videos_raw"][0].numpy(), video)


@pytest.mark.parametrize("is_video", [False, True])
def test_patch_packing_matches_the_actual_stock_processor(processors, is_video):
    stock, _ = processors
    processor = stock.video_processor if is_video else stock.image_processor
    frames = torch.arange(3 * 64 * 96, dtype=torch.int64).remainder(256).to(torch.uint8)
    frames = frames.reshape(1, 3, 64, 96)
    if is_video:
        frames = torch.cat([frames, frames.roll(1, dims=-1), frames.roll(2, dims=-2)])
        result = processor(
            videos=[frames],
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            do_sample_frames=False,
            return_tensors="pt",
        )
        pixel_key, grid_key = "pixel_values_videos", "video_grid_thw"
    else:
        result = processor(
            images=[frames[0]],
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            return_tensors="pt",
        )
        pixel_key, grid_key = "pixel_values", "image_grid_thw"
    actual = pack_cosmos3_frames(
        frames,
        result[grid_key][0].tolist(),
        patch_size=processor.patch_size,
        temporal_patch_size=processor.temporal_patch_size,
        merge_size=processor.merge_size,
        is_video=is_video,
    )
    torch.testing.assert_close(actual, result[pixel_key], rtol=0, atol=0)


@pytest.mark.parametrize("disable_grouping", [False, True])
@pytest.mark.parametrize("is_video", [False, True])
def test_explicit_opt_out_keeps_stock_pixels_and_normalization(
    processors, disable_grouping, is_video
):
    stock, adapter = processors
    pixels = np.random.default_rng(23).integers(0, 256, (3, 67, 99, 3), dtype=np.uint8)
    kwargs = dict(
        text=[
            "<|vision_start|>"
            + ("<|video_pad|>" if is_video else "<|image_pad|>")
            + "<|vision_end|>"
        ],
        return_tensors="pt",
        disable_grouping=disable_grouping,
    )
    if is_video:
        kwargs.update(
            videos=[pixels],
            do_sample_frames=False,
            video_metadata=[
                VideoMetadata(total_num_frames=3, fps=7, frames_indices=[0, 1, 2])
            ],
        )
    else:
        kwargs["images"] = [Image.fromarray(pixels[0])]
    expected = stock(**copy.deepcopy(kwargs))
    actual = adapter(**copy.deepcopy(kwargs), device_resize=False)
    assert actual.keys() == expected.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_rgb_conversion_retains_stock_source_resolution_bytes(processors, mode):
    stock, adapter = processors
    image = Image.new(mode, (67, 99), 37)
    expected = stock.image_processor.process_image(image, do_convert_rgb=True)
    actual = adapter.image_processor(images=[image], return_tensors="pt")
    torch.testing.assert_close(
        actual["pixel_values_raw"][0][0].permute(2, 0, 1), expected, rtol=0, atol=0
    )


def test_sampling_keeps_stock_frame_selection_and_timestamp_tokens(processors):
    stock, adapter = processors
    video = np.random.default_rng(29).integers(0, 256, (12, 65, 97, 3), dtype=np.uint8)
    kwargs = dict(
        text=["<|vision_start|><|video_pad|><|vision_end|>"],
        videos=[video],
        video_metadata=[VideoMetadata(total_num_frames=12, fps=6)],
        num_frames=4,
        fps=None,
        do_sample_frames=True,
        return_tensors="pt",
    )
    expected = stock(**copy.deepcopy(kwargs))
    actual = adapter(**copy.deepcopy(kwargs))
    torch.testing.assert_close(
        actual["input_ids"], expected["input_ids"], rtol=0, atol=0
    )
    selected = stock.video_processor.sample_frames(
        VideoMetadata(total_num_frames=12, fps=6),
        num_frames=4,
        fps=None,
    )
    np.testing.assert_array_equal(
        actual["pixel_values_videos_raw"][0].numpy(), video[selected]
    )


def test_float_pixels_are_rejected_instead_of_silently_quantized(processors):
    _, adapter = processors
    with pytest.raises(ValueError, match="CPU uint8 RGB"):
        adapter.image_processor(images=[torch.ones(3, 65, 97)], return_tensors="pt")


@pytest.mark.parametrize("numpy_input", [False, True])
@pytest.mark.parametrize("is_video", [False, True])
def test_source_rgb_transport_avoids_full_resolution_cpu_copies(
    processors, numpy_input, is_video
):
    _, adapter = processors
    frames = 3 if is_video else 1
    source = torch.arange(frames * 64 * 96 * 3).remainder(256).to(torch.uint8)
    source = source.reshape(frames, 64, 96, 3)
    pixels = source if is_video else source[0]
    pixels = pixels.numpy() if numpy_input else pixels
    if is_video:
        actual = adapter.video_processor(
            videos=[pixels], do_sample_frames=False, return_tensors="pt"
        )
        raw = actual["pixel_values_videos_raw"][0]
    else:
        actual = adapter.image_processor(images=[pixels], return_tensors="pt")
        raw = actual["pixel_values_raw"][0]
    torch.testing.assert_close(raw, source, rtol=0, atol=0)
    assert raw.data_ptr() == source.data_ptr()


def test_factory_loading_uses_the_same_pinned_checkpoint_contract(processors):
    stock, _ = processors
    model = stock.tokenizer.name_or_path
    revision = MODELS[model.rsplit("Cosmos3-", 1)[1]]
    adapter = Cosmos3CVCUDAProcessor.from_pretrained(model, revision=revision)
    assert adapter.image_processor.patch_size == stock.image_processor.patch_size
    assert adapter.video_processor.temporal_patch_size == 2
    actual = adapter.image_processor(
        images=[Image.new("RGB", (67, 99))], return_tensors="pt"
    )
    assert actual["pixel_values_raw"][0].shape == (1, 99, 67, 3)


def make_processing_info(stock, **config_kwargs):
    model = stock.tokenizer.name_or_path
    config = ModelConfig(
        model=model,
        revision=MODELS[model.rsplit("Cosmos3-", 1)[1]],
        dtype="bfloat16",
        seed=0,
        max_model_len=4096,
        mm_processor_cache_gb=0,
        **config_kwargs,
    )
    return Cosmos3ProcessingInfo(
        InputProcessingContext(config, tokenizer=stock.tokenizer)
    )


def test_native_default_keeps_stock_pixels_and_prompt_tokens(processors):
    stock, _ = processors
    info = make_processing_info(stock, mm_device_do_normalize=False)
    inputs = dict(
        text=["<|vision_start|><|image_pad|><|vision_end|>"],
        images=[Image.new("RGB", (67, 99), color=(47, 19, 211))],
        return_tensors="pt",
    )
    expected = stock(**copy.deepcopy(inputs))
    actual = info.get_hf_processor()(**copy.deepcopy(inputs))
    assert "pixel_values_raw" not in actual
    for key in ("input_ids", "image_grid_thw", "pixel_values"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


def test_native_apply_preserves_raw_videos_and_stock_prompt_contract(processors):
    """Exercise real public processing, including per-video raw-field transport."""
    stock, _ = processors
    stock_info = make_processing_info(
        stock,
        mm_device_do_normalize=True,
        limit_mm_per_prompt={"image": 1, "video": 2},
    )
    raw_info = make_processing_info(
        stock,
        mm_device_do_normalize=True,
        mm_processor_kwargs={"device_resize": True},
        limit_mm_per_prompt={"image": 1, "video": 2},
    )
    baseline = Cosmos3RawMultiModalProcessor(
        stock_info, Qwen3VLDummyInputsBuilder(stock_info)
    )
    candidate = Cosmos3RawMultiModalProcessor(
        raw_info, Qwen3VLDummyInputsBuilder(raw_info)
    )
    generator = torch.Generator().manual_seed(131)
    image = torch.randint(0, 256, (67, 99, 3), dtype=torch.uint8, generator=generator)
    videos = [
        torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)
        for shape in ((4, 67, 99, 3), (6, 99, 67, 3))
    ]
    video_data = [
        (
            frames.numpy(),
            {
                "total_num_frames": len(frames),
                "fps": fps,
                "frames_indices": list(range(len(frames))),
                "video_backend": "opencv",
                "do_sample_frames": False,
            },
        )
        for frames, fps in zip(videos, (7, 11), strict=True)
    ]
    prompt = (
        "<|vision_start|><|image_pad|><|vision_end|>"
        + "<|vision_start|><|video_pad|><|vision_end|>" * 2
        + "Describe the image and both videos."
    )
    mm_data = {"image": [image.numpy()], "video": video_data}

    def apply_processor(processor):
        inputs = ProcessorInputs(
            prompt=stock.tokenizer.encode(prompt),
            mm_data_items=processor.info.parse_mm_data(mm_data),
            hf_processor_mm_kwargs={"videos_kwargs": {"do_sample_frames": False}},
        )
        return processor.apply(inputs, TimingContext(enabled=False))

    expected = apply_processor(baseline)
    actual = apply_processor(candidate)
    assert actual["prompt_token_ids"] == expected["prompt_token_ids"]
    assert actual["mm_placeholders"] == expected["mm_placeholders"]
    assert len(actual["mm_kwargs"]["video"]) == len(videos)
    assert len(actual["mm_hashes"]["video"]) == len(videos)
    raw_image = actual["mm_kwargs"]["image"][0]
    assert "pixel_values" not in raw_image
    assert raw_image["pixel_values_raw"].field.keep_on_cpu
    assert raw_image["pixel_values_raw"].data.data_ptr() == image.data_ptr()
    torch.testing.assert_close(
        raw_image["pixel_values_raw"].data[0], image, rtol=0, atol=0
    )
    torch.testing.assert_close(
        raw_image["image_grid_thw"].data,
        expected["mm_kwargs"]["image"][0]["image_grid_thw"].data,
        rtol=0,
        atol=0,
    )
    token_count = 0
    for source, raw_video, stock_video in zip(
        videos,
        actual["mm_kwargs"]["video"],
        expected["mm_kwargs"]["video"],
        strict=True,
    ):
        assert "pixel_values_videos" not in raw_video
        assert "pixel_values_videos_raw" not in stock_video
        assert stock_video["pixel_values_videos"].data.numel() > 0
        transported = raw_video["pixel_values_videos_raw"]
        assert transported.field.keep_on_cpu
        assert transported.data.device.type == "cpu"
        assert transported.data.data_ptr() == source.data_ptr()
        torch.testing.assert_close(transported.data, source, rtol=0, atol=0)
        torch.testing.assert_close(
            raw_video["video_grid_thw"].data,
            stock_video["video_grid_thw"].data,
            rtol=0,
            atol=0,
        )
        assert raw_video["timestamps"].data == stock_video["timestamps"].data
        token_count += int(raw_video["video_grid_thw"].data.prod())
    token_count //= stock.image_processor.merge_size**2
    video_token_id = raw_info.get_hf_config().video_token_id
    assert actual["prompt_token_ids"].count(video_token_id) == token_count


@pytest.mark.parametrize(
    "request_kwargs",
    [
        {"device_resize": True},
        {"images_kwargs": {"device_resize": True}},
        {"videos_kwargs": {"device_resize": True}},
    ],
)
def test_native_request_cannot_enable_worker_resize(processors, request_kwargs):
    stock, _ = processors
    info = make_processing_info(stock)
    with pytest.raises(ValueError, match="enabled in engine mm_processor_kwargs"):
        info.get_hf_processor(**request_kwargs)


def test_native_resize_rejects_disabled_worker_normalization(processors):
    stock, _ = processors
    info = make_processing_info(
        stock,
        mm_device_do_normalize=False,
        mm_processor_kwargs={"device_resize": True},
    )
    with pytest.raises(ValueError, match="requires mm_device_do_normalize=True"):
        info.get_hf_processor()


@pytest.mark.parametrize(
    "cuda_version,distribution",
    [
        ("12.8", "cvcuda-cu12"),
        ("13.2", "cvcuda-cu13"),
    ],
)
def test_public_release_uses_matching_cuda_distribution(cuda_version, distribution):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _cvcuda_distribution,
    )

    assert _cvcuda_distribution(cuda_version) == distribution


@pytest.mark.parametrize("cuda_version", [None, "11.8", "14.0"])
def test_public_release_rejects_unsupported_cuda_runtime(cuda_version):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _cvcuda_distribution,
    )

    with pytest.raises(RuntimeError, match="CUDA 12 or 13"):
        _cvcuda_distribution(cuda_version)


@pytest.mark.parametrize(
    "version",
    [
        "0.18.0",
        "0.21.0",
        "99.0.0",
        "0.19.0+private",
        "0.20.0+private",
        "0.20.0rc1",
        "0.20.0.post1",
    ],
)
def test_public_release_rejects_other_dependency_versions(version):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _validate_cvcuda_release_version,
    )

    with pytest.raises(RuntimeError, match="CV-CUDA 0.19.0 or 0.20.0"):
        _validate_cvcuda_release_version(version)


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
def test_public_release_accepts_exact_dependency_version(version):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _validate_cvcuda_release_version,
    )

    _validate_cvcuda_release_version(version)


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
@pytest.mark.parametrize("module_version", [None, "0.21.0"])
def test_public_release_api_rejects_mismatched_module_version(version, module_version):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _validate_cvcuda_api,
    )

    with pytest.raises(
        RuntimeError, match="module and distribution versions must match"
    ):
        _validate_cvcuda_api(version, {"__version__": module_version})


@pytest.mark.parametrize("version", ["0.19.0", "0.20.0"])
def test_public_release_api_requires_exported_resize(version):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _validate_cvcuda_api,
    )

    with pytest.raises(
        RuntimeError, match="Required public CV-CUDA API.*tvresize_into"
    ):
        _validate_cvcuda_api(version, {"__version__": version})


@pytest.mark.parametrize(
    "frames,size,budget",
    [
        (torch.zeros((0, 5, 7, 3), dtype=torch.uint8), (4, 4), 1024),
        (torch.zeros((1, 5, 7, 4), dtype=torch.uint8), (4, 4), 1024),
        (torch.zeros((1, 5, 7, 3), dtype=torch.float32), (4, 4), 1024),
        (torch.zeros((1, 5, 7, 3), dtype=torch.uint8), (0, 4), 1024),
        (torch.zeros((1, 5, 7, 3), dtype=torch.uint8), (4, 4), 0),
    ],
)
def test_public_resize_rejects_invalid_inputs_before_cuda(frames, size, budget):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        cvcuda_bicubic_resize,
    )

    with pytest.raises(ValueError):
        cvcuda_bicubic_resize(frames, size, torch.device("cpu"), budget)


def test_public_resize_requires_real_cuda_device():
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        cvcuda_bicubic_resize,
    )

    with pytest.raises(ValueError, match="requires a CUDA device"):
        cvcuda_bicubic_resize(
            torch.zeros((1, 5, 7, 3), dtype=torch.uint8), (4, 4), torch.device("cpu")
        )


@pytest.mark.parametrize(
    "source,size,frames,expected",
    [
        ((5, 7), (5, 7), 3, 0),
        ((5, 7), (5, 4), 3, 720),
        ((5, 7), (4, 7), 3, 1260),
        ((5, 7), (4, 4), 1, 240),
        ((5, 7), (4, 4), 3, 720),
        ((1, 3), (1, 7), 1, 84),
        ((1, 32), (1, 1), 1, 12),
        ((16, 16), (32, 32), 1, 14848),
    ],
)
def test_public_workspace_matches_original_release(source, size, frames, expected):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _cvcuda_workspace_bytes,
    )

    assert _cvcuda_workspace_bytes(source, size, frames) == expected


@pytest.mark.parametrize("budget", [1024 * 1024, 512 * 1024 * 1024])
def test_public_warmup_is_maximal_without_exceeding_native_budget(budget):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _cvcuda_warmup_rows,
        _cvcuda_workspace_bytes,
    )

    rows = _cvcuda_warmup_rows(budget)
    assert _cvcuda_workspace_bytes((rows, 16), (16, 4096), 1) <= budget
    upper = budget // (12 * 4096)
    assert all(
        _cvcuda_workspace_bytes((candidate, 16), (16, 4096), 1) > budget
        for candidate in range(rows + 1, upper + 2)
    )


@pytest.mark.parametrize(
    "source,size,frames",
    [
        ((0, 7), (4, 4), 1),
        ((5, 7), (2**31, 4), 1),
        ((5, 7), (4, 4), 0),
        ((5, 7), (4, 4), 65536),
    ],
)
def test_public_workspace_rejects_unsupported_native_dimensions(source, size, frames):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        _cvcuda_workspace_bytes,
    )

    with pytest.raises(ValueError, match="CV-CUDA"):
        _cvcuda_workspace_bytes(source, size, frames)


def test_public_chunk_plan_charges_uint8_uploads_resize_and_native_scratch():
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        ALLOCATION_OVERHEAD_BYTES,
        resize_plan,
    )

    source, size = (193, 319), (160, 224)
    frame_bytes = 6 * source[0] * source[1] + 3 * size[0] * size[1]
    budget = 2 * frame_bytes + ALLOCATION_OVERHEAD_BYTES
    plan = resize_plan(source, size, 3, budget)
    assert plan.frame_step == 2
    assert plan.temporary_bytes == budget
    single = resize_plan(source, size, 3, frame_bytes + ALLOCATION_OVERHEAD_BYTES)
    assert single.frame_step == 1
    assert single.native_bytes < plan.native_bytes


@pytest.mark.parametrize(
    "source,target,axis",
    [
        ((17, 23), 11, 1),
        ((17, 23), 19, 2),
        ((33, 48), 41, 1),
        ((33, 48), 19, 2),
    ],
)
def test_cpu_fp32_oracle_rounds_and_clamps_only_after_resize(source, target, axis):
    pixels = torch.randint(
        0,
        256,
        (1, *source, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(1234),
    )
    size = list(source)
    size[axis - 1] = target
    floating = torch.nn.functional.interpolate(
        pixels.permute(0, 3, 1, 2).float(),
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    oracle = floating.round().clamp(0, 255).to(torch.uint8)
    assert oracle.dtype == torch.uint8
    assert tuple(oracle.shape) == (1, 3, *size)
    assert torch.equal(oracle, torch.clip(torch.round(floating), 0, 255).byte())


@pytest.mark.parametrize(
    "source,size",
    [
        ((32, 48), (32, 48)),
        ((17, 23), (25, 31)),
        ((67, 97), (32, 48)),
        ((67, 97), (67, 64)),
        ((67, 97), (48, 97)),
        ((193, 319), (160, 224)),
    ],
)
@pytest.mark.parametrize("num_frames", [1, 3])
@pytest.mark.parametrize("default_stream", [False, True])
def test_gpu_public_resize_preserves_pixels_and_stream_lifetime(
    source, size, num_frames, default_stream
):
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        cvcuda_bicubic_resize,
    )

    frames = torch.randint(
        0,
        256,
        (num_frames, *source, 3),
        dtype=torch.uint8,
        generator=torch.Generator().manual_seed(1234),
    )
    reference = torch.nn.functional.interpolate(
        frames.permute(0, 3, 1, 2).float(),
        size=size,
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    device = torch.device("cuda", 0)
    stream = (
        torch.cuda.default_stream(device)
        if default_stream
        else torch.cuda.Stream(device)
    )
    with torch.cuda.stream(stream):
        resized = cvcuda_bicubic_resize(frames, size, device)
        retained = resized.clone()
        del resized
        torch.empty_like(retained).fill_(99)
        completion = torch.cuda.Event()
        completion.record(stream)
    completion.synchronize()
    reference = reference.round().clamp(0, 255).to(torch.uint8)
    difference = (retained.cpu().int() - reference.int()).abs()
    assert difference.max() <= 1
    assert (difference > 0).float().mean() < 1e-3
