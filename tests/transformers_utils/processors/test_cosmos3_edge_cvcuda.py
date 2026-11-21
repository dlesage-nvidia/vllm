# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import AutoProcessor
from transformers.image_utils import PILImageResampling
from transformers.video_utils import VideoMetadata

from vllm.config import ModelConfig
from vllm.model_executor.models.cosmos3_edge import (
    Cosmos3EdgeDummyInputsBuilder,
    Cosmos3EdgeMultiModalProcessor,
    Cosmos3EdgeProcessingInfo,
)
from vllm.multimodal.inputs import MultiModalKwargsItems
from vllm.multimodal.processing.context import InputProcessingContext
from vllm.transformers_utils.processors.cosmos3_edge_cvcuda import (
    Cosmos3EdgeCVCUDAProcessor,
    cvcuda_cosmos3_edge_patchify,
    normalize_cosmos3_edge_frames,
    pack_cosmos3_edge_frames,
)

MODELS = {
    "Edge": "6afc1256321c3a8e623d89be3bc5117a1e90f96b",
    "Edge-FP8": "9a55df1413b0ca2a95cc5175476fd3dee64a3d67",
}
IMAGE_PROMPT = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PROMPT = "<|vision_start|><|video_pad|><|vision_end|>"


@pytest.fixture(scope="module", params=MODELS)
def processors(request):
    name = request.param
    stock = AutoProcessor.from_pretrained(
        f"nvidia/Cosmos3-{name}", revision=MODELS[name], local_files_only=True
    )
    adapter = Cosmos3EdgeCVCUDAProcessor(
        image_processor=stock.image_processor,
        video_processor=stock.video_processor,
        tokenizer=stock.tokenizer,
        chat_template=stock.chat_template,
    )
    return stock, adapter


@pytest.mark.parametrize("shape", [(64, 96), (193, 319)])
@pytest.mark.parametrize("is_video", [False, True], ids=["image", "video"])
def test_gpu_edge_patchify_matches_stock_uint8(processors, shape, is_video):
    """Compare real SigLIP2 CUDA patches with HF bytes at the unchanged <=1 gate."""
    stock, adapter = processors
    processor = stock.video_processor if is_video else stock.image_processor
    pixel_key = "pixel_values_videos" if is_video else "pixel_values"
    grid_key = "video_grid_thw" if is_video else "image_grid_thw"
    pixels = np.random.default_rng(71).integers(
        0, 256, (3 if is_video else 1, *shape, 3), dtype=np.uint8
    )
    inputs = dict(
        text=[VIDEO_PROMPT if is_video else IMAGE_PROMPT],
        return_tensors="pt",
        do_rescale=False,
        do_normalize=False,
        size={"shortest_edge": 4096, "longest_edge": 8192},
    )
    if is_video:
        inputs.update(
            videos=[pixels],
            do_sample_frames=False,
            video_metadata=[
                VideoMetadata(total_num_frames=3, fps=7, frames_indices=[0, 1, 2])
            ],
        )
    else:
        inputs["images"] = [pixels[0]]
    expected = stock(**copy.deepcopy(inputs))
    raw = adapter(**copy.deepcopy(inputs))
    torch.testing.assert_close(raw["input_ids"], expected["input_ids"], rtol=0, atol=0)
    torch.testing.assert_close(raw[grid_key], expected[grid_key], rtol=0, atol=0)
    assert expected[pixel_key].dtype == torch.uint8
    device = torch.device("cuda", 0)
    actual = cvcuda_cosmos3_edge_patchify(
        raw[f"{pixel_key}_raw"],
        raw[grid_key],
        device,
        image_processor=processor,
    )
    assert actual.device == device
    assert actual.dtype == torch.uint8
    expected_pixels = expected[pixel_key].reshape(-1, 3 * processor.patch_size**2)
    assert actual.shape == expected_pixels.shape
    max_uint8_error = int(
        (actual.cpu().to(torch.int16) - expected_pixels.to(torch.int16)).abs().max()
    )
    assert max_uint8_error <= 1, (
        f"{stock.tokenizer.name_or_path}, shape={shape}, is_video={is_video}: "
        f"max_uint8_error={max_uint8_error} exceeds 1"
    )


def make_native_processor(stock):
    model = stock.tokenizer.name_or_path
    name = model.rsplit("Cosmos3-", 1)[1]
    config = ModelConfig(
        model=model,
        revision=MODELS[name],
        dtype="bfloat16",
        seed=0,
        max_model_len=4096,
        mm_processor_cache_gb=0,
    )
    info = Cosmos3EdgeProcessingInfo(
        InputProcessingContext(config, tokenizer=stock.tokenizer)
    )
    return Cosmos3EdgeMultiModalProcessor(
        info=info, dummy_inputs=Cosmos3EdgeDummyInputsBuilder(info)
    )


@pytest.mark.parametrize("shape", [(31, 65), (67, 97), (160, 320)])
@pytest.mark.parametrize("size", [None, {"shortest_edge": 4096, "longest_edge": 8192}])
def test_edge_images_preserve_source_bytes_grid_and_tokens(processors, shape, size):
    stock, adapter = processors
    pixels = np.random.default_rng(42).integers(0, 256, (*shape, 3), dtype=np.uint8)
    images = [Image.fromarray(pixels), Image.fromarray(pixels.transpose(1, 0, 2))]
    kwargs = dict(text=[IMAGE_PROMPT * 2], images=images, return_tensors="pt")
    if size is not None:
        kwargs["images_kwargs"] = {"size": size}
    expected = stock(**copy.deepcopy(kwargs))
    actual = adapter(**copy.deepcopy(kwargs))
    for key in ("input_ids", "attention_mask", "image_grid_thw"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["pixel_values"].shape == (0, 768)
    for (raw, transform), image in zip(actual["pixel_values_raw"], images):
        assert raw.dtype == torch.uint8 and raw.device.type == "cpu"
        np.testing.assert_array_equal(raw[0].numpy(), np.asarray(image))
        assert transform.device.type == "cpu" and transform.dtype == torch.float64


@pytest.mark.parametrize("num_frames", [1, 3, 8])
@pytest.mark.parametrize("sample", [False, True])
def test_edge_video_sampling_has_no_temporal_padding_and_keeps_timestamps(
    processors, num_frames, sample
):
    stock, adapter = processors
    video = np.random.default_rng(7).integers(
        0, 256, (num_frames, 67, 97, 3), dtype=np.uint8
    )
    metadata = VideoMetadata(
        total_num_frames=num_frames,
        fps=4,
        duration=num_frames / 4,
        frames_indices=list(range(num_frames)),
    )
    kwargs = dict(
        text=[VIDEO_PROMPT],
        videos=[video],
        video_metadata=[metadata],
        do_sample_frames=sample,
        return_tensors="pt",
    )
    expected = stock(**copy.deepcopy(kwargs))
    actual = adapter(**copy.deepcopy(kwargs))
    for key in ("input_ids", "attention_mask", "video_grid_thw"):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual["pixel_values_videos"].shape == (0, 768)
    indices = (
        stock.video_processor.sample_frames(copy.deepcopy(metadata))
        if sample
        else np.arange(num_frames)
    )
    raw, _ = actual["pixel_values_videos_raw"][0]
    np.testing.assert_array_equal(raw.numpy(), video[indices])
    assert actual["video_grid_thw"][0, 0] == len(indices)
    tokens = stock.tokenizer.batch_decode(actual["input_ids"])[0]
    assert tokens.count("<|vision_start|>") == len(indices)
    assert tokens.count("<|vision_end|>") == len(indices)


@pytest.mark.parametrize("is_video", [False, True])
@pytest.mark.parametrize(
    "flags", [(True, True), (True, False), (False, True), (False, False)]
)
def test_edge_hwc_packing_and_effective_normalization_match_stock(
    processors, is_video, flags
):
    stock, adapter = processors
    image_processor = stock.video_processor if is_video else stock.image_processor
    raw_processor = adapter.video_processor if is_video else adapter.image_processor
    frames = (
        torch.arange(3 * 3 * 64 * 96)
        .remainder(256)
        .to(torch.uint8)
        .reshape(3, 3, 64, 96)
    )
    if not is_video:
        frames = frames[:1]
    kwargs = dict(
        do_resize=False,
        do_rescale=flags[0],
        do_normalize=flags[1],
        rescale_factor=1 / 127,
        image_mean=[0.1, 0.2, 0.3],
        image_std=[0.25, 0.5, 0.75],
        return_tensors="pt",
    )
    if is_video:
        kwargs.update(videos=[frames], do_sample_frames=False)
        pixel_key, grid_key = "pixel_values_videos", "video_grid_thw"
    else:
        kwargs.update(images=[frames[0]])
        pixel_key, grid_key = "pixel_values", "image_grid_thw"
    expected = image_processor(**kwargs)
    actual = raw_processor(**kwargs)
    raw, transform = actual[f"{pixel_key}_raw"][0]
    normalized = normalize_cosmos3_edge_frames(
        raw.permute(0, 3, 1, 2), transform, image_processor=image_processor
    )
    packed = pack_cosmos3_edge_frames(normalized, actual[grid_key][0].tolist())
    torch.testing.assert_close(
        packed, expected[pixel_key].reshape(-1, 768), rtol=0, atol=0
    )
    if not any(flags):
        np.testing.assert_array_equal(
            packed[0, :6].numpy(), raw[0, 0, :2].numpy().reshape(-1)
        )


@pytest.mark.parametrize("mode", ["RGB", "RGBA", "L"])
def test_edge_stock_rgb_conversion_is_before_raw_transport(processors, mode):
    stock, adapter = processors
    image = Image.new(mode, (67, 99), 37)
    actual = adapter.image_processor(images=[image], return_tensors="pt")
    raw, _ = actual["pixel_values_raw"][0]
    expected = stock.image_processor.process_image(image, do_convert_rgb=True)
    torch.testing.assert_close(raw[0].permute(2, 0, 1), expected, rtol=0, atol=0)


@pytest.mark.parametrize("is_video", [False, True])
def test_edge_explicit_opt_out_retains_stock_dense_output(processors, is_video):
    stock, adapter = processors
    pixels = np.zeros((3, 67, 97, 3), dtype=np.uint8)
    kwargs: dict[str, object] = dict(
        text=[VIDEO_PROMPT if is_video else IMAGE_PROMPT], return_tensors="pt"
    )
    if is_video:
        kwargs.update(videos=[pixels], do_sample_frames=False)
    else:
        kwargs.update(images=[Image.fromarray(pixels[0])])
    expected = stock(**copy.deepcopy(kwargs))
    actual = adapter(**copy.deepcopy(kwargs), device_resize=False)
    assert expected.keys() == actual.keys()
    for key in expected:
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)


@pytest.mark.parametrize("is_video", [False, True])
def test_edge_native_field_batching_keeps_raw_settings_and_drops_empty_dense(
    processors, is_video
):
    stock, adapter = processors
    native = make_native_processor(stock)
    media = np.zeros((3, 64, 96, 3), dtype=np.uint8)
    if is_video:
        output = adapter(
            text=[VIDEO_PROMPT * 2],
            videos=[media, media[:, :, :64]],
            do_sample_frames=False,
            return_tensors="pt",
        )
        modality, pixel_key = "video", "pixel_values_videos"
    else:
        output = adapter(
            text=[IMAGE_PROMPT * 2],
            images=[Image.fromarray(media[0]), Image.fromarray(media[0, :, :64])],
            return_tensors="pt",
        )
        modality, pixel_key = "image", "pixel_values"
    fields = native._get_mm_fields_config(output, {})
    items = MultiModalKwargsItems.from_hf_inputs(output, fields)
    batched = items.get_data(device="cpu", pin_memory=False)
    assert pixel_key not in batched
    assert len(items[modality]) == 2
    for expected, actual in zip(
        output[f"{pixel_key}_raw"], batched[f"{pixel_key}_raw"]
    ):
        for original, transported in zip(expected, actual):
            torch.testing.assert_close(transported, original, rtol=0, atol=0)
    assert not native._expands_only_video_token(stock)
    assert type(native.info.get_hf_processor()) is type(stock)


@pytest.mark.parametrize("is_video", [False, True])
@pytest.mark.parametrize(
    "overrides",
    [
        {"resample": PILImageResampling.BILINEAR},
        {"patch_size": 8},
        {"merge_size": 1},
        {"temporal_patch_size": 2},
    ],
)
def test_edge_opt_in_rejects_unsupported_resampling_and_encoder_shapes(
    processors, is_video, overrides
):
    _, adapter = processors
    processor = adapter.video_processor if is_video else adapter.image_processor
    kwargs = dict(return_tensors="pt", **overrides)
    if is_video:
        kwargs.update(
            videos=[torch.zeros((3, 3, 64, 96), dtype=torch.uint8)],
            do_sample_frames=False,
        )
    else:
        kwargs.update(images=[Image.new("RGB", (96, 64))])
    with pytest.raises(ValueError, match="Edge|Cosmos3 Edge"):
        processor(**kwargs)


def test_edge_pack_rejects_qwen_temporal_doubling():
    with pytest.raises(ValueError, match="unpadded"):
        pack_cosmos3_edge_frames(torch.zeros((2, 3, 64, 96)), [1, 4, 6])


@pytest.mark.parametrize("rescale", [False, True])
def test_edge_zero_scalar_mean_is_not_replaced_by_checkpoint_default(
    processors, rescale
):
    stock, adapter = processors
    image = torch.full((3, 64, 96), 37, dtype=torch.uint8)
    kwargs = dict(
        images=[image],
        do_resize=False,
        do_rescale=rescale,
        image_mean=0.0,
        image_std=0.25,
        return_tensors="pt",
    )
    expected = stock.image_processor(**kwargs)
    actual = adapter.image_processor(**kwargs)
    raw, transform = actual["pixel_values_raw"][0]
    normalized = normalize_cosmos3_edge_frames(
        raw.permute(0, 3, 1, 2), transform, image_processor=stock.image_processor
    )
    torch.testing.assert_close(
        pack_cosmos3_edge_frames(normalized, actual["image_grid_thw"][0].tolist()),
        expected["pixel_values"].reshape(-1, 768),
        rtol=0,
        atol=0,
    )


def test_edge_factory_uses_both_pinned_checkpoint_processors(processors):
    stock, _ = processors
    model = stock.tokenizer.name_or_path
    name = model.rsplit("Cosmos3-", 1)[1]
    adapter = Cosmos3EdgeCVCUDAProcessor.from_pretrained(
        model, revision=MODELS[name], local_files_only=True
    )
    assert adapter.image_processor.patch_size == 16
    assert adapter.video_processor.temporal_patch_size == 1
    output = adapter.image_processor(
        images=[Image.new("RGB", (67, 99))], return_tensors="pt"
    )
    assert output["pixel_values_raw"][0][0].shape == (1, 99, 67, 3)


@pytest.mark.parametrize("is_video", [False, True])
def test_edge_raw_transport_rejects_non_uint8_sources(processors, is_video):
    _, adapter = processors
    if is_video:
        processor = adapter.video_processor
        kwargs = dict(videos=[torch.zeros((3, 3, 64, 96))], do_sample_frames=False)
    else:
        processor = adapter.image_processor
        kwargs = dict(images=[torch.zeros((3, 64, 96))])
    with pytest.raises(ValueError, match="CPU uint8 RGB"):
        processor(**kwargs)


@pytest.mark.parametrize("is_video", [False, True])
@pytest.mark.parametrize("as_numpy", [False, True])
@pytest.mark.parametrize("batched", [False, True])
def test_edge_interleaved_source_transport_preserves_alias(
    processors, is_video, as_numpy, batched
):
    _, adapter = processors
    shape = (3, 64, 96, 3) if is_video else (64, 96, 3)
    if batched:
        shape = (2, *shape)
    source = torch.arange(np.prod(shape)).remainder(256).to(torch.uint8).reshape(shape)
    original = source.clone()
    supplied = source.numpy() if as_numpy else source
    if is_video:
        output = adapter.video_processor(
            videos=supplied, do_sample_frames=False, return_tensors="pt"
        )
        raw_items = output["pixel_values_videos_raw"]
    else:
        output = adapter.image_processor(images=supplied, return_tensors="pt")
        raw_items = output["pixel_values_raw"]
    expected_items = list(source.unbind(0)) if batched else [source]
    torch.testing.assert_close(source, original, rtol=0, atol=0)
    assert len(raw_items) == len(expected_items)
    for (raw, _), expected in zip(raw_items, expected_items):
        assert raw.data_ptr() == expected.data_ptr()
        torch.testing.assert_close(
            raw if is_video else raw[0], expected, rtol=0, atol=0
        )
