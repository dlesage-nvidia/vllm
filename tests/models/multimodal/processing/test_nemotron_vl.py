# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for Nemotron-Nano-VL's multimodal preprocessing kwargs."""

from collections.abc import Mapping

import numpy as np
import pytest
import torch
from PIL import Image
from transformers import PreTrainedConfig

from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.cache import MultiModalProcessorOnlyCache
from vllm.multimodal.image import rescale_image_size
from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.processing import BaseMultiModalProcessor
from vllm.transformers_utils.processors.nano_nemotron_vl import dynamic_preprocess

from ....conftest import ImageTestAssets
from ...utils import build_model_context


@pytest.mark.parametrize(
    "model_id",
    [
        "nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16",
        "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16",
    ],
)
@pytest.mark.parametrize("cache_enabled", [False, True])
@pytest.mark.parametrize("device_normalization", [False, True])
@pytest.mark.parametrize(
    "interleaved_storage", [False, True], ids=["planar", "interleaved"]
)
def test_nemotron_tensor_processor_matches_pillow(
    model_id: str,
    cache_enabled: bool,
    device_normalization: bool,
    interleaved_storage: bool,
) -> None:
    model_config = ModelConfig(
        model_id,
        trust_remote_code=True,
        dtype="bfloat16",
        seed=0,
        max_model_len=4096,
        limit_mm_per_prompt={"image": 1, "video": 0},
        mm_processor_cache_gb=int(cache_enabled),
        mm_device_do_normalize=device_normalization,
    )
    processor = MULTIMODAL_REGISTRY.create_processor(model_config)
    cache = MultiModalProcessorOnlyCache(model_config) if cache_enabled else None
    pixels = (np.arange(64 * 200 * 3).reshape(64, 200, 3) % 251).astype(np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
    if interleaved_storage:
        tensor = torch.from_numpy(pixels).permute(2, 0, 1)
    snapshot = tensor.clone()
    image = MediaWithBytes(tensor, b"same RGB image", {"backend": "nvimagecodec"})

    def process(item):
        items = processor.info.parse_mm_data({"image": [item]})
        return processor("<image>Describe the image.", mm_items=items, cache=cache)

    reference = process(Image.fromarray(pixels))
    results = [process(image), process(image)]
    for result in results:
        for key in ("mm_kwargs", "prompt_token_ids", "mm_placeholders"):
            assert result[key] == reference[key]
    if cache is not None:
        assert cache.make_stats().hits == 1
    assert torch.equal(tensor, snapshot)


@pytest.mark.parametrize("use_thumbnail", [False, True])
@pytest.mark.parametrize("do_cpu_normalize", [False, True])
def test_nemotron_tensor_tiles_match_pillow(
    use_thumbnail: bool, do_cpu_normalize: bool
) -> None:
    pixels = (np.arange(48 * 128 * 3).reshape(48, 128, 3) % 251).astype(np.uint8)
    tensor = torch.from_numpy(np.ascontiguousarray(pixels.transpose(2, 0, 1)))
    kwargs = dict(
        image_size=32,
        max_num_tiles=6,
        use_thumbnail=use_thumbnail,
        do_cpu_normalize=do_cpu_normalize,
    )
    reference = dynamic_preprocess(Image.fromarray(pixels), **kwargs)
    result = dynamic_preprocess(tensor, **kwargs)
    assert torch.equal(result, reference)


@pytest.mark.parametrize("interleaved_storage", [False, True])
def test_nemotron_chw_view_preserves_shared_storage(
    interleaved_storage: bool,
) -> None:
    from vllm.transformers_utils.processors.nano_nemotron_vl import (
        _image_to_nhwc_tensor,
    )

    image = torch.arange(3 * 8 * 12, dtype=torch.uint8).reshape(3, 8, 12)
    if interleaved_storage:
        image = image.permute(1, 2, 0).contiguous().permute(2, 0, 1)
    result = _image_to_nhwc_tensor(image)
    assert result.shape == (1, 8, 12, 3)
    assert result.untyped_storage().data_ptr() == image.untyped_storage().data_ptr()
    assert torch.equal(result.permute(0, 3, 1, 2)[0], image)


@pytest.mark.parametrize(
    "image",
    [
        torch.zeros(3, 8, 12, dtype=torch.float32),
        torch.zeros(8, 12, 3, dtype=torch.uint8),
        torch.zeros(3, 8, 12, dtype=torch.uint8).transpose(1, 2),
    ],
)
def test_nemotron_rejects_invalid_tensor_image(image: torch.Tensor) -> None:
    from vllm.transformers_utils.processors.nano_nemotron_vl import (
        _image_to_nhwc_tensor,
    )

    with pytest.raises(ValueError, match="dense CPU uint8 CHW RGB"):
        _image_to_nhwc_tensor(image)


def _get_expected_num_patches(
    config: PreTrainedConfig,
    image: Image.Image,
    num_imgs: int,
    min_num: int,
    max_num: int,
):
    from vllm.transformers_utils.processors.nemotron_vl import (
        calculate_nemotron_vl_targets,
        get_nemotron_vl_target_ratios,
    )

    width, height = image.size

    blocks, _, _ = calculate_nemotron_vl_targets(
        orig_width=width,
        orig_height=height,
        target_ratios=get_nemotron_vl_target_ratios(
            min_num,
            max_num,
        ),
        image_size=config.force_image_size,
        use_thumbnail=False,
    )
    expected_num_patches = blocks

    if config.use_thumbnail and expected_num_patches > 1:
        expected_num_patches += 1

    return expected_num_patches


def _run_check(
    processor: BaseMultiModalProcessor,
    images: list[Image.Image],
    min_num: int,
    max_num: int,
    mm_processor_kwargs: Mapping[str, object],
):
    tokenizer = processor.info.get_tokenizer()
    config = processor.info.get_hf_config()
    image_processor = processor.info.get_image_processor()

    config.use_thumbnail = image_processor.use_thumbnail
    prompt = "<image>" * len(images)
    mm_data = {"image": images}

    total_expected_num_patches = sum(
        _get_expected_num_patches(config, image, len(images), min_num, max_num)
        for image in images
    )
    print(total_expected_num_patches)
    processed_inputs = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=mm_processor_kwargs,
    )

    # Ensure we have the right number of placeholders per num_crops size
    image_token_id = tokenizer.convert_tokens_to_ids("<image>")
    img_tok_count = processed_inputs["prompt_token_ids"].count(image_token_id)
    pixel_values = processed_inputs["mm_kwargs"].get_data()["pixel_values_flat"]
    assert isinstance(pixel_values, torch.Tensor)
    pixel_shape = pixel_values.shape
    print("Image token count:", img_tok_count, "Pixel shape:", pixel_shape)
    assert img_tok_count == 256 * total_expected_num_patches
    assert pixel_shape[0] == total_expected_num_patches


@pytest.mark.parametrize("model_id", ["nvidia/Llama-3.1-Nemotron-Nano-VL-8B-V1"])
@pytest.mark.parametrize(
    "size_factors",
    [
        # Single-scale
        [1.0],
        # Single-scale, batched
        [1.0, 1.0, 1.0],
        # Multi-scale
        [0.25, 0.5, 1.0],
        [4.0, 2.0, 1.0],
    ],
)
@pytest.mark.parametrize(
    ("min_dynamic_patch", "max_dynamic_patch"),
    [(1, 1), (1, 2), (1, 4), (1, 8), (2, 4), (4, 8)],
)
@pytest.mark.parametrize("dynamic_image_size", [True, False])
@pytest.mark.parametrize("kwargs_on_init", [True, False])
def test_processor_override(
    model_id: str,
    image_assets: ImageTestAssets,
    size_factors: list[int],
    min_dynamic_patch: int,
    max_dynamic_patch: int,
    dynamic_image_size: bool | None,
    kwargs_on_init: bool,
):
    mm_processor_kwargs = {
        "min_dynamic_patch": min_dynamic_patch,
        "max_dynamic_patch": max_dynamic_patch,
        "dynamic_image_size": dynamic_image_size,
    }

    ctx = build_model_context(
        model_id,
        mm_processor_kwargs=mm_processor_kwargs if kwargs_on_init else None,
        limit_mm_per_prompt={"image": len(size_factors)},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)
    hf_processor_mm_kwargs = {} if kwargs_on_init else mm_processor_kwargs

    min_num = min_dynamic_patch if dynamic_image_size else 1
    max_num = max_dynamic_patch if dynamic_image_size else 1

    _run_check(
        processor,
        [rescale_image_size(image_assets[0].pil_image, f) for f in size_factors],
        min_num,
        max_num,
        hf_processor_mm_kwargs,
    )
