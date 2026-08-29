# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import numpy as np
import pytest
import torch

from vllm.assets.video import VideoAsset
from vllm.config import ModelConfig
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.video import VLLM_VIDEO_INPUT_DATA_FORMAT_KEY
from vllm.multimodal.video_decoders import processor_video_resize_target

from ....conftest import ImageTestAssets
from ...utils import build_model_context

MODEL_ID = "nvidia/Cosmos3-Edge"
IMAGE_PLACEHOLDER = "<|vision_start|><|image_pad|><|vision_end|>"
VIDEO_PLACEHOLDER = "<|vision_start|><|video_pad|><|vision_end|>"
LOCAL_MODEL_PATH = os.getenv("COSMOS3_EDGE_MODEL_PATH")


@pytest.fixture(scope="module")
def processor():
    if LOCAL_MODEL_PATH:
        model_config = ModelConfig(
            LOCAL_MODEL_PATH,
            tokenizer=LOCAL_MODEL_PATH,
            max_model_len=4096,
            limit_mm_per_prompt={"image": 2, "video": 1},
        )
    else:
        ctx = build_model_context(
            MODEL_ID,
            limit_mm_per_prompt={"image": 2, "video": 1},
        )
        model_config = ctx.model_config

    return MULTIMODAL_REGISTRY.create_processor(model_config)


def _assert_image_outputs(processor, processed, num_images: int) -> None:
    mm_data = processed["mm_kwargs"].get_data()
    grid_thw = mm_data["image_grid_thw"]
    pixel_values = mm_data["pixel_values"]

    assert grid_thw.shape == (num_images, 3)
    assert pixel_values.shape[0] == int(grid_thw.prod(dim=-1).sum())

    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size
    expected_tokens = (grid_thw.prod(dim=-1) // merge_size**2).tolist()
    image_placeholders = processed["mm_placeholders"]["image"]

    assert len(image_placeholders) == num_images
    assert [placeholder.length for placeholder in image_placeholders] == (
        expected_tokens
    )

    image_token_id = processor.info.get_hf_processor().image_token_id
    assert processed["prompt_token_ids"].count(image_token_id) == sum(expected_tokens)


def _assert_video_outputs(processor, processed) -> None:
    mm_data = processed["mm_kwargs"].get_data()
    grid_thw = mm_data["video_grid_thw"]
    pixel_values = mm_data["pixel_values_videos"]

    assert grid_thw.shape == (1, 3)
    assert pixel_values.shape[0] == int(grid_thw.prod())
    assert len(processed["mm_placeholders"]["video"]) == 1

    merge_size = processor.info.get_hf_config().vision_config.spatial_merge_size
    expected_tokens = int(grid_thw.prod()) // merge_size**2
    video_token_id = processor.info.get_hf_config().video_token_id
    prompt_token_ids = processed["prompt_token_ids"]
    assert prompt_token_ids.count(video_token_id) == expected_tokens

    hf_processor = processor.info.get_hf_processor()
    expected_frame_wrappers = int(grid_thw[:, 0].sum())
    assert (
        prompt_token_ids.count(hf_processor.vision_start_token_id)
        == expected_frame_wrappers
    )
    assert (
        prompt_token_ids.count(hf_processor.vision_end_token_id)
        == expected_frame_wrappers
    )


def test_device_normalization_is_disabled() -> None:
    ctx = build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )

    assert ctx.model_config.multimodal_config.mm_device_do_normalize is False


def test_gpu_video_resize_target_resolves_live_processor(processor) -> None:
    target = processor_video_resize_target(
        processor,
        {"max_pixels": 32 * 1024 * 576},
    )

    assert target is not None
    assert target(3840, 2160, 32) == (1024, 576)


@pytest.mark.parametrize("num_images", [1, 2])
def test_process_images(
    processor,
    image_assets: ImageTestAssets,
    num_images: int,
) -> None:
    images = [asset.pil_image for asset in image_assets[:num_images]]
    processed = processor(
        IMAGE_PLACEHOLDER * num_images,
        mm_items=processor.info.parse_mm_data({"image": images}),
        hf_processor_mm_kwargs={},
    )

    _assert_image_outputs(processor, processed, num_images)


def test_process_video(processor) -> None:
    video_asset = VideoAsset(name="baby_reading", num_frames=8)
    video = (video_asset.np_ndarrays, video_asset.metadata)
    processed = processor(
        VIDEO_PLACEHOLDER,
        mm_items=processor.info.parse_mm_data({"video": [video]}),
        hf_processor_mm_kwargs={},
    )

    _assert_video_outputs(processor, processed)


def test_tchw_video_matches_thwc(processor) -> None:
    num_frames, height, width = 4, 65, 97
    thwc = np.arange(num_frames * height * width * 3, dtype=np.uint8).reshape(
        num_frames, height, width, 3
    )
    tchw = np.ascontiguousarray(thwc.transpose(0, 3, 1, 2))
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "pynvvideocodec",
        "do_sample_frames": False,
    }

    def process(video, video_metadata):
        return processor(
            VIDEO_PLACEHOLDER,
            mm_items=processor.info.parse_mm_data({"video": [(video, video_metadata)]}),
            hf_processor_mm_kwargs={},
        )

    baseline = process(thwc, metadata)
    candidate = process(
        tchw,
        {
            **metadata,
            VLLM_VIDEO_INPUT_DATA_FORMAT_KEY: "channels_first",
        },
    )

    assert candidate.keys() == baseline.keys()
    assert candidate["type"] == baseline["type"]
    assert candidate["prompt_token_ids"] == baseline["prompt_token_ids"]
    baseline_mm = baseline["mm_kwargs"].get_data()
    candidate_mm = candidate["mm_kwargs"].get_data()
    assert candidate_mm.keys() == baseline_mm.keys()
    assert torch.equal(
        candidate_mm["pixel_values_videos"], baseline_mm["pixel_values_videos"]
    )
    assert torch.equal(candidate_mm["video_grid_thw"], baseline_mm["video_grid_thw"])
    assert candidate_mm["timestamps"] == baseline_mm["timestamps"]

    baseline_phs = baseline["mm_placeholders"]["video"]
    candidate_phs = candidate["mm_placeholders"]["video"]
    assert len(candidate_phs) == len(baseline_phs)
    for candidate_ph, baseline_ph in zip(candidate_phs, baseline_phs):
        assert candidate_ph.offset == baseline_ph.offset
        assert candidate_ph.length == baseline_ph.length
        assert torch.equal(candidate_ph.is_embed, baseline_ph.is_embed)

    assert candidate["mm_hashes"]["video"] != baseline["mm_hashes"]["video"]
