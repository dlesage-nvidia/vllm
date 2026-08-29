# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression tests for Qwen3-VL processing and vision inputs."""

from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn

from vllm.model_executor.models.qwen3_vl import (
    Qwen3_VisionTransformer,
    Qwen3VLForConditionalGeneration,
)
from vllm.model_executor.models.vision import FusedInputNorm
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.video import VLLM_VIDEO_INPUT_DATA_FORMAT_KEY
from vllm.multimodal.video_decoders import processor_video_resize_target

from ....conftest import ImageTestAssets
from ...utils import build_model_context

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"
MOE_MODEL_ID = "Qwen/Qwen3-VL-30B-A3B-Instruct"
QWEN3_5_MODEL_IDS = ("Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-35B-A3B")


def _build_video_mm_data(
    num_frames: int,
    width: int = 128,
    height: int = 128,
    original_fps: float = 30.0,
) -> dict[str, Any]:
    """Create synthetic video data with metadata indicating that
    HF processor should re-sample frames (do_sample_frames=True).

    ``total_num_frames`` is set equal to the ndarray frame count so
    that HF's ``sample_frames`` indices stay within bounds of the
    actual tensor that is passed."""
    frame, row, column = np.indices((num_frames, height, width), dtype=np.uint16)
    video = np.stack(
        (
            (17 * frame + 3 * row + column) % 256,
            (29 * frame + row + 5 * column + 41) % 256,
            (7 * frame + 11 * row + 2 * column + 137) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    metadata = {
        "fps": original_fps,
        "duration": num_frames / original_fps,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "opencv",
        "do_sample_frames": True,
    }
    return {"video": [(video, metadata)]}


def test_gpu_video_resize_target_resolves_live_processor() -> None:
    max_pixels = 32 * 1024 * 576
    ctx = build_model_context(
        MODEL_ID,
        mm_processor_kwargs={"max_pixels": max_pixels},
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    target = processor_video_resize_target(
        processor,
        ctx.model_config.multimodal_config.mm_processor_kwargs,
    )

    assert target is not None
    assert target(3840, 2160, 32) == (1024, 576)


@pytest.mark.parametrize("model_id", [MODEL_ID, MOE_MODEL_ID])
@pytest.mark.parametrize("modality", ["image", "video"])
def test_mm_device_do_normalize(
    image_assets: ImageTestAssets,
    model_id: str,
    modality: str,
) -> None:
    limits = {"image": int(modality == "image"), "video": int(modality == "video")}
    ctx = build_model_context(model_id, limit_mm_per_prompt=limits)
    assert ctx.model_config.multimodal_config.mm_device_do_normalize is True
    ctx.model_config.multimodal_config.mm_device_do_normalize = False
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    if modality == "image":
        prompt = "<|vision_start|><|image_pad|><|vision_end|>"
        mm_data = {"image": [image_assets[0].pil_image]}
        pixel_key = "pixel_values"
        hf_mm_kwargs: dict[str, Any] = {}
    else:
        prompt = "<|vision_start|><|video_pad|><|vision_end|>"
        mm_data = _build_video_mm_data(num_frames=8)
        pixel_key = "pixel_values_videos"
        hf_mm_kwargs = {"num_frames": 8}

    normalized = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )["mm_kwargs"].get_data()[pixel_key]
    raw = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs
        | {"do_normalize": False, "do_rescale": False},
    )["mm_kwargs"].get_data()[pixel_key]

    assert raw.dtype == torch.uint8
    ctx.model_config.multimodal_config.mm_device_do_normalize = True
    input_norm = FusedInputNorm.from_model_config(ctx.model_config)
    device_normalized = input_norm(raw, normalized.dtype)

    assert device_normalized.dtype == normalized.dtype
    torch.testing.assert_close(normalized, device_normalized)


@pytest.mark.parametrize("model_id", QWEN3_5_MODEL_IDS)
def test_qwen3_5_does_not_enable_device_normalization(model_id: str) -> None:
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 1, "video": 1},
    )

    assert ctx.model_config.multimodal_config.mm_device_do_normalize is False


def test_vision_forward_normalizes_before_patch_embed() -> None:
    observed_dtypes: list[torch.dtype] = []

    class RecordingNorm(nn.Module):
        def forward(self, inputs: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
            observed_dtypes.append(inputs.dtype)
            return inputs.to(dtype)

    class RecordingPatchEmbed(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(1, 1, bias=False)

        def forward(self, inputs: torch.Tensor) -> torch.Tensor:
            observed_dtypes.append(inputs.dtype)
            return inputs[:, :1]

    visual = Qwen3_VisionTransformer.__new__(Qwen3_VisionTransformer)
    nn.Module.__init__(visual)
    visual.input_norm = RecordingNorm()
    visual.patch_embed = RecordingPatchEmbed()
    visual.blocks = nn.ModuleList()
    visual.deepstack_visual_indexes = []
    visual.deepstack_merger_list = nn.ModuleList()
    visual.merger = nn.Identity()

    raw = torch.arange(8, dtype=torch.uint8).reshape(2, 4)
    output = visual(
        raw,
        [[1, 1, 2]],
        encoder_metadata={"pos_embeds": torch.zeros(2, 1)},
    )

    assert observed_dtypes == [torch.uint8, torch.float32]
    assert output.shape == (2, 1, 1)


@pytest.mark.parametrize(
    ("device_normalize", "expected_dtype"),
    [(False, torch.bfloat16), (True, torch.uint8)],
)
def test_encoder_cudagraph_capture_pixel_dtype(
    device_normalize: bool,
    expected_dtype: torch.dtype,
) -> None:
    class FakeVisual:
        spatial_merge_size = 2
        patch_embed = SimpleNamespace(
            proj=SimpleNamespace(in_channels=3),
            patch_size=14,
            temporal_patch_size=2,
        )

        @staticmethod
        def prepare_encoder_metadata(*args, **kwargs) -> dict:
            return {}

    model = SimpleNamespace(
        visual=FakeVisual(),
        multimodal_config=SimpleNamespace(
            mm_device_do_normalize=device_normalize,
        ),
    )
    capture_inputs = (
        Qwen3VLForConditionalGeneration.prepare_encoder_cudagraph_capture_inputs(
            model,
            token_budget=64,
            max_batch_size=1,
            max_frames_per_batch=1,
            device=torch.device("cpu"),
            dtype=torch.bfloat16,
        )
    )

    assert capture_inputs.values["pixel_values"].dtype == expected_dtype


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "num_frames",
    [8, 16],
)
def test_processor_num_frames_timestamp(
    model_id: str,
    num_frames: int,
) -> None:
    """Regression test: using ``num_frames`` (without ``fps``) must not
    cause a timestamp / token-count mismatch.

    Before the fix, ``_get_video_second_idx`` ignored the explicit
    ``num_frames`` and fell back to an fps-based calculation, which
    produced a different number of timestamp entries and ultimately led
    to shape mismatches in downstream token construction.

    We deliberately choose ``num_frames`` values (8, 16) that differ
    from what the default fps-based path would compute (which clamps
    to ``min_frames=4`` for a short video at 30 fps), so this test
    would fail without the fix.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>"
    mm_data = _build_video_mm_data(num_frames=num_frames)

    # Process with explicit num_frames (no fps) -- this is the path
    # that was broken before the fix.
    hf_mm_kwargs: dict[str, Any] = {"num_frames": num_frames}
    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )

    # Basic sanity: the processor must produce video tokens.
    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0, "Processor produced empty token list"

    # Verify that video placeholders were actually inserted.
    assert "mm_placeholders" in processed
    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 1, (
        f"Expected exactly 1 video placeholder, got {len(video_phs)}"
    )


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize("num_videos", [2, 4])
def test_processor_multi_video(
    model_id: str,
    num_videos: int,
) -> None:
    """Verify that multi-video processing produces correct placeholders.

    This exercises the token-level replacement path in
    ``_apply_hf_processor_main`` which avoids the quadratic text-level
    prompt expansion.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": num_videos},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = "<|vision_start|><|video_pad|><|vision_end|>" * num_videos
    mm_data = {"video": [_build_video_mm_data(num_frames=8)["video"][0]] * num_videos}

    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs={"num_frames": 8},
    )

    token_ids = processed["prompt_token_ids"]
    assert len(token_ids) > 0

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == num_videos, (
        f"Expected {num_videos} video placeholders, got {len(video_phs)}"
    )

    # All placeholders should have the same length (same video params)
    # and must not overlap.
    lengths = {ph.length for ph in video_phs}
    assert len(lengths) == 1, f"Placeholder lengths differ: {lengths}"
    for i in range(1, len(video_phs)):
        prev_end = video_phs[i - 1].offset + video_phs[i - 1].length
        assert video_phs[i].offset >= prev_end, (
            f"Placeholder {i} overlaps with placeholder {i - 1}"
        )


@pytest.mark.parametrize("model_id", [MODEL_ID])
@pytest.mark.parametrize(
    "hf_mm_kwargs",
    [{"num_frames": [8, 16]}, {"fps": [2.0, 4.0]}],
)
def test_processor_multi_video_list_kwargs(
    model_id: str,
    hf_mm_kwargs: dict[str, Any],
) -> None:
    """Regression test: a multi-video request with list-valued per-video
    ``mm_processor_kwargs`` (one ``fps``/``num_frames`` per video) must not
    crash.

    Before the fix, ``_apply_hf_processor_main`` copied the whole kwargs to every
    video without slicing, so ``_get_video_second_idx`` received the list
    where a scalar was expected and raised ``TypeError``.
    """
    ctx = build_model_context(
        model_id,
        limit_mm_per_prompt={"image": 0, "video": 2},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    prompt = (
        "<|vision_start|><|video_pad|><|vision_end|>"
        "<|vision_start|><|video_pad|><|vision_end|>"
    )
    mm_data = {
        "video": [
            _build_video_mm_data(num_frames=16)["video"][0],
            _build_video_mm_data(num_frames=32)["video"][0],
        ]
    }

    processed = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(mm_data),
        hf_processor_mm_kwargs=hf_mm_kwargs,
    )

    video_phs = processed["mm_placeholders"].get("video", [])
    assert len(video_phs) == 2, (
        f"Expected exactly 2 video placeholders, got {len(video_phs)}"
    )


def test_processor_tchw_matches_thwc() -> None:
    ctx = build_model_context(
        MODEL_ID,
        limit_mm_per_prompt={"image": 0, "video": 1},
    )
    processor = MULTIMODAL_REGISTRY.create_processor(ctx.model_config)

    num_frames, height, width = 8, 128, 160
    thwc = np.arange(num_frames * height * width * 3, dtype=np.uint8).reshape(
        num_frames, height, width, 3
    )
    metadata = {
        "fps": 2.0,
        "duration": num_frames / 2.0,
        "total_num_frames": num_frames,
        "frames_indices": list(range(num_frames)),
        "video_backend": "pynvvideocodec",
        "do_sample_frames": False,
    }
    prompt = "<|vision_start|><|video_pad|><|vision_end|>"

    baseline = processor(
        prompt,
        mm_items=processor.info.parse_mm_data({"video": [(thwc, metadata)]}),
        hf_processor_mm_kwargs={},
    )
    tchw_metadata = {
        **metadata,
        VLLM_VIDEO_INPUT_DATA_FORMAT_KEY: "channels_first",
    }
    candidate = processor(
        prompt,
        mm_items=processor.info.parse_mm_data(
            {"video": [(thwc.transpose(0, 3, 1, 2).copy(), tchw_metadata)]}
        ),
        hf_processor_mm_kwargs={},
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

    # Raw THWC and TCHW representations intentionally have distinct cache
    # hashes even though every model-visible processor output is identical.
    assert candidate["mm_hashes"]["video"] != baseline["mm_hashes"]["video"]
