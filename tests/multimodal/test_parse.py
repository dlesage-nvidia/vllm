# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.media import MediaWithBytes
from vllm.multimodal.parse import (
    ImageProcessorItems,
    MultiModalDataParser,
    VideoProcessorItems,
)

H, W = 480, 640


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (W, H)),
        # HWC, e.g. from np.array(PIL.Image)
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        # CHW, standard PyTorch / numpy convention
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_image_size_hwc_chw(image):
    """Image sizes must be channel-layout agnostic.

    `get_image_size` determines the multimodal placeholder count; reading an
    HWC array (the layout `np.array(PIL.Image)` produces) as CHW yields a
    bogus size and a placeholder/embedding count mismatch at inference time.
    """
    items = ImageProcessorItems([image])

    assert items.get_image_size(0) == (W, H)


@pytest.mark.parametrize("width", [1, 3, 4])
def test_tagged_chw_image_preserves_layout_through_size_and_reparse(width):
    chw = np.zeros((3, H, width), dtype=np.uint8)
    config = {"backend": "nvimagecodec", "output_layout": "CHW"}
    wrapped = MediaWithBytes(chw, b"encoded", config)
    items = ImageProcessorItems([wrapped])

    assert items.get_image_size(0) == (width, H)
    assert items.get_processor_data()["images"][0] is chw
    assert items.get_item_for_reparse(0) is wrapped


@pytest.mark.parametrize(
    "frame",
    [
        Image.new("RGB", (W, H)),
        np.zeros((H, W, 3), dtype=np.uint8),
        torch.zeros((H, W, 3), dtype=torch.uint8),
        np.zeros((3, H, W), dtype=np.uint8),
        torch.zeros((3, H, W), dtype=torch.uint8),
    ],
)
def test_frame_size_hwc_chw(frame):
    """`get_frame_size` must stay consistent with `get_image_size`."""
    items = VideoProcessorItems([[frame]])

    assert items.get_frame_size(0) == (W, H)


@pytest.mark.parametrize(
    "modality,processor_cls",
    [
        ("image", ImageProcessorItems),
        ("video", VideoProcessorItems),
    ],
)
def test_parse_mm_data_accepts_none_cached_item(modality, processor_cls):
    mm_items = MultiModalDataParser().parse_mm_data({modality: [None]})
    items = mm_items[modality]
    assert isinstance(items, processor_cls)
    assert len(items) == 1
    assert items.get(0) is None
