# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.image_decoders.nvimagecodec import (
    _NvImageCodecDecoder,
    preflight_image_nvimagecodec,
)
from vllm.multimodal.media import ImageMediaIO
from vllm.multimodal.media.image import initialize_nvimagecodec_decode_service

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _jpeg(width: int = 193, height: int = 97, offset: int = 0) -> bytes:
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x * 3, y * 5, x + y * 2), axis=-1).astype(np.uint8)
    pixels[..., 0] += offset
    with BytesIO() as buffer:
        Image.fromarray(pixels).save(buffer, "JPEG", quality=95)
        return buffer.getvalue()


@pytest.mark.asyncio
async def test_selected_backend_returns_owned_pinned_chw() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    release = initialize_nvimagecodec_decode_service()
    try:
        data = _jpeg()
        result = await ImageMediaIO(backend="nvimagecodec").load_bytes_async(data)
        host = result.media
        expected = np.moveaxis(np.asarray(ImageMediaIO().load_bytes(data).media), -1, 0)
        assert host.is_pinned() and tuple(host.shape) == expected.shape
        np.testing.assert_allclose(host.numpy(), expected, rtol=0, atol=6)
        snapshot = host.clone()
        for offset in (1, 2):
            await ImageMediaIO(backend="nvimagecodec").load_bytes_async(
                _jpeg(offset=offset)
            )
        assert torch.equal(host, snapshot)
    finally:
        release()


def test_device_images_resize_without_invalidating_owned_outputs() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    decoder = _NvImageCodecDecoder(batch_cap=1)
    images = []
    retained = []
    try:
        for index, (width, height) in enumerate(
            ((193, 97), (385, 193), (97, 49), (513, 257))
        ):
            item = preflight_image_nvimagecodec(_jpeg(width, height, index))
            arena = decoder.submit((item,))
            if index < 2:
                images.append(arena.images[0])
            else:
                assert arena.images[0] is images[index % 2]
            (host,) = decoder.collect(arena)
            assert host is not None
            assert host.is_pinned() and tuple(host.shape) == (3, height, width)
            retained.append((host, host.clone()))
        for host, snapshot in retained:
            assert torch.equal(host, snapshot)
    finally:
        decoder.close()
