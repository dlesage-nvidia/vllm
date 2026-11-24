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


@pytest.mark.parametrize("batch_cap", [4, 20])
def test_mixed_shape_waves_preserve_retained_outputs(batch_cap: int) -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    shapes = ((193, 97), (385, 193), (97, 49), (513, 257), (1920, 1080), (3840, 2160))
    encoded = [
        _jpeg(width, height, index) for index, (width, height) in enumerate(shapes)
    ]
    expected = [
        np.moveaxis(np.asarray(ImageMediaIO().load_bytes(data).media), -1, 0)
        for data in encoded
    ]
    widths = (batch_cap, batch_cap, batch_cap // 2, batch_cap - 1, batch_cap, batch_cap)
    decoder = _NvImageCodecDecoder(batch_cap=batch_cap)
    retained = []
    try:
        for first_wave in range(0, len(widths), 2):
            pending = []
            for wave in (first_wave, first_wave + 1):
                indices = [
                    (wave + index) % len(shapes) for index in range(widths[wave])
                ]
                items = tuple(
                    preflight_image_nvimagecodec(encoded[index]) for index in indices
                )
                pending.append((decoder.submit(items), indices))
            for arena, indices in pending:
                for host, index in zip(decoder.collect(arena), indices, strict=True):
                    assert host is not None and host.is_pinned()
                    assert tuple(host.shape) == expected[index].shape
                    np.testing.assert_allclose(
                        host.numpy(), expected[index], rtol=0, atol=6
                    )
                    if first_wave == 0:
                        retained.append((host, host.clone()))
                for host, snapshot in retained:
                    assert torch.equal(host, snapshot)
    finally:
        decoder.close()
    for host, snapshot in retained:
        assert torch.equal(host, snapshot)
