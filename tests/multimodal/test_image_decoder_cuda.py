# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

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
