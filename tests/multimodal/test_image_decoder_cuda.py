# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecInput,
    _PinnedImageLease,
    create_nvimagecodec_decode_service,
)
from vllm.multimodal.media import ImageMediaIO

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _jpeg(width: int = 193, height: int = 97) -> bytes:
    y, x = np.mgrid[:height, :width]
    pixels = np.stack((x * 3, y * 5, x + y * 2), axis=-1).astype(np.uint8)
    with BytesIO() as buffer:
        Image.fromarray(pixels).save(buffer, "JPEG", quality=95)
        return buffer.getvalue()


def test_real_jpeg_decode_is_pinned_chw_and_matches_pillow() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    service = create_nvimagecodec_decode_service()
    try:
        service.wait_until_ready()
        data = _jpeg()
        image_io = ImageMediaIO(backend="nvimagecodec", _borrow_output=True)
        prepared = image_io._prepare_bytes(data)
        assert isinstance(prepared, NvImageCodecInput)

        result = service.submit(prepared).result()
        assert isinstance(result, _PinnedImageLease)
        host = result.borrow_tensor()
        expected = np.moveaxis(np.asarray(ImageMediaIO().load_bytes(data).media), -1, 0)
        assert host.is_pinned() and tuple(host.shape) == expected.shape
        np.testing.assert_allclose(host.numpy(), expected, rtol=0, atol=6)

        result.release()
        with pytest.raises(RuntimeError, match="expired"):
            result.borrow_tensor()
    finally:
        service.close()
