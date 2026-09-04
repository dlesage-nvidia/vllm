# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.media import ImageMediaIO
from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="Requires NVIDIA CUDA"
)


def test_nvimagecodec_decodes_owned_rgb_images() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    image_io = ImageMediaIO(image_backend="nvimagecodec")
    encoded = []
    references = []

    for width, height in ((193, 97), (384, 216)):
        y, x = np.mgrid[:height, :width]
        pixels = np.stack((x * 3, y * 5, x + y * 2), axis=-1).astype(np.uint8)
        buffer = BytesIO()
        Image.fromarray(pixels).save(buffer, "JPEG", quality=95)
        data = buffer.getvalue()
        encoded.append(data)
        references.append(np.asarray(ImageMediaIO().load_bytes(data).media))

    first_image = image_io.load_bytes(encoded[0]).media
    first_pixels = np.asarray(first_image).copy()
    with ThreadPoolExecutor(max_workers=8) as clients:
        results = list(clients.map(image_io.load_bytes, encoded * 4))

    for index, native in enumerate(results):
        assert isinstance(native.media, Image.Image)
        assert native.io_config == {"image_backend": "nvimagecodec"}
        np.testing.assert_allclose(
            np.asarray(native.media), references[index % len(encoded)], rtol=0, atol=6
        )

    np.testing.assert_array_equal(np.asarray(first_image), first_pixels)
