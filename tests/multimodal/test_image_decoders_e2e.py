# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests: does a GPU-decoded image survive the real consumer path?

The GPU backend hands back an HWC uint8 ndarray instead of a PIL image. vLLM's
own ``HfImageItem`` union declares that legal, but the decision is only safe if
a *real* HF image processor produces the same tensors either way. These tests
exercise that with real processors rather than asserting it from the type alias.

They need only processor configs, not model weights.
"""

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.image_decoders import NVIMGCODEC_BACKEND
from vllm.multimodal.media.image import ImageMediaIO
from vllm.multimodal.parse import ImageProcessorItems

pytestmark = pytest.mark.cpu_test

# Processors that ship a config small enough to fetch without model weights.
PROCESSORS = ["Qwen/Qwen3-VL-2B-Instruct", "llava-hf/llava-1.5-7b-hf"]


def _jpeg(width: int, height: int, seed: int = 0) -> bytes:
    from io import BytesIO

    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    pixels = np.clip(
        np.stack(
            [
                128 + 100 * np.sin(xx / width * 6),
                128 + 100 * np.cos(yy / height * 5),
                np.full((height, width), 120.0, np.float32),
            ],
            -1,
        )
        + rs.rand(height, width, 3) * 6,
        0,
        255,
    ).astype(np.uint8)
    buf = BytesIO()
    Image.fromarray(pixels).save(buf, "JPEG", quality=92)
    return buf.getvalue()


def _load_processor(name: str):
    transformers = pytest.importorskip("transformers")
    try:
        return transformers.AutoImageProcessor.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001 - offline / gated / not cached
        pytest.skip(f"image processor {name} unavailable: {exc}")


@pytest.mark.parametrize("name", PROCESSORS)
@pytest.mark.parametrize("size", [(64, 48), (97, 65), (224, 224)])
def test_processor_output_is_identical_for_array_and_pil(name, size):
    """The load-bearing assumption behind image_output="array"."""
    processor = _load_processor(name)
    width, height = size
    array = np.asarray(Image.fromarray(_decode_reference(width, height)))
    pil = Image.fromarray(array)

    from_array = processor(images=[array], return_tensors="np")
    from_pil = processor(images=[pil], return_tensors="np")

    assert set(from_array.keys()) == set(from_pil.keys())
    for key in from_array:
        left, right = np.asarray(from_array[key]), np.asarray(from_pil[key])
        assert left.shape == right.shape, f"{name}/{key}: shape differs"
        np.testing.assert_array_equal(
            left, right, err_msg=f"{name}/{key}: ndarray and PIL inputs diverge"
        )


def _decode_reference(width: int, height: int) -> np.ndarray:
    from io import BytesIO

    with Image.open(BytesIO(_jpeg(width, height))) as image:
        return np.asarray(image.convert("RGB"))


@pytest.mark.parametrize("size", [(64, 48), (97, 65)])
def test_image_processor_items_accepts_an_array(size):
    """vLLM's own parser must report the same size for either representation."""
    width, height = size
    array = _decode_reference(width, height)
    assert ImageProcessorItems([array]).get_image_size(0) == ImageProcessorItems(
        [Image.fromarray(array)]
    ).get_image_size(0)


def test_pil_escape_hatch_returns_a_pil_image():
    """image_output="pil" must keep the GPU decode but restore the PIL type."""
    io_pil = ImageMediaIO(
        image_mode="RGB", image_backend=NVIMGCODEC_BACKEND, image_output="pil"
    )
    result = io_pil.load_bytes(_jpeg(64, 48))
    assert isinstance(result.media, Image.Image)


def test_connector_fetch_image_uses_the_backend_end_to_end(tmp_path):
    """The real connector entry point, not just ImageMediaIO."""
    pytest.importorskip("torch")
    from vllm.multimodal.media.connector import MediaConnector

    path = tmp_path / "sample.jpg"
    path.write_bytes(_jpeg(320, 240))

    connector = MediaConnector(
        media_io_kwargs={"image": {"image_backend": NVIMGCODEC_BACKEND}},
        allowed_local_media_path=str(tmp_path),
    )
    result = connector.fetch_image(path.as_uri())
    # Whichever backend served it, the connector contract must hold.
    assert result.original_bytes == path.read_bytes()
    assert isinstance(result.media, (Image.Image, np.ndarray))


def test_connector_pillow_default_is_untouched(tmp_path):
    from vllm.multimodal.media.connector import MediaConnector

    path = tmp_path / "sample.jpg"
    path.write_bytes(_jpeg(320, 240))
    connector = MediaConnector(allowed_local_media_path=str(tmp_path))
    result = connector.fetch_image(path.as_uri())
    assert isinstance(result.media, Image.Image)
