# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

import vllm.envs as envs
import vllm.multimodal.image_decoders.nvimagecodec as nvcodec

pytestmark = pytest.mark.cpu_test


def _jpeg(*, exif: bool = False) -> bytes:
    kwargs = {}
    if exif:
        metadata = Image.Exif()
        metadata[Image.ExifTags.Base.Orientation] = 6
        kwargs["exif"] = metadata
    with BytesIO() as buffer:
        Image.new("RGB", (8, 4)).save(buffer, "JPEG", **kwargs)
        return buffer.getvalue()


def _install_codec(monkeypatch, *, error=None, **metadata):
    stream = SimpleNamespace(
        **{
            "codec_name": "jpeg",
            "width": 8,
            "height": 4,
            "precision": 8,
            "num_channels": 3,
            **metadata,
        }
    )

    def code_stream(_data):
        if error is not None:
            raise error
        return stream

    monkeypatch.setattr(
        nvcodec, "_load_nvimgcodec", lambda: SimpleNamespace(CodeStream=code_stream)
    )


@pytest.mark.parametrize(
    ("data", "image_mode", "metadata", "error"),
    [
        (b"not jpeg", "RGB", {}, "JPEG images only"),
        (_jpeg()[:-2], "RGB", {}, "complete JPEG"),
        (_jpeg(), None, {}, "requires image_mode='RGB'"),
        (_jpeg(exif=True), "RGB", {}, "does not support EXIF"),
        (_jpeg(), "RGB", {"precision": 12}, "8-bit, three-channel"),
        (_jpeg(), "RGB", {"num_channels": 1}, "8-bit, three-channel"),
        (_jpeg(), "RGB", {"width": 4}, "width of at least 5"),
        (_jpeg(), "RGB", {"width": 3841, "height": 2160}, "8,294,400"),
    ],
)
def test_preflight_rejects_unsupported_inputs(
    monkeypatch, data, image_mode, metadata, error
):
    _install_codec(monkeypatch, **metadata)
    with pytest.raises(ValueError, match=error):
        nvcodec.preflight_image_nvimagecodec(data, image_mode=image_mode)


def test_preflight_rejects_native_parser_failure_and_pixel_limit(monkeypatch):
    _install_codec(monkeypatch, error=RuntimeError("bad JPEG"))
    with pytest.raises(ValueError, match="could not parse"):
        nvcodec.preflight_image_nvimagecodec(_jpeg())

    _install_codec(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 31)
    with pytest.raises(ValueError, match="maximum of 31 pixels"):
        nvcodec.preflight_image_nvimagecodec(_jpeg())
