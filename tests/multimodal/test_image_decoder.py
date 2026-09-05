# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

import vllm.envs as envs
import vllm.multimodal.image_decoders.nvimagecodec as nvcodec

pytestmark = pytest.mark.cpu_test


def _jpeg(*, exif: bool = False, progressive: bool = False) -> bytes:
    image = Image.new("RGB", (8, 4))
    kwargs = {"progressive": progressive}
    if exif:
        metadata = Image.Exif()
        metadata[Image.ExifTags.Base.Orientation] = 6
        kwargs["exif"] = metadata
    with BytesIO() as buffer:
        image.save(buffer, "JPEG", **kwargs)
        return buffer.getvalue()


def _install_codec(monkeypatch, *, error=None, **metadata):
    stream = SimpleNamespace(
        **dict(
            codec_name="jpeg",
            width=8,
            height=4,
            precision=8,
            num_channels=3,
        )
        | metadata
    )

    def code_stream(_data):
        if error is not None:
            raise error
        return stream

    monkeypatch.setattr(
        nvcodec, "_load_nvimgcodec", lambda: SimpleNamespace(CodeStream=code_stream)
    )


def test_package_availability_without_import(monkeypatch):
    monkeypatch.setattr(nvcodec, "_load_nvimgcodec", lambda: pytest.fail("imported"))
    monkeypatch.setattr(nvcodec.importlib.util, "find_spec", lambda _name: object())
    nvcodec.ensure_nvimagecodec_available()

    monkeypatch.setattr(nvcodec.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="requires.*nvidia-nvimgcodec"):
        nvcodec.ensure_nvimagecodec_available()


def test_rejects_non_jpeg_and_non_rgb_mode(monkeypatch):
    monkeypatch.setattr(nvcodec, "_load_nvimgcodec", lambda: pytest.fail("loaded"))
    with pytest.raises(ValueError, match="only JPEG"):
        nvcodec.preflight_image_nvimagecodec(b"not an image")
    with pytest.raises(ValueError, match="image_mode='RGB'"):
        nvcodec.preflight_image_nvimagecodec(_jpeg(), image_mode=None)


@pytest.mark.parametrize(
    "metadata",
    [
        {"precision": 12},
        {"num_channels": 1},
        {"width": 0},
    ],
)
def test_unsupported_jpeg_metadata_fails(monkeypatch, metadata):
    _install_codec(monkeypatch, **metadata)
    with pytest.raises(ValueError, match="requires 8-bit|invalid JPEG dimensions"):
        nvcodec.preflight_image_nvimagecodec(_jpeg())


def test_exif_and_native_parser_failure_fail(monkeypatch):
    _install_codec(monkeypatch)
    with pytest.raises(ValueError, match="EXIF"):
        nvcodec.preflight_image_nvimagecodec(_jpeg(exif=True))

    _install_codec(monkeypatch, error=RuntimeError("unsupported JPEG"))
    with pytest.raises(ValueError, match="rejected the JPEG metadata"):
        nvcodec.preflight_image_nvimagecodec(_jpeg())


def test_exif_signature_outside_app1_is_allowed(monkeypatch):
    _install_codec(monkeypatch)
    data = _jpeg()
    comment = b"not metadata: Exif\x00\x00"
    data = (
        data[:2]
        + b"\xff\xfe"
        + (len(comment) + 2).to_bytes(2, "big")
        + comment
        + data[2:]
    )

    result = nvcodec.preflight_image_nvimagecodec(data)
    assert (result.width, result.height) == (8, 4)


def test_incomplete_jpeg_fails_before_native_parser(monkeypatch):
    monkeypatch.setattr(nvcodec, "_load_nvimgcodec", lambda: pytest.fail("loaded"))
    for progressive in (False, True):
        data = _jpeg(progressive=progressive)
        with pytest.raises(ValueError, match="structurally complete"):
            nvcodec.preflight_image_nvimagecodec(data[:-2])


@pytest.mark.parametrize(
    ("width", "height", "limit", "outcome"),
    [
        (8, 4, 32, None),
        (8, 4, 31, "maximum of 31 pixels"),
        (4, 4, 0, "width greater than 4"),
        (3840, 2160, 0, None),
        (3841, 2160, 0, "limited to 8,294,400 pixels"),
    ],
)
def test_pixel_boundaries(monkeypatch, width, height, limit, outcome):
    _install_codec(monkeypatch, width=width, height=height)
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", limit)
    if outcome:
        with pytest.raises(ValueError, match=outcome):
            nvcodec.preflight_image_nvimagecodec(_jpeg())
    else:
        result = nvcodec.preflight_image_nvimagecodec(_jpeg())
        assert (result.width, result.height) == (width, height)
