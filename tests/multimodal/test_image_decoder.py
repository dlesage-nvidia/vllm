# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from io import BytesIO
from types import SimpleNamespace

import pytest
from PIL import Image

import vllm.envs as envs
import vllm.multimodal.image_decoders.nvimagecodec as nvcodec

pytestmark = pytest.mark.cpu_test


def _jpeg(mode: str = "RGB", *, exif: bool = False, progressive: bool = False) -> bytes:
    image = Image.new(mode, (8, 4))
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


def test_admits_only_simple_rgb_jpeg(monkeypatch):
    _install_codec(monkeypatch)
    result = nvcodec.preflight_image_nvimagecodec(_jpeg())
    assert result is not None
    assert (result.width, result.height) == (8, 4)

    monkeypatch.setattr(nvcodec, "_load_nvimgcodec", lambda: pytest.fail("loaded"))
    assert nvcodec.preflight_image_nvimagecodec(b"not a jpeg") is None
    assert nvcodec.preflight_image_nvimagecodec(_jpeg(), image_mode=None) is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"precision": 12},
        {"num_channels": 1},
        {"width": 0},
    ],
)
def test_unsupported_jpeg_metadata_falls_back(monkeypatch, metadata):
    _install_codec(monkeypatch, **metadata)
    assert nvcodec.preflight_image_nvimagecodec(_jpeg()) is None


def test_exif_and_native_parser_failure_fall_back(monkeypatch):
    _install_codec(monkeypatch)
    assert nvcodec.preflight_image_nvimagecodec(_jpeg(exif=True)) is None

    _install_codec(monkeypatch, error=RuntimeError("unsupported JPEG"))
    assert nvcodec.preflight_image_nvimagecodec(_jpeg()) is None


def test_jpeg_structural_validation(monkeypatch):
    for progressive in (False, True):
        data = _jpeg(progressive=progressive)
        assert nvcodec.jpeg_has_complete_scan_and_eoi(data)
        assert not nvcodec.jpeg_has_complete_scan_and_eoi(data[:-2])

    _install_codec(monkeypatch)
    assert nvcodec.preflight_image_nvimagecodec(data[:-2]) is None


@pytest.mark.parametrize(
    ("width", "height", "limit", "outcome"),
    [
        (8, 4, 32, None),
        (8, 4, 31, "maximum of 31 pixels"),
        (3840, 2160, 0, None),
        (3841, 2160, 0, "fallback"),
    ],
)
def test_pixel_boundaries(monkeypatch, width, height, limit, outcome):
    _install_codec(monkeypatch, width=width, height=height)
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", limit)
    if outcome and outcome != "fallback":
        with pytest.raises(ValueError, match=outcome):
            nvcodec.preflight_image_nvimagecodec(_jpeg())
    else:
        result = nvcodec.preflight_image_nvimagecodec(_jpeg())
        if outcome == "fallback":
            assert result is None
        else:
            assert result is not None and (result.width, result.height) == (
                width,
                height,
            )
