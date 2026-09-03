# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from io import BytesIO
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageOps

import vllm.envs as envs
import vllm.multimodal.image_decoders.nvimagecodec as nvcodec

pytestmark = pytest.mark.cpu_test


def _encode(image_format: str, mode: str = "RGB", exif=None, **kwargs) -> bytes:
    if exif is not None:
        metadata = Image.Exif()
        metadata[exif[0]] = exif[1]
        kwargs["exif"] = metadata
    try:
        with BytesIO() as buffer:
            image = Image.new(mode, (8, 4))
            if mode == "P":
                image.putpalette(list(range(256)) * 3)
            image.save(buffer, image_format, **kwargs)
            return buffer.getvalue()
    except OSError:
        pytest.skip(f"Pillow was built without {image_format} support")
        return b""


def _rgba_bitfields_bmp() -> bytes:
    width, height = 2, 1
    pixels = struct.pack("<II", 0x001E140A, 0xFF3C3228)
    dib = struct.pack(
        "<IiiHHIIiiII",
        108,
        width,
        height,
        1,
        32,
        3,
        len(pixels),
        2835,
        2835,
        0,
        0,
    )
    dib += struct.pack(
        "<IIIII",
        0x00FF0000,
        0x0000FF00,
        0x000000FF,
        0xFF000000,
        0x73524742,
    )
    dib += bytes(48)
    offset = 14 + len(dib)
    file_header = b"BM" + struct.pack("<IHHI", offset + len(pixels), 0, 0, offset)
    return file_header + dib + pixels


def _rgb16_bmp() -> bytes:
    pixels = struct.pack("<HH", 0x7C00, 0x03E0)
    offset = 14 + 40
    dib = struct.pack("<IiiHHIIiiII", 40, 2, 1, 1, 16, 0, len(pixels), 0, 0, 0, 0)
    return (
        struct.pack("<2sIHHI", b"BM", offset + len(pixels), 0, 0, offset) + dib + pixels
    )


def _palette1_tiff() -> bytes:
    width, height = 8, 2
    entries = [
        (256, 4, 1, width),
        (257, 4, 1, height),
        (258, 3, 1, 1),
        (259, 3, 1, 1),
        (262, 3, 1, 3),
        (277, 3, 1, 1),
        (278, 4, 1, height),
        (279, 4, 1, height),
    ]
    color_map = struct.pack("<6H", 0, 65535, 0, 0, 0, 65535)
    ifd_size = 2 + 10 * 12 + 4
    color_map_offset = 8 + ifd_size
    pixels = bytes((0b01010101, 0b10101010))
    pixel_offset = color_map_offset + len(color_map)
    entries.extend([(273, 4, 1, pixel_offset), (320, 3, 6, color_map_offset)])

    data = bytearray(b"II*\x00" + struct.pack("<I", 8) + struct.pack("<H", 10))
    for tag, field_type, count, value in sorted(entries):
        data += struct.pack("<HHI", tag, field_type, count)
        data += (
            struct.pack("<H", value) + bytes(2)
            if field_type == 3 and count == 1
            else struct.pack("<I", value)
        )
    return bytes(data + struct.pack("<I", 0) + color_map + pixels)


def _install_codec(monkeypatch, *, codec="jpeg", error=None, **metadata):
    metadata = dict(width=8, height=4, precision=8, num_channels=3) | metadata
    stream = SimpleNamespace(codec_name=codec, **metadata)
    stream.get_sub_code_stream = lambda _index: stream

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
    assert nvcodec.ensure_nvimagecodec_available() is None
    monkeypatch.setattr(nvcodec.importlib.util, "find_spec", lambda _name: None)
    with pytest.raises(RuntimeError, match="requires.*nvidia-nvimgcodec"):
        nvcodec.ensure_nvimagecodec_available()


@pytest.mark.parametrize(
    ("codec", "signature"),
    [
        ("bmp", b"BMbroken"),
        ("jpeg", b"\xff\xd8broken"),
        ("png", b"\x89PNG\r\n\x1a\nbroken"),
        ("pnm", b"P6\n1 1\n255\nbr"),
        ("webp", b"RIFF\x04\x00\x00\x00WEBPbroken"),
        ("tiff", b"II*\x00broken"),
        ("jpeg2k", b"\x00\x00\x00\x0cjP  \r\n\x87\nbroken"),
    ],
)
def test_native_codec_admission_and_signed_corruption(monkeypatch, codec, signature):
    image_format = {"pnm": "PPM", "jpeg2k": "JPEG2000"}.get(codec, codec.upper())
    data = _encode(image_format)
    _install_codec(monkeypatch, codec=codec)
    result = nvcodec.preflight_image_nvimagecodec(data)
    assert result is not None
    assert result.output_layout == "hwc_rgb"
    _install_codec(monkeypatch, error=RuntimeError("bad header"))
    assert nvcodec.preflight_image_nvimagecodec(data) is None
    with pytest.raises(ValueError, match="Invalid .* image"):
        nvcodec.preflight_image_nvimagecodec(signature)


@pytest.mark.parametrize(
    ("codec", "mode", "orientation"),
    [
        ("jpeg", "RGB", 1),
        ("jpeg", "L", 1),
        ("png", "RGB", 1),
        ("jpeg", "RGB", 6),
    ],
)
def test_requested_and_device_layouts(monkeypatch, codec, mode, orientation):
    data = _encode(
        codec.upper(),
        mode,
        exif=(Image.ExifTags.Base.Orientation, orientation),
    )
    _install_codec(monkeypatch, codec=codec, num_channels=len(mode))
    result = nvcodec.preflight_image_nvimagecodec(data, output_layout="chw_rgb")
    assert result is not None
    assert result.orientation == orientation
    assert result.output_layout == "chw_rgb"


@pytest.mark.parametrize(
    ("width", "height", "limit", "error"),
    [
        (8, 4, 32, None),
        (8, 4, 31, "maximum of 31 pixels"),
        (3840, 2160, 0, None),
        (3841, 2160, 0, "fixed raster ceiling"),
    ],
)
def test_pixel_boundaries(monkeypatch, width, height, limit, error):
    data = _encode("JPEG")
    _install_codec(monkeypatch, width=width, height=height)
    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", limit)
    if error:
        with pytest.raises(ValueError, match=error):
            nvcodec.preflight_image_nvimagecodec(data)
    else:
        result = nvcodec.preflight_image_nvimagecodec(data)
        assert result is not None and (result.width, result.height) == (width, height)


def test_unsupported_output_mode_falls_back_before_loading_package(monkeypatch):
    monkeypatch.setattr(nvcodec, "_load_nvimgcodec", lambda: pytest.fail("loaded"))
    assert nvcodec.preflight_image_nvimagecodec(b"image", image_mode="L") is None


def test_preserved_mode_only_admits_opaque_rgb(monkeypatch):
    rgb = _encode("PNG", "RGB")
    _install_codec(monkeypatch, codec="png", num_channels=3)
    assert nvcodec.preflight_image_nvimagecodec(rgb, image_mode=None) is not None

    transparent = _encode("PNG", "RGBA")
    _install_codec(monkeypatch, codec="png", num_channels=4)
    assert nvcodec.preflight_image_nvimagecodec(transparent, image_mode=None) is None

    keyed = _encode("PNG", "RGB", transparency=(0, 0, 0))
    _install_codec(monkeypatch, codec="png", num_channels=3)
    assert nvcodec.preflight_image_nvimagecodec(keyed, image_mode=None) is None

    grayscale = _encode("JPEG", "L")
    _install_codec(monkeypatch, codec="jpeg", num_channels=1)
    assert nvcodec.preflight_image_nvimagecodec(grayscale, image_mode=None) is None


def test_jpeg_structural_validation(monkeypatch):
    for progressive in (False, True):
        data = _encode("JPEG", progressive=progressive)
        assert nvcodec.jpeg_has_complete_scan_and_eoi(data)
        assert not nvcodec.jpeg_has_complete_scan_and_eoi(data[:-2])

    sos = data.index(b"\xff\xda")
    scan = sos + 2 + int.from_bytes(data[sos + 2 : sos + 4], "big")
    malformed = data[:scan] + b"\xff\xc1" + data[scan + 2 :]
    assert not nvcodec.jpeg_has_complete_scan_and_eoi(malformed)
    _install_codec(monkeypatch)
    with pytest.raises(ValueError, match="Invalid JPEG image"):
        nvcodec.preflight_image_nvimagecodec(malformed)


def test_jp2_and_bmp_header_structure_fail_closed(monkeypatch):
    invalid_jp2 = nvcodec._JPEG2000_SIGNATURE + struct.pack(
        ">I4sI4sI", 20, b"jp2h", 12, b"res ", 9
    )
    bmp = _encode("BMP")
    mutations = (
        (10, 4, len(bmp)),
        (14, 4, 66),
        (18, 4, -64),
        (26, 2, 0),
        (28, 2, 2),
        (30, 4, 1),
    )
    _install_codec(monkeypatch)

    cases = [(invalid_jp2, "JPEG 2000")]
    for offset, size, value in mutations:
        invalid_bmp = bytearray(bmp)
        invalid_bmp[offset : offset + size] = value.to_bytes(
            size, "little", signed=value < 0
        )
        cases.append((bytes(invalid_bmp), "BMP"))
    for data, label in cases:
        with pytest.raises(ValueError, match=f"Invalid {label} image"):
            nvcodec.preflight_image_nvimagecodec(data)


@pytest.mark.parametrize(
    ("image_format", "mode", "codec"),
    [
        ("PNG", "RGBA", "png"),
        ("WEBP", "RGB", "webp"),
        ("TIFF", "CMYK", "tiff"),
        ("TIFF", "LAB", "tiff"),
        ("JPEG", "RGB", "jpeg"),
        (None, "RGBA", "jpeg2k"),
        (None, "RGB", "gif"),
    ],
)
def test_pillow_semantic_fallbacks(monkeypatch, image_format, mode, codec):
    if image_format is None:
        data = b"jpeg2k"
    else:
        kwargs: dict[str, object] = {}
        if codec == "webp":
            kwargs.update(save_all=True, append_images=[Image.new(mode, (8, 4), "red")])
        elif codec == "jpeg":
            kwargs["exif"] = (Image.ExifTags.Base.ImageID, "stable-image-id")
        data = _encode(image_format, mode, **kwargs)
    _install_codec(monkeypatch, codec=codec, num_channels=len(mode))
    assert nvcodec.preflight_image_nvimagecodec(data) is None


def test_native_alpha_semantics_fall_back_to_pillow(monkeypatch):
    cases = [
        (_rgba_bitfields_bmp(), "bmp", 4),
        (_encode("JPEG2000", "LA", no_jp2=True), "jpeg2k", 2),
    ]
    for data, codec, channels in cases:
        _install_codec(monkeypatch, codec=codec, num_channels=channels)
        assert nvcodec.preflight_image_nvimagecodec(data) is None


@pytest.mark.parametrize("mode", ["1", "P"])
def test_paletted_bmp_falls_back_to_pillow(monkeypatch, mode):
    _install_codec(monkeypatch, codec="bmp")

    assert nvcodec.preflight_image_nvimagecodec(_encode("BMP", mode)) is None


def test_eight_bit_palette_tiff_uses_native_rgb_decode(monkeypatch):
    image = Image.fromarray(np.arange(32, dtype=np.uint8).reshape(4, 8), mode="P")
    image.putpalette([value for index in range(256) for value in (index, 0, 0)])
    with BytesIO() as buffer:
        image.save(buffer, "TIFF", compression="raw")
        data = buffer.getvalue()

    _install_codec(monkeypatch, codec="tiff", precision=16, num_channels=3)

    assert nvcodec.preflight_image_nvimagecodec(data) is not None


@pytest.mark.parametrize(
    ("data", "codec", "precision", "channels"),
    [
        (_rgb16_bmp(), "bmp", 0, 0),
        (_rgb16_bmp(), "bmp", 8, 3),
        (_palette1_tiff(), "tiff", 16, 3),
    ],
)
def test_valid_ambiguous_native_metadata_falls_back(
    monkeypatch, data, codec, precision, channels
):
    _install_codec(monkeypatch, codec=codec, precision=precision, num_channels=channels)

    assert nvcodec.preflight_image_nvimagecodec(data) is None


@pytest.mark.parametrize(
    "data",
    [
        b"P2\n3 1\n100\n0 50 100\n",
        b"P5\n# scaled gray\n3 1\n15\n" + bytes([0, 7, 15]),
        b"P5\n# scaled gray\r3 1\n15\n" + bytes([0, 7, 15]),
        b"P6\n1 1\n15\n" + bytes([0, 7, 15]),
    ],
)
def test_scaled_pnm_falls_back_to_pillow(monkeypatch, data):
    _install_codec(monkeypatch, codec="pnm")

    assert nvcodec.preflight_image_nvimagecodec(data) is None


def test_valid_image_falls_back_on_invalid_native_dimensions(monkeypatch):
    data = b"P5\n# comment\r3 1\n255\n" + bytes([0, 127, 255])
    _install_codec(monkeypatch, codec="pnm", width=0, height=0, num_channels=0)

    assert nvcodec.preflight_image_nvimagecodec(data) is None


def test_group4_tiff_falls_back_to_pillow(monkeypatch):
    data = _encode(
        "TIFF",
        "1",
        exif=(Image.ExifTags.Base.Orientation, 6),
        compression="group4",
    )
    _install_codec(monkeypatch, codec="tiff", precision=1, num_channels=1)

    assert nvcodec.preflight_image_nvimagecodec(data) is None


def test_unavailable_tiff_compression_falls_back_to_pillow(monkeypatch):
    data = _encode("TIFF", "RGB", compression="lzma")
    _install_codec(monkeypatch, codec="tiff")

    assert nvcodec.preflight_image_nvimagecodec(data) is None


@pytest.mark.parametrize("tiffinfo", [{262: 0}, {266: 2}, {262: 0, 266: 2}])
def test_unsupported_tiff_sample_order_falls_back_to_pillow(monkeypatch, tiffinfo):
    data = _encode("TIFF", "L", tiffinfo=tiffinfo)
    _install_codec(monkeypatch, codec="tiff", num_channels=1)

    assert nvcodec.preflight_image_nvimagecodec(data) is None


def test_valid_pillow_image_falls_back_on_native_parser_failure(monkeypatch):
    exif = (Image.ExifTags.Base.Orientation, 6)
    cases = [
        _encode("JPEG2000", "LA"),
        _encode("WEBP", "RGB", exif=exif, lossless=True),
    ]
    _install_codec(monkeypatch, error=RuntimeError("unsupported component layout"))
    for data in cases:
        assert nvcodec.preflight_image_nvimagecodec(data) is None

    monkeypatch.setattr(
        nvcodec,
        "_load_nvimgcodec",
        lambda: SimpleNamespace(
            CodeStream=lambda _data: SimpleNamespace(codec_name="webp")
        ),
    )
    assert nvcodec.preflight_image_nvimagecodec(cases[1]) is None


def test_parser_failure_probes_pillow_in_normal_load_order(monkeypatch):
    class _RetryableImage:
        size = (8, 4)
        primed = False

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def load(self):
            if not self.primed:
                raise OSError("first load attempt failed")

    image = _RetryableImage()

    def prime_then_fail(opened):
        opened.primed = True
        raise OSError("normalization failed after touching the decoder")

    data = _encode("WEBP")
    _install_codec(monkeypatch, error=RuntimeError("native parser failed"))
    monkeypatch.setattr(nvcodec.Image, "open", lambda _source: image)
    monkeypatch.setattr(nvcodec.ImageOps, "exif_transpose", prime_then_fail)

    assert nvcodec.preflight_image_nvimagecodec(data) is None


@pytest.mark.parametrize("orientation", range(1, 9))
def test_exif_orientation_view_matches_pillow(orientation):
    array = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)
    image = Image.fromarray(array, mode="RGB")
    image.getexif()[Image.ExifTags.Base.Orientation] = orientation
    np.testing.assert_array_equal(
        nvcodec._apply_exif_orientation_view(array, orientation),
        np.asarray(ImageOps.exif_transpose(image)),
    )
