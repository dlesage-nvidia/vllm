# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from io import BytesIO
from pathlib import Path

import numpy as np
import pybase64 as base64
import pytest
import torch
from PIL import Image, features

from tests.multimodal.test_image_decoder import (
    _palette1_tiff,
    _rgb16_bmp,
    _rgba_bitfields_bmp,
)
from vllm.multimodal.image_decoders.nvimagecodec import (
    NvImageCodecInput,
    create_nvimagecodec_decode_service,
)
from vllm.multimodal.media import ImageMediaIO

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")


def _encode(image: Image.Image, image_format: str, **kwargs) -> bytes:
    with BytesIO() as buffer:
        image.save(buffer, image_format, **kwargs)
        return buffer.getvalue()


def _rgb32_v2_bmp() -> bytes:
    pixels = struct.pack("<II", 0x001E140A, 0x003C3228)
    dib = struct.pack("<IiiHHIIiiII", 52, 2, 1, 1, 32, 3, len(pixels), 0, 0, 0, 0)
    dib += struct.pack("<III", 0x00FF0000, 0x0000FF00, 0x000000FF)
    offset = 14 + len(dib)
    return (
        struct.pack("<2sIHHI", b"BM", offset + len(pixels), 0, 0, offset) + dib + pixels
    )


def _jpeg(image: Image.Image, **kwargs) -> tuple[bytes, int]:
    return _encode(image, "JPEG", **kwargs), 6


def _gradient(width: int, height: int, offset: int = 0) -> Image.Image:
    y, x = np.mgrid[:height, :width]
    channels = (
        (x * 3 + offset * 17) % 256,
        (y * 5 + offset * 29) % 256,
        (x + y * 2 + offset * 41) % 256,
    )
    return Image.fromarray(np.stack(channels, axis=-1).astype(np.uint8))


def _pillow_pixels(data: bytes) -> np.ndarray:
    media = ImageMediaIO(backend="pillow").load_bytes(data).media
    return np.array(media, copy=True)


def _decode_batch(service, encoded: list[bytes], layout: str = "hwc_rgb"):
    image_io = ImageMediaIO(backend="nvimagecodec", output_layout=layout)
    prepared = [image_io._prepare_bytes(data) for data in encoded]
    futures = [
        service.submit(item) for item in prepared if isinstance(item, NvImageCodecInput)
    ]
    outputs = iter(future.result() for future in futures)
    for index, (data, item) in enumerate(zip(encoded, prepared, strict=True)):
        if isinstance(item, NvImageCodecInput):
            prepared[index] = image_io._wrap_native(next(outputs), data)
    return prepared


def _assert_native(item, expected: np.ndarray, layout="HWC", atol=0) -> None:
    assert item.io_config == {"backend": "nvimagecodec", "output_layout": layout}
    array = item.media
    assert type(array) is np.ndarray and array.dtype == np.uint8
    assert array.flags.c_contiguous and array.flags.writeable
    assert array.flags.owndata and array.base is None
    np.testing.assert_allclose(array, expected, rtol=0, atol=atol)


def _assert_pillow_batch(loaded, cases, layout="hwc_rgb") -> None:
    chw = layout == "chw_rgb"
    for item, (data, atol) in zip(loaded, cases, strict=True):
        expected = _pillow_pixels(data)
        if chw:
            expected = np.moveaxis(expected, -1, 0)
        _assert_native(item, expected, "CHW" if chw else "HWC", atol)


@pytest.fixture
def nvimagecodec_service():
    pytest.importorskip("nvidia.nvimgcodec")
    service = create_nvimagecodec_decode_service()
    try:
        service.wait_until_ready()
        yield service
    finally:
        service.close()


# Lines: libjpeg-turbo YCCK JPEG, then nvJPEG2000-generated raw and boxed
# HTJ2K codestreams.
_FIXTURES = Path(__file__).with_name("assets").joinpath("nvimagecodec.b64")
_YCCK_JPEG, _HTJ2K_J2K, _HTJ2K_JP2 = map(
    base64.b64decode, _FIXTURES.read_bytes().splitlines()
)


def test_real_core_codec_matrix_is_owned_after_close_and_matches_pillow(
    nvimagecodec_service,
) -> None:
    """Cover both layouts, JPEG variants, CPU plugins, tRNS, and ordering."""
    jpeg = _gradient(193, 97)
    rgb = [_gradient(65 + index * 2, 33 + index, index) for index in range(4)]
    pbm = Image.new("1", (61, 31), 1)
    pgm = Image.fromarray(np.arange(64 * 32, dtype=np.uint8).reshape(32, 64))
    pnm = [_encode(image, "PPM") for image in (pbm, pgm, rgb[2])]
    assert [data[:2] for data in pnm] == [b"P4", b"P5", b"P6"]

    keyed = Image.new("RGB", (4, 1))
    keyed.putdata([(10, 20, 30), (200, 100, 50), (10, 20, 30), (7, 8, 9)])
    hwc_cases = [
        _jpeg(jpeg, quality=95, subsampling=2),
        (_encode(rgb[0], "PNG"), 0),
        _jpeg(jpeg, quality=95, subsampling=0, progressive=True),
        (_encode(rgb[1], "BMP"), 0),
        _jpeg(jpeg.convert("L"), quality=95),
        (pnm[0], 0),
        _jpeg(jpeg.convert("CMYK"), quality=95),
        (pnm[1], 0),
        (_YCCK_JPEG, 6),
        (pnm[2], 0),
        (_encode(rgb[3], "WEBP", lossless=True), 0),
        (_encode(keyed, "PNG", transparency=(10, 20, 30)), 0),
    ]
    chw_cases = hwc_cases
    hwc = _decode_batch(nvimagecodec_service, [data for data, _ in hwc_cases])
    chw = _decode_batch(
        nvimagecodec_service, [data for data, _ in chw_cases], "chw_rgb"
    )
    nvimagecodec_service.close()

    _assert_pillow_batch(hwc, hwc_cases)
    _assert_pillow_batch(chw, chw_cases, "chw_rgb")


def test_real_jpeg2000_and_htj2k_codestreams_match_references(
    nvimagecodec_service,
) -> None:
    if not features.check("jpg_2000"):
        pytest.skip("Requires Pillow JPEG 2000")
    pytest.importorskip("nvidia.nvjpeg2k")
    image = _gradient(131, 73)
    jpeg2000 = [
        _encode(image, "JPEG2000", irreversible=False),
        _encode(image, "JPEG2000", irreversible=True),
        _encode(image, "JPEG2000", irreversible=False, no_jp2=True),
    ]
    y, x = np.mgrid[:8, :8]
    htj2k_expected = np.dstack(
        (
            (x * 13 + y * 3) % 256,
            (y * 17 + x * 5) % 256,
            (x * 3 + y * 5) % 256,
        )
    ).astype(np.uint8)
    encoded = [*jpeg2000, _HTJ2K_J2K, _HTJ2K_JP2]
    expected = [*map(_pillow_pixels, jpeg2000), htj2k_expected, htj2k_expected]
    tolerances = [2, 2, 2, 0, 0]
    for layout in ("hwc_rgb", "chw_rgb"):
        loaded = _decode_batch(nvimagecodec_service, encoded, layout)
        for item, pixels, atol in zip(loaded, expected, tolerances, strict=True):
            if layout == "chw_rgb":
                pixels = np.moveaxis(pixels, -1, 0)
            _assert_native(item, pixels, layout[:3].upper(), atol)


def test_real_non_native_image_semantics_fall_back_to_pillow() -> None:
    pytest.importorskip("nvidia.nvimgcodec")
    if not features.check("jpg_2000"):
        pytest.skip("Requires Pillow JPEG 2000")

    alpha = Image.new("LA", (2, 1))
    alpha.putdata([(50, 128), (200, 255)])
    palette = Image.new("P", (8, 4))
    palette.putpalette(list(range(256)) * 3)
    cases = [
        _rgba_bitfields_bmp(),
        _rgb16_bmp(),
        _rgb32_v2_bmp(),
        _encode(Image.new("1", (8, 4)), "BMP"),
        _encode(palette, "BMP"),
        _palette1_tiff(),
        _encode(alpha, "JPEG2000", irreversible=False),
        _encode(alpha, "JPEG2000", irreversible=False, no_jp2=True),
    ]
    image_io = ImageMediaIO(backend="nvimagecodec")
    for data in cases:
        loaded = image_io._prepare_bytes(data)
        assert not isinstance(loaded, NvImageCodecInput)
        assert (loaded.io_config or {}).get("backend") is None
        np.testing.assert_array_equal(np.asarray(loaded.media), _pillow_pixels(data))


def test_real_lossless_webp_with_exif_matches_pillow(nvimagecodec_service) -> None:
    exif = Image.Exif()
    exif[Image.ExifTags.Base.Orientation] = 6
    data = _encode(_gradient(79, 47), "WEBP", lossless=True, exif=exif)

    [loaded] = _decode_batch(nvimagecodec_service, [data])

    np.testing.assert_array_equal(np.asarray(loaded.media), _pillow_pixels(data))


def test_real_scaled_pnm_matches_pillow(nvimagecodec_service) -> None:
    encoded = [
        b"P2\n3 1\n100\n0 50 100\n",
        b"P5\n3 1\n15\n" + bytes([0, 7, 15]),
        b"P5\n# comment\r3 1\n15\n" + bytes([0, 7, 15]),
        b"P5\n# comment\r3 1\n255\n" + bytes([0, 127, 255]),
        b"P6\n1 1\n15\n" + bytes([0, 7, 15]),
    ]

    loaded = _decode_batch(nvimagecodec_service, encoded)

    for item, data in zip(loaded, encoded, strict=True):
        assert (item.io_config or {}).get("backend") is None
        np.testing.assert_array_equal(np.asarray(item.media), _pillow_pixels(data))


def test_real_native_codec_truncation_matches_pillow_failure(
    nvimagecodec_service,
) -> None:
    if not features.check("jpg_2000"):
        pytest.skip("Requires Pillow JPEG 2000")
    pytest.importorskip("nvidia.nvjpeg2k")
    pytest.importorskip("nvidia.nvtiff")
    image = _gradient(96, 64)
    truncated = [
        _encode(image, "WEBP", lossless=True)[:-1],
        _encode(image, "TIFF", compression="raw")[:-1],
        _encode(Image.new("1", image.size, 1), "PPM")[:-1],
        _encode(image.convert("L"), "PPM")[:-1],
        _encode(image, "PPM")[:-1],
        _encode(image, "JPEG2000", irreversible=False)[:-1],
        _encode(image, "JPEG2000", irreversible=False, no_jp2=True)[:-1],
        _HTJ2K_J2K[:-8],
        _HTJ2K_JP2[:-8],
    ]

    for data in truncated:
        with pytest.raises(OSError), Image.open(BytesIO(data)) as source:
            source.load()
        with pytest.raises(ValueError):
            _decode_batch(nvimagecodec_service, [data])


def test_real_tiff_compressions_orientations_and_multipage_match_pillow(
    nvimagecodec_service,
) -> None:
    pytest.importorskip("nvidia.nvtiff")
    first = _gradient(79, 41)
    cases = [
        (_encode(first, "TIFF", compression="raw"), 0),
        (_encode(first, "TIFF", compression="tiff_lzw"), 0),
        (_encode(first, "TIFF", compression="tiff_deflate"), 0),
        (_encode(first, "TIFF", compression="jpeg"), 6),
    ]
    palette = Image.fromarray(
        np.arange(79 * 41, dtype=np.uint8).reshape(41, 79), mode="P"
    )
    palette.putpalette(
        [value for index in range(256) for value in (index, index // 2, 255 - index)]
    )
    cases.append((_encode(palette, "TIFF", compression="raw"), 1))
    oriented = _gradient(53, 31)
    for orientation in range(1, 9):
        exif = Image.Exif()
        exif[Image.ExifTags.Base.Orientation] = orientation
        cases.append((_encode(oriented, "TIFF", compression="raw", exif=exif), 0))

    multipage = _encode(
        first,
        "TIFF",
        save_all=True,
        append_images=[_gradient(37, 23, 4)],
        compression="tiff_lzw",
    )
    loaded = _decode_batch(
        nvimagecodec_service,
        [data for data, _ in cases] + [multipage],
    )
    chw = _decode_batch(nvimagecodec_service, [data for data, _ in cases], "chw_rgb")

    _assert_pillow_batch(loaded[:-1], cases)
    _assert_pillow_batch(chw, cases, "chw_rgb")

    fallback = loaded[-1]
    assert fallback.io_config is None
    np.testing.assert_array_equal(np.asarray(fallback.media), _pillow_pixels(multipage))

    lab = _encode(first.convert("LAB"), "TIFF", compression="raw")
    exif = Image.Exif()
    exif[Image.ExifTags.Base.Orientation] = 6
    group4 = _encode(first.convert("1"), "TIFF", compression="group4", exif=exif)
    lzma = _encode(first, "TIFF", compression="lzma")
    inverse = _encode(first.convert("L"), "TIFF", tiffinfo={262: 0})
    reversed_order = _encode(first.convert("L"), "TIFF", tiffinfo={266: 2})
    fallback_data = (lab, group4, lzma, inverse, reversed_order)
    fallbacks = _decode_batch(nvimagecodec_service, list(fallback_data))
    for fallback, data in zip(fallbacks, fallback_data, strict=True):
        assert (fallback.io_config or {}).get("backend") is None
        np.testing.assert_array_equal(np.asarray(fallback.media), _pillow_pixels(data))


def test_signed_truncation_and_malformed_input_fail_closed(
    nvimagecodec_service,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _gradient(193, 97)
    jpeg = _encode(source, "JPEG", quality=95)
    png = _encode(source, "PNG")
    bmp = _encode(source, "BMP")
    sos = jpeg.index(b"\xff\xda")
    scan = sos + 2 + int.from_bytes(jpeg[sos + 2 : sos + 4], "big")
    corrupt_at = scan + (jpeg.rindex(b"\xff\xd9") - scan) // 2
    cases = [
        (jpeg[: len(jpeg) // 2], "Invalid JPEG image"),
        (
            jpeg[:corrupt_at] + b"\xff\xc1" + jpeg[corrupt_at + 2 :],
            "Invalid JPEG image",
        ),
        (png[:-12], "Invalid PNG image"),
        (bmp[:-3], "Invalid BMP image"),
    ]
    image_io = ImageMediaIO(backend="nvimagecodec")

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"not an image")
    monkeypatch.setattr(
        ImageMediaIO,
        "_load_bytes_pillow",
        lambda *_args, **_kwargs: pytest.fail(
            "signed native input unexpectedly spilled to Pillow"
        ),
    )
    for data, message in cases:
        with pytest.raises(ValueError, match=message):
            image_io.load_bytes(data)
