# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from vllm.multimodal.image_decoders import NvImageCodecBatchItemError
from vllm.multimodal.media import ImageBatchItemError, ImageMediaIO

pytestmark = pytest.mark.cpu_test

ASSETS_DIR = Path(__file__).parent.parent / "assets"
assert ASSETS_DIR.exists()


def test_image_media_io_rgba_custom_background(tmp_path):
    """Test RGBA to RGB conversion with custom background colors."""
    # Create a simple RGBA image with transparent and opaque pixels
    rgba_image = Image.new("RGBA", (10, 10), (255, 0, 0, 255))  # Red with full opacity

    # Make top-left quadrant transparent
    for i in range(5):
        for j in range(5):
            rgba_image.putpixel((i, j), (0, 0, 0, 0))  # Fully transparent

    # Save the test image to tmp_path
    test_image_path = tmp_path / "test_rgba.png"
    rgba_image.save(test_image_path)

    # Test 1: Default white background (backward compatibility)
    image_io_default = ImageMediaIO()
    converted_default = image_io_default.load_file(test_image_path)
    default_numpy = np.array(converted_default)

    # Check transparent pixels are white
    assert default_numpy[0][0][0] == 255  # R
    assert default_numpy[0][0][1] == 255  # G
    assert default_numpy[0][0][2] == 255  # B
    # Check opaque pixels remain red
    assert default_numpy[5][5][0] == 255  # R
    assert default_numpy[5][5][1] == 0  # G
    assert default_numpy[5][5][2] == 0  # B

    # Test 2: Custom black background via kwargs
    image_io_black = ImageMediaIO(rgba_background_color=(0, 0, 0))
    converted_black = image_io_black.load_file(test_image_path)
    black_numpy = np.array(converted_black)

    # Check transparent pixels are black
    assert black_numpy[0][0][0] == 0  # R
    assert black_numpy[0][0][1] == 0  # G
    assert black_numpy[0][0][2] == 0  # B
    # Check opaque pixels remain red
    assert black_numpy[5][5][0] == 255  # R
    assert black_numpy[5][5][1] == 0  # G
    assert black_numpy[5][5][2] == 0  # B

    # Test 3: Custom blue background via kwargs (as list)
    image_io_blue = ImageMediaIO(rgba_background_color=[0, 0, 255])
    converted_blue = image_io_blue.load_file(test_image_path)
    blue_numpy = np.array(converted_blue)

    # Check transparent pixels are blue
    assert blue_numpy[0][0][0] == 0  # R
    assert blue_numpy[0][0][1] == 0  # G
    assert blue_numpy[0][0][2] == 255  # B

    # Test 4: Test with load_bytes method
    with open(test_image_path, "rb") as f:
        image_data = f.read()

    image_io_green = ImageMediaIO(rgba_background_color=(0, 255, 0))
    converted_green = image_io_green.load_bytes(image_data)
    green_numpy = np.array(converted_green)

    # Check transparent pixels are green
    assert green_numpy[0][0][0] == 0  # R
    assert green_numpy[0][0][1] == 255  # G
    assert green_numpy[0][0][2] == 0  # B


def test_image_media_io_no_mode_conversion(tmp_path):
    """image_mode=None skips conversion and preserves the original mode."""
    # RGBA image: opaque black pixel on a fully transparent background
    rgba_image = Image.new("RGBA", (10, 10), (0, 0, 0, 0))
    rgba_image.putpixel((5, 5), (0, 0, 0, 255))
    test_image_path = tmp_path / "test_rgba.png"
    rgba_image.save(test_image_path)

    # Default behavior: RGBA is composited onto a white background
    image_io_default = ImageMediaIO()
    converted_default = image_io_default.load_file(test_image_path)
    assert converted_default.media.mode == "RGB"
    assert converted_default.media.getpixel((0, 0)) == (255, 255, 255)
    assert converted_default.media.getpixel((5, 5)) == (0, 0, 0)

    # image_mode=None: original mode and alpha channel are preserved
    image_io_keep = ImageMediaIO(image_mode=None)
    converted_keep = image_io_keep.load_file(test_image_path)
    assert converted_keep.media.mode == "RGBA"
    assert converted_keep.media.getpixel((0, 0)) == (0, 0, 0, 0)
    assert converted_keep.media.getpixel((5, 5)) == (0, 0, 0, 255)


def test_image_media_io_rgba_background_color_validation():
    """Test that invalid rgba_background_color values are properly rejected."""

    # Test invalid types
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color="255,255,255")

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=255)

    # Test wrong number of elements
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, 255, 255, 255))

    # Test non-integer values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255.0, 255.0, 255.0))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, "255", 255))

    # Test out of range values
    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(256, 255, 255))

    with pytest.raises(
        ValueError, match="rgba_background_color must be a list or tuple"
    ):
        ImageMediaIO(rgba_background_color=(255, -1, 255))

    # Test that valid values work
    ImageMediaIO(rgba_background_color=(0, 0, 0))  # Should not raise
    ImageMediaIO(rgba_background_color=[255, 255, 255])  # Should not raise
    ImageMediaIO(rgba_background_color=(128, 128, 128))  # Should not raise


def test_image_media_io_load_bytes(tmp_path):
    """Test load_bytes with valid and invalid image data."""
    # Save a valid RGB image to use as source bytes
    valid_image = Image.new("RGB", (8, 8), (100, 150, 200))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)

    valid_data = valid_path.read_bytes()

    # Test 1: Valid image bytes load successfully and are fully decoded
    image_io = ImageMediaIO()
    result = image_io.load_bytes(valid_data)

    # Check the returned media is a properly loaded image
    assert isinstance(result.media, Image.Image)
    assert result.media.size == (8, 8)
    assert result.media.getpixel((0, 0)) == (100, 150, 200)

    # Test 2: Garbage bytes raise ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"not an image")

    # Test 3: Truncated PNG header raises ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    # Test 4: Real PNG truncated mid-stream raises ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(valid_data[: len(valid_data) // 2])

    # Test 5: Empty bytes raise ValueError
    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_bytes(b"")


def test_image_media_io_load_bytes_many_reports_open_error_index(tmp_path):
    valid_image = Image.new("RGB", (2, 2))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)

    with pytest.raises(ImageBatchItemError) as exc_info:
        ImageMediaIO().load_bytes_many(
            [valid_path.read_bytes(), b"not an image", valid_path.read_bytes()]
        )

    assert exc_info.value.index == 1
    assert type(exc_info.value.error) is ValueError
    assert "Failed to load image" in str(exc_info.value.error)


def test_image_media_io_load_bytes_many_reports_load_error_index(tmp_path, monkeypatch):
    valid_image = Image.new("RGB", (2, 2))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)
    encoded = valid_path.read_bytes()

    original_open = Image.open
    open_index = 0

    def indexed_open(*args, **kwargs):
        nonlocal open_index
        image = original_open(*args, **kwargs)
        image._batch_test_index = open_index
        open_index += 1
        return image

    original_load = Image.Image.load

    def fail_second_load(image, *args, **kwargs):
        if image._batch_test_index == 1:
            raise OSError("simulated pixel load failure")
        return original_load(image, *args, **kwargs)

    monkeypatch.setattr(Image, "open", indexed_open)
    monkeypatch.setattr(Image.Image, "load", fail_second_load)
    monkeypatch.setattr(
        "vllm.multimodal.media.image.normalize_image", lambda image: image
    )

    with pytest.raises(ImageBatchItemError) as exc_info:
        ImageMediaIO().load_bytes_many([encoded, encoded, encoded])

    assert exc_info.value.index == 1
    assert type(exc_info.value.error) is ValueError
    assert str(exc_info.value.error).endswith("simulated pixel load failure")


def test_image_media_io_load_bytes_many_reports_conversion_error_index(
    tmp_path, monkeypatch
):
    valid_image = Image.new("RGB", (2, 2))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)
    encoded = valid_path.read_bytes()
    original_convert = ImageMediaIO._convert_image_mode
    conversion_index = 0
    converted_images = []

    def fail_second_conversion(image_io, image):
        nonlocal conversion_index
        index = conversion_index
        conversion_index += 1
        if index == 1:
            raise ValueError("simulated conversion failure")
        converted = original_convert(image_io, image)
        converted_images.append(converted)
        return converted

    monkeypatch.setattr(ImageMediaIO, "_convert_image_mode", fail_second_conversion)

    with pytest.raises(ImageBatchItemError) as exc_info:
        ImageMediaIO(image_mode="RGBA").load_bytes_many([encoded, encoded, encoded])

    assert exc_info.value.index == 1
    assert type(exc_info.value.error) is ValueError
    assert str(exc_info.value.error) == "simulated conversion failure"
    with pytest.raises(ValueError, match="closed image"):
        converted_images[0].load()


def test_nvimagecodec_candidate_failure_reports_original_batch_index(monkeypatch):
    noncandidate = Image.new("P", (2, 2))
    candidate = Image.new("RGB", (2, 2))
    with BytesIO() as buffer:
        noncandidate.save(buffer, "GIF")
        gif_data = buffer.getvalue()
    with BytesIO() as buffer:
        candidate.save(buffer, "JPEG")
        jpeg_data = buffer.getvalue()
    original_error = ValueError("metadata copy failed")

    class RaisingInfo(dict):
        def update(self, *args, **kwargs):
            raise original_error

    class DecodedImage:
        info = RaisingInfo()

        def close(self):
            pass

    def decode_only_candidate(encoded_images, **kwargs):
        assert encoded_images == [jpeg_data]
        return [DecodedImage()]

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        decode_only_candidate,
    )

    with pytest.raises(ImageBatchItemError) as exc_info:
        ImageMediaIO(backend="nvimagecodec").load_bytes_many([gif_data, jpeg_data])

    assert exc_info.value.index == 1
    assert exc_info.value.error is original_error


def test_nvimagecodec_native_item_error_maps_to_original_batch_index(monkeypatch):
    noncandidate = Image.new("P", (2, 2))
    candidate = Image.new("RGB", (2, 2))
    with BytesIO() as buffer:
        noncandidate.save(buffer, "GIF")
        gif_data = buffer.getvalue()
    with BytesIO() as buffer:
        candidate.save(buffer, "JPEG")
        jpeg_data = buffer.getvalue()
    original_error = ValueError("GPU budget too small")

    def fail_native_candidate(encoded_images, **kwargs):
        assert encoded_images == [jpeg_data]
        raise NvImageCodecBatchItemError(0, original_error)

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        fail_native_candidate,
    )

    with pytest.raises(ImageBatchItemError) as exc_info:
        ImageMediaIO(backend="nvimagecodec").load_bytes_many([gif_data, jpeg_data])

    assert exc_info.value.index == 1
    assert exc_info.value.error is original_error


def test_image_media_io_load_file(tmp_path):
    """Test load_file with valid and invalid image files."""
    # Save a valid RGB image to disk
    valid_image = Image.new("RGB", (4, 4), (10, 20, 30))
    valid_path = tmp_path / "valid.png"
    valid_image.save(valid_path)

    # Test 1: Valid image file loads successfully and is fully decoded
    image_io = ImageMediaIO()
    result = image_io.load_file(valid_path)

    # Check the returned media is a properly loaded image
    assert isinstance(result.media, Image.Image)
    assert result.media.size == (4, 4)
    assert result.media.getpixel((0, 0)) == (10, 20, 30)

    # Test 2: File with garbage content raises ValueError
    bad_file = tmp_path / "bad.png"
    bad_file.write_bytes(b"this is not an image")

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(bad_file)

    # Test 3: File with truncated PNG header raises ValueError
    truncated_file = tmp_path / "truncated.png"
    truncated_file.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10)

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(truncated_file)

    # Test 4: Real PNG file truncated mid-stream raises ValueError
    valid_data = valid_path.read_bytes()
    truncated_real_file = tmp_path / "truncated_real.png"
    truncated_real_file.write_bytes(valid_data[: len(valid_data) // 2])

    with pytest.raises(ValueError, match="Failed to load image"):
        image_io.load_file(truncated_real_file)


def test_image_pixel_limit_respected():
    """A small image within the pixel limit loads successfully."""
    import vllm.envs as envs

    image = Image.new("RGB", (100, 100), (255, 0, 0))
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    assert envs.VLLM_MAX_IMAGE_PIXELS >= 100 * 100

    image_io = ImageMediaIO()
    result = image_io.load_bytes(data)
    assert result.media.size == (100, 100)


def test_image_pixel_limit_rejected(monkeypatch):
    """An image exceeding the pixel limit is rejected before raster decode."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 100)

    image = Image.new("RGB", (20, 20), (0, 255, 0))
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    image_io = ImageMediaIO()
    with pytest.raises(ValueError, match="exceed"):
        image_io.load_bytes(data)


def test_image_pixel_limit_disabled(monkeypatch):
    """Setting VLLM_MAX_IMAGE_PIXELS=0 disables the pixel limit."""
    import vllm.envs as envs

    monkeypatch.setattr(envs, "VLLM_MAX_IMAGE_PIXELS", 0)

    image = Image.new("RGB", (1000, 1000), (0, 0, 255))
    from io import BytesIO

    buf = BytesIO()
    image.save(buf, format="PNG")
    data = buf.getvalue()

    image_io = ImageMediaIO()
    result = image_io.load_bytes(data)
    assert result.media.size == (1000, 1000)


def _encode_test_image(image: Image.Image, image_format: str, **save_kwargs) -> bytes:
    with BytesIO() as buffer:
        image.save(buffer, format=image_format, **save_kwargs)
        return buffer.getvalue()


def test_nvimagecodec_backend_decodes_rgb_jpeg(monkeypatch):
    source = Image.new("RGB", (8, 4), (10, 20, 30))
    with BytesIO() as buffer:
        source.save(buffer, "JPEG", subsampling=2)
        data = buffer.getvalue()
    decoded = Image.new("RGB", source.size, (11, 22, 33))
    calls = []

    def fake_decode(encoded, *, output_modes, decoders, batch_size, pipeline_depth):
        calls.append((encoded, output_modes, decoders, batch_size, pipeline_depth))
        return [decoded]

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec", fake_decode
    )

    result = ImageMediaIO(backend="nvimagecodec", decoders=3).load_bytes(data)

    assert result.media is decoded
    assert result.io_config == {"backend": "nvimagecodec"}
    assert calls == [([data], ["RGB"], 3, 5, 4)]


def test_nvimagecodec_batches_all_supported_format_families(monkeypatch):
    formats = ["JPEG", "JPEG2000", "TIFF", "BMP", "PNG", "PPM", "WEBP"]
    encoded = [
        _encode_test_image(Image.new("RGB", (index + 2, 3)), image_format)
        for index, image_format in enumerate(formats)
    ]
    decoded = [
        Image.new("RGB", (index + 2, 3), (index, index, index))
        for index in range(len(formats))
    ]
    calls = []

    def fake_decode(datas, *, output_modes, decoders, batch_size, pipeline_depth):
        calls.append((datas, output_modes, decoders, batch_size, pipeline_depth))
        return decoded

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec", fake_decode
    )

    results = ImageMediaIO(
        backend="nvimagecodec", decoders=3, batch_size=4, pipeline_depth=3
    ).load_bytes_many(encoded)

    assert [result.media for result in results] == decoded
    assert [result.original_bytes for result in results] == encoded
    assert [result.io_config for result in results] == [
        {"backend": "nvimagecodec"}
    ] * len(formats)
    assert calls == [(encoded, ["RGB"] * len(formats), 3, 4, 3)]


def test_nvimagecodec_decodes_opaque_gpu_formats_as_rgb_for_rgba_target(monkeypatch):
    formats = ["JPEG", "JPEG2000", "TIFF"]
    encoded = [
        _encode_test_image(Image.new("RGB", (2, 3)), image_format)
        for image_format in formats
    ]
    decoded = [Image.new("RGB", (2, 3), (1, 2, 3)) for _ in formats]
    calls = []

    def fake_decode(datas, *, output_modes, **kwargs):
        calls.append((datas, output_modes))
        return decoded

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec", fake_decode
    )

    results = ImageMediaIO(backend="nvimagecodec", image_mode="RGBA").load_bytes_many(
        encoded
    )

    assert calls == [(encoded, ["RGB"] * len(formats))]
    assert [result.media.mode for result in results] == ["RGBA"] * len(formats)
    assert [result.io_config for result in results] == [
        {
            "image_mode": "RGBA",
            "rgba_background_color": (255, 255, 255),
            "backend": "nvimagecodec",
        }
    ] * len(formats)


@pytest.mark.parametrize(
    ("mode", "subsampling"),
    [("RGB", 0), ("RGB", 1), ("CMYK", None)],
)
def test_nvimagecodec_routes_extended_jpeg_inputs(
    monkeypatch, mode: str, subsampling: int | None
):
    source = Image.new(mode, (8, 4))
    with BytesIO() as buffer:
        save_kwargs = {} if subsampling is None else {"subsampling": subsampling}
        source.save(buffer, "JPEG", **save_kwargs)
        data = buffer.getvalue()

    calls = []

    def fake_decode(datas, *, output_modes, **kwargs):
        calls.append((datas, output_modes))
        return [Image.new("RGB", source.size)]

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec", fake_decode
    )

    result = ImageMediaIO(backend="nvimagecodec").load_bytes(data)

    assert result.io_config == {"backend": "nvimagecodec"}
    assert calls == [([data], ["RGB"])]


def test_nvimagecodec_backend_decodes_grayscale_jpeg(monkeypatch):
    source = Image.new("L", (8, 4), 10)
    data = _encode_test_image(source, "JPEG")
    decoded = Image.new("RGB", source.size, (11, 22, 33))

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        lambda *args, **kwargs: [decoded],
    )

    result = ImageMediaIO(backend="nvimagecodec").load_bytes(data)

    assert result.media is decoded
    assert result.io_config == {"backend": "nvimagecodec"}


def test_nvimagecodec_preserves_alpha_for_background_composite(monkeypatch):
    rgba = Image.new("RGBA", (2, 1), (1, 2, 3, 0))
    rgba.putpixel((1, 0), (40, 50, 60, 255))
    palette = Image.new("P", (2, 1))
    palette.putpalette([1, 2, 3, 40, 50, 60] + [0] * (256 * 3 - 6))
    palette.putdata([0, 1])
    encoded = [
        _encode_test_image(rgba, "PNG"),
        _encode_test_image(rgba, "WEBP", lossless=True),
        _encode_test_image(palette, "PNG", transparency=0),
    ]
    calls = []

    def fake_decode(datas, *, output_modes, **kwargs):
        calls.append((datas, output_modes))
        return [rgba.copy() for _ in datas]

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec", fake_decode
    )

    results = ImageMediaIO(
        backend="nvimagecodec",
        rgba_background_color=(9, 8, 7),
    ).load_bytes_many(encoded)

    assert calls == [(encoded, ["RGBA", "RGBA", "RGBA"])]
    for result in results:
        assert result.media.mode == "RGB"
        assert result.media.getpixel((0, 0)) == (9, 8, 7)
        assert result.media.getpixel((1, 0)) == (40, 50, 60)
        assert result.io_config == {
            "image_mode": "RGB",
            "rgba_background_color": (9, 8, 7),
            "backend": "nvimagecodec",
        }


def test_nvimagecodec_truecolor_png_transparency_matches_pillow(monkeypatch):
    source = Image.new("RGB", (2, 1))
    source.putdata([(10, 20, 30), (200, 100, 50)])
    data = _encode_test_image(source, "PNG", transparency=(10, 20, 30))
    expected = ImageMediaIO(backend="pillow").load_bytes(data).media
    calls = []

    def fake_decode(encoded, *, output_modes, **kwargs):
        calls.append(output_modes)
        with Image.open(BytesIO(encoded[0])) as image:
            return [image.convert(output_modes[0])]

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec", fake_decode
    )
    actual = ImageMediaIO(backend="nvimagecodec").load_bytes(data).media

    assert calls == [["RGB"]]
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_nvimagecodec_known_semantic_traps_stay_on_pillow(monkeypatch):
    alpha_tiff = _encode_test_image(Image.new("RGBA", (2, 2)), "TIFF")
    high_depth_png = _encode_test_image(
        Image.fromarray(np.array([[0, 65535]], dtype=np.uint16)), "PNG"
    )
    first_frame = Image.new("RGB", (2, 2), "red")
    animated_webp = _encode_test_image(
        first_frame,
        "WEBP",
        save_all=True,
        append_images=[Image.new("RGB", (2, 2), "blue")],
        duration=10,
        lossless=True,
    )

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        lambda *args, **kwargs: pytest.fail("semantic trap reached nvImageCodec"),
    )

    results = ImageMediaIO(backend="nvimagecodec").load_bytes_many(
        [alpha_tiff, high_depth_png, animated_webp]
    )

    assert [result.media.mode for result in results] == ["RGB", "RGB", "RGB"]
    assert all(
        (result.io_config or {}).get("backend") != "nvimagecodec" for result in results
    )


def test_nvimagecodec_batch_preserves_positional_fallback(monkeypatch):
    encoded = [
        _encode_test_image(Image.new("RGB", (2, 2), color), image_format)
        for color, image_format in [
            ((1, 2, 3), "JPEG"),
            ((4, 5, 6), "PNG"),
            ((7, 8, 9), "BMP"),
        ]
    ]
    first = Image.new("RGB", (2, 2), (10, 20, 30))
    third = Image.new("RGB", (2, 2), (70, 80, 90))
    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        lambda *args, **kwargs: [first, None, third],
    )

    results = ImageMediaIO(backend="nvimagecodec").load_bytes_many(encoded)

    assert results[0].media is first
    assert results[1].media.getpixel((0, 0)) == (4, 5, 6)
    assert results[2].media is third
    assert [result.io_config for result in results] == [
        {"backend": "nvimagecodec"},
        None,
        {"backend": "nvimagecodec"},
    ]


def test_nvimagecodec_codec_miss_falls_back_to_pillow(monkeypatch):
    source = Image.new("RGB", (4, 4), (40, 50, 60))
    data = _encode_test_image(source, "JPEG")
    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        lambda *args, **kwargs: [None],
    )

    result = ImageMediaIO(backend="nvimagecodec").load_bytes(data)

    assert result.media.mode == "RGB"
    assert result.media.size == source.size
    assert result.io_config is None


def test_nvimagecodec_applies_exif_orientation_once(monkeypatch):
    source = Image.new("RGB", (6, 4))
    source.putdata([(x * 30, y * 50, (x + y) * 20) for y in range(4) for x in range(6)])
    exif = Image.Exif()
    exif[Image.ExifTags.Base.Orientation] = 6
    exif[Image.ExifTags.Base.ImageDescription] = "preserved metadata"
    with BytesIO() as buffer:
        source.save(buffer, format="JPEG", quality=95, exif=exif)
        data = buffer.getvalue()

    expected = ImageMediaIO(backend="pillow").load_bytes(data).media

    def fake_decode(encoded, **kwargs):
        with Image.open(BytesIO(encoded[0])) as encoded_image:
            return [encoded_image.convert("RGB")]

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec", fake_decode
    )
    actual = ImageMediaIO(backend="nvimagecodec").load_bytes(data).media

    assert actual.size == expected.size == (4, 6)
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
    assert actual.getexif() == expected.getexif()


@pytest.mark.parametrize("orientation", range(2, 9))
def test_nvimagecodec_applies_tiff_exif_orientation_once(monkeypatch, orientation):
    source = Image.new("RGB", (6, 4))
    source.putdata([(x * 30, y * 50, (x + y) * 20) for y in range(4) for x in range(6)])
    exif = Image.Exif()
    exif[Image.ExifTags.Base.Orientation] = orientation
    data = _encode_test_image(source, "TIFF", compression="raw", exif=exif)
    expected = ImageMediaIO(backend="pillow").load_bytes(data).media

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        lambda *args, **kwargs: [source.copy()],
    )
    actual = ImageMediaIO(backend="nvimagecodec").load_bytes(data).media

    assert actual.size == expected.size
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))


def test_nvimagecodec_truncated_jpeg_uses_pillow_validation(monkeypatch):
    source = Image.new("RGB", (32, 16), (10, 20, 30))
    data = _encode_test_image(source, "JPEG", quality=95)[:-2]

    monkeypatch.setattr(
        "vllm.multimodal.media.image.decode_images_nvimagecodec",
        lambda *args, **kwargs: pytest.fail(
            "JPEG without an end-of-image marker reached nvImageCodec"
        ),
    )

    with pytest.raises(ValueError, match="Failed to load image"):
        ImageMediaIO(backend="nvimagecodec").load_bytes(data)


def test_image_merge_kwargs_strips_unconfigured_gpu_backend(monkeypatch):
    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "pillow")

    merged = ImageMediaIO.merge_kwargs(
        default_kwargs={"rgba_background_color": [0, 0, 0]},
        runtime_kwargs={"backend": "nvimagecodec", "image_mode": "RGB"},
    )

    assert merged == {
        "rgba_background_color": [0, 0, 0],
        "image_mode": "RGB",
    }


def test_image_merge_kwargs_preserves_statically_configured_gpu_backend():
    merged = ImageMediaIO.merge_kwargs(
        default_kwargs={
            "backend": "nvimagecodec",
            "decoders": 3,
            "batch_size": 4,
            "pipeline_depth": 3,
            "coalesce_timeout_ms": 0.25,
            "image_mode": None,
        },
        runtime_kwargs={
            "backend": "nvimagecodec",
            "decoders": 8,
            "batch_size": 16,
            "pipeline_depth": 7,
            "coalesce_timeout_ms": 1,
            "image_mode": "RGB",
        },
    )

    assert merged == {
        "backend": "nvimagecodec",
        "decoders": 3,
        "batch_size": 4,
        "pipeline_depth": 3,
        "coalesce_timeout_ms": 0.25,
        "image_mode": "RGB",
    }


def test_image_merge_kwargs_none_cannot_reveal_gpu_environment(monkeypatch):
    monkeypatch.setenv("VLLM_IMAGE_LOADER_BACKEND", "nvimagecodec")

    merged = ImageMediaIO.merge_kwargs(
        default_kwargs={"backend": "pillow"},
        runtime_kwargs={"backend": None},
    )

    assert merged == {"backend": "pillow"}


@pytest.mark.parametrize("backend", ["opencv", "unknown", ""])
def test_image_media_io_rejects_unknown_backend(backend: str):
    with pytest.raises(ValueError, match="Unknown image backend"):
        ImageMediaIO(backend=backend)
