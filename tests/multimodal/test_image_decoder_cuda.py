# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib import metadata
from io import BytesIO

import numpy as np
import pybase64 as base64
import pytest
import torch
from PIL import Image, Jpeg2KImagePlugin, features

from vllm.multimodal.gpu_ipc_memory import (
    MultiModalGPUMemoryPool,
    set_mm_gpu_ipc_pool,
)
from vllm.multimodal.image_decoders.nvimagecodec import (
    _nvimagecodec_decoder_pool,
    decode_images_nvimagecodec,
)
from vllm.multimodal.media import ImageMediaIO


@contextmanager
def _fresh_decoder_pool():
    pool = _nvimagecodec_decoder_pool
    old_state = (
        pool.slots,
        pool.active,
        pool.cond,
        pool.max_slots,
        pool.batch_size,
        pool.pipeline_depth,
        pool.owner_pid,
        pool.closing,
        pool.generation,
    )
    pool.slots = []
    pool.active = 0
    pool.cond = threading.Condition()
    pool.max_slots = None
    pool.batch_size = None
    pool.pipeline_depth = None
    pool.owner_pid = os.getpid()
    pool.closing = False
    pool.generation = 0
    try:
        yield
    finally:
        (
            pool.slots,
            pool.active,
            pool.cond,
            pool.max_slots,
            pool.batch_size,
            pool.pipeline_depth,
            pool.owner_pid,
            pool.closing,
            pool.generation,
        ) = old_state


def _encode(image: Image.Image, image_format: str, **kwargs) -> bytes:
    with BytesIO() as buffer:
        image.save(buffer, image_format, **kwargs)
        return buffer.getvalue()


def _pillow_decode(data: bytes, mode: str) -> Image.Image:
    with Image.open(BytesIO(data)) as image:
        return image.convert(mode)


def _has_distribution(*names: str) -> bool:
    for name in names:
        try:
            metadata.version(name)
            return True
        except metadata.PackageNotFoundError:
            pass
    return False


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Requires CUDA"
)


# Generated with libjpeg-turbo's JCS_YCCK output color space. The APP14 Adobe
# marker has transform=2, unlike the transform=0 CMYK files Pillow can create.
_YCCK_JPEG = base64.b64decode(
    "/9j/7gAOQWRvYmUAZAAAAAAC/9sAQwABAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEB/9sAQwEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB"
    "AQEBAQEBAQEBAQEBAQEBAQEBAQEBAQEB/8AAFAgAEAAQBAERAAIRAQMRAQQRAP/EAB8AAAEFAQEBAQEBAAAAAAAA"
    "AAABAgMEBQYHCAkKC//EALUQAAIBAwMCBAMFBQQEAAABfQECAwAEEQUSITFBBhNRYQcicRQygZGhCCNCscEVUtHw"
    "JDNicoIJChYXGBkaJSYnKCkqNDU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6g4SFhoeIiYqS"
    "k5SVlpeYmZqio6Slpqeoqaqys7S1tre4ubrCw8TFxsfIycrS09TV1tfY2drh4uPk5ebn6Onq8fLz9PX29/j5+v/E"
    "AB8BAAMBAQEBAQEBAQEAAAAAAAABAgMEBQYHCAkKC//EALURAAIBAgQEAwQHBQQEAAECdwABAgMRBAUhMQYSQVEH"
    "YXETIjKBCBRCkaGxwQkjM1LwFWJy0QoWJDThJfEXGBkaJicoKSo1Njc4OTpDREVGR0hJSlNUVVZXWFlaY2RlZmdo"
    "aWpzdHV2d3h5eoKDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uLj"
    "5OXm5+jp6vLz9PX29/j5+v/aAA4EAQACEQMRBAAAPwD+h/8AaL/ap/4/8al/z0/5bfX3/wDrg8d6/D6Pift/tH/k"
    "339fvXzR6XiN4W/x/wDZu/2PXy1/8m/z/kz/AGW/DX/IO/d/88u30r8F/wBov9qn/j+/4mX/AD0/5a/X3H4dweOl"
    "etQ8T9v9o/8AJ/8Ag/c+vqj+CfEbwt/j/wCzdJ/Y830t+i9e/wCmH7VWt23gX9izx1ajXdQ8O634/wBQ8GeAPDja"
    "Y2qwXOs3OqeI9P1nxL4ekvNLjK2un6t8PdA8Zw6smpz2ulappS3uhXMlxLqtvp97xfxu+NHibX4r64huI7C0eOSS"
    "K71S4ktYrgMsTx/Z40imupo5YplkiuVtzZuqOFufMXYfg/o7/Qv+lH9IjKqfEuRcOZfwBwbi8vhmGT8YeKuMzPhX"
    "KuI6dfDZLj8ufDuW4PJ874szfL83yrO6OaZPxRheHJcG4/D4PMcPQ4klmOHhga3+9/i3w5kmUzxFGqpYrEKcoVMP"
    "gacK9Sk06sZ+2nKdKhCdOdN06lGVZ4iMpQbo8jc18n/st+Gv+Qd+7/55dvpX48/HD4peHtNe5n1DVZ9auY/NJS4k"
    "WDT2kWcSRSfYomeR9saLDJDdXl3azbpWkgKuiRf7jeAn7NH6OHhfjMl4h8R854g8ceJ8r5q1XB8RwwvD/hxWzHD5"
    "9QzXKcyhwLllTF5hi/qmX4Shk+ZZHxVxjxZwrn1LFZvWzHIJ0Mbg8Dlf+Yfizleb42OMo4PB0stoz5kpUoyq4xQd"
    "OUKkHiZxjGPNOTqQq0MNh69JxpqFW8ZSn6h/wUf1u5i0r9nH4WWOu6ebS/1Dxf4/8UeFI20qbVY7nSbbRPDvgTxD"
    "exmOTW9O0+WLWviNpumOktrpWtXMOrLJHf3egRNp3//Z"
)

# Generated once with nvImageCodec 0.9.0.20 and nvJPEG2000 0.11.0.51
# from the 8x8 lossless-HT source pixels reconstructed in the test below.
_HTJ2K_J2K = base64.b64decode(
    "/0//UQAvQAAAAAAIAAAACAAAAAAAAAAAAAAACAAAAAgAAAAAAAAAAAADBwEBBwEBBwEB/1AACAACAAAAA/9SAAwAAgA"
    "BAAMEBEAB/1wADUBASEhQSEhQSEhQ/5AACgAAAAAAtgAA/5PAJbUBB3QAwCWZAQd0AMAl2QEHdADAEsASgPQAh3QA+gD"
    "HdADAEsASgOwAB3QAmgEHdADAEsASgPoAx3QA7AAHdADAEsASABgAAnQAzBNTAMASwBMARADCdABK5QAjVADAEsASgOQ"
    "AQnQAlP4zUwDAE8ATAIiIoKfCdQAE/tXstADAE8AVIADwoKdCdQAgAODAAAHstgDAE0ATgACgKeJ1AAjgwEnstQD/2Q=="
)
_HTJ2K_JP2 = base64.b64decode(
    "AAAADGpQICANCocKAAAAFGZ0eXBqcDIgAAAAAGpwMiAAAAAtanAyaAAAABZpaGRyAAAACAAAAAgAAwcHAAAAAAAPY29scgE"
    "AAAAAABAAAAEaanAyY/9P/1EAL0AAAAAACAAAAAgAAAAAAAAAAAAAAAgAAAAIAAAAAAAAAAAAAwcBAQcBAQcBAf9QAAgAAgA"
    "AAAP/UgAMAAIAAQADBARAAf9cAA1AQEhIUEhIUEhIUP+QAAoAAAAAALYAAP+TwCW1AQd0AMAlmQEHdADAJdkBB3QAwBLAEo"
    "D0AId0APoAx3QAwBLAEoDsAAd0AJoBB3QAwBLAEoD6AMd0AOwAB3QAwBLAEgAYAAJ0AMwTUwDAEsATAEQAwnQASuUAI1QAw"
    "BLAEoDkAEJ0AJT+M1MAwBPAEwCIiKCnwnUABP7V7LQAwBPAFSAA8KCnQnUAIADgwAAB7LYAwBNAE4AAoCnidQAI4MBJ7LUA"
    "/9k="
)


@requires_cuda
def test_nvimagecodec_real_jpeg_native_batches_match_pillow():
    pytest.importorskip("nvidia.nvimgcodec")

    height, width = 97, 193
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(
        ((x * 3) % 256, (y * 5) % 256, (x + y * 2) % 256),
        axis=-1,
    ).astype(np.uint8)
    rgb = Image.fromarray(pixels)
    grayscale = Image.fromarray(((x * 3 + y * 5) % 256).astype(np.uint8))
    cmyk = rgb.convert("CMYK")
    data = [
        _encode(rgb, "JPEG", quality=95, subsampling=subsampling, progressive=prog)
        for subsampling, prog in [
            (0, False),
            (1, False),
            (2, False),
            (0, True),
            (1, True),
            (2, True),
        ]
    ]
    data.extend(
        [
            _encode(grayscale, "JPEG", quality=95),
            _encode(cmyk, "JPEG", quality=95),
            _encode(rgb, "JPEG", quality=80, subsampling=2),
            _encode(rgb, "JPEG", quality=85, subsampling=1),
            _encode(rgb, "JPEG", quality=90, subsampling=0),
            _encode(grayscale, "JPEG", quality=85, progressive=True),
        ]
    )
    data.extend(data[:10])
    expected = [
        np.array(_pillow_decode(encoded, "RGB"), dtype=np.int16) for encoded in data
    ]
    memory_pool = MultiModalGPUMemoryPool(len(data) * width * height * 3 + 1)
    set_mm_gpu_ipc_pool(memory_pool)
    try:
        with _fresh_decoder_pool():
            # Reuse the retained decoder and its four ring streams across more
            # than one ringful of chunks, repeatedly. nvImageCodec has fixed
            # back-to-back parameter races in the past, so a single decode call
            # is not sufficient lifetime coverage for the serving path.
            for _ in range(8):
                actual = decode_images_nvimagecodec(
                    data,
                    batch_size=5,
                    pipeline_depth=4,
                )

                assert memory_pool.available_bytes == memory_pool.total_bytes
                assert all(image is not None for image in actual)
                for decoded, expected_pixels in zip(actual, expected):
                    assert decoded is not None
                    np.testing.assert_allclose(
                        np.asarray(decoded, dtype=np.int16),
                        expected_pixels,
                        atol=6,
                    )
                    decoded.close()
    finally:
        set_mm_gpu_ipc_pool(None)


def test_nvimagecodec_real_cpu_codec_batch_preserves_rgb_and_rgba():
    pytest.importorskip("nvidia.nvimgcodec")

    rgb = Image.new("RGB", (65, 33), (10, 20, 30))
    rgba = Image.new("RGBA", (63, 35), (40, 50, 60, 70))
    pbm = Image.new("1", (61, 31), 1)
    pgm = Image.new("L", (64, 32), 127)
    data = [
        _encode(rgb, "BMP"),
        _encode(rgb, "PNG"),
        _encode(pbm, "PPM"),
        _encode(pgm, "PPM"),
        _encode(rgb, "PPM"),
        _encode(rgb, "WEBP", lossless=True),
        _encode(rgba, "PNG"),
        _encode(rgba, "WEBP", lossless=True),
    ]
    modes = ["RGB", "RGB", "RGB", "RGB", "RGB", "RGB", "RGBA", "RGBA"]
    assert [encoded[:2] for encoded in data[2:5]] == [b"P4", b"P5", b"P6"]

    with _fresh_decoder_pool():
        actual = decode_images_nvimagecodec(
            data,
            output_modes=modes,
            batch_size=5,
        )

    for encoded, mode, decoded in zip(data, modes, actual):
        assert decoded is not None
        expected = _pillow_decode(encoded, mode)
        np.testing.assert_array_equal(np.asarray(decoded), np.asarray(expected))


@requires_cuda
def test_nvimagecodec_real_heterogeneous_batch_keeps_positions():
    pytest.importorskip("nvidia.nvimgcodec")

    data = [
        _encode(Image.new("RGB", (31, 17), (1, 2, 3)), "JPEG", subsampling=2),
        _encode(Image.new("RGBA", (29, 19), (4, 5, 6, 7)), "PNG"),
        _encode(Image.new("RGB", (27, 21), (8, 9, 10)), "WEBP", lossless=True),
        b"not an image",
        _encode(Image.new("RGB", (25, 23), (11, 12, 13)), "JPEG", subsampling=0),
    ]
    modes = ["RGB", "RGBA", "RGB", "RGB", "RGB"]
    raw_bytes = sum(
        width * height * channels
        for width, height, channels in [(31, 17, 3), (25, 23, 3)]
    )
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(raw_bytes + 1))
    try:
        with _fresh_decoder_pool():
            actual = decode_images_nvimagecodec(
                data,
                output_modes=modes,
                batch_size=5,
            )
    finally:
        set_mm_gpu_ipc_pool(None)

    assert [image is not None for image in actual] == [
        True,
        True,
        True,
        False,
        True,
    ]
    for index in (0, 1, 2, 4):
        assert actual[index] is not None
        expected = _pillow_decode(data[index], modes[index])
        np.testing.assert_allclose(
            np.asarray(actual[index], dtype=np.int16),
            np.asarray(expected, dtype=np.int16),
            atol=6 if index in (0, 4) else 0,
        )


@requires_cuda
@pytest.mark.skipif(
    not features.check("jpg_2000"),
    reason="Pillow was built without JPEG 2000",
)
def test_nvimagecodec_real_jpeg2000_batch_matches_pillow():
    pytest.importorskip("nvidia.nvimgcodec")

    height, width = 73, 131
    y, x = np.mgrid[:height, :width]
    pixels = np.stack(
        ((x * 7) % 256, (y * 11) % 256, (x + y * 3) % 256),
        axis=-1,
    ).astype(np.uint8)
    image = Image.fromarray(pixels)
    data = [
        _encode(image, "JPEG2000", irreversible=False),
        _encode(image, "JPEG2000", irreversible=True),
        _encode(image, "JPEG2000", irreversible=False, no_jp2=True),
    ]
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(len(data) * width * height * 3 + 1))
    try:
        with _fresh_decoder_pool():
            actual = decode_images_nvimagecodec(data)
    finally:
        set_mm_gpu_ipc_pool(None)

    assert all(image is not None for image in actual)
    for encoded, decoded in zip(data, actual):
        assert decoded is not None
        expected = _pillow_decode(encoded, "RGB")
        np.testing.assert_allclose(
            np.asarray(decoded, dtype=np.int16),
            np.asarray(expected, dtype=np.int16),
            atol=2,
        )


@requires_cuda
@pytest.mark.skipif(
    not _has_distribution(
        "nvidia-nvjpeg2k-cu12",
        "nvidia-nvjpeg2k-cu13",
        "nvidia-nvjpeg2k-tegra-cu12",
    ),
    reason="nvJPEG2000 optional dependency is not installed",
)
def test_nvimagecodec_htj2k_j2k_and_jp2_reach_media_io_native_batch(
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("nvidia.nvimgcodec")

    height = width = 8
    y, x = np.mgrid[:height, :width]
    expected = np.stack(
        (
            (x * 13 + y * 3) % 256,
            (y * 17 + x * 5) % 256,
            (x * 3 + y * 5) % 256,
        ),
        axis=-1,
    ).astype(np.uint8)
    encoded = [_HTJ2K_J2K, _HTJ2K_JP2]

    def fail_pillow_pixel_decode(*args, **kwargs):
        raise AssertionError("HTJ2K pixel decoding unexpectedly reached Pillow")

    monkeypatch.setattr(
        Jpeg2KImagePlugin.Jpeg2KImageFile,
        "load",
        fail_pillow_pixel_decode,
    )

    raw_bytes = len(encoded) * width * height * 3
    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(raw_bytes + 1))
    try:
        with _fresh_decoder_pool():
            loaded = ImageMediaIO(
                backend="nvimagecodec",
                image_mode="RGB",
            ).load_bytes_many(encoded)
    finally:
        set_mm_gpu_ipc_pool(None)

    assert [item.io_config for item in loaded] == [
        {"backend": "nvimagecodec"},
        {"backend": "nvimagecodec"},
    ]
    for item in loaded:
        np.testing.assert_array_equal(np.asarray(item.media), expected)


@requires_cuda
def test_nvimagecodec_adobe_ycck_matches_pillow():
    pytest.importorskip("nvidia.nvimgcodec")

    with Image.open(BytesIO(_YCCK_JPEG)) as image:
        assert image.info["adobe_transform"] == 2
        expected = np.asarray(image.convert("RGB"), dtype=np.int16)

    set_mm_gpu_ipc_pool(MultiModalGPUMemoryPool(16 * 16 * 3 + 1))
    try:
        with _fresh_decoder_pool():
            loaded = ImageMediaIO(backend="nvimagecodec", image_mode="RGB").load_bytes(
                _YCCK_JPEG
            )
    finally:
        set_mm_gpu_ipc_pool(None)

    assert loaded.io_config == {"backend": "nvimagecodec"}
    np.testing.assert_allclose(
        np.asarray(loaded.media, dtype=np.int16),
        expected,
        atol=6,
    )


@requires_cuda
@pytest.mark.skipif(
    not _has_distribution(
        "nvidia-nvtiff-cu12",
        "nvidia-nvtiff-cu13",
        "nvidia-nvtiff-tegra-cu12",
    ),
    reason="nvTIFF optional dependency is not installed",
)
def test_nvimagecodec_real_tiff_batch_uses_first_page():
    pytest.importorskip("nvidia.nvimgcodec")

    first = Image.new("RGB", (79, 41), (10, 20, 30))
    second = Image.new("RGB", (37, 23), (40, 50, 60))
    multipage = _encode(
        first,
        "TIFF",
        save_all=True,
        append_images=[second],
        compression="tiff_lzw",
    )
    data = [
        _encode(first, "TIFF", compression="raw"),
        _encode(first, "TIFF", compression="tiff_lzw"),
        _encode(first, "TIFF", compression="tiff_deflate"),
        _encode(first, "TIFF", compression="jpeg"),
        multipage,
    ]
    set_mm_gpu_ipc_pool(
        MultiModalGPUMemoryPool(len(data) * first.width * first.height * 3 + 1)
    )
    try:
        with _fresh_decoder_pool():
            actual = decode_images_nvimagecodec(data)
    finally:
        set_mm_gpu_ipc_pool(None)

    assert all(image is not None for image in actual)
    for encoded, decoded in zip(data, actual):
        assert decoded is not None
        assert decoded.size == first.size
        expected = _pillow_decode(encoded, "RGB")
        np.testing.assert_allclose(
            np.asarray(decoded, dtype=np.int16),
            np.asarray(expected, dtype=np.int16),
            atol=6,
        )


@requires_cuda
@pytest.mark.skipif(
    not features.check("jpg_2000"),
    reason="Pillow was built without JPEG 2000",
)
def test_nvimagecodec_high_precision_stream_is_not_implicitly_rescaled():
    pytest.importorskip("nvidia.nvimgcodec")

    pixels = np.arange(31 * 17, dtype=np.uint16).reshape(17, 31) * 97
    data = _encode(Image.fromarray(pixels), "JPEG2000", irreversible=False)

    with _fresh_decoder_pool():
        assert decode_images_nvimagecodec([data]) == [None]


@requires_cuda
def test_nvimagecodec_two_real_native_batches_can_run_concurrently():
    pytest.importorskip("nvidia.nvimgcodec")

    image = Image.new("RGB", (193, 97), (10, 20, 30))
    data = [_encode(image, "JPEG", subsampling=2)] * 5
    per_batch_bytes = len(data) * image.width * image.height * 3
    memory_pool = MultiModalGPUMemoryPool(per_batch_bytes * 2 + 1)
    set_mm_gpu_ipc_pool(memory_pool)
    try:
        with _fresh_decoder_pool(), ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(
                    decode_images_nvimagecodec,
                    data,
                    decoders=2,
                    batch_size=5,
                )
                for _ in range(2)
            ]
            results = [future.result(timeout=30) for future in futures]
    finally:
        set_mm_gpu_ipc_pool(None)

    assert all(image is not None for batch in results for image in batch)
    assert memory_pool.available_bytes == memory_pool.total_bytes
