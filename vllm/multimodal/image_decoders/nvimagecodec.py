# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib.metadata
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from PIL import Image

from vllm.multimodal.image_decoders import NVIMAGECODEC_DECODER_THREADS
from vllm.utils.mem_constants import MiB_bytes

NVIMAGECODEC_MAX_ENCODED_BYTES = 64 * MiB_bytes

_MISSING_PACKAGE_ERROR = (
    "nvImageCodec requires the CUDA-matched nvidia-nvimgcodec package."
)
_owner = threading.local()
# Bound native concurrency to vLLM's default media-loading concurrency. Each
# owner thread reuses its own decoder because the decoder is not thread-safe.
_executor = ThreadPoolExecutor(
    max_workers=NVIMAGECODEC_DECODER_THREADS,
    thread_name_prefix="vllm-nvimagecodec",
)


def ensure_nvimagecodec_available() -> None:
    """Check CUDA-matched package availability without initializing CUDA."""
    cuda_major = (torch.version.cuda or "").partition(".")[0]
    packages = (
        f"nvidia-nvimgcodec-cu{cuda_major}",
        "nvidia-nvjpeg-cu12" if cuda_major == "12" else "nvidia-nvjpeg",
    )
    try:
        for package in packages:
            importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError(_MISSING_PACKAGE_ERROR) from exc


def _load_nvimgcodec():
    try:
        from nvidia import nvimgcodec
    except ImportError as exc:
        raise RuntimeError(_MISSING_PACKAGE_ERROR) from exc
    return nvimgcodec


def _get_decoder(nvimgcodec):
    decoder = getattr(_owner, "decoder", None)
    if decoder is None or getattr(_owner, "module", None) is not nvimgcodec:
        _owner.__dict__.clear()
        backends = [
            nvimgcodec.Backend(nvimgcodec.BackendKind.HW_GPU_ONLY),
            nvimgcodec.Backend(nvimgcodec.BackendKind.GPU_ONLY),
            nvimgcodec.Backend(nvimgcodec.BackendKind.HYBRID_CPU_GPU),
        ]
        decoder = nvimgcodec.Decoder(
            device_id=0,
            max_num_cpu_threads=1,
            backends=backends,
        )
        _owner.decoder = decoder
        _owner.module = nvimgcodec
        _owner.params = nvimgcodec.DecodeParams(
            allow_any_depth=False,
            apply_exif_orientation=False,
            color_spec=nvimgcodec.ColorSpec.SRGB,
            sample_format=nvimgcodec.SampleFormat.I_RGB,
        )
    return decoder, _owner.params


def _decode_on_owner(data: bytes, width: int, height: int) -> Image.Image:
    nvimgcodec = _load_nvimgcodec()
    stream = nvimgcodec.CodeStream(data)
    if (
        str(stream.codec_name).lower() != "jpeg"
        or int(stream.precision) != 8
        or int(stream.num_channels) != 3
        or (int(stream.width), int(stream.height)) != (width, height)
    ):
        raise ValueError("nvImageCodec JPEG metadata did not match Pillow")

    try:
        decoder, params = _get_decoder(nvimgcodec)
        decoded = decoder.decode(stream, params=params)
        if decoded is None:
            raise RuntimeError("nvImageCodec did not return a decoded image")
        host = decoded.cpu()
        if host is None:
            raise RuntimeError("nvImageCodec failed to copy an image to host")
        array = np.asarray(host)
        if array.dtype != np.uint8 or array.shape != (height, width, 3):
            raise RuntimeError(
                "nvImageCodec returned an unexpected host image: "
                f"shape={array.shape}, dtype={array.dtype}"
            )
        return Image.frombytes("RGB", (width, height), np.ascontiguousarray(array))
    except Exception:
        _owner.__dict__.clear()
        raise


def decode_image_nvimagecodec(
    data: bytes,
    *,
    width: int,
    height: int,
) -> Image.Image:
    """Decode one JPEG to an owned RGB image."""
    if len(data) > NVIMAGECODEC_MAX_ENCODED_BYTES:
        raise ValueError("nvImageCodec JPEG inputs are limited to 64 MiB")
    try:
        return _executor.submit(_decode_on_owner, data, width, height).result()
    except ValueError:
        raise
    except Exception as exc:
        raise RuntimeError(
            f"nvImageCodec failed to decode the JPEG ({type(exc).__name__})"
        ) from exc
