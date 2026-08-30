# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pybase64
import torch
from PIL import Image

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import check_sparse_tensor_invariants_threadsafe

from ..image import convert_image_mode, normalize_image, rgba_to_rgb
from ..image_decoders import (
    NVIMGCODEC_BACKEND,
    PILLOW_BACKEND,
    decode_batch,
    output_layout,
)
from .base import MediaIO, MediaWithBytes

logger = init_logger(__name__)

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0

# Settings that shape retained, process-wide resources. A request must never be
# able to change these, so they are stripped from request-level kwargs.
_STARTUP_ONLY_KWARGS = (
    "image_backend",
    "num_decoders",
    "image_output",
    "coalesce_width",
    "min_gpu_pixels",
    "gpu_resize",
    "resize_prefilter",
    "min_resize_ratio",
)


def _as_pil(array, source_layout: str) -> Image.Image:
    """Wrap a decoded array as a PIL image, given the layout it is actually in.

    The layout is negotiated once at startup and is known to every caller, so it
    is passed in rather than inferred. Inferring it from shape is wrong for real
    images: a CHW array whose width is 1, 3 or 4 is indistinguishable from HWC
    by shape alone. Guessing raised "Cannot handle this data type" at width 1
    and silently produced wrong dimensions -- and an RGBA mode -- at widths 3
    and 4. Only reachable when a deployment overrides image_output="pil" while
    a raw-array layout was negotiated.
    """
    if source_layout == "device":
        # Device mode is only negotiated when the processor itself runs on the
        # accelerator, so materializing a PIL image here would undo the copy the
        # mode exists to avoid. It is still supported rather than refused,
        # because image_output="pil" is a documented escape hatch.
        array = array.permute(1, 2, 0).cpu().numpy()
    elif source_layout == "chw":
        array = np.ascontiguousarray(array.transpose(1, 2, 0))
    return Image.fromarray(array)


class ImageMediaIO(MediaIO[Image.Image]):
    """Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
        *,
        trusted: bool = False,
    ) -> dict[str, Any]:
        """Merge, stripping startup-only keys from untrusted overrides.

        ``trusted=True`` means the caller has already applied the request trust
        boundary and is re-merging its own authorized result. Renderers that
        fold server configuration into the override position (Kimi-K3 does)
        must say so, or this guard silently deletes the server's own settings
        and the GPU backend never activates.
        """
        if runtime_kwargs and not trusted:
            stripped = [key for key in _STARTUP_ONLY_KWARGS if key in runtime_kwargs]
            if stripped:
                # A request cannot select a GPU backend whose device memory was
                # not reserved at startup, nor resize the decoder pool.
                runtime_kwargs = {
                    key: value
                    for key, value in runtime_kwargs.items()
                    if key not in _STARTUP_ONLY_KWARGS
                }
                logger.warning_once(
                    "Ignoring startup-only image media_io_kwargs from a request: %s",
                    ", ".join(stripped),
                )
        return super().merge_kwargs(default_kwargs, runtime_kwargs, trusted=trusted)

    def __init__(self, image_mode: str | None = "RGB", **kwargs) -> None:
        super().__init__()

        # Which decoder this instance may use. Pillow is the default and always
        # the fallback; the GPU backend is opt-in and never activates itself.
        self.image_backend = kwargs.pop("image_backend", PILLOW_BACKEND)
        if self.image_backend not in (PILLOW_BACKEND, NVIMGCODEC_BACKEND):
            raise ValueError(
                f"Unknown image_backend {self.image_backend!r}; expected "
                f"{PILLOW_BACKEND!r} or {NVIMGCODEC_BACKEND!r}."
            )
        # "array" hands back the decoded raster directly, which is most of why
        # the GPU path is worth having: a PIL image would be copied into
        # PIL-owned memory and then copied straight back out by the HF
        # processor. "pil" is the escape hatch for a processor that duck-types
        # PIL.Image instead of honouring vLLM's declared HfImageItem union.
        self.image_output = kwargs.pop("image_output", "array")
        if self.image_output not in ("array", "pil"):
            raise ValueError("image_output must be 'array' or 'pil'")

        # Target mode for loaded images; `None` keeps the original mode
        # (i.e. no conversion, alpha channel is preserved as-is).
        self.image_mode = image_mode
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.
        self.kwargs = kwargs

        # Extract RGBA background color from kwargs if provided
        # Default to white background for backward compatibility
        rgba_bg = kwargs.get("rgba_background_color", (255, 255, 255))
        # Convert list to tuple for consistency
        if isinstance(rgba_bg, list):
            rgba_bg = tuple(rgba_bg)

        # Validate rgba_background_color format
        if not (
            isinstance(rgba_bg, tuple)
            and len(rgba_bg) == 3
            and all(isinstance(c, int) and 0 <= c <= 255 for c in rgba_bg)
        ):
            raise ValueError(
                "rgba_background_color must be a list or tuple of 3 integers "
                "in the range [0, 255]."
            )
        self.rgba_background_color = rgba_bg

    def _convert_image_mode(
        self, image: Image.Image | MediaWithBytes[Image.Image]
    ) -> Image.Image:
        """Convert image mode with custom background color."""
        if isinstance(image, MediaWithBytes):
            image = image.media
        if self.image_mode is None or image.mode == self.image_mode:
            return image
        elif image.mode == "RGBA" and self.image_mode == "RGB":
            return rgba_to_rgb(image, self.rgba_background_color)
        else:
            return convert_image_mode(
                image, self.image_mode, self.rgba_background_color
            )

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image | np.ndarray]:
        if self.image_backend == NVIMGCODEC_BACKEND:
            # decode_batch never raises and never returns wrong pixels: it
            # returns None for anything it cannot prove matches Pillow, which
            # includes every failure. So there is nothing to catch here, and a
            # fallback is indistinguishable from the default path.
            array = decode_batch([data], self.image_mode)[0]
            if array is not None:
                # The decoder emits whatever layout the startup capability probe
                # proved this processor accepts bit-exactly. "pil" means no
                # bypass was provable, so re-wrap and keep the old contract.
                # An explicit image_output="pil" overrides the probe.
                decoder_layout = output_layout()
                layout = "pil" if self.image_output == "pil" else decoder_layout
                media = (
                    _as_pil(array, decoder_layout) if layout == "pil" else array
                )
                # Record the backend so the multimodal cache cannot serve GPU
                # pixels for a Pillow key or vice versa; the two agree only
                # within the documented lossy-codec tolerance, not bit-exactly.
                return MediaWithBytes(
                    media,
                    data,
                    {
                        "image_mode": self.image_mode,
                        "rgba_background_color": self.rgba_background_color,
                        "image_backend": NVIMGCODEC_BACKEND,
                    },
                )
        return self._load_bytes_pillow(data)

    def load_bytes_many(
        self, datas: Sequence[bytes]
    ) -> list[MediaWithBytes[Image.Image | np.ndarray]]:
        """Decode several images, preserving input order.

        Multi-image requests and ``video/jpeg`` frame batches arrive together,
        so decoding them in one native call is strictly better than N calls:
        it is the only place native batch width above one occurs in practice.
        Positions the accelerator declines fall back to Pillow individually, so
        a mixed batch behaves exactly like N independent ``load_bytes`` calls.
        """
        if not datas:
            return []
        arrays: list[np.ndarray | None] = [None] * len(datas)
        if self.image_backend == NVIMGCODEC_BACKEND:
            arrays = decode_batch(list(datas), self.image_mode)

        results: list[MediaWithBytes[Image.Image | np.ndarray]] = []
        decoder_layout = output_layout()
        layout = "pil" if self.image_output == "pil" else decoder_layout
        for data, array in zip(datas, arrays):
            if array is None:
                results.append(self._load_bytes_pillow(data))
                continue
            media = _as_pil(array, decoder_layout) if layout == "pil" else array
            results.append(
                MediaWithBytes(
                    media,
                    data,
                    {
                        "image_mode": self.image_mode,
                        "rgba_background_color": self.rgba_background_color,
                        "image_backend": NVIMGCODEC_BACKEND,
                    },
                )
            )
        return results

    def _load_bytes_pillow(self, data: bytes) -> MediaWithBytes[Image.Image]:
        try:
            image = Image.open(BytesIO(data))
            w, h = image.size
            max_pixels = envs.VLLM_MAX_IMAGE_PIXELS
            if max_pixels > 0 and w * h > max_pixels:
                raise ValueError(
                    f"Image dimensions {w}x{h} ({w * h} pixels) exceed "
                    f"the maximum of {max_pixels} pixels. Set "
                    f"VLLM_MAX_IMAGE_PIXELS to increase this limit."
                )
            image = normalize_image(image)
            image.load()
            converted = self._convert_image_mode(image)
        except (OSError, Image.UnidentifiedImageError) as e:
            raise ValueError(f"Failed to load image: {e}") from e

        io_config = None
        if converted is not image:
            io_config = {
                "image_mode": self.image_mode,
                "rgba_background_color": self.rgba_background_color,
            }
        return MediaWithBytes(converted, data, io_config)

    def load_base64(self, media_type: str, data: str) -> MediaWithBytes[Image.Image]:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> MediaWithBytes[Image.Image]:
        return self.load_bytes(filepath.read_bytes())

    def encode_base64(
        self,
        media: Image.Image,
        *,
        image_format: str = "PNG",
    ) -> str:
        image = media

        with BytesIO() as buffer:
            image = self._convert_image_mode(image)
            image.save(buffer, image_format)
            data = buffer.getvalue()

        return pybase64.b64encode(data).decode("utf-8")


class ImageEmbeddingMediaIO(MediaIO[torch.Tensor]):
    """Image embedding MediaIO implementation.

    Configuration values can be user-provided either by --media-io-kwargs or
    by the runtime API field "media_io_kwargs". Ensure proper validation and
    error handling.
    """

    def __init__(self) -> None:
        super().__init__()

    def _load_pickled_torch(self, data: bytes) -> torch.Tensor:
        buffer = BytesIO(data)
        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(buffer, weights_only=True)
            return tensor.to_dense()

    def _load_numpy(self, data: bytes) -> torch.Tensor:
        with BytesIO(data) as buffer:
            return torch.from_numpy(np.load(buffer))

    def load_bytes(self, data: bytes) -> torch.Tensor:
        if data[:6] == MAGIC_NUMPY_PREFIX:
            return self._load_numpy(data)

        return self._load_pickled_torch(data)

    def load_base64(self, media_type: str, data: str) -> torch.Tensor:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> torch.Tensor:
        if filepath.suffix == ".npy":
            return torch.from_numpy(np.load(filepath))

        with check_sparse_tensor_invariants_threadsafe():
            tensor = torch.load(filepath, weights_only=True)
            return tensor.to_dense()

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
