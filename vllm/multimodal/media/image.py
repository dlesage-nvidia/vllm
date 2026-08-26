# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from io import BytesIO
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pybase64
import torch
from PIL import Image

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.multimodal.image_decoders import (
    NVIMAGECODEC_DEFAULT_BATCH_SIZE,
    NVIMAGECODEC_DEFAULT_DECODERS,
    NVIMAGECODEC_IMAGE_BACKEND,
    PILLOW_IMAGE_BACKEND,
    NvImageCodecBatchItemError,
    decode_images_nvimagecodec,
    validate_nvimagecodec_batch_size,
    validate_nvimagecodec_decoders,
)
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import check_sparse_tensor_invariants_threadsafe

from ..image import (
    _has_transparency,
    check_image_pixel_limit,
    convert_image_mode,
    normalize_image,
    rgba_to_rgb,
)
from .base import MediaIO, MediaWithBytes

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0

logger = init_logger(__name__)

_NvImageCodecOutputMode = Literal["RGB", "RGBA"]
_NVIMAGECODEC_PIL_FORMATS = frozenset(
    {"JPEG", "JPEG2000", "TIFF", "BMP", "PNG", "PPM", "WEBP"}
)
_NVIMAGECODEC_ALPHA_FORMATS = frozenset({"PNG", "WEBP"})


class _ImageBatchItemError(ValueError):
    """An image-batch failure whose position must survive async batching."""

    def __init__(self, index: int, error: Exception) -> None:
        super().__init__(str(error))
        self.index = index
        self.error = error


def _indexed_image_error(index: int, error: Exception) -> _ImageBatchItemError:
    if isinstance(error, OSError):
        error = ValueError(f"Failed to load image: {error}")
    return _ImageBatchItemError(index, error)


def _nvimagecodec_output_mode(
    image: Image.Image, target_mode: str | None
) -> _NvImageCodecOutputMode | None:
    image_format = image.format
    if image_format not in _NVIMAGECODEC_PIL_FORMATS:
        return None

    # nvImageCodec scales higher precision samples to uint8, while Pillow's
    # RGB conversion clips some integer and floating-point modes instead.
    bits = getattr(image, "bits", 8) or 8
    if image.mode in ("I", "F") or image.mode.startswith("I;") or int(bits) > 8:
        return None

    # nvImageCodec's still-image decoders do not define animated PNG/WebP
    # compositing semantics. Multi-page TIFF is safe because the decoder
    # explicitly selects the first IFD, matching Pillow's initial frame.
    if image_format in ("PNG", "WEBP") and getattr(image, "is_animated", False):
        return None

    has_transparency = _has_transparency(image)
    if has_transparency and image_format not in _NVIMAGECODEC_ALPHA_FORMATS:
        return None

    if target_mode == "RGB":
        # Retain alpha until _convert_image_mode composites the configured
        # background color.
        return "RGBA" if has_transparency else "RGB"
    if target_mode == "RGBA":
        # The GPU JPEG, JPEG 2000, and TIFF backends reject channel expansion.
        # Decode opaque sources as RGB and add the opaque alpha channel in PIL.
        return "RGBA" if has_transparency else "RGB"
    if target_mode is None:
        if image.mode == "RGB":
            return "RGB"
        if image.mode == "RGBA" and has_transparency:
            return "RGBA"
    return None


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
    ) -> dict[str, Any]:
        if runtime_kwargs:
            runtime_kwargs = dict(runtime_kwargs)
            runtime_kwargs.pop("decoders", None)
            runtime_kwargs.pop("batch_size", None)

            static_backend = (default_kwargs or {}).get("backend")
            if static_backend is None:
                static_backend = envs.VLLM_IMAGE_LOADER_BACKEND
            requested = runtime_kwargs.get("backend")
            if "backend" in runtime_kwargs and requested is None:
                # `None` means "use the environment default" in __init__.
                # Retain the startup value instead so a request cannot reveal
                # a GPU backend hidden by an explicit static Pillow override.
                runtime_kwargs.pop("backend")
            if (
                requested == NVIMAGECODEC_IMAGE_BACKEND
                and static_backend != NVIMAGECODEC_IMAGE_BACKEND
            ):
                logger.warning_once(
                    "Stripping request-level image backend=%r: GPU image "
                    "backend not configured at startup.",
                    requested,
                )
                runtime_kwargs.pop("backend")

        return super().merge_kwargs(default_kwargs, runtime_kwargs)

    def __init__(
        self,
        image_mode: str | None = "RGB",
        *,
        backend: str | None = None,
        decoders: int = NVIMAGECODEC_DEFAULT_DECODERS,
        batch_size: int = NVIMAGECODEC_DEFAULT_BATCH_SIZE,
        **kwargs,
    ) -> None:
        super().__init__()

        # Target mode for loaded images; `None` keeps the original mode
        # (i.e. no conversion, alpha channel is preserved as-is).
        self.image_mode = image_mode
        self.backend = envs.VLLM_IMAGE_LOADER_BACKEND if backend is None else backend
        if self.backend not in (PILLOW_IMAGE_BACKEND, NVIMAGECODEC_IMAGE_BACKEND):
            raise ValueError(
                f"Unknown image backend {self.backend!r}; expected "
                f"{PILLOW_IMAGE_BACKEND!r} or {NVIMAGECODEC_IMAGE_BACKEND!r}."
            )
        self.decoders = (
            validate_nvimagecodec_decoders(decoders)
            if self.backend == NVIMAGECODEC_IMAGE_BACKEND
            else decoders
        )
        self.batch_size = (
            validate_nvimagecodec_batch_size(batch_size)
            if self.backend == NVIMAGECODEC_IMAGE_BACKEND
            else batch_size
        )
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

    def load_bytes_many(
        self, encoded_images: Sequence[bytes]
    ) -> list[MediaWithBytes[Image.Image]]:
        if not encoded_images:
            return []

        images: list[Image.Image] = []
        try:
            for index, data in enumerate(encoded_images):
                try:
                    image = Image.open(BytesIO(data))
                    images.append(image)
                    check_image_pixel_limit(*image.size)
                except Exception as e:
                    raise _indexed_image_error(index, e) from e

            candidate_indices: list[int] = []
            candidate_output_modes: list[_NvImageCodecOutputMode] = []
            if self.backend == NVIMAGECODEC_IMAGE_BACKEND:
                for index, image in enumerate(images):
                    try:
                        output_mode = _nvimagecodec_output_mode(image, self.image_mode)
                    except Exception as e:
                        raise _indexed_image_error(index, e) from e
                    if output_mode is not None:
                        candidate_indices.append(index)
                        candidate_output_modes.append(output_mode)
            used_backends = [PILLOW_IMAGE_BACKEND] * len(images)

            if candidate_indices:
                source_infos = []
                orientations = []
                for index in candidate_indices:
                    try:
                        source_infos.append(images[index].info.copy())
                        orientations.append(
                            images[index].getexif().get(Image.ExifTags.Base.Orientation)
                        )
                    except Exception as e:
                        raise _indexed_image_error(index, e) from e
                try:
                    decoded_images = decode_images_nvimagecodec(
                        [encoded_images[index] for index in candidate_indices],
                        output_modes=candidate_output_modes,
                        decoders=self.decoders,
                        batch_size=self.batch_size,
                    )
                except NvImageCodecBatchItemError as e:
                    raise _indexed_image_error(
                        candidate_indices[e.index], e.error
                    ) from e
                if len(decoded_images) != len(candidate_indices):
                    for decoded in decoded_images:
                        if decoded is not None:
                            decoded.close()
                    raise RuntimeError(
                        "nvImageCodec returned an unexpected number of images: "
                        f"expected {len(candidate_indices)}, got {len(decoded_images)}"
                    )

                try:
                    for index, source_info, orientation, decoded in zip(
                        candidate_indices,
                        source_infos,
                        orientations,
                        decoded_images,
                    ):
                        if decoded is None:
                            continue
                        try:
                            images[index].close()
                            images[index] = decoded
                            decoded.info.update(source_info)
                            if isinstance(orientation, int) and 2 <= orientation <= 8:
                                normalized = normalize_image(decoded)
                                images[index] = normalized
                                if normalized is not decoded:
                                    decoded.close()
                            used_backends[index] = NVIMAGECODEC_IMAGE_BACKEND
                        except Exception as e:
                            raise _indexed_image_error(index, e) from e
                except Exception:
                    for decoded in decoded_images:
                        if decoded is not None:
                            decoded.close()
                    raise
        except Exception:
            for image in images:
                image.close()
            raise

        loaded_images: list[MediaWithBytes[Image.Image]] = []
        created_images: list[Image.Image] = []
        try:
            for index, (image, data, used_backend) in enumerate(
                zip(images, encoded_images, used_backends)
            ):
                try:
                    if used_backend == PILLOW_IMAGE_BACKEND:
                        normalized = normalize_image(image)
                        if normalized is not image:
                            created_images.append(normalized)
                        image = normalized
                        image.load()
                    converted = self._convert_image_mode(image)
                    if converted is not image:
                        created_images.append(converted)

                    io_config: dict[str, Any] | None = None
                    if converted is not image:
                        io_config = {
                            "image_mode": self.image_mode,
                            "rgba_background_color": self.rgba_background_color,
                        }
                    if used_backend == NVIMAGECODEC_IMAGE_BACKEND:
                        io_config = {**(io_config or {}), "backend": used_backend}
                    loaded_images.append(MediaWithBytes(converted, data, io_config))
                except Exception as e:
                    raise _indexed_image_error(index, e) from e
        except Exception:
            for created in created_images:
                created.close()
            for image in images:
                image.close()
            raise
        return loaded_images

    def load_bytes(self, data: bytes) -> MediaWithBytes[Image.Image]:
        try:
            return self.load_bytes_many([data])[0]
        except _ImageBatchItemError as e:
            raise e.error from None

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
