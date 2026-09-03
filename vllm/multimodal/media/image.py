# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import contextlib
import os
import threading
from collections.abc import Callable
from concurrent.futures import Executor, Future
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pybase64
import torch
from PIL import Image

import vllm.envs as envs
from vllm.multimodal.image_decoders.nvimagecodec import (
    NVIMAGECODEC_IMAGE_BACKEND,
    PILLOW_IMAGE_BACKEND,
    NvImageCodecInput,
    NvImageCodecResult,
    _PinnedImageLease,
    create_nvimagecodec_decode_service,
    preflight_image_nvimagecodec,
    validate_image_backend,
)
from vllm.utils.serial_utils import tensor2base64
from vllm.utils.sparse_utils import (
    check_sparse_tensor_invariants_threadsafe,
    safe_to_dense,
)

from ..image import convert_image_mode, normalize_image, rgba_to_rgb
from .base import MediaIO, MediaWithBytes

MAGIC_NUMPY_PREFIX = b"\x93NUMPY"  # https://numpy.org/devdocs/reference/generated/numpy.lib.format.html#format-version-1-0

DecodedImage: TypeAlias = Image.Image | _PinnedImageLease
LoadedImage: TypeAlias = MediaWithBytes[DecodedImage]


def _release_abandoned_nvimagecodec_result(
    future: Future[NvImageCodecResult],
) -> None:
    if future.cancelled():
        return
    try:
        result = future.result()
    except BaseException:
        return
    if isinstance(result, _PinnedImageLease):
        result.release()


@dataclass
class _NvImageCodecState:
    api_process_count: int
    device_index: int
    service: Any
    users: int = 1


_nvimagecodec_state: _NvImageCodecState | None = None
_nvimagecodec_service_lock = threading.Lock()


def _get_nvimagecodec_decode_service():
    """Return the service created during post-fork server initialization."""
    state = _nvimagecodec_state
    if state is None:
        raise RuntimeError(
            "The nvImageCodec image backend was not initialized in this process."
        )
    return state.service


def initialize_nvimagecodec_decode_service(
    api_process_count: int = 1,
    device_index: int = 0,
) -> Callable[[], None]:
    """Acquire the process-local decoder and return an idempotent release."""
    global _nvimagecodec_state

    with _nvimagecodec_service_lock:
        if _nvimagecodec_state is not None:
            if (
                _nvimagecodec_state.api_process_count != api_process_count
                or _nvimagecodec_state.device_index != device_index
            ):
                raise RuntimeError(
                    "The nvImageCodec image backend was already initialized "
                    "with different process topology."
                )
            _nvimagecodec_state.users += 1
        else:
            service = create_nvimagecodec_decode_service(
                api_process_count, device_index=device_index
            )
            try:
                service.wait_until_ready()
            except BaseException:
                with contextlib.suppress(BaseException):
                    service.close()
                raise
            _nvimagecodec_state = _NvImageCodecState(
                api_process_count, device_index, service
            )
        acquired_state = _nvimagecodec_state

    released = False

    def release() -> None:
        nonlocal released
        global _nvimagecodec_state

        with _nvimagecodec_service_lock:
            if released:
                return
            released = True
            state = _nvimagecodec_state
            if state is not acquired_state:
                return
            state.users -= 1
            if state.users == 0:
                _nvimagecodec_state = None
                state.service.close()

    return release


def initialize_image_decode_backend(
    vllm_config: Any,
) -> Callable[[], None] | None:
    """Acquire a configured GPU image backend after process creation."""
    mm_config = vllm_config.model_config.multimodal_config
    if mm_config is not None and mm_config.use_gpu_image_backend():
        parallel_config = vllm_config.parallel_config
        return initialize_nvimagecodec_decode_service(
            parallel_config._api_process_count,
            device_index=_nvimagecodec_device_index(
                parallel_config.assigned_physical_gpu_ids
            ),
        )
    return None


def _nvimagecodec_device_index(assigned_physical_gpu_ids: list[int] | None) -> int:
    if not assigned_physical_gpu_ids:
        return 0

    from vllm.platforms import current_platform

    physical_id = assigned_physical_gpu_ids[0]
    visible = os.environ.get(current_platform.device_control_env_var, "")
    if not visible:
        return physical_id
    visible_physical_ids = [
        current_platform.device_control_id_to_physical_device_id(item)
        for item in visible.split(",")
    ]
    if physical_id not in visible_physical_ids:
        raise RuntimeError(
            f"Assigned physical GPU {physical_id} is not visible in "
            f"{current_platform.device_control_env_var}={visible}."
        )
    return visible_physical_ids.index(physical_id)


def _reset_nvimagecodec_decode_service_after_fork() -> None:
    global _nvimagecodec_state, _nvimagecodec_service_lock

    # Only the thread that called fork survives in the child. Never inherit a
    # service owner thread or a lock that another parent thread may have held.
    _nvimagecodec_state = None
    _nvimagecodec_service_lock = threading.Lock()


os.register_at_fork(after_in_child=_reset_nvimagecodec_decode_service_after_fork)


class ImageMediaIO(MediaIO[LoadedImage]):
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
        if runtime_kwargs and "backend" in runtime_kwargs:
            runtime_kwargs = dict(runtime_kwargs)
            requested = runtime_kwargs.pop("backend")
            configured = (default_kwargs or {}).get("backend", PILLOW_IMAGE_BACKEND)
            if requested != configured:
                raise ValueError(
                    f"Image backend is fixed at startup as {configured!r}, "
                    f"not {requested!r}."
                )
        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        merged.pop("_borrow_output", None)
        return merged

    def __init__(
        self,
        image_mode: str | None = "RGB",
        *,
        backend: str = PILLOW_IMAGE_BACKEND,
        _borrow_output: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        # Target mode for loaded images; `None` keeps the original mode
        # (i.e. no conversion, alpha channel is preserved as-is).
        self.image_mode = image_mode
        self.backend = validate_image_backend(backend)
        self._borrow_output = _borrow_output
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

    def _load_bytes_pillow(self, data: bytes) -> LoadedImage:
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

    def _prepare_bytes(self, data: bytes) -> LoadedImage | NvImageCodecInput:
        if self.backend == PILLOW_IMAGE_BACKEND or not self._borrow_output:
            return self._load_bytes_pillow(data)

        prepared = preflight_image_nvimagecodec(
            data,
            image_mode=self.image_mode,
        )
        if prepared is None:
            return self._load_bytes_pillow(data)
        return prepared

    def load_bytes(self, data: bytes) -> LoadedImage:
        return self._load_bytes_pillow(data)

    async def load_bytes_async(
        self,
        data: bytes,
        *,
        executor: Executor | None = None,
    ) -> LoadedImage:
        loop = asyncio.get_running_loop()
        prepared = await loop.run_in_executor(executor, self._prepare_bytes, data)
        if isinstance(prepared, MediaWithBytes):
            return prepared

        future = _get_nvimagecodec_decode_service().submit(prepared)
        try:
            lease = await asyncio.wrap_future(future)
        except BaseException:
            future.add_done_callback(_release_abandoned_nvimagecodec_result)
            raise
        if lease is None:
            return await loop.run_in_executor(executor, self._load_bytes_pillow, data)
        return MediaWithBytes(lease, data, {"backend": NVIMAGECODEC_IMAGE_BACKEND})

    async def load_base64_async(
        self,
        media_type: str,
        data: str,
        *,
        executor: Executor | None = None,
    ) -> LoadedImage:
        loop = asyncio.get_running_loop()
        encoded = await loop.run_in_executor(
            executor,
            lambda: pybase64.b64decode(data, validate=True),
        )
        return await self.load_bytes_async(encoded, executor=executor)

    async def load_file_async(
        self,
        filepath: Path,
        *,
        executor: Executor | None = None,
    ) -> LoadedImage:
        loop = asyncio.get_running_loop()
        data = await loop.run_in_executor(executor, filepath.read_bytes)
        return await self.load_bytes_async(data, executor=executor)

    def load_base64(self, media_type: str, data: str) -> LoadedImage:
        return self.load_bytes(pybase64.b64decode(data, validate=True))

    def load_file(self, filepath: Path) -> LoadedImage:
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
            return safe_to_dense(tensor, parameter="image_embeds")

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
            return safe_to_dense(tensor, parameter="image_embeds")

    def encode_base64(self, media: torch.Tensor) -> str:
        return tensor2base64(media)
