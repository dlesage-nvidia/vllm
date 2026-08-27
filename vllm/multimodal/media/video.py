# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
from concurrent.futures import Executor
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pybase64
from PIL import Image

from vllm import envs
from vllm.logger import init_logger
from vllm.multimodal.image_decoders import NVIMAGECODEC_IMAGE_BACKEND

from ..video import VIDEO_LOADER_REGISTRY
from .base import MediaIO, MediaWithBytes
from .image import ImageMediaIO
from .image_decode_service import (
    load_images_with_service,
    load_images_with_service_async,
    reserve_image_decode_request_async,
)

logger = init_logger(__name__)


def _close_loaded_frames(
    loaded_frames: list[MediaWithBytes[Image.Image]],
) -> None:
    for frame in loaded_frames:
        frame.media.close()


def _consume_future_exception(future: asyncio.Future[Any]) -> None:
    if not future.cancelled():
        future.exception()


class VideoMediaIO(MediaIO[MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]]):
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
            # Decoder GPU memory is reserved from the startup value.
            runtime_kwargs = dict(runtime_kwargs)
            runtime_kwargs.pop("hw_decoders", None)
            runtime_kwargs.pop("pool_size", None)

            # Block request-level selection of GPU video backends that
            # were not configured (and VRAM-reserved) at startup.
            for key in ("video_backend", "backend"):
                requested = runtime_kwargs.get(key)
                if requested and VIDEO_LOADER_REGISTRY.backend_requires_gpu(requested):
                    static_val = (default_kwargs or {}).get(key)
                    if static_val != requested:
                        logger.warning_once(
                            "Stripping request-level %s=%r: GPU video "
                            "backend not configured at startup.",
                            key,
                            requested,
                        )
                        runtime_kwargs = {
                            k: v for k, v in runtime_kwargs.items() if k != key
                        }

        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        # fps and num_frames interact with each other, so if either is
        # overridden at request time, wipe the other from defaults to
        # avoid unintuitive cross-field interactions.
        if runtime_kwargs:
            if "num_frames" in runtime_kwargs and "fps" not in runtime_kwargs:
                merged.pop("fps", None)
            elif "fps" in runtime_kwargs and "num_frames" not in runtime_kwargs:
                merged.pop("num_frames", None)
        return merged

    def __init__(
        self,
        image_io: ImageMediaIO,
        num_frames: int = 32,
        **kwargs,
    ) -> None:
        super().__init__()

        self.image_io = image_io
        self.num_frames = num_frames
        # `kwargs` contains custom arguments from
        # --media-io-kwargs for this modality, merged with
        # per-request runtime media_io_kwargs via merge_kwargs().
        # They can be passed to the underlying
        # media loaders (e.g. custom implementations)
        # for flexible control.

        # Allow per-request override of video backend via kwargs.
        # This enables users to specify a different backend than the
        # global VLLM_VIDEO_LOADER_BACKEND env var, e.g.:
        #   --media-io-kwargs '{"video": {"video_backend": "torchcodec"}}'
        video_loader_backend = (
            kwargs.pop("video_backend", None) or envs.VLLM_VIDEO_LOADER_BACKEND
        )
        self.kwargs = kwargs
        self.video_loader = VIDEO_LOADER_REGISTRY.load(video_loader_backend)

    def load_bytes(
        self, data: bytes
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        video = self.video_loader.load_bytes(
            data, num_frames=self.num_frames, **self.kwargs
        )
        return MediaWithBytes(video, data)

    def _jpeg_sequence_frame_count(self, data: str) -> int:
        if self.num_frames == 0:
            raise ValueError("num_frames must be greater than 0 or -1")

        available_frames = data.count(",") + 1
        if self.num_frames > 0:
            return min(self.num_frames, available_frames)
        return available_frames

    def _decode_jpeg_sequence_base64(self, data: str) -> list[bytes]:
        if self.num_frames > 0:
            frame_parts = data.split(",", self.num_frames)[: self.num_frames]
        elif self.num_frames == 0:
            raise ValueError("num_frames must be greater than 0 or -1")
        else:
            frame_parts = data.split(",")

        return [
            pybase64.b64decode(frame_data, validate=True) for frame_data in frame_parts
        ]

    def _finalize_jpeg_sequence(
        self,
        data: str,
        loaded_frames: list[MediaWithBytes[Image.Image]],
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        try:
            frames = np.stack([np.asarray(frame) for frame in loaded_frames])
            frame_io_configs = [frame.io_config for frame in loaded_frames]
        finally:
            _close_loaded_frames(loaded_frames)

        total = int(frames.shape[0])
        fps = float(self.kwargs.get("fps", 1))

        # validate and extract frames_indices
        frames_indices = self.kwargs.get("frames_indices")
        if frames_indices is not None:
            if not (
                isinstance(frames_indices, list)
                and all(isinstance(i, int) for i in frames_indices)
            ):
                raise ValueError("frames_indices must be a list of integers")
            if len(frames_indices) != total:
                raise ValueError(
                    f"frames_indices length ({len(frames_indices)}) must "
                    f"match number of frames sent ({total})"
                )
        else:
            frames_indices = list(range(total))

        # validate and extract total_num_frames
        total_num_frames = self.kwargs.get("total_num_frames", total)
        if not isinstance(total_num_frames, int) or total_num_frames < 1:
            raise ValueError("total_num_frames must be a positive integer")
        if total_num_frames < total:
            raise ValueError(
                f"total_num_frames ({total_num_frames}) must be >= "
                f"number of frames sent ({total})"
            )

        # validate and extract duration
        duration = self.kwargs.get("duration")
        if duration is not None:
            if not isinstance(duration, (int, float)) or duration < 0:
                raise ValueError("duration must be a non-negative number")
        else:
            duration = total_num_frames / fps if fps > 0 else 0.0

        metadata = {
            "total_num_frames": total_num_frames,
            "fps": fps,
            "duration": duration,
            "video_backend": "jpeg_sequence",
            "frames_indices": frames_indices,
            "do_sample_frames": self.kwargs.get("do_sample_frames", False),
        }
        io_config = (
            {"frame_io_configs": frame_io_configs}
            if any(config is not None for config in frame_io_configs)
            else None
        )
        return MediaWithBytes((frames, metadata), data.encode(), io_config)

    async def load_base64_async(
        self,
        media_type: str,
        data: str,
        *,
        executor: Executor,
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        loop = asyncio.get_running_loop()
        if (
            media_type.lower() != "video/jpeg"
            or self.image_io.backend != NVIMAGECODEC_IMAGE_BACKEND
        ):
            return await loop.run_in_executor(
                executor, self.load_base64, media_type, data
            )

        frame_count = await loop.run_in_executor(
            executor, self._jpeg_sequence_frame_count, data
        )
        async with reserve_image_decode_request_async(self.image_io, frame_count):
            frame_bytes = await loop.run_in_executor(
                executor, self._decode_jpeg_sequence_base64, data
            )
            loaded_frames = await load_images_with_service_async(
                self.image_io, frame_bytes
            )
            try:
                finalizer = loop.run_in_executor(
                    executor,
                    self._finalize_jpeg_sequence,
                    data,
                    loaded_frames,
                )
            except BaseException:
                _close_loaded_frames(loaded_frames)
                raise
            try:
                return await asyncio.shield(finalizer)
            except asyncio.CancelledError:
                # The executor call owns and closes the decoded frames. Let it
                # finish after request cancellation and consume any exception.
                finalizer.add_done_callback(_consume_future_exception)
                raise

    def load_base64(
        self, media_type: str, data: str
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        if media_type.lower() == "video/jpeg":
            frame_bytes = self._decode_jpeg_sequence_base64(data)
            if self.image_io.backend == NVIMAGECODEC_IMAGE_BACKEND:
                loaded_frames = load_images_with_service(self.image_io, frame_bytes)
            else:
                loaded_frames = self.image_io.load_bytes_many(frame_bytes)
            return self._finalize_jpeg_sequence(data, loaded_frames)

        return self.load_bytes(pybase64.b64decode(data))

    def load_file(
        self, filepath: Path
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        with filepath.open("rb") as f:
            data = f.read()

        return self.load_bytes(data)

    def encode_base64(
        self,
        media: npt.NDArray,
        *,
        video_format: str = "JPEG",
    ) -> str:
        video = media

        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in video)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
