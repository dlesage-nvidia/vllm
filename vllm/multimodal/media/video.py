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

from ..video import (
    PYNVVIDEOCODEC_VIDEO_BACKEND,
    VIDEO_LOADER_REGISTRY,
    VLLM_VIDEO_INPUT_DATA_FORMAT_KEY,
    uses_pynvvideocodec_video_io,
)
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

    @staticmethod
    def uses_pynvvideocodec(kwargs: dict[str, Any]) -> bool:
        return uses_pynvvideocodec_video_io(kwargs)

    @classmethod
    def merge_kwargs(
        cls,
        default_kwargs: dict[str, Any] | None,
        runtime_kwargs: dict[str, Any] | None,
    ) -> dict[str, Any]:
        default_kwargs = dict(default_kwargs or {})
        default_uses_pynvvideocodec = cls.uses_pynvvideocodec(default_kwargs)
        stripped_pynvvideocodec_backend = False
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
                        stripped_pynvvideocodec_backend |= (
                            requested == PYNVVIDEOCODEC_VIDEO_BACKEND
                        )

            if stripped_pynvvideocodec_backend:
                runtime_kwargs.pop("output_layout", None)

        merged = super().merge_kwargs(default_kwargs, runtime_kwargs)
        if default_uses_pynvvideocodec:
            if cls.uses_pynvvideocodec(merged):
                # The static PyNvVideoCodec layout owns the retained decoder
                # slots and cannot be changed by a request.
                if "output_layout" in default_kwargs:
                    merged["output_layout"] = default_kwargs["output_layout"]
                else:
                    merged.pop("output_layout", None)
            else:
                # A request may fall back to another loader or codec. Remove
                # static PyNv-only options, while preserving an explicitly
                # request-owned output_layout for a custom loader.
                merged.pop("hw_decoders", None)
                if not runtime_kwargs or "output_layout" not in runtime_kwargs:
                    merged.pop("output_layout", None)
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
        input_data_format = (
            video[1].get(VLLM_VIDEO_INPUT_DATA_FORMAT_KEY)
            if isinstance(video, tuple) and isinstance(video[1], dict)
            else None
        )
        io_config = (
            {"pynvvideocodec_input_data_format": input_data_format}
            if input_data_format is not None
            else None
        )
        return MediaWithBytes(video, data, io_config)

    def _jpeg_sequence_frame_count(self, data: str) -> int:
        if self.num_frames == 0:
            raise ValueError("num_frames must be greater than 0 or -1")

        available_frames = data.count(",") + 1
        if self.num_frames > 0:
            return min(self.num_frames, available_frames)
        return available_frames

    def _jpeg_sequence_parts(self, data: str) -> list[str]:
        if self.num_frames > 0:
            return data.split(",", self.num_frames)[: self.num_frames]
        elif self.num_frames == 0:
            raise ValueError("num_frames must be greater than 0 or -1")
        return data.split(",")

    def _decode_jpeg_sequence_base64(self, data: str) -> list[bytes]:
        return [
            pybase64.b64decode(frame_data, validate=True)
            for frame_data in self._jpeg_sequence_parts(data)
        ]

    def _build_jpeg_sequence_result(
        self,
        data: str,
        frames: npt.NDArray,
        frame_io_configs: list[dict[str, Any] | None],
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
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

    def _finalize_jpeg_sequence(
        self,
        data: str,
        loaded_frames: list[MediaWithBytes[Image.Image]],
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        try:
            frames = np.stack([np.asarray(frame.media) for frame in loaded_frames])
            frame_io_configs = [frame.io_config for frame in loaded_frames]
        finally:
            _close_loaded_frames(loaded_frames)
        return self._build_jpeg_sequence_result(data, frames, frame_io_configs)

    def _load_jpeg_sequence_pillow(
        self, data: str
    ) -> MediaWithBytes[tuple[npt.NDArray, dict[str, Any]]]:
        frame_arrays: list[npt.NDArray] = []
        frame_io_configs: list[dict[str, Any] | None] = []
        for frame_data in self._jpeg_sequence_parts(data):
            loaded_frame = self.image_io.load_base64("image/jpeg", frame_data)
            try:
                # Copy each frame before closing it so only one Pillow raster
                # needs to be live in addition to the final NumPy video.
                frame_arrays.append(np.array(loaded_frame.media, copy=True))
                frame_io_configs.append(loaded_frame.io_config)
            finally:
                loaded_frame.media.close()

        frames = np.stack(frame_arrays)
        return self._build_jpeg_sequence_result(data, frames, frame_io_configs)

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
                self.image_io, frame_bytes, executor=executor
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
            if self.image_io.backend == NVIMAGECODEC_IMAGE_BACKEND:
                frame_bytes = self._decode_jpeg_sequence_base64(data)
                loaded_frames = load_images_with_service(self.image_io, frame_bytes)
                return self._finalize_jpeg_sequence(data, loaded_frames)
            return self._load_jpeg_sequence_pillow(data)

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
        if video_format == "JPEG":
            encode_frame = partial(
                self.image_io.encode_base64,
                image_format=video_format,
            )

            return ",".join(encode_frame(Image.fromarray(frame)) for frame in media)

        msg = "Only JPEG format is supported for now."
        raise NotImplementedError(msg)
