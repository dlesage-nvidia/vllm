# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from transformers import BatchFeature
from transformers.image_transforms import convert_to_rgb
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
)
from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
    Qwen2VLImageProcessor,
)
from transformers.models.qwen3_vl.processing_qwen3_vl import (
    Qwen3VLProcessor,
)
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

if TYPE_CHECKING:
    from typing import TypedDict as Qwen2VLImageProcessorKwargs
    from typing import TypedDict as Qwen3VLProcessorKwargs
    from typing import TypedDict as Qwen3VLVideoProcessorInitKwargs
else:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessorKwargs,
    )
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessorKwargs
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessorInitKwargs,
    )


def reserve_qwen3_5_cvcuda_workspace(device: torch.device) -> torch.Tensor:
    """Account for bounded upload/intermediate buffers in the startup profile."""
    from vllm.transformers_utils.processors.cvcuda_uint8_resize import (
        reserve_cvcuda_workspace,
    )

    return reserve_cvcuda_workspace(device)


class Qwen3_5CVCUDAImageKwargs(Qwen2VLImageProcessorKwargs, total=False):
    device_resize: bool


class Qwen3_5CVCUDAVideoKwargs(Qwen3VLVideoProcessorInitKwargs, total=False):
    device_resize: bool


class Qwen3_5CVCUDAProcessorKwargs(Qwen3VLProcessorKwargs, total=False):
    images_kwargs: Qwen3_5CVCUDAImageKwargs
    videos_kwargs: Qwen3_5CVCUDAVideoKwargs


def _validate_raw_pixels(pixels: Sequence[torch.Tensor], kwargs: dict) -> None:
    if kwargs.get("do_rescale") or kwargs.get("do_normalize"):
        raise ValueError("device_resize requires mm_device_do_normalize=True")
    if kwargs.get("do_resize") and kwargs.get("resample") != PILImageResampling.BICUBIC:
        raise ValueError("device_resize requires bicubic interpolation")
    if any(pixel.dtype != torch.uint8 or pixel.shape[-3] != 3 for pixel in pixels):
        raise ValueError("device_resize requires uint8 RGB pixels")


class Qwen3_5CVCUDAImageProcessor(Qwen2VLImageProcessor):
    valid_kwargs = Qwen3_5CVCUDAImageKwargs
    device_resize = True

    def process_image(
        self, image, do_convert_rgb=None, input_data_format=None, device=None, **kwargs
    ) -> torch.Tensor:
        if isinstance(image, Image.Image):
            image = (
                convert_to_rgb(image, return_numpy=True)
                if do_convert_rgb
                else np.array(image)
            )
            input_data_format = ChannelDimension.LAST
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image)
        if (
            isinstance(image, torch.Tensor)
            and image.ndim == 3
            and image.dtype == torch.uint8
        ):
            if input_data_format is None:
                input_data_format = infer_channel_dimension_format(
                    image, num_channels=(1, 3, 4)
                )
            if input_data_format == ChannelDimension.LAST:
                image = image.permute(2, 0, 1)
                input_data_format = ChannelDimension.FIRST
            if image.shape[0] == 3:
                return image.to(device) if device is not None else image
        return super().process_image(
            image,
            do_convert_rgb=do_convert_rgb,
            input_data_format=input_data_format,
            device=device,
            **kwargs,
        )

    def _preprocess(self, images: list[torch.Tensor], **kwargs) -> BatchFeature:
        """Run HF's grid and patch-shape logic without allocating resized pixels."""
        if not kwargs.pop("device_resize", True):
            return super()._preprocess(images, **kwargs)
        _validate_raw_pixels(images, kwargs)
        output = super()._preprocess(
            [torch.empty_like(image, device="meta") for image in images], **kwargs
        )
        output["pixel_values"] = torch.empty(
            (0, output["pixel_values"].shape[-1]), dtype=torch.uint8
        )
        output["pixel_values_raw"] = [
            image.permute(1, 2, 0).unsqueeze(0).contiguous() for image in images
        ]
        return output


class Qwen3_5CVCUDAVideoProcessor(Qwen3VLVideoProcessor):
    valid_kwargs = Qwen3_5CVCUDAVideoKwargs
    device_resize = True

    def _prepare_input_videos(
        self, videos, do_convert_rgb=None, input_data_format=None, device=None
    ) -> list[torch.Tensor]:
        videos = [torch.as_tensor(video) for video in videos]
        if input_data_format is None:
            input_data_format = infer_channel_dimension_format(videos[0])
        if input_data_format == ChannelDimension.LAST:
            videos = [video.permute(0, 3, 1, 2) for video in videos]
        return super()._prepare_input_videos(
            videos,
            do_convert_rgb=do_convert_rgb,
            input_data_format=ChannelDimension.FIRST,
            device=device,
        )

    def _preprocess(self, videos: list[torch.Tensor], **kwargs) -> BatchFeature:
        """Keep HF's frame padding and pixel budgets, including per-frame caps."""
        if not kwargs.pop("device_resize", True):
            return super()._preprocess(videos, **kwargs)
        _validate_raw_pixels(videos, kwargs)
        output = super()._preprocess(
            [torch.empty_like(video, device="meta") for video in videos], **kwargs
        )
        output["pixel_values_videos"] = torch.empty(
            (0, output["pixel_values_videos"].shape[-1]), dtype=torch.uint8
        )
        output["pixel_values_videos_raw"] = [
            video.permute(0, 2, 3, 1).contiguous() for video in videos
        ]
        return output


class Qwen3_5CVCUDAProcessor(Qwen3VLProcessor):
    valid_processor_kwargs = Qwen3_5CVCUDAProcessorKwargs
    skip_tensor_conversion = list(Qwen3VLProcessor.skip_tensor_conversion) + [
        "pixel_values_raw",
        "pixel_values_videos_raw",
    ]

    def __init__(
        self, image_processor=None, tokenizer=None, video_processor=None, **kwargs
    ):
        image_processor = Qwen3_5CVCUDAImageProcessor(**image_processor.to_dict())
        video_processor = Qwen3_5CVCUDAVideoProcessor(**video_processor.to_dict())
        super().__init__(
            image_processor=image_processor,
            tokenizer=tokenizer,
            video_processor=video_processor,
            **kwargs,
        )


def cvcuda_qwen3_5_patchify(
    pixels: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    device: torch.device,
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    is_video: bool,
    max_chunk_bytes: int = 256 * 1024 * 1024,
) -> torch.Tensor:
    """Resize raw RGB frames and pack Qwen's merged spatial/temporal patches."""
    from vllm.transformers_utils.processors.cvcuda_utils import (
        cvcuda_bicubic_resize,
        cvcuda_cpu_uint8_scratch_bytes,
    )

    if len(pixels) != len(grid_thw):
        raise ValueError("Raw pixel count must match the number of vision grids")
    patches = []
    for frames, (grid_t, grid_h, grid_w) in zip(pixels, grid_thw.tolist()):
        scratch_bytes = cvcuda_cpu_uint8_scratch_bytes(
            frames.shape[1], frames.shape[2], grid_h * patch_size, grid_w * patch_size
        )
        if scratch_bytes > 512 * 1024 * 1024:
            raise ValueError("A source frame exceeds the CV-CUDA scratch memory budget")
        resized = cvcuda_bicubic_resize(
            frames,
            (grid_h * patch_size, grid_w * patch_size),
            device,
            max_chunk_bytes=max_chunk_bytes,
            quantize_intermediate=True,
        )
        if not is_video:
            resized = resized.expand(temporal_patch_size, -1, -1, -1)
        elif pad := -len(resized) % temporal_patch_size:
            resized = torch.cat((resized, resized[-1:].expand(pad, -1, -1, -1)))
        if len(resized) != grid_t * temporal_patch_size:
            raise ValueError("Raw frame count does not match the temporal vision grid")
        packed = resized.reshape(
            grid_t,
            temporal_patch_size,
            3,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        patches.append(
            packed.permute(0, 3, 6, 4, 7, 2, 1, 5, 8).reshape(
                grid_t * grid_h * grid_w,
                3 * temporal_patch_size * patch_size * patch_size,
            )
        )
    return torch.cat(patches)
