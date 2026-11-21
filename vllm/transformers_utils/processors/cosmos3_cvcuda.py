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
from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
    Qwen3VLVideoProcessor,
)

if TYPE_CHECKING:
    from typing import TypedDict as ProcessingKwargs
    from typing import TypedDict as Qwen2VLImageProcessorKwargs
    from typing import TypedDict as Qwen3VLVideoProcessorInitKwargs
else:
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import (
        Qwen2VLImageProcessorKwargs,
    )
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import (
        Qwen3VLVideoProcessorInitKwargs,
    )
    from transformers.processing_utils import ProcessingKwargs


class Cosmos3ImageKwargs(Qwen2VLImageProcessorKwargs, total=False):
    device_resize: bool


class Cosmos3VideoKwargs(Qwen3VLVideoProcessorInitKwargs, total=False):
    device_resize: bool


class Cosmos3ProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Cosmos3ImageKwargs
    videos_kwargs: Cosmos3VideoKwargs


def _validate_contract(processor, pixels, kwargs):
    if any(
        pixel.dtype != torch.uint8 or pixel.device.type != "cpu" or pixel.shape[-3] != 3
        for pixel in pixels
    ):
        raise ValueError("device_resize requires source-resolution CPU uint8 RGB")
    if kwargs.get("do_resize") and kwargs.get("resample") != PILImageResampling.BICUBIC:
        raise ValueError("device_resize requires bicubic interpolation")
    for key in ("patch_size", "temporal_patch_size", "merge_size"):
        if kwargs.get(key, getattr(processor, key)) != getattr(processor, key):
            raise ValueError(f"device_resize requires the checkpoint's {key}")
    for key in ("image_mean", "image_std", "rescale_factor"):
        if kwargs.get(key, getattr(processor, key)) != getattr(processor, key):
            raise ValueError(f"device_resize requires the checkpoint's {key}")


class Cosmos3CVCUDAImageProcessor(Qwen2VLImageProcessor):
    valid_kwargs = Cosmos3ImageKwargs
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

    def _preprocess(self, images, **kwargs) -> BatchFeature:
        if not kwargs.pop("device_resize", True):
            return super()._preprocess(images, **kwargs)
        _validate_contract(self, images, kwargs)
        output = super()._preprocess(
            [torch.empty_like(image, device="meta") for image in images], **kwargs
        )
        output["pixel_values"] = torch.empty(
            (0, output["pixel_values"].shape[-1]),
            dtype=output["pixel_values"].dtype,
            device="cpu",
        )
        output["pixel_values_raw"] = [
            image.permute(1, 2, 0).unsqueeze(0).contiguous() for image in images
        ]
        return output


class Cosmos3CVCUDAVideoProcessor(Qwen3VLVideoProcessor):
    valid_kwargs = Cosmos3VideoKwargs
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

    def _preprocess(self, videos, **kwargs) -> BatchFeature:
        if not kwargs.pop("device_resize", True):
            return super()._preprocess(videos, **kwargs)
        _validate_contract(self, videos, kwargs)
        output = super()._preprocess(
            [torch.empty_like(video, device="meta") for video in videos], **kwargs
        )
        output["pixel_values_videos"] = torch.empty(
            (0, output["pixel_values_videos"].shape[-1]),
            dtype=output["pixel_values_videos"].dtype,
            device="cpu",
        )
        output["pixel_values_videos_raw"] = [
            video.permute(0, 2, 3, 1).contiguous() for video in videos
        ]
        return output


class Cosmos3CVCUDAProcessor(Qwen3VLProcessor):
    valid_processor_kwargs = Cosmos3ProcessorKwargs
    skip_tensor_conversion = Qwen3VLProcessor.skip_tensor_conversion + [
        "pixel_values_raw",
        "pixel_values_videos_raw",
    ]

    def __init__(
        self,
        image_processor=None,
        tokenizer=None,
        video_processor=None,
        chat_template=None,
        **kwargs,
    ):
        super().__init__(
            image_processor=Cosmos3CVCUDAImageProcessor(**image_processor.to_dict()),
            video_processor=Cosmos3CVCUDAVideoProcessor(**video_processor.to_dict()),
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )


def pack_cosmos3_frames(
    frames: torch.Tensor,
    grid: Sequence[int],
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    is_video: bool,
) -> torch.Tensor:
    """Pack resized NCHW frames using Cosmos3 Nano/Super's Qwen3-VL contract."""
    grid_t, grid_h, grid_w = grid
    if min(grid) < 1 or grid_h % merge_size or grid_w % merge_size:
        raise ValueError("Vision grid must be positive and divisible by merge_size")
    if frames.ndim != 4 or tuple(frames.shape[1:]) != (
        3,
        grid_h * patch_size,
        grid_w * patch_size,
    ):
        raise ValueError("Resized RGB frame dimensions must match the vision grid")
    if not is_video:
        if len(frames) != 1 or grid_t != 1:
            raise ValueError("An image must have exactly one frame and temporal grid")
        frames = frames.expand(temporal_patch_size, -1, -1, -1)
    elif pad := -len(frames) % temporal_patch_size:
        frames = torch.cat([frames, frames[-1:].expand(pad, -1, -1, -1)])
    if len(frames) != grid_t * temporal_patch_size:
        raise ValueError("Raw frame count does not match the temporal vision grid")
    patches = frames.reshape(
        grid_t,
        temporal_patch_size,
        3,
        grid_h // merge_size,
        merge_size,
        patch_size,
        grid_w // merge_size,
        merge_size,
        patch_size,
    ).permute(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return patches.reshape(
        grid_t * grid_h * grid_w, 3 * temporal_patch_size * patch_size**2
    )


def cvcuda_cosmos3_patchify(
    pixels: Sequence[torch.Tensor],
    grid_thw: torch.Tensor,
    device: torch.device,
    *,
    image_processor,
    is_video: bool,
) -> torch.Tensor:
    """Resize raw RGB transport; leave normalization in Nano/Super's input norm."""
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        cvcuda_bicubic_resize_many,
    )

    if grid_thw.device.type != "cpu" or len(pixels) != len(grid_thw):
        raise ValueError("Raw pixel count must match the CPU vision grids")
    grids = grid_thw.tolist()
    sizes = [
        (grid[1] * image_processor.patch_size, grid[2] * image_processor.patch_size)
        for grid in grids
    ]
    resized_items = cvcuda_bicubic_resize_many(pixels, sizes, device)
    output = []
    for resized, grid in zip(resized_items, grids):
        output.append(
            pack_cosmos3_frames(
                resized,
                grid,
                patch_size=image_processor.patch_size,
                temporal_patch_size=image_processor.temporal_patch_size,
                merge_size=image_processor.merge_size,
                is_video=is_video,
            )
        )
    return torch.cat(output)
