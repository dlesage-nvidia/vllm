# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch
from transformers import BatchFeature, Cosmos3EdgeProcessor
from transformers.image_utils import (
    ChannelDimension,
    PILImageResampling,
    infer_channel_dimension_format,
)
from transformers.models.cosmos3_edge.image_processing_cosmos3_edge import (
    Cosmos3EdgeImageProcessor,
)
from transformers.models.cosmos3_edge.video_processing_cosmos3_edge import (
    Cosmos3EdgeVideoProcessor,
)

if TYPE_CHECKING:
    from typing import TypedDict as Cosmos3EdgeImageProcessorKwargs
    from typing import TypedDict as Cosmos3EdgeVideoProcessorInitKwargs
    from typing import TypedDict as ProcessingKwargs
else:
    from transformers.models.cosmos3_edge.image_processing_cosmos3_edge import (
        Cosmos3EdgeImageProcessorKwargs,
    )
    from transformers.models.cosmos3_edge.video_processing_cosmos3_edge import (
        Cosmos3EdgeVideoProcessorInitKwargs,
    )
    from transformers.processing_utils import ProcessingKwargs


class Cosmos3EdgeImageKwargs(Cosmos3EdgeImageProcessorKwargs, total=False):
    device_resize: bool


class Cosmos3EdgeVideoKwargs(Cosmos3EdgeVideoProcessorInitKwargs, total=False):
    device_resize: bool


class Cosmos3EdgeProcessorKwargs(ProcessingKwargs, total=False):
    images_kwargs: Cosmos3EdgeImageKwargs
    videos_kwargs: Cosmos3EdgeVideoKwargs


def _raw_transform(processor, pixels, kwargs) -> torch.Tensor:
    if any(
        pixel.device.type != "cpu" or pixel.dtype != torch.uint8 or pixel.shape[-3] != 3
        for pixel in pixels
    ):
        raise ValueError("Edge device_resize requires source-resolution CPU uint8 RGB")
    if kwargs["do_resize"] and kwargs["resample"] != PILImageResampling.BICUBIC:
        raise ValueError("Edge device_resize requires bicubic interpolation")
    for key, expected in (
        ("patch_size", 16),
        ("temporal_patch_size", 1),
        ("merge_size", 2),
    ):
        if kwargs.get(key, getattr(processor, key)) != expected:
            raise ValueError(f"Edge device_resize requires {key}={expected}")
    channels = []
    for key in ("image_mean", "image_std"):
        configured = kwargs.get(key)
        if configured is None:
            configured = getattr(processor, key)
        values = torch.as_tensor(configured, dtype=torch.float64).flatten()
        if len(values) == 1:
            values = values.expand(3)
        if len(values) != 3:
            raise ValueError(f"Edge device_resize requires three RGB {key} values")
        channels.extend(values.tolist())
    return torch.tensor(
        [
            kwargs["do_rescale"],
            kwargs["do_normalize"],
            kwargs["rescale_factor"],
            *channels,
        ],
        dtype=torch.float64,
    )


class Cosmos3EdgeCVCUDAImageProcessor(Cosmos3EdgeImageProcessor):
    valid_kwargs = Cosmos3EdgeImageKwargs
    device_resize = True

    def process_image(
        self, image, do_convert_rgb=None, input_data_format=None, device=None, **kwargs
    ) -> torch.Tensor:
        tensor = torch.from_numpy(image) if isinstance(image, np.ndarray) else image
        if (
            isinstance(tensor, torch.Tensor)
            and tensor.ndim == 3
            and tensor.dtype == torch.uint8
        ):
            channel_format = input_data_format
            if channel_format is None:
                channel_format = infer_channel_dimension_format(
                    tensor, num_channels=(1, 3, 4)
                )
            if channel_format == ChannelDimension.LAST:
                tensor = tensor.permute(2, 0, 1)
            if tensor.shape[0] == 3:
                return tensor.to(device) if device is not None else tensor
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
        transform = _raw_transform(self, images, kwargs)
        output = super()._preprocess(
            [torch.empty_like(image, device="meta") for image in images], **kwargs
        )
        output["pixel_values"] = torch.empty(
            (0, output["pixel_values"].shape[-1]), dtype=output["pixel_values"].dtype
        )
        output["pixel_values_raw"] = [
            [image.permute(1, 2, 0).unsqueeze(0).contiguous(), transform.clone()]
            for image in images
        ]
        return output


class Cosmos3EdgeCVCUDAVideoProcessor(Cosmos3EdgeVideoProcessor):
    valid_kwargs = Cosmos3EdgeVideoKwargs
    device_resize = True

    def _prepare_input_videos(
        self, videos, do_convert_rgb=None, input_data_format=None, device=None
    ) -> list[torch.Tensor]:
        videos = [torch.as_tensor(video) for video in videos]
        if videos and input_data_format is None:
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
        transform = _raw_transform(self, videos, kwargs)
        output = super()._preprocess(
            [torch.empty_like(video, device="meta") for video in videos], **kwargs
        )
        output["pixel_values_videos"] = torch.empty(
            (0, output["pixel_values_videos"].shape[-1]),
            dtype=output["pixel_values_videos"].dtype,
        )
        output["pixel_values_videos_raw"] = [
            [video.permute(0, 2, 3, 1).contiguous(), transform.clone()]
            for video in videos
        ]
        return output


class Cosmos3EdgeCVCUDAProcessor(Cosmos3EdgeProcessor):
    valid_processor_kwargs = Cosmos3EdgeProcessorKwargs
    skip_tensor_conversion = Cosmos3EdgeProcessor.skip_tensor_conversion + [
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
            image_processor=Cosmos3EdgeCVCUDAImageProcessor(
                **image_processor.to_dict()
            ),
            video_processor=Cosmos3EdgeCVCUDAVideoProcessor(
                **video_processor.to_dict()
            ),
            tokenizer=tokenizer,
            chat_template=chat_template,
            **kwargs,
        )


def pack_cosmos3_edge_frames(
    frames: torch.Tensor,
    grid: Sequence[int],
    *,
    patch_size: int = 16,
    merge_size: int = 2,
) -> torch.Tensor:
    """Pack Edge frames in merge-window order with HWC values inside each patch."""
    grid_t, grid_h, grid_w = grid
    if (
        min(grid) < 1
        or grid_h % merge_size
        or grid_w % merge_size
        or tuple(frames.shape) != (grid_t, 3, grid_h * patch_size, grid_w * patch_size)
    ):
        raise ValueError("Edge frames must match the unpadded THW grid")
    return (
        frames.reshape(
            grid_t,
            3,
            grid_h // merge_size,
            merge_size,
            patch_size,
            grid_w // merge_size,
            merge_size,
            patch_size,
        )
        .permute(0, 2, 5, 3, 6, 4, 7, 1)
        .reshape(grid_t * grid_h * grid_w, 3 * patch_size**2)
    )


def normalize_cosmos3_edge_frames(
    frames: torch.Tensor,
    transform: torch.Tensor,
    *,
    image_processor: Cosmos3EdgeImageProcessor | Cosmos3EdgeVideoProcessor,
) -> torch.Tensor:
    """Apply the actual stock fused rescale/normalize after uint8 resize."""
    if transform.device.type != "cpu" or transform.shape != (9,):
        raise ValueError(
            "Edge normalization settings must remain a CPU nine-value tensor"
        )
    do_rescale, do_normalize, rescale_factor, *channels = transform.tolist()
    return image_processor.rescale_and_normalize(
        frames,
        bool(do_rescale),
        rescale_factor,
        bool(do_normalize),
        tuple(channels[:3]),
        tuple(channels[3:]),
    )


def cvcuda_cosmos3_edge_patchify(
    items: Sequence[Sequence[torch.Tensor]],
    grid_thw: torch.Tensor,
    device: torch.device,
    *,
    image_processor: Cosmos3EdgeImageProcessor | Cosmos3EdgeVideoProcessor,
) -> torch.Tensor:
    """Resize source RGB, normalize on device, then pack the distinct SigLIP2 layout."""
    from vllm.transformers_utils.processors.cosmos3_cvcuda_resize import (
        cvcuda_bicubic_resize_many,
    )

    if grid_thw.device.type != "cpu" or grid_thw.ndim != 2 or grid_thw.shape[1] != 3:
        raise ValueError("Edge raw pixels require CPU THW grids")
    if len(items) != len(grid_thw) or not items:
        raise ValueError("Edge raw item count must match the vision grids")
    grids = grid_thw.tolist()
    for item, grid in zip(items, grids):
        if len(item) != 2:
            raise ValueError("Edge raw items require frames and normalization settings")
        frames, transform = item
        if (
            frames.device.type != "cpu"
            or frames.dtype != torch.uint8
            or frames.ndim != 4
            or frames.shape[-1] != 3
            or len(frames) != grid[0]
            or transform.device.type != "cpu"
            or transform.shape != (9,)
        ):
            raise ValueError("Edge raw items must preserve CPU RGB frames and settings")
    sizes = [
        (grid[1] * image_processor.patch_size, grid[2] * image_processor.patch_size)
        for grid in grids
    ]
    resized_items = cvcuda_bicubic_resize_many(
        [item[0] for item in items], sizes, device
    )
    output = []
    for resized, item, grid in zip(resized_items, items, grids):
        transform = item[1]
        output.append(
            pack_cosmos3_edge_frames(
                normalize_cosmos3_edge_frames(
                    resized, transform, image_processor=image_processor
                ),
                grid,
                patch_size=image_processor.patch_size,
                merge_size=image_processor.merge_size,
            )
        )
    return torch.cat(output)
