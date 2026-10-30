# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Experimental FP32 image preprocessing using only shipping CV-CUDA operators."""

import math
from collections import deque

import torch


class PureImagePreprocessor:
    """Retain floating-point bicubic overshoot until the native model-dtype cast."""

    def __init__(self, mean: list[float], std: list[float]):
        import cvcuda

        if cvcuda.__version__ != "0.19.0":
            raise ValueError("Pure CV-CUDA preprocessing requires CV-CUDA 0.19.0")
        if (
            len(mean) != 3
            or len(std) != 3
            or not all(math.isfinite(value) for value in (*mean, *std))
            or not all(value > 0 for value in std)
        ):
            raise ValueError(
                "Normalization requires three finite means and positive stds"
            )
        self.cvcuda = cvcuda
        self.mean_host = torch.tensor(
            mean, dtype=torch.float32, device="cpu"
        ).pin_memory()
        self.std_host = torch.tensor(
            std, dtype=torch.float32, device="cpu"
        ).pin_memory()
        self.parameters: dict[
            tuple[torch.device, int], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self.pending: deque = deque()

    def __call__(self, source, size, dtype) -> torch.Tensor:
        """Convert CUDA NHWC uint8 RGB to normalized model-dtype NCHW pixels."""
        if (
            not source.is_cuda
            or source.ndim != 4
            or source.dtype != torch.uint8
            or source.shape[-1] != 3
            or not source.is_contiguous()
            or any(not 1 <= extent <= 16384 for extent in source.shape[1:3])
        ):
            raise ValueError("Source must be contiguous CUDA NHWC RGB uint8")
        if (
            not isinstance(size, (list, tuple))
            or len(size) != 2
            or any(
                type(extent) is not int or not 1 <= extent <= 16384 for extent in size
            )
            or dtype not in (torch.bfloat16, torch.float16, torch.float32)
        ):
            raise ValueError("Target requires valid image extents and a floating dtype")
        if torch.cuda.is_current_stream_capturing():
            raise ValueError("Pure CV-CUDA preprocessing does not support CUDA Graphs")
        while self.pending and self.pending[0][0].query():
            self.pending.popleft()
        torch_stream = torch.cuda.current_stream(source.device)
        parameter_key = (source.device, torch_stream.cuda_stream)
        if parameter_key not in self.parameters:
            mean = self.mean_host.to(source.device, non_blocking=True)
            inverse_std = self.std_host.to(
                source.device, non_blocking=True
            ).reciprocal()
            self.parameters[parameter_key] = (
                mean.reshape(1, 1, 1, 3),
                inverse_std.reshape(1, 1, 1, 3),
            )
        mean, inverse_std = self.parameters[parameter_key]
        stream = self.cvcuda.as_stream(torch_stream)
        floating = torch.empty_like(source, dtype=torch.float32)
        resized = torch.empty(
            (source.shape[0], *size, 3), dtype=torch.float32, device=source.device
        )
        scaled, normalized = torch.empty_like(resized), torch.empty_like(resized)
        planar = torch.empty(
            (source.shape[0], 3, *size), dtype=torch.float32, device=source.device
        )
        buffers = [
            source,
            floating,
            resized,
            scaled,
            normalized,
            planar,
            mean,
            inverse_std,
        ]
        views = [
            self.cvcuda.as_tensor(buffer, "NCHW" if index == 5 else "NHWC")
            for index, buffer in enumerate(buffers)
        ]
        source_cv, floating_cv, resized_cv, scaled_cv = views[:4]
        normalized_cv, planar_cv, mean_cv, inverse_cv = views[4:]
        operations = [
            self.cvcuda.convertto_into(floating_cv, source_cv, stream=stream),
            self.cvcuda.tvresize_into(
                resized_cv,
                floating_cv,
                interp=self.cvcuda.Interp.CUBIC,
                antialias=True,
                stream=stream,
            ),
            self.cvcuda.convertto_into(
                scaled_cv, resized_cv, scale=1.0 / 255.0, stream=stream
            ),
            self.cvcuda.normalize_into(
                normalized_cv,
                scaled_cv,
                mean_cv,
                inverse_cv,
                globalscale=1.0,
                globalshift=0.0,
                epsilon=0.0,
                stream=stream,
            ),
            self.cvcuda.reformat_into(planar_cv, normalized_cv, stream=stream),
        ]
        result = planar.to(dtype)
        event = torch.cuda.Event()
        event.record(torch_stream)
        self.pending.append(
            (
                event,
                [*buffers, *views, *operations, result, self.mean_host, self.std_host],
            )
        )
        return result

    def process_images(self, images, sizes, device, dtype) -> list[torch.Tensor]:
        """Upload CPU RGB images while retaining asynchronous buffer ownership."""
        if (
            not isinstance(images, (list, torch.Tensor))
            or not isinstance(sizes, list)
            or len(images) != len(sizes)
        ):
            raise ValueError(
                "Raw image tensors and corresponding target sizes required"
            )
        if isinstance(images, torch.Tensor) and images.ndim != 4:
            raise ValueError("A raw image batch must be NHWC")
        outputs = []
        for image, size in zip(images, sizes, strict=True):
            if (
                not isinstance(image, torch.Tensor)
                or image.device.type != "cpu"
                or image.ndim != 3
                or image.dtype != torch.uint8
                or image.shape[-1] != 3
                or not image.is_contiguous()
                or any(not 1 <= extent <= 16384 for extent in image.shape[:2])
            ):
                raise ValueError("Raw images must be contiguous CPU HWC RGB uint8")
            staging = image[None].pin_memory()
            outputs.append(self(staging.to(device, non_blocking=True), size, dtype)[0])
            self.pending[-1][1].append(staging)
        return outputs
