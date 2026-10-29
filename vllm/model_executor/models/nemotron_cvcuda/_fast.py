# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""B-exact CV-CUDA resize with direct RGBX upload and fused U8 patch packing."""

import functools
import hashlib
from pathlib import Path

from ._resize import ExactResize


@functools.lru_cache(maxsize=1)
def finish_module():
    from torch.utils.cpp_extension import load

    paths = [Path(__file__).with_suffix(suffix) for suffix in (".cpp", ".cu")]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    return load(
        name="nemotron_B_exact_cvcuda_finish_" + digest[:16],
        sources=[str(path) for path in paths],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    ), digest


def retire_scratch(pixels):
    import torch

    torch.cuda.current_stream().synchronize()
    pixels._d_cvcuda_owners = ()
    return pixels


class RawRGBXUploader:
    def __init__(self):
        from ._upload import CudaRuntimeCopy

        self.copy = CudaRuntimeCopy()

    def __call__(self, items):
        import torch

        if not items:
            raise ValueError("Direct D upload requires a nonempty RGBX batch")
        shape = tuple(items[0].shape)
        if (
            any(
                item.device.type != "cpu"
                or item.dtype != torch.uint8
                or not item.is_contiguous()
                or tuple(item.shape) != shape
                for item in items
            )
            or len(shape) != 3
            or shape[-1] != 4
        ):
            raise ValueError("Direct D upload requires same-shape contiguous CPU RGBX")
        output = torch.empty((len(items), *shape), dtype=torch.uint8, device="cuda")
        output._d_cvcuda_raw_owners = tuple(items)
        stream = torch.cuda.current_stream()
        image_bytes = items[0].numel()
        try:
            for index, item in enumerate(items):
                self.copy(
                    output.data_ptr() + index * image_bytes,
                    item.data_ptr(),
                    image_bytes,
                    stream.cuda_stream,
                )
        except BaseException:
            stream.synchronize()
            raise
        return output


class PackedExactResize(ExactResize):
    def __init__(self, cvcuda, patch):
        import torch

        super().__init__(cvcuda)
        self.finish, self.finish_digest = finish_module()
        self.patch = patch
        self.empty_int = torch.empty(0, dtype=torch.int32, device="cuda")
        self.empty_float = torch.empty(0, dtype=torch.float32, device="cuda")
        self.cache_sentinel = torch.empty(
            (1, 1, 1, 4), dtype=torch.uint8, device="cuda"
        )

    def retire(self, pixels):
        """Retire completed non-reusable tensor wrappers, preserving operator pools."""
        retire_scratch(pixels)
        self.cvcuda.as_tensor(self.cache_sentinel, "NHWC")
        return pixels

    def __call__(self, source, size):
        import torch

        if (
            source.ndim != 4
            or source.dtype != torch.uint8
            or not source.is_cuda
            or not source.is_contiguous()
            or source.shape[-1] not in (3, 4)
        ):
            raise ValueError(
                "Packed D resize requires contiguous CUDA NHWC RGB/RGBX U8"
            )
        if any(extent % self.patch for extent in size):
            raise ValueError("Packed D resize requires patch-aligned target extents")
        axes = [axis for axis in (2, 1) if source.shape[axis] != size[axis - 1]]
        current = source
        owners = [source]
        stream = self.cvcuda.as_stream(torch.cuda.current_stream())
        for axis in axes:
            destination = size[axis - 1]
            key = (current.shape[axis], destination, axis, current.shape[-1])
            if key not in self.axes:
                raise ValueError(
                    "Packed resize geometry has not completed untimed setup"
                )
            floating = self.cvcuda.convertto(
                self.cvcuda.as_tensor(current, "NHWC"),
                self.cvcuda.Type.F32,
                stream=stream,
            )
            target = (
                (current.shape[1], destination)
                if axis == 2
                else (destination, current.shape[2])
            )
            proxy = torch.empty(
                (current.shape[0], *target, current.shape[-1]),
                dtype=torch.float32,
                device=current.device,
            )
            resized = self.cvcuda.tvresize_into(
                self.cvcuda.as_tensor(proxy, "NHWC"),
                floating,
                interp=self.cvcuda.Interp.CUBIC,
                antialias=True,
                stream=stream,
            )
            owners.extend((floating, resized, proxy))
            if axis == axes[-1]:
                result = self.finish.finish(
                    current, proxy, *self.axes[key], axis, *size, self.patch
                )
            else:
                current = self.module.correct(current, proxy, *self.axes[key], axis)
                owners.append(current)
        if not axes:
            result = self.finish.finish(
                current,
                self.empty_float,
                self.empty_int,
                self.empty_int,
                self.empty_int,
                self.empty_float,
                0,
                0,
                *size,
                self.patch,
            )
        result._d_cvcuda_owners = tuple(owners)
        return result
