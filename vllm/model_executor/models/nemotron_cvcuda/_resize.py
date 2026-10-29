# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CV-CUDA resampling with pinned PR-B integer-rounding compatibility."""

import functools
import hashlib
import math
import os
from pathlib import Path

TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"


def cubic(value):
    distance = abs(value)
    if distance < 1.0:
        return ((1.5 * distance - 2.5) * distance) * distance + 1.0
    if distance < 2.0:
        return ((-0.5 * distance + 2.5) * distance - 4.0) * distance + 2.0
    return 0.0


@functools.lru_cache(maxsize=64)
def axis_coefficients(input_size, output_size):
    import numpy as np

    if min(input_size, output_size) < 1 or max(input_size, output_size) > 8192:
        raise ValueError("D CV-CUDA supports positive resize extents up to 8192")
    scale = input_size / output_size
    support = 2.0 * max(scale, 1.0)
    inverse_scale = 1.0 / scale if scale >= 1.0 else 1.0
    taps = math.ceil(support) * 2 + 1
    starts = np.empty(output_size, dtype=np.int32)
    counts = np.empty(output_size, dtype=np.int32)
    weights = np.zeros((output_size, taps), dtype=np.float64)
    maximum = 0.0
    for destination in range(output_size):
        center = scale * (destination + 0.5)
        start = max(int(center - support + 0.5), 0)
        count = max(min(int(center + support + 0.5), input_size) - start, 0)
        count = min(count, taps)
        values = [
            cubic((index + start - center + 0.5) * inverse_scale)
            for index in range(count)
        ]
        total = sum(values)
        starts[destination] = start
        counts[destination] = count
        for index, value in enumerate(values):
            weight = value / total if total else value
            weights[destination, index] = weight
            maximum = max(maximum, weight)
    precision = 0
    while precision < 22:
        if int(0.5 + maximum * (1 << (precision + 1))) >= (1 << 15):
            break
        precision += 1
    scaled = weights * (1 << precision)
    integers = np.trunc(scaled + np.where(scaled < 0, -0.5, 0.5)).astype(np.int32)
    if np.any(
        np.sum(np.abs(integers.astype(np.int64)), axis=1) * 255 + (1 << (precision - 1))
        >= 2**31
    ):
        raise ValueError("Pinned B resize accumulation would overflow int32")
    return starts, counts, integers, precision


@functools.lru_cache(maxsize=1)
def compatibility_module():
    from torch.utils.cpp_extension import load

    paths = [Path(__file__).with_name("_resize" + suffix) for suffix in (".cpp", ".cu")]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    os.environ.setdefault("MAX_JOBS", "2")
    return (
        load(
            name="nemotron_B_exact_cvcuda_" + digest[:16],
            sources=[str(path) for path in paths],
            extra_cflags=["-O3"],
            extra_cuda_cflags=["-O3"],
            with_cuda=True,
        ),
        digest,
    )


class ExactResize:
    def __init__(self, cvcuda):
        import torch

        if torch.version.git_version != TORCH_GIT or cvcuda.__version__ != "0.19.0":
            raise ValueError("CV-CUDA D requires the frozen Torch/CV-CUDA runtime")
        self.cvcuda = cvcuda
        self.module, self.source_digest = compatibility_module()
        self.axes = {}
        self.records = {}

    def prepare_axis(self, input_size, output_size, axis, channels):
        import numpy as np
        import torch

        key = (input_size, output_size, axis, channels)
        if key in self.axes:
            return self.axes[key]
        starts, counts, weights, precision = axis_coefficients(input_size, output_size)
        basis = torch.eye(input_size, dtype=torch.float32)
        basis = (
            basis[None, :, :, None]
            .expand(1, input_size, input_size, channels)
            .contiguous()
            .cuda()
        )
        size = (output_size, input_size) if axis == 1 else (input_size, output_size)
        stream = self.cvcuda.as_stream(torch.cuda.current_stream())
        resized = self.cvcuda.tvresize(
            self.cvcuda.as_tensor(basis, "NHWC"),
            size=size,
            interp=self.cvcuda.Interp.CUBIC,
            antialias=True,
            stream=stream,
        )
        measured = (
            torch.as_tensor(resized.cuda(), device="cuda")[0, :, :, 0].cpu().numpy()
        )
        if axis == 2:
            measured = measured.T
        expected = np.zeros(measured.shape, dtype=np.float64)
        for destination in range(output_size):
            start = starts[destination]
            count = counts[destination]
            expected[destination, start : start + count] = weights[
                destination, :count
            ] / (1 << precision)
        magnitude = np.sum(np.abs(measured.astype(np.float64)), axis=1)
        support = np.count_nonzero(measured, axis=1)
        operations = 4 * support + 16
        epsilon = np.finfo(np.float32).eps
        gamma = operations * epsilon / (1 - operations * epsilon)
        bounds = (
            255 * (np.sum(np.abs(measured - expected), axis=1) + gamma * magnitude)
            + 1e-4
        )
        if not np.isfinite(bounds).all():
            raise ValueError("Nonfinite CV-CUDA coefficient error bound")
        payload = (
            torch.from_numpy(starts.copy()).cuda(),
            torch.from_numpy(counts.copy()).cuda(),
            torch.from_numpy(weights.copy()).cuda(),
            torch.from_numpy(bounds.astype(np.float32)).cuda(),
            precision,
        )
        self.axes[key] = payload
        self.records[str(key)] = {
            "precision": precision,
            "taps": weights.shape[1],
            "maximum_error_bound": float(bounds.max()),
            "coefficient_sha256": hashlib.sha256(weights.tobytes()).hexdigest(),
            "CV_CUDA_impulse_coefficients_sha256": hashlib.sha256(
                measured.tobytes()
            ).hexdigest(),
            "basis_inputs": (
                "Shape-only identity constants; no timed image/output cache"
            ),
        }
        return payload

    def prepare(self, source_size, target_size, channels=4):
        height, width = source_size
        target_height, target_width = target_size
        if width != target_width:
            self.prepare_axis(width, target_width, 2, channels)
            channels = 3
        if height != target_height:
            self.prepare_axis(height, target_height, 1, channels)

    def __call__(self, source, size):
        import torch

        if (
            source.ndim != 4
            or source.dtype != torch.uint8
            or not source.is_cuda
            or not source.is_contiguous()
            or source.shape[-1] not in (3, 4)
        ):
            raise ValueError("D CV-CUDA requires contiguous CUDA NHWC RGB/RGBX uint8")
        current = source
        owners = [source]
        stream = self.cvcuda.as_stream(torch.cuda.current_stream())
        for axis in (2, 1):
            destination = size[axis - 1]
            if current.shape[axis] == destination:
                continue
            key = (current.shape[axis], destination, axis, current.shape[-1])
            if key not in self.axes:
                raise ValueError(
                    "Resize geometry has not completed untimed CV-CUDA setup"
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
            current = self.module.correct(current, proxy, *self.axes[key], axis)
            owners.extend((floating, resized, proxy, current))
        if current.shape[-1] == 4:
            current = self.module.strip(current)
            owners.append(current)
        result = torch.empty(
            (current.shape[0], 3, current.shape[1], current.shape[2]),
            dtype=torch.uint8,
            device=current.device,
        )
        self.cvcuda.reformat_into(
            self.cvcuda.as_tensor(result, "NCHW"),
            self.cvcuda.as_tensor(current, "NHWC"),
            stream=stream,
        )
        result._d_cvcuda_owners = tuple(owners)
        return result
