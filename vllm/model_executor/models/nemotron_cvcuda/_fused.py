# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in B-exact resize/packing/normalization with completed-scratch retirement."""

import functools
import hashlib
import weakref
from pathlib import Path

from ._fast import PackedExactResize


@functools.lru_cache(maxsize=1)
def fused_module():
    from torch.utils.cpp_extension import load

    paths = [Path(__file__).with_suffix(suffix) for suffix in (".cpp", ".cu")]
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    return load(
        name="nemotron_B_exact_cvcuda_normalized_finish_" + digest[:16],
        sources=[str(path) for path in paths],
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3"],
        with_cuda=True,
    ), digest


class TrustedNormalized:
    def __init__(self):
        self.values: weakref.WeakValueDictionary[int, object] = (
            weakref.WeakValueDictionary()
        )

    def add(self, pixels):
        self.values[id(pixels)] = pixels
        return pixels

    def contains(self, pixels):
        return pixels is not None and self.values.get(id(pixels)) is pixels


class NormalizedFinisher:
    def __init__(self, module, weight, bias):
        self.module = module
        self.weight = weight
        self.bias = bias

    def finish(self, *arguments):
        return self.module.finish(*arguments, self.weight, self.bias)


class NormalizedPackedResize(PackedExactResize):
    def __init__(self, cvcuda, patch, weight, bias):
        super().__init__(cvcuda, patch)
        module, self.finish_digest = fused_module()
        self.finish = NormalizedFinisher(module, weight, bias)
