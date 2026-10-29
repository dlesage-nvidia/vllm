# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from pathlib import Path


class CudaRuntimeCopy:
    def __init__(self):
        libraries = {
            fields[-1]
            for line in Path("/proc/self/maps").read_text().splitlines()
            if len(fields := line.split()) >= 6
            and Path(fields[-1]).name.startswith("libcudart.so")
        }
        if len(libraries) != 1:
            raise RuntimeError("require exactly one already-loaded CUDA runtime")
        self.path = libraries.pop()
        self.library = ctypes.CDLL(self.path)
        self.library.cudaMemcpyAsync.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.cudaMemcpyAsync.restype = ctypes.c_int
        self.library.cudaGetErrorString.argtypes = [ctypes.c_int]
        self.library.cudaGetErrorString.restype = ctypes.c_char_p

    def __call__(self, destination, source, byte_count, stream):
        status = self.library.cudaMemcpyAsync(
            destination, source, byte_count, 1, stream
        )
        if status:
            message = self.library.cudaGetErrorString(status)
            raise RuntimeError(f"RGBX cudaMemcpyAsync failed ({status}): {message!r}")
