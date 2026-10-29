# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Untimed setup for source-bundled, byte-qualified portable encoder kernels."""

import functools
import hashlib
from pathlib import Path

GELU_AVAILABLE = True
TORCH_GIT = "cf30153c4c131c8164ee7798e5022d810682e2cb"


@functools.cache
def compiled_module(capability, gelu):
    from torch.utils.cpp_extension import load

    if len(capability) != 2 or capability[0] < 8:
        raise ValueError("Portable BF16 kernels require compute capability >= 8")
    root = Path(__file__).parent
    paths = [root / "_leads.cpp", root / "_scalar_norm.cu"]
    if GELU_AVAILABLE:
        paths.append(root / "_gelu.cu")
    digest = hashlib.sha256(b"".join(path.read_bytes() for path in paths)).hexdigest()
    target = "".join(map(str, capability))
    module = load(
        name="nemotron_pr_portable_" + digest[:16] + "_sm" + target,
        sources=[str(path) for path in paths],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-O3",
            "--fmad=false",
            "-gencode=arch=compute_" + target + ",code=sm_" + target,
        ],
        with_cuda=True,
    )
    return module, digest


class PortableLeads:
    def __init__(self, gelu=False):
        import torch

        if torch.version.git_version != TORCH_GIT:
            raise ValueError("Portable encoder requires the qualified native runtime")
        if gelu and not GELU_AVAILABLE:
            raise ValueError("Exact GELU has not been added to this staged package")
        self.module, self.digest = compiled_module(
            torch.cuda.get_device_capability(), gelu
        )
        self.table = None
        self.table_gate = None
        if gelu:
            codes = torch.arange(65536, dtype=torch.int32, device="cuda").to(
                torch.int16
            )
            inputs = codes.view(torch.bfloat16)
            expected = torch.nn.functional.gelu(inputs, approximate="none")
            self.table = expected.view(torch.int16).clone()
            finite = torch.isfinite(inputs)
            candidate = self.module.gelu(inputs[finite], self.table)
            reference = expected[finite]
            if not torch.isfinite(candidate).all() or not torch.equal(
                candidate.view(torch.int16), reference.view(torch.int16)
            ):
                raise ValueError("Exact GELU fails exhaustive finite BF16 byte parity")
            self.table_gate = {
                "elements": reference.numel(),
                "bit_exact": True,
                "signed_zero_bits_compared": True,
                "all_finite_bf16_codes": True,
            }
        self.ready = torch.cuda.Event()
        self.ready.record()

    def norm(self, residual, branch, weight, bias, epsilon):
        import torch

        torch.cuda.current_stream(residual.device).wait_event(self.ready)
        return self.module.norm(residual, branch, weight, bias, epsilon)

    def gelu(self, inputs):
        import torch

        if self.table is None:
            raise ValueError("Exact GELU is disabled")
        torch.cuda.current_stream(inputs.device).wait_event(self.ready)
        return self.module.gelu(inputs, self.table)
