// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

__global__ void lookup_gelu(const uint16_t* __restrict__ source,
                            const uint16_t* __restrict__ table,
                            uint16_t* __restrict__ output, int64_t elements) {
  const int64_t offset = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x) * 8;
  if (offset + 8 <= elements) {
    const uint4 packed = *reinterpret_cast<const uint4*>(source + offset);
    uint4 result;
    const uint32_t* input_words = reinterpret_cast<const uint32_t*>(&packed);
    uint32_t* output_words = reinterpret_cast<uint32_t*>(&result);
#pragma unroll
    for (int index = 0; index < 4; ++index) {
      const uint32_t word = input_words[index];
      output_words[index] =
          uint32_t(table[word & 65535]) | (uint32_t(table[word >> 16]) << 16);
    }
    *reinterpret_cast<uint4*>(output + offset) = result;
  } else {
    for (int64_t index = offset; index < elements; ++index) {
      output[index] = table[source[index]];
    }
  }
}

torch::Tensor portable_gelu(torch::Tensor inputs, torch::Tensor table) {
  TORCH_CHECK(
      inputs.is_cuda() && inputs.scalar_type() == at::kBFloat16 &&
          inputs.is_contiguous() && table.device() == inputs.device() &&
          table.scalar_type() == at::kShort && table.dim() == 1 &&
          table.numel() == 65536 && table.is_contiguous() &&
          reinterpret_cast<uintptr_t>(inputs.data_ptr()) % 16 == 0,
      "GELU requires aligned contiguous CUDA BF16 inputs and a BF16-bit table");
  c10::cuda::CUDAGuard guard(inputs.device());
  auto output = torch::empty_like(inputs);
  if (inputs.numel()) {
    lookup_gelu<<<(inputs.numel() + 2047) / 2048, 256, 0,
                  c10::cuda::getCurrentCUDAStream()>>>(
        static_cast<const uint16_t*>(inputs.data_ptr()),
        static_cast<const uint16_t*>(table.data_ptr()),
        static_cast<uint16_t*>(output.data_ptr()), inputs.numel());
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  return output;
}
