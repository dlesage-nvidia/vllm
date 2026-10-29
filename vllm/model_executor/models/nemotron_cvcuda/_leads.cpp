// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cmath>

void launch_plain(const void*, const void*, const void*, const void*, void*,
                  void*, int64_t, float, cudaStream_t);
torch::Tensor portable_gelu(torch::Tensor, torch::Tensor);

std::tuple<torch::Tensor, torch::Tensor> portable_norm(torch::Tensor residual,
                                                       torch::Tensor branch,
                                                       torch::Tensor weight,
                                                       torch::Tensor bias,
                                                       double epsilon) {
  TORCH_CHECK(residual.is_cuda() && residual.scalar_type() == at::kBFloat16 &&
                  residual.is_contiguous() && residual.dim() >= 2 &&
                  residual.size(-1) == 1280 && residual.numel() > 0,
              "Expected nonempty contiguous CUDA BF16 width1280 residual");
  for (const auto& tensor : {branch, weight, bias}) {
    TORCH_CHECK(tensor.device() == residual.device() &&
                    tensor.scalar_type() == at::kBFloat16 &&
                    tensor.is_contiguous() &&
                    reinterpret_cast<uintptr_t>(tensor.data_ptr()) % 16 == 0,
                "Expected aligned same-device contiguous BF16 norm tensors");
  }
  TORCH_CHECK(reinterpret_cast<uintptr_t>(residual.data_ptr()) % 16 == 0 &&
                  branch.sizes() == residual.sizes() && weight.dim() == 1 &&
                  bias.dim() == 1 && weight.numel() == 1280 &&
                  bias.numel() == 1280 && std::isfinite(epsilon) &&
                  epsilon > 0 && std::isfinite(static_cast<float>(epsilon)) &&
                  static_cast<float>(epsilon) > 0,
              "Invalid norm geometry or epsilon");
  c10::cuda::CUDAGuard guard(residual.device());
  auto rounded = torch::empty_like(residual);
  auto normalized = torch::empty_like(residual);
  launch_plain(residual.data_ptr(), branch.data_ptr(), weight.data_ptr(),
               bias.data_ptr(), rounded.data_ptr(), normalized.data_ptr(),
               residual.numel() / 1280, static_cast<float>(epsilon),
               c10::cuda::getCurrentCUDAStream());
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return {rounded, normalized};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("norm", &portable_norm);
  module.def("gelu", &portable_gelu);
}
