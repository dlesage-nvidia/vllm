// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

torch::Tensor finish_cuda(torch::Tensor source, torch::Tensor proxy,
                          torch::Tensor starts, torch::Tensor counts,
                          torch::Tensor coefficients, torch::Tensor bounds,
                          int precision, int axis, int height, int width,
                          int patch, torch::Tensor weight, torch::Tensor bias);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("finish", &finish_cuda);
}
