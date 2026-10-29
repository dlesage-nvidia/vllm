// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

torch::Tensor correct_cuda(torch::Tensor source, torch::Tensor proxy,
                           torch::Tensor starts, torch::Tensor counts,
                           torch::Tensor weights, torch::Tensor bounds,
                           int precision, int axis);
torch::Tensor strip_cuda(torch::Tensor source);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("correct", &correct_cuda);
  module.def("strip", &strip_cuda);
}
