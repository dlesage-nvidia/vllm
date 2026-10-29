// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <climits>

__global__ void correction_kernel(const unsigned char* source,
                                  const float* proxy, const int* starts,
                                  const int* counts, const int* weights,
                                  const float* bounds, unsigned char* output,
                                  int elements, int height, int width,
                                  int source_height, int source_width,
                                  int channels, int taps, int precision,
                                  int axis) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= elements) return;
  int channel = offset % 3;
  int pixel = offset / 3;
  int column = pixel % width;
  int row = (pixel / width) % height;
  int sample = pixel / (height * width);
  int destination = axis == 2 ? column : row;
  float value = proxy[pixel * channels + channel];
  float rounded = floorf(value + 0.5f);
  if (isfinite(value) &&
      fabsf(value - (floorf(value) + 0.5f)) > bounds[destination]) {
    output[offset] =
        static_cast<unsigned char>(fminf(fmaxf(rounded, 0.f), 255.f));
    return;
  }
  int accumulation = 1 << (precision - 1);
  int start = starts[destination];
  int count = counts[destination];
  for (int tap = 0; tap < count; ++tap) {
    int source_row = axis == 1 ? start + tap : row;
    int source_column = axis == 2 ? start + tap : column;
    int index =
        ((sample * source_height + source_row) * source_width + source_column) *
            channels +
        channel;
    accumulation += source[index] * weights[destination * taps + tap];
  }
  int result = accumulation >> precision;
  output[offset] = static_cast<unsigned char>(min(max(result, 0), 255));
}

__global__ void strip_kernel(const unsigned char* source, unsigned char* output,
                             int elements) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset < elements) output[offset] = source[(offset / 3) * 4 + offset % 3];
}

torch::Tensor correct_cuda(torch::Tensor source, torch::Tensor proxy,
                           torch::Tensor starts, torch::Tensor counts,
                           torch::Tensor weights, torch::Tensor bounds,
                           int precision, int axis) {
  TORCH_CHECK(source.is_cuda() && source.is_contiguous() && source.dim() == 4 &&
              source.scalar_type() == torch::kUInt8);
  TORCH_CHECK(proxy.is_contiguous() && proxy.dim() == 4 &&
              proxy.scalar_type() == torch::kFloat32 &&
              proxy.device() == source.device());
  TORCH_CHECK(source.size(0) == proxy.size(0) &&
              source.size(3) == proxy.size(3) &&
              (source.size(3) == 3 || source.size(3) == 4));
  TORCH_CHECK((axis == 1 || axis == 2) && precision > 0 && precision <= 22);
  TORCH_CHECK(source.size(3 - axis) == proxy.size(3 - axis));
  for (auto parameter : {starts, counts, weights})
    TORCH_CHECK(parameter.device() == source.device() &&
                parameter.is_contiguous() &&
                parameter.scalar_type() == torch::kInt32);
  TORCH_CHECK(bounds.device() == source.device() && bounds.is_contiguous() &&
              bounds.scalar_type() == torch::kFloat32);
  TORCH_CHECK(weights.dim() == 2 && weights.size(0) == proxy.size(axis) &&
              weights.size(1) > 0);
  TORCH_CHECK(starts.numel() == proxy.size(axis) &&
              counts.numel() == proxy.size(axis) &&
              bounds.numel() == proxy.size(axis));
  TORCH_CHECK(proxy.numel() > 0 && proxy.numel() < INT_MAX &&
              source.numel() < INT_MAX);
  c10::cuda::CUDAGuard guard(source.device());
  auto output = torch::empty({proxy.size(0), proxy.size(1), proxy.size(2), 3},
                             source.options());
  int elements = int(output.numel());
  correction_kernel<<<(elements + 255) / 256, 256, 0,
                      at::cuda::getCurrentCUDAStream()>>>(
      source.data_ptr<unsigned char>(), proxy.data_ptr<float>(),
      starts.data_ptr<int>(), counts.data_ptr<int>(), weights.data_ptr<int>(),
      bounds.data_ptr<float>(), output.data_ptr<unsigned char>(), elements,
      int(proxy.size(1)), int(proxy.size(2)), int(source.size(1)),
      int(source.size(2)), int(source.size(3)), int(weights.size(1)), precision,
      axis);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}

torch::Tensor strip_cuda(torch::Tensor source) {
  TORCH_CHECK(source.is_cuda() && source.is_contiguous() && source.dim() == 4 &&
              source.size(3) == 4 && source.scalar_type() == torch::kUInt8);
  TORCH_CHECK(source.numel() > 0 && source.numel() < INT_MAX);
  c10::cuda::CUDAGuard guard(source.device());
  auto output = torch::empty(
      {source.size(0), source.size(1), source.size(2), 3}, source.options());
  int elements = int(output.numel());
  strip_kernel<<<(elements + 255) / 256, 256, 0,
                 at::cuda::getCurrentCUDAStream()>>>(
      source.data_ptr<unsigned char>(), output.data_ptr<unsigned char>(),
      elements);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
