// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_bf16.h>
#include <climits>

__global__ void finish_kernel(const unsigned char* source, const float* proxy,
                              const int* starts, const int* counts,
                              const int* coefficients, const float* bounds,
                              __nv_bfloat16* output, const float* weight,
                              const float* bias, int elements,
                              int source_height, int source_width, int channels,
                              int height, int width, int patch, int taps,
                              int precision, int axis) {
  int offset = blockIdx.x * blockDim.x + threadIdx.x;
  if (offset >= elements) return;
  int patch_area = patch * patch;
  int channel = (offset / patch_area) % 3;
  int packed_patch = offset / (3 * patch_area);
  int patches_x = width / patch;
  int patches_per_image = (height / patch) * patches_x;
  int sample = packed_patch / patches_per_image;
  int image_patch = packed_patch % patches_per_image;
  int row = (image_patch / patches_x) * patch + (offset % patch_area) / patch;
  int column = (image_patch % patches_x) * patch + offset % patch;
  int byte_value;
  if (axis == 0) {
    byte_value =
        source[((sample * height + row) * width + column) * channels + channel];
  } else {
    int destination = axis == 2 ? column : row;
    float value =
        proxy[((sample * height + row) * width + column) * channels + channel];
    if (isfinite(value) &&
        fabsf(value - (floorf(value) + 0.5f)) > bounds[destination]) {
      byte_value = int(fminf(fmaxf(floorf(value + 0.5f), 0.f), 255.f));
    } else {
      int accumulation = 1 << (precision - 1);
      for (int tap = 0; tap < counts[destination]; ++tap) {
        int source_row = axis == 1 ? starts[destination] + tap : row;
        int source_column = axis == 2 ? starts[destination] + tap : column;
        int index = ((sample * source_height + source_row) * source_width +
                     source_column) *
                        channels +
                    channel;
        accumulation += source[index] * coefficients[destination * taps + tap];
      }
      byte_value = min(max(accumulation >> precision, 0), 255);
    }
  }
  output[offset] = __float2bfloat16_rn(
      __fmaf_rn(float(byte_value), weight[channel], bias[channel]));
}

torch::Tensor finish_cuda(torch::Tensor source, torch::Tensor proxy,
                          torch::Tensor starts, torch::Tensor counts,
                          torch::Tensor coefficients, torch::Tensor bounds,
                          int precision, int axis, int height, int width,
                          int patch, torch::Tensor weight, torch::Tensor bias) {
  TORCH_CHECK(source.is_cuda() && source.is_contiguous() && source.dim() == 4 &&
              source.scalar_type() == torch::kUInt8);
  TORCH_CHECK(source.size(3) == 3 || source.size(3) == 4);
  TORCH_CHECK(source.numel() > 0 && source.numel() < INT_MAX);
  TORCH_CHECK(height > 0 && width > 0 && patch > 0 && patch <= 256 &&
              height % patch == 0 && width % patch == 0);
  TORCH_CHECK(axis >= 0 && axis <= 2);
  for (auto parameter : {weight, bias})
    TORCH_CHECK(
        parameter.device() == source.device() && parameter.is_contiguous() &&
        parameter.scalar_type() == torch::kFloat32 && parameter.numel() == 3);
  int taps = 0;
  if (axis == 0) {
    TORCH_CHECK(source.size(1) == height && source.size(2) == width);
  } else {
    TORCH_CHECK(proxy.device() == source.device() &&
                proxy.scalar_type() == torch::kFloat32 &&
                proxy.is_contiguous() && proxy.dim() == 4 &&
                proxy.numel() < INT_MAX);
    TORCH_CHECK(proxy.size(0) == source.size(0) && proxy.size(1) == height &&
                proxy.size(2) == width && proxy.size(3) == source.size(3));
    TORCH_CHECK(source.size(3 - axis) == proxy.size(3 - axis));
    TORCH_CHECK(precision > 0 && precision <= 22);
    for (auto parameter : {starts, counts, coefficients})
      TORCH_CHECK(parameter.device() == source.device() &&
                  parameter.is_contiguous() &&
                  parameter.scalar_type() == torch::kInt32);
    TORCH_CHECK(bounds.device() == source.device() && bounds.is_contiguous() &&
                bounds.scalar_type() == torch::kFloat32);
    TORCH_CHECK(coefficients.dim() == 2 &&
                coefficients.size(0) == proxy.size(axis) &&
                coefficients.size(1) > 0);
    TORCH_CHECK(starts.numel() == proxy.size(axis) &&
                counts.numel() == proxy.size(axis) &&
                bounds.numel() == proxy.size(axis));
    taps = int(coefficients.size(1));
  }
  c10::cuda::CUDAGuard guard(source.device());
  auto output =
      torch::empty({1, source.size(0) * (height / patch) * (width / patch),
                    3 * patch * patch},
                   source.options().dtype(torch::kBFloat16));
  TORCH_CHECK(output.numel() < INT_MAX);
  int elements = int(output.numel());
  finish_kernel<<<(elements + 255) / 256, 256, 0,
                  at::cuda::getCurrentCUDAStream()>>>(
      source.data_ptr<unsigned char>(), proxy.data_ptr<float>(),
      starts.data_ptr<int>(), counts.data_ptr<int>(),
      coefficients.data_ptr<int>(), bounds.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr()),
      weight.data_ptr<float>(), bias.data_ptr<float>(), elements,
      int(source.size(1)), int(source.size(2)), int(source.size(3)), height,
      width, patch, taps, precision, axis);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return output;
}
