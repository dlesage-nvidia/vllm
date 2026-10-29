// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// All rights reserved. SPDX-License-Identifier: Apache-2.0

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

using F32x2 = unsigned long long;

struct VectorState {
  F32x2 mean, m2;
  float count;
};
struct VectorMoments {
  F32x2 mean, m2;
};
struct ScalarMoments {
  float mean, m2;
};

__device__ __forceinline__ F32x2 join_bits(uint32_t lo, uint32_t hi) {
  return F32x2(lo) | (F32x2(hi) << 32);
}
__device__ __forceinline__ F32x2 join_float(float lo, float hi) {
  return join_bits(__float_as_uint(lo), __float_as_uint(hi));
}
__device__ __forceinline__ F32x2 splat(float x) {
  uint32_t bits = __float_as_uint(x);
  return join_bits(bits, bits);
}
__device__ __forceinline__ float first_float(F32x2 x) {
  return __uint_as_float(uint32_t(x));
}
__device__ __forceinline__ float second_float(F32x2 x) {
  return __uint_as_float(uint32_t(x >> 32));
}
__device__ __forceinline__ float low_bf16(uint32_t x) {
  return __uint_as_float(x << 16);
}
__device__ __forceinline__ float high_bf16(uint32_t x) {
  return __uint_as_float(x & 0xffff0000u);
}
__device__ __forceinline__ F32x2 widen_pair(uint32_t x) {
  return join_bits(x << 16, x & 0xffff0000u);
}

__device__ __forceinline__ F32x2 add2(F32x2 first, F32x2 second) {
  return join_float(__fadd_rn(first_float(first), first_float(second)),
                    __fadd_rn(second_float(first), second_float(second)));
}
__device__ __forceinline__ F32x2 sub2(F32x2 first, F32x2 second) {
  return join_float(__fsub_rn(first_float(first), first_float(second)),
                    __fsub_rn(second_float(first), second_float(second)));
}
__device__ __forceinline__ F32x2 mul2(F32x2 first, F32x2 second) {
  return join_float(__fmul_rn(first_float(first), first_float(second)),
                    __fmul_rn(second_float(first), second_float(second)));
}
__device__ __forceinline__ F32x2 fma2(F32x2 first, F32x2 second, F32x2 third) {
  return join_float(
      __fmaf_rn(first_float(first), first_float(second), first_float(third)),
      __fmaf_rn(second_float(first), second_float(second),
                second_float(third)));
}
__device__ __forceinline__ F32x2 shuffle_down16(F32x2 x, int delta) {
  uint32_t lo = __shfl_down_sync(0xffffffff, uint32_t(x), delta, 16);
  uint32_t hi = __shfl_down_sync(0xffffffff, uint32_t(x >> 32), delta, 16);
  return join_bits(lo, hi);
}

__device__ __forceinline__ uint32_t add_bf16_pair(uint32_t a, uint32_t b) {
  const auto low = __float2bfloat16_rn(__fadd_rn(low_bf16(a), low_bf16(b)));
  const auto high = __float2bfloat16_rn(__fadd_rn(high_bf16(a), high_bf16(b)));
  return uint32_t(__bfloat16_as_ushort(low)) |
         (uint32_t(__bfloat16_as_ushort(high)) << 16);
}
__device__ __forceinline__ uint4 add_eight(uint4 a, uint4 b) {
  return make_uint4(add_bf16_pair(a.x, b.x), add_bf16_pair(a.y, b.y),
                    add_bf16_pair(a.z, b.z), add_bf16_pair(a.w, b.w));
}

__device__ __forceinline__ void update2(VectorState& s, F32x2 x) {
  F32x2 delta = sub2(x, s.mean);
  s.count = __fadd_rn(s.count, 1.f);
  F32x2 reciprocal = splat(__frcp_rn(s.count));
  s.mean = fma2(delta, reciprocal, s.mean);
  s.m2 = fma2(delta, sub2(x, s.mean), s.m2);
}

__device__ __forceinline__ void update_two_threads(VectorState& s, uint4 x) {
  update2(s, join_float(low_bf16(x.x), low_bf16(x.z)));
  update2(s, join_float(high_bf16(x.x), high_bf16(x.z)));
  update2(s, join_float(low_bf16(x.y), low_bf16(x.w)));
  update2(s, join_float(high_bf16(x.y), high_bf16(x.w)));
}

__device__ __forceinline__ VectorMoments combine_equal2(VectorMoments b,
                                                        VectorMoments a,
                                                        float count) {
  F32x2 half = splat(0.5f);
  F32x2 delta = sub2(b.mean, a.mean);
  F32x2 mean = fma2(half, a.mean, mul2(half, b.mean));
  F32x2 square = mul2(mul2(delta, delta), splat(count));
  F32x2 m2 = fma2(half, square, add2(a.m2, b.m2));
  return {mean, m2};
}
__device__ __forceinline__ ScalarMoments combine_equal(ScalarMoments b,
                                                       ScalarMoments a,
                                                       float count) {
  float delta = b.mean - a.mean;
  float mean = __fmaf_rn(0.5f, a.mean, __fmul_rn(0.5f, b.mean));
  float square = __fmul_rn(__fmul_rn(delta, delta), count);
  float m2 = __fmaf_rn(0.5f, square, __fadd_rn(a.m2, b.m2));
  return {mean, m2};
}
__device__ __forceinline__ ScalarMoments combine_unequal(ScalarMoments b,
                                                         ScalarMoments a) {
  float delta = b.mean - a.mean;
  float mean = __fmaf_rn(0.4f, a.mean, __fmul_rn(0.6f, b.mean));
  float square = __fmul_rn(__fmul_rn(delta, delta), 256.f);
  float m2 = __fmaf_rn(0.6f, square, __fadd_rn(a.m2, b.m2));
  return {mean, m2};
}

__device__ __forceinline__ uint32_t affine_pair(uint32_t x, uint32_t weight,
                                                uint32_t bias, F32x2 mean,
                                                F32x2 rstd) {
  F32x2 centered = sub2(widen_pair(x), mean);
  F32x2 scaled = mul2(centered, rstd);
  F32x2 result = fma2(scaled, widen_pair(weight), widen_pair(bias));
  float lo = first_float(result);
  float hi = second_float(result);
  return uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(lo))) |
         (uint32_t(__bfloat16_as_ushort(__float2bfloat16_rn(hi))) << 16);
}
__device__ __forceinline__ uint4 affine_eight(uint4 x, uint4 weight, uint4 bias,
                                              F32x2 mean, F32x2 rstd) {
  return make_uint4(affine_pair(x.x, weight.x, bias.x, mean, rstd),
                    affine_pair(x.y, weight.y, bias.y, mean, rstd),
                    affine_pair(x.z, weight.z, bias.z, mean, rstd),
                    affine_pair(x.w, weight.w, bias.w, mean, rstd));
}

__global__ __launch_bounds__(128) void scalar_pairs_residual_norm(
    const uint4* __restrict__ a, const uint4* __restrict__ b,
    const uint4* __restrict__ weight, const uint4* __restrict__ bias,
    uint4* __restrict__ rounded, uint4* __restrict__ normalized, int64_t rows,
    float eps) {
  const int lane = threadIdx.x & 31;
  const int64_t row = int64_t(blockIdx.x) * 4 + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int64_t base = row * 160;

  uint4 values_first[3];
  uint4 values_second[2];
  VectorState s01 = {0ull, 0ull, 0.f};
  VectorState s23 = {0ull, 0ull, 0.f};

#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int col = lane + 64 * k;
    uint4 v01 = add_eight(a[base + col], b[base + col]);
    values_first[k] = v01;
    rounded[base + col] = v01;
    update_two_threads(s01, v01);
    if (k < 2) {
      uint4 v23 = add_eight(a[base + col + 32], b[base + col + 32]);
      values_second[k] = v23;
      rounded[base + col + 32] = v23;
      update_two_threads(s23, v23);
    }
  }

  VectorMoments p01 = {s01.mean, s01.m2};
  VectorMoments p23 = {s23.mean, s23.m2};
#pragma unroll
  for (int stage = 0; stage < 4; ++stage) {
    const int shift = 8 >> stage;
    VectorMoments other01 = {shuffle_down16(p01.mean, shift),
                             shuffle_down16(p01.m2, shift)};
    VectorMoments other23 = {shuffle_down16(p23.mean, shift),
                             shuffle_down16(p23.m2, shift)};
    p01 = combine_equal2(p01, other01, float(12 << stage));
    p23 = combine_equal2(p23, other23, float(8 << stage));
  }

  ScalarMoments even01 = {first_float(p01.mean), first_float(p01.m2)};
  ScalarMoments odd01 = {second_float(p01.mean), second_float(p01.m2)};
  ScalarMoments even23 = {first_float(p23.mean), first_float(p23.m2)};
  ScalarMoments odd23 = {second_float(p23.mean), second_float(p23.m2)};
  ScalarMoments first_native = combine_equal(even01, odd01, 192.f);
  ScalarMoments second_native = combine_equal(even23, odd23, 128.f);

  ScalarMoments half = combine_unequal(first_native, second_native);
  ScalarMoments partner = {__shfl_down_sync(0xffffffff, half.mean, 16),
                           __shfl_down_sync(0xffffffff, half.m2, 16)};
  ScalarMoments total = combine_equal(half, partner, 640.f);
  float mean = __shfl_sync(0xffffffff, total.mean, 0);
  float variance = __shfl_sync(0xffffffff, total.m2, 0);
  float rstd = rsqrtf(__fdiv_rn(variance, 1280.f) + eps);
  F32x2 mean2 = splat(mean);
  F32x2 rstd2 = splat(rstd);

#pragma unroll
  for (int k = 0; k < 3; ++k) {
    const int col = lane + 64 * k;
    normalized[base + col] =
        affine_eight(values_first[k], weight[col], bias[col], mean2, rstd2);
    if (k < 2) {
      normalized[base + col + 32] = affine_eight(
          values_second[k], weight[col + 32], bias[col + 32], mean2, rstd2);
    }
  }
}

void launch_plain(const void* a, const void* b, const void* weight,
                  const void* bias, void* rounded, void* normalized,
                  int64_t rows, float eps, cudaStream_t stream) {
  scalar_pairs_residual_norm<<<(rows + 3) / 4, 128, 0, stream>>>(
      static_cast<const uint4*>(a), static_cast<const uint4*>(b),
      static_cast<const uint4*>(weight), static_cast<const uint4*>(bias),
      static_cast<uint4*>(rounded), static_cast<uint4*>(normalized), rows, eps);
}
