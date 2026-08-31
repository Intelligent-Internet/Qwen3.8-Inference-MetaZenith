// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {

constexpr int kNumQueryHeads = 24;
constexpr int kNumKvHeads = 4;
constexpr int kHeadDim = 256;
constexpr int kBlockSize = 16;
constexpr int kSlotSize = 268;
constexpr int kPackedValueOffset = 128;
constexpr int kKeyNormOffset = 256;
constexpr int kValueScaleOffset = 258;
constexpr int kValueZeroOffset = 260;
constexpr int kCachedInvNormOffset = 264;
static_assert(kPackedValueOffset + kHeadDim / 2 == kKeyNormOffset);
static_assert(kKeyNormOffset + sizeof(std::uint16_t) == kValueScaleOffset);
static_assert(kCachedInvNormOffset + sizeof(float) == kSlotSize);
constexpr int kNumSplits = 32;
constexpr int kHeadWarps = 3;
constexpr int kHeadsPerWarp = 2;
constexpr float kAttentionScale = 0.0625f;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta >= 1; delta >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, delta);
  }
  return value;
}

__device__ __forceinline__ float lane_reduce8(const float (&value)[8]) {
  float result = value[0] + value[1];
#pragma unroll
  for (int i = 2; i < 8; ++i) {
    result += value[i];
  }
  return result;
}

__device__ __forceinline__ float unpack_half(const uint8_t* ptr) {
  __half_raw raw;
  raw.x = static_cast<unsigned short>(ptr[0]) |
          (static_cast<unsigned short>(ptr[1]) << 8);
  return __half2float(raw);
}

__device__ __forceinline__ float triton_exp(float value) {
  const float scaled = value * 1.4426950408889634074f;
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(scaled));
  return result;
}

__device__ __forceinline__ float triton_div(float numerator,
                                             float denominator) {
  float result;
  asm("div.full.f32 %0, %1, %2;"
      : "=f"(result)
      : "f"(numerator), "f"(denominator));
  return result;
}

// Each CTA handles all six query heads sharing one KV head and split.  Loader
// warps stage an even token tile in parallel; the first three warps then apply
// its two-token online-softmax updates in the exact order used by the Triton
// reference.  The tile is chosen from the static CUDA-graph batch size so the
// total loader parallelism remains high without changing any attention math.
template <int kTileTokens>
__global__ void turboquant_head_parallel_stage1_kernel(
    const float* __restrict__ query, const uint8_t* __restrict__ cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ centroids, float* __restrict__ output,
    int64_t cache_block_stride, int64_t cache_position_stride,
    int64_t cache_head_stride, int64_t table_stride) {
  static_assert(kTileTokens >= 6 && kTileTokens % 2 == 0);
  const int batch = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool compute_warp = warp < kHeadWarps;

  const int seq_len = seq_lens[batch];
  const int split_len = (seq_len + kNumSplits - 1) / kNumSplits;
  const int split_start = split_len * split;
  const int split_end = min(split_start + split_len, seq_len);
  const int split_tokens = max(split_end - split_start, 0);

  float q[kHeadsPerWarp][8];
  float accum[kHeadsPerWarp][8];
#pragma unroll
  for (int h = 0; h < kHeadsPerWarp; ++h) {
    const int local_head = warp * kHeadsPerWarp + h;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + i * 32;
      q[h][i] = compute_warp
                    ? query[(batch * kNumQueryHeads +
                             kv_head * (kHeadWarps * kHeadsPerWarp) +
                             local_head) *
                                kHeadDim +
                            d]
                    : 0.0f;
      accum[h][i] = 0.0f;
    }
  }

  const float negative_infinity = __int_as_float(0xff800000);
  float m[kHeadsPerWarp];
  float l[kHeadsPerWarp];
#pragma unroll
  for (int h = 0; h < kHeadsPerWarp; ++h) {
    m[h] = negative_infinity;
    l[h] = 0.0f;
  }

  __shared__ float shared_key[kTileTokens][kHeadDim];
  __shared__ float shared_value[kTileTokens][kHeadDim];
  __shared__ float shared_key_norm[kTileTokens];

  for (int tile = 0; tile * kTileTokens < split_tokens; ++tile) {
    const int token_in_tile = warp;
    const int position = split_start + tile * kTileTokens + token_in_tile;
    const bool valid = position < split_end;
    const int page = position / kBlockSize;
    const int page_offset = position - page * kBlockSize;
    const int block = valid
                          ? block_table[batch * table_stride + page]
                          : 0;
    const uint8_t* slot =
        cache + static_cast<int64_t>(block) * cache_block_stride +
        static_cast<int64_t>(page_offset) * cache_position_stride +
        static_cast<int64_t>(kv_head) * cache_head_stride;

    const float value_scale =
        valid ? unpack_half(slot + kValueScaleOffset) : 0.0f;
    const float value_zero =
        valid ? unpack_half(slot + kValueZeroOffset) : 0.0f;
    float key_part[8];
    float value_part[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + i * 32;
      const uint8_t key_byte = valid ? slot[d >> 1] : 0;
      const int key_idx = (key_byte >> ((d & 1) * 4)) & 15;
      key_part[i] = valid ? centroids[key_idx] : 0.0f;
      const uint8_t value_byte =
          valid ? slot[kPackedValueOffset + (d >> 1)] : 0;
      const float value_idx = static_cast<float>(
          (value_byte >> ((d & 1) * 4)) & 15);
      value_part[i] = value_idx * value_scale + value_zero;
    }

    const float inv_norm =
        valid
        ? *reinterpret_cast<const float*>(slot + kCachedInvNormOffset)
        : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + i * 32;
      shared_key[token_in_tile][d] = key_part[i] * inv_norm;
      shared_value[token_in_tile][d] = value_part[i];
    }
    if (lane == 0) {
      shared_key_norm[token_in_tile] =
          valid ? unpack_half(slot + kKeyNormOffset) : 0.0f;
    }
    __syncthreads();

    if (compute_warp) {
#pragma unroll
      for (int pair = 0; pair < kTileTokens / 2; ++pair) {
        const int token0 = pair * 2;
        const int token1 = token0 + 1;
        const bool valid0 =
            tile * kTileTokens + token0 < split_tokens;
        const bool valid1 =
            tile * kTileTokens + token1 < split_tokens;
        float score0[kHeadsPerWarp];
        float score1[kHeadsPerWarp];
#pragma unroll
        for (int h = 0; h < kHeadsPerWarp; ++h) {
          float products0[8];
          float products1[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            products0[i] = q[h][i] * shared_key[token0][d];
            products1[i] = q[h][i] * shared_key[token1][d];
          }
          float s0 = warp_sum(lane_reduce8(products0));
          float s1 = warp_sum(lane_reduce8(products1));
          s0 = s0 * shared_key_norm[token0] * kAttentionScale;
          s1 = s1 * shared_key_norm[token1] * kAttentionScale;
          score0[h] = valid0 ? s0 : negative_infinity;
          score1[h] = valid1 ? s1 : negative_infinity;
        }

#pragma unroll
        for (int h = 0; h < kHeadsPerWarp; ++h) {
          const float next_m = fmaxf(fmaxf(score0[h], score1[h]), m[h]);
          const float rescale = triton_exp(m[h] - next_m);
          const float p0 = triton_exp(score0[h] - next_m);
          const float p1 = triton_exp(score1[h] - next_m);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            const float weighted = p0 * shared_value[token0][d] +
                                   p1 * shared_value[token1][d];
            accum[h][i] = accum[h][i] * rescale + weighted;
          }
          l[h] = l[h] * rescale + (p0 + p1);
          m[h] = next_m;
        }
      }
    }
    __syncthreads();
  }

  if (compute_warp) {
#pragma unroll
    for (int h = 0; h < kHeadsPerWarp; ++h) {
      const int local_head = warp * kHeadsPerWarp + h;
      const float safe_l = l[h] > 0.0f ? l[h] : 1.0f;
      float* out_head =
          output + ((batch * kNumQueryHeads +
                     kv_head * (kHeadWarps * kHeadsPerWarp) + local_head) *
                        kNumSplits +
                    split) *
                       (kHeadDim + 1);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int d = lane + i * 32;
        out_head[d] = triton_div(accum[h][i], safe_l);
      }
      if (lane == 0) {
        out_head[kHeadDim] = m[h] + logf(safe_l);
      }
    }
  }
}

// Producer/consumer form of the exact head-parallel kernel.  Dedicated loader
// warps prepare the following token tile while the first three warps execute
// the unchanged per-head attention math on the current tile.  The two
// CTA-wide barriers form the buffer-ready/buffer-released handoff.
template <int kTileTokens, int kActiveSplits = kNumSplits,
          bool kBoundaryOnly = false>
__global__ void turboquant_head_parallel_stage1_pipeline_kernel(
    const float* __restrict__ query, const uint8_t* __restrict__ cache,
    const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_lens,
    const float* __restrict__ centroids, float* __restrict__ output,
    int64_t cache_block_stride, int64_t cache_position_stride,
    int64_t cache_head_stride, int64_t table_stride,
    const int32_t* __restrict__ table_mismatch) {
  static_assert(kTileTokens >= 4 && kTileTokens % 2 == 0);
  const int batch = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool compute_warp = warp < kHeadWarps;
  const int loader_token = warp - kHeadWarps;

  // Complement the synthetic-row shared kernel.  The preceding proof launch
  // folds physical-page identity and authoritative split width into one
  // immutable stream-ordered flag, so every thread makes the same decision.
  if constexpr (kBoundaryOnly) {
    if (table_mismatch[0] == 0) {
      return;
    }
  }

  const int seq_len = seq_lens[batch];
  const int split_len = (seq_len + kActiveSplits - 1) / kActiveSplits;
  const int split_start = split_len * split;
  const int split_end = min(split_start + split_len, seq_len);
  const int split_tokens = max(split_end - split_start, 0);
  const int num_tiles =
      (split_tokens + kTileTokens - 1) / kTileTokens;

  float q[kHeadsPerWarp][8];
  float accum[kHeadsPerWarp][8];
#pragma unroll
  for (int h = 0; h < kHeadsPerWarp; ++h) {
    const int local_head = warp * kHeadsPerWarp + h;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + i * 32;
      q[h][i] = compute_warp
                    ? query[(batch * kNumQueryHeads +
                             kv_head * (kHeadWarps * kHeadsPerWarp) +
                             local_head) *
                                kHeadDim +
                            d]
                    : 0.0f;
      accum[h][i] = 0.0f;
    }
  }

  const float negative_infinity = __int_as_float(0xff800000);
  float m[kHeadsPerWarp];
  float l[kHeadsPerWarp];
#pragma unroll
  for (int h = 0; h < kHeadsPerWarp; ++h) {
    m[h] = negative_infinity;
    l[h] = 0.0f;
  }

  __shared__ float shared_key[2][kTileTokens][kHeadDim];
  __shared__ float shared_value[2][kTileTokens][kHeadDim];
  __shared__ float shared_key_norm[2][kTileTokens];

  auto load_tile = [&](int tile, int buffer) {
    if (loader_token < 0 || loader_token >= kTileTokens) {
      return;
    }
    const int position =
        split_start + tile * kTileTokens + loader_token;
    const bool valid = position < split_end;
    const int page = position / kBlockSize;
    const int page_offset = position - page * kBlockSize;
    const int block = valid
                          ? block_table[batch * table_stride + page]
                          : 0;
    const uint8_t* slot =
        cache + static_cast<int64_t>(block) * cache_block_stride +
        static_cast<int64_t>(page_offset) * cache_position_stride +
        static_cast<int64_t>(kv_head) * cache_head_stride;

    const float value_scale =
        valid ? unpack_half(slot + kValueScaleOffset) : 0.0f;
    const float value_zero =
        valid ? unpack_half(slot + kValueZeroOffset) : 0.0f;
    float key_part[8];
    float value_part[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + i * 32;
      const uint8_t key_byte = valid ? slot[d >> 1] : 0;
      const int key_idx = (key_byte >> ((d & 1) * 4)) & 15;
      key_part[i] = valid ? centroids[key_idx] : 0.0f;
      const uint8_t value_byte =
          valid ? slot[kPackedValueOffset + (d >> 1)] : 0;
      const float value_idx = static_cast<float>(
          (value_byte >> ((d & 1) * 4)) & 15);
      value_part[i] = value_idx * value_scale + value_zero;
    }

    const float inv_norm =
        valid
        ? *reinterpret_cast<const float*>(slot + kCachedInvNormOffset)
        : 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int d = lane + i * 32;
      shared_key[buffer][loader_token][d] = key_part[i] * inv_norm;
      shared_value[buffer][loader_token][d] = value_part[i];
    }
    if (lane == 0) {
      shared_key_norm[buffer][loader_token] =
          valid ? unpack_half(slot + kKeyNormOffset) : 0.0f;
    }
  };

  if (num_tiles > 0) {
    load_tile(0, 0);
  }
  __syncthreads();

  for (int tile = 0; tile < num_tiles; ++tile) {
    const int buffer = tile & 1;
    if (tile + 1 < num_tiles) {
      load_tile(tile + 1, buffer ^ 1);
    }

    if (compute_warp) {
#pragma unroll
      for (int pair = 0; pair < kTileTokens / 2; ++pair) {
        const int token0 = pair * 2;
        const int token1 = token0 + 1;
        const bool valid0 =
            tile * kTileTokens + token0 < split_tokens;
        const bool valid1 =
            tile * kTileTokens + token1 < split_tokens;
        float score0[kHeadsPerWarp];
        float score1[kHeadsPerWarp];
#pragma unroll
        for (int h = 0; h < kHeadsPerWarp; ++h) {
          float products0[8];
          float products1[8];
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            products0[i] = q[h][i] * shared_key[buffer][token0][d];
            products1[i] = q[h][i] * shared_key[buffer][token1][d];
          }
          float s0 = warp_sum(lane_reduce8(products0));
          float s1 = warp_sum(lane_reduce8(products1));
          s0 = s0 * shared_key_norm[buffer][token0] * kAttentionScale;
          s1 = s1 * shared_key_norm[buffer][token1] * kAttentionScale;
          score0[h] = valid0 ? s0 : negative_infinity;
          score1[h] = valid1 ? s1 : negative_infinity;
        }

#pragma unroll
        for (int h = 0; h < kHeadsPerWarp; ++h) {
          const float next_m = fmaxf(fmaxf(score0[h], score1[h]), m[h]);
          const float rescale = triton_exp(m[h] - next_m);
          const float p0 = triton_exp(score0[h] - next_m);
          const float p1 = triton_exp(score1[h] - next_m);
#pragma unroll
          for (int i = 0; i < 8; ++i) {
            const int d = lane + i * 32;
            const float weighted =
                p0 * shared_value[buffer][token0][d] +
                p1 * shared_value[buffer][token1][d];
            accum[h][i] = accum[h][i] * rescale + weighted;
          }
          l[h] = l[h] * rescale + (p0 + p1);
          m[h] = next_m;
        }
      }
    }
    __syncthreads();
  }

  if (compute_warp) {
#pragma unroll
    for (int h = 0; h < kHeadsPerWarp; ++h) {
      const int local_head = warp * kHeadsPerWarp + h;
      const float safe_l = l[h] > 0.0f ? l[h] : 1.0f;
      float* out_head =
          output + ((batch * kNumQueryHeads +
                     kv_head * (kHeadWarps * kHeadsPerWarp) + local_head) *
                        kActiveSplits +
                    split) *
                       (kHeadDim + 1);
#pragma unroll
      for (int i = 0; i < 8; ++i) {
        const int d = lane + i * 32;
        out_head[d] = triton_div(accum[h][i], safe_l);
      }
      if (lane == 0) {
        out_head[kHeadDim] =
            split_tokens > 0 ? m[h] + logf(safe_l) : 0.0f;
      }
    }
  }
}

template <int kTileTokens>
void launch_turboquant_head_parallel(
    const float* query, const uint8_t* cache, const int32_t* block_table,
    const int32_t* seq_lens, const float* centroids, float* output,
    int64_t batch_size, int64_t cache_block_stride,
    int64_t cache_position_stride, int64_t cache_head_stride,
    int64_t table_stride, cudaStream_t stream) {
  const dim3 grid(static_cast<unsigned int>(batch_size), kNumKvHeads,
                  kNumSplits);
  turboquant_head_parallel_stage1_kernel<kTileTokens>
      <<<grid, kTileTokens * 32, 0, stream>>>(
          query, cache, block_table, seq_lens, centroids, output,
          cache_block_stride, cache_position_stride, cache_head_stride,
          table_stride);
}

template <int kTileTokens, int kActiveSplits = kNumSplits,
          bool kBoundaryOnly = false>
void launch_turboquant_head_parallel_pipeline(
    const float* query, const uint8_t* cache, const int32_t* block_table,
    const int32_t* seq_lens, const float* centroids, float* output,
    int64_t batch_size, int64_t cache_block_stride,
    int64_t cache_position_stride, int64_t cache_head_stride,
    int64_t table_stride, cudaStream_t stream,
    const int32_t* table_mismatch = nullptr) {
  const dim3 grid(static_cast<unsigned int>(batch_size), kNumKvHeads,
                  kActiveSplits);
  constexpr int kWarps = kHeadWarps + kTileTokens;
  turboquant_head_parallel_stage1_pipeline_kernel<
      kTileTokens, kActiveSplits, kBoundaryOnly>
      <<<grid, kWarps * 32, 0, stream>>>(
          query, cache, block_table, seq_lens, centroids, output,
          cache_block_stride, cache_position_stride, cache_head_stride,
          table_stride, table_mismatch);
}

void turboquant_head_parallel_stage1_impl(
    const torch::stable::Tensor& query,
    const torch::stable::Tensor& cache,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens,
    const torch::stable::Tensor& centroids,
    torch::stable::Tensor& output,
    const torch::stable::Tensor* table_mismatch, bool boundary_only) {
  STD_TORCH_CHECK(query.is_cuda() && cache.is_cuda() &&
                      block_table.is_cuda() && seq_lens.is_cuda() &&
                      centroids.is_cuda() && output.is_cuda(),
                  "turboquant_head_parallel_stage1: tensors must be CUDA");
  const auto device = query.get_device_index();
  STD_TORCH_CHECK(cache.get_device_index() == device &&
                      block_table.get_device_index() == device &&
                      seq_lens.get_device_index() == device &&
                      centroids.get_device_index() == device &&
                      output.get_device_index() == device,
                  "turboquant_head_parallel_stage1: device mismatch");
  STD_TORCH_CHECK(
      query.scalar_type() == torch::headeronly::ScalarType::Float &&
          cache.scalar_type() == torch::headeronly::ScalarType::Byte &&
          block_table.scalar_type() == torch::headeronly::ScalarType::Int &&
          seq_lens.scalar_type() == torch::headeronly::ScalarType::Int &&
          centroids.scalar_type() == torch::headeronly::ScalarType::Float &&
          output.scalar_type() == torch::headeronly::ScalarType::Float,
      "turboquant_head_parallel_stage1: unsupported dtype");
  STD_TORCH_CHECK(query.dim() == 3 && query.size(0) >= 1 &&
                      query.size(0) <= 4 &&
                      query.size(1) == kNumQueryHeads &&
                      query.size(2) == kHeadDim,
                  "turboquant_head_parallel_stage1: query shape mismatch");
  const int64_t batch_size = query.size(0);
  STD_TORCH_CHECK(cache.dim() == 4 && cache.size(1) == kBlockSize &&
                      cache.size(2) == kNumKvHeads &&
                      cache.size(3) == kSlotSize,
                  "turboquant_head_parallel_stage1: cache shape mismatch");
  STD_TORCH_CHECK(block_table.dim() == 2 &&
                      block_table.size(0) == batch_size &&
                      seq_lens.dim() == 1 &&
                      seq_lens.size(0) == batch_size &&
                      centroids.dim() == 1 && centroids.size(0) == 16,
                  "turboquant_head_parallel_stage1: metadata shape mismatch");
  const int64_t active_splits = output.dim() == 4 ? output.size(2) : -1;
  STD_TORCH_CHECK(output.dim() == 4 && output.size(0) == batch_size &&
                      output.size(1) == kNumQueryHeads &&
                      (active_splits == kNumSplits ||
                       (batch_size == 4 && active_splits == 28)) &&
                      output.size(3) == kHeadDim + 1,
                  "turboquant_head_parallel_stage1: output shape mismatch");
  STD_TORCH_CHECK(query.is_contiguous() && seq_lens.is_contiguous() &&
                      centroids.is_contiguous() && output.is_contiguous() &&
                      cache.stride(3) == 1 && block_table.stride(1) == 1,
                  "turboquant_head_parallel_stage1: stride mismatch");

  const torch::stable::accelerator::DeviceGuard device_guard(device);
  const auto stream = get_current_cuda_stream(device);
  const float* query_ptr = query.const_data_ptr<float>();
  const uint8_t* cache_ptr = cache.const_data_ptr<uint8_t>();
  const int32_t* table_ptr = block_table.const_data_ptr<int32_t>();
  const int32_t* seq_ptr = seq_lens.const_data_ptr<int32_t>();
  const float* centroids_ptr = centroids.const_data_ptr<float>();
  float* output_ptr = output.mutable_data_ptr<float>();
  const int64_t cache_block_stride = cache.stride(0);
  const int64_t cache_position_stride = cache.stride(1);
  const int64_t cache_head_stride = cache.stride(2);
  const int64_t table_stride = block_table.stride(0);

  if (boundary_only) {
    STD_TORCH_CHECK(batch_size == 4,
                    "boundary-only TurboQuant requires B4");
    STD_TORCH_CHECK(table_mismatch != nullptr && table_mismatch->is_cuda() &&
                        table_mismatch->get_device_index() == device &&
                        table_mismatch->scalar_type() ==
                            torch::headeronly::ScalarType::Int &&
                        table_mismatch->numel() == 1 &&
                        table_mismatch->is_contiguous(),
                    "boundary-only TurboQuant requires one int32 CUDA proof");
    if (active_splits == 28) {
      launch_turboquant_head_parallel_pipeline<8, 28, true>(
          query_ptr, cache_ptr, table_ptr, seq_ptr, centroids_ptr, output_ptr,
          batch_size, cache_block_stride, cache_position_stride,
          cache_head_stride, table_stride, stream,
          table_mismatch->const_data_ptr<int32_t>());
    } else {
      launch_turboquant_head_parallel_pipeline<8, kNumSplits, true>(
          query_ptr, cache_ptr, table_ptr, seq_ptr, centroids_ptr, output_ptr,
          batch_size, cache_block_stride, cache_position_stride,
          cache_head_stride, table_stride, stream,
          table_mismatch->const_data_ptr<int32_t>());
    }
  } else {
    if (active_splits == 28) {
      launch_turboquant_head_parallel_pipeline<8, 28, false>(
          query_ptr, cache_ptr, table_ptr, seq_ptr, centroids_ptr, output_ptr,
          batch_size, cache_block_stride, cache_position_stride,
          cache_head_stride, table_stride, stream);
    } else {
      launch_turboquant_head_parallel_pipeline<8, kNumSplits, false>(
          query_ptr, cache_ptr, table_ptr, seq_ptr, centroids_ptr, output_ptr,
          batch_size, cache_block_stride, cache_position_stride,
          cache_head_stride, table_stride, stream);
    }
  }
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "turboquant_head_parallel_stage1 failed: ",
                  cudaGetErrorString(error));
}

void turboquant_head_parallel_stage1(
    const torch::stable::Tensor& query,
    const torch::stable::Tensor& cache,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens,
    const torch::stable::Tensor& centroids,
    torch::stable::Tensor& output) {
  turboquant_head_parallel_stage1_impl(
      query, cache, block_table, seq_lens, centroids, output, nullptr, false);
}

void turboquant_head_parallel_stage1_boundary_b4(
    const torch::stable::Tensor& query,
    const torch::stable::Tensor& cache,
    const torch::stable::Tensor& block_table,
    const torch::stable::Tensor& seq_lens,
    const torch::stable::Tensor& centroids,
    torch::stable::Tensor& output,
    const torch::stable::Tensor& table_mismatch) {
  turboquant_head_parallel_stage1_impl(
      query, cache, block_table, seq_lens, centroids, output,
      &table_mismatch, true);
}

}  // namespace

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "turboquant_head_parallel_stage1(Tensor query, Tensor cache, Tensor "
      "block_table, Tensor seq_lens, Tensor centroids, Tensor! output) -> ()");
  m.def(
      "turboquant_head_parallel_stage1_boundary_b4(Tensor query, Tensor cache, "
      "Tensor block_table, Tensor seq_lens, Tensor centroids, Tensor! output, "
      "Tensor table_mismatch) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("turboquant_head_parallel_stage1",
         TORCH_BOX(&turboquant_head_parallel_stage1));
  m.impl("turboquant_head_parallel_stage1_boundary_b4",
         TORCH_BOX(&turboquant_head_parallel_stage1_boundary_b4));
}
