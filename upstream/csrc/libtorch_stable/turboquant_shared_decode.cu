// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "core/registration.h"
#include "libtorch_stable/torch_utils.h"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <torch/csrc/stable/library.h>

namespace vllm {
namespace {

constexpr int kHeads = 24;
constexpr int kKvHeads = 4;
constexpr int kHeadsPerKv = 6;
constexpr int kHeadDim = 256;
constexpr int kBlockSize = 16;
constexpr int kSlotSize = 262;
constexpr int kSplits = 32;
constexpr float kAttentionScale = 0.0625f;

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int delta = 16; delta >= 1; delta >>= 1) {
    value += __shfl_xor_sync(0xffffffff, value, delta);
  }
  return value;
}

__device__ __forceinline__ float lane_reduce8(const float (&values)[8]) {
  float result = values[0] + values[1];
#pragma unroll
  for (int k = 2; k < 8; ++k) {
    result += values[k];
  }
  return result;
}

__device__ __forceinline__ float unpack_half(const uint8_t* ptr) {
  __half_raw raw;
  raw.x = static_cast<unsigned short>(ptr[0]) |
          (static_cast<unsigned short>(ptr[1]) << 8);
  return __half2float(raw);
}

// Match the PTX operations emitted by the existing Triton stage-1 kernel.
__device__ __forceinline__ float tq_exp(float value) {
  const float scaled = value * 1.4426950408889634074f;
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(scaled));
  return result;
}

__device__ __forceinline__ float tq_sqrt(float value) {
  float result;
  asm("sqrt.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ float tq_div(float numerator, float denominator) {
  float result;
  asm("div.full.f32 %0, %1, %2;"
      : "=f"(result)
      : "f"(numerator), "f"(denominator));
  return result;
}

template <int kRows>
__global__ __launch_bounds__(128, 2) void turboquant_shared_rows_stage1_kernel(
    const float* __restrict__ query, const uint8_t* __restrict__ cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const float* __restrict__ centroids, float* __restrict__ output,
    long long cache_block_stride, long long cache_position_stride,
    long long cache_head_stride, int table_stride) {
  const int group = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  const int row = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int batch = group * kRows + row;

  const int seq_len = seq_lens[batch];
  const int split_len = (seq_len + kSplits - 1) / kSplits;
  const int common_split_len =
      (seq_lens[group * kRows] + kSplits - 1) / kSplits;
  const int split_start = split_len * split;
  const int split_end = min(split_start + split_len, seq_len);
  const int row_split_tokens = max(split_end - split_start, 0);

  float q[6][8];
  float accum[6][8];
#pragma unroll
  for (int head = 0; head < kHeadsPerKv; ++head) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane + k * 32;
      q[head][k] =
          query[(batch * kHeads + kv_head * kHeadsPerKv + head) * kHeadDim + d];
      accum[head][k] = 0.0f;
    }
  }

  const float negative_infinity = __int_as_float(0xff800000);
  float m[6];
  float l[6];
#pragma unroll
  for (int head = 0; head < kHeadsPerKv; ++head) {
    m[head] = negative_infinity;
    l[head] = 0.0f;
  }

  __shared__ float shared_key[kRows][2][kHeadDim];
  __shared__ float shared_value[kRows][2][kHeadDim];
  __shared__ float shared_key_norm[kRows][2];

  int max_split_tokens = 0;
#pragma unroll
  for (int other_row = 0; other_row < kRows; ++other_row) {
    const int other_seq_len = seq_lens[group * kRows + other_row];
    const int other_split_len =
        (other_seq_len + kSplits - 1) / kSplits;
    const int other_start = other_split_len * split;
    const int other_end = min(other_start + other_split_len, other_seq_len);
    max_split_tokens = max(max_split_tokens, max(other_end - other_start, 0));
  }

  for (int tile = 0; tile * 2 < max_split_tokens; ++tile) {
#pragma unroll
    for (int load_iter = 0; load_iter < 2; ++load_iter) {
      const bool loader_active = row < 2 && load_iter == 0;
      if (!loader_active) {
        continue;
      }
      const int token_in_tile = row;
      constexpr int storage_row = 0;
      const int load_batch = group * kRows;
      const int load_split_start = common_split_len * split;
      const int position = load_split_start + tile * 2 + token_in_tile;
      int load_split_end = 0;
#pragma unroll
      for (int other_row = 0; other_row < kRows; ++other_row) {
        load_split_end = max(
            load_split_end,
            min(load_split_start + common_split_len,
                seq_lens[group * kRows + other_row]));
      }
      const bool valid = position < load_split_end;
      const int page = position / kBlockSize;
      const int page_offset = position - page * kBlockSize;
      const int block = valid
          ? block_table[load_batch * table_stride + page]
          : 0;
      const uint8_t* slot =
          cache + static_cast<long long>(block) * cache_block_stride +
          static_cast<long long>(page_offset) * cache_position_stride +
          static_cast<long long>(kv_head) * cache_head_stride;

      float key_part[8];
      float value_part[8];
      const float value_scale = valid ? unpack_half(slot + 258) : 0.0f;
      const float value_zero = valid ? unpack_half(slot + 260) : 0.0f;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = lane + k * 32;
        const uint8_t key_byte = valid ? slot[d >> 1] : 0;
        const int key_idx = (key_byte >> ((d & 1) * 4)) & 15;
        key_part[k] = valid ? centroids[key_idx] : 0.0f;
        const uint8_t value_byte = valid ? slot[130 + (d >> 1)] : 0;
        const float value_idx = static_cast<float>(
            (value_byte >> ((d & 1) * 4)) & 15);
        value_part[k] = value_idx * value_scale + value_zero;
      }

      float norm_terms[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        norm_terms[k] = key_part[k] * key_part[k];
      }
      const float norm_sq = warp_sum(lane_reduce8(norm_terms));
      const float inv_norm = tq_div(1.0f, tq_sqrt(norm_sq + 1.0e-16f));
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = lane + k * 32;
        shared_key[storage_row][token_in_tile][d] = key_part[k] * inv_norm;
        shared_value[storage_row][token_in_tile][d] = value_part[k];
      }
      if (lane == 0) {
        shared_key_norm[storage_row][token_in_tile] =
            valid ? unpack_half(slot + 128) : 0.0f;
      }
    }
    __syncthreads();

    if (tile * 2 < row_split_tokens) {
      constexpr int storage_row = 0;
      const bool valid1 = tile * 2 + 1 < row_split_tokens;
      float score0[6];
      float score1[6];
#pragma unroll
      for (int head = 0; head < kHeadsPerKv; ++head) {
        float products0[8];
        float products1[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int d = lane + k * 32;
          products0[k] = q[head][k] * shared_key[storage_row][0][d];
          products1[k] = q[head][k] * shared_key[storage_row][1][d];
        }
        float s0 = warp_sum(lane_reduce8(products0));
        float s1 = warp_sum(lane_reduce8(products1));
        s0 = s0 * shared_key_norm[storage_row][0] * kAttentionScale;
        s1 = s1 * shared_key_norm[storage_row][1] * kAttentionScale;
        score0[head] = s0;
        score1[head] = valid1 ? s1 : negative_infinity;
      }

#pragma unroll
      for (int head = 0; head < kHeadsPerKv; ++head) {
        const float next_m =
            fmaxf(fmaxf(score0[head], score1[head]), m[head]);
        const float rescale = tq_exp(m[head] - next_m);
        const float p0 = tq_exp(score0[head] - next_m);
        const float p1 = tq_exp(score1[head] - next_m);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int d = lane + k * 32;
          const float weighted =
              p0 * shared_value[storage_row][0][d] +
              p1 * shared_value[storage_row][1][d];
          accum[head][k] = accum[head][k] * rescale + weighted;
        }
        l[head] = l[head] * rescale + (p0 + p1);
        m[head] = next_m;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int head = 0; head < kHeadsPerKv; ++head) {
    const float safe_l = l[head] > 0.0f ? l[head] : 1.0f;
    float* out_head = output +
        ((batch * kHeads + kv_head * kHeadsPerKv + head) * kSplits + split) *
            (kHeadDim + 1);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane + k * 32;
      out_head[d] = tq_div(accum[head][k], safe_l);
    }
    if (lane == 0) {
      out_head[kHeadDim] = seq_len > 0 ? m[head] + logf(safe_l) : 0.0f;
    }
  }
}

// Four B16 warps can cooperatively dequantize a larger KV tile without
// serializing the loaders.  Keep the exact two-token online-softmax update
// order, but amortize the CTA barriers over ten pairs.  B8 deliberately keeps
// the two-token kernel above: its two loader warps lose to serialized loading
// with this larger tile.
__global__ __launch_bounds__(128, 2)
void turboquant_shared_rows_tile20_stage1_kernel(
    const float* __restrict__ query, const uint8_t* __restrict__ cache,
    const int* __restrict__ block_table, const int* __restrict__ seq_lens,
    const float* __restrict__ centroids, float* __restrict__ output,
    long long cache_block_stride, long long cache_position_stride,
    long long cache_head_stride, int table_stride) {
  constexpr int kRows = 4;
  constexpr int kTileTokens = 20;
  const int group = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  const int row = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int batch = group * kRows + row;

  const int seq_len = seq_lens[batch];
  const int split_len = (seq_len + kSplits - 1) / kSplits;
  const int common_split_len =
      (seq_lens[group * kRows] + kSplits - 1) / kSplits;
  const int split_start = split_len * split;
  const int split_end = min(split_start + split_len, seq_len);
  const int row_split_tokens = max(split_end - split_start, 0);

  float q[6][8];
  float accum[6][8];
#pragma unroll
  for (int head = 0; head < kHeadsPerKv; ++head) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane + k * 32;
      q[head][k] =
          query[(batch * kHeads + kv_head * kHeadsPerKv + head) * kHeadDim + d];
      accum[head][k] = 0.0f;
    }
  }

  const float negative_infinity = __int_as_float(0xff800000);
  float m[6];
  float l[6];
#pragma unroll
  for (int head = 0; head < kHeadsPerKv; ++head) {
    m[head] = negative_infinity;
    l[head] = 0.0f;
  }

  __shared__ float shared_key[kTileTokens][kHeadDim];
  __shared__ float shared_value[kTileTokens][kHeadDim];
  __shared__ float shared_key_norm[kTileTokens];

  int max_split_tokens = 0;
  int load_split_end = 0;
#pragma unroll
  for (int other_row = 0; other_row < kRows; ++other_row) {
    const int other_seq_len = seq_lens[group * kRows + other_row];
    const int other_split_len =
        (other_seq_len + kSplits - 1) / kSplits;
    const int other_start = other_split_len * split;
    const int other_end = min(other_start + other_split_len, other_seq_len);
    max_split_tokens = max(max_split_tokens, max(other_end - other_start, 0));
    load_split_end = max(load_split_end, other_end);
  }

  for (int tile = 0; tile * kTileTokens < max_split_tokens; ++tile) {
#pragma unroll
    for (int token_in_tile = row; token_in_tile < kTileTokens;
         token_in_tile += kRows) {
      const int load_batch = group * kRows;
      const int load_split_start = common_split_len * split;
      const int position =
          load_split_start + tile * kTileTokens + token_in_tile;
      const bool valid = position < load_split_end;
      const int page = position / kBlockSize;
      const int page_offset = position - page * kBlockSize;
      const int block = valid
          ? block_table[load_batch * table_stride + page]
          : 0;
      const uint8_t* slot =
          cache + static_cast<long long>(block) * cache_block_stride +
          static_cast<long long>(page_offset) * cache_position_stride +
          static_cast<long long>(kv_head) * cache_head_stride;

      const float value_scale = valid ? unpack_half(slot + 258) : 0.0f;
      const float value_zero = valid ? unpack_half(slot + 260) : 0.0f;
      float key_part[8];
      float value_part[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = lane + k * 32;
        const uint8_t key_byte = valid ? slot[d >> 1] : 0;
        const int key_idx = (key_byte >> ((d & 1) * 4)) & 15;
        key_part[k] = valid ? centroids[key_idx] : 0.0f;
        const uint8_t value_byte = valid ? slot[130 + (d >> 1)] : 0;
        const float value_idx = static_cast<float>(
            (value_byte >> ((d & 1) * 4)) & 15);
        value_part[k] = value_idx * value_scale + value_zero;
      }

      float norm_terms[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        norm_terms[k] = key_part[k] * key_part[k];
      }
      const float norm_sq = warp_sum(lane_reduce8(norm_terms));
      const float inv_norm = tq_div(1.0f, tq_sqrt(norm_sq + 1.0e-16f));
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = lane + k * 32;
        shared_key[token_in_tile][d] = key_part[k] * inv_norm;
        shared_value[token_in_tile][d] = value_part[k];
      }
      if (lane == 0) {
        shared_key_norm[token_in_tile] =
            valid ? unpack_half(slot + 128) : 0.0f;
      }
    }
    __syncthreads();

#pragma unroll
    for (int pair = 0; pair < kTileTokens / 2; ++pair) {
      const int token0 = pair * 2;
      const int token1 = token0 + 1;
      if (tile * kTileTokens + token0 >= row_split_tokens) {
        continue;
      }
      const bool valid1 =
          tile * kTileTokens + token1 < row_split_tokens;
      float score0[6];
      float score1[6];
#pragma unroll
      for (int head = 0; head < kHeadsPerKv; ++head) {
        float products0[8];
        float products1[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int d = lane + k * 32;
          products0[k] = q[head][k] * shared_key[token0][d];
          products1[k] = q[head][k] * shared_key[token1][d];
        }
        float s0 = warp_sum(lane_reduce8(products0));
        float s1 = warp_sum(lane_reduce8(products1));
        s0 = s0 * shared_key_norm[token0] * kAttentionScale;
        s1 = s1 * shared_key_norm[token1] * kAttentionScale;
        score0[head] = s0;
        score1[head] = valid1 ? s1 : negative_infinity;
      }

#pragma unroll
      for (int head = 0; head < kHeadsPerKv; ++head) {
        const float next_m =
            fmaxf(fmaxf(score0[head], score1[head]), m[head]);
        const float rescale = tq_exp(m[head] - next_m);
        const float p0 = tq_exp(score0[head] - next_m);
        const float p1 = tq_exp(score1[head] - next_m);
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int d = lane + k * 32;
          const float weighted =
              p0 * shared_value[token0][d] +
              p1 * shared_value[token1][d];
          accum[head][k] = accum[head][k] * rescale + weighted;
        }
        l[head] = l[head] * rescale + (p0 + p1);
        m[head] = next_m;
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int head = 0; head < kHeadsPerKv; ++head) {
    const float safe_l = l[head] > 0.0f ? l[head] : 1.0f;
    float* out_head = output +
        ((batch * kHeads + kv_head * kHeadsPerKv + head) * kSplits + split) *
            (kHeadDim + 1);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane + k * 32;
      out_head[d] = tq_div(accum[head][k], safe_l);
    }
    if (lane == 0) {
      out_head[kHeadDim] =
          row_split_tokens > 0 ? m[head] + logf(safe_l) : 0.0f;
    }
  }
}

// Exact same-split B4 producer/consumer kernel. Three two-head CTA groups
// retain target parallelism while sharing historical KV across four causal rows.
template <int kTileTokens = 20, int kLoaderWarps = 5,
          int kHeadsPerWarp = 6, int kHeadGroups = 1>
__global__ __launch_bounds__((4 + kLoaderWarps) * 32,
                             kHeadGroups == 1 ? 1 : 2)
void turboquant_shared4_grid3_pipeline_kernel(
    const float* __restrict__ q, const uint8_t* __restrict__ cache,
    const int* __restrict__ table, const int* __restrict__ seq,
    const float* __restrict__ centroids, float* __restrict__ out,
    long long cache_block_stride, long long cache_position_stride,
    long long cache_head_stride, int table_stride,
    const int* __restrict__ table_mismatch) {
  constexpr int kRows = 4;
  constexpr int kComputeWarps = kRows;
  static_assert(6 % kHeadsPerWarp == 0);
  static_assert(kHeadGroups == 1 ||
                kHeadGroups * kHeadsPerWarp == 6);
  constexpr int kBlockSize = 16;
  constexpr int kSplits = 32;
  constexpr float kScale = 0.0625f;
  const int group = blockIdx.x / kHeadGroups;
  const int head_group = blockIdx.x - group * kHeadGroups;
  const int kv_head = blockIdx.y;
  const int split = blockIdx.z;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const bool compute_warp = warp < kComputeWarps;
  const int row = compute_warp ? warp : 0;
  const int loader_warp = warp - kComputeWarps;
  const int batch = group * kRows + row;

  // This B4 specialization is selected only for four synthetic verification
  // rows whose physical block table is repeated from one request.  Async MTP
  // may still correct the authoritative lengths after graph dispatch, so
  // prove on device that all rows retain one common 32-way split width.  The
  // complementary head-parallel launch handles the boundary case exactly.
  __shared__ int share_safe;
  if (threadIdx.x == 0) {
    const int common_len =
        (seq[group * kRows] + kSplits - 1) / kSplits;
    share_safe = table_mismatch[0] == 0;
#pragma unroll
    for (int other_row = 1; other_row < kRows; ++other_row) {
      const int candidate =
          (seq[group * kRows + other_row] + kSplits - 1) / kSplits;
      if (candidate != common_len) {
        share_safe = 0;
      }
    }
  }
  __syncthreads();
  if (!share_safe) {
    return;
  }

  const int seq_len = seq[batch];
  const int split_len = (seq_len + kSplits - 1) / kSplits;
  const int common_split_len =
      (seq[group * kRows] + kSplits - 1) / kSplits;
  const int split_start = split_len * split;
  const int split_end = min(split_start + split_len, seq_len);
  const int row_split_tokens = max(split_end - split_start, 0);
  const int load_split_start = common_split_len * split;
  int load_split_end = 0;
#pragma unroll
  for (int other_row = 0; other_row < kRows; ++other_row) {
    const int other_seq_len = seq[group * kRows + other_row];
    const int other_split_len =
        (other_seq_len + kSplits - 1) / kSplits;
    const int other_start = other_split_len * split;
    const int other_end =
        min(other_start + other_split_len, other_seq_len);
    load_split_end = max(load_split_end, other_end);
  }
  const int max_split_tokens = max(load_split_end - load_split_start, 0);
  const int num_tiles =
      (max_split_tokens + kTileTokens - 1) / kTileTokens;

  float query[kHeadsPerWarp][8];
  float accum[kHeadsPerWarp][8];
#pragma unroll
  for (int h = 0; h < kHeadsPerWarp; ++h) {
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int d = lane + k * 32;
      const int local_head = head_group * kHeadsPerWarp + h;
      query[h][k] = compute_warp
          ? q[(batch * 24 + kv_head * 6 + local_head) * 256 + d]
          : 0.0f;
      accum[h][k] = 0.0f;
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

  extern __shared__ float shared_storage[];
  float (*shared_key)[kTileTokens][256] =
      reinterpret_cast<float (*)[kTileTokens][256]>(shared_storage);
  float (*shared_value)[kTileTokens][256] =
      reinterpret_cast<float (*)[kTileTokens][256]>(
          shared_storage + 2 * kTileTokens * 256);
  float (*shared_key_norm)[kTileTokens] =
      reinterpret_cast<float (*)[kTileTokens]>(
          shared_storage + 4 * kTileTokens * 256);

  auto load_tile = [&](int tile, int buffer) {
    if (loader_warp < 0 || loader_warp >= kLoaderWarps) {
      return;
    }
#pragma unroll
    for (int load_iter = 0;
         load_iter < (kTileTokens + kLoaderWarps - 1) / kLoaderWarps;
         ++load_iter) {
      const int loader_token = loader_warp + load_iter * kLoaderWarps;
      if (loader_token >= kTileTokens) {
        continue;
      }
      const int position =
          load_split_start + tile * kTileTokens + loader_token;
      const bool valid = position < load_split_end;
      const int page = position / kBlockSize;
      const int page_offset = position - page * kBlockSize;
      const int block =
          valid ? table[group * kRows * table_stride + page] : 0;
      const uint8_t* slot =
          cache + static_cast<long long>(block) * cache_block_stride +
          static_cast<long long>(page_offset) * cache_position_stride +
          static_cast<long long>(kv_head) * cache_head_stride;
      const float value_scale = valid ? unpack_half(slot + 258) : 0.0f;
      const float value_zero = valid ? unpack_half(slot + 260) : 0.0f;
      float key_part[8];
      float value_part[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = lane + k * 32;
        const uint8_t key_byte = valid ? slot[d >> 1] : 0;
        const int key_idx = (key_byte >> ((d & 1) * 4)) & 15;
        key_part[k] = valid ? centroids[key_idx] : 0.0f;
        const uint8_t value_byte = valid ? slot[130 + (d >> 1)] : 0;
        const float value_idx = static_cast<float>(
            (value_byte >> ((d & 1) * 4)) & 15);
        value_part[k] = value_idx * value_scale + value_zero;
      }
      float norm_terms[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        norm_terms[k] = key_part[k] * key_part[k];
      }
      const float norm_sq = warp_sum(lane_reduce8(norm_terms));
      const float inv_norm =
          tq_div(1.0f, tq_sqrt(norm_sq + 1.0e-16f));
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = lane + k * 32;
        shared_key[buffer][loader_token][d] = key_part[k] * inv_norm;
        shared_value[buffer][loader_token][d] = value_part[k];
      }
      if (lane == 0) {
        shared_key_norm[buffer][loader_token] =
            valid ? unpack_half(slot + 128) : 0.0f;
      }
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
        if (tile * kTileTokens + token0 >= row_split_tokens) {
          continue;
        }
        const bool valid1 =
            tile * kTileTokens + token1 < row_split_tokens;
        float score0[kHeadsPerWarp];
        float score1[kHeadsPerWarp];
#pragma unroll
        for (int h = 0; h < kHeadsPerWarp; ++h) {
          float products0[8];
          float products1[8];
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int d = lane + k * 32;
            products0[k] = query[h][k] * shared_key[buffer][token0][d];
            products1[k] = query[h][k] * shared_key[buffer][token1][d];
          }
          float s0 = warp_sum(lane_reduce8(products0));
          float s1 = warp_sum(lane_reduce8(products1));
          s0 = s0 * shared_key_norm[buffer][token0] * kScale;
          s1 = s1 * shared_key_norm[buffer][token1] * kScale;
          score0[h] = s0;
          score1[h] = valid1 ? s1 : negative_infinity;
        }
#pragma unroll
        for (int h = 0; h < kHeadsPerWarp; ++h) {
          const float next_m =
              fmaxf(fmaxf(score0[h], score1[h]), m[h]);
          const float rescale = tq_exp(m[h] - next_m);
          const float p0 = tq_exp(score0[h] - next_m);
          const float p1 = tq_exp(score1[h] - next_m);
#pragma unroll
          for (int k = 0; k < 8; ++k) {
            const int d = lane + k * 32;
            const float weighted =
                p0 * shared_value[buffer][token0][d] +
                p1 * shared_value[buffer][token1][d];
            accum[h][k] = accum[h][k] * rescale + weighted;
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
      const int local_head = head_group * kHeadsPerWarp + h;
      const float safe_l = l[h] > 0.0f ? l[h] : 1.0f;
      float* out_head = out +
          ((batch * 24 + kv_head * 6 + local_head) * kSplits + split) *
              257;
#pragma unroll
      for (int k = 0; k < 8; ++k) {
        const int d = lane + k * 32;
        out_head[d] = tq_div(accum[h][k], safe_l);
      }
      if (lane == 0) {
        out_head[256] = row_split_tokens > 0
            ? m[h] + logf(safe_l)
            : 0.0f;
      }
    }
  }
}

// Hoist the physical-page proof out of the attention CTAs.  One small launch
// compares all active entries exactly once per layer; both following kernels
// consume the same stream-ordered mismatch scalar and form exact complements.
__global__ void turboquant_b4_table_identity_proof_kernel(
    const int* __restrict__ table, const int* __restrict__ seq,
    int* __restrict__ mismatch, int table_stride) {
  int max_seq = 0;
#pragma unroll
  for (int row = 0; row < 4; ++row) {
    max_seq = max(max_seq, seq[row]);
  }
  const int active_pages = (max_seq + kBlockSize - 1) / kBlockSize;
  if (threadIdx.x == 0 && blockIdx.x == 0 && active_pages > table_stride) {
    atomicExch(mismatch, 1);
  }
  const int checked_pages = min(active_pages, table_stride);
  for (int page = blockIdx.x * blockDim.x + threadIdx.x;
       page < checked_pages; page += blockDim.x * gridDim.x) {
    const int reference = table[page];
#pragma unroll
    for (int row = 1; row < 4; ++row) {
      if (table[row * table_stride + page] != reference) {
        atomicExch(mismatch, 1);
      }
    }
  }
}


}  // namespace

void turboquant_shared_rows_stage1(const torch::stable::Tensor& query,
                                   const torch::stable::Tensor& cache,
                                   const torch::stable::Tensor& block_table,
                                   const torch::stable::Tensor& seq_lens,
                                   const torch::stable::Tensor& centroids,
                                   torch::stable::Tensor& output,
                                   torch::stable::Tensor& table_mismatch) {
  using ScalarType = torch::headeronly::ScalarType;
  STD_TORCH_CHECK(query.is_cuda() && cache.is_cuda() && block_table.is_cuda() &&
                      seq_lens.is_cuda() && centroids.is_cuda() &&
                      output.is_cuda() && table_mismatch.is_cuda(),
                  "turboquant_shared_rows_stage1 expects CUDA tensors");
  STD_TORCH_CHECK(query.scalar_type() == ScalarType::Float &&
                      cache.scalar_type() == ScalarType::Byte &&
                      block_table.scalar_type() == ScalarType::Int &&
                      seq_lens.scalar_type() == ScalarType::Int &&
                      centroids.scalar_type() == ScalarType::Float &&
                      output.scalar_type() == ScalarType::Float &&
                      table_mismatch.scalar_type() == ScalarType::Int,
                  "turboquant_shared_rows_stage1 received unsupported dtypes");
  STD_TORCH_CHECK(query.dim() == 3 && query.size(1) == kHeads &&
                      query.size(2) == kHeadDim &&
                      (query.size(0) == 4 || query.size(0) == 8 ||
                       query.size(0) == 16),
                  "turboquant_shared_rows_stage1 expects [B,24,256], "
                  "B in {4,8,16}");
  STD_TORCH_CHECK(cache.dim() == 4 && cache.size(1) == kBlockSize &&
                      cache.size(2) == kKvHeads && cache.size(3) == kSlotSize,
                  "turboquant_shared_rows_stage1 received unsupported cache layout");
  STD_TORCH_CHECK(block_table.dim() == 2 &&
                      block_table.size(0) == query.size(0) &&
                      seq_lens.dim() == 1 &&
                      seq_lens.size(0) == query.size(0) &&
                      centroids.dim() == 1 && centroids.size(0) == 16,
                  "turboquant_shared_rows_stage1 received incompatible metadata");
  STD_TORCH_CHECK(query.is_contiguous() && cache.stride(3) == 1 &&
                      block_table.is_contiguous() && seq_lens.is_contiguous() &&
                      centroids.is_contiguous() && output.is_contiguous() &&
                      table_mismatch.is_contiguous() &&
                      table_mismatch.numel() == 1,
                  "turboquant_shared_rows_stage1 received unsupported strides");
  STD_TORCH_CHECK(output.dim() == 4 && output.size(0) == query.size(0) &&
                      output.size(1) == kHeads && output.size(2) == kSplits &&
                      output.size(3) == kHeadDim + 1,
                  "turboquant_shared_rows_stage1 received unsupported output shape");

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();
  const int batch_size = static_cast<int>(query.size(0));
  if (batch_size == 4) {
    constexpr int kTileTokens = 12;
    constexpr int kLoaderWarps = 8;
    constexpr int kHeadsPerWarp = 2;
    constexpr int kHeadGroups = 3;
    constexpr int kSharedBytes =
        sizeof(float) * (4 * kTileTokens * kHeadDim + 2 * kTileTokens);
    const cudaError_t attr_error = cudaFuncSetAttribute(
        turboquant_shared4_grid3_pipeline_kernel<
            kTileTokens, kLoaderWarps, kHeadsPerWarp, kHeadGroups>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, kSharedBytes);
    STD_TORCH_CHECK(attr_error == cudaSuccess,
                    "turboquant B4 grid3 shared-memory opt-in failed: ",
                    cudaGetErrorString(attr_error));
    const dim3 grid(batch_size / 4 * kHeadGroups, kKvHeads, kSplits);
    cudaMemsetAsync(table_mismatch.mutable_data_ptr<int>(), 0, sizeof(int),
                    stream);
    turboquant_b4_table_identity_proof_kernel<<<4, 256, 0, stream>>>(
        block_table.const_data_ptr<int>(), seq_lens.const_data_ptr<int>(),
        table_mismatch.mutable_data_ptr<int>(),
        static_cast<int>(block_table.stride(0)));
    turboquant_shared4_grid3_pipeline_kernel<
        kTileTokens, kLoaderWarps, kHeadsPerWarp, kHeadGroups>
        <<<grid, (4 + kLoaderWarps) * 32, kSharedBytes, stream>>>(
            query.const_data_ptr<float>(), cache.const_data_ptr<uint8_t>(),
            block_table.const_data_ptr<int>(), seq_lens.const_data_ptr<int>(),
            centroids.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
            static_cast<long long>(cache.stride(0)),
            static_cast<long long>(cache.stride(1)),
            static_cast<long long>(cache.stride(2)),
            static_cast<int>(block_table.stride(0)),
            table_mismatch.const_data_ptr<int>());
  } else if (batch_size == 8) {
    const dim3 grid(batch_size / 2, kKvHeads, kSplits);
    turboquant_shared_rows_stage1_kernel<2><<<grid, 64, 0, stream>>>(
        query.const_data_ptr<float>(), cache.const_data_ptr<uint8_t>(),
        block_table.const_data_ptr<int>(), seq_lens.const_data_ptr<int>(),
        centroids.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
        static_cast<long long>(cache.stride(0)),
        static_cast<long long>(cache.stride(1)),
        static_cast<long long>(cache.stride(2)),
        static_cast<int>(block_table.stride(0)));
  } else {
    const dim3 grid(batch_size / 4, kKvHeads, kSplits);
    turboquant_shared_rows_tile20_stage1_kernel<<<grid, 128, 0, stream>>>(
        query.const_data_ptr<float>(), cache.const_data_ptr<uint8_t>(),
        block_table.const_data_ptr<int>(), seq_lens.const_data_ptr<int>(),
        centroids.const_data_ptr<float>(), output.mutable_data_ptr<float>(),
        static_cast<long long>(cache.stride(0)),
        static_cast<long long>(cache.stride(1)),
        static_cast<long long>(cache.stride(2)),
        static_cast<int>(block_table.stride(0)));
  }
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "turboquant_shared_rows_stage1 launch failed: ",
                  cudaGetErrorString(error));
}

}  // namespace vllm

STABLE_TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "turboquant_shared_rows_stage1(Tensor query, Tensor cache, Tensor "
      "block_table, Tensor seq_lens, Tensor centroids, Tensor! output, "
      "Tensor! table_mismatch) -> ()");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("turboquant_shared_rows_stage1",
         TORCH_BOX(&vllm::turboquant_shared_rows_stage1));
}
