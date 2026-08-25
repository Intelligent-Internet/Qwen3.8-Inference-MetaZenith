# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Triton fused TurboQuant decode attention.

Decode path: Triton stage1 (split-KV tiled attention scoring + value
accumulation) + stage2 (log-sum-exp reduction across splits).

Supports FP8 (E4M3) keys, 3-bit and 4-bit uniform quantized values.
"""

import math
from typing import Any

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.triton_decode_attention import (
    _fwd_kernel_stage2,
)

_FP8_E4B15: dict[int, int] = {}


def _use_fp8_e4b15(device: int = 0) -> int:
    """Return 1 if device needs fp8e4b15 (Ampere/Ada, SM < 8.9), else 0.
    On non-CUDA platforms (e.g. XPU), always returns 0 (use e4nv format).
    """
    if device not in _FP8_E4B15:
        if current_platform.is_cuda_alike():
            cap = torch.cuda.get_device_capability(device)
            _FP8_E4B15[device] = 1 if cap < (8, 9) else 0
        else:
            _FP8_E4B15[device] = 0
    return _FP8_E4B15[device]


# ---------------------------------------------------------------------------
# Stage 1: Fused TQ score + value accumulation (BLOCK_KV tiled)
# ---------------------------------------------------------------------------


@triton.jit
def _tq_decode_stage1(
    # Precomputed query projection
    Q_rot_ptr,  # [B, Hq, D] float32
    # Compressed KV cache (combined K+V)
    KV_cache_ptr,  # [num_blocks, block_size, Hk, padded_slot] uint8
    # Block table and sequence info
    Block_table_ptr,  # [B, max_num_blocks] int32
    Seq_lens_ptr,  # [B] int32
    # TQ parameters
    Centroids_ptr,  # [n_centroids] float32
    # Output (intermediate for stage2)
    Mid_o_ptr,  # [B, Hq, NUM_KV_SPLITS, D+1] float32
    # Strides
    stride_qb,
    stride_qh,  # Q strides: [B, Hq, D]
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,  # KV cache
    stride_bt_b,  # block_table stride per batch
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,  # mid_o strides
    # Constexpr dims
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # KV cache block_size (pages)
    NUM_KV_SPLITS: tl.constexpr,
    KV_GROUP_SIZE: tl.constexpr,  # Hq // Hk
    # TQ layout constants
    MSE_BITS: tl.constexpr,  # 3 or 4
    MSE_BYTES: tl.constexpr,  # ceil(D * mse_bits / 8)
    KPS: tl.constexpr,  # key_packed_size
    VQB: tl.constexpr,  # value_quant_bits (4 or 8=FP8)
    VAL_DATA_BYTES: tl.constexpr,  # ceil(D * vqb / 8) or D for FP8
    # Score constants
    ATTN_SCALE: tl.constexpr,  # 1/sqrt(D)
    # Block tile sizes
    BLOCK_D: tl.constexpr,  # next_power_of_2(HEAD_DIM)
    BLOCK_KV: tl.constexpr,  # tokens per tile (16)
    HEAD_GROUP: tl.constexpr,  # query heads sharing one KV head/load
    KEY_FP8: tl.constexpr,  # 1 if K is stored as FP8
    NORM_CORRECTION: tl.constexpr = 0,  # 1 = re-normalize centroids
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    bid = tl.program_id(0)  # batch index
    hgid = tl.program_id(1)  # query-head group index
    sid = tl.program_id(2)  # kv_split index

    head_local = tl.arange(0, 8)
    head_offs = hgid * HEAD_GROUP + head_local
    head_mask = (head_local < HEAD_GROUP) & (head_offs < NUM_QUERY_HEADS)
    kv_head = (hgid * HEAD_GROUP) // KV_GROUP_SIZE

    # Sequence length for this batch
    seq_len = tl.load(Seq_lens_ptr + bid)

    # KV split range
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)

    if split_start >= split_end:
        return

    # Dimension offsets
    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM
    kv_range = tl.arange(0, BLOCK_KV)

    # Load query vector: q_rot — [BLOCK_D] float32
    q_addrs = (
        bid * stride_qb
        + head_offs[:, None] * stride_qh
        + d_offs[None, :]
    )
    q_rot = tl.load(
        Q_rot_ptr + q_addrs,
        mask=head_mask[:, None] & d_mask[None, :],
        other=0.0,
    ).to(tl.float32)

    # Precompute byte/bit index vectors for MSE gather loads
    if not KEY_FP8:
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_mask = (1 << MSE_BITS) - 1

    # Precompute value bit/byte index vectors (loop-invariant)
    if VQB == 3:
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8

    # Online softmax accumulators
    m_prev = tl.full([8], -float("inf"), dtype=tl.float32)
    l_prev = tl.zeros([8], dtype=tl.float32)
    acc = tl.zeros([8, BLOCK_D], dtype=tl.float32)

    bt_base = bid * stride_bt_b

    # ================================================================
    # TILED LOOP: process BLOCK_KV tokens per iteration
    # ================================================================
    for start_n in range(split_start, split_end, BLOCK_KV):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end

        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx,
            mask=kv_mask,
            other=0,
        ).to(tl.int64)

        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        # ============================================================
        # COMPUTE ATTENTION SCORES: [BLOCK_KV]
        # ============================================================
        if KEY_FP8:
            k_addrs = slot_bases[:, None] + d_offs[None, :]
            k_raw = tl.load(
                KV_cache_ptr + k_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            )
            if FP8_E4B15:
                k_float = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
            else:
                k_float = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            scores = (
                tl.sum(
                    tl.where(
                        d_mask[None, None, :],
                        q_rot[:, None, :] * k_float[None, :, :],
                        0.0,
                    ),
                    axis=2,
                )
                * ATTN_SCALE
            )
            scores = tl.where(kv_mask[None, :], scores, -float("inf"))
        else:
            # MSE unpack + norms
            mse_addrs0 = slot_bases[:, None] + mse_byte_idx[None, :]
            mse_raw0 = tl.load(
                KV_cache_ptr + mse_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            mse_raw1 = tl.load(
                KV_cache_ptr + mse_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = mse_raw0 | (mse_raw1 << 8)
            mse_idx = (raw16 >> mse_bit_shift[None, :]) & mse_mask

            # Centroid gather + dot product
            c_vals = tl.load(
                Centroids_ptr + mse_idx,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0.0,
            )

            # Norm correction: re-normalize centroid vector to unit norm
            if NORM_CORRECTION:
                c_norm_sq = tl.sum(
                    tl.where(d_mask[None, :], c_vals * c_vals, 0.0),
                    axis=1,
                )
                c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
                c_vals = c_vals * c_inv_norm[:, None]

            term1 = tl.sum(
                tl.where(
                    d_mask[None, None, :],
                    q_rot[:, None, :] * c_vals[None, :, :],
                    0.0,
                ),
                axis=2,
            )

            # Load norms (fp16 -> fp32): norms are at MSE_BYTES offset
            norm_bases = slot_bases + MSE_BYTES
            n_lo = tl.load(KV_cache_ptr + norm_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            n_hi = tl.load(KV_cache_ptr + norm_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            vec_norms = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

            scores = vec_norms[None, :] * term1 * ATTN_SCALE
            scores = tl.where(kv_mask[None, :], scores, -float("inf"))

        # ============================================================
        # ONLINE SOFTMAX UPDATE (block-level)
        # ============================================================
        n_e_max = tl.maximum(tl.max(scores, 1), m_prev)
        re_scale = tl.exp(m_prev - n_e_max)
        p = tl.exp(scores - n_e_max[:, None])

        # ============================================================
        # VALUE LOAD + DEQUANTIZE: [BLOCK_KV, BLOCK_D]
        # ============================================================
        val_bases = slot_bases + KPS

        if VQB == 3:
            val_addrs0 = val_bases[:, None] + val_byte_idx[None, :]
            val_raw0 = tl.load(
                KV_cache_ptr + val_addrs0,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            val_raw1 = tl.load(
                KV_cache_ptr + val_addrs0 + 1,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            raw16 = val_raw0 | (val_raw1 << 8)
            v_idx = ((raw16 >> val_bit_shift[None, :]) & 0x7).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]
        else:  # VQB == 4
            vb_idx = d_offs // 2
            vb_shift = (d_offs % 2) * 4
            val_addrs = val_bases[:, None] + vb_idx[None, :]
            val_raw = tl.load(
                KV_cache_ptr + val_addrs,
                mask=kv_mask[:, None] & d_mask[None, :],
                other=0,
            ).to(tl.int32)
            v_idx = ((val_raw >> vb_shift[None, :]) & 0xF).to(tl.float32)

            sc_bases = val_bases + VAL_DATA_BYTES
            sc_lo = tl.load(KV_cache_ptr + sc_bases, mask=kv_mask, other=0).to(
                tl.uint16
            )
            sc_hi = tl.load(KV_cache_ptr + sc_bases + 1, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_scales = (
                (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            )
            zr_lo = tl.load(KV_cache_ptr + sc_bases + 2, mask=kv_mask, other=0).to(
                tl.uint16
            )
            zr_hi = tl.load(KV_cache_ptr + sc_bases + 3, mask=kv_mask, other=0).to(
                tl.uint16
            )
            v_zeros = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
            values = v_idx * v_scales[:, None] + v_zeros[:, None]

        # ============================================================
        # WEIGHTED VALUE ACCUMULATION
        # ============================================================
        acc = acc * re_scale[:, None] + tl.sum(
            p[:, :, None] * values[None, :, :], 1
        )
        l_prev = l_prev * re_scale + tl.sum(p, 1)
        m_prev = n_e_max

    # Store partial result
    out_base = (
        bid * stride_mid_b
        + head_offs[:, None] * stride_mid_h
        + sid * stride_mid_s
    )
    safe_l = tl.where(l_prev > 0.0, l_prev, 1.0)
    tl.store(
        Mid_o_ptr + out_base + d_offs[None, :],
        acc / safe_l[:, None],
        mask=head_mask[:, None] & d_mask[None, :],
    )
    lse = m_prev + tl.log(safe_l)
    lse_base = (
        bid * stride_mid_b + head_offs * stride_mid_h + sid * stride_mid_s
    )
    tl.store(Mid_o_ptr + lse_base + HEAD_DIM, lse, mask=head_mask)


# The target model has six query heads per KV head.  Expressing those heads as
# one Triton tensor rounds the leading dimension to eight, inflating live state
# and register pressure.  This exact-format specialization keeps the shared KV
# load/dequantization but spells out the six independent FP32 reductions.  It
# is selected only for the bitwise-equivalent KV2 path below.
@triton.jit
def _tq_decode_stage1_six_scalar_mse4_v4_nc(
    Q_rot_ptr,
    KV_cache_ptr,
    Block_table_ptr,
    Seq_lens_ptr,
    Centroids_ptr,
    Mid_o_ptr,
    stride_qb,
    stride_qh,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    stride_mid_b,
    stride_mid_h,
    stride_mid_s,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_SPLITS: tl.constexpr,
    ATTN_SCALE: tl.constexpr,
):
    bid = tl.program_id(0)
    kv_head = tl.program_id(1)
    sid = tl.program_id(2)

    seq_len = tl.load(Seq_lens_ptr + bid)
    split_len = tl.cdiv(seq_len, NUM_KV_SPLITS)
    split_start = split_len * sid
    split_end = tl.minimum(split_start + split_len, seq_len)
    if split_start >= split_end:
        return

    d_offs = tl.arange(0, 256)
    kv_range = tl.arange(0, 2)
    q_base = bid * stride_qb + kv_head * 6 * stride_qh
    q0 = tl.load(Q_rot_ptr + q_base + 0 * stride_qh + d_offs).to(tl.float32)
    q1 = tl.load(Q_rot_ptr + q_base + 1 * stride_qh + d_offs).to(tl.float32)
    q2 = tl.load(Q_rot_ptr + q_base + 2 * stride_qh + d_offs).to(tl.float32)
    q3 = tl.load(Q_rot_ptr + q_base + 3 * stride_qh + d_offs).to(tl.float32)
    q4 = tl.load(Q_rot_ptr + q_base + 4 * stride_qh + d_offs).to(tl.float32)
    q5 = tl.load(Q_rot_ptr + q_base + 5 * stride_qh + d_offs).to(tl.float32)

    byte_idx = d_offs // 2
    bit_shift = (d_offs % 2) * 4
    m0 = tl.full([], -float("inf"), tl.float32)
    m1 = tl.full([], -float("inf"), tl.float32)
    m2 = tl.full([], -float("inf"), tl.float32)
    m3 = tl.full([], -float("inf"), tl.float32)
    m4 = tl.full([], -float("inf"), tl.float32)
    m5 = tl.full([], -float("inf"), tl.float32)
    l0 = tl.zeros([], tl.float32)
    l1 = tl.zeros([], tl.float32)
    l2 = tl.zeros([], tl.float32)
    l3 = tl.zeros([], tl.float32)
    l4 = tl.zeros([], tl.float32)
    l5 = tl.zeros([], tl.float32)
    acc0 = tl.zeros([256], tl.float32)
    acc1 = tl.zeros([256], tl.float32)
    acc2 = tl.zeros([256], tl.float32)
    acc3 = tl.zeros([256], tl.float32)
    acc4 = tl.zeros([256], tl.float32)
    acc5 = tl.zeros([256], tl.float32)
    bt_base = bid * stride_bt_b

    for start_n in range(split_start, split_end, 2):
        kv_offs = start_n + kv_range
        kv_mask = kv_offs < split_end
        page_idx = kv_offs // BLOCK_SIZE
        page_off = kv_offs % BLOCK_SIZE
        block_nums = tl.load(
            Block_table_ptr + bt_base + page_idx, mask=kv_mask, other=0
        ).to(tl.int64)
        slot_bases = (
            block_nums * stride_cache_block
            + page_off.to(tl.int64) * stride_cache_pos
            + tl.cast(kv_head, tl.int64) * stride_cache_head
        )

        key_raw = tl.load(
            KV_cache_ptr + slot_bases[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.int32)
        key_idx = (key_raw >> bit_shift[None, :]) & 0xF
        key = tl.load(
            Centroids_ptr + key_idx, mask=kv_mask[:, None], other=0.0
        )
        norm_sq = tl.sum(key * key, axis=1)
        key = key * (1.0 / tl.sqrt(norm_sq + 1e-16))[:, None]
        norm_lo = tl.load(
            KV_cache_ptr + slot_bases + 128, mask=kv_mask, other=0
        ).to(tl.uint16)
        norm_hi = tl.load(
            KV_cache_ptr + slot_bases + 129, mask=kv_mask, other=0
        ).to(tl.uint16)
        key_norm = (
            (norm_lo | (norm_hi << 8))
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )

        score0 = tl.sum(q0[None, :] * key, axis=1) * key_norm * ATTN_SCALE
        score1 = tl.sum(q1[None, :] * key, axis=1) * key_norm * ATTN_SCALE
        score2 = tl.sum(q2[None, :] * key, axis=1) * key_norm * ATTN_SCALE
        score3 = tl.sum(q3[None, :] * key, axis=1) * key_norm * ATTN_SCALE
        score4 = tl.sum(q4[None, :] * key, axis=1) * key_norm * ATTN_SCALE
        score5 = tl.sum(q5[None, :] * key, axis=1) * key_norm * ATTN_SCALE
        score0 = tl.where(kv_mask, score0, -float("inf"))
        score1 = tl.where(kv_mask, score1, -float("inf"))
        score2 = tl.where(kv_mask, score2, -float("inf"))
        score3 = tl.where(kv_mask, score3, -float("inf"))
        score4 = tl.where(kv_mask, score4, -float("inf"))
        score5 = tl.where(kv_mask, score5, -float("inf"))

        next_m0 = tl.maximum(tl.max(score0, axis=0), m0)
        next_m1 = tl.maximum(tl.max(score1, axis=0), m1)
        next_m2 = tl.maximum(tl.max(score2, axis=0), m2)
        next_m3 = tl.maximum(tl.max(score3, axis=0), m3)
        next_m4 = tl.maximum(tl.max(score4, axis=0), m4)
        next_m5 = tl.maximum(tl.max(score5, axis=0), m5)
        rescale0, p0 = tl.exp(m0 - next_m0), tl.exp(score0 - next_m0)
        rescale1, p1 = tl.exp(m1 - next_m1), tl.exp(score1 - next_m1)
        rescale2, p2 = tl.exp(m2 - next_m2), tl.exp(score2 - next_m2)
        rescale3, p3 = tl.exp(m3 - next_m3), tl.exp(score3 - next_m3)
        rescale4, p4 = tl.exp(m4 - next_m4), tl.exp(score4 - next_m4)
        rescale5, p5 = tl.exp(m5 - next_m5), tl.exp(score5 - next_m5)

        value_base = slot_bases + 130
        value_raw = tl.load(
            KV_cache_ptr + value_base[:, None] + byte_idx[None, :],
            mask=kv_mask[:, None],
            other=0,
        ).to(tl.int32)
        value_idx = ((value_raw >> bit_shift[None, :]) & 0xF).to(tl.float32)
        scale_lo = tl.load(
            KV_cache_ptr + value_base + 128, mask=kv_mask, other=0
        ).to(tl.uint16)
        scale_hi = tl.load(
            KV_cache_ptr + value_base + 129, mask=kv_mask, other=0
        ).to(tl.uint16)
        value_scale = (
            (scale_lo | (scale_hi << 8))
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        zero_lo = tl.load(
            KV_cache_ptr + value_base + 130, mask=kv_mask, other=0
        ).to(tl.uint16)
        zero_hi = tl.load(
            KV_cache_ptr + value_base + 131, mask=kv_mask, other=0
        ).to(tl.uint16)
        value_zero = (
            (zero_lo | (zero_hi << 8))
            .to(tl.float16, bitcast=True)
            .to(tl.float32)
        )
        value = value_idx * value_scale[:, None] + value_zero[:, None]

        acc0 = acc0 * rescale0 + tl.sum(p0[:, None] * value, axis=0)
        acc1 = acc1 * rescale1 + tl.sum(p1[:, None] * value, axis=0)
        acc2 = acc2 * rescale2 + tl.sum(p2[:, None] * value, axis=0)
        acc3 = acc3 * rescale3 + tl.sum(p3[:, None] * value, axis=0)
        acc4 = acc4 * rescale4 + tl.sum(p4[:, None] * value, axis=0)
        acc5 = acc5 * rescale5 + tl.sum(p5[:, None] * value, axis=0)
        l0, m0 = l0 * rescale0 + tl.sum(p0, axis=0), next_m0
        l1, m1 = l1 * rescale1 + tl.sum(p1, axis=0), next_m1
        l2, m2 = l2 * rescale2 + tl.sum(p2, axis=0), next_m2
        l3, m3 = l3 * rescale3 + tl.sum(p3, axis=0), next_m3
        l4, m4 = l4 * rescale4 + tl.sum(p4, axis=0), next_m4
        l5, m5 = l5 * rescale5 + tl.sum(p5, axis=0), next_m5

    out_base = (
        bid * stride_mid_b + kv_head * 6 * stride_mid_h + sid * stride_mid_s
    )
    safe_l0 = tl.where(l0 > 0.0, l0, 1.0)
    safe_l1 = tl.where(l1 > 0.0, l1, 1.0)
    safe_l2 = tl.where(l2 > 0.0, l2, 1.0)
    safe_l3 = tl.where(l3 > 0.0, l3, 1.0)
    safe_l4 = tl.where(l4 > 0.0, l4, 1.0)
    safe_l5 = tl.where(l5 > 0.0, l5, 1.0)
    tl.store(Mid_o_ptr + out_base + 0 * stride_mid_h + d_offs, acc0 / safe_l0)
    tl.store(Mid_o_ptr + out_base + 1 * stride_mid_h + d_offs, acc1 / safe_l1)
    tl.store(Mid_o_ptr + out_base + 2 * stride_mid_h + d_offs, acc2 / safe_l2)
    tl.store(Mid_o_ptr + out_base + 3 * stride_mid_h + d_offs, acc3 / safe_l3)
    tl.store(Mid_o_ptr + out_base + 4 * stride_mid_h + d_offs, acc4 / safe_l4)
    tl.store(Mid_o_ptr + out_base + 5 * stride_mid_h + d_offs, acc5 / safe_l5)
    tl.store(Mid_o_ptr + out_base + 0 * stride_mid_h + 256, m0 + tl.log(safe_l0))
    tl.store(Mid_o_ptr + out_base + 1 * stride_mid_h + 256, m1 + tl.log(safe_l1))
    tl.store(Mid_o_ptr + out_base + 2 * stride_mid_h + 256, m2 + tl.log(safe_l2))
    tl.store(Mid_o_ptr + out_base + 3 * stride_mid_h + 256, m3 + tl.log(safe_l3))
    tl.store(Mid_o_ptr + out_base + 4 * stride_mid_h + 256, m4 + tl.log(safe_l4))
    tl.store(Mid_o_ptr + out_base + 5 * stride_mid_h + 256, m5 + tl.log(safe_l5))


# ---------------------------------------------------------------------------
# Pre-dequant kernel: Bulk dequant K (MSE+norms) and V to fp16
# ---------------------------------------------------------------------------


@triton.jit
def _tq_full_dequant_kv(
    KV_cache_ptr,
    Block_table_ptr,
    Centroids_ptr,
    K_out_ptr,  # [B, Hk, max_seq, D] float16
    V_out_ptr,  # [B, Hk, max_seq, D] float16
    stride_ko_b,
    stride_ko_h,
    stride_ko_s,
    stride_vo_b,
    stride_vo_h,
    stride_vo_s,
    stride_cache_block,
    stride_cache_pos,
    stride_cache_head,
    stride_bt_b,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
    MSE_BYTES: tl.constexpr,
    KPS: tl.constexpr,
    VQB: tl.constexpr,
    VAL_DATA_BYTES: tl.constexpr,
    MSE_BITS: tl.constexpr,
    KEY_FP8: tl.constexpr,
    BLOCK_D: tl.constexpr,
    NORM_CORRECTION: tl.constexpr = 0,
    FP8_E4B15: tl.constexpr = 0,  # 1 = use e4b15 (Ampere/Ada), 0 = e4nv (Hopper+)
):
    """Full dequant: reconstruct K (MSE centroids * norm or FP8) and V to fp16."""
    pos = tl.program_id(0)
    bh = tl.program_id(1)
    bid = bh // NUM_KV_HEADS
    hid = bh % NUM_KV_HEADS

    page_idx = pos // BLOCK_SIZE
    page_off = pos % BLOCK_SIZE
    block_num = tl.load(Block_table_ptr + bid * stride_bt_b + page_idx).to(tl.int64)
    slot_base = (
        block_num * stride_cache_block
        + tl.cast(page_off, tl.int64) * stride_cache_pos
        + tl.cast(hid, tl.int64) * stride_cache_head
    )

    d_offs = tl.arange(0, BLOCK_D)
    d_mask = d_offs < HEAD_DIM

    # === K dequant ===
    ko_base = bid * stride_ko_b + hid * stride_ko_h + pos * stride_ko_s
    if KEY_FP8:
        k_raw = tl.load(KV_cache_ptr + slot_base + d_offs, mask=d_mask, other=0)
        if FP8_E4B15:
            k_recon = k_raw.to(tl.float8e4b15, bitcast=True).to(tl.float32)
        else:
            k_recon = k_raw.to(tl.float8e4nv, bitcast=True).to(tl.float32)
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)
    else:
        # MSE unpack (3-bit or 4-bit) + norms
        mse_bit_off = d_offs * MSE_BITS
        mse_byte_idx = mse_bit_off // 8
        mse_bit_shift = mse_bit_off % 8
        mse_umask = (1 << MSE_BITS) - 1

        mse_raw0 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        mse_raw1 = tl.load(
            KV_cache_ptr + slot_base + mse_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_key = mse_raw0 | (mse_raw1 << 8)
        mse_idx = (raw16_key >> mse_bit_shift) & mse_umask

        k_mse = tl.load(Centroids_ptr + mse_idx, mask=d_mask, other=0.0)

        # Norm correction: re-normalize centroid vector to unit norm
        if NORM_CORRECTION:
            c_norm_sq = tl.sum(tl.where(d_mask, k_mse * k_mse, 0.0), axis=0)
            c_inv_norm = 1.0 / tl.sqrt(c_norm_sq + 1e-16)
            k_mse = k_mse * c_inv_norm

        # Norms at MSE_BYTES offset (no QJL bytes)
        norm_base = slot_base + MSE_BYTES
        n_lo = tl.load(KV_cache_ptr + norm_base).to(tl.uint16)
        n_hi = tl.load(KV_cache_ptr + norm_base + 1).to(tl.uint16)
        vec_norm = (n_lo | (n_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)

        k_recon = vec_norm * k_mse
        tl.store(K_out_ptr + ko_base + d_offs, k_recon.to(tl.float16), mask=d_mask)

    # === V dequant ===
    val_base = slot_base + KPS
    if VQB == 4:
        vb_idx = d_offs // 2
        vb_shift = (d_offs % 2) * 4
        val_raw = tl.load(KV_cache_ptr + val_base + vb_idx, mask=d_mask, other=0).to(
            tl.int32
        )
        v_idx = ((val_raw >> vb_shift) & 0xF).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    elif VQB == 3:
        # 3-bit value unpack: 8 values per 3 bytes
        val_bit_off = d_offs * 3
        val_byte_idx = val_bit_off // 8
        val_bit_shift = val_bit_off % 8
        val_raw0 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx, mask=d_mask, other=0
        ).to(tl.int32)
        val_raw1 = tl.load(
            KV_cache_ptr + val_base + val_byte_idx + 1, mask=d_mask, other=0
        ).to(tl.int32)
        raw16_val = val_raw0 | (val_raw1 << 8)
        v_idx = ((raw16_val >> val_bit_shift) & 0x7).to(tl.float32)

        sc_base = val_base + VAL_DATA_BYTES
        sc_lo = tl.load(KV_cache_ptr + sc_base).to(tl.uint16)
        sc_hi = tl.load(KV_cache_ptr + sc_base + 1).to(tl.uint16)
        v_scale = (sc_lo | (sc_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        zr_lo = tl.load(KV_cache_ptr + sc_base + 2).to(tl.uint16)
        zr_hi = tl.load(KV_cache_ptr + sc_base + 3).to(tl.uint16)
        v_zero = (zr_lo | (zr_hi << 8)).to(tl.float16, bitcast=True).to(tl.float32)
        v_vals = v_idx * v_scale + v_zero
    else:
        v_vals = tl.zeros([BLOCK_D], dtype=tl.float32)

    vo_base = bid * stride_vo_b + hid * stride_vo_h + pos * stride_vo_s
    tl.store(V_out_ptr + vo_base + d_offs, v_vals.to(tl.float16), mask=d_mask)


# ---------------------------------------------------------------------------
# Stage 2: Reuse from triton_decode_attention.py
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Launcher — cached constants + fused GEMM
# ---------------------------------------------------------------------------

_layout_cache: dict = {}


def _get_layout(D, mse_bits, value_quant_bits, key_packed_size):
    """Get cached layout constants."""
    key = (D, mse_bits, value_quant_bits, key_packed_size)
    cfg = _layout_cache.get(key)
    if cfg is None:
        val_data_bytes = math.ceil(D * value_quant_bits / 8)
        cfg = {
            "mse_bytes": math.ceil(D * mse_bits / 8),
            "val_data_bytes": val_data_bytes,
            "mse_bits": mse_bits,
            "n_centroids": 2**mse_bits,
            "BLOCK_D": triton.next_power_of_2(D),
        }
        _layout_cache[key] = cfg
    return cfg


def triton_turboquant_decode_attention(
    query: torch.Tensor,  # [B, Hq, D] — original query
    kv_cache: torch.Tensor,  # [num_blocks, block_size, Hk, padded_slot] uint8
    block_table: torch.Tensor,  # [B, max_num_blocks] int32
    seq_lens: torch.Tensor,  # [B] int32
    Pi: torch.Tensor,  # [D, D] float32
    centroids: torch.Tensor,  # [n_centroids] float32
    scale: float,
    mse_bits: int,
    key_packed_size: int,
    value_quant_bits: int,
    key_fp8: bool = False,
    norm_correction: bool = False,
    PiT: torch.Tensor | None = None,  # [D, D] pre-computed Pi.T contiguous
    # Pre-allocated buffers (optional, avoids per-call allocation)
    mid_o_buf: torch.Tensor | None = None,
    output_buf: torch.Tensor | None = None,
    lse_buf: torch.Tensor | None = None,
    buf_holder: Any = None,
    max_num_kv_splits: int = 32,  # fixed split count (must be constant for cudagraph)
    block_kv: int = 2,
) -> torch.Tensor:
    """Launch fused TQ decode attention (Triton stage1 + stage2).

    Returns: output tensor [B, Hq, D] in query's dtype.
    """
    B, Hq, D = query.shape
    Hk = kv_cache.shape[2]
    block_size = kv_cache.shape[1]
    kv_group_size = Hq // Hk
    device = query.device

    cfg = _get_layout(D, mse_bits, value_quant_bits, key_packed_size)

    # Compute q_rot = q @ Pi.T (rotated query for MSE key scoring)
    # FP8 path: pass query directly (float16); kernel casts inline.
    # MSE path: still needs external GEMM (cuBLAS), so q_rot is float32.
    if key_fp8:
        q_rot = query.contiguous()
    else:
        q_float = query.float()
        if PiT is None:
            PiT = Pi.T.contiguous()
        q_rot = (q_float @ PiT).contiguous()

    NUM_KV_SPLITS = max_num_kv_splits

    if (
        mid_o_buf is not None
        and mid_o_buf.shape[0] >= B
        and mid_o_buf.shape[2] >= NUM_KV_SPLITS
    ):
        mid_o = mid_o_buf[:B, :Hq, :NUM_KV_SPLITS, :]
    else:
        mid_o = torch.empty(
            B,
            Hq,
            NUM_KV_SPLITS,
            D + 1,
            dtype=torch.float32,
            device=device,
        )
        if buf_holder is not None:
            buf_holder._tq_mid_o_buf = mid_o

    # Stage 1: split-KV tiled attention scoring + value accumulation
    fp8_e4b15 = _use_fp8_e4b15(device.index or 0)
    BLOCK_KV = block_kv
    HEAD_GROUP = 6 if kv_group_size % 6 == 0 else 1
    use_six_scalar = (
        BLOCK_KV == 2
        and Hq == 24
        and Hk == 4
        and D == 256
        and kv_group_size == 6
        and mse_bits == 4
        and cfg["mse_bytes"] == 128
        and key_packed_size == 130
        and value_quant_bits == 4
        and cfg["val_data_bytes"] == 128
        and not key_fp8
        and norm_correction
    )
    if use_six_scalar:
        grid = (B, Hk, NUM_KV_SPLITS)
        _tq_decode_stage1_six_scalar_mse4_v4_nc[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            ATTN_SCALE=scale,
            num_warps=1,
            num_stages=1,
        )
    else:
        grid = (B, triton.cdiv(Hq, HEAD_GROUP), NUM_KV_SPLITS)
        _tq_decode_stage1[grid](
            q_rot,
            kv_cache,
            block_table,
            seq_lens,
            centroids,
            mid_o,
            q_rot.stride(0),
            q_rot.stride(1),
            kv_cache.stride(0),
            kv_cache.stride(1),
            kv_cache.stride(2),
            block_table.stride(0),
            mid_o.stride(0),
            mid_o.stride(1),
            mid_o.stride(2),
            NUM_QUERY_HEADS=Hq,
            NUM_KV_HEADS=Hk,
            HEAD_DIM=D,
            BLOCK_SIZE=block_size,
            NUM_KV_SPLITS=NUM_KV_SPLITS,
            KV_GROUP_SIZE=kv_group_size,
            MSE_BITS=mse_bits,
            MSE_BYTES=cfg["mse_bytes"],
            KPS=key_packed_size,
            VQB=value_quant_bits,
            VAL_DATA_BYTES=cfg["val_data_bytes"],
            ATTN_SCALE=scale,
            BLOCK_D=cfg["BLOCK_D"],
            BLOCK_KV=BLOCK_KV,
            HEAD_GROUP=HEAD_GROUP,
            KEY_FP8=1 if key_fp8 else 0,
            NORM_CORRECTION=1 if norm_correction else 0,
            FP8_E4B15=fp8_e4b15,
            num_warps=1,
            num_stages=1,
        )

    # Stage 2: Reduce across KV splits
    # Output in query dtype — eliminates float16_copy kernel after stage2
    out_dtype = query.dtype
    if (
        output_buf is not None
        and output_buf.shape[0] >= B
        and output_buf.dtype == out_dtype
    ):
        output = output_buf[:B, :Hq, :D]
    else:
        output = torch.empty(B, Hq, D, dtype=out_dtype, device=device)
        if buf_holder is not None:
            buf_holder._tq_output_buf = output
    if lse_buf is not None and lse_buf.shape[0] >= B:
        lse = lse_buf[:B, :Hq]
    else:
        lse = torch.empty(B, Hq, dtype=torch.float32, device=device)
        if buf_holder is not None:
            buf_holder._tq_lse_buf = lse

    grid2 = (B, Hq)
    _fwd_kernel_stage2[grid2](
        mid_o,
        output,
        lse,
        seq_lens,
        mid_o.stride(0),
        mid_o.stride(1),
        mid_o.stride(2),
        output.stride(0),
        output.stride(1),
        lse.stride(0),
        NUM_KV_SPLITS=NUM_KV_SPLITS,
        BLOCK_DV=cfg["BLOCK_D"],
        Lv=D,
        OUTPUT_FP16=1 if out_dtype == torch.float16 else 0,
        num_warps=4,
        num_stages=2,
    )

    return output  # already in query dtype
