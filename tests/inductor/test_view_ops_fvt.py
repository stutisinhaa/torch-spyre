# Copyright 2025 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
FVT: View Operations — reshape, squeeze, unsqueeze, expand
==========================================================

Two levels of testing:
  1. MODULE LEVEL  – whole Gemma-4 sub-module forward passes that rely on
                    view ops internally (extracted shapes from the 26B-A4B-it
                    model trace).
  2. OPERATION LEVEL – direct standalone tests of each of the four priority
                    view ops across 0-D to 6-D, all three floating-point
                    dtypes (fp16, bf16, fp32), and both aligned and
                    *unaligned stick* scenarios.

Stick alignment notes
---------------------
The Spyre hardware stores tensors in 128-byte "sticks".  The innermost
(last) tensor dimension is the "stick dimension".  The number of elements
that fit in one stick depends on dtype:

    fp16 / bf16  →  128 B / 2 B  =  64 elements per stick
    fp32         →  128 B / 4 B  =  32 elements per stick

An innermost dimension is **aligned** when its size is divisible by the
per-stick element count (64 or 32).  It is **unaligned** when:
  * the size is *less than* one stick width, or
  * the size is *not divisible* by the stick width.

Examples for fp16/bf16 (stick = 64 elems):
  aligned   :  64, 128, 256, 704, 2816  (multiples of 64)
  unaligned :  1, 7, 13, 30, 33, 63, 65, 70, 72, 127  (not multiples of 64)

Examples for fp32 (stick = 32 elems):
  aligned   :  32, 64, 128, 256
  unaligned :  1, 7, 15, 33, 63, 65

Size-1 trailing dimensions are a special "ultra-unaligned" case (1 element
is always less than one stick) and are exercised throughout both sections.
"""

import math
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils_inductor import compare_with_cpu, cached_randn

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _randn(shape, dtype):
    """Produce a deterministic float tensor, supporting 0-D (shape=())."""
    torch.manual_seed(0xAFFE)
    return torch.randn(shape, dtype=dtype)


def _atol(dtype):
    """Return a suitable absolute tolerance for each dtype."""
    return 1e-2 if dtype == torch.float16 else 1e-3


# ---------------------------------------------------------------------------
# Section 1 — Module-Level Tests (Gemma-4 sub-module shapes)
# ---------------------------------------------------------------------------
# Each test instantiates a minimal nn.Module that mirrors the view operations
# performed inside a real Gemma-4 sub-module, and runs compare_with_cpu so
# the compiled Spyre result is checked against the CPU reference.


class TestModuleLevel(unittest.TestCase):
    """Module-level view-op tests using shapes from the Gemma-4 26B-A4B trace."""

    # ------------------------------------------------------------------
    # Reshape / view — QKV projection split and reassemble
    # ------------------------------------------------------------------

    def test_module_qkv_view_split_and_flatten(self):
        """
        Gemma4TextAttention QKV projection:
          • split:   [B, S, H*D]  →  [B, S, H, D]   (view)
          • flatten: [B, S, H, D] →  [B, S, H*D]    (reshape)
        Covers sliding (H=16,D=256) and full-attention (H=16,D=512) layers.
        """
        for B, S, H, D in [
            (1, 32, 16, 256),   # sliding, prefill
            (1,  1, 16, 256),   # sliding, single-token decode
            (1, 32, 16, 512),   # full-attention (Config D)
            (1,  1, 16, 512),   # full-attention decode
        ]:
            x_flat = cached_randn((B, S, H * D), dtype=torch.bfloat16)
            x_4d   = cached_randn((B, S, H, D),  dtype=torch.bfloat16)
            compare_with_cpu(lambda t, b=B,s=S,h=H,d=D: t.view(b, s, h, d),
                             x_flat, run_eager=True, atol=0.0)
            compare_with_cpu(lambda t, b=B,s=S,h=H,d=D: t.reshape(b, s, h * d),
                             x_4d,   run_eager=True, atol=0.0)

    def test_module_moe_token_flatten_unflatten(self):
        """
        Gemma4TextMoE:  [B, S, hidden] ↔ [B*S, hidden]
        Standard routing flatten + reassembly.
        """
        B, S, H = 4, 32, 2816
        x = cached_randn((B, S, H), dtype=torch.bfloat16)
        compare_with_cpu(lambda t: t.view(-1, H),     x, run_eager=True, atol=0.0)
        x2 = cached_randn((B * S, H), dtype=torch.bfloat16)
        compare_with_cpu(lambda t: t.view(B, S, H),  x2, run_eager=True, atol=0.0)

    def test_module_moe_dispatch_index_flatten(self):
        """
        Gemma4TextMoE:  [B*S, K]  →  [B*S*K]   (int64 indices)
        Dispatching top-k expert indices, 8 experts per token.
        """
        x = torch.arange(32 * 8, dtype=torch.int64).reshape(32, 8)
        compare_with_cpu(lambda t: t.reshape(-1), x, run_eager=True, atol=0.0)

    def test_module_attention_head_transpose_and_restore(self):
        """
        Gemma4TextAttention:
          • head transpose:    [B, S, H, D]  →  [B, H, S, D]   (transpose+contiguous)
          • post-attn restore: [B, H, S, D]  →  [B, S, H, D]   (transpose+reshape)
        """
        B, S, H, D = 1, 32, 16, 256
        x = cached_randn((B, S, H, D), dtype=torch.bfloat16)
        y = cached_randn((B, H, S, D), dtype=torch.bfloat16)

        compare_with_cpu(lambda t: t.transpose(1, 2).contiguous(),
                         x, run_eager=True, atol=0.0)
        compare_with_cpu(lambda t, b=B,s=S,h=H,d=D:
                         t.transpose(1, 2).reshape(b, s, h * d),
                         y, run_eager=True, atol=0.0)

    def test_module_gqa_unsqueeze_expand(self):
        """
        Gemma4TextAttention GQA KV broadcast:
          [B, KVH, S, D]  →  [B, H, S, D]
        via unsqueeze(2) → expand → reshape.
        Covers 2x (sliding) and 8x (full-attention) expansion.
        """
        for B, KVH, H, S, D in [
            (1,  8, 16, 32, 256),  # 2x expansion (Config A)
            (1,  2, 16, 32, 512),  # 8x expansion (Config D)
            (1,  1, 16, 32, 256),  # 16x MQA mode (Config C)
        ]:
            exp = H // KVH
            x = cached_randn((B, KVH, S, D), dtype=torch.bfloat16)
            compare_with_cpu(
                lambda t, b=B, kvh=KVH, h=H, s=S, d=D, e=exp:
                    t.unsqueeze(2).expand(b, kvh, e, s, d).reshape(b, h, s, d),
                x, run_eager=True, atol=0.0,
            )

    def test_module_rmsnorm_unsqueeze_broadcast(self):
        """
        Gemma4RMSNorm:  weight [D] broadcast over [B, S, D]  via unsqueeze(0).unsqueeze(0)
        """
        B, S, D = 4, 128, 2816
        x = cached_randn((B, S, D), dtype=torch.bfloat16)
        w = cached_randn((D,),       dtype=torch.bfloat16)
        # Mirrors: (x * w.unsqueeze(0).unsqueeze(0))
        compare_with_cpu(
            lambda a, b: a * b.unsqueeze(0).unsqueeze(0),
            x, w, run_eager=True, atol=1e-2,
        )

    def test_module_rope_unsqueeze_freq_broadcast(self):
        """
        Gemma4RotaryEmbedding: freq table unsqueezed for head broadcast
          [B, S, D]  →  [B, S, 1, D]  via unsqueeze(2)
        """
        B, S, D = 1, 32, 256
        x = cached_randn((B, S, D), dtype=torch.bfloat16)
        compare_with_cpu(lambda t: t.unsqueeze(2), x, run_eager=True, atol=0.0)

    def test_module_vision_patch_flatten(self):
        """
        Gemma4VisionPatchEmbedder:
          [B, D, H_p, W_p]  →  [B, N, D]
        where N = H_p * W_p = 256 patches.
        """
        B, D, H_p, W_p = 1, 1152, 16, 16
        x = cached_randn((B, D, H_p, W_p), dtype=torch.bfloat16)
        compare_with_cpu(
            lambda t: t.view(B, D, -1).transpose(1, 2),
            x, run_eager=True, atol=0.0,
        )

    def test_module_vision_attention_head_split_d72(self):
        """
        Gemma4VisionEncoderLayer QKV split with non-power-of-2 head dim D=72.
        [B, N, 1152]  →  [B, N, H, D]  where H=16, D=72.
        D=72 is unaligned for fp16/bf16 (72 % 64 != 0).
        """
        B, N, H, D = 1, 280, 16, 72
        x = cached_randn((B, N, H * D), dtype=torch.bfloat16)
        compare_with_cpu(
            lambda t: t.view(B, N, H, D),
            x, run_eager=True, atol=0.0,
        )

    def test_module_vision_projector_reshape_pooling(self):
        """
        Gemma4VisionProjector:  [B, N, 1152] pool then project → [B, N', 2816]
        Simplified to pure reshape: [1, 280, 1152] → [1, 70, 4, 1152]
        (pool=4 to represent spatial downsampling step).
        """
        B, N, D = 1, 280, 1152
        pool = 4
        x = cached_randn((B, N, D), dtype=torch.bfloat16)
        compare_with_cpu(
            lambda t: t.reshape(B, N // pool, pool, D),
            x, run_eager=True, atol=0.0,
        )

    def test_module_attention_mask_expand(self):
        """
        Attention mask broadcast:
          [1, 1, S, S]  →  [B, H, S, S]  via expand
        """
        B, H, S = 4, 16, 32
        mask = cached_randn((1, 1, S, S), dtype=torch.bfloat16)
        compare_with_cpu(
            lambda t: t.expand(B, H, S, S),
            mask, run_eager=True, atol=0.0,
        )

    def test_module_token_embedding_squeeze(self):
        """
        After embedding lookup, squeeze a batch dimension if B=1.
          [1, S, D]  →  [S, D]  via squeeze(0)
        """
        S, D = 32, 2816
        x = cached_randn((1, S, D), dtype=torch.bfloat16)
        compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=0.0)

    def test_module_scalar_reshape_to_1d(self):
        """
        0-D scalar elevated to [1] via reshape(1) — arises with scalar bias terms.
        """
        x = _randn((), dtype=torch.bfloat16)
        compare_with_cpu(lambda t: t.reshape(1), x, run_eager=True, atol=0.0)


# ---------------------------------------------------------------------------
# Section 2 — Operation-Level: reshape
# ---------------------------------------------------------------------------

DTYPES = [torch.float16, torch.bfloat16, torch.float32]


class TestReshape(unittest.TestCase):
    """
    Standalone reshape tests: 0-D to 6-D, all three dtypes,
    aligned and unaligned stick innermost dimensions.
    """

    # ---- 0-D ----

    def test_reshape_0d_scalar_to_1d(self):
        """0-D scalar → [1]"""
        for dt in DTYPES:
            x = _randn((), dt)
            compare_with_cpu(lambda t: t.reshape(1), x, run_eager=True, atol=_atol(dt))

    def test_reshape_0d_scalar_to_2d(self):
        """0-D scalar → [1, 1]"""
        for dt in DTYPES:
            x = _randn((), dt)
            compare_with_cpu(lambda t: t.reshape(1, 1), x, run_eager=True, atol=_atol(dt))

    # ---- 1-D ----

    def test_reshape_1d_aligned_to_2d(self):
        """[256] → [16, 16]  (aligned: 256 % 64 == 0)"""
        for dt in DTYPES:
            x = cached_randn((256,), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(16, 16), x, run_eager=True, atol=_atol(dt))

    def test_reshape_1d_unaligned_to_2d(self):
        """[63] → [9, 7]  (unaligned: 63 % 64 != 0, size < 64)"""
        for dt in DTYPES:
            x = cached_randn((63,), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(9, 7), x, run_eager=True, atol=_atol(dt))

    def test_reshape_1d_size1_trailing(self):
        """[1] → [1, 1, 1]  (size-1 → size-1 dims)"""
        for dt in DTYPES:
            x = _randn((1,), dt)
            compare_with_cpu(lambda t: t.reshape(1, 1, 1), x, run_eager=True, atol=_atol(dt))

    def test_reshape_1d_gemma_hidden_to_2d(self):
        """[2816] → [128, 22]  (2816 % 64 == 0, aligned)"""
        for dt in DTYPES:
            x = cached_randn((2816,), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(128, 22), x, run_eager=True, atol=_atol(dt))

    # ---- 2-D ----

    def test_reshape_2d_aligned_flatten(self):
        """[64, 64] → [4096]  (aligned innermost)"""
        for dt in DTYPES:
            x = cached_randn((64, 64), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(-1), x, run_eager=True, atol=_atol(dt))

    def test_reshape_2d_unaligned_inner(self):
        """[32, 33] → [1056]  (33 % 64 != 0, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((32, 33), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(-1), x, run_eager=True, atol=_atol(dt))

    def test_reshape_2d_size1_col(self):
        """[128, 1] → [128]  (size-1 innermost dimension)"""
        for dt in DTYPES:
            x = cached_randn((128, 1), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(128), x, run_eager=True, atol=_atol(dt))

    def test_reshape_2d_to_3d_split(self):
        """[4, 2048] → [4, 8, 256]  (aligned split)"""
        for dt in DTYPES:
            x = cached_randn((4, 2048), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(4, 8, 256), x, run_eager=True, atol=_atol(dt))

    def test_reshape_2d_to_3d_unaligned_inner(self):
        """[4, 70] → [4, 2, 35]  (35 % 64 != 0, unaligned; 72 % 64 = 8)"""
        for dt in DTYPES:
            x = cached_randn((4, 70), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(4, 2, 35), x, run_eager=True, atol=_atol(dt))

    # ---- 3-D ----

    def test_reshape_3d_aligned_flatten_last_two(self):
        """[2, 16, 256] → [2, 4096]  (aligned innermost)"""
        for dt in DTYPES:
            x = cached_randn((2, 16, 256), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(2, -1), x, run_eager=True, atol=_atol(dt))

    def test_reshape_3d_unaligned_seq(self):
        """[3, 71, 2048] → [3, 71, 8, 256]  (S=71 is non-power-of-2 but inner is aligned 256)"""
        for dt in DTYPES:
            x = cached_randn((3, 71, 2048), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(3, 71, 8, 256),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_3d_unaligned_inner_d72(self):
        """[1, 280, 1152] → [1, 280, 16, 72]  (D=72 is unaligned: 72 % 64 != 0)"""
        for dt in DTYPES:
            x = cached_randn((1, 280, 1152), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(1, 280, 16, 72),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_3d_unaligned_inner_small(self):
        """[2, 8, 13] → [16, 13]  (13 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((2, 8, 13), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(16, 13), x, run_eager=True, atol=_atol(dt))

    def test_reshape_3d_size1_inner(self):
        """[4, 32, 1] → [4, 32]  (size-1 innermost)"""
        for dt in DTYPES:
            x = cached_randn((4, 32, 1), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(4, 32), x, run_eager=True, atol=_atol(dt))

    # ---- 4-D ----

    def test_reshape_4d_aligned_head_flatten(self):
        """[1, 16, 32, 256] → [1, 32, 4096]  (attention heads merge)"""
        for dt in DTYPES:
            x = cached_randn((1, 16, 32, 256), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(1, 32, 16 * 256),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_4d_unaligned_inner_d72(self):
        """[1, 16, 280, 72] → [1, 280, 1152]  (merge heads with D=72, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 16, 280, 72), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(1, 280, 16 * 72),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_4d_unaligned_inner_33(self):
        """[2, 4, 8, 33] → [2, 32, 33]  (33 is unaligned)"""
        for dt in DTYPES:
            x = cached_randn((2, 4, 8, 33), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(2, 32, 33),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_4d_size1_dim(self):
        """[1, 8, 1, 256] → [8, 256]  (size-1 batch and seq dims)"""
        for dt in DTYPES:
            x = cached_randn((1, 8, 1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(8, 256), x, run_eager=True, atol=_atol(dt))

    # ---- 5-D ----

    def test_reshape_5d_aligned_flatten(self):
        """[2, 4, 8, 16, 32] → [2, 4, 4096]  (aligned innermost 32 for fp32, 32<64 for fp16 unaligned)"""
        for dt in DTYPES:
            x = cached_randn((2, 4, 8, 16, 32), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(2, 4, -1),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_5d_unaligned_inner_7(self):
        """[1, 2, 4, 8, 7] → [1, 8, 56]  (7 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 8, 7), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(1, 8, 56),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_5d_to_6d_unfold(self):
        """[1, 2, 4, 8, 256] → [1, 2, 4, 8, 4, 64]  (aligned, last-dim split)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 8, 256), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(1, 2, 4, 8, 4, 64),
                             x, run_eager=True, atol=_atol(dt))

    # ---- 6-D ----

    def test_reshape_6d_aligned_collapse(self):
        """[1, 2, 4, 8, 4, 64] → [1, 2, 4, 2048]  (flatten last 3 dims)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 8, 4, 64), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(1, 2, 4, -1),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_6d_unaligned_inner(self):
        """[1, 2, 4, 8, 4, 7] → [1, 8, 224]  (7 is unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 8, 4, 7), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(1, 8, 224),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_6d_size1_innermost(self):
        """[2, 3, 4, 5, 6, 1] → [2, 3, 4, 5, 6]  (trailing size-1)"""
        for dt in DTYPES:
            x = cached_randn((2, 3, 4, 5, 6, 1), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(2, 3, 4, 5, 6),
                             x, run_eager=True, atol=_atol(dt))

    # ---- Inference dimension (-1) ----

    def test_reshape_inferred_dim_2d(self):
        """[2, 32, 16, 256] → [2, -1, 256]  (infer middle dim = 512)"""
        for dt in DTYPES:
            x = cached_randn((2, 32, 16, 256), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(2, -1, 256),
                             x, run_eager=True, atol=_atol(dt))

    def test_reshape_inferred_dim_unaligned(self):
        """[3, 7, 6] → [3, -1]  (infer dim = 42, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((3, 7, 6), dtype=dt)
            compare_with_cpu(lambda t: t.reshape(3, -1),
                             x, run_eager=True, atol=_atol(dt))


# ---------------------------------------------------------------------------
# Section 3 — Operation-Level: squeeze
# ---------------------------------------------------------------------------

class TestSqueeze(unittest.TestCase):
    """
    Standalone squeeze tests: 0-D to 6-D, all dtypes,
    aligned and unaligned stick scenarios.
    """

    # ---- 0-D / 1-D ----

    def test_squeeze_1d_size1_to_scalar(self):
        """[1] → []  (squeeze scalar out)"""
        for dt in DTYPES:
            x = _randn((1,), dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_1d_no_effect_aligned(self):
        """[256] → [256]  (no-op: dim 0 is not size-1)"""
        for dt in DTYPES:
            x = cached_randn((256,), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_1d_no_effect_unaligned(self):
        """[63] → [63]  (no-op: dim 0 not size-1, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((63,), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))

    # ---- 2-D ----

    def test_squeeze_2d_leading_dim_aligned(self):
        """[1, 256] → [256]  (remove batch dim; aligned inner)"""
        for dt in DTYPES:
            x = cached_randn((1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_2d_leading_dim_unaligned(self):
        """[1, 33] → [33]  (remove batch dim; 33 is unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 33), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_2d_trailing_size1(self):
        """[128, 1] → [128]  (trailing size-1 innermost)"""
        for dt in DTYPES:
            x = cached_randn((128, 1), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(-1), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_2d_all_singletons(self):
        """[1, 1] → []  (squeeze all; 0-D result)"""
        for dt in DTYPES:
            x = _randn((1, 1), dt)
            compare_with_cpu(lambda t: t.squeeze(), x, run_eager=True, atol=_atol(dt))

    # ---- 3-D ----

    def test_squeeze_3d_batch_dim_aligned(self):
        """[1, 32, 2816] → [32, 2816]  (batch squeeze; 2816 aligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 32, 2816), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_3d_seq_dim_aligned(self):
        """[2, 1, 256] → [2, 256]  (seq=1 decode step; 256 aligned)"""
        for dt in DTYPES:
            x = cached_randn((2, 1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(1), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_3d_seq_dim_unaligned(self):
        """[2, 1, 70] → [2, 70]  (seq=1 decode, 70 is unaligned: 70 % 64 != 0)"""
        for dt in DTYPES:
            x = cached_randn((2, 1, 70), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(1), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_3d_no_effect_non_singleton(self):
        """[1, 32, 2816] .squeeze(1) → [1, 32, 2816]  (dim-1 size=32, not 1)"""
        for dt in DTYPES:
            x = cached_randn((1, 32, 2816), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(1), x, run_eager=True, atol=_atol(dt))

    # ---- 4-D ----

    def test_squeeze_4d_decode_seq_dim(self):
        """[1, 1, 16, 256] → [1, 16, 256]  (single-token decode step; 256 aligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 16, 256), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(1), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_4d_decode_seq_dim_unaligned(self):
        """[1, 1, 16, 72] → [1, 16, 72]  (vision decode; D=72 unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 16, 72), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(1), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_4d_all_singletons(self):
        """[1, 1, 32, 1] → [32]  (squeeze all size-1 dims)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 32, 1), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_4d_all_singletons_unaligned(self):
        """[1, 1, 63, 1] → [63]  (63 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 63, 1), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_4d_tuple_dims(self):
        """[1, 16, 1, 256] .squeeze((0, 2)) → [16, 256]"""
        for dt in DTYPES:
            x = cached_randn((1, 16, 1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze((0, 2)), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_4d_negative_index(self):
        """[32, 8, 64, 1] .squeeze(-1) → [32, 8, 64]  (trailing size-1, inner 64 aligned)"""
        for dt in DTYPES:
            x = cached_randn((32, 8, 64, 1), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(-1), x, run_eager=True, atol=_atol(dt))

    # ---- 5-D ----

    def test_squeeze_5d_all_singletons(self):
        """[1, 1, 32, 1, 256] → [32, 256]  (aligned inner 256)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 32, 1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_5d_all_singletons_unaligned(self):
        """[1, 1, 32, 1, 30] → [32, 30]  (30 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 32, 1, 30), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(), x, run_eager=True, atol=_atol(dt))

    # ---- 6-D ----

    def test_squeeze_6d_leading_batch(self):
        """[1, 2, 4, 1, 8, 16] .squeeze(0) → [2, 4, 1, 8, 16]"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 1, 8, 16), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))

    def test_squeeze_6d_unaligned_inner(self):
        """[1, 2, 4, 1, 8, 13] .squeeze(0) → [2, 4, 1, 8, 13]  (13 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 1, 8, 13), dtype=dt)
            compare_with_cpu(lambda t: t.squeeze(0), x, run_eager=True, atol=_atol(dt))


# ---------------------------------------------------------------------------
# Section 4 — Operation-Level: unsqueeze
# ---------------------------------------------------------------------------

class TestUnsqueeze(unittest.TestCase):
    """
    Standalone unsqueeze tests: 0-D to 5-D inputs (result up to 6-D),
    all dtypes, aligned and unaligned stick scenarios.
    """

    # ---- 0-D input ----

    def test_unsqueeze_scalar_to_1d(self):
        """[] → [1]  via unsqueeze(0)"""
        for dt in DTYPES:
            x = _randn((), dt)
            compare_with_cpu(lambda t: t.unsqueeze(0), x, run_eager=True, atol=_atol(dt))

    # ---- 1-D input ----

    def test_unsqueeze_1d_aligned_to_2d_front(self):
        """[256] → [1, 256]  (leading dim; inner 256 aligned)"""
        for dt in DTYPES:
            x = cached_randn((256,), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(0), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_1d_unaligned_to_2d_front(self):
        """[70] → [1, 70]  (70 % 64 != 0, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((70,), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(0), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_1d_size1_to_2d(self):
        """[1] → [1, 1]  (size-1 input + size-1 result)"""
        for dt in DTYPES:
            x = _randn((1,), dt)
            compare_with_cpu(lambda t: t.unsqueeze(0), x, run_eager=True, atol=_atol(dt))

    # ---- 2-D input ----

    def test_unsqueeze_2d_aligned_to_3d_dim0(self):
        """[32, 2816] → [1, 32, 2816]  (batch dim insert; 2816 aligned)"""
        for dt in DTYPES:
            x = cached_randn((32, 2816), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(0), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_2d_unaligned_to_3d_dim0(self):
        """[32, 13] → [1, 32, 13]  (batch dim insert; 13 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((32, 13), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(0), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_2d_to_3d_trailing(self):
        """[32, 256] → [32, 256, 1]  (trailing size-1 becomes new unaligned inner)"""
        for dt in DTYPES:
            x = cached_randn((32, 256), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(-1), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_2d_attn_mask_double(self):
        """[1, 32] → [1, 1, 1, 32] via two unsqueezes  (attention mask broadcast)"""
        for dt in DTYPES:
            x = cached_randn((1, 32), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(1).unsqueeze(2),
                             x, run_eager=True, atol=_atol(dt))

    # ---- 3-D input ----

    def test_unsqueeze_3d_rope_freq_dim2_aligned(self):
        """[1, 32, 256] → [1, 32, 1, 256]  (RoPE head broadcast; 256 aligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 32, 256), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(2), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_3d_rope_freq_dim2_unaligned(self):
        """[1, 32, 70] → [1, 32, 1, 70]  (70 is unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 32, 70), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(2), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_3d_size1_inner(self):
        """[4, 8, 1] → [4, 1, 8, 1]  (middle insert; size-1 trailing stays)"""
        for dt in DTYPES:
            x = cached_randn((4, 8, 1), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(1), x, run_eager=True, atol=_atol(dt))

    # ---- 4-D input ----

    def test_unsqueeze_4d_gqa_head_dim2_aligned(self):
        """[2, 8, 32, 256] → [2, 8, 1, 32, 256]  (GQA head group dim; 256 aligned)"""
        for dt in DTYPES:
            x = cached_randn((2, 8, 32, 256), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(2), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_4d_gqa_head_dim2_unaligned(self):
        """[2, 8, 32, 72] → [2, 8, 1, 32, 72]  (D=72 unaligned vision heads)"""
        for dt in DTYPES:
            x = cached_randn((2, 8, 32, 72), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(2), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_4d_negative_index(self):
        """[1, 16, 32, 256] → [1, 16, 32, 256, 1]  (unsqueeze(-1))"""
        for dt in DTYPES:
            x = cached_randn((1, 16, 32, 256), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(-1), x, run_eager=True, atol=_atol(dt))

    # ---- 5-D input → 6-D ----

    def test_unsqueeze_5d_to_6d_aligned(self):
        """[1, 2, 4, 8, 16] → [1, 2, 4, 1, 8, 16]  (256 aligned if full; 16 < 64 unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 8, 16), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(3), x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_5d_to_6d_unaligned(self):
        """[1, 2, 4, 8, 13] → [1, 2, 4, 1, 8, 13]  (13 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 4, 8, 13), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(3), x, run_eager=True, atol=_atol(dt))

    # ---- Multiple unsqueeze in one call (Gemma-specific patterns) ----

    def test_unsqueeze_dual_axis_aligned(self):
        """[32, 256] → [1, 32, 256, 1]  via unsqueeze(0).unsqueeze(3)"""
        for dt in DTYPES:
            x = cached_randn((32, 256), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(3),
                             x, run_eager=True, atol=_atol(dt))

    def test_unsqueeze_dual_axis_unaligned(self):
        """[32, 70] → [1, 32, 70, 1]  via unsqueeze(0).unsqueeze(3)  (70 unaligned, 1 new inner)"""
        for dt in DTYPES:
            x = cached_randn((32, 70), dtype=dt)
            compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(3),
                             x, run_eager=True, atol=_atol(dt))


# ---------------------------------------------------------------------------
# Section 5 — Operation-Level: expand
# ---------------------------------------------------------------------------

class TestExpand(unittest.TestCase):
    """
    Standalone expand tests: 0-D to 5-D inputs (result up to 5-D),
    all dtypes, aligned and unaligned stick scenarios.
    """

    # ---- 0-D input ----

    def test_expand_scalar_to_4d_aligned(self):
        """[] → [1, 16, 32, 256]  (aligned inner 256)"""
        for dt in DTYPES:
            x = _randn((), dt)
            compare_with_cpu(lambda t: t.expand(1, 16, 32, 256),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_scalar_to_4d_unaligned(self):
        """[] → [1, 16, 32, 70]  (70 is unaligned)"""
        for dt in DTYPES:
            x = _randn((), dt)
            compare_with_cpu(lambda t: t.expand(1, 16, 32, 70),
                             x, run_eager=True, atol=_atol(dt))

    # ---- 1-D input ----

    def test_expand_1d_bias_to_3d_aligned(self):
        """[2816] → [2, 32, 2816]  (1D bias broadcast; 2816 aligned)"""
        for dt in DTYPES:
            x = cached_randn((2816,), dtype=dt)
            compare_with_cpu(lambda t: t.expand(2, 32, 2816),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_1d_bias_to_3d_unaligned(self):
        """[70] → [2, 32, 70]  (1D bias; 70 unaligned)"""
        for dt in DTYPES:
            x = cached_randn((70,), dtype=dt)
            compare_with_cpu(lambda t: t.expand(2, 32, 70),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_1d_size1_to_3d(self):
        """[1] → [2, 32, 1]  (expand from size-1 vector; trailing 1 is unaligned)"""
        for dt in DTYPES:
            x = _randn((1,), dt)
            compare_with_cpu(lambda t: t.expand(2, 32, 1),
                             x, run_eager=True, atol=_atol(dt))

    # ---- 2-D input ----

    def test_expand_2d_batch_aligned(self):
        """[1, 2816] → [4, 2816]  (batch broadcast; 2816 aligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 2816), dtype=dt)
            compare_with_cpu(lambda t: t.expand(4, 2816),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_2d_batch_unaligned(self):
        """[1, 33] → [4, 33]  (33 is unaligned: < 64)"""
        for dt in DTYPES:
            x = cached_randn((1, 33), dtype=dt)
            compare_with_cpu(lambda t: t.expand(4, 33),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_2d_size1_inner(self):
        """[32, 1] → [32, 2816]  (trailing size-1 → aligned)"""
        for dt in DTYPES:
            x = cached_randn((32, 1), dtype=dt)
            compare_with_cpu(lambda t: t.expand(32, 2816),
                             x, run_eager=True, atol=_atol(dt))

    # ---- 3-D input ----

    def test_expand_3d_vision_cls_token_aligned(self):
        """[1, 1, 1152] → [2, 1, 1152]  (CLS token; 1152 aligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 1152), dtype=dt)
            compare_with_cpu(lambda t: t.expand(2, 1, 1152),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_3d_vision_cls_token_unaligned(self):
        """[1, 1, 70] → [2, 1, 70]  (70 unaligned; size-1 seq dim kept)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 70), dtype=dt)
            compare_with_cpu(lambda t: t.expand(2, 1, 70),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_3d_minus_one_dims(self):
        """[2, 1, 256] → [2, 16, 256]  (expand only head dim; -1 keeps others)"""
        for dt in DTYPES:
            x = cached_randn((2, 1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.expand(-1, 16, -1),
                             x, run_eager=True, atol=_atol(dt))

    # ---- 4-D input ----

    def test_expand_4d_gqa_kv_sliding_aligned(self):
        """[1, 8, 1, 32, 256] used after unsqueeze; expand heads 2x  (Config A)
        Here simplified: [1, 8, 1, 256] → [1, 8, 2, 256]"""
        for dt in DTYPES:
            x = cached_randn((1, 8, 1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.expand(1, 8, 2, 256),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_4d_gqa_kv_full_unaligned(self):
        """[1, 2, 1, 72] → [1, 2, 8, 72]  (D=72 unaligned, 8x GQA)"""
        for dt in DTYPES:
            x = cached_randn((1, 2, 1, 72), dtype=dt)
            compare_with_cpu(lambda t: t.expand(1, 2, 8, 72),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_4d_attn_mask_batch_heads_aligned(self):
        """[1, 1, 32, 32] → [4, 16, 32, 32]  (mask broadcast; 32 < 64, unaligned inner)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 32, 32), dtype=dt)
            compare_with_cpu(lambda t: t.expand(4, 16, 32, 32),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_4d_attn_mask_unaligned_seq(self):
        """[1, 1, 33, 33] → [2, 8, 33, 33]  (33 unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 33, 33), dtype=dt)
            compare_with_cpu(lambda t: t.expand(2, 8, 33, 33),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_4d_rope_freqs_heads_aligned(self):
        """[1, 32, 1, 256] → [1, 32, 16, 256]  (RoPE freq broadcast; 256 aligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 32, 1, 256), dtype=dt)
            compare_with_cpu(lambda t: t.expand(1, 32, 16, 256),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_4d_size1_trailing_to_aligned(self):
        """[32, 8, 4, 1] → [32, 8, 4, 2816]  (trailing 1 → aligned)"""
        for dt in DTYPES:
            x = cached_randn((32, 8, 4, 1), dtype=dt)
            compare_with_cpu(lambda t: t.expand(32, 8, 4, 2816),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_4d_odd_batch_heads(self):
        """[1, 1, 33, 256] → [3, 7, 33, 256]  (B=3, H=7 non-power-of-2; S=33 unaligned)"""
        for dt in DTYPES:
            x = cached_randn((1, 1, 33, 256), dtype=dt)
            compare_with_cpu(lambda t: t.expand(3, 7, 33, 256),
                             x, run_eager=True, atol=_atol(dt))

    # ---- 5-D input ----

    def test_expand_5d_block_cache_heads_aligned(self):
        """[16, 1, 4, 16, 256] → [16, 8, 4, 16, 256]  (paged cache; 256 aligned)"""
        for dt in DTYPES:
            x = cached_randn((16, 1, 4, 16, 256), dtype=dt)
            compare_with_cpu(lambda t: t.expand(16, 8, 4, 16, 256),
                             x, run_eager=True, atol=_atol(dt))

    def test_expand_5d_block_cache_unaligned(self):
        """[16, 1, 4, 16, 13] → [16, 8, 4, 16, 13]  (13 < 64, unaligned)"""
        for dt in DTYPES:
            x = cached_randn((16, 1, 4, 16, 13), dtype=dt)
            compare_with_cpu(lambda t: t.expand(16, 8, 4, 16, 13),
                             x, run_eager=True, atol=_atol(dt))


# ---------------------------------------------------------------------------
# Section 6 — Composite: Chained view ops (module-like patterns)
# ---------------------------------------------------------------------------

class TestCompositeViewChains(unittest.TestCase):
    """
    Multi-op chains that mirror compound view sequences found in Gemma modules.
    Tests are dtype-parameterized and cover both aligned and unaligned cases.
    """

    def test_chain_unsqueeze_expand_reshape_gqa_aligned(self):
        """
        GQA KV broadcast chain (Config A, sliding):
          [B, KVH, S, D] → unsqueeze(2) → expand → reshape → [B, H, S, D]
        D=256 aligned.
        """
        for dt in DTYPES:
            B, KVH, H, S, D = 1, 8, 16, 32, 256
            exp = H // KVH  # 2
            x = cached_randn((B, KVH, S, D), dtype=dt)
            compare_with_cpu(
                lambda t, b=B, kvh=KVH, h=H, s=S, d=D, e=exp:
                    t.unsqueeze(2).expand(b, kvh, e, s, d).reshape(b, h, s, d),
                x, run_eager=True, atol=_atol(dt),
            )

    def test_chain_unsqueeze_expand_reshape_gqa_unaligned(self):
        """
        GQA KV broadcast chain with unaligned D=72 (vision encoder):
          [B, KVH, S, D] → unsqueeze(2) → expand → reshape → [B, H, S, D]
        D=72 unaligned.
        """
        for dt in DTYPES:
            B, KVH, H, S, D = 1, 8, 16, 32, 72
            exp = H // KVH  # 2
            x = cached_randn((B, KVH, S, D), dtype=dt)
            compare_with_cpu(
                lambda t, b=B, kvh=KVH, h=H, s=S, d=D, e=exp:
                    t.unsqueeze(2).expand(b, kvh, e, s, d).reshape(b, h, s, d),
                x, run_eager=True, atol=_atol(dt),
            )

    def test_chain_reshape_unsqueeze_expand_aligned(self):
        """
        Attention output re-assembly with extra broadcast:
          [B, S, H*D] → reshape [B, S, H, D] → unsqueeze(0) → expand [B, S, H, D]
        (Synthetic broadcast test to check the chain; D=256 aligned.)
        """
        for dt in DTYPES:
            B, S, H, D = 2, 32, 8, 256
            x = cached_randn((B, S, H * D), dtype=dt)
            compare_with_cpu(
                lambda t, b=B, s=S, h=H, d=D:
                    t.reshape(b, s, h, d).unsqueeze(0).expand(1, b, s, h, d),
                x, run_eager=True, atol=_atol(dt),
            )

    def test_chain_squeeze_expand_unaligned(self):
        """
        Squeeze singleton batch dim then broadcast:
          [1, 32, 70] → squeeze(0) → [32, 70] → unsqueeze(0) → expand [3, 32, 70]
        70 is unaligned.
        """
        for dt in DTYPES:
            x = cached_randn((1, 32, 70), dtype=dt)
            compare_with_cpu(
                lambda t: t.squeeze(0).unsqueeze(0).expand(3, 32, 70),
                x, run_eager=True, atol=_atol(dt),
            )

    def test_chain_view_split_and_head_transpose_aligned(self):
        """
        QKV split + head transpose:
          [B, S, H*D] → view [B, S, H, D] → transpose(1,2) → [B, H, S, D]
        D=256 aligned.
        """
        for dt in DTYPES:
            B, S, H, D = 1, 32, 16, 256
            x = cached_randn((B, S, H * D), dtype=dt)
            compare_with_cpu(
                lambda t, b=B, s=S, h=H, d=D:
                    t.view(b, s, h, d).transpose(1, 2),
                x, run_eager=True, atol=_atol(dt),
            )

    def test_chain_view_split_and_head_transpose_unaligned(self):
        """
        QKV split + head transpose with D=72 (vision encoder):
          [1, 280, 1152] → view [1, 280, 16, 72] → transpose(1,2) → [1, 16, 280, 72]
        D=72 unaligned.
        """
        for dt in DTYPES:
            x = cached_randn((1, 280, 1152), dtype=dt)
            compare_with_cpu(
                lambda t: t.view(1, 280, 16, 72).transpose(1, 2),
                x, run_eager=True, atol=_atol(dt),
            )

    def test_chain_reshape_infer_then_expand_aligned(self):
        """
        MoE flatten + expand to broadcast routing weights:
          [4, 32, 2816] → reshape(-1, 2816) → unsqueeze(-1) → expand [128, 2816, 8]
        D=2816 aligned.
        """
        for dt in DTYPES:
            B, S, D, K = 4, 32, 2816, 8
            x = cached_randn((B, S, D), dtype=dt)
            compare_with_cpu(
                lambda t, d=D, k=K: t.reshape(-1, d).unsqueeze(-1).expand(-1, d, k),
                x, run_eager=True, atol=_atol(dt),
            )

    def test_chain_squeeze_reshape_expand_size1_inner(self):
        """
        Squeeze + reshape + expand with size-1 trailing (ultra-unaligned inner):
          [1, 32, 1] → squeeze(0) → [32, 1] → expand [32, 2816]
        Inner dim goes from 1 → 2816 (aligned destination).
        """
        for dt in DTYPES:
            x = cached_randn((1, 32, 1), dtype=dt)
            compare_with_cpu(
                lambda t: t.squeeze(0).expand(32, 2816),
                x, run_eager=True, atol=_atol(dt),
            )


if __name__ == "__main__":
    unittest.main()
