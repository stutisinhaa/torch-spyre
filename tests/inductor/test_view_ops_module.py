# Copyright 2026 The Torch-Spyre Authors.
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
FVT Section 1 — Module-Level View-Op Tests
==========================================

Each test reproduces the exact ATen op chain from a named Gemma-4 sub-module,
grounded in three source HTML traces:
  - gemma4_26b_module_operator_report_v2.html  (CPU, seq=32)
  - gemma4_fx_module_trace_v2.html             (Spyre compile, seq=64)
  - gemma4-view-ops-aten.html                  (CPU ATen layer)

All tests use bfloat16 unless noted (int64 for index tensors).
Pass criteria: compare_with_cpu(fn, *inputs, run_eager=True), atol=1e-2.

MOD-01–21:  reshape / unsqueeze / expand  (original)
MOD-22–26:  transpose
MOD-27–30:  t (2D weight transpose)
MOD-31–34:  permute (Spyre compile trace)
MOD-35–38:  clone
MOD-39–42:  cat (concat)
MOD-43–44:  select
MOD-45–49:  narrow / slice
MOD-50–51:  split
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)


_ATOL_F16 = 1e-2
_ATOL_BF16 = 2e-2
_ATOL_F32 = 1e-5
_RTOL = 1e-2


def _get_tolerances(dtype):
    if dtype == torch.float16:
        return _ATOL_F16, _RTOL
    elif dtype == torch.bfloat16:
        return _ATOL_BF16, _RTOL
    elif dtype == torch.float32:
        return _ATOL_F32, _RTOL
    return _ATOL_F16, _RTOL


def _compare_modes(execution_mode, fn, *args, atol=None, rtol=None):
    if atol is None or rtol is None:
        dtype = next(
            (a.dtype for a in args if isinstance(a, torch.Tensor)), torch.bfloat16
        )
        def_atol, def_rtol = _get_tolerances(dtype)
        atol = def_atol if atol is None else atol
        rtol = def_rtol if rtol is None else rtol

    compare_with_cpu(
        fn,
        *args,
        atol=atol,
        rtol=rtol,
        run_compile=(execution_mode == "compiled"),
        run_eager=(execution_mode == "eager"),
    )


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestModuleLevel:
    """
    Section 1 — Module-level view-op tests grounded in Gemma-4 26B-A4B-it traces.
    All inputs are bfloat16 unless noted (int64 for index/mask tensors).
    Pass criteria: compare_with_cpu, atol=1e-2.
    """

    # ------------------------------------------------------------------
    # Reshape / view — Decoder token flatten / unflatten
    # ------------------------------------------------------------------

    def test_mod_decoder_token_flatten(self, execution_mode):
        """MOD-01: Gemma4TextDecoderLayer — view(-1,2816): [1,32,2816]→[32,2816]."""
        x = cached_randn((1, 32, 2816), dtype=torch.bfloat16,
                         differentiation="mod01_decoder_flatten")
        _compare_modes(execution_mode, lambda t: t.view(-1, 2816), x)

    def test_mod_decoder_token_unflatten(self, execution_mode):
        """MOD-02: Gemma4TextDecoderLayer — view(1,32,2816): [32,2816]→[1,32,2816]."""
        x = cached_randn((32, 2816), dtype=torch.bfloat16,
                         differentiation="mod02_decoder_unflatten")
        _compare_modes(execution_mode, lambda t: t.view(1, 32, 2816), x)

    # ------------------------------------------------------------------
    # Reshape / view — Attention QKV split
    # ------------------------------------------------------------------

    def test_mod_attn_qsplit_sliding(self, execution_mode):
        """MOD-03: Gemma4TextAttention — view(1,32,-1,256): [1,32,4096]→[1,32,16,256]."""
        x = cached_randn((1, 32, 4096), dtype=torch.bfloat16,
                         differentiation="mod03_qsplit_sliding")
        _compare_modes(execution_mode, lambda t: t.view(1, 32, -1, 256), x)

    def test_mod_attn_kvsplit_sliding(self, execution_mode):
        """MOD-04: Gemma4TextAttention — view(1,32,-1,256): [1,32,2048]→[1,32,8,256]."""
        x = cached_randn((1, 32, 2048), dtype=torch.bfloat16,
                         differentiation="mod04_kvsplit_sliding")
        _compare_modes(execution_mode, lambda t: t.view(1, 32, -1, 256), x)

    def test_mod_attn_qsplit_spyre_s64(self, execution_mode):
        """MOD-05: Gemma4Attention (Spyre seq=64) — view(1,64,-1,256): [1,64,4096]→[1,64,16,256]."""
        x = cached_randn((1, 64, 4096), dtype=torch.bfloat16,
                         differentiation="mod05_qsplit_spyre_s64")
        _compare_modes(execution_mode, lambda t: t.view(1, 64, -1, 256), x)

    def test_mod_attn_head_merge(self, execution_mode):
        """MOD-06: Gemma4TextAttention — reshape(1,32,-1): [1,32,16,256]→[1,32,4096]."""
        x = cached_randn((1, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod06_head_merge")
        _compare_modes(execution_mode, lambda t: t.reshape(1, 32, -1), x)

    # ------------------------------------------------------------------
    # Reshape / view — BMM preparation (expand→view)
    # ------------------------------------------------------------------

    def test_mod_attn_bmm_prep_expand_view(self, execution_mode):
        """MOD-07: Gemma4TextAttention — expand(1,16,32,256)→view([16,32,256])."""
        x = cached_randn((1, 16, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod07_bmm_prep")
        _compare_modes(
            execution_mode,
            lambda t: t.expand(1, 16, 32, 256).view(16, 32, 256),
            x,
        )

    # ------------------------------------------------------------------
    # Reshape / view — RoPE view chain (4D→5D→6D→4D)
    # ------------------------------------------------------------------

    def test_mod_rope_view_chain(self, execution_mode):
        """MOD-08: Gemma4TextRotaryEmbedding — view(5D)→unsqueeze(-3)→mul→view(4D)."""
        x = cached_randn((1, 64, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod08_rope_x")
        freq = cached_randn((1, 64, 1, 2, 128), dtype=torch.bfloat16,
                            differentiation="mod08_rope_freq")

        def rope_chain(t, f):
            return t.view(1, 64, 16, 2, 128).unsqueeze(-3).mul(f).view(1, 64, 16, 256)

        _compare_modes(execution_mode, rope_chain, x, freq)

    # ------------------------------------------------------------------
    # Reshape / view — MoE index flatten / output unflatten
    # ------------------------------------------------------------------

    def test_mod_moe_index_flatten(self, execution_mode):
        """MOD-09: Gemma4TextMoE — view(-1): [32,8]→[256] (int64)."""
        x = torch.arange(256, dtype=torch.int64).view(32, 8)
        _compare_modes(execution_mode, lambda t: t.view(-1), x)

    def test_mod_moe_output_unflatten(self, execution_mode):
        """MOD-10: Gemma4TextMoE — view(32,8,2816): [256,2816]→[32,8,2816]."""
        x = cached_randn((256, 2816), dtype=torch.bfloat16,
                         differentiation="mod10_moe_unflatten")
        _compare_modes(execution_mode, lambda t: t.view(32, 8, 2816), x)

    # ------------------------------------------------------------------
    # Unsqueeze — Mask chain and RoPE
    # ------------------------------------------------------------------

    def test_mod_mask_unsqueeze_chain(self, execution_mode):
        """MOD-11: Gemma4Model — unsqueeze(0)→unsqueeze(1)→unsqueeze(3): [32]→[1,1,32,1]."""
        x = torch.arange(32, dtype=torch.int64)
        _compare_modes(
            execution_mode,
            lambda t: t.unsqueeze(0).unsqueeze(1).unsqueeze(3),
            x,
        )

    def test_mod_rope_unsqueeze_freq(self, execution_mode):
        """MOD-12: Gemma4TextAttention — unsqueeze(2): [1,32,256]→[1,32,1,256]."""
        x = cached_randn((1, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod12_rope_unsqueeze")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(2), x)

    def test_mod_gqa_unsqueeze_sliding(self, execution_mode):
        """MOD-13: Gemma4TextAttention — unsqueeze(2): [1,8,32,256]→[1,8,1,32,256]."""
        x = cached_randn((1, 8, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod13_gqa_unsqueeze_sliding")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(2), x)

    def test_mod_moe_routing_unsqueeze(self, execution_mode):
        """MOD-14: Gemma4TextMoE — unsqueeze(-1): [256]→[256,1]."""
        x = cached_randn((256,), dtype=torch.bfloat16,
                         differentiation="mod14_moe_routing_unsqueeze")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(-1), x)

    def test_mod_linear_weight_unsqueeze(self, execution_mode):
        """MOD-15: Linear (Spyre lowering) — unsqueeze(0): [4096,2816]→[1,4096,2816]."""
        x = cached_randn((4096, 2816), dtype=torch.bfloat16,
                         differentiation="mod15_linear_weight_unsqueeze")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(0), x)

    def test_mod_rope_unsqueeze_6d(self, execution_mode):
        """MOD-16: Gemma4TextRotaryEmbedding — view(5D)→unsqueeze(-3): [1,64,16,2,128]→[1,64,16,1,2,128]."""
        x = cached_randn((1, 64, 16, 2, 128), dtype=torch.bfloat16,
                         differentiation="mod16_rope_unsqueeze_6d")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(-3), x)

    # ------------------------------------------------------------------
    # Expand — Mask same-shape and GQA materialise
    # ------------------------------------------------------------------

    def test_mod_mask_expand(self, execution_mode):
        """MOD-17: Gemma4Model — expand(1,-1,32,32) same-shape: [1,1,32,32]→[1,1,32,32]."""
        x = cached_randn((1, 1, 32, 32), dtype=torch.bfloat16,
                         differentiation="mod17_mask_expand")
        _compare_modes(execution_mode, lambda t: t.expand(1, -1, 32, 32), x)

    def test_mod_gqa_expand_clone_sliding(self, execution_mode):
        """MOD-18: Gemma4TextAttention — unsqueeze(2)→expand(1,8,2,32,256)→clone: GQA 2×."""
        x = cached_randn((1, 8, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod18_gqa_expand_clone_sliding")
        _compare_modes(
            execution_mode,
            lambda t: t.unsqueeze(2).expand(1, 8, 2, 32, 256).clone(),
            x,
        )

    def test_mod_gqa_expand_clone_full_attn(self, execution_mode):
        """MOD-19: Gemma4TextAttention — unsqueeze(2)→expand(1,2,8,32,512)→clone: GQA 8× full-attn."""
        x = cached_randn((1, 2, 32, 512), dtype=torch.bfloat16,
                         differentiation="mod19_gqa_expand_clone_full_attn")
        _compare_modes(
            execution_mode,
            lambda t: t.unsqueeze(2).expand(1, 2, 8, 32, 512).clone(),
            x,
        )

    def test_mod_linear_unsqueeze_expand_bmm(self, execution_mode):
        """MOD-20: Linear (Spyre) — w.unsqueeze(0).expand(1,2816,4096) then bmm(x, w_e)."""
        w = cached_randn((2816, 4096), dtype=torch.bfloat16,
                         differentiation="mod20_w")
        x = cached_randn((1, 64, 2816), dtype=torch.bfloat16,
                         differentiation="mod20_x")

        def linear_bmm(weight, inp):
            return torch.bmm(inp, weight.unsqueeze(0).expand(1, 2816, 4096))

        _compare_modes(execution_mode, linear_bmm, w, x, atol=0.5, rtol=0.1)

    def test_mod_moe_token_unsqueeze_expand(self, execution_mode):
        """MOD-21: Gemma4TextMoE — unsqueeze(1)→expand(1,8,2816): [1,2816]→[1,8,2816]."""
        x = cached_randn((1, 2816), dtype=torch.bfloat16,
                         differentiation="mod21_moe_token_dispatch")
        _compare_modes(
            execution_mode,
            lambda t: t.unsqueeze(1).expand(1, 8, 2816),
            x,
        )

    # ------------------------------------------------------------------
    # Transpose — Attention head-axis swap and key transpose
    # ------------------------------------------------------------------

    def test_mod_transpose_attn_head_swap_sliding(self, execution_mode):
        """MOD-22: Gemma4TextAttention sliding — transpose(1,2) after RoPE: [1,32,16,256]→[1,16,32,256]."""
        x = cached_randn((1, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod22_trp_head_swap_sliding")
        _compare_modes(execution_mode, lambda t: t.transpose(1, 2), x)

    def test_mod_transpose_attn_key_for_matmul(self, execution_mode):
        """MOD-23: Gemma4TextAttention — transpose(2,3) for K in QK matmul: [1,16,32,256]→[1,16,256,32]."""
        x = cached_randn((1, 16, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod23_trp_key_matmul")
        _compare_modes(execution_mode, lambda t: t.transpose(2, 3), x)

    def test_mod_transpose_attn_head_swap_full_attn(self, execution_mode):
        """MOD-24: Gemma4TextAttention full-attn — transpose(1,2): [1,32,16,512]→[1,16,32,512]."""
        x = cached_randn((1, 32, 16, 512), dtype=torch.bfloat16,
                         differentiation="mod24_trp_head_swap_full_attn")
        _compare_modes(execution_mode, lambda t: t.transpose(1, 2), x)

    def test_mod_transpose_moe_expert_gemm(self, execution_mode):
        """MOD-25: Gemma4TextMoE expert GEMM — transpose(-2,-1): [128,1408,2816]→[128,2816,1408]."""
        x = cached_randn((128, 1408, 2816), dtype=torch.bfloat16,
                         differentiation="mod25_trp_moe_expert")
        _compare_modes(execution_mode, lambda t: t.transpose(-2, -1), x)

    def test_mod_transpose_qk_attention_chain(self, execution_mode):
        """MOD-26: Gemma4TextAttention — q @ k.transpose(2,3): full QK attention score chain."""
        q = cached_randn((1, 16, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod26_q")
        k = cached_randn((1, 16, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod26_k")
        _compare_modes(
            execution_mode,
            lambda q, k: torch.matmul(q, k.transpose(2, 3)),
            q, k,
            atol=0.5, rtol=0.1,
        )

    # ------------------------------------------------------------------
    # t — 2D weight transpose before mm
    # ------------------------------------------------------------------

    def test_mod_t_linear_weight_q(self, execution_mode):
        """MOD-27: Gemma4TextAttention q_proj — t(): [4096,2816]→[2816,4096]."""
        x = cached_randn((4096, 2816), dtype=torch.bfloat16,
                         differentiation="mod27_t_weight_q")
        _compare_modes(execution_mode, lambda t: t.t(), x)

    def test_mod_t_linear_weight_kv(self, execution_mode):
        """MOD-28: Gemma4TextAttention k/v_proj — t(): [2048,2816]→[2816,2048]."""
        x = cached_randn((2048, 2816), dtype=torch.bfloat16,
                         differentiation="mod28_t_weight_kv")
        _compare_modes(execution_mode, lambda t: t.t(), x)

    def test_mod_t_linear_weight_mlp(self, execution_mode):
        """MOD-29: Gemma4TextMLP gate/up_proj — t(): [2112,2816]→[2816,2112]."""
        x = cached_randn((2112, 2816), dtype=torch.bfloat16,
                         differentiation="mod29_t_weight_mlp")
        _compare_modes(execution_mode, lambda t: t.t(), x)

    def test_mod_t_moe_output_weight(self, execution_mode):
        """MOD-30: Gemma4TextMoE output proj — t(): [128,2816]→[2816,128]."""
        x = cached_randn((128, 2816), dtype=torch.bfloat16,
                         differentiation="mod30_t_moe_output")
        _compare_modes(execution_mode, lambda t: t.t(), x)

    # ------------------------------------------------------------------
    # Permute — Spyre compile trace (t decomposes to permute)
    # ------------------------------------------------------------------

    def test_mod_permute_weight_transpose_q(self, execution_mode):
        """MOD-31: Spyre Linear q_proj — permute(1,0): [4096,2816]→[2816,4096]."""
        x = cached_randn((4096, 2816), dtype=torch.bfloat16,
                         differentiation="mod31_prm_weight_q")
        _compare_modes(execution_mode, lambda t: t.permute(1, 0), x)

    def test_mod_permute_head_swap_sliding(self, execution_mode):
        """MOD-32: Spyre Gemma4Attention — permute(0,2,1,3): [1,64,8,256]→[1,8,64,256]."""
        x = cached_randn((1, 64, 8, 256), dtype=torch.bfloat16,
                         differentiation="mod32_prm_head_swap_sliding")
        _compare_modes(execution_mode, lambda t: t.permute(0, 2, 1, 3), x)

    def test_mod_permute_head_swap_q_s64(self, execution_mode):
        """MOD-33: Spyre Gemma4Attention — permute(0,2,1,3): [1,64,16,256]→[1,16,64,256]."""
        x = cached_randn((1, 64, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod33_prm_head_swap_q_s64")
        _compare_modes(execution_mode, lambda t: t.permute(0, 2, 1, 3), x)

    def test_mod_permute_kv_cache_layout(self, execution_mode):
        """MOD-34: Spyre KV-cache — permute(2,0,1,3): [1,8,512,256]→[512,1,8,256]."""
        x = cached_randn((1, 8, 512, 256), dtype=torch.bfloat16,
                         differentiation="mod34_prm_kv_cache")
        _compare_modes(execution_mode, lambda t: t.permute(2, 0, 1, 3), x)

    # ------------------------------------------------------------------
    # Clone — standalone materialise and post-transpose restickify
    # ------------------------------------------------------------------

    def test_mod_clone_attn_output_sliding(self, execution_mode):
        """MOD-35: Gemma4TextAttention — clone(): [1,32,16,256] attention output buffer."""
        x = cached_randn((1, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod35_cln_attn_out_sliding")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_mod_clone_attn_output_full_attn(self, execution_mode):
        """MOD-36: Gemma4TextAttention full-attn — clone(): [1,32,16,512]."""
        x = cached_randn((1, 32, 16, 512), dtype=torch.bfloat16,
                         differentiation="mod36_cln_attn_out_full_attn")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_mod_clone_after_transpose_sliding(self, execution_mode):
        """MOD-37: Gemma4TextAttention — transpose(1,2).clone(): [1,16,32,256]→[1,32,16,256]. Restickify path."""
        x = cached_randn((1, 16, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod37_cln_after_trp_sliding")
        _compare_modes(execution_mode, lambda t: t.transpose(1, 2).clone(), x)

    def test_mod_clone_after_transpose_full_attn(self, execution_mode):
        """MOD-38: Gemma4TextAttention full-attn — transpose(1,2).clone(): [1,16,32,512]→[1,32,16,512]."""
        x = cached_randn((1, 16, 32, 512), dtype=torch.bfloat16,
                         differentiation="mod38_cln_after_trp_full_attn")
        _compare_modes(execution_mode, lambda t: t.transpose(1, 2).clone(), x)

    # ------------------------------------------------------------------
    # Cat — RoPE half-rotation concat and KV-cache append
    # ------------------------------------------------------------------

    def test_mod_cat_rope_halves_sliding(self, execution_mode):
        """MOD-39: Gemma4TextAttention — narrow→narrow→cat(dim=-1): [1,32,16,256]→halves→[1,32,16,256]."""
        x = cached_randn((1, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod39_cat_rope_sliding")

        def rope_cat(t):
            a = t.narrow(3, 0, 128)
            b = t.narrow(3, 128, 128)
            return torch.cat([a, b], dim=-1)

        _compare_modes(execution_mode, rope_cat, x)

    def test_mod_cat_rope_halves_full_attn(self, execution_mode):
        """MOD-40: Gemma4TextAttention full-attn — narrow→narrow→cat(dim=-1): [1,32,16,512]→halves→[1,32,16,512]."""
        x = cached_randn((1, 32, 16, 512), dtype=torch.bfloat16,
                         differentiation="mod40_cat_rope_full_attn")

        def rope_cat_full(t):
            a = t.narrow(3, 0, 256)
            b = t.narrow(3, 256, 256)
            return torch.cat([a, b], dim=-1)

        _compare_modes(execution_mode, rope_cat_full, x)

    def test_mod_cat_kv_cache_append(self, execution_mode):
        """MOD-41: Gemma4TextAttention — cat([past_kv, new_kv], dim=-2): [1,8,32,256]+[1,8,32,256]→[1,8,64,256]."""
        past = cached_randn((1, 8, 32, 256), dtype=torch.bfloat16,
                            differentiation="mod41_cat_kv_past")
        new = cached_randn((1, 8, 32, 256), dtype=torch.bfloat16,
                           differentiation="mod41_cat_kv_new")
        _compare_modes(
            execution_mode,
            lambda a, b: torch.cat([a, b], dim=-2),
            past, new,
        )

    def test_mod_cat_pos_id_concat(self, execution_mode):
        """MOD-42: Gemma4Model — cat([[1,1],[1,32]], dim=-1)→[1,33] int64. Position-id concat."""
        pytest.xfail(
            "known failure for both eager and compiled; "
            "tracked in https://github.com/torch-spyre/torch-spyre/issues/3916"
        )
        a = torch.zeros(1, 1, dtype=torch.int64)
        b = torch.arange(32, dtype=torch.int64).unsqueeze(0)
        _compare_modes(execution_mode, lambda a, b: torch.cat([a, b], dim=-1), a, b)

    # ------------------------------------------------------------------
    # Select — last-token position id and routing index
    # ------------------------------------------------------------------

    def test_mod_select_pos_id_last_token(self, execution_mode):
        """MOD-43: Gemma4Model — select(1,-1): [1,32] int64 → [1] (last token position)."""
        x = torch.arange(32, dtype=torch.int64).unsqueeze(0)
        _compare_modes(execution_mode, lambda t: t.select(1, -1), x)

    def test_mod_select_routing_index(self, execution_mode):
        """MOD-44: Gemma4TextMoE — select(2,0): [1,8,32] int32 → [1,8] (routing index)."""
        x = torch.zeros(1, 8, 32, dtype=torch.int32)
        _compare_modes(execution_mode, lambda t: t.select(2, 0), x)

    # ------------------------------------------------------------------
    # Narrow / slice — RoPE half-split and KV-cache sliding window
    # ------------------------------------------------------------------

    def test_mod_narrow_rope_first_half_sliding(self, execution_mode):
        """MOD-45: Gemma4TextAttention — narrow(3,0,128): [1,32,16,256]→[1,32,16,128]. RoPE first half."""
        x = cached_randn((1, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod45_narrow_rope_first")
        _compare_modes(execution_mode, lambda t: t.narrow(3, 0, 128), x)

    def test_mod_narrow_rope_second_half_sliding(self, execution_mode):
        """MOD-46: Gemma4TextAttention — narrow(3,128,128): [1,32,16,256]→[1,32,16,128]. RoPE second half."""
        x = cached_randn((1, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="mod46_narrow_rope_second")
        _compare_modes(execution_mode, lambda t: t.narrow(3, 128, 128), x)

    def test_mod_narrow_rope_kv_sliding(self, execution_mode):
        """MOD-47: Gemma4TextAttention KV — narrow(3,0,128): [1,32,8,256]→[1,32,8,128]. KV RoPE first half."""
        x = cached_randn((1, 32, 8, 256), dtype=torch.bfloat16,
                         differentiation="mod47_narrow_kv_sliding")
        _compare_modes(execution_mode, lambda t: t.narrow(3, 0, 128), x)

    def test_mod_narrow_kv_cache_window(self, execution_mode):
        """MOD-48: Gemma4TextAttention — narrow(2,0,32) sliding-window KV crop: [1,8,32,256]→[1,8,32,256]."""
        x = cached_randn((1, 8, 32, 256), dtype=torch.bfloat16,
                         differentiation="mod48_narrow_kv_window")
        _compare_modes(execution_mode, lambda t: t.narrow(2, 0, 32), x)

    def test_mod_narrow_rope_full_attn(self, execution_mode):
        """MOD-49: Gemma4TextAttention full-attn — narrow(3,0,256): [1,32,16,512]→[1,32,16,256]. Full-attn RoPE half."""
        x = cached_randn((1, 32, 16, 512), dtype=torch.bfloat16,
                         differentiation="mod49_narrow_rope_full_attn")
        _compare_modes(execution_mode, lambda t: t.narrow(3, 0, 256), x)

    # ------------------------------------------------------------------
    # Split — MoE expert weight split
    # ------------------------------------------------------------------

    def test_mod_split_moe_weight(self, execution_mode):
        """MOD-50: Gemma4TextMoE — split(704,dim=-1): [256,1408]→([256,704],[256,704])."""
        x = cached_randn((256, 1408), dtype=torch.bfloat16,
                         differentiation="mod50_split_moe_weight")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(704, dim=-1)]),
            x,
        )

    def test_mod_split_moe_weight_then_transpose(self, execution_mode):
        """MOD-51: Gemma4TextMoE — split(704,dim=-1) then transpose on each chunk: the natural MoE GEMM pattern."""
        x = cached_randn((256, 1408), dtype=torch.bfloat16,
                         differentiation="mod51_split_moe_transpose")

        def split_and_transpose(t):
            chunks = t.split(704, dim=-1)
            return torch.stack([c.transpose(0, 1) for c in chunks])

        _compare_modes(execution_mode, split_and_transpose, x, atol=0.5, rtol=0.1)
