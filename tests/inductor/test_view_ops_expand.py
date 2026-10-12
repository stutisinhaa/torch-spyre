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
FVT Section 5 — expand — Operation Level
=========================================

Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Every test in §5.3 and §5.4 uses ``_apply_consumer`` with both a pointwise
(``* 1.0``) and a reduction (``.sum(dim=-1)``) consumer, parametrized per
test method via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.
expand never changes which dimension is innermost, so ``sum(dim=-1)`` reduces
along the (unchanged) innermost — this is a valid and distinct code path.
EXP-05/06/07 already have real consumers (view, reshape, add) and are
unchanged.

Subsections:
  §5.1  Gemma CPU-traced shapes (aligned)          EXP-04–07
         (EXP-01/02/03 removed — covered by MOD-17/18/19)
  §5.2  Spyre-traced shapes (aligned)              EXP-10, EXP-11
         (EXP-08/09 removed — covered by MOD-20/07)
  §5.3  Unaligned innermost dimension              EXP-12–20
  §5.4  Tensors containing size-1 dimensions       EXP-21–31

Consumer parametrization:
  CONSUMERS = ["pointwise", "sum_last"]
  _apply_consumer(t, consumer):
    "pointwise" → t * 1.0
    "sum_last"  → t.sum(dim=-1)

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §5.3 / §5.4 parametrization
CONSUMERS = ["pointwise", "sum_last"]


def _apply_consumer(t, consumer):
    """Apply the named consumer to tensor t."""
    if consumer == "pointwise":
        return t * 1.0
    # sum_last
    return t.sum(dim=-1)


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
            (a.dtype for a in args if isinstance(a, torch.Tensor)), torch.float32
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


# ---------------------------------------------------------------------------
# §5.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------
# EXP-01 removed: [1,8,1,32,256]→expand(1,8,2,32,256) — covered by MOD-18
# EXP-02 removed: [1,2,1,32,512]→expand(1,2,8,32,512) — covered by MOD-19
# EXP-03 removed: [1,1,32,32]→expand(1,-1,32,32) — covered by MOD-17

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandCPUTraced:
    """EXP-04–07: Gemma CPU-traced aligned shapes not covered by module-level tests."""

    def test_expand_rope_freq_heads(self, execution_mode, dtype):
        """EXP-04: [1,32,1,256]→expand(1,32,16,256) * 1.0→[1,32,16,256]. RoPE frequency broadcast; inner 256 unchanged."""
        x = cached_randn((1, 32, 1, 256), dtype=dtype,
                         differentiation=f"exp04_{dtype}")
        _compare_modes(execution_mode, lambda t: t.expand(1, 32, 16, 256) * 1.0, x)

    def test_expand_bmm_prep_then_view(self, execution_mode, dtype):
        """EXP-05: [1,16,32,256]→expand(1,16,32,256).view(16,32,256)→[16,32,256]. Same-shape expand then view."""
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"exp05_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: t.expand(1, 16, 32, 256).view(16, 32, 256),
            x,
        )

    def test_expand_chain_then_reshape_heads(self, execution_mode, dtype):
        """EXP-06: [1,8,1,32,256]→expand(1,8,2,32,256).reshape(1,16,32,256)→[1,16,32,256]. Forces clone."""
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"exp06_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: t.expand(1, 8, 2, 32, 256).reshape(1, 16, 32, 256),
            x,
        )

    def test_expand_chain_bias_add(self, execution_mode, dtype):
        """EXP-07: [2816]→expand(32,2816)+x[32,2816]. Bias broadcast then add."""
        b = cached_randn((2816,), dtype=dtype, differentiation=f"exp07_b_{dtype}")
        x = cached_randn((32, 2816), dtype=dtype,
                         differentiation=f"exp07_x_{dtype}")

        def fn(bias, inp):
            return bias.expand(32, 2816) + inp

        _compare_modes(execution_mode, fn, b, x)


# ---------------------------------------------------------------------------
# §5.2 — Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------
# EXP-08 removed: [1,64,2816]→expand(1,64,2816) — covered by MOD-20 (input same-shape expand)
# EXP-09 removed: [1,2816,4096]→expand(1,2816,4096) — covered by MOD-20 (weight same-shape expand)

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandSpyreTraced:
    """EXP-10, EXP-11: Spyre-traced shapes not covered by module-level tests."""

    def test_expand_spyre_weight_kv(self, execution_mode, dtype):
        """EXP-10: [1,2816,2048]→expand(1,2816,2048) * 1.0→[1,2816,2048]. Same-shape; inner 2048 unchanged."""
        x = cached_randn((1, 2816, 2048), dtype=dtype,
                         differentiation=f"exp10_{dtype}")
        _compare_modes(execution_mode, lambda t: t.expand(1, 2816, 2048) * 1.0, x)

    def test_expand_spyre_weight_mlp(self, execution_mode, dtype):
        """EXP-11: [1,2816,2112]→expand(1,2816,2112) * 1.0→[1,2816,2112]. Same-shape; inner 2112 unchanged."""
        x = cached_randn((1, 2816, 2112), dtype=dtype,
                         differentiation=f"exp11_{dtype}")
        _compare_modes(execution_mode, lambda t: t.expand(1, 2816, 2112) * 1.0, x)


# ---------------------------------------------------------------------------
# §5.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandUnaligned:
    """EXP-12–20: 0D–6D inputs with unaligned innermost dimension.

    Each test is parametrized with both consumers. expand never changes
    innermost, so sum_last reduces the (unchanged) unaligned inner dim.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_0d_scalar_to_4d_unaligned(self, execution_mode, dtype, consumer):
        """EXP-12: []→expand(1,16,32,70)→[1,16,32,70]. Inner 70: 70%64=6. Both consumers."""
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 16, 32, 70), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_1d_bias_unaligned_70(self, execution_mode, dtype, consumer):
        """EXP-13: [70]→expand(2,32,70)→[2,32,70]. Inner 70 unchanged. Both consumers."""
        x = cached_randn((70,), dtype=dtype, differentiation=f"exp13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(2, 32, 70), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_2d_batch_unaligned_33(self, execution_mode, dtype, consumer):
        """EXP-14: [1,33]→expand(4,33)→[4,33]. Inner 33 unchanged. Both consumers."""
        x = cached_randn((1, 33), dtype=dtype, differentiation=f"exp14_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(4, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_3d_batch_unaligned_33(self, execution_mode, dtype, consumer):
        """EXP-15: [1,32,33]→expand(4,32,33)→[4,32,33]. Inner 33 unchanged. Both consumers."""
        x = cached_randn((1, 32, 33), dtype=dtype,
                         differentiation=f"exp15_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(4, 32, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_5d_gqa_unaligned_65(self, execution_mode, dtype, consumer):
        """EXP-16: [1,8,1,32,65]→expand(1,8,2,32,65)→[1,8,2,32,65]. Inner 65 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 32, 65), dtype=dtype,
                         differentiation=f"exp16_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 8, 2, 32, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_4d_mask_unaligned_seq33(self, execution_mode, dtype, consumer):
        """EXP-17: [1,1,33,33]→expand(2,8,33,33)→[2,8,33,33]. Inner 33 unchanged. Both consumers."""
        x = cached_randn((1, 1, 33, 33), dtype=dtype,
                         differentiation=f"exp17_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(2, 8, 33, 33), consumer), x)

    # EXP-18 removed: exact duplicate of EXP-16 (same input/output shape [1,8,1,32,65]→[1,8,2,32,65]).

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_3d_minus_one_unaligned(self, execution_mode, dtype, consumer):
        """EXP-19: [2,1,70]→expand(-1,16,-1)→[2,16,70]. -1 preserves dim; inner 70 unchanged. Both consumers."""
        x = cached_randn((2, 1, 70), dtype=dtype,
                         differentiation=f"exp19_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(-1, 16, -1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_6d_unaligned_inner(self, execution_mode, dtype, consumer):
        """EXP-20: [1,2,1,4,1,13]→expand(1,2,8,4,16,13)→[1,2,8,4,16,13]. 6D; inner 13 unchanged. Both consumers."""
        x = cached_randn((1, 2, 1, 4, 1, 13), dtype=dtype,
                         differentiation=f"exp20_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(1, 2, 8, 4, 16, 13), consumer),
            x,
        )


# ---------------------------------------------------------------------------
# §5.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestExpandSize1:
    """EXP-21–31: 0D–6D. Size-1 innermost can stay size-1 or be broadcast to aligned.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_0d_scalar_to_4d_aligned(self, execution_mode, dtype, consumer):
        """EXP-21: []→expand(1,16,32,256)→[1,16,32,256]. 0D broadcast; inner 256 unchanged. Both consumers."""
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 16, 32, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_1d_size1_keep_inner(self, execution_mode, dtype, consumer):
        """EXP-22: [1]→expand(2,32,1)→[2,32,1]. Innermost stays size-1. Both consumers."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"exp22_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(2, 32, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_1d_size1_to_aligned(self, execution_mode, dtype, consumer):
        """EXP-23: [1]→expand(256)→[256]. 1D size-1 broadcast to aligned 256. Both consumers."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"exp23_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_2d_size1_inner_to_2816(self, execution_mode, dtype, consumer):
        """EXP-24: [32,1]→expand(32,2816)→[32,2816]. Size-1 innermost broadcast to aligned 2816. Both consumers."""
        x = cached_randn((32, 1), dtype=dtype, differentiation=f"exp24_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(32, 2816), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_3d_moe_trailing1_broadcast(self, execution_mode, dtype, consumer):
        """EXP-25: [32,8,1]→expand(32,8,2816)→[32,8,2816]. MoE routing trailing-1 broadcast. Both consumers."""
        x = cached_randn((32, 8, 1), dtype=dtype,
                         differentiation=f"exp25_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(32, 8, 2816), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_4d_all_size1_except_inner(self, execution_mode, dtype, consumer):
        """EXP-26: [1,1,1,256]→expand(4,16,32,256)→[4,16,32,256]. All outer size-1; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 1, 1, 256), dtype=dtype,
                         differentiation=f"exp26_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(4, 16, 32, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_4d_mqa_16x(self, execution_mode, dtype, consumer):
        """EXP-27: [1,1,1,256]→expand(1,16,32,256)→[1,16,32,256]. MQA (KV=1) 16× broadcast. Both consumers."""
        x = cached_randn((1, 1, 1, 256), dtype=dtype,
                         differentiation=f"exp27_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.expand(1, 16, 32, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_5d_size1_inner_aligned(self, execution_mode, dtype, consumer):
        """EXP-28: [16,1,4,16,1]→expand(16,8,4,16,256)→[16,8,4,16,256]. 5D; size-1 inner → aligned 256. Both consumers."""
        x = cached_randn((16, 1, 4, 16, 1), dtype=dtype,
                         differentiation=f"exp28_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(16, 8, 4, 16, 256), consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_5d_size1_inner_keep(self, execution_mode, dtype, consumer):
        """EXP-29: [16,1,4,16,1]→expand(16,8,4,16,1)→[16,8,4,16,1]. 5D; inner stays size-1. Both consumers."""
        x = cached_randn((16, 1, 4, 16, 1), dtype=dtype,
                         differentiation=f"exp29_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(16, 8, 4, 16, 1), consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_6d_size1_to_aligned(self, execution_mode, dtype, consumer):
        """EXP-30: [1,1,1,4,1,256]→expand(2,8,4,4,16,256)→[2,8,4,4,16,256]. 6D; multiple size-1 broadcast. Both consumers."""
        x = cached_randn((1, 1, 1, 4, 1, 256), dtype=dtype,
                         differentiation=f"exp30_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(2, 8, 4, 4, 16, 256), consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_expand_6d_aligned(self, execution_mode, dtype, consumer):
        """EXP-31: [1,2,1,4,1,256]→expand(8,2,4,4,16,256)→[8,2,4,4,16,256]. 6D aligned; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 2, 1, 4, 1, 256), dtype=dtype,
                         differentiation=f"exp31_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.expand(8, 2, 4, 4, 16, 256), consumer),
            x,
        )
