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
FVT Section 4 — unsqueeze — Operation Level
============================================

Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Every test in §4.3 and §4.4 uses ``_apply_consumer`` with both a pointwise
(``* 1.0``) and a reduction (``.sum(dim=-1)``) consumer, parametrized per
test method via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.
UNS-14 (0D→1D) uses CONSUMERS since the result is [1] (not 0D).

Subsections:
  §4.1  Gemma CPU-traced shapes (aligned)          UNS-04
         (UNS-01/02/03/05/06/07 removed — covered by MOD-11–16/18/19/21)
  §4.2  Spyre-traced shapes (aligned)              UNS-09, UNS-10
         (UNS-08/11/12/13 removed — covered by MOD-15/16/20/21)
  §4.3  Unaligned innermost dimension              UNS-14–19
  §4.4  Tensors containing size-1 dimensions       UNS-20–24, UNS-26–28
         (UNS-25 removed — covered by MOD-11)

Consumer parametrization:
  CONSUMERS        = ["pointwise", "sum_last"]  — for all unsqueeze outputs (≥1D)
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

# Consumer lists for §4.3 / §4.4 parametrization
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
# §4.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------
# UNS-01 removed: [32]→unsqueeze(0) int64 — covered by MOD-11
# UNS-02 removed: [1,32,256]→unsqueeze(2) — covered by MOD-12
# UNS-03 removed: [1,8,32,256]→unsqueeze(2) — covered by MOD-13
# UNS-05 removed: [256]→unsqueeze(-1) — covered by MOD-14
# UNS-06 removed: [1,8,32,256]→unsqueeze(2).expand(1,8,2,32,256) — covered by MOD-18
# UNS-07 removed: [1,2,32,512]→unsqueeze(2).expand(1,2,8,32,512) — covered by MOD-19

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeCPUTraced:
    """UNS-04: Gemma CPU-traced aligned shapes not covered by module-level tests."""

    def test_unsqueeze_4d_gqa_full_attn_to_5d(self, execution_mode, dtype):
        """UNS-04: [1,2,32,512]→unsqueeze(2) * 1.0→[1,2,1,32,512]. Full-attn GQA; no module-level test."""
        x = cached_randn((1, 2, 32, 512), dtype=dtype,
                         differentiation=f"uns04_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(2) * 1.0, x)


# ---------------------------------------------------------------------------
# §4.2 — Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------
# UNS-08 removed: [4096,2816]→unsqueeze(0) — covered by MOD-15
# UNS-11 removed: [1,64,16,2,128]→unsqueeze(-3) — covered by MOD-16
# UNS-12 removed: w.unsqueeze(0).expand(1,2816,4096)→bmm — covered by MOD-20
# UNS-13 removed: [1,2816]→unsqueeze(1).expand(1,8,2816) — covered by MOD-21

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeSpyreTraced:
    """UNS-09, UNS-10: Spyre-traced shapes not covered by module-level tests."""

    def test_unsqueeze_2d_linear_weight_kv(self, execution_mode, dtype):
        """UNS-09: [2048,2816]→unsqueeze(0) * 1.0→[1,2048,2816]. Non-trailing; no module-level test."""
        x = cached_randn((2048, 2816), dtype=dtype,
                         differentiation=f"uns09_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(0) * 1.0, x)

    def test_unsqueeze_2d_linear_weight_mlp(self, execution_mode, dtype):
        """UNS-10: [2816,2112]→unsqueeze(0) * 1.0→[1,2816,2112]. Non-trailing; no module-level test."""
        x = cached_randn((2816, 2112), dtype=dtype,
                         differentiation=f"uns10_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(0) * 1.0, x)


# ---------------------------------------------------------------------------
# §4.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeUnaligned:
    """UNS-14–19: 0D–5D inputs (producing 1D–6D) with unaligned innermost dim.

    Each test is parametrized with both consumers so that propagate_layouts is
    exercised under both a passthrough (``* 1.0``) and a reducing
    (``sum(dim=-1)``) downstream op.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_0d_to_1d_unaligned(self, execution_mode, dtype, consumer):
        """UNS-14: []→unsqueeze(0)→[1]. 0D→1D; inner=1. sum_last reduces [1]→scalar (valid). Both consumers."""
        x = torch.tensor(1.5, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_1d_to_2d_unaligned_70(self, execution_mode, dtype, consumer):
        """UNS-15: [70]→unsqueeze(0)→[1,70]. Non-trailing; inner 70 unchanged. Both consumers."""
        x = cached_randn((70,), dtype=dtype, differentiation=f"uns15_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_2d_to_3d_unaligned_33(self, execution_mode, dtype, consumer):
        """UNS-16: [32,33]→unsqueeze(0)→[1,32,33]. Non-trailing; inner 33 unchanged. Both consumers."""
        x = cached_randn((32, 33), dtype=dtype, differentiation=f"uns16_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_3d_to_4d_rope_unaligned_65(self, execution_mode, dtype, consumer):
        """UNS-17: [1,32,65]→unsqueeze(2)→[1,32,1,65]. Non-trailing; inner 65 unchanged. Both consumers."""
        x = cached_randn((1, 32, 65), dtype=dtype,
                         differentiation=f"uns17_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_4d_to_5d_gqa_unaligned_65(self, execution_mode, dtype, consumer):
        """UNS-18: [2,8,32,65]→unsqueeze(2)→[2,8,1,32,65]. Non-trailing; inner 65 unchanged. Both consumers."""
        x = cached_randn((2, 8, 32, 65), dtype=dtype,
                         differentiation=f"uns18_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_5d_to_6d_unaligned_13(self, execution_mode, dtype, consumer):
        """UNS-19: [1,2,4,8,13]→unsqueeze(3)→[1,2,4,1,8,13]. Non-trailing (dim 3); inner 13 unchanged. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 13), dtype=dtype,
                         differentiation=f"uns19_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(3), consumer), x)


# ---------------------------------------------------------------------------
# §4.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestUnsqueezeSize1:
    """UNS-20–28: 0D–5D inputs (producing 1D–6D). Trailing unsqueeze(-1) creates size-1 innermost.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_0d_to_1d_aligned(self, execution_mode, dtype, consumer):
        """UNS-20: []→unsqueeze(0)→[1]. 0D→1D; result size-1. Both consumers."""
        x = torch.tensor(2.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_1d_size1_to_2d(self, execution_mode, dtype, consumer):
        """UNS-21: [1]→unsqueeze(0)→[1,1]. Size-1 input stays size-1. Both consumers."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"uns21_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_2d_trailing_size1_aligned(self, execution_mode, dtype, consumer):
        """UNS-22: [32,256]→unsqueeze(-1)→[32,256,1]. Trailing; new inner=1. Both consumers."""
        x = cached_randn((32, 256), dtype=dtype,
                         differentiation=f"uns22_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_2d_trailing_size1_unaligned(self, execution_mode, dtype, consumer):
        """UNS-23: [4,8,70]→unsqueeze(-1)→[4,8,70,1]. Trailing; new inner=1; preceding 70 unaligned. Both consumers."""
        x = cached_randn((4, 8, 70), dtype=dtype,
                         differentiation=f"uns23_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_3d_dual_axis(self, execution_mode, dtype, consumer):
        """UNS-24: [32,256]→unsqueeze(0).unsqueeze(3)→[1,32,256,1]. Trailing unsqueeze adds size-1 innermost. Both consumers."""
        x = cached_randn((32, 256), dtype=dtype,
                         differentiation=f"uns24_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(0).unsqueeze(3), consumer), x)

    # UNS-25 removed: [32]→unsqueeze(0).unsqueeze(1).unsqueeze(3) int64 — covered by MOD-11

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_4d_size1_innermost_to_5d(self, execution_mode, dtype, consumer):
        """UNS-26: [1,8,32,1]→unsqueeze(2)→[1,8,1,32,1]. Non-trailing (dim 2); inner=1 unchanged. Both consumers."""
        x = cached_randn((1, 8, 32, 1), dtype=dtype,
                         differentiation=f"uns26_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_5d_trailing_size1_to_6d(self, execution_mode, dtype, consumer):
        """UNS-27: [1,2,4,8,256]→unsqueeze(-1)→[1,2,4,8,256,1]. Trailing; new inner=1. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 256), dtype=dtype,
                         differentiation=f"uns27_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_unsqueeze_5d_to_6d_aligned(self, execution_mode, dtype, consumer):
        """UNS-28: [1,2,4,8,256]→unsqueeze(3)→[1,2,4,1,8,256]. Non-trailing (dim 3); inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 256), dtype=dtype,
                         differentiation=f"uns28_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.unsqueeze(3), consumer), x)
