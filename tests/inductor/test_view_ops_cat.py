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
FVT Section 10 — cat — Operation Level
========================================

``aten.cat.default`` observed in the Gemma CPU trace only.
Key patterns:
  - RoPE half-rotation concat: [B,S,H,D/2] × 2 → [B,S,H,D]  (dim=-1)
  - KV-cache append: [B,KV,past,D] + [B,KV,new,D] → concat along dim=2
  - Position-id concat: [1,1] + [1,S] → [1,S+1]  int64

Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Consumer selection:
  ``cat`` itself is a compute op (it copies data), so it is already its own
  consumer — but we still need a downstream op to force ``propagate_layouts``
  to commit to a stick layout.  ``* 1.0`` covers all cat cases here because:
  - cat along dim=-1 produces a larger inner dim (aligned or unaligned)
  - cat along a non-trailing dim leaves the inner dim unchanged
  int64 tests have no consumer.

Subsections:
  §10.1  Gemma CPU-traced shapes (aligned)         CAT-01–05
  §10.2  Unaligned innermost dimension             CAT-06–11
  §10.3  Tensors containing size-1 dimensions      CAT-12–14

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §10.2 / §10.3 parametrization
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


# CAT-01, CAT-02 (KV RoPE — distinct halves shape [1,32,8,128]), CAT-03,
# CAT-04, CAT-05 are covered by module-level tests MOD-39–42.
# CAT-02 is the only CPU-traced shape without a direct module test match
# (MOD-39/40 use the full RoPE chain on Q; KV halves shape is distinct).

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCatCPUTraced:
    """CAT-02: KV RoPE half-rotation concat — no matching module-level test."""

    def test_cat_rope_halves_kv_sliding(self, execution_mode, dtype):
        """CAT-02: [1,32,8,128] × 2 → cat(dim=-1) → [1,32,8,256] * 1.0. KV RoPE concat; distinct shape from MOD-39."""
        a = cached_randn((1, 32, 8, 128), dtype=dtype,
                         differentiation=f"cat02a_{dtype}")
        b = cached_randn((1, 32, 8, 128), dtype=dtype,
                         differentiation=f"cat02b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: torch.cat([a, b], dim=-1) * 1.0, a, b
        )


# ---------------------------------------------------------------------------
# §10.2 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCatUnaligned:
    """CAT-06–11: 1D–6D inputs with unaligned innermost dimension.

    Each test is parametrized with both consumers.
    Note: for non-trailing-dim cat (CAT-08, CAT-10), inner is unchanged so
    sum_last reduces along the unchanged inner dim — still a valid path.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_1d_unaligned_halves(self, execution_mode, dtype, consumer):
        """CAT-06: [33] + [33] → cat(dim=0) → [66]. Output inner 66; 66%64=2. Both consumers."""
        a = cached_randn((33,), dtype=dtype,
                         differentiation=f"cat06a_{dtype}")
        b = cached_randn((33,), dtype=dtype,
                         differentiation=f"cat06b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b]), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_2d_trailing_unaligned(self, execution_mode, dtype, consumer):
        """CAT-07: [32,33] + [32,33] → cat(dim=-1) → [32,66]. Output inner 66 unaligned. Both consumers."""
        a = cached_randn((32, 33), dtype=dtype,
                         differentiation=f"cat07a_{dtype}")
        b = cached_randn((32, 33), dtype=dtype,
                         differentiation=f"cat07b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_3d_seq_dim_unaligned(self, execution_mode, dtype, consumer):
        """CAT-08: [1,33,256] + [1,33,256] → cat(dim=1) → [1,66,256]. Seq dim unaligned; inner 256 unchanged. Both consumers."""
        a = cached_randn((1, 33, 256), dtype=dtype,
                         differentiation=f"cat08a_{dtype}")
        b = cached_randn((1, 33, 256), dtype=dtype,
                         differentiation=f"cat08b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_4d_unaligned_inner_halves(self, execution_mode, dtype, consumer):
        """CAT-09: [1,16,32,33] × 2 → cat(dim=-1) → [1,16,32,66]. Unaligned inputs → unaligned output. Both consumers."""
        a = cached_randn((1, 16, 32, 33), dtype=dtype,
                         differentiation=f"cat09a_{dtype}")
        b = cached_randn((1, 16, 32, 33), dtype=dtype,
                         differentiation=f"cat09b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_5d_seq_unaligned(self, execution_mode, dtype, consumer):
        """CAT-10: [1,8,1,33,256] × 2 → cat(dim=3) → [1,8,1,66,256]. Seq dim unaligned; inner 256 unchanged. Both consumers."""
        a = cached_randn((1, 8, 1, 33, 256), dtype=dtype,
                         differentiation=f"cat10a_{dtype}")
        b = cached_randn((1, 8, 1, 33, 256), dtype=dtype,
                         differentiation=f"cat10b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=3), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_6d_inner_unaligned(self, execution_mode, dtype, consumer):
        """CAT-11: [1,2,4,8,16,13] × 2 → cat(dim=-1) → [1,2,4,8,16,26]. 6D; inner 26 < 64. Both consumers."""
        a = cached_randn((1, 2, 4, 8, 16, 13), dtype=dtype,
                         differentiation=f"cat11a_{dtype}")
        b = cached_randn((1, 2, 4, 8, 16, 13), dtype=dtype,
                         differentiation=f"cat11b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )


# ---------------------------------------------------------------------------
# §10.3 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCatSize1:
    """CAT-12–14: Size-1 dimension interactions.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_3d_size1_seq_expand(self, execution_mode, dtype, consumer):
        """CAT-12: [1,1,256] + [1,32,256] → cat(dim=1) → [1,33,256]. Size-1 seq prepended. Both consumers."""
        a = cached_randn((1, 1, 256), dtype=dtype,
                         differentiation=f"cat12a_{dtype}")
        b = cached_randn((1, 32, 256), dtype=dtype,
                         differentiation=f"cat12b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_4d_size1_trailing_halves(self, execution_mode, dtype, consumer):
        """CAT-13: [32,8,16,1] × 2 → cat(dim=-1) → [32,8,16,2]. Two size-1 halves → size-2 inner. Both consumers."""
        a = cached_randn((32, 8, 16, 1), dtype=dtype,
                         differentiation=f"cat13a_{dtype}")
        b = cached_randn((32, 8, 16, 1), dtype=dtype,
                         differentiation=f"cat13b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=-1), consumer), a, b
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_cat_5d_gqa_kv_cache(self, execution_mode, dtype, consumer):
        """CAT-14: [1,8,1,32,256] × 2 → cat(dim=2) → [1,8,2,32,256]. GQA-like cache; inner 256 unchanged. Both consumers."""
        a = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"cat14a_{dtype}")
        b = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"cat14b_{dtype}")
        _compare_modes(
            execution_mode, lambda a, b: _apply_consumer(torch.cat([a, b], dim=2), consumer), a, b
        )
