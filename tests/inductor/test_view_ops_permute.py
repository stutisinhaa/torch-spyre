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
FVT Section 8 — permute — Operation Level
==========================================

``aten.permute.default`` observed in the Gemma Spyre compile trace.
In the Spyre backend, ``aten.t.default`` (CPU trace) is decomposed to
``aten.permute.default`` during lowering, so this file covers both cases.

Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Consumer selection:
  * §8.1 (Gemma-traced shapes): single baked-in ``* 1.0`` consumer.
  * §8.2/§8.3 (unaligned + size-1): parametrized with both ``"pointwise"``
    and ``"sum_last"`` consumers via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.

Spyre-traced shapes PRM-02, PRM-04, PRM-05, PRM-06 are covered by
module-level tests MOD-31–34 in test_view_ops_module.py. PRM-01 ([128,2816])
and PRM-03 ([2048,2816]) have no matching module test and are kept.

Subsections:
  §8.1  Remaining Spyre-traced shapes              PRM-01, PRM-03
  §8.2  Unaligned innermost dimension              PRM-07–11
  §8.3  Tensors containing size-1 dimensions       PRM-12–15

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §8.2 / §8.3 parametrization
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
# §8.1 — Gemma Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestPermuteSpyreTraced:
    """PRM-01, PRM-03: Spyre-traced shapes not covered by module-level tests."""

    def test_permute_weight_transpose_2d(self, execution_mode, dtype):
        """PRM-01: [128,2816]→permute(1,0)→[2816,128] * 1.0. MoE output weight; no module test at this shape."""
        x = cached_randn((128, 2816), dtype=dtype,
                         differentiation=f"prm01_{dtype}")
        _compare_modes(execution_mode, lambda t: t.permute(1, 0) * 1.0, x)

    def test_permute_weight_transpose_kv(self, execution_mode, dtype):
        """PRM-03: [2048,2816]→permute(1,0)→[2816,2048] * 1.0. KV weight; no module test at this shape."""
        x = cached_randn((2048, 2816), dtype=dtype,
                         differentiation=f"prm03_{dtype}")
        _compare_modes(execution_mode, lambda t: t.permute(1, 0) * 1.0, x)


# ---------------------------------------------------------------------------
# §8.2 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestPermuteUnaligned:
    """PRM-07–11: 2D–6D inputs with unaligned innermost dimension.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_2d_unaligned_new_inner(self, execution_mode, dtype, consumer):
        """PRM-07: [65,256]→permute(1,0)→[256,65]. New inner 65; 65%64=1. Both consumers."""
        x = cached_randn((65, 256), dtype=dtype,
                         differentiation=f"prm07_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.permute(1, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_3d_last_to_first(self, execution_mode, dtype, consumer):
        """PRM-08: [256,32,70]→permute(2,1,0)→[70,32,256]. New inner 256; aligned. Both consumers."""
        x = cached_randn((256, 32, 70), dtype=dtype,
                         differentiation=f"prm08_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.permute(2, 1, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_4d_unaligned_inner_swap(self, execution_mode, dtype, consumer):
        """PRM-09: [1,16,32,65]→permute(0,1,3,2)→[1,16,65,32]. Inner changes 65→32. Both consumers."""
        x = cached_randn((1, 16, 32, 65), dtype=dtype,
                         differentiation=f"prm09_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.permute(0, 1, 3, 2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_5d_cycle(self, execution_mode, dtype, consumer):
        """PRM-10: [1,2,4,8,13]→permute(0,4,2,3,1)→[1,13,4,8,2]. New inner 2 < 64. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 13), dtype=dtype,
                         differentiation=f"prm10_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.permute(0, 4, 2, 3, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_6d_unaligned(self, execution_mode, dtype, consumer):
        """PRM-11: [1,2,4,8,16,13]→permute(0,5,2,3,4,1)→[1,13,4,8,16,2]. New inner 2. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 16, 13), dtype=dtype,
                         differentiation=f"prm11_{dtype}")
        _compare_modes(
            execution_mode, lambda t: _apply_consumer(t.permute(0, 5, 2, 3, 4, 1), consumer), x
        )


# ---------------------------------------------------------------------------
# §8.3 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestPermuteSize1:
    """PRM-12–15: Size-1 dimension interactions.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_3d_size1_cycle(self, execution_mode, dtype, consumer):
        """PRM-12: [1,256,64]→permute(2,0,1)→[64,1,256]. Inner 256 unchanged; size-1 moves to middle. Both consumers."""
        x = cached_randn((1, 256, 64), dtype=dtype,
                         differentiation=f"prm12_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.permute(2, 0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_4d_size1_to_trailing(self, execution_mode, dtype, consumer):
        """PRM-13: [1,8,32,256]→permute(1,2,3,0)→[8,32,256,1]. Size-1 moved to innermost. Both consumers."""
        x = cached_randn((1, 8, 32, 256), dtype=dtype,
                         differentiation=f"prm13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.permute(1, 2, 3, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_5d_size1_leading_and_inner(self, execution_mode, dtype, consumer):
        """PRM-14: [1,8,1,32,256]→permute(0,1,4,2,3)→[1,8,256,1,32]. New inner 32; two size-1 dims. Both consumers."""
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"prm14_{dtype}")
        _compare_modes(
            execution_mode, lambda t: _apply_consumer(t.permute(0, 1, 4, 2, 3), consumer), x
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_permute_6d_size1_to_inner(self, execution_mode, dtype, consumer):
        """PRM-15: [2,4,8,16,256,1]→permute(0,1,2,5,3,4)→[2,4,8,1,16,256]. Inner 256; size-1 trailing→middle. Both consumers."""
        x = cached_randn((2, 4, 8, 16, 256, 1), dtype=dtype,
                         differentiation=f"prm15_{dtype}")
        _compare_modes(
            execution_mode, lambda t: _apply_consumer(t.permute(0, 1, 2, 5, 3, 4), consumer), x
        )
