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
FVT Section 11 — select — Operation Level
==========================================

``aten.select.int`` observed in both traces.
  CPU trace: ``[1,32]→select(1,-1)→[1]``  (last token position, int64)
  Spyre trace: ``[1,8,32]→select(2,0)→[1,8]``  (routing index, int32)

``select`` removes exactly one dimension (rank drops by 1).

Pass criteria: compare_with_cpu, atol=0.0.

Consumer selection:
  * §11.1 (int32/int64 Gemma-traced shapes): no consumer.
  * §11.2 (float aligned shapes): single baked-in ``* 1.0`` consumer.
    SEL-03 (0D scalar output) also uses ``* 1.0`` (sum undefined on scalar).
  * §11.3/§11.4 (unaligned + size-1): parametrized with both ``"pointwise"``
    and ``"sum_last"`` consumers via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.

Subsections:
  §11.1  Gemma traced shapes (int types)           SEL-01–02
  §11.2  Float shapes — 1D–6D                     SEL-03–09
  §11.3  Unaligned innermost dimension             SEL-10–12
  §11.4  Tensors containing size-1 dimensions     SEL-13–15

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §11.3 / §11.4 parametrization
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


# SEL-01 and SEL-02 (Gemma traced int shapes) are covered by module-level
# tests MOD-43 and MOD-44 in test_view_ops_module.py.

# ---------------------------------------------------------------------------
# §11.2 — Float shapes, 1D–6D
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSelectFloat:
    """SEL-03–09: Float dtype shapes covering 1D→0D through 6D→5D."""

    def test_select_1d_to_scalar(self, execution_mode, dtype):
        """SEL-03: [256] → select(0,0) → [] * 1.0. 0D output; sum undefined on scalar."""
        x = cached_randn((256,), dtype=dtype,
                         differentiation=f"sel03_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(0, 0) * 1.0, x)

    def test_select_2d_leading_dim(self, execution_mode, dtype):
        """SEL-04: [8,256] → select(0,3) → [256] * 1.0. Leading dim selected; inner 256 unchanged."""
        x = cached_randn((8, 256), dtype=dtype,
                         differentiation=f"sel04_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(0, 3) * 1.0, x)

    def test_select_3d_seq_select(self, execution_mode, dtype):
        """SEL-05: [1,32,256] → select(1,0) → [1,256] * 1.0. Middle dim selected; inner 256 unchanged."""
        x = cached_randn((1, 32, 256), dtype=dtype,
                         differentiation=f"sel05_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(1, 0) * 1.0, x)

    def test_select_4d_head_select(self, execution_mode, dtype):
        """SEL-06: [1,16,32,256] → select(1,0) → [1,32,256] * 1.0. Head dim selected; inner 256 unchanged."""
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"sel06_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(1, 0) * 1.0, x)

    def test_select_4d_last_token_trailing(self, execution_mode, dtype):
        """SEL-07: [1,16,32,256] → select(2,-1) → [1,16,256] * 1.0. Seq dim selected; inner 256 unchanged."""
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"sel07_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(2, -1) * 1.0, x)

    def test_select_5d_middle_dim(self, execution_mode, dtype):
        """SEL-08: [1,8,2,32,256] → select(2,0) → [1,8,32,256] * 1.0. GQA-like; inner 256 unchanged."""
        x = cached_randn((1, 8, 2, 32, 256), dtype=dtype,
                         differentiation=f"sel08_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(2, 0) * 1.0, x)

    def test_select_6d_to_5d(self, execution_mode, dtype):
        """SEL-09: [1,2,4,8,16,256] → select(3,0) → [1,2,4,16,256] * 1.0. 6D→5D; inner 256 unchanged."""
        x = cached_randn((1, 2, 4, 8, 16, 256), dtype=dtype,
                         differentiation=f"sel09_{dtype}")
        _compare_modes(execution_mode, lambda t: t.select(3, 0) * 1.0, x)


# ---------------------------------------------------------------------------
# §11.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSelectUnaligned:
    """SEL-10–12: Unaligned innermost dimension.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_2d_leading_unaligned(self, execution_mode, dtype, consumer):
        """SEL-10: [8,70] → select(0,0) → [70]. Inner 70 unchanged; 70%64=6. Both consumers."""
        x = cached_randn((8, 70), dtype=dtype,
                         differentiation=f"sel10_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(0, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_3d_trailing_unaligned_new_inner(self, execution_mode, dtype, consumer):
        """SEL-11: [4,256,33] → select(1,0) → [4,33]. Inner 33 unchanged from dim-2; 33 < 64. Both consumers."""
        x = cached_randn((4, 256, 33), dtype=dtype,
                         differentiation=f"sel11_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(1, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_4d_unaligned_inner(self, execution_mode, dtype, consumer):
        """SEL-12: [1,16,32,65] → select(1,0) → [1,32,65]. Inner 65 unchanged; 65%64=1. Both consumers."""
        x = cached_randn((1, 16, 32, 65), dtype=dtype,
                         differentiation=f"sel12_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(1, 0), consumer), x)


# ---------------------------------------------------------------------------
# §11.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSelectSize1:
    """SEL-13–15: Size-1 dimension interactions.

    SEL-03 (1D→scalar, 0D output) uses CONSUMERS_SCALAR (pointwise only).
    All other tests use CONSUMERS (both consumers).
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_2d_size1_leading(self, execution_mode, dtype, consumer):
        """SEL-13: [1,256] → select(0,0) → [256]. Removes size-1 leading dim. Both consumers."""
        x = cached_randn((1, 256), dtype=dtype,
                         differentiation=f"sel13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(0, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_3d_size1_gqa(self, execution_mode, dtype, consumer):
        """SEL-14: [1,8,1] → select(2,0) → [1,8]. Select removes trailing size-1; inner changes 1→8. Both consumers."""
        x = cached_randn((1, 8, 1), dtype=dtype,
                         differentiation=f"sel14_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(2, 0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_select_5d_size1_dim(self, execution_mode, dtype, consumer):
        """SEL-15: [1,8,1,32,256] → select(2,0) → [1,8,32,256]. Select the size-1 dim; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"sel15_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.select(2, 0), consumer), x)
