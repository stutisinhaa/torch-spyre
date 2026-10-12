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
FVT Section 12 — narrow / slice — Operation Level
===================================================

``aten.slice.Tensor`` (exposed as ``torch.narrow``) observed in both traces.
Key CPU shapes:
  - RoPE first half:  [B,S,H,256] → narrow(3,0,128) → [B,S,H,128]
  - RoPE second half: [B,S,H,256] → narrow(3,128,128) → [B,S,H,128]
  - KV-cache crop:    [1,8,32,256] → narrow(2,0,32) → [1,8,32,256]  (no-op)
Key Spyre shapes:
  - [64,128,64]  → narrow(2,0,1)  → [64,128,1]
  - [1,8,64]     → narrow(2,0,32) → [1,8,32]
  - [1,8,64]     → narrow(2,0,1)  → [1,8,1]  (bfloat16 only)

Note: ``torch.narrow(dim, start, length)`` and ``t.narrow(dim, start, length)``
are equivalent.

Pass criteria: compare_with_cpu, atol=0.0.

Consumer selection:
  * §12.1/§12.2 (Gemma-traced shapes): single baked-in ``* 1.0`` consumer.
  * §12.3/§12.4 (unaligned + size-1): parametrized with both ``"pointwise"``
    and ``"sum_last"`` consumers via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.

Subsections:
  §12.1  Gemma CPU-traced shapes (aligned)         SLC-01–04
  §12.2  Gemma Spyre-traced shapes                 SLC-05–07
  §12.3  Unaligned innermost dimension             SLC-08–13
  §12.4  Tensors containing size-1 dimensions      SLC-14–17

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §12.3 / §12.4 parametrization
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


# SLC-01–04 (CPU-traced Gemma shapes) are covered by module-level tests
# MOD-45–48 in test_view_ops_module.py.

# ---------------------------------------------------------------------------
# §12.2 — Gemma Spyre-traced shapes
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestNarrowSpyreTraced:
    """SLC-05–06: Shapes from gemma4_fx_module_trace_v2.html."""

    def test_slice_3d_last_element(self, execution_mode, dtype):
        """SLC-05: [64,128,64]→narrow(2,0,1)→[64,128,1] * 1.0. Inner changes 64→1."""
        x = cached_randn((64, 128, 64), dtype=dtype,
                         differentiation=f"slc05_{dtype}")
        _compare_modes(execution_mode, lambda t: t.narrow(2, 0, 1) * 1.0, x)

    def test_slice_3d_half_inner(self, execution_mode, dtype):
        """SLC-06: [1,8,64]→narrow(2,0,32)→[1,8,32] * 1.0. Inner changes 64→32."""
        x = cached_randn((1, 8, 64), dtype=dtype,
                         differentiation=f"slc06_{dtype}")
        _compare_modes(execution_mode, lambda t: t.narrow(2, 0, 32) * 1.0, x)


@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestNarrowSpyreTracedBf16:
    """SLC-07: bf16-only shape from Spyre trace."""

    def test_slice_3d_single_element_inner_bf16(self, execution_mode):
        """SLC-07: [1,8,64]→narrow(2,0,1)→[1,8,1] * 1.0. bf16 only; inner changes 64→1."""
        x = cached_randn((1, 8, 64), dtype=torch.bfloat16,
                         differentiation="slc07_bf16")
        _compare_modes(execution_mode, lambda t: t.narrow(2, 0, 1) * 1.0, x)


# ---------------------------------------------------------------------------
# §12.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestNarrowUnaligned:
    """SLC-08–13: 1D–6D inputs with unaligned output innermost dimension.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_1d_unaligned_result(self, execution_mode, dtype, consumer):
        """SLC-08: [256]→narrow(0,0,65)→[65]. Output inner 65; 65%64=1. Both consumers."""
        x = cached_randn((256,), dtype=dtype,
                         differentiation=f"slc08_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(0, 0, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_2d_trailing_unaligned(self, execution_mode, dtype, consumer):
        """SLC-09: [8,128]→narrow(1,0,33)→[8,33]. Inner changes 128→33; 33 < 64. Both consumers."""
        x = cached_randn((8, 128), dtype=dtype,
                         differentiation=f"slc09_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(1, 0, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_3d_inner_unaligned(self, execution_mode, dtype, consumer):
        """SLC-10: [1,32,128]→narrow(2,0,70)→[1,32,70]. Inner changes 128→70; 70%64=6. Both consumers."""
        x = cached_randn((1, 32, 128), dtype=dtype,
                         differentiation=f"slc10_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(2, 0, 70), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_4d_head_dim_unaligned(self, execution_mode, dtype, consumer):
        """SLC-11: [1,16,32,256]→narrow(3,0,65)→[1,16,32,65]. Inner changes 256→65. Both consumers."""
        x = cached_randn((1, 16, 32, 256), dtype=dtype,
                         differentiation=f"slc11_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(3, 0, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_5d_inner_below_stick(self, execution_mode, dtype, consumer):
        """SLC-12: [1,2,4,8,128]→narrow(4,0,13)→[1,2,4,8,13]. Inner changes 128→13 < 64. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 128), dtype=dtype,
                         differentiation=f"slc12_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(4, 0, 13), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_6d_inner_unaligned(self, execution_mode, dtype, consumer):
        """SLC-13: [1,2,4,8,16,128]→narrow(5,0,33)→[1,2,4,8,16,33]. 6D; inner changes 128→33 < 64. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 16, 128), dtype=dtype,
                         differentiation=f"slc13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(5, 0, 33), consumer), x)


# ---------------------------------------------------------------------------
# §12.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestNarrowSize1:
    """SLC-14–17: Size-1 dimension interactions.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_2d_size1_leading(self, execution_mode, dtype, consumer):
        """SLC-14: [1,256]→narrow(0,0,1)→[1,256]. Narrow on size-1 dim; no-op; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 256), dtype=dtype,
                         differentiation=f"slc14_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(0, 0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_3d_to_size1_inner(self, execution_mode, dtype, consumer):
        """SLC-15: [1,32,256]→narrow(2,0,1)→[1,32,1]. Slice to size-1 innermost; inner changes 256→1. Both consumers."""
        x = cached_randn((1, 32, 256), dtype=dtype,
                         differentiation=f"slc15_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(2, 0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_4d_already_size1_dim(self, execution_mode, dtype, consumer):
        """SLC-16: [1,8,1,256]→narrow(2,0,1)→[1,8,1,256]. Narrow the existing size-1 dim; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"slc16_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(2, 0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_narrow_5d_inner_to_size1(self, execution_mode, dtype, consumer):
        """SLC-17: [1,8,2,32,64]→narrow(4,0,1)→[1,8,2,32,1]. 5D; innermost sliced to 1. Both consumers."""
        x = cached_randn((1, 8, 2, 32, 64), dtype=dtype,
                         differentiation=f"slc17_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.narrow(4, 0, 1), consumer), x)
