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
FVT Section 7 — t (2D transpose) — Operation Level
====================================================

``torch.t()`` / ``aten.t.default`` is the 2D-only special case of transpose.
Observed in the Gemma CPU trace only; the Spyre compile trace decomposes
``aten.t`` to ``aten.permute`` during lowering.

Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Consumer selection:
  * §7.2/§7.3 (unaligned + size-1): parametrized with both ``"pointwise"``
    and ``"sum_last"`` consumers via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.
  * 0D / 1D ``t()`` is a no-op in PyTorch — tests use ``* 1.0``.

Gemma CPU-traced shapes (T2D-01–04) are covered by module-level tests
MOD-27–30 in test_view_ops_module.py and are not duplicated here.

Subsections:
  §7.2  Unaligned innermost dimension              T2D-05–08
  §7.3  Tensors containing size-1 dimensions       T2D-09–11

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §7.2 / §7.3 parametrization
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
# §7.2 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTUnaligned:
    """T2D-05–08: Unaligned / swapped dimensions.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_1d_noop(self, execution_mode, dtype, consumer):
        """T2D-05: [256].t()→[256]. 1D t() is a no-op; identity. Both consumers."""
        x = cached_randn((256,), dtype=dtype,
                         differentiation=f"t05_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_unaligned_inner_to_outer(self, execution_mode, dtype, consumer):
        """T2D-06: [65,256].t()→[256,65]. New inner 65; 65%64=1. Both consumers."""
        x = cached_randn((65, 256), dtype=dtype,
                         differentiation=f"t06_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_both_unaligned(self, execution_mode, dtype, consumer):
        """T2D-07: [33,70].t()→[70,33]. New inner 33 < 64; both dims unaligned. Both consumers."""
        x = cached_randn((33, 70), dtype=dtype,
                         differentiation=f"t07_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_aligned_swap(self, execution_mode, dtype, consumer):
        """T2D-08: [128,64].t()→[64,128]. New inner 128; both aligned; inner changes. Both consumers."""
        x = cached_randn((128, 64), dtype=dtype,
                         differentiation=f"t08_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)


# ---------------------------------------------------------------------------
# §7.3 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTSize1:
    """T2D-09–11: Size-1 dimension interactions with t().

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_size1_trailing(self, execution_mode, dtype, consumer):
        """T2D-09: [256,1].t()→[1,256]. Trailing size-1 → leading; new inner 256. Both consumers."""
        x = cached_randn((256, 1), dtype=dtype,
                         differentiation=f"t09_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_size1_leading(self, execution_mode, dtype, consumer):
        """T2D-10: [1,256].t()→[256,1]. Size-1 leading → trailing; new inner 1 (ultra-unaligned). Both consumers."""
        x = cached_randn((1, 256), dtype=dtype,
                         differentiation=f"t10_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_t_2d_both_size1(self, execution_mode, dtype, consumer):
        """T2D-11: [1,1].t()→[1,1]. All dims=1; no layout change. Both consumers."""
        x = cached_randn((1, 1), dtype=dtype,
                         differentiation=f"t11_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.t(), consumer), x)
