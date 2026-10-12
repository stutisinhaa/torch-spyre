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
FVT Section 6 — transpose — Operation Level
=============================================

Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Consumer selection:
  * §6.3/§6.4 (unaligned + size-1): parametrized with both ``"pointwise"``
    and ``"sum_last"`` consumers via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.

Gemma CPU-traced shapes (TRP-01–08) are covered by module-level tests
MOD-22–26 in test_view_ops_module.py and are not duplicated here.

Subsections:
  §6.3  Unaligned innermost dimension              TRP-09–14
  §6.4  Tensors containing size-1 dimensions       TRP-15–19

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §6.3 / §6.4 parametrization
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
# §6.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTransposeUnaligned:
    """TRP-09–14: 2D–6D inputs with unaligned innermost dimension.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_2d_unaligned_inner_swap(self, execution_mode, dtype, consumer):
        """TRP-09: [65,256]→transpose(0,1)→[256,65]. Inner changes 256→65; 65%64=1. Both consumers."""
        x = cached_randn((65, 256), dtype=dtype,
                         differentiation=f"trp09_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_3d_last_two_unaligned(self, execution_mode, dtype, consumer):
        """TRP-10: [4,32,70]→transpose(1,2)→[4,70,32]. Inner changes 70→32; input inner 70 unaligned. Both consumers."""
        x = cached_randn((4, 32, 70), dtype=dtype,
                         differentiation=f"trp10_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_4d_key_unaligned_65(self, execution_mode, dtype, consumer):
        """TRP-11: [1,16,32,65]→transpose(2,3)→[1,16,65,32]. Inner changes 65→32. Both consumers."""
        x = cached_randn((1, 16, 32, 65), dtype=dtype,
                         differentiation=f"trp11_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(2, 3), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_4d_head_swap_unaligned_inner(self, execution_mode, dtype, consumer):
        """TRP-12: [2,32,8,70]→transpose(1,2)→[2,8,32,70]. Inner 70 unchanged; 70%64=6. Both consumers."""
        x = cached_randn((2, 32, 8, 70), dtype=dtype,
                         differentiation=f"trp12_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_5d_unaligned(self, execution_mode, dtype, consumer):
        """TRP-13: [1,8,1,32,65]→transpose(1,3)→[1,32,1,8,65]. Inner 65 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 32, 65), dtype=dtype,
                         differentiation=f"trp13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 3), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_6d_unaligned(self, execution_mode, dtype, consumer):
        """TRP-14: [1,2,4,8,16,13]→transpose(1,4)→[1,16,4,8,2,13]. Inner 13 unchanged; 13 < 64. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 16, 13), dtype=dtype,
                         differentiation=f"trp14_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 4), consumer), x)


# ---------------------------------------------------------------------------
# §6.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestTransposeSize1:
    """TRP-15–19: Size-1 dimension interactions.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_2d_size1_trailing(self, execution_mode, dtype, consumer):
        """TRP-15: [256,1]→transpose(0,1)→[1,256]. Trailing size-1 moves to leading. Both consumers."""
        x = cached_randn((256, 1), dtype=dtype,
                         differentiation=f"trp15_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_3d_size1_middle(self, execution_mode, dtype, consumer):
        """TRP-16: [4,1,256]→transpose(0,1)→[1,4,256]. Size-1 at middle; inner 256 unchanged. Both consumers."""
        x = cached_randn((4, 1, 256), dtype=dtype,
                         differentiation=f"trp16_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_4d_size1_gqa_expand(self, execution_mode, dtype, consumer):
        """TRP-17: [1,8,1,256]→transpose(1,2)→[1,1,8,256]. GQA-like; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"trp17_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(1, 2), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_5d_size1_to_trailing(self, execution_mode, dtype, consumer):
        """TRP-18: [1,8,32,1,256]→transpose(3,4)→[1,8,32,256,1]. Size-1 moved to innermost. Both consumers."""
        x = cached_randn((1, 8, 32, 1, 256), dtype=dtype,
                         differentiation=f"trp18_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(3, 4), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_transpose_6d_all_size1_except_inner(self, execution_mode, dtype, consumer):
        """TRP-19: [1,1,1,1,1,256]→transpose(0,4)→[1,1,1,1,1,256]. No-op in practice; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 1, 1, 1, 1, 256), dtype=dtype,
                         differentiation=f"trp19_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.transpose(0, 4), consumer), x)
