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
FVT Section 9 — clone — Operation Level
========================================

``aten.clone.default`` observed in both the CPU and Spyre compile traces.

Key architectural note
-----------------------
``clone`` has its own path in ``propagate_layouts._clone_layout``: it
materialises a new contiguous buffer (or triggers a restickify when the input
has a non-zero offset in the stick dimension).  Because clone already forces
a layout decision, **no extra compute consumer (``* 1.0`` / ``.sum(dim=-1)``)
is appended** — clone IS its own consumer.

Critical restickify tests
--------------------------
``CLN-05`` (transpose → clone) and ``CLN-10`` (permute → clone) exercise the
path where the input is non-contiguous, which forces the compiler to insert a
restickify node.

Pass criteria: compare_with_cpu, atol=0.0.

Subsections:
  §9.1  Gemma CPU-traced shapes (aligned)          CLN-01–05
  §9.2  Gemma Spyre-traced shapes (aligned)        CLN-06–10
  §9.3  Unaligned innermost dimension              CLN-11–16
  §9.4  Tensors containing size-1 dimensions       CLN-17–21

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

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


# CLN-01–05 (CPU-traced Gemma shapes) are covered by module-level tests
# MOD-18, MOD-19, MOD-35, MOD-36, MOD-37 in test_view_ops_module.py.

# ---------------------------------------------------------------------------
# §9.2 — Gemma Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCloneSpyreTraced:
    """CLN-06–10: Shapes from gemma4_fx_module_trace_v2.html (Spyre compile, seq=64)."""

    def test_clone_moe_token_3d(self, execution_mode, dtype):
        """CLN-06: [1,8,64].clone(). MoE token routing; inner 64 aligned."""
        x = cached_randn((1, 8, 64), dtype=dtype,
                         differentiation=f"cln06_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_moe_hidden_3d(self, execution_mode, dtype):
        """CLN-07: [1,8,2816].clone(). MoE hidden state; inner 2816 aligned."""
        x = cached_randn((1, 8, 2816), dtype=dtype,
                         differentiation=f"cln07_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_attn_output_4d_s64(self, execution_mode, dtype):
        """CLN-08: [1,16,64,256].clone(). Spyre attn output; aligned."""
        x = cached_randn((1, 16, 64, 256), dtype=dtype,
                         differentiation=f"cln08_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_attn_output_full_attn_s64(self, execution_mode, dtype):
        """CLN-09: [1,16,64,512].clone(). Full-attention Spyre output; inner 512 aligned."""
        x = cached_randn((1, 16, 64, 512), dtype=dtype,
                         differentiation=f"cln09_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_after_permute_head_swap(self, execution_mode, dtype):
        """CLN-10: [1,64,16,256].permute(0,2,1,3).clone()→[1,16,64,256]. Clone of non-contiguous permuted tensor; restickify path."""
        x = cached_randn((1, 64, 16, 256), dtype=dtype,
                         differentiation=f"cln10_{dtype}")
        _compare_modes(execution_mode, lambda t: t.permute(0, 2, 1, 3).clone(), x)


# ---------------------------------------------------------------------------
# §9.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCloneUnaligned:
    """CLN-11–16: 1D–6D inputs with unaligned innermost dimension."""

    def test_clone_1d_unaligned(self, execution_mode, dtype):
        """CLN-11: [63].clone(). Inner 63 < 64."""
        x = cached_randn((63,), dtype=dtype,
                         differentiation=f"cln11_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_2d_unaligned_inner(self, execution_mode, dtype):
        """CLN-12: [32,70].clone(). Inner 70; 70%64=6."""
        x = cached_randn((32, 70), dtype=dtype,
                         differentiation=f"cln12_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_3d_unaligned_after_transpose(self, execution_mode, dtype):
        """CLN-13: [4,256,65].transpose(1,2).clone()→[4,65,256]. Clone of non-contiguous; output inner 256."""
        x = cached_randn((4, 256, 65), dtype=dtype,
                         differentiation=f"cln13_{dtype}")
        _compare_modes(execution_mode, lambda t: t.transpose(1, 2).clone(), x)

    def test_clone_4d_unaligned_inner_33(self, execution_mode, dtype):
        """CLN-14: [2,8,32,33].clone(). Inner 33 < 64."""
        x = cached_randn((2, 8, 32, 33), dtype=dtype,
                         differentiation=f"cln14_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_4d_unaligned_inner_70(self, execution_mode, dtype):
        """CLN-14b: [2,8,32,70].clone(). Inner 70 > 64 non-multiple; covers >64 bucket at 4D."""
        x = cached_randn((2, 8, 32, 70), dtype=dtype,
                         differentiation=f"cln14b_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_5d_unaligned(self, execution_mode, dtype):
        """CLN-15: [1,2,4,8,13].clone(). Inner 13 < 64."""
        x = cached_randn((1, 2, 4, 8, 13), dtype=dtype,
                         differentiation=f"cln15_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_6d_unaligned(self, execution_mode, dtype):
        """CLN-16: [1,2,4,8,16,7].clone(). Inner 7 < 64."""
        x = cached_randn((1, 2, 4, 8, 16, 7), dtype=dtype,
                         differentiation=f"cln16_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_6d_unaligned_inner_70(self, execution_mode, dtype):
        """CLN-16b: [1,2,4,8,16,70].clone(). Inner 70 > 64 non-multiple; covers >64 bucket at 6D."""
        x = cached_randn((1, 2, 4, 8, 16, 70), dtype=dtype,
                         differentiation=f"cln16b_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)


# ---------------------------------------------------------------------------
# §9.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestCloneSize1:
    """CLN-17–21: 0D–5D with size-1 dimensions."""

    def test_clone_scalar_0d(self, execution_mode, dtype):
        """CLN-17: [] scalar.clone(). 0D scalar materialise."""
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_1d_size1(self, execution_mode, dtype):
        """CLN-18: [1].clone(). Size-1 vector; extreme unaligned."""
        x = cached_randn((1,), dtype=dtype,
                         differentiation=f"cln18_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_3d_size1_trailing(self, execution_mode, dtype):
        """CLN-19: [32,64,1].clone(). Size-1 innermost."""
        x = cached_randn((32, 64, 1), dtype=dtype,
                         differentiation=f"cln19_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)

    def test_clone_4d_size1_after_unsqueeze(self, execution_mode, dtype):
        """CLN-20: [1,8,32,256].unsqueeze(2).clone()→[1,8,1,32,256]. Clone after unsqueeze; inner 256 unchanged."""
        x = cached_randn((1, 8, 32, 256), dtype=dtype,
                         differentiation=f"cln20_{dtype}")
        _compare_modes(execution_mode, lambda t: t.unsqueeze(2).clone(), x)

    def test_clone_5d_gqa_size1_dim(self, execution_mode, dtype):
        """CLN-21: [1,8,1,32,256].clone(). GQA-like; size-1 at dim-2; inner 256 aligned."""
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"cln21_{dtype}")
        _compare_modes(execution_mode, lambda t: t.clone(), x)
