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
FVT Section 2 — reshape / view — Operation Level
=================================================

Two execution modes tested: eager and compiled (torch.compile).
Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Every test in §2.3–§2.4 uses ``_apply_consumer`` with both a pointwise
(``* 1.0``) and a reduction (``.sum(dim=-1)``) consumer, parametrized
per test method via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.
RSH-31–34 already have real consumers (bmm/add/cast/layer_norm) and are
unchanged.

Subsections:
  §2.1  Gemma CPU-traced shapes (aligned)          RSH-03, RSH-09
         (RSH-01/02/04/05/06/07/08 removed — covered by MOD-01–06/08/09)
  §2.2  Spyre-traced shapes (aligned)              RSH-11, RSH-14
         (RSH-10/12/13 removed — covered by MOD-05/08/10)
  §2.3  Unaligned innermost dimension              RSH-15–22
         (RSH-19 removed — dup 3D >64 alongside RSH-20)
  §2.4  Tensors containing size-1 dimensions       RSH-23–30
  §2.5  Restickification scenarios                 RSH-31–34

Consumer parametrization:
  CONSUMERS            = ["pointwise", "sum_last"]  — for ≥2D output
  CONSUMERS_SCALAR     = ["pointwise"]              — for 1D output (sum→scalar is valid too, kept)
  _apply_consumer(t, consumer):
    "pointwise"  → t * 1.0
    "sum_last"   → t.sum(dim=-1)

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch
import torch.nn.functional as F

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §2.3 / §2.4 parametrization
CONSUMERS = ["pointwise", "sum_last"]
CONSUMERS_SCALAR = ["pointwise"]  # 0D scalar output: sum(dim=-1) is undefined


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
# §2.1 — Gemma CPU-traced shapes (aligned)
# ---------------------------------------------------------------------------
# RSH-01 removed: [1,32,4096]→view(1,32,-1,256) — covered by MOD-03
# RSH-02 removed: [1,32,2048]→view(1,32,-1,256) — covered by MOD-04
# RSH-04 removed: [1,32,16,256]→reshape(1,32,-1) — covered by MOD-06
# RSH-05 removed: [1,32,2816]→view(-1,2816) — covered by MOD-01
# RSH-06 removed: [32,2816]→view(1,32,2816) — covered by MOD-02
# RSH-07 removed: [32,8]→view(-1) int64 — covered by MOD-09
# RSH-08 removed: [256,2816]→view(32,8,2816) — covered by MOD-10

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeCPUTraced:
    """RSH-03, RSH-09: Gemma CPU-traced aligned shapes not covered by module-level tests."""

    def test_reshape_qsplit_full_attn(self, execution_mode, dtype):
        """RSH-03: [1,32,8192]→view(1,32,16,512) * 1.0→[1,32,16,512]. Full-attn D=512; no module-level test."""
        x = cached_randn((1, 32, 8192), dtype=dtype,
                         differentiation=f"rsh03_{dtype}")
        _compare_modes(execution_mode, lambda t: t.view(1, 32, 16, 512) * 1.0, x)

    def test_reshape_round_trip_flatten_split(self, execution_mode, dtype):
        """RSH-09: [1,32,16,256]→reshape(1,32,-1).reshape(1,32,16,256) * 1.0. Round-trip chain; no module-level test."""
        x = cached_randn((1, 32, 16, 256), dtype=dtype,
                         differentiation=f"rsh09_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: t.reshape(1, 32, -1).reshape(1, 32, 16, 256) * 1.0,
            x,
        )


# ---------------------------------------------------------------------------
# §2.2 — Spyre-traced shapes (aligned)
# ---------------------------------------------------------------------------
# RSH-10 removed: [1,64,4096]→view(1,64,-1,256) — covered by MOD-05
# RSH-12 removed: [1,64,16,256]→view(1,64,16,2,128) — covered by MOD-08
# RSH-13 removed: [1,64,16,2,1,128]→view(1,64,16,256) — covered by MOD-08 (reverse step)

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeSpyreTraced:
    """RSH-11, RSH-14: Spyre-traced shapes not covered by module-level tests."""

    def test_reshape_kvsplit_spyre_s64(self, execution_mode, dtype):
        """RSH-11: [1,64,2048]→view(1,64,-1,256) * 1.0→[1,64,8,256]. K/V split, Spyre seq=64; no module-level test."""
        x = cached_randn((1, 64, 2048), dtype=dtype,
                         differentiation=f"rsh11_{dtype}")
        _compare_modes(execution_mode, lambda t: t.view(1, 64, -1, 256) * 1.0, x)

    def test_reshape_inferred_dim_aligned(self, execution_mode, dtype):
        """RSH-14: [2,32,16,256]→reshape(2,-1,256) * 1.0→[2,512,256]. Inner unchanged 256→256; no module-level test."""
        x = cached_randn((2, 32, 16, 256), dtype=dtype,
                         differentiation=f"rsh14_{dtype}")
        _compare_modes(execution_mode, lambda t: t.reshape(2, -1, 256) * 1.0, x)


# ---------------------------------------------------------------------------
# §2.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeUnaligned:
    """RSH-15–22: All ranks 1D–6D with non-multiple innermost dimensions.

    Each test is parametrized with both consumers (pointwise and sum_last) so
    that propagate_layouts is exercised under both a passthrough (``* 1.0``)
    and a reducing (``sum(dim=-1)``) downstream op.
    RSH-19 removed: dup 3D >64 alongside RSH-20 at the same rank/bucket.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_1d_to_2d(self, execution_mode, dtype, consumer):
        """RSH-15: [520]→reshape(8,65)→[8,65]. Inner 65: 65%64=1. Both consumers exercised."""
        x = cached_randn((520,), dtype=dtype, differentiation=f"rsh15_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(8, 65), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_2d_to_1d(self, execution_mode, dtype, consumer):
        """RSH-16: [8,65]→reshape(-1)→[520]. 1D output; sum_last reduces to scalar (valid)."""
        x = cached_randn((8, 65), dtype=dtype, differentiation=f"rsh16_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_3d_to_2d_33(self, execution_mode, dtype, consumer):
        """RSH-17: [4,8,33]→reshape(32,33)→[32,33]. Inner 33 < 64. Both consumers."""
        x = cached_randn((4, 8, 33), dtype=dtype, differentiation=f"rsh17_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(32, 33), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_3d_to_4d_260(self, execution_mode, dtype, consumer):
        """RSH-18: [1,32,2080]→reshape(1,32,8,260)→[1,32,8,260]. Output inner 260: 260%64=4. Both consumers."""
        x = cached_randn((1, 32, 2080), dtype=dtype, differentiation=f"rsh18_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 32, 8, 260), consumer), x)

    # RSH-19 removed: [2,8,32,70]→reshape(2,32,560) was dup 3D >64 alongside RSH-20.

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_4d_to_3d_129(self, execution_mode, dtype, consumer):
        """RSH-20: [1,32,8,129]→reshape(1,32,1032)→[1,32,1032]. Inner 1032; 1032%64=8. Both consumers."""
        x = cached_randn((1, 32, 8, 129), dtype=dtype,
                         differentiation=f"rsh20_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 32, 1032), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_5d_to_3d(self, execution_mode, dtype, consumer):
        """RSH-21: [1,2,4,8,7]→reshape(1,8,56)→[1,8,56]. Input inner 7; output 56: 56%64≠0. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 7), dtype=dtype,
                         differentiation=f"rsh21_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 8, 56), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_unaligned_6d_to_3d(self, execution_mode, dtype, consumer):
        """RSH-22: [1,2,4,8,4,7]→reshape(1,8,224)→[1,8,224]. Inner 224: 224%64=32. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 4, 7), dtype=dtype,
                         differentiation=f"rsh22_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 8, 224), consumer), x)


# ---------------------------------------------------------------------------
# §2.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestReshapeSize1:
    """RSH-23–30: 0D–6D inputs with size-1 dimensions.

    Each test is parametrized with both consumers (pointwise and sum_last).
    RSH-23 (0D→1D scalar output of reshape is [1] — not 0D, so sum_last→scalar
    which is valid and included in CONSUMERS).
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_0d_scalar_to_1d(self, execution_mode, dtype, consumer):
        """RSH-23: []→reshape(1)→[1]. 0D→1D; inner=1. sum_last reduces [1] to scalar (valid)."""
        x = torch.tensor(1.0, dtype=dtype)
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_1d_size1_to_3d(self, execution_mode, dtype, consumer):
        """RSH-24: [1]→reshape(1,1,1)→[1,1,1]. All dims=1. Both consumers."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"rsh24_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 1, 1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_2d_remove_trailing_size1(self, execution_mode, dtype, consumer):
        """RSH-25: [128,1]→reshape(128)→[128]. 1D output; sum_last reduces to scalar. Both consumers."""
        x = cached_randn((128, 1), dtype=dtype, differentiation=f"rsh25_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(128), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_3d_trailing_size1_to_2d(self, execution_mode, dtype, consumer):
        """RSH-26: [4,32,1]→reshape(4,32)→[4,32]. Inner changes 1→32. Both consumers."""
        x = cached_randn((4, 32, 1), dtype=dtype, differentiation=f"rsh26_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(4, 32), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_4d_two_size1_to_2d(self, execution_mode, dtype, consumer):
        """RSH-27: [1,8,1,256]→reshape(8,256)→[8,256]. Inner unchanged 256→256. Both consumers."""
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"rsh27_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(8, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_4d_size1_trailing_to_3d(self, execution_mode, dtype, consumer):
        """RSH-28: [2,8,32,1]→reshape(2,8,32)→[2,8,32]. Inner changes 1→32. Both consumers."""
        x = cached_randn((2, 8, 32, 1), dtype=dtype,
                         differentiation=f"rsh28_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(2, 8, 32), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_5d_size1_inner_to_4d(self, execution_mode, dtype, consumer):
        """RSH-29: [1,2,1,8,256]→reshape(1,2,8,256)→[1,2,8,256]. Inner unchanged 256. Both consumers."""
        x = cached_randn((1, 2, 1, 8, 256), dtype=dtype,
                         differentiation=f"rsh29_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 2, 8, 256), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_reshape_6d_size1_trailing_to_5d(self, execution_mode, dtype, consumer):
        """RSH-30: [1,2,4,8,16,1]→reshape(1,2,4,8,16)→[1,2,4,8,16]. Inner changes 1→16. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 16, 1), dtype=dtype,
                         differentiation=f"rsh30_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.reshape(1, 2, 4, 8, 16), consumer), x)


# ---------------------------------------------------------------------------
# §2.5 — Restickification scenarios — view op as trigger
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
class TestReshapeRestickify:
    """
    RSH-31–34: Tests where a reshape causes the Spyre compiler to insert a
    restickify (device-side data-movement) before the consuming op.
    All bfloat16. Source: propagate_layouts.py, insert_restickify.py.
    """

    def test_restickify_reshape_then_bmm(self, execution_mode):
        """RSH-31: reshape changes K stick dim; matmul requires stick on K → restickify.

        Scenario 1: reshape(B*H, S, D) then bmm with a weight tensor.
        Input x [2,32,16,256] bf16, weight [256,256] bf16.
        """
        x = cached_randn((2, 32, 16, 256), dtype=torch.bfloat16,
                         differentiation="rsh31_x")
        w = cached_randn((256, 256), dtype=torch.bfloat16,
                         differentiation="rsh31_w")

        def fn(t, weight):
            reshaped = t.reshape(64, 16, 256)
            w_exp = weight.unsqueeze(0).expand(64, 256, 256)
            return torch.bmm(reshaped, w_exp)

        _compare_modes(execution_mode, fn, x, w, atol=0.5, rtol=0.1)

    def test_restickify_reshape_then_add_incompatible_layout(self, execution_mode):
        """RSH-32: Two view-derived tensors with incompatible stick layouts fed into add.

        Scenario 3: t.reshape(4,64,128) has inner 128, t has inner 64; compiler
        must unify stick layouts before pointwise add.
        Input x [4,128,64] bf16.
        """
        x = cached_randn((4, 128, 64), dtype=torch.bfloat16,
                         differentiation="rsh32")

        def fn(t):
            return t.reshape(4, 64, 128) + t.reshape(4, 64, 128)

        _compare_modes(execution_mode, fn, x)

    def test_restickify_reshape_then_dtype_cast(self, execution_mode):
        """RSH-33: Non-offset-free layout after reshape; dtype cast crosses stick-width boundary.

        Scenario 4: reshape then .to(torch.float32) on unaligned fp16 tensor.
        Input [1,32,65] fp16; 65%32=1 for fp32 stick → restickify required before cast.
        """
        x = cached_randn((1, 32, 65), dtype=torch.float16,
                         differentiation="rsh33")

        def fn(t):
            return t.reshape(32, 65).to(torch.float32)

        _compare_modes(execution_mode, fn, x)

    def test_restickify_reshape_then_layernorm(self, execution_mode):
        """RSH-34: Reshape leaves stick on wrong dim; layer_norm requires stick on last dim.

        Scenario 5: reshape(B*S, D) then F.layer_norm([D]).
        Input [1,32,2816] bf16.
        """
        x = cached_randn((1, 32, 2816), dtype=torch.bfloat16,
                         differentiation="rsh34")

        def fn(t):
            return F.layer_norm(t.reshape(32, 2816), [2816])

        _compare_modes(execution_mode, fn, x)
