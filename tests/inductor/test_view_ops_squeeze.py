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
FVT Section 3 — squeeze — Operation Level
==========================================

squeeze was not observed in either Gemma ATen trace; all tests are standalone.
Pass criteria: compare_with_cpu, atol=0.0 unless noted.

Every test in §3.3 and §3.4 uses ``_apply_consumer`` with both a pointwise
(``* 1.0``) and a reduction (``.sum(dim=-1)``) consumer, parametrized per
test method via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.
0D scalar output tests (SQZ-18/19) use CONSUMERS_SCALAR (pointwise only).

Subsections:
  §3.1  No-op cases (dim is not size-1)            SQZ-01–03
  §3.2  Aligned result after squeeze               SQZ-04–11
  §3.3  Unaligned result after squeeze             SQZ-12–17
  §3.4  Tensors containing size-1 dimensions       SQZ-18–25

Consumer parametrization:
  CONSUMERS        = ["pointwise", "sum_last"]  — for ≥1D output
  CONSUMERS_SCALAR = ["pointwise"]              — for 0D scalar output

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §3.3 / §3.4 parametrization
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
# §3.1 — No-op cases (dim is not size-1 — verifies no-op path)
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeNoop:
    """SQZ-01–03: squeeze on a non-size-1 dim — verifies the no-op path."""

    def test_squeeze_noop_1d_aligned(self, execution_mode, dtype):
        """SQZ-01: [256]→squeeze(0) * 1.0→[256]. No-op; inner 256 unchanged."""
        x = cached_randn((256,), dtype=dtype, differentiation=f"sqz01_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_noop_1d_unaligned(self, execution_mode, dtype):
        """SQZ-02: [63]→squeeze(0) * 1.0→[63]. No-op; inner 63 unchanged."""
        x = cached_randn((63,), dtype=dtype, differentiation=f"sqz02_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_noop_4d_no_singletons(self, execution_mode, dtype):
        """SQZ-03: [2,16,32,256]→squeeze() * 1.0→[2,16,32,256]. No-op; inner 256 unchanged."""
        x = cached_randn((2, 16, 32, 256), dtype=dtype,
                         differentiation=f"sqz03_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze() * 1.0, x)


# ---------------------------------------------------------------------------
# §3.2 — Aligned result after squeeze
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeAligned:
    """SQZ-04–11: 1D–6D inputs producing aligned innermost dim after squeeze."""

    def test_squeeze_2d_leading_to_1d_aligned(self, execution_mode, dtype):
        """SQZ-04: [1,256]→squeeze(0) * 1.0→[256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((1, 256), dtype=dtype, differentiation=f"sqz04_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_3d_batch_to_2d_aligned(self, execution_mode, dtype):
        """SQZ-05: [1,32,2816]→squeeze(0) * 1.0→[32,2816]. Non-trailing squeeze; inner 2816 unchanged."""
        x = cached_randn((1, 32, 2816), dtype=dtype,
                         differentiation=f"sqz05_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)

    def test_squeeze_3d_mid_decode_aligned(self, execution_mode, dtype):
        """SQZ-06: [2,1,256]→squeeze(1) * 1.0→[2,256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((2, 1, 256), dtype=dtype, differentiation=f"sqz06_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(1) * 1.0, x)

    def test_squeeze_4d_decode_seq_aligned(self, execution_mode, dtype):
        """SQZ-07: [1,1,16,256]→squeeze(1) * 1.0→[1,16,256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((1, 1, 16, 256), dtype=dtype,
                         differentiation=f"sqz07_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(1) * 1.0, x)

    def test_squeeze_4d_tuple_dims_aligned(self, execution_mode, dtype):
        """SQZ-08: [1,16,1,256]→squeeze((0,2)) * 1.0→[16,256]. Neither dim is trailing; inner 256 unchanged."""
        x = cached_randn((1, 16, 1, 256), dtype=dtype,
                         differentiation=f"sqz08_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze((0, 2)) * 1.0, x)

    def test_squeeze_4d_trailing_to_3d_aligned(self, execution_mode, dtype):
        """SQZ-09: [32,8,64,1]→squeeze(-1) * 1.0→[32,8,64]. Trailing squeeze; inner changes 1→64."""
        x = cached_randn((32, 8, 64, 1), dtype=dtype,
                         differentiation=f"sqz09_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(-1) * 1.0, x)

    def test_squeeze_5d_all_singletons_aligned(self, execution_mode, dtype):
        """SQZ-10: [1,1,32,1,256]→squeeze() * 1.0→[32,256]. Inner 256 unchanged."""
        x = cached_randn((1, 1, 32, 1, 256), dtype=dtype,
                         differentiation=f"sqz10_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze() * 1.0, x)

    def test_squeeze_6d_leading_to_5d_aligned(self, execution_mode, dtype):
        """SQZ-11: [1,2,4,1,8,256]→squeeze(0) * 1.0→[2,4,1,8,256]. Non-trailing squeeze; inner 256 unchanged."""
        x = cached_randn((1, 2, 4, 1, 8, 256), dtype=dtype,
                         differentiation=f"sqz11_{dtype}")
        _compare_modes(execution_mode, lambda t: t.squeeze(0) * 1.0, x)


# ---------------------------------------------------------------------------
# §3.3 — Unaligned result after squeeze
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeUnaligned:
    """SQZ-12–17: 1D–6D inputs producing unaligned innermost dim after squeeze.

    Each test is parametrized with both consumers so that propagate_layouts is
    exercised under both a passthrough (``* 1.0``) and a reducing
    (``sum(dim=-1)``) downstream op.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_2d_leading_unaligned_33(self, execution_mode, dtype, consumer):
        """SQZ-12: [1,33]→squeeze(0)→[33]. Non-trailing squeeze; inner 33 unchanged. Both consumers."""
        x = cached_randn((1, 33), dtype=dtype, differentiation=f"sqz12_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_3d_mid_unaligned_70(self, execution_mode, dtype, consumer):
        """SQZ-13: [2,1,70]→squeeze(1)→[2,70]. Non-trailing squeeze; inner 70 unchanged. Both consumers."""
        x = cached_randn((2, 1, 70), dtype=dtype, differentiation=f"sqz13_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_4d_decode_unaligned_65(self, execution_mode, dtype, consumer):
        """SQZ-14: [1,1,16,65]→squeeze(1)→[1,16,65]. Non-trailing squeeze; inner 65 unchanged. Both consumers."""
        x = cached_randn((1, 1, 16, 65), dtype=dtype,
                         differentiation=f"sqz14_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_4d_tuple_unaligned_33(self, execution_mode, dtype, consumer):
        """SQZ-15: [1,16,1,33]→squeeze((0,2))→[16,33]. Neither dim trailing; inner 33 unchanged. Both consumers."""
        x = cached_randn((1, 16, 1, 33), dtype=dtype,
                         differentiation=f"sqz15_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze((0, 2)), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_5d_all_singletons_unaligned(self, execution_mode, dtype, consumer):
        """SQZ-16: [1,1,32,1,30]→squeeze()→[32,30]. Inner 30 unchanged. Both consumers."""
        x = cached_randn((1, 1, 32, 1, 30), dtype=dtype,
                         differentiation=f"sqz16_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_6d_leading_unaligned_13(self, execution_mode, dtype, consumer):
        """SQZ-17: [1,2,4,1,8,13]→squeeze(0)→[2,4,1,8,13]. Non-trailing squeeze; inner 13 unchanged. Both consumers."""
        x = cached_randn((1, 2, 4, 1, 8, 13), dtype=dtype,
                         differentiation=f"sqz17_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(0), consumer), x)


# ---------------------------------------------------------------------------
# §3.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSqueezeSize1:
    """SQZ-18–25: 0D–6D edge cases involving size-1 dims, scalars, and multi-squeeze chains.

    Tests producing 0D scalar output (SQZ-18, SQZ-19) use
    CONSUMERS_SCALAR (pointwise only) since sum(dim=-1) on a 0D tensor is
    undefined.  All other tests use CONSUMERS (both consumers).
    """

    @pytest.mark.parametrize("consumer", CONSUMERS_SCALAR)
    def test_squeeze_1d_size1_to_scalar(self, execution_mode, dtype, consumer):
        """SQZ-18: [1]→squeeze(0)→[] scalar. 0D output; pointwise only."""
        x = cached_randn((1,), dtype=dtype, differentiation=f"sqz18_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(0), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS_SCALAR)
    def test_squeeze_2d_all_size1_to_scalar(self, execution_mode, dtype, consumer):
        """SQZ-19: [1,1]→squeeze()→[] scalar. 0D output; pointwise only."""
        x = cached_randn((1, 1), dtype=dtype, differentiation=f"sqz19_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_2d_trailing_size1(self, execution_mode, dtype, consumer):
        """SQZ-20: [128,1]→squeeze(-1)→[128]. Trailing squeeze; inner changes 1→128. Both consumers."""
        x = cached_randn((128, 1), dtype=dtype, differentiation=f"sqz20_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(-1), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_3d_all_size1_to_1d(self, execution_mode, dtype, consumer):
        """SQZ-21: [1,1,256]→squeeze()→[256]. Inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 1, 256), dtype=dtype,
                         differentiation=f"sqz21_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_4d_mixed_size1_to_1d(self, execution_mode, dtype, consumer):
        """SQZ-22: [1,1,32,1]→squeeze()→[32]. Three size-1 dims removed; inner changes 1→32."""
        x = cached_randn((1, 1, 32, 1), dtype=dtype,
                         differentiation=f"sqz22_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_5d_size1_inner(self, execution_mode, dtype, consumer):
        """SQZ-23: [2,1,32,1,256]→squeeze()→[2,32,256]. Inner 256 unchanged. Both consumers."""
        x = cached_randn((2, 1, 32, 1, 256), dtype=dtype,
                         differentiation=f"sqz23_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_6d_size1_to_3d(self, execution_mode, dtype, consumer):
        """SQZ-24: [1,1,4,1,8,256]→squeeze()→[4,8,256]. Inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 1, 4, 1, 8, 256), dtype=dtype,
                         differentiation=f"sqz24_{dtype}")
        _compare_modes(execution_mode, lambda t: _apply_consumer(t.squeeze(), consumer), x)

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_squeeze_chain_multi_then_reshape(self, execution_mode, dtype, consumer):
        """SQZ-25: [1,8,1,256]→squeeze(0).squeeze(1).reshape(-1)→[2048]. Both consumers."""
        x = cached_randn((1, 8, 1, 256), dtype=dtype,
                         differentiation=f"sqz25_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.squeeze(0).squeeze(1).reshape(-1), consumer),
            x,
        )
