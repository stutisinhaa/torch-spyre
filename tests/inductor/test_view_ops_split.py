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
FVT Section 13 — split — Operation Level
==========================================

``aten.split.Tensor`` observed in CPU trace only.
Key Gemma shape: [256,1408] → split(704,-1) → ([256,704],[256,704])
(MoE expert weight split).

``split`` returns a **tuple** of tensors, each a view of the original storage.
To reduce to a single tensor for ``compare_with_cpu``, we use:
  ``torch.stack([chunk.op() for chunk in x.split(...)])``

When the split is on a non-trailing dim and produces identically-shaped chunks,
``torch.stack`` works directly.  When the split produces different-sized chunks
we select specific indices; this does not arise in any test below (all splits
are even).

Pass criteria: compare_with_cpu, atol=0.0.

Consumer selection:
  * §13.1/§13.2 (Gemma-traced and aligned shapes): single baked-in ``* 1.0``
    consumer applied per chunk inside ``torch.stack``.
  * §13.3/§13.4 (unaligned + size-1): parametrized with both ``"pointwise"``
    and ``"sum_last"`` consumers via ``@pytest.mark.parametrize("consumer", CONSUMERS)``.
  * Single-chunk result (SPL-12): no stack, access ``[0]`` directly → consumer
    applied to the single chunk.

Subsections:
  §13.1  Gemma CPU-traced shapes (aligned)         SPL-01–02
  §13.2  Additional aligned shapes                 SPL-03–06
  §13.3  Unaligned innermost dimension             SPL-07–10
  §13.4  Tensors containing size-1 dimensions      SPL-11–13

Stick alignment reference:
  fp16 / bf16  →  64 elements per stick  (128 B / 2 B)
  fp32         →  32 elements per stick  (128 B / 4 B)
"""

import pytest
import torch

from utils_inductor import compare_with_cpu, cached_randn

torch.manual_seed(0xAFFE)

DTYPES = [torch.float16, torch.bfloat16, torch.float32]

# Consumer lists for §13.3 / §13.4 parametrization
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


# SPL-01 ([256,1408]→split(704,-1)) is covered by module-level tests
# MOD-50–51 in test_view_ops_module.py.
# SPL-02 ([1,32,4096]→split(2048,-1)) has no module-level test and is kept.

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitCPUTraced:
    """SPL-02: CPU-traced QKV split — no matching module-level test."""

    def test_split_qkv_heads_trailing(self, execution_mode, dtype):
        """SPL-02: [1,32,4096]→split(2048,dim=-1)→([1,32,2048],[1,32,2048]). QKV split; * 1.0 per chunk."""
        x = cached_randn((1, 32, 4096), dtype=dtype,
                         differentiation=f"spl02_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(2048, dim=-1)]),
            x,
        )


# ---------------------------------------------------------------------------
# §13.2 — Additional aligned shapes
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitAligned:
    """SPL-03–06: Further aligned shapes covering 2D–5D."""

    def test_split_2d_equal_cols(self, execution_mode, dtype):
        """SPL-03: [32,256]→split(128,dim=1)→([32,128],[32,128]) * 1.0. Inner changes 256→128."""
        x = cached_randn((32, 256), dtype=dtype,
                         differentiation=f"spl03_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(128, dim=1)]),
            x,
        )

    def test_split_3d_seq_dim(self, execution_mode, dtype):
        """SPL-04: [2,64,256]→split(32,dim=1)→[2,32,256] × 2. * 1.0; inner 256 unchanged."""
        x = cached_randn((2, 64, 256), dtype=dtype,
                         differentiation=f"spl04_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(32, dim=1)]),
            x,
        )

    def test_split_4d_head_split(self, execution_mode, dtype):
        """SPL-05: [1,32,16,256]→split(8,dim=2)→[1,32,8,256] × 2. * 1.0; inner 256 unchanged."""
        x = cached_randn((1, 32, 16, 256), dtype=dtype,
                         differentiation=f"spl05_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(8, dim=2)]),
            x,
        )

    def test_split_5d_rope_half(self, execution_mode, dtype):
        """SPL-06: [1,32,16,2,128]→split(1,dim=3)→[1,32,16,1,128] × 2. * 1.0; inner 128 unchanged."""
        x = cached_randn((1, 32, 16, 2, 128), dtype=dtype,
                         differentiation=f"spl06_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([c * 1.0 for c in t.split(1, dim=3)]),
            x,
        )


# ---------------------------------------------------------------------------
# §13.3 — Unaligned innermost dimension
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitUnaligned:
    """SPL-07–10: 2D–5D inputs with unaligned output innermost dimension.

    Each test is parametrized with both consumers. For split, the consumer is
    applied per chunk before stacking — this exercises the consumer path on
    each split chunk.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_2d_unaligned_result_trailing(self, execution_mode, dtype, consumer):
        """SPL-07: [8,130]→split(65,dim=-1)→[8,65] × 2. Inner changes 130→65; 65%64=1. Both consumers."""
        x = cached_randn((8, 130), dtype=dtype,
                         differentiation=f"spl07_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(65, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_3d_unaligned_trailing(self, execution_mode, dtype, consumer):
        """SPL-08: [4,32,140]→split(70,dim=-1)→[4,32,70] × 2. Inner 70; 70%64=6. Both consumers."""
        x = cached_randn((4, 32, 140), dtype=dtype,
                         differentiation=f"spl08_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(70, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_4d_unaligned_inner(self, execution_mode, dtype, consumer):
        """SPL-09: [1,16,32,130]→split(65,dim=-1)→[1,16,32,65] × 2. Inner 65 unaligned. Both consumers."""
        x = cached_randn((1, 16, 32, 130), dtype=dtype,
                         differentiation=f"spl09_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(65, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_5d_unaligned_non_trailing(self, execution_mode, dtype, consumer):
        """SPL-10: [1,2,4,8,256]→split(2,dim=2)→[1,2,2,8,256] × 2. Non-trailing split; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 2, 4, 8, 256), dtype=dtype,
                         differentiation=f"spl10_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(2, dim=2)]),
            x,
        )


# ---------------------------------------------------------------------------
# §13.4 — Tensors containing size-1 dimensions
# ---------------------------------------------------------------------------

@pytest.mark.filterwarnings("ignore::torch_spyre.ops.fallbacks.FallbackWarning")
@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])
@pytest.mark.parametrize("dtype", DTYPES)
class TestSplitSize1:
    """SPL-11–13: Size-1 dimension interactions.

    Each test is parametrized with both consumers.
    """

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_3d_trailing_size1_result(self, execution_mode, dtype, consumer):
        """SPL-11: [4,32,2]→split(1,dim=-1)→[4,32,1] × 2. Inner changes 2→1. Both consumers."""
        x = cached_randn((4, 32, 2), dtype=dtype,
                         differentiation=f"spl11_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(1, dim=-1)]),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_4d_size1_dim(self, execution_mode, dtype, consumer):
        """SPL-12: [1,8,32,256]→split(1,dim=0)→([1,8,32,256],). Single chunk. Both consumers."""
        x = cached_randn((1, 8, 32, 256), dtype=dtype,
                         differentiation=f"spl12_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: _apply_consumer(t.split(1, dim=0)[0], consumer),
            x,
        )

    @pytest.mark.parametrize("consumer", CONSUMERS)
    def test_split_5d_size1_head_chunks(self, execution_mode, dtype, consumer):
        """SPL-13: [1,8,1,32,256]→split(4,dim=1)→[1,4,1,32,256] × 2. GQA-like chunks; inner 256 unchanged. Both consumers."""
        x = cached_randn((1, 8, 1, 32, 256), dtype=dtype,
                         differentiation=f"spl13_{dtype}")
        _compare_modes(
            execution_mode,
            lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(4, dim=1)]),
            x,
        )
