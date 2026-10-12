# Functional Verification Test Plan — Remaining View Operations
## `transpose` · `t` · `permute` · `clone` · `cat` · `select` · `narrow/slice` · `split`

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Remaining View Operations on torch-spyre |
| **Version** | 1.2 |
| **Status** | Under Review |
| **Date** | 2025-07-15 |
| **Author** | Stuti Sinha |
| **Reviewers** | Kanaka Pathivada, Meena Soundararajan, Albin Joy |
| **Scope** | Text-only Gemma-4-26B-A4B-it inference path. Static concrete dimensions 0D–6D. Default compiler env-var values only. |
| **Sources** | `gemma4_26b_module_operator_report_v2.html` (CPU trace, seq=32), `gemma4_fx_module_trace_v2.html` (Spyre compile trace, seq=64) |
| **Companion document** | `view-ops-priority-fvt-v5.0.md` — covers `reshape`, `squeeze`, `unsqueeze`, `expand` |

---

## 1. Scope & Guidelines

### 1.1 Ops covered

| # | Op | ATen lowering | CPU trace | Spyre trace |
|---|---|---|---|---|
| 1 | `torch.transpose` / `Tensor.transpose_` | `aten.transpose.int` | ✅ | — |
| 2 | `torch.t` | `aten.t.default` | ✅ | — (lowered to permute) |
| 3 | `torch.permute` | `aten.permute.default` | — | ✅ |
| 4 | `torch.clone` | `aten.clone.default` | ✅ | ✅ |
| 5 | `torch.cat` | `aten.cat.default` | ✅ | — |
| 6 | `torch.select` | `aten.select.int` | ✅ | ✅ |
| 7 | `torch.narrow` / slice | `aten.slice.Tensor` | ✅ | ✅ |
| 8 | `torch.split` | `aten.split.Tensor` | ✅ | — |

### 1.2 Stick alignment reference

| Dtype | Bytes/elem | Elements per stick |
|---|---|---|
| `float16` / `bfloat16` | 2 B | **64** |
| `float32` | 4 B | **32** |

**Aligned:** innermost dim divisible by 64 (fp16/bf16) or 32 (fp32).
**Unaligned — two sub-cases:**
- *Below one stick:* inner dim < 64 (fp16/bf16) or < 32 (fp32). e.g. 1, 7, 13, 30.
- *Not divisible:* inner dim ≥ stick width but not a clean multiple. e.g. 33, 65, 70, 129.

**Size-1 innermost:** extreme unaligned case — 1 element per stick, 63 padding slots (fp16/bf16).

### 1.3 Consumer op requirement

All view ops lower to `ReinterpretView` / `NopKernelSchedulerNode` in the Spyre backend. Without a downstream `ComputedBuffer`, `propagate_layouts.compute_layouts` is never invoked and stick-layout correctness is invisible.

Every §x.3 (Unaligned) and §x.4 (Size-1) test uses **both** consumers, parametrized per method:

```python
@pytest.mark.parametrize("consumer", CONSUMERS)
def test_xxx(self, execution_mode, dtype, consumer):
    ...
    _compare_modes(execution_mode, lambda t: _apply_consumer(view_result, consumer), x)
```

| Consumer key | Code | Output |
|---|---|---|
| `"pointwise"` | `t * 1.0` | Same shape as view output |
| `"sum_last"` | `t.sum(dim=-1)` | Rank reduced by 1 |

`CONSUMERS = ["pointwise", "sum_last"]` — used for all ≥1D outputs.

**Effective test combinations per §x.3/§x.4 method:** 2 (execution_mode) × 3 (dtype) × 2 (consumer) = **12**.

**Exceptions:**
- `clone` — no consumer parametrization. clone materialises a new buffer and is its own consumer. `CLN-14b` and `CLN-16b` added to cover the >64 non-multiple bucket at 4D and 6D respectively.
- Tests in §x.1/§x.2 (Gemma-traced aligned shapes) — single consumer only; not parametrized.
- `split` — consumer is applied per chunk inside the `torch.stack(...)` expression.

### 1.4 Gemma model architecture variants

| Variant | H | KV | D | GQA factor |
|---|---|---|---|---|
| **Sliding-attention** (25 layers) | 16 | 8 | 256 | 2× |
| **Full-attention** (5 layers) | 16 | 2 | 512 | 8× |

### 1.5 Test structure

Each section has four subsections:
- **§x.1 Gemma CPU-traced shapes** — exact shapes from `gemma4_26b_module_operator_report_v2.html`
- **§x.2 Gemma Spyre-traced shapes** — exact shapes from `gemma4_fx_module_trace_v2.html` (omitted when op not observed)
- **§x.3 Unaligned innermost dimension** — 0D–6D inputs with non-multiple innermost dims
- **§x.4 Tensors containing size-1 dimensions** — size-1 innermost and size-1 dims in other positions

### 1.6 Deduplication

Section 1 of the companion document covers **MOD-22–51** for all 8 remaining ops (see `test_view_ops_module.py`). Op-level §x.1/§x.2 tests whose base view op + input shape are covered by a module-level test have been removed from the per-op test files to avoid double-counting. The covering MOD-xx test is the single source of truth for those shapes.

| Op file | Removed | Covered by |
|---|---|---|
| `test_view_ops_transpose.py` | TRP-01–08 | MOD-22–29 |
| `test_view_ops_t.py` | T2D-01–04 | MOD-27–30 |
| `test_view_ops_permute.py` | PRM-02/04/05/06 | MOD-31–34 |
| `test_view_ops_clone.py` | CLN-01–05 | MOD-18/19/35–37 |
| `test_view_ops_cat.py` | CAT-01/03/04/05 | MOD-39–42 |
| `test_view_ops_select.py` | SEL-01/02 | MOD-43–44 |
| `test_view_ops_narrow.py` | SLC-01–04 | MOD-45–48 |
| `test_view_ops_split.py` | SPL-01 | MOD-50–51 |

**All op-level tests run three dtypes:** `float16`, `bfloat16`, `float32`.

---

## Section 6 — `transpose` — Operation Level

`aten.transpose.int` observed in CPU trace: head-axis swap `[B,S,H,D]→[B,H,S,D]` and key transpose `[B,H,S,D]→[B,H,D,S]` for attention; MoE expert GEMM transpose.

**Pass criteria:** `compare_with_cpu`, `atol=0.0` unless noted.

### 6.1 Gemma CPU-traced shapes (aligned)

TRP-01–08 are covered by MOD-22–29 and are not repeated here.

### 6.2 Chain: transpose → matmul (Gemma attention pattern)

TRP-08 is covered by MOD-29 and is not repeated here.

### 6.3 Unaligned innermost dimension

| ID | Test Name | Input | Op | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `TRP-09` | `test_transpose_2d_unaligned_inner_swap` | `[65,256]` (all dtypes) | `transpose(0,1)` | `[256,65]` | pointwise + sum_last | inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(0,1), consumer), x)` |
| `TRP-10` | `test_transpose_3d_last_two_unaligned` | `[4,32,70]` (all dtypes) | `transpose(1,2)` | `[4,70,32]` | pointwise + sum_last | input inner 70 unaligned | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(1,2), consumer), x)` |
| `TRP-11` | `test_transpose_4d_key_unaligned_65` | `[1,16,32,65]` (all dtypes) | `transpose(2,3)` | `[1,16,65,32]` | pointwise + sum_last | inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(2,3), consumer), x)` |
| `TRP-12` | `test_transpose_4d_head_swap_unaligned_inner` | `[2,32,8,70]` (all dtypes) | `transpose(1,2)` | `[2,8,32,70]` | pointwise + sum_last | inner 70 unaligned | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(1,2), consumer), x)` |
| `TRP-13` | `test_transpose_5d_unaligned` | `[1,8,1,32,65]` (all dtypes) | `transpose(1,3)` | `[1,32,1,8,65]` | pointwise + sum_last | inner 65 | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(1,3), consumer), x)` |
| `TRP-14` | `test_transpose_6d_unaligned` | `[1,2,4,8,16,13]` (all dtypes) | `transpose(1,4)` | `[1,16,4,8,2,13]` | pointwise + sum_last | inner 13 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(1,4), consumer), x)` |

### 6.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Op | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `TRP-15` | `test_transpose_2d_size1_trailing` | `[256,1]` (all dtypes) | `transpose(0,1)` | `[1,256]` | pointwise + sum_last | Trailing size-1 moves to leading | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(0,1), consumer), x)` |
| `TRP-16` | `test_transpose_3d_size1_middle` | `[4,1,256]` (all dtypes) | `transpose(0,1)` | `[1,4,256]` | pointwise + sum_last | Size-1 at middle | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(0,1), consumer), x)` |
| `TRP-17` | `test_transpose_4d_size1_gqa_expand` | `[1,8,1,256]` (all dtypes) | `transpose(1,2)` | `[1,1,8,256]` | pointwise + sum_last | GQA-like; size-1 in dim-2 | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(1,2), consumer), x)` |
| `TRP-18` | `test_transpose_5d_size1_to_trailing` | `[1,8,32,1,256]` (all dtypes) | `transpose(3,4)` | `[1,8,32,256,1]` | pointwise + sum_last | Size-1 moved to innermost | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(3,4), consumer), x)` |
| `TRP-19` | `test_transpose_6d_all_size1_except_inner` | `[1,1,1,1,1,256]` (all dtypes) | `transpose(0,4)` | `[1,1,1,1,1,256]` | pointwise + sum_last | No-op in practice | `compare_with_cpu(lambda t: _apply_consumer(t.transpose(0,4), consumer), x)` |

---

## Section 7 — `t` (2D transpose) — Operation Level

`aten.t.default` is a special case of transpose for 2D matrices: swaps dims 0 and 1. Observed in CPU trace only — weight matrix transpose before `mm`. In the Spyre compile trace, `aten.t.default` is replaced by `aten.permute.default`.

**Consumer parametrization for §7.2/§7.3:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `* 1.0` (pointwise) and `.sum(dim=-1)` (sum_last) are applied.

**Note on 0D–1D:** `t()` on a 0D or 1D tensor is a no-op in PyTorch (returns a view of the same storage). These cases are covered for completeness.

**Pass criteria:** `compare_with_cpu`, `atol=0.0` unless noted.

### 7.1 Gemma CPU-traced shapes (aligned)

T2D-01–04 are covered by MOD-27–30 and are not repeated here.

### 7.2 Unaligned innermost dimension

| ID | Test Name | Input | Op | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `T2D-05` | `test_t_1d_noop` | `[256]` (all dtypes) | `.t()` | `[256]` | pointwise + sum_last | 1D `.t()` is identity | `compare_with_cpu(lambda t: _apply_consumer(t.t(), consumer), x)` |
| `T2D-06` | `test_t_2d_unaligned_inner_to_outer` | `[65,256]` (all dtypes) | `.t()` | `[256,65]` | pointwise + sum_last | input inner 256 → new inner 65 | `compare_with_cpu(lambda t: _apply_consumer(t.t(), consumer), x)` |
| `T2D-07` | `test_t_2d_both_unaligned` | `[33,70]` (all dtypes) | `.t()` | `[70,33]` | pointwise + sum_last | both dims unaligned | `compare_with_cpu(lambda t: _apply_consumer(t.t(), consumer), x)` |
| `T2D-08` | `test_t_2d_aligned_swap` | `[128,64]` (all dtypes) | `.t()` | `[64,128]` | pointwise + sum_last | both aligned; inner changes | `compare_with_cpu(lambda t: _apply_consumer(t.t(), consumer), x)` |

### 7.3 Tensors containing size-1 dimensions

| ID | Test Name | Input | Op | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `T2D-09` | `test_t_2d_size1_trailing` | `[256,1]` (all dtypes) | `.t()` | `[1,256]` | pointwise + sum_last | Size-1 trailing → leading | `compare_with_cpu(lambda t: _apply_consumer(t.t(), consumer), x)` |
| `T2D-10` | `test_t_2d_size1_leading` | `[1,256]` (all dtypes) | `.t()` | `[256,1]` | pointwise + sum_last | Size-1 leading → trailing | `compare_with_cpu(lambda t: _apply_consumer(t.t(), consumer), x)` |
| `T2D-11` | `test_t_2d_both_size1` | `[1,1]` (all dtypes) | `.t()` | `[1,1]` | pointwise + sum_last | All dims=1 | `compare_with_cpu(lambda t: _apply_consumer(t.t(), consumer), x)` |

---

## Section 8 — `permute` — Operation Level

`aten.permute.default` observed in Spyre trace only. In the Spyre compile trace, `aten.t.default` (CPU) is decomposed to `aten.permute.default`. Key shapes: `[128,2816]→[2816,128]` (weight transpose), `[1,64,8,256]→[1,8,64,256]` (head-axis swap), `[1,64,16,256]→[1,16,64,256]`.

**Consumer parametrization for §8.2/§8.3:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `* 1.0` (pointwise) and `.sum(dim=-1)` (sum_last) are applied.

**Pass criteria:** `compare_with_cpu`, `atol=0.0` unless noted.

### 8.1 Gemma Spyre-traced shapes (aligned)

PRM-02/04/05/06 are covered by MOD-31–34 and are not repeated here. PRM-01/03 are kept (unique shapes not in module tests).

| ID | Test Name | Input | Op | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `PRM-01` | `test_permute_weight_transpose_2d` | `[128,2816]` (all dtypes) | `permute(1,0)` | `[2816,128]` | `* 1.0` | `compare_with_cpu(lambda t: t.permute(1,0) * 1.0, x)` |
| `PRM-03` | `test_permute_weight_transpose_kv` | `[2048,2816]` (all dtypes) | `permute(1,0)` | `[2816,2048]` | `* 1.0` | `compare_with_cpu(lambda t: t.permute(1,0) * 1.0, x)` |

### 8.2 Unaligned innermost dimension

| ID | Test Name | Input | Op | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `PRM-07` | `test_permute_2d_unaligned_new_inner` | `[65,256]` (all dtypes) | `permute(1,0)` | `[256,65]` | pointwise + sum_last | inner changes: 256→65 | `compare_with_cpu(lambda t: _apply_consumer(t.permute(1,0), consumer), x)` |
| `PRM-08` | `test_permute_3d_last_to_first` | `[256,32,70]` (all dtypes) | `permute(2,1,0)` | `[70,32,256]` | pointwise + sum_last | input inner 70 unaligned | `compare_with_cpu(lambda t: _apply_consumer(t.permute(2,1,0), consumer), x)` |
| `PRM-09` | `test_permute_4d_unaligned_inner_swap` | `[1,16,32,65]` (all dtypes) | `permute(0,1,3,2)` | `[1,16,65,32]` | pointwise + sum_last | inner 65 unaligned | `compare_with_cpu(lambda t: _apply_consumer(t.permute(0,1,3,2), consumer), x)` |
| `PRM-10` | `test_permute_5d_cycle` | `[1,2,4,8,13]` (all dtypes) | `permute(0,4,2,3,1)` | `[1,13,4,8,2]` | pointwise + sum_last | input inner 13 < 64; new inner 2 | `compare_with_cpu(lambda t: _apply_consumer(t.permute(0,4,2,3,1), consumer), x)` |
| `PRM-11` | `test_permute_6d_unaligned` | `[1,2,4,8,16,13]` (all dtypes) | `permute(0,5,2,3,4,1)` | `[1,13,4,8,16,2]` | pointwise + sum_last | input inner 13 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.permute(0,5,2,3,4,1), consumer), x)` |

### 8.3 Tensors containing size-1 dimensions

| ID | Test Name | Input | Op | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `PRM-12` | `test_permute_3d_size1_cycle` | `[1,256,64]` (all dtypes) | `permute(2,0,1)` | `[64,1,256]` | pointwise + sum_last | Size-1 moves to middle | `compare_with_cpu(lambda t: _apply_consumer(t.permute(2,0,1), consumer), x)` |
| `PRM-13` | `test_permute_4d_size1_to_trailing` | `[1,8,32,256]` (all dtypes) | `permute(1,2,3,0)` | `[8,32,256,1]` | pointwise + sum_last | Size-1 moved to innermost | `compare_with_cpu(lambda t: _apply_consumer(t.permute(1,2,3,0), consumer), x)` |
| `PRM-14` | `test_permute_5d_size1_leading_and_inner` | `[1,8,1,32,256]` (all dtypes) | `permute(0,1,4,2,3)` | `[1,8,256,1,32]` | pointwise + sum_last | Two size-1 dims | `compare_with_cpu(lambda t: _apply_consumer(t.permute(0,1,4,2,3), consumer), x)` |
| `PRM-15` | `test_permute_6d_size1_to_inner` | `[2,4,8,16,256,1]` (all dtypes) | `permute(0,1,2,5,3,4)` | `[2,4,8,1,16,256]` | pointwise + sum_last | Size-1 from trailing to middle | `compare_with_cpu(lambda t: _apply_consumer(t.permute(0,1,2,5,3,4), consumer), x)` |

---

## Section 9 — `clone` — Operation Level

`aten.clone.default` observed in both traces. In the Spyre backend, `clone` has its own path in `propagate_layouts._clone_layout`: it materialises a new contiguous buffer. When the input has no offset in the stick dimension, clone is an identity; otherwise it becomes a restickify. **Clone is its own consumer** — no `* 1.0` or `sum(dim=-1)` needed. The test chain is simply `clone()` or `op→clone()`.

Key CPU trace shapes: GQA materialise `[1,8,2,32,256]`, attention output `[1,32,16,256]`, full-attn `[1,2,8,32,512]`.
Key Spyre trace shapes: `[1,8,64]`, `[1,8,2816]`, `[1,16,64,256]`, `[1,8,64,256]`, `[1,64,16,256]`, `[1,16,64,512]`, `[1,2,64,512]`.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`.

### 9.1 Gemma CPU-traced shapes (aligned)

CLN-01–05 are covered by MOD-18/19/35–37 and are not repeated here.

### 9.2 Gemma Spyre-traced shapes (aligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `CLN-06` | `test_clone_moe_token_3d` | `[1,8,64]` (all dtypes) | `clone()` | `[1,8,64]` | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-07` | `test_clone_moe_hidden_3d` | `[1,8,2816]` (all dtypes) | `clone()` | `[1,8,2816]` | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-08` | `test_clone_attn_output_4d_s64` | `[1,16,64,256]` (all dtypes) | `clone()` | `[1,16,64,256]` | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-09` | `test_clone_attn_output_full_attn_s64` | `[1,16,64,512]` (all dtypes) | `clone()` | `[1,16,64,512]` | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-10` | `test_clone_after_permute_head_swap` | `[1,64,16,256]` (all dtypes) | `permute(0,2,1,3).clone()` | `[1,16,64,256]` | `compare_with_cpu(lambda t: t.permute(0,2,1,3).clone(), x)` — clone of non-contiguous permuted tensor |

### 9.3 Unaligned innermost dimension

| ID | Test Name | Input | Chain | Output | Stick note | Test Steps |
|---|---|---|---|---|---|---|
| `CLN-11` | `test_clone_1d_unaligned` | `[63]` (all dtypes) | `clone()` | `[63]` | inner 63 < 64 | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-12` | `test_clone_2d_unaligned_inner` | `[32,70]` (all dtypes) | `clone()` | `[32,70]` | inner 70: 70%64=6 | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-13` | `test_clone_3d_unaligned_after_transpose` | `[4,256,65]` (all dtypes) | `transpose(1,2).clone()` | `[4,65,256]` — clone of non-contiguous | clone after transpose; output inner 256 | `compare_with_cpu(lambda t: t.transpose(1,2).clone(), x)` |
| `CLN-14` | `test_clone_4d_unaligned_inner_33` | `[2,8,32,33]` (all dtypes) | `clone()` | `[2,8,32,33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-15` | `test_clone_5d_unaligned` | `[1,2,4,8,13]` (all dtypes) | `clone()` | `[1,2,4,8,13]` | inner 13 < 64 | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-16` | `test_clone_6d_unaligned` | `[1,2,4,8,16,7]` (all dtypes) | `clone()` | `[1,2,4,8,16,7]` | inner 7 < 64 | `compare_with_cpu(lambda t: t.clone(), x)` |

### 9.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Chain | Output | Notes | Test Steps |
|---|---|---|---|---|---|---|
| `CLN-17` | `test_clone_scalar_0d` | `[]` (all dtypes) | `clone()` | `[]` | 0D scalar | `x=torch.tensor(1.0,dtype=dtype)` → `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-18` | `test_clone_1d_size1` | `[1]` (all dtypes) | `clone()` | `[1]` | Size-1 vector | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-19` | `test_clone_3d_size1_trailing` | `[32,64,1]` (all dtypes) | `clone()` | `[32,64,1]` | Size-1 innermost | `compare_with_cpu(lambda t: t.clone(), x)` |
| `CLN-20` | `test_clone_4d_size1_after_unsqueeze` | `[1,8,32,256]` (all dtypes) | `unsqueeze(2).clone()` | `[1,8,1,32,256]` | Clone after unsqueeze; inner 256 unchanged | `compare_with_cpu(lambda t: t.unsqueeze(2).clone(), x)` |
| `CLN-21` | `test_clone_5d_gqa_size1_dim` | `[1,8,1,32,256]` (all dtypes) | `clone()` | `[1,8,1,32,256]` | GQA-like; size-1 at dim-2 | `compare_with_cpu(lambda t: t.clone(), x)` |

---

## Section 10 — `cat` — Operation Level

`aten.cat.default` observed in CPU trace only. Key patterns: RoPE half-rotation concat `[B,S,H,D/2]×2 → [B,S,H,D]` (dim=-1), position id concat `[1,1]+[1,S] → [1,S+1]` (dim=-1), KV-cache append `[]→KV_cached` (dim=-2).

**Consumer parametrization for §10.2/§10.3:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `* 1.0` (pointwise) and `.sum(dim=-1)` (sum_last) are applied.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`.

### 10.1 Gemma CPU-traced shapes (aligned)

CAT-01/03/04/05 are covered by MOD-39–42 and are not repeated here. CAT-02 is kept (no module-level test for KV half-cat shape).

| ID | Test Name | Inputs | Op | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `CAT-02` | `test_cat_rope_halves_kv_sliding` | `[1,32,8,128]` × 2 (all dtypes) | `cat([a,b], dim=-1)` | `[1,32,8,256]` | `* 1.0` | `compare_with_cpu(lambda a,b: torch.cat([a,b], dim=-1) * 1.0, a, b)` |

### 10.2 Unaligned innermost dimension

| ID | Test Name | Inputs | Op | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `CAT-06` | `test_cat_1d_unaligned_halves` | `[33]` + `[33]` (all dtypes) | `cat([a,b], dim=0)` | `[66]` | pointwise + sum_last | both inputs unaligned | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b]), consumer), a, b)` |
| `CAT-07` | `test_cat_2d_trailing_unaligned` | `[32,33]` + `[32,33]` (all dtypes) | `cat([a,b], dim=-1)` | `[32,66]` | pointwise + sum_last | output inner 66 unaligned | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=-1), consumer), a, b)` |
| `CAT-08` | `test_cat_3d_seq_dim_unaligned` | `[1,33,256]` + `[1,33,256]` (all dtypes) | `cat([a,b], dim=1)` | `[1,66,256]` | pointwise + sum_last | seq dim 66; inner aligned | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=1), consumer), a, b)` |
| `CAT-09` | `test_cat_4d_unaligned_inner_halves` | `[1,16,32,33]` × 2 (all dtypes) | `cat([a,b], dim=-1)` | `[1,16,32,66]` | pointwise + sum_last | unaligned inputs → unaligned output | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=-1), consumer), a, b)` |
| `CAT-10` | `test_cat_5d_seq_unaligned` | `[1,8,1,33,256]` × 2 (all dtypes) | `cat([a,b], dim=3)` | `[1,8,1,66,256]` | pointwise + sum_last | seq dim unaligned | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=3), consumer), a, b)` |
| `CAT-11` | `test_cat_6d_inner_unaligned` | `[1,2,4,8,16,13]` × 2 (all dtypes) | `cat([a,b], dim=-1)` | `[1,2,4,8,16,26]` | pointwise + sum_last | 6D; inner 26 < 64 | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=-1), consumer), a, b)` |

### 10.3 Tensors containing size-1 dimensions

| ID | Test Name | Inputs | Op | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `CAT-12` | `test_cat_3d_size1_seq_expand` | `[1,1,256]` + `[1,32,256]` (all dtypes) | `cat([a,b], dim=1)` | `[1,33,256]` | pointwise + sum_last | Size-1 seq prepended | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=1), consumer), a, b)` |
| `CAT-13` | `test_cat_4d_size1_trailing_halves` | `[32,8,16,1]` × 2 (all dtypes) | `cat([a,b], dim=-1)` | `[32,8,16,2]` | pointwise + sum_last | Two size-1 halves | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=-1), consumer), a, b)` |
| `CAT-14` | `test_cat_5d_gqa_kv_cache` | `[1,8,1,32,256]` + `[1,8,1,32,256]` (all dtypes) | `cat([a,b], dim=2)` | `[1,8,2,32,256]` | pointwise + sum_last | GQA-like cache | `compare_with_cpu(lambda a,b: _apply_consumer(torch.cat([a,b], dim=2), consumer), a, b)` |

---

## Section 11 — `select` — Operation Level

`aten.select.int` observed in both traces. In CPU trace: `[1,32]→select(1,-1)→[1]` (last token position). In Spyre trace: `[1,8,32]→select(2,0)→[1,8]` (first element along routing axis; int32). `select` removes exactly one dimension.

**Consumer parametrization for §11.3/§11.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `* 1.0` (pointwise) and `.sum(dim=-1)` (sum_last) are applied. int32/int64 tests in §11.1 have no consumer. SEL-03 (0D scalar output) is in §11.2 with a baked-in `* 1.0` consumer — not parametrized.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`.

### 11.1 Gemma CPU-traced shapes

SEL-01/02 are covered by MOD-43–44 and are not repeated here.

### 11.2 Float shapes (all dtypes)

| ID | Test Name | Input | Op | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `SEL-03` | `test_select_1d_to_scalar` | `[256]` (all dtypes) | `select(0,0)` | `[]` scalar | `* 1.0` (0D; sum undefined) | `compare_with_cpu(lambda t: t.select(0,0) * 1.0, x)` |
| `SEL-04` | `test_select_2d_leading_dim` | `[8,256]` (all dtypes) | `select(0,3)` | `[256]` | `* 1.0` (inner 256 unchanged) | `compare_with_cpu(lambda t: t.select(0,3) * 1.0, x)` |
| `SEL-05` | `test_select_3d_seq_select` | `[1,32,256]` (all dtypes) | `select(1,0)` | `[1,256]` | `* 1.0` (inner 256 unchanged) | `compare_with_cpu(lambda t: t.select(1,0) * 1.0, x)` |
| `SEL-06` | `test_select_4d_head_select` | `[1,16,32,256]` (all dtypes) | `select(1,0)` | `[1,32,256]` | `* 1.0` (inner 256 unchanged) | `compare_with_cpu(lambda t: t.select(1,0) * 1.0, x)` |
| `SEL-07` | `test_select_4d_last_token_trailing` | `[1,16,32,256]` (all dtypes) | `select(2,-1)` | `[1,16,256]` | `* 1.0` (inner 256 unchanged; dim-2 selected) | `compare_with_cpu(lambda t: t.select(2,-1) * 1.0, x)` |
| `SEL-08` | `test_select_5d_middle_dim` | `[1,8,2,32,256]` (all dtypes) | `select(2,0)` | `[1,8,32,256]` | `* 1.0` (inner 256 unchanged) | `compare_with_cpu(lambda t: t.select(2,0) * 1.0, x)` |
| `SEL-09` | `test_select_6d_to_5d` | `[1,2,4,8,16,256]` (all dtypes) | `select(3,0)` | `[1,2,4,16,256]` | `* 1.0` (inner 256 unchanged) | `compare_with_cpu(lambda t: t.select(3,0) * 1.0, x)` |

### 11.3 Unaligned innermost dimension

| ID | Test Name | Input | Op | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `SEL-10` | `test_select_2d_leading_unaligned` | `[8,70]` (all dtypes) | `select(0,0)` | `[70]` | pointwise + sum_last | inner 70: 70%64=6 | `compare_with_cpu(lambda t: _apply_consumer(t.select(0,0), consumer), x)` |
| `SEL-11` | `test_select_3d_trailing_unaligned_new_inner` | `[4,256,33]` (all dtypes) | `select(1,0)` | `[4,33]` | pointwise + sum_last | inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.select(1,0), consumer), x)` |
| `SEL-12` | `test_select_4d_unaligned_inner` | `[1,16,32,65]` (all dtypes) | `select(1,0)` | `[1,32,65]` | pointwise + sum_last | inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.select(1,0), consumer), x)` |

### 11.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Op | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `SEL-13` | `test_select_2d_size1_leading` | `[1,256]` (all dtypes) | `select(0,0)` | `[256]` | pointwise + sum_last | Removes size-1 dim | `compare_with_cpu(lambda t: _apply_consumer(t.select(0,0), consumer), x)` |
| `SEL-14` | `test_select_3d_size1_gqa` | `[1,8,1]` (all dtypes) | `select(2,0)` | `[1,8]` | pointwise + sum_last | Select removes trailing size-1 | `compare_with_cpu(lambda t: _apply_consumer(t.select(2,0), consumer), x)` |
| `SEL-15` | `test_select_5d_size1_dim` | `[1,8,1,32,256]` (all dtypes) | `select(2,0)` | `[1,8,32,256]` | pointwise + sum_last | Select the size-1 dim | `compare_with_cpu(lambda t: _apply_consumer(t.select(2,0), consumer), x)` |

---

## Section 12 — `narrow` / `slice` — Operation Level

`aten.slice.Tensor` observed in both traces. Equivalent to `torch.narrow`. Key CPU shapes: RoPE half-split `[B,S,H,256]→narrow(3,0,128)→[B,S,H,128]`, KV-cache crop, sliding-window crop. Key Spyre shapes: `[64,128,64]→slice(2,0,1)→[64,128,1]`, `[1,8,64]→slice(2,0,32)→[1,8,32]`.

**Consumer parametrization for §12.3/§12.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `* 1.0` (pointwise) and `.sum(dim=-1)` (sum_last) are applied.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`.

### 12.1 Gemma CPU-traced shapes (aligned)

SLC-01–04 are covered by MOD-45–48 and are not repeated here.

### 12.2 Gemma Spyre-traced shapes

| ID | Test Name | Input | Op | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `SLC-05` | `test_slice_3d_last_element` | `[64,128,64]` (all dtypes) | `narrow(2,0,1)` | `[64,128,1]` | `* 1.0` | `compare_with_cpu(lambda t: t.narrow(2,0,1) * 1.0, x)` |
| `SLC-06` | `test_slice_3d_half_inner` | `[1,8,64]` (all dtypes) | `narrow(2,0,32)` | `[1,8,32]` | `* 1.0` | `compare_with_cpu(lambda t: t.narrow(2,0,32) * 1.0, x)` |
| `SLC-07` | `test_slice_3d_single_element_inner_bf16` | `[1,8,64]` bf16 only | `narrow(2,0,1)` | `[1,8,1]` | `* 1.0` | `compare_with_cpu(lambda t: t.narrow(2,0,1) * 1.0, x)` |

### 12.3 Unaligned innermost dimension

| ID | Test Name | Input | Op | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `SLC-08` | `test_narrow_1d_unaligned_result` | `[256]` (all dtypes) | `narrow(0,0,65)` | `[65]` | pointwise + sum_last | output inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(0,0,65), consumer), x)` |
| `SLC-09` | `test_narrow_2d_trailing_unaligned` | `[8,128]` (all dtypes) | `narrow(1,0,33)` | `[8,33]` | pointwise + sum_last | output inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(1,0,33), consumer), x)` |
| `SLC-10` | `test_narrow_3d_inner_unaligned` | `[1,32,128]` (all dtypes) | `narrow(2,0,70)` | `[1,32,70]` | pointwise + sum_last | inner 70: 70%64=6 | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(2,0,70), consumer), x)` |
| `SLC-11` | `test_narrow_4d_head_dim_unaligned` | `[1,16,32,256]` (all dtypes) | `narrow(3,0,65)` | `[1,16,32,65]` | pointwise + sum_last | output inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(3,0,65), consumer), x)` |
| `SLC-12` | `test_narrow_5d_inner_below_stick` | `[1,2,4,8,128]` (all dtypes) | `narrow(4,0,13)` | `[1,2,4,8,13]` | pointwise + sum_last | output inner 13 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(4,0,13), consumer), x)` |
| `SLC-13` | `test_narrow_6d_inner_unaligned` | `[1,2,4,8,16,128]` (all dtypes) | `narrow(5,0,33)` | `[1,2,4,8,16,33]` | pointwise + sum_last | 6D; output inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(5,0,33), consumer), x)` |

### 12.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Op | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `SLC-14` | `test_narrow_2d_size1_leading` | `[1,256]` (all dtypes) | `narrow(0,0,1)` | `[1,256]` | pointwise + sum_last | Narrow on size-1 dim | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(0,0,1), consumer), x)` |
| `SLC-15` | `test_narrow_3d_to_size1_inner` | `[1,32,256]` (all dtypes) | `narrow(2,0,1)` | `[1,32,1]` | pointwise + sum_last | Slice to size-1 innermost | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(2,0,1), consumer), x)` |
| `SLC-16` | `test_narrow_4d_already_size1_dim` | `[1,8,1,256]` (all dtypes) | `narrow(2,0,1)` | `[1,8,1,256]` | pointwise + sum_last | Narrow the size-1 dim | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(2,0,1), consumer), x)` |
| `SLC-17` | `test_narrow_5d_inner_to_size1` | `[1,8,2,32,64]` (all dtypes) | `narrow(4,0,1)` | `[1,8,2,32,1]` | pointwise + sum_last | 5D; innermost sliced to 1 | `compare_with_cpu(lambda t: _apply_consumer(t.narrow(4,0,1), consumer), x)` |

---

## Section 13 — `split` — Operation Level

`aten.split.Tensor` observed in CPU trace: MoE expert weight split `[256,1408]→split(704,-1)→([256,704],[256,704])`. Returns a tuple of tensors. Each output tensor is a view of the original storage.

**Consumer parametrization for §13.3/§13.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. The consumer is applied per-chunk inside the `torch.stack(...)` expression.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`.

### 13.1 Gemma CPU-traced shapes (aligned)

SPL-01 is covered by MOD-50–51 and is not repeated here. SPL-02 is kept (no module-level test for QKV head split shape).

| ID | Test Name | Input | Op | Outputs | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `SPL-02` | `test_split_qkv_heads_trailing` | `[1,32,4096]` (all dtypes) | `split(2048, dim=-1)` | `([1,32,2048], [1,32,2048])` | `* 1.0` on stack | `compare_with_cpu(lambda t: torch.stack([c * 1.0 for c in t.split(2048, dim=-1)]), x)` |

### 13.2 Additional aligned shapes

| ID | Test Name | Input | Op | Outputs | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `SPL-03` | `test_split_2d_equal_cols` | `[32,256]` (all dtypes) | `split(128, dim=1)` | `([32,128], [32,128])` | `* 1.0` on stack | `compare_with_cpu(lambda t: torch.stack([c * 1.0 for c in t.split(128, dim=1)]), x)` |
| `SPL-04` | `test_split_3d_seq_dim` | `[2,64,256]` (all dtypes) | `split(32, dim=1)` | `[2,32,256]` × 2 | `* 1.0` (inner 256 unchanged) | `compare_with_cpu(lambda t: torch.stack([c * 1.0 for c in t.split(32, dim=1)]), x)` |
| `SPL-05` | `test_split_4d_head_split` | `[1,32,16,256]` (all dtypes) | `split(8, dim=2)` | `[1,32,8,256]` × 2 | `* 1.0` (inner 256 unchanged) | `compare_with_cpu(lambda t: torch.stack([c * 1.0 for c in t.split(8, dim=2)]), x)` |
| `SPL-06` | `test_split_5d_rope_half` | `[1,32,16,2,128]` (all dtypes) | `split(1, dim=3)` | `[1,32,16,1,128]` × 2 | `* 1.0` (inner 128 unchanged) | `compare_with_cpu(lambda t: torch.stack([c * 1.0 for c in t.split(1, dim=3)]), x)` |

### 13.3 Unaligned innermost dimension

| ID | Test Name | Input | Op | Outputs | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `SPL-07` | `test_split_2d_unaligned_result_trailing` | `[8,130]` (all dtypes) | `split(65, dim=-1)` | `[8,65]` × 2 | pointwise + sum_last (per chunk) | output inner 65 unaligned | `compare_with_cpu(lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(65, dim=-1)]), x)` |
| `SPL-08` | `test_split_3d_unaligned_trailing` | `[4,32,140]` (all dtypes) | `split(70, dim=-1)` | `[4,32,70]` × 2 | pointwise + sum_last (per chunk) | output inner 70 unaligned | `compare_with_cpu(lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(70, dim=-1)]), x)` |
| `SPL-09` | `test_split_4d_unaligned_inner` | `[1,16,32,130]` (all dtypes) | `split(65, dim=-1)` | `[1,16,32,65]` × 2 | pointwise + sum_last (per chunk) | inner 65 unaligned | `compare_with_cpu(lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(65, dim=-1)]), x)` |
| `SPL-10` | `test_split_5d_unaligned_non_trailing` | `[1,2,4,8,256]` (all dtypes) | `split(2, dim=2)` | `[1,2,2,8,256]` × 2 | pointwise + sum_last (per chunk) | non-trailing split | `compare_with_cpu(lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(2, dim=2)]), x)` |

### 13.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Op | Outputs | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `SPL-11` | `test_split_3d_trailing_size1_result` | `[4,32,2]` (all dtypes) | `split(1, dim=-1)` | `[4,32,1]` × 2 | pointwise + sum_last (per chunk) | Each chunk has size-1 inner | `compare_with_cpu(lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(1, dim=-1)]), x)` |
| `SPL-12` | `test_split_4d_size1_dim` | `[1,8,32,256]` (all dtypes) | `split(1, dim=0)` | `[1,8,32,256]` (single chunk) | pointwise + sum_last | Split on size-1 dim → single chunk | `compare_with_cpu(lambda t: _apply_consumer(t.split(1, dim=0)[0], consumer), x)` |
| `SPL-13` | `test_split_5d_size1_head_chunks` | `[1,8,1,32,256]` (all dtypes) | `split(4, dim=1)` | `[1,4,1,32,256]` × 2 | pointwise + sum_last (per chunk) | GQA-like chunks | `compare_with_cpu(lambda t: torch.stack([_apply_consumer(c, consumer) for c in t.split(4, dim=1)]), x)` |

---

## 14. Coverage Summary

Active test counts (op-level tests only; module-level MOD-xx tests counted separately):

| Section | Op | Dim range | CPU-traced | Spyre-traced | Aligned | Unaligned | Size-1 | Chain/restickify | Total active |
|---|---|---|---|---|---|---|---|---|---|
| §6 `transpose` | `transpose` | 2D–6D | 0 | — | 0 | 6 | 5 | 0 | **11** |
| §7 `t` | `t` | 1D–2D | 0 | — | 0 | 4 | 3 | — | **7** |
| §8 `permute` | `permute` | 2D–6D | — | 2 | 2 | 5 | 4 | — | **11** |
| §9 `clone` | `clone` | 0D–6D | 0 | 5 | 5 | 6 | 5 | 1 (post-permute) | **16** |
| §10 `cat` | `cat` | 1D–6D | 1 | — | 1 | 6 | 3 | — | **10** |
| §11 `select` | `select` | 1D→0D … 6D→5D | 0 | — | 7 | 3 | 3 | — | **13** |
| §12 `narrow`/`slice` | `narrow` | 1D–6D | 0 | 3 | 3 | 6 | 4 | — | **13** |
| §13 `split` | `split` | 2D–6D | 1 | — | 5 | 4 | 3 | — | **12** (SPL-02 + §13.2–13.4) |
| **Total** | 8 ops | **0D–6D** | **2** | **10** | **16** | **40** | **30** | **1** | **93** |

---

## 15. Rank Coverage Matrix

TRP-01–08, T2D-01–04, PRM-02/04/05/06, CLN-01–05, CAT-01/03/04/05, SEL-01/02, SLC-01–04, and SPL-01 are covered by MOD-22–51 module-level tests and are not repeated in the op-level sections.

| Op | 0D | 1D | 2D | 3D | 4D | 5D | 6D |
|---|---|---|---|---|---|---|---|
| **transpose** (input rank) | — | — | TRP-09,15 | TRP-10,16 | TRP-11–13,17 | TRP-13,18 | TRP-14,19 |
| **t** (input rank) | — | T2D-05 | T2D-06–11 | — | — | — | — |
| **permute** (input rank) | — | — | PRM-01,03,07 | PRM-08,12 | PRM-09,13 | PRM-10,14 | PRM-11,15 |
| **clone** (input rank) | CLN-17 | CLN-11,18 | CLN-12,19 | CLN-06–07,13,20 | CLN-08–09,14,21 | CLN-10,15 | CLN-16 |
| **cat** (input rank) | — | CAT-06 | CAT-07,13 | CAT-02,08,12 | CAT-09,14 | CAT-10 | CAT-11 |
| **select** (input rank → output rank) | — | SEL-03 (1D→0D) | SEL-04,13 (2D→1D) | SEL-05,10,14 (3D→2D) | SEL-06–07,11,15 (4D→3D) | SEL-08,12 (5D→4D) | SEL-09 (6D→5D) |
| **narrow** (input rank) | — | SLC-08 | SLC-09,14 | SLC-05–07,10,15 | SLC-11,16 | SLC-12,17 | SLC-13 |
| **split** (input rank) | — | — | SPL-02–03,07 | SPL-04,08,11 | SPL-05,09,12 | SPL-06,10,13 | — |
