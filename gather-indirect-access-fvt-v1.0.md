# Gather Indirect Access — Comprehensive FVT Plan

---

| Field | Value |
|---|---|
| **Title** | Gather Indirect Access — Comprehensive FVT Plan for torch-spyre |
| **Version** | 1.6 |
| **Date** | 2026-08-12 |
| **Author** | Mahalakshmi Pathivada |
| **Reviewers** | Shruthip Venkatesh, Tanushree Haldar |
| **Approver** | Albin Joy, Pradipta Ghosh, Anto John |
| **FRS Source** | `FRS/FRS_GATHER_INDIRECT_ADDRESS_UPDATED.txt`, `FRS/Indirect_Access_Entirety_View.md` |

---

## Scope

This document defines **net-new** test cases for gather indirect access validation on torch-spyre. All scenarios here are unique and do not duplicate:

- Existing torch-spyre tests: `tests/inductor/test_indirect_access_gather.py` (TestGather, 45 tests), `test_indirect_access_scatter.py` (TestScatter, 16 tests), `test_indirect_access_internals.py` (TestInternals, 63 tests)
- Upstream YAML configs: `tests/configs/upstream_tests_beta/test_indexing.yaml`, `tests/configs/upstream_tests_beta/test_torch.yaml`, `tests/configs/upstream_tests_beta/nn-test_embedding.yaml`, `tests/configs/upstream_tests_beta/test_scatter_gather_ops.yaml`
- Prior FVT plans in `FVT/torch-gather-fvt-v1.0.md`

**Inference-only scope**: No backward, no autograd, no training, no CUDA, no JIT. CPU-based compilation, Spyre device execution.

**Covered operations**: `x[i]` (advanced indexing), `torch.gather`, `torch.index_select`, `torch.take_along_dim`, `torch.take`, `F.embedding`, `nn.Embedding`, `F.embedding_bag`, `nn.EmbeddingBag`, `torch.masked_select`, `torch.repeat_interleave`.

---

## Test Environment Variables

| Variable | Default | Test Values | Purpose |
|---|---|---|---|
| `SENCORES` | 1 | 1, 4, 32 | Number of Spyre cores (`config.patch({"sencores": N})` in tests) |
| `LX_PLANNING` | 0 | 0, 1 | Enable LX scratchpad planning |
| `CO_OPTIMIZING_LX_PLANNING` | 0 | 0, 1 | Co-optimized LX placement |
| `SPYRE_INDUCTOR_IGNORE_HINTS` | 0 | 0, 1 | Disable WSR/span-overflow coarse-tiling hints |
| `GLOBAL_STICK_OPTIMIZER` | 1 | 0, 1 | Global stick optimizer |
| `BUNDLE_SYMBOLIC_ARGS` | 1 | 0, 1 | Bundle symbolic arguments as runtime symbols |
| `UNROLL_LOOPS` | 1 | 0, 1 | Loop unrolling |
| `DXP_LX_FRAC_AVAIL` | 0.2 | 0.1, 0.5, 0.8 | LX allocation fraction |
| `LAYOUT_SOLVER` | greedy | greedy, firstfit, bestfit | LX layout solver algorithm |
| `TORCH_SPYRE_DOWNCAST_WARN` | 1 | 0, 1 | int64→int32 downcast warning |

---

## Table of Contents

| Section | Category | Test IDs | Count |
|---|---|---|---|
| 1 | Core Functional Correctness | GFC-01..GFC-20 | 20 |
| 2 | Data Type Coverage | GDT-01..GDT-16 | 16 |
| 3 | Index Access Patterns | GIDX-01..GIDX-14 | 14 |
| 4 | SpyreTensorLayout (STL) | GSTL-01..GSTL-15 | 15 |
| 5 | Downstream Co-Scheduled Operations | GDS-01..GDS-17 | 17 |
| 6 | Named Dimensions | GNDM-01..GNDM-10 | 10 |
| 7 | Execution Modes | GEM-01..GEM-10 | 10 |
| 8 | Embedding Table Lookup | GEMB-01..GEMB-15 | 15 |
| 9 | MoE Expert Routing | GMOE-01..GMOE-10 | 10 |
| 10 | Paged Attention and KV Cache | GPA-01..GPA-26 | 26 |
| 11 | vLLM Serving Patterns | GVLLM-01..GVLLM-15 | 15 |
| 12 | FMS Serving Patterns | GFMS-01..GFMS-08 | 8 |
| 13 | Real Model Patterns | GMDL-01..GMDL-13 | 13 |
| 14 | STL Paired Comparison | GSTLCMP-01..GSTLCMP-09 | 9 |
| 15 | Compiler Pipeline Validation | GCP-01..GCP-23 | 23 |
| 16 | Memory, Layout, and Scale | GMEM-01..GMEM-07 | 7 |
| 17 | Dynamic and Symbolic Shapes | GDYN-01..GDYN-10 | 10 |
| 18 | Cross-Feature Dependencies | GCFG-01..GCFG-10 | 10 |
| 19 | Chained and Composed Gather | GCH-01..GCH-08 | 8 |
| 20 | Edge Cases and Negative Tests | GEC-01..GEC-19 | 19 |
| 21 | torch.masked_select and Mask-Based Gather | GMSK-01..GMSK-16 | 16 |
| 22 | torch.index_select — Dedicated Corner Cases | GIDXS-01..GIDXS-16 | 16 |
| 23 | EmbeddingBag and Lookup Table Advanced Coverage | GEMB2-01..GEMB2-15 | 15 |
| 24 | Python API Forms — Complete Coverage | GAPI-01..GAPI-25 | 25 |
| 25 | Non-Contiguous Source Patterns | GNCS-01..GNCS-10 | 10 |
| 26 | Known Bug Regression Cases | GBUG-01..GBUG-08 | 8 |
| 27 | Additional Corner Cases | GCOR-01..GCOR-14 | 14 |
| 28 | STL Advanced and Production Patterns | GSTLADV-01..GSTLADV-10 | 10 |
| 29 | Stride-Exact and Precision Real Model Cases | GSTR-01..GSTR-10 | 10 |
| 30 | Chained Gather with chunk, split, and RoPE Patterns | GCHNK-01..GCHNK-15 | 15 |
| 31 | E2E vLLM Decode and Prefill Inference Patterns | GE2E-01..GE2E-14 | 14 |
| 32 | LX Scratchpad and Compiler Optimization Depth | GLXD-01..GLXD-10 | 10 |
| 33 | SDPA with Paged KV Cache (FlashAttention Patterns) | GSDPA-01..GSDPA-20 | 20 |
| 34 | Non-Leading-Dim Gather (dim=1, dim=2 Production Patterns) | GNLD-01..GNLD-07 | 7 |
| — | **Total** | | **472** |

---

## Section 1 — Core Functional Correctness

**Test class:** `TestGatherShapeRankAndIndexPatterns`
**File:** `tests/inductor/indirect_access/test_gather_functional.py`

> Covers gather semantics across tensor ranks, shapes, and API forms not already validated in the existing `TestGather` suite. The existing suite covers (128,256)/(32,256) 2D shapes, (64,8,64) 3D shapes, and the basic `x[idx]`/`index_select`/`torch.gather` APIs. All cases here use distinct shapes or structural patterns.
>
> **Indirect Access Validation Focus:** Validates that the Spyre compiler emits a correct `GATHER_OP_SPEC` for the indirect read path across diverse 2D/3D/4D shapes — square, wide, tall, non-power-of-2 inner dims, and full-range permutations — ensuring the SDSC stick-grid address formula is correct regardless of tensor geometry. Failure here indicates a broken `_add_indices_to_address_index` pass or incorrect SDSC `device_size`/`stride_map` encoding.

| ID | Test Name | Input Shape | Index Shape | dim | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GFC-01 | `test_gather_2d_square` | (64, 64) | (32,) | 0 | float16 | Square tensor gather; GATHER_OP_SPEC emitted; output shape (32, 64) | Output shape (32,64); values match CPU reference | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_square |
| GFC-02 | `test_gather_2d_large_inner` | (16, 1024) | (8,) | 0 | float16 | Wide inner dim (N=1024, 16 sticks/row); no SDSC stride error | Output shape (8,1024); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_large_inner |
| GFC-03 | `test_gather_2d_wide` | (8, 4096) | (4,) | 0 | float16 | Very wide row (64 sticks); memory layout correct | Output shape (4,4096); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_wide |
| GFC-04 | `test_gather_2d_tall` | (256, 64) | (128,) | 0 | float16 | Tall tensor (many rows, narrow inner); large M correctly handled | Output shape (128,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_tall |
| GFC-05 | `test_gather_2d_small` | (4, 16) | (2,) | 0 | float16 | Minimal tensor; N=16 < one stick (64 elems); padding handled | Output shape (2,16); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_small |
| GFC-06 | `test_gather_2d_non_power_of_2_inner` | (16, 100) | (8,) | 0 | float16 | N=100 not multiple of 64; stick-level padding does not corrupt output | Output shape (8,100); values match CPU; no padding leak | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_non_power_of_2_inner |
| GFC-07 | `test_gather_2d_full_range` | (32, 64) | (32,) | 0 | float16 | Index = full row permutation; every row read exactly once | Output is permuted version of input; row-by-row match | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_full_range |
| GFC-08 | `test_gather_2d_all_same_row` | (32, 64) | (16,) | 0 | float16 | All idx values = 0; repeated reads of row 0; no caching side-effects | All 16 output rows == input row 0 | exact | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_all_same_row |
| GFC-09 | `test_gather_2d_ascending_idx` | (32, 128) | (16,) | 0 | float16 | Sequential ascending idx [0..15]; cache-friendly access order | Output matches input rows 0–15 in order | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_ascending_idx |
| GFC-10 | `test_gather_2d_descending_idx` | (32, 128) | (16,) | 0 | float16 | Descending idx [31..16]; reverse access order | Output matches input rows 31–16 in reverse | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_descending_idx |
| GFC-11 | `test_gather_2d_repeated_idx` | (32, 64) | (20,) | 0 | float16 | idx has repeated values; each repeated row must produce identical output | Duplicate output rows are bit-identical | exact | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_repeated_idx |
| GFC-12 | `test_gather_3d_wide` | (8, 32, 512) | (16,) | 0 | float16 | 3D value tensor; dim=0 gather; inner dims (32,512) contiguous | Output shape (16,32,512); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_3d_wide |
| GFC-13 | `test_gather_3d_small` | (4, 4, 16) | (2,) | 0 | float16 | Smallest valid 3D tensor; sub-stick inner dims | Output shape (2,4,16); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_3d_small |
| GFC-14 | `test_gather_3d_interior_dim` | (8, 16, 64) | (8,) | 1 | float16 | `x[:, idx, :]` — interior dim=1 gather; default layout already outermost; no STL required | Output shape (8,8,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_3d_interior_dim |
| GFC-15 | `test_gather_4d_data` | (2, 8, 4, 64) | (4,) | 0 | float16 | 4D value tensor; gather at dim=0 collapses first dim | Output shape (4,8,4,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_4d_data |
| GFC-16 | `test_index_select_new_shape` | (16, 256) | (12,) | 0 | float16 | `torch.index_select(x, 0, idx)` — distinct shape from existing suite; GATHER_OP_SPEC | Output shape (12,256); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_index_select_new_shape |
| GFC-17 | `test_take_along_dim` | (32, 64) | (16, 64) | 0 | float16 | `torch.take_along_dim(x, idx, dim=0)` same-rank index; topk-style gather | Output shape (16,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_take_along_dim |
| GFC-18 | `test_torch_gather_dim0` | (32, 64) | (16, 64) | 0 | float16 | `torch.gather(x, 0, idx)` functional form; same-rank index | Output shape (16,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_torch_gather_dim0 |
| GFC-19 | `test_gather_result_exact` | (16, 64) | (8,) | 0 | float16 | Integer-representable float16 values (powers of 2); verify no rounding at output | Output values match exact integer targets; no fp16 rounding | exact | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_result_exact |
| GFC-20 | `test_gather_2d_interleaved_idx` | (32, 128) | (16,) | 0 | float16 | Interleaved idx [0,16,1,17,...]; alternating low/high row access | Output rows in interleaved order match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_gather_2d_interleaved_idx |

---

## Section 2 — Data Type Coverage

**Test class:** `TestGatherValueAndIndexDtypeCoverage`
**File:** `tests/inductor/indirect_access/test_gather_dtypes.py`

> The existing `TestGather` validates float16 value + int32 index (primary) and int64 index on float16. These cases cover the remaining value dtypes, cross-combinations, and SDSC wordLength encoding.
>
> **Indirect Access Validation Focus:** Validates that the SDSC bundle's `wordLength` field is set correctly for every supported value dtype (fp16/bf16/fp32/int32/int64/bool) and index dtype (int32, int64 with auto-downcast), ensuring the hardware address formula — which multiplies element size by index value — remains numerically correct for each byte width. Failure here produces silently wrong values (shifted by an element-size factor) for any dtype not exercised in the primary test.

| ID | Test Name | Input Shape | Value dtype | Index dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|
| GDT-01 | `test_gather_bfloat16_value` | (32, 128) | bfloat16 | int32 | bfloat16 GATHER_OP_SPEC emitted; SDSC wordLength=4; correct output | Output matches CPU bfloat16 reference | atol=2e-2 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_bfloat16_value |
| GDT-02 | `test_gather_float32_value` | (32, 128) | float32 | int32 | float32 GATHER_OP_SPEC emitted; 4 bytes/element layout; correct output | Output matches CPU float32 reference | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_float32_value |
| GDT-03 | `test_gather_int32_value` | (32, 64) | int32 | int32 | Integer value tensor gather; exact integer output expected | Output matches CPU reference exactly; no rounding | exact | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_int32_value |
| GDT-04 | `test_gather_int64_value` | (32, 64) | int64 | int32 | int64 value tensor; 8 bytes/element; SDSC stride correct | Output matches CPU reference exactly | exact | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_int64_value |
| GDT-05 | `test_gather_bool_value` | (32, 64) | bool | int32 | Boolean value gather; 1-byte elements; output is bool tensor | Output bool values match CPU reference element-wise | exact | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_bool_value |
| GDT-06 | `test_gather_int64_index_float16` | (32, 128) | float16 | int64 | int64 index auto-downcast to int32; downcast warning emitted in logs | Output matches CPU; int64→int32 warning present in stderr | atol=1e-2 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_int64_index_float16 |
| GDT-07 | `test_gather_int32_index_bfloat16` | (32, 128) | bfloat16 | int32 | bfloat16 value + int32 index — standard production combination | Output matches CPU bfloat16 reference | atol=2e-2 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_int32_index_bfloat16 |
| GDT-08 | `test_gather_int64_index_bfloat16` | (32, 128) | bfloat16 | int64 | bfloat16 value + int64 index → auto-downcast; both features interact | Output correct after downcast; downcast warning emitted | atol=2e-2 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_int64_index_bfloat16 |
| GDT-09 | `test_gather_int32_index_float32` | (32, 128) | float32 | int32 | float32 value + int32 index; no downcast path exercised | Output matches CPU float32 reference | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_int32_index_float32 |
| GDT-10 | `test_gather_int64_index_float32` | (32, 128) | float32 | int64 | float32 value + int64 index → downcast; combined dtype path | Output correct; downcast warning present | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_int64_index_float32 |
| GDT-11 | `test_gather_float64_not_gathered` | (32, 64) | float64 | int32 | float64 must NOT produce GATHER_OP_SPEC; compiler falls back to CPU path | GATHER_OP_SPEC absent in FX graph; output computed on CPU | N/A | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_float64_not_gathered |
| GDT-12 | `test_gather_downcast_warning_suppressed` | (32, 128) | float16 | int64 | TORCH_SPYRE_DOWNCAST_WARN=0 suppresses int64→int32 warning | No downcast warning in stderr; output still correct | atol=1e-2 | TORCH_SPYRE_DOWNCAST_WARN=0 | TestGatherValueAndIndexDtypeCoverage::test_gather_downcast_warning_suppressed |
| GDT-13 | `test_gather_bfloat16_with_exp` | (32, 128) | bfloat16 | int32 | bfloat16 gather + exp() downstream fused; GATHER_OP_SPEC + exp co-scheduled | Output = exp(gathered rows); matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_bfloat16_with_exp |
| GDT-14 | `test_gather_float32_with_abs` | (32, 128) | float32 | int32 | float32 gather + abs() downstream; fused execution on Spyre | Output = abs(gathered rows); matches CPU float32 | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_gather_float32_with_abs |
| GDT-15 | `test_sdsc_wordlength_int32` | (32, 128) | float16 | int32 | SDSC bundle: index wordLength field = 4 (int32 = 4 bytes) | Compiled artifact contains SDSC wordLength=4 | N/A | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_sdsc_wordlength_int32 |
| GDT-16 | `test_sdsc_wordlength_int64` | (32, 128) | float16 | int64 | SDSC bundle: index wordLength = 8 before downcast; SDSC reflects original type | Compiled artifact contains SDSC wordLength=8 prior to downcast | N/A | SENCORES=1 | TestGatherValueAndIndexDtypeCoverage::test_sdsc_wordlength_int64 |

---

## Section 3 — Index Access Patterns

**Test class:** `TestGatherShapeRankAndIndexPatterns`
**File:** `tests/inductor/indirect_access/test_gather_functional.py`

> Validates correctness across different index value distributions and structural properties. Existing suite covers random int32 (P=32) and single-row (P=1). These cases cover additional patterns.
>
> **Indirect Access Validation Focus:** Validates that the indirect read path produces correct output regardless of index value distribution — hotspot-heavy (80% same row), strided (stride=4), oversize (P>M), full-range sequential, and repeated pairs — ensuring the SDSC address computation does not depend on index value ordering or uniqueness. Any caching, stride shortcut, or address-reuse optimization in the compiler that incorrectly assumes index independence would be caught here.

| ID | Test Name | Input Shape | Index Values | P (index length) | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GIDX-01 | `test_idx_random_permutation` | (32, 128) | Random permutation of [0..31] | 32 | float16 | Every row accessed exactly once; GATHER_OP_SPEC handles full-range random | Output is permuted input; values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_random_permutation |
| GIDX-02 | `test_idx_hotspot_80pct` | (32, 128) | 80% same value, 20% varied | 24 | float16 | Heavy hotspot access (one row ~19× out of 24); repeated-read correctness | Output rows for repeated index all identical; match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_hotspot_80pct |
| GIDX-03 | `test_idx_alternating` | (32, 64) | Alternates [0,1,0,1,...] | 16 | float16 | Only 2 unique rows repeated; minimal working set; no caching side-effects | Even output rows = row 0; odd = row 1; exact match | exact | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_alternating |
| GIDX-04 | `test_idx_strided` | (64, 64) | [0, 4, 8, 12, ...] | 16 | float16 | Strided row access (stride=4); non-contiguous access pattern on Spyre | Output rows match input rows 0,4,8,... in order | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_strided |
| GIDX-05 | `test_idx_near_full` | (32, 64) | All rows except row 31 | P=31 | float16 | P=M-1; all but one row; near-complete scan correctness | 31 output rows match corresponding input rows | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_near_full |
| GIDX-06 | `test_idx_oversize` | (16, 64) | Random with repeats, P=32>M=16 | P=32 | float16 | P > M (more output rows than input); each input row read ≥twice | Output shape (32,64); values match CPU reference | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_oversize |
| GIDX-07 | `test_idx_two_elements` | (32, 128) | [idx_a, idx_b] | P=2 | float16 | Minimal index (P=2); smallest valid gather output | 2 output rows match indexed input rows | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_two_elements |
| GIDX-08 | `test_idx_single_row` | (32, 256) | [idx] | P=1 | float16 | P=1 triggers known BackendCompilerFailed; error path validated, not output | Raises BackendCompilerFailed (or equivalent) at compile | N/A | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_single_row |
| GIDX-09 | `test_idx_full_range` | (32, 64) | [0,1,...,31] | P=M=32 | float16 | Exact full-range sequential; identity-permutation gather | Output identical to input tensor order | exact | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_full_range |
| GIDX-10 | `test_idx_2d_shape` | (16, 64) | Shape (8, 4) 2D index | P=32 (flattened) | float16 | 2D-shaped index compiles; device output diverges from CPU (known behavior) | Compile succeeds; divergence from CPU is documented expected behavior | N/A | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_2d_shape |
| GIDX-11 | `test_idx_sorted_ascending` | (64, 128) | [0,1,2,...,P-1] | P=32 | float16 | Sorted ascending order; cache-friendly pattern; validate correctness | Output = input rows 0–31; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_sorted_ascending |
| GIDX-12 | `test_idx_sorted_descending` | (64, 128) | [P-1,...,1,0] | P=32 | float16 | Reverse-sorted index; reverse-cache-friendly access; correctness | Output = input rows 31–0 in reverse; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_sorted_descending |
| GIDX-13 | `test_idx_boundary_values` | (32, 64) | Contains 0 and M-1=31 plus mid values | P=16 | float16 | Both boundary rows (first and last) in index; boundary read correctness | Row 0 and row 31 output correct; full output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_boundary_values |
| GIDX-14 | `test_idx_repeated_pairs` | (16, 64) | [0,0,1,1,2,2,...] | P=16 | float16 | Each row appears twice consecutively; adjacent duplicate reads | Consecutive duplicate output rows are bit-identical; match CPU | exact | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_idx_repeated_pairs |

---

## Section 4 — SpyreTensorLayout (STL)

**Test class:** `TestGatherSpyreTensorLayoutAnnotation`
**File:** `tests/inductor/indirect_access/test_gather_stl.py`

> STL is required when the gather dimension is not already the outermost device dimension (FR-3.1). Index tensors never receive STL. These cases extend the STL-A/STL-B coverage from the previous FVT with new shapes and combinations.
>
> **Indirect Access Validation Focus:** Validates the `SpyreTensorLayout` annotation mechanism: when the gather dimension is not the outermost device dimension, the compiler must inject an STL specifying `device_size` and `stride_map` to remap the dimension ordering in the SDSC `device_coordinates` field. Failure here — a missing STL, wrong `stride_map`, or STL incorrectly applied to the index tensor — produces an incorrect base address and silently reads from wrong memory locations on Spyre.

| ID | Test Name | Input Shape | Index Shape | STL device_size / stride_map | dim | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GSTL-01 | `test_stl_2d_square` | (64, 64) | (32,) | `[64,1,64]` / `[64,1]` | 0 | float16 | STL required: gather dim=0 is NOT outermost device dim; SDSC layout encoded correctly | Output (32,64) matches CPU; GATHER_OP_SPEC present | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_2d_square |
| GSTL-02 | `test_stl_2d_frs_example_a` | (12, 1024) | (8,) | `[12,16,64]` / `[1024,64,1]` | 0 | float16 | FRS canonical Example A shape and stride; device_size and stride_map exact match | Output (8,1024) matches CPU; STL strides match FRS spec | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_2d_frs_example_a |
| GSTL-03 | `test_stl_2d_16x512` | (16, 512) | (8,) | `[16,8,64]` / `[512,64,1]` | 0 | float16 | 8 sticks in inner dim; STL slicing into stick grid | Output (8,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_2d_16x512 |
| GSTL-04 | `test_stl_2d_int64_idx` | (32, 256) | (16,) | `[32,4,64]` / `[256,64,1]` | 0 | float16 (value) / int64 (idx) | int64 index + STL co-exist; downcast does not corrupt STL encoding | Output (16,256) matches CPU; wordLength=8 in SDSC | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_2d_int64_idx |
| GSTL-05 | `test_stl_2d_with_tanh` | (32, 256) | (16,) | `[32,4,64]` / `[256,64,1]` | 0 | float16 | STL + tanh downstream co-scheduled; GATHER_OP_SPEC + tanh kernel fused | Output = tanh(gathered rows); matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_2d_with_tanh |
| GSTL-06 | `test_stl_2d_with_sum_reduction` | (32, 256) | (16,) | `[32,4,64]` / `[256,64,1]` | 0 | float16 | STL + sum(dim=1) reduction after gather; shape collapses | Output shape (16,); sum values match CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_2d_with_sum_reduction |
| GSTL-07 | `test_stl_interior_dim_no_stl_needed` | (5, 64, 512) | (8,) | None required | 1 | float16 | `x[:, idx, :]` — outermost device dim already matches gather dim=1; no STL needed | Compiler does NOT inject STL; output (5,8,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_interior_dim_no_stl_needed |
| GSTL-08 | `test_stl_3d_explicit` | (8, 32, 128) | (4,) | `[8,32,128]` / `[4096,128,1]` | 0 | float16 | 3D value tensor with explicit STL at dim=0; 4096 elements/row | Output (4,32,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_3d_explicit |
| GSTL-09 | `test_stl_with_named_dims` | (32, 256) | (16,) | `[32,4,64]` / `[256,64,1]` | 0 | float16 | STL + named dims (`name_tensor_dims` on value and index); both annotations present | Compiled graph carries named dim annotations; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_with_named_dims |
| GSTL-10 | `test_stl_dynamic_compile` | (32, 256) | (16,) | `[32,4,64]` / `[256,64,1]` | 0 | float16 | STL + `torch.compile(dynamic=True)`; symbolic shapes do not break STL inference | Output matches CPU for multiple runtime shapes | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_dynamic_compile |
| GSTL-11 | `test_stl_with_lx_planning` | (32, 256) | (16,) | `[32,4,64]` / `[256,64,1]` | 0 | float16 | STL + LX scratchpad planning; LX planner sees STL-annotated buffer | Compile succeeds; output matches CPU; no LX allocation crash | atol=1e-2 | LX_PLANNING=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_with_lx_planning |
| GSTL-12 | `test_stl_embedding_shape` | (512, 64) | (32,) | `[512,1,64]` / `[64,1]` | 0 | float16 | Vocabulary embedding shape (512 rows × 64 dims); production embedding STL | Output (32,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_embedding_shape |
| GSTL-13 | `test_stl_kv_cache_shape` | (512, 8, 64) | (32,) | `[512,8,64]` / `[512,64,1]` | 0 | float16 | KV cache 3D shape; paged attention page lookup via STL | Output (32,8,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_kv_cache_shape |
| GSTL-14 | `test_stl_moe_shape` | (8, 512, 64) | (4,) | `[8,512,64]` / `[32768,64,1]` | 0 | float16 | MoE expert weight shape; large inner dims; STL stride map encoding | Output (4,512,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_moe_shape |
| GSTL-15 | `test_stl_bfloat16_value` | (32, 256) | (16,) | `[32,4,64]` / `[256,64,1]` | 0 | bfloat16 | bfloat16 value + STL; wordLength and stride map correct for bf16 element size | Output (16,256) matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_bfloat16_value |

---

## Section 5 — Downstream Co-Scheduled Operations

**Test class:** `TestGatherFusedDownstreamOperations`
**File:** `tests/inductor/indirect_access/test_gather_downstream.py`

> The existing suite covers all 7 basic unaries on 2D (32,256) shape, chained exp→tanh, scalar ×2.0, scalar +1.0, sum(dim=1), dual-gather binary, and sin-fallback. These cases extend to 3D tensors, new dtypes, new downstream types, and novel chains.
>
> **Indirect Access Validation Focus:** Validates that pointwise and reduction ops immediately following a `GATHER_OP_SPEC` are correctly fused into the same Spyre kernel bundle — meaning the gather output buffer feeds directly into the co-scheduled kernel without a host round-trip. A bug in the downstream fusion pass would either produce incorrect values (if the wrong buffer is wired) or a compile failure. CPU-fallback ops (cos, log) must cause a clean split-device dispatch, not a crash.

| ID | Test Name | Input Shape | Index Shape | Downstream Op | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GDS-01 | `test_gather_neg_3d` | (8, 32, 128) | (4,) | `neg` | float16 | 3D value gather + unary neg co-scheduled; GATHER_OP_SPEC fused with neg kernel | Output = neg(gathered rows); shape (4,32,128); matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_neg_3d |
| GDS-02 | `test_gather_sigmoid_3d` | (8, 32, 128) | (4,) | `sigmoid` | float16 | 3D value + sigmoid; non-linear activation after gather | Output = sigmoid(gathered rows); shape (4,32,128); matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_sigmoid_3d |
| GDS-03 | `test_gather_relu_3d` | (8, 32, 128) | (4,) | `relu` | float16 | 3D value + relu; piecewise activation after gather | Output = relu(gathered rows); zero-clipped values match CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_relu_3d |
| GDS-04 | `test_gather_sqrt_bfloat16` | (32, 128) | (16,) | `sqrt` | bfloat16 | bfloat16 value + sqrt; mixed dtype activation; SDSC wordLength=2 | Output = sqrt(gathered rows) in bfloat16; matches CPU | atol=2e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_sqrt_bfloat16 |
| GDS-05 | `test_gather_exp_float32` | (32, 128) | (16,) | `exp` | float32 | float32 value + exp downstream; no fp16 precision loss in activation | Output = exp(gathered rows) in float32; matches CPU | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_exp_float32 |
| GDS-06 | `test_gather_triple_unary` | (32, 256) | (16,) | `exp → tanh → sigmoid` | float16 | Triple-chained unary after gather; all three fused in one Spyre kernel | Output = sigmoid(tanh(exp(gathered))); matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_triple_unary |
| GDS-07 | `test_gather_scalar_sub` | (32, 128) | (16,) | `- 0.5` | float16 | Scalar subtraction after gather; scalar broadcast co-scheduled | Output = gathered - 0.5; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_scalar_sub |
| GDS-08 | `test_gather_scalar_div` | (32, 256) | (16,) | `/ 8.0` | float16 | Scalar divide (attention head scale 1/√d); co-scheduled with gather | Output = gathered / 8.0; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_scalar_div |
| GDS-09 | `test_gather_mean_reduction` | (32, 128) | (16,) | `mean(dim=1)` | float16 | Mean reduction along inner dim after gather; output shape collapses | Output shape (16,); mean values match CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_mean_reduction |
| GDS-10 | `test_gather_sum_dim0` | (32, 128) | (16,) | `sum(dim=0)` | float16 | Sum reduction along gather-output's first dim; shape (128,) | Output shape (128,); sum values match CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_sum_dim0 |
| GDS-11 | `test_gather_cos_cpu_fallback` | (32, 64) | (16,) | `cos` | float16 | cos has no Spyre kernel; gather remains on Spyre while cos falls back to CPU; split-device scheduling | Gather on Spyre; cos on CPU; combined output matches CPU reference | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_cos_cpu_fallback |
| GDS-12 | `test_gather_log_cpu_fallback` | (32, 64) | (16,) | `log` | float16 | log unsupported on Spyre; gather on Spyre, log on CPU; correct split | Gather on Spyre; log on CPU; output matches CPU reference | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_log_cpu_fallback |
| GDS-13 | `test_gather_softmax` | (32, 128) | (16,) | `softmax(dim=-1)` | float16 | Attention scoring: gather + softmax; softmax normalises over inner dim | Output = softmax(gathered rows); rows sum to 1.0; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_softmax |
| GDS-14 | `test_gather_abs_then_sum` | (32, 256) | (16,) | `abs → sum(dim=1)` | float16 | Two-stage downstream: unary then reduction; both co-scheduled | Output shape (16,); abs+sum values match CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_abs_then_sum |
| GDS-15 | `test_gather_scalar_mul_3d` | (8, 16, 64) | (4,) | `* 2.0` | float16 | Scalar multiply on 3D gathered output; broadcast over all inner dims | Output = gathered * 2.0; shape (4,16,64); matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_scalar_mul_3d |
| GDS-16 | `test_gather_then_add_broadcast_bias` | (64, 128) | (16,) | `+ bias (128,)` | float16 | Gather rows then add broadcast bias vector; common transformer pre-norm pattern | Output = gathered + bias; broadcast aligns; values match CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_then_add_broadcast_bias |
| GDS-17 | `test_gather_then_mul_and_add_residual` | (64, 128) | (16,) | `* gate + residual` | float16 | Gather rows, scale by per-row gate, add residual; MoE output accumulation pattern | Output = gathered * gate + residual; values match CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_gather_then_mul_and_add_residual |

---

## Section 6 — Named Dimensions

**Test class:** `TestGatherFusedDownstreamOperations`
**File:** `tests/inductor/indirect_access/test_gather_downstream.py`

> `declare_tensor_dim` and named dimensions are required by FR-4. The existing suite validates named dims on 2D (TestGather::test_gather_sdsc_fields). These cases cover broader naming patterns and combinations.
>
> **Indirect Access Validation Focus:** Validates the `declare_tensor_dim` (FR-4) annotation mechanism that enables the `propagate_named_dims` pass to identify value↔index tensor relationships. Named dimension labels on both the value tensor (e.g. `['M','N']`) and index tensor (e.g. `['P']`) must survive compilation and appear in FX graph buffer annotations; without them, the indirect access detection pass at `propagate_named_dims.py:187` cannot fire, and `GATHER_OP_SPEC` is never emitted even with the gate enabled.

| ID | Test Name | Input Shape | Index Shape | Value Named Dims | Index Named Dims | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|---|
| GNDM-01 | `test_named_2d_basic` | (32, 256) | (16,) | `['M','N']` | `['P']` | float16 | `name_tensor_dims` on 2D value + 1D index; baseline named-dim gather | Named dims present in compiled graph; output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_2d_basic |
| GNDM-02 | `test_named_3d_basic` | (8, 32, 128) | (4,) | `['A','B','C']` | `['P']` | float16 | 3D named dims; all three axes annotated; annotations survive compilation | Named dims on all 3 value axes; output (4,32,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_3d_basic |
| GNDM-03 | `test_named_kv_pattern` | (512, 8, 64) | (B*Lk,) | `['cache','H','D']` | `['B','Lk']` | float16 | Paged KV naming; 2D index with semantic names; canonical paged-attention annotation | Named dims present; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_kv_pattern |
| GNDM-04 | `test_named_embedding` | (32000, 512) | (32,) | `['vocab','dim']` | `['seq']` | float16 | Embedding table naming; vocabulary axis labeled; dim axis labeled | Named dims in graph; output (32,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_embedding |
| GNDM-05 | `test_named_moe` | (8, 512, 64) | (16,) | `['E','D','F']` | `['tokens']` | float16 | MoE expert weight naming; E=expert, D=hidden, F=feature; routing index named | Named dims in graph; output (16,512,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_moe |
| GNDM-06 | `test_named_stl_combo` | (32, 256) | (16,) | `['M','N']` | `['P']` | float16 | Named dims + explicit STL; both annotations co-exist in compiled graph | STL and named dims both present; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_stl_combo |
| GNDM-07 | `test_named_full_attention` | q:(B,Lq,H,D), k/v:(cache,H,D) | slot:(B,Lk) | `q:['B','Lq','H','D']`, `k/v:['cache','H','D']` | `['B','Lk']` | float16 | Full four-tensor attention naming (q, k, v, slot); all semantic names | All four tensors carry names in graph; attention output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_full_attention |
| GNDM-08 | `test_named_dims_propagate` | (32, 128) | (16,) | `['M','N']` | `['P']` | float16 | Named dims on value propagate through gather to output buffer | Output buffer also carries named dim annotations in FX graph | N/A | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_dims_propagate |
| GNDM-09 | `test_named_idx_only` | (32, 128) | (16,) | None (unnamed) | `['P']` | float16 | Partial naming: index named, value unnamed; verify index names respected | Index named dims preserved; value has no names; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_idx_only |
| GNDM-10 | `test_named_with_downstream` | (32, 128) | (16,) | `['M','N']` | `['P']` | float16 | Named dims preserved through gather → exp → sum chain; annotations survive multi-op fusion | Named dims present in all intermediate buffers; final output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_named_with_downstream |

---

## Section 7 — Execution Modes

**Test class:** `TestGatherFusedDownstreamOperations`
**File:** `tests/inductor/indirect_access/test_gather_downstream.py`

> Validates gather correctness across SENCORES settings, dynamic compilation, and environment variable gates.
>
> **Indirect Access Validation Focus:** Validates that the `GATHER_OP_SPEC` path is invariant across the full execution-mode space: indirect access is detected automatically via `MemoryDep.is_indirect()` and always emits `GATHER_OP_SPEC`; multi-core work division (SENCORES=1/4/32) must not produce partial-result merge errors; `dynamic=True` compilation must preserve the SDSC encoding for multiple runtime shapes; and the compile cache must produce separate graph entries for distinct execution configurations.

| ID | Test Name | Input Shape | Index Shape | Env / Config | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GEM-01 | `test_sencores_1_2d` | (32, 256) | (16,) | SENCORES=1 | float16 | Single-core gather; no work-division; SDSC correctly serialised | Output (16,256) matches CPU; no compile error | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_sencores_1_2d |
| GEM-02 | `test_sencores_4_2d` | (32, 256) | (16,) | SENCORES=4 | float16 | 4-core gather; work-division across 4 senCOREs; partial-gather merge correct | Output matches CPU; all 4 cores contribute correct partial results | atol=1e-2 | SENCORES=4 | TestGatherFusedDownstreamOperations::test_sencores_4_2d |
| GEM-03 | `test_sencores_32_2d` | (32, 256) | (16,) | SENCORES=32 | float16 | Full-chip 32-core gather; maximum parallelism; no correctness regression | Output matches CPU at full-chip scale | atol=1e-2 | SENCORES=32 | TestGatherFusedDownstreamOperations::test_sencores_32_2d |
| GEM-04 | `test_sencores_4_3d` | (8, 32, 128) | (4,) | SENCORES=4 | float16 | Multi-core 3D gather; work-division on 3D tensor; inner dims intact | Output (4,32,128) matches CPU | atol=1e-2 | SENCORES=4 | TestGatherFusedDownstreamOperations::test_sencores_4_3d |
| GEM-05 | `test_sencores_32_embedding` | (32000, 512) | (32,) | SENCORES=32 | float16 | Full-chip embedding table lookup; large V=32000 split across 32 cores | Output (32,512) matches CPU | atol=1e-2 | SENCORES=32 | TestGatherFusedDownstreamOperations::test_sencores_32_embedding |
| GEM-06 | `test_sencores_32_paged_kv` | (1024, 8, 64) | (32,) | SENCORES=32 | float16 | Full-chip 3D KV pool gather; paged slot access at scale | Output (32,8,64) matches CPU | atol=1e-2 | SENCORES=32 | TestGatherFusedDownstreamOperations::test_sencores_32_paged_kv |
| GEM-07 | `test_dynamic_compile` | (32, 256) | (P,) dynamic | `dynamic=True` | float16 | Symbolic P; compile once, run at P=8 and P=16; GATHER_OP_SPEC emitted for both shapes | Output matches CPU at each runtime P value | atol=1e-2 | SENCORES=1 | TestGatherFusedDownstreamOperations::test_dynamic_compile |
| GEM-08 | `test_fusion_disabled` | (32, 256) | (16,) | ENABLE_FUSION=0 | float16 | Downstream fusion disabled; gather and downstream appear as separate kernel specs | Gather and exp are separate nodes in compiled graph; output still matches CPU | atol=1e-2 | ENABLE_FUSION=0 | TestGatherFusedDownstreamOperations::test_fusion_disabled |
| GEM-09 | `test_lx_planning_enabled` | (32, 256) | (16,) | LX_PLANNING=1 | float16 | LX scratchpad planning active; downstream buffer eligible for LX but index must remain in HBM | Index buffer allocated in HBM (not LX); downstream output in LX if eligible; output matches CPU | atol=1e-2 | LX_PLANNING=1 | TestGatherFusedDownstreamOperations::test_lx_planning_enabled |
| GEM-10 | `test_compile_cache_hit` | (32, 256) | (16,) | default | float16 | Compile same function twice; second call should use cached graph; no recompile triggered | Compile count = 1; both calls return identical output | exact | SENCORES=1 | TestGatherFusedDownstreamOperations::test_compile_cache_hit |

---

## Section 8 — Embedding Table Lookup

**Test class:** `TestGatherEmbeddingTableLookupAndPooling`
**File:** `tests/inductor/indirect_access/test_gather_embedding.py`

> The existing suite validates vocabulary size V=32000, D=512, B=2, S=128 with int64 indices, and the `F.embedding`/`nn.Embedding` CPU fallback paths. These cases add new vocab sizes, batch configurations, RoPE patterns, and dtype variants.
>
> **Indirect Access Validation Focus:** Validates the primary real-model indirect access pattern — vocabulary embedding lookup via `weight[token_ids]` (advanced indexing) or `index_select(0, ids)` — across LLM-scale table sizes (GPT-2: V=50257, Llama-3: V=128256), multi-token batches, bfloat16 weights, RoPE cosine-sine cache lookup, and weight-tied LM-head shared matrix. Also confirms the `NO_SPYRE_OP` / CPU-fallback classification for `F.embedding` and `nn.Embedding` (which route through `aten.embedding.default`, not `aten.index.Tensor`), ensuring these two API forms are not mistakenly emitting `GATHER_OP_SPEC`.

| ID | Test Name | Table Shape (V, D) | Index Shape | Embedding API | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GEMB-01 | `test_embedding_small_vocab` | (512, 64) | (32,) | `weight[idx]` (advanced indexing) | float16 | Small vocabulary advanced-index gather on Spyre; GATHER_OP_SPEC emitted | Output (32,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_small_vocab |
| GEMB-02 | `test_embedding_medium_vocab` | (8192, 128) | (64,) | `weight[idx]` | float16 | Medium vocabulary; 128-dim embeddings; stick-aligned layout | Output (64,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_medium_vocab |
| GEMB-03 | `test_embedding_gpt2_vocab` | (50257, 768) | (16,) | `weight[idx]` | float16 | GPT-2 vocabulary (V=50257, D=768 = 12 sticks); large table gather | Output (16,768) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_gpt2_vocab |
| GEMB-04 | `test_embedding_llama3_vocab` | (128256, 4096) | (8,) | `weight[idx]` | float16 | Llama-3 vocabulary (V=128256, D=4096 = 64 sticks); very large table | Output (8,4096) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_llama3_vocab |
| GEMB-05 | `test_embedding_batch_2d_ids` | (32000, 512) | (4, 128) | `weight[idx]` | float16 | 2D token IDs (B=4, S=128); output shape (4,128,512); batched sequence lookup | Output (4,128,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_batch_2d_ids |
| GEMB-06 | `test_embedding_batch_large` | (32000, 512) | (8, 512) | `weight[idx]` | float16 | Large batch (B=8, S=512 = 4096 lookups); maximum-scale embedding | Output (8,512,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_batch_large |
| GEMB-07 | `test_embedding_positional` | (2048, 512) | (128,) | `pos_emb[pos_ids]` (index_select) | float16 | Positional encoding table; `index_select(0, pos_ids)` pattern | Output (128,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_positional |
| GEMB-08 | `test_embedding_rope_cos_sin` | (4096, 128) | (seq,) | `cos_sin_cache[positions]` (index_select) | float16 | RoPE cosine-sine cache lookup; `index_select(0, positions)` produces (seq,128) for downstream `.chunk(2,-1)` | Output (seq,128) matches CPU; chunk into (seq,64) cos and sin correct | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_rope_cos_sin |
| GEMB-09 | `test_embedding_bfloat16` | (4096, 64) | (32,) | `weight[idx]` | bfloat16 | bfloat16 embedding table; SDSC wordLength=2; GATHER_OP_SPEC for bfloat16 | Output (32,64) matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_bfloat16 |
| GEMB-10 | `test_embedding_float32` | (4096, 64) | (32,) | `weight[idx]` | float32 | float32 embedding table; 4-byte elements; GATHER_OP_SPEC for float32 | Output (32,64) matches CPU float32 | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_float32 |
| GEMB-11 | `test_embedding_int64_ids` | (32000, 512) | (32,) int64 | `weight[idx]` | float16 | int64 token IDs auto-downcast to int32; embedding lookup still correct | Output matches CPU; downcast warning emitted | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_int64_ids |
| GEMB-12 | `test_embedding_with_exp` | (4096, 64) | (32,) | `weight[idx]` + `exp()` | float16 | Embedding lookup + exp() downstream co-scheduled; fused GATHER+exp kernel | Output = exp(embedded); matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_with_exp |
| GEMB-13 | `test_embedding_functional_api` | (4096, 64) | (32,) | `F.embedding(idx, weight)` | float16 | `F.embedding` maps to `aten.embedding` → NO_SPYRE_OP; CPU fallback; no GATHER_OP_SPEC | FX graph shows CPU fallback for `aten.embedding`; output correct | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_functional_api |
| GEMB-14 | `test_embedding_nn_module` | (4096, 64) | (32,) | `nn.Embedding.forward` | float16 | `nn.Embedding` forward → `aten.embedding` → CPU fallback; NO_SPYRE_OP documented path | No GATHER_OP_SPEC for `aten.embedding`; CPU fallback correct | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_nn_module |
| GEMB-15 | `test_embedding_tied_lm_head` | (vocab=4096, dim=64) | (seq=32,) | `weight[idx]` (embedding) + `linear(h, weight)` (lm_head) | float16 | Weight-tied model: same matrix used for both input embedding and LM head; weight sharing intact | Both embedding and LM-head outputs match CPU; weight is not duplicated | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_tied_lm_head |

---

## Section 9 — MoE Expert Routing

**Test class:** `TestGatherProductionInferencePatterns`
**File:** `tests/inductor/indirect_access/test_gather_production.py`

> The existing `test_moe` covers a single expert_w shape with B×S int64 IDs. These cases add expert count variations, real model shapes, and the full routing + aggregation pattern.
>
> **Indirect Access Validation Focus:** Validates the Mixture-of-Experts indirect access pattern where `expert_weights[expert_ids]` selects 3D expert weight rows (shape `(E, D_in, D_out)`) based on router output — the canonical "dispatch" step in every MoE model. Tests cover 4-to-256 expert counts, top-k routing with flattened IDs, bfloat16 weights, multi-core partitioning, Mixtral-8x7B and DeepSeek-MoE production shapes, and the full routing+aggregation pipeline (gather → multiply by gating weight → sum). A bug in the 3D gather or SDSC stride-map encoding for the inner `(D_in, D_out)` plane would produce wrong expert outputs silently.

| ID | Test Name | Expert Weight Shape (E, D_in, D_out) | Router Index Shape | num_experts | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GMOE-01 | `test_moe_small` | (4, 64, 32) | (8,) | 4 | float16 | Minimal MoE: 4 experts, 8 tokens routed; `expert_w[expert_ids]` GATHER_OP_SPEC | Output (8,64,32) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_small |
| GMOE-02 | `test_moe_large_expert_count` | (64, 512, 64) | (32,) | 64 | float16 | 64 experts; large expert count; SDSC correctly scaled | Output (32,512,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_large_expert_count |
| GMOE-03 | `test_moe_256_experts` | (256, 256, 32) | (16,) | 256 | float16 | DeepSeek-MoE style: 256 experts; M=256 at max scale | Output (16,256,32) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_256_experts |
| GMOE-04 | `test_moe_topk_routing` | (8, 512, 64) | (tokens×2,) | 8 | float16 | Top-k=2 routing: 2 expert IDs per token flattened; duplicates handled correctly | Output shape (tokens×2, 512, 64); matches CPU for each routed expert | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_topk_routing |
| GMOE-05 | `test_moe_bfloat16` | (8, 512, 64) | (32,) | 8 | bfloat16 | bfloat16 expert weights; SDSC wordLength=2; GATHER_OP_SPEC for bfloat16 | Output (32,512,64) matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_bfloat16 |
| GMOE-06 | `test_moe_sencores4` | (8, 512, 64) | (32,) | 8 | float16 | 4-core MoE gather; work-division across SENCORES=4 | Output matches CPU; all 4 cores contribute | atol=1e-2 | SENCORES=4 | TestGatherProductionInferencePatterns::test_moe_sencores4 |
| GMOE-07 | `test_moe_mixtral_shape` | (8, 4096, 1024) | (32,) | 8 | float16 | Mixtral-8x7B approximated expert shape (E=8, D_in=4096, D_out=1024) | Output (32,4096,1024) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_mixtral_shape |
| GMOE-08 | `test_moe_deepseek_shape` | (256, 1024, 256) | (16,) | 256 | float16 | DeepSeek-MoE large-expert-count shape (E=256, D_in=1024, D_out=256) | Output (16,1024,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_deepseek_shape |
| GMOE-09 | `test_moe_expert_aggregation` | (8, 512, 64) | (32,) | 8 | float16 | Route via gather then aggregate weighted sum; gather feeds into pointwise multiply + sum | Aggregated output shape (tokens, 512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_expert_aggregation |
| GMOE-10 | `test_moe_multi_layer` | (8, 512, 64) | (32,) each | 8 | float16 | Two consecutive MoE routing layers in one compiled graph; two GATHER_OP_SPECs | Both gather outputs correct; two SDSC bundles in compiled graph | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_moe_multi_layer |

---

## Section 10 — Paged Attention and KV Cache

**Test class:** `TestGatherProductionInferencePatterns`
**File:** `tests/inductor/indirect_access/test_gather_production.py`

> The existing `test_paged_kv` validates cache_size=32768, H=32, Lq=256, Lk=256, D=128 at SENCORES=1. These cases add pool size variations, GQA/MQA, decode/prefill phases, chunked prefill, online softmax, FMS 4D/5D layouts, and vLLM block_table patterns.
>
> **Indirect Access Validation Focus:** Validates the paged KV-cache gather (`keys = kv_pool[slot_idxs]`) — the central indirect access operation in production LLM serving — where a 3D pool of `(pool_size, kv_heads, head_dim)` is indexed by a 2D slot map `(B, Lk)` to retrieve the exact KV entries for each active decode or prefill position. A wrong SDSC `device_size` or `stride_map` for the `(kv_heads, head_dim)` inner plane silently returns values from adjacent heads, causing numerically plausible but incorrect attention. Tests cover the full serving complexity: pool sizes from 256 to 32768, GQA/MQA head ratios, block-table derived indices, page boundary correctness, FMS 4D/5D cache layouts with `dim_order` reordering, and dynamic sequence lengths.

| ID | Test Name | KV Cache Shape (pool, H, D) | Slot Index Shape (B, Lk) | Config | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GPA-01 | `test_paged_kv_small_pool` | (256, 8, 64) | (4, 32) | — | float16 | Small pool (256 slots); `keys=cache[slot_idxs]`; GATHER_OP_SPEC on 3D cache | Output (4,32,8,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_small_pool |
| GPA-02 | `test_paged_kv_medium_pool` | (1024, 8, 64) | (2, 64) | — | float16 | 1024-slot pool; wider slot map; correct slot-to-cache mapping | Output (2,64,8,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_medium_pool |
| GPA-03 | `test_paged_kv_large_pool` | (8192, 16, 128) | (8, 128) | — | float16 | 8K-slot pool; H=16, D=128; large-scale production-like KV pool | Output (8,128,16,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_large_pool |
| GPA-04 | `test_paged_kv_decode_single` | (512, 8, 64) | (12, 1) | decode step | float16 | Single decode step: B=12, one slot per request; Lk=1 index | Output (12,1,8,64) matches CPU; single-token KV read | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_decode_single |
| GPA-05 | `test_paged_kv_prefill` | (512, 8, 64) | (1, 256) | prefill | float16 | Prefill: B=1, full sequence length Lk=256; contiguous slot range | Output (1,256,8,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_prefill |
| GPA-06 | `test_paged_kv_chunked_prefill` | (512, 8, 64) | (1, 64) | chunk_size=64 | float16 | Chunked prefill: 4 chunks × 64 slots; each chunk a separate gather call | Each chunk output (1,64,8,64) matches CPU; results concatenated correctly | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_chunked_prefill |
| GPA-07 | `test_paged_kv_gqa` | (1024, 8, 64) | (4, 64) | kv_heads=8, q_heads=32 | float16 | GQA (group-query attention): 4× head ratio; cache gathered at kv_heads=8, expanded for 32 q_heads | K/V output (4,64,8,64) correct; attention with head expansion matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_gqa |
| GPA-08 | `test_paged_kv_mqa` | (1024, 1, 64) | (4, 64) | kv_heads=1, q_heads=32 | float16 | MQA (multi-query attention): single KV head; gather produces (4,64,1,64) | Output (4,64,1,64) matches CPU; MQA head broadcast correct | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_mqa |
| GPA-09 | `test_paged_kv_head_dim_128` | (512, 16, 128) | (4, 64) | D=128 | float16 | Head dim D=128 (2 sticks per head); standard Llama-style dimension | Output (4,64,16,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_head_dim_128 |
| GPA-10 | `test_paged_kv_head_dim_256` | (512, 8, 256) | (4, 32) | D=256 | float16 | Head dim D=256 (4 sticks per head); large head dim correctness | Output (4,32,8,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_head_dim_256 |
| GPA-11 | `test_paged_kv_bfloat16` | (512, 8, 64) | (4, 32) | — | bfloat16 | bfloat16 KV cache; SDSC wordLength=2; gather with bfloat16 slots | Output (4,32,8,64) matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_bfloat16 |
| GPA-12 | `test_paged_kv_block_table` | (512, 8, 64) | (B, Tk) 2D | block_table + offset | float16 | 2D block_table: `slot_idxs = block_table[b, t] * page_size + offset`; derived index | Slot-derived gather output matches CPU; block boundary correct | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_block_table |
| GPA-13 | `test_paged_kv_page_reuse` | (512, 8, 64) | (4, 32) | prefix sharing | float16 | Same page referenced by multiple requests (prefix caching); repeated-slot gather | Multiple requests sharing prefix page produce identical KV outputs | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_page_reuse |
| GPA-14 | `test_paged_kv_online_softmax` | (512, 8, 64) | (4, 64) | — | float16 | K gather + online softmax (running max/denominator); Flash-Attention style accumulation | Attention scores with online softmax match CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_online_softmax |
| GPA-15 | `test_paged_kv_full_attention` | (512, 8, 64) | (4, 64) | full attn | float16 | Full paged attention: gather K + gather V + online softmax + weighted V sum; two GATHER_OP_SPECs | Attention output matches CPU; two SDSC bundles in compiled graph | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_full_attention |
| GPA-16 | `test_paged_kv_named_dims` | (512, 8, 64) | (4, 64) | named dims | float16 | Named dims on K/V (`['cache','H','D']`) and slot_idxs (`['B','Lk']`); all annotations present | All named dims in FX graph; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_named_dims |
| GPA-17 | `test_paged_kv_with_stl` | (512, 8, 64) | (4, 32) | STL | float16 | Explicit SpyreTensorLayout on KV cache; 4-arg form with `dim_order`; SDSC layout correct | STL annotation in compiled graph; output (4,32,8,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_with_stl |
| GPA-18 | `test_paged_kv_sencores4` | (512, 8, 64) | (4, 32) | SENCORES=4 | float16 | Paged KV gather across 4 cores; work-division for 3D tensor | Output (4,32,8,64) matches CPU; 4-core partition | atol=1e-2 | SENCORES=4 | TestGatherProductionInferencePatterns::test_paged_kv_sencores4 |
| GPA-19 | `test_paged_kv_sencores32` | (1024, 8, 64) | (8, 128) | SENCORES=32 | float16 | Full-chip 32-core paged attention; production-scale parallelism | Output matches CPU at SENCORES=32 | atol=1e-2 | SENCORES=32 | TestGatherProductionInferencePatterns::test_paged_kv_sencores32 |
| GPA-20 | `test_paged_kv_lx_planning` | (512, 8, 64) | (4, 32) | LX_PLANNING=1 | float16 | LX-eligible downstream after KV gather; slot_idxs must stay in HBM not LX | slot_idxs in HBM; KV output in LX if eligible; output matches CPU | atol=1e-2 | LX_PLANNING=1 | TestGatherProductionInferencePatterns::test_paged_kv_lx_planning |
| GPA-21 | `test_paged_kv_two_gathers` | (512, 8, 64) | (4, 32) | dual gather | float16 | K and V gathered from same pool with same slots; two independent GATHER_OP_SPECs | Two SDSC bundles in compiled graph; K and V outputs both match CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_two_gathers |
| GPA-22 | `test_paged_kv_dynamic_seqlen` | (512, 8, 64) | (B, Lk) dynamic | dynamic=True | float16 | Dynamic Lk at inference; compile once, run at Lk=32 and Lk=64 | Outputs match CPU at each runtime Lk value | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_dynamic_seqlen |
| GPA-23 | `test_paged_kv_4d_fms_layout` | (pages, kv_h, page_sz, head_d) | (B, Lk) | FMS 4D | float16 | FMS-style 4D KV layout: page as first dim, kv_heads second; STL with dim_order reordering | Output matches CPU with FMS 4D layout | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_4d_fms_layout |
| GPA-24 | `test_paged_kv_5d_split_kv` | (pages, 2, kv_h, page_sz, head_d) | (B, Lk) | FMS 5D | float16 | FMS 5D: combined K+V along dim=1, split after gather; single gather + split downstream | K and V outputs after split match CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_5d_split_kv |
| GPA-25 | `test_paged_kv_sentinel_boundary` | (512, 8, 64) | (4, 32) | boundary | float16 | Values at page boundaries (last slot of a page = slot % page_size == page_size-1); no off-by-one | Boundary slot values match CPU; no page-crossing corruption | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_paged_kv_sentinel_boundary |
| GPA-26 | `test_float32_kv_cache_access` | (512, 8, 64) | (32,) int64 | 0 | float32 | float32 3D KV pool (512,8,64) gather; 4-byte element, int64 slot index; wordLength=4 | Output shape (32,8,64); float32 values match CPU | atol=1e-5 | SENCORES=1 | TestGatherProductionInferencePatterns::test_float32_kv_cache_access |

---

## Section 11 — vLLM Serving Patterns

**Test class:** `TestGatherProductionInferencePatterns`
**File:** `tests/inductor/indirect_access/test_gather_production.py`

> Mimics the six core indirect access scenarios from vLLM serving (INDIRECT-001 through INDIRECT-006) and additional vLLM-specific patterns. Target models: Llama-3.1-8B, Mistral-Small-3.2-24B, Granite-3.3-8b, gpt-oss-20b.
>
> **Indirect Access Validation Focus:** Validates all indirect access touch-points in the vLLM serving loop: on-device token embedding lookup (replacing the CPU `aten.embedding` fallback), paged KV cache write (`index_put_`), prefill/decode KV read (sequential vs paged slots), decode-step last-token logit extraction, ragged-batch per-request cache indexing, and beam search KV reorder via `index_select` at the batch dimension. Each test confirms a specific `GATHER_OP_SPEC` emission that eliminates a host-device round-trip and therefore directly impacts inference latency on Spyre.

| ID | Test Name | Tensor Shape | Index Shape | Gather Pattern | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GVLLM-01 | `test_vllm_token_embedding` | (32000, 4096) | (seq=128,) | `embedding_table[token_ids]` | float16 | On-device token embedding lookup (issue #420); GATHER_OP_SPEC instead of CPU aten.embedding | Output (128,4096) matches CPU; GATHER_OP_SPEC in graph | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_token_embedding |
| GVLLM-02 | `test_vllm_kv_cache_write` | (1,8,2048,64) | seq positions | `index_put + copy_` → `index_put_` FX transform | float16 | KV cache write; FX graph transforms `index_put + copy_` into in-place `index_put_` | FX graph shows `index_put_`; written positions match expected values | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_kv_cache_write |
| GVLLM-03 | `test_vllm_kv_prefill_read` | (512, 8, 64) | (1, 256) contiguous | `cache[:, :, :seq_len, :]` sequential | float16 | Prefill KV read (sequential, contiguous slots); GATHER_OP_SPEC on slot range | Output (1,256,8,64) matches CPU; contiguous slot access | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_kv_prefill_read |
| GVLLM-04 | `test_vllm_kv_decode_read` | (512, 8, 64) | (12, 1) paged | `cache[slot_idxs]` — paged slot gather | float16 | Decode read via paged slot indices; single-step decode B=12, Lk=1 | Output (12,1,8,64) matches CPU slot mapping | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_kv_decode_read |
| GVLLM-05 | `test_vllm_logit_extraction` | (B=12, S=1, V=32000) | (B,) last-pos | `logits[:, -1, :]` last-token indexing | float16 | Decode-step last-token logit extraction; advanced indexing on sequence dim | Output (12,32000) matches CPU last-token slice | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_logit_extraction |
| GVLLM-06 | `test_vllm_batch_variable` | (512, 8, 64) | ragged (per-request lengths) | Per-request gather with variable active lengths | float16 | Ragged-batch inference; each request has different active cache length; correct masking | Each request output matches its expected CPU result | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_batch_variable |
| GVLLM-07 | `test_vllm_4d_kv_layout` | (num_blocks, block_sz, kv_h, head_d) | (B, Lk) | vLLM native 4D: block-first layout | float16 | vLLM native KV layout (block_size as second dim); STL with dim_order=[0,1,2,3]; gather at dim=0 | Output matches CPU with vLLM 4D layout | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_4d_kv_layout |
| GVLLM-08 | `test_vllm_prefill_decode_combined` | (512, 8, 64) | mixed (B prefill + B decode) | Mixed batch: prefill slots + decode slots in same gather | float16 | Combined prefill+decode batch; different slot lengths per request in one call | Each request's output matches CPU for its length | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_prefill_decode_combined |
| GVLLM-09 | `test_vllm_rope_position_lookup` | (4096, 128) | (seq=128,) | `cos_sin_cache.index_select(0, positions)` | float16 | RoPE position cache lookup (SpyreRotaryEmbedding); `index_select` → GATHER_OP_SPEC | Output (128,128) matches CPU; then `.chunk(2,-1)` produces correct cos/sin | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_rope_position_lookup |
| GVLLM-10 | `test_vllm_beam_search_reorder` | (B=4, cache_len, 8, 64) | beam_idx (B,) | `past_kv.index_select(0, beam_idx)` — beam reorder | float16 | Beam search KV reordering; `index_select` at batch dim; all beams correct | Output (4, cache_len, 8, 64) matches CPU beam reorder | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_beam_search_reorder |
| GVLLM-11 | `test_vllm_llama31_8b` | kv:(pool, 8, 128), emb:(128256, 4096) | slot:(B,Lk), tok:(seq,) | Dual gather: token emb + KV cache | float16 | Llama-3.1-8B shapes (H=8, D=128, V=128256); two GATHER_OP_SPECs in one graph | Both gather outputs match CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_llama31_8b |
| GVLLM-12 | `test_vllm_mistral_small_24b` | kv:(pool, 16, 128) | (B, Lk) sliding window | Sliding window KV gather | float16 | Mistral-Small-3.2-24B sliding-window attention; KV gathered from sliding window slots only | Output for window positions matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_mistral_small_24b |
| GVLLM-13 | `test_vllm_granite_8b` | kv:(pool, 8, 128) | (B, Lk) GQA | GQA paged KV: kv_heads=8, q_heads=32 | float16 | Granite-3.3-8b GQA (ratio=4); K/V gathered at kv_heads=8; head broadcast for 32 q_heads | Attention output with head expansion matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_granite_8b |
| GVLLM-14 | `test_vllm_continuous_batching` | (pool, 8, 64) | (B, variable_Lk) | Per-request variable active lengths in a batch | float16 | Continuous batching: B requests each with a different number of active KV positions | Each request output matches CPU for its active length | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_continuous_batching |
| GVLLM-15 | `test_vllm_speculative_decode` | (V=32000, D=64) | (draft_tokens=5,) | Draft token distribution gather | float16 | Speculative decode: gather target probs at draft token positions for verification | Output (5,) matches CPU gather of target probs | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_vllm_speculative_decode |

---

## Section 12 — FMS Serving Patterns

**Test class:** `TestGatherProductionInferencePatterns`
**File:** `tests/inductor/indirect_access/test_gather_production.py`

> Foundation Model Stack (FMS) uses distinct KV cache layouts: 4D `(num_pages, kv_heads, page_size, head_dim)` and 5D `(pages, 2, kv_heads, page_size, head_dim)` for combined K+V. Explicit STL required for both.
>
> **Indirect Access Validation Focus:** Validates FMS-specific KV cache indirect access, where the memory layout differs from the standard 3D vLLM format and requires the 4-argument `SpyreTensorLayout(device_size, stride_map, dtype, dim_order)` to reorder axes before the SDSC `device_coordinates` field is computed. Specifically: FMS places `kv_heads` before `page_size` in memory but wants gather at the page (outermost) axis, requiring `dim_order=[0,2,1,3]` to swap them. A missing or incorrect `dim_order` would make the compiler read across kv-head boundaries instead of page boundaries, silently returning wrong KV values.

| ID | Test Name | KV Cache Shape | Slot Index Shape | STL dim_order | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GFMS-01 | `test_fms_4d_layout` | (128, 8, 16, 128) — (pages, kv_h, page_sz, D) | (B=4, Lk=16) | `[0,2,1,3]` — kv_heads third | float16 | FMS 4D KV layout: STL with dim_order reorders kv_heads; gather pages at dim=0 | Output (4,16,8,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_fms_4d_layout |
| GFMS-02 | `test_fms_5d_split_kv` | (128, 2, 8, 16, 128) — (pages, 2, kv_h, page_sz, D) | (B=4, Lk=16) | `[0,2,1,3,4]` | float16 | FMS 5D combined K+V; dim=1 selects K or V; gather at dim=0 then split on dim=1 | K output (4,16,8,128) and V output (4,16,8,128) match CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_fms_5d_split_kv |
| GFMS-03 | `test_fms_cache_seqlens` | (128, 8, 16, 128) | ragged (B, variable_Lk) | `[0,2,1,3]` | float16 | Per-request `cache_seqlens`; gather only up to valid cache length per request | Each request's output matches CPU for its valid length only | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_fms_cache_seqlens |
| GFMS-04 | `test_fms_gqa_8kv_heads` | (128, 8, 16, 128) | (B=4, Lk=64) | `[0,2,1,3]` | float16 | GQA ratio=4: kv_heads=8, q_heads=32; KV gathered at kv_heads, broadcast for q_heads | Attention output with GQA head broadcast matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_fms_gqa_8kv_heads |
| GFMS-05 | `test_fms_decode_granite` | (128, 8, 16, 128) — Granite-3.3-8b shape | (B=1, Lk=1) — decode | `[0,2,1,3]` | float16 | Granite-3.3-8b single decode step with FMS layout; H=8, D=128, page_sz=16 | Output (1,1,8,128) matches CPU Granite decode | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_fms_decode_granite |
| GFMS-06 | `test_fms_bfloat16` | (128, 8, 16, 128) | (B=4, Lk=16) | `[0,2,1,3]` | bfloat16 | bfloat16 KV cache with FMS 4D layout; SDSC wordLength=2 | Output matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_fms_bfloat16 |
| GFMS-07 | `test_fms_named_dims` | (128, 8, 16, 128) | (B=4, Lk=16) | `[0,2,1,3]` | float16 | Named dims on FMS layout: `['pages','kv_h','page_sz','D']` on KV, `['B','Lk']` on slots | All named dims in compiled graph; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_fms_named_dims |
| GFMS-08 | `test_fms_sencores32` | (128, 8, 16, 128) | (B=4, Lk=64) | `[0,2,1,3]` | float16 | Full-chip 32-core FMS gather; SENCORES=32 with 4D layout | Output matches CPU at SENCORES=32 | atol=1e-2 | SENCORES=32 | TestGatherProductionInferencePatterns::test_fms_sencores32 |

---

## Section 13 — Real Model Patterns

**Test class:** `TestGatherProductionInferencePatterns`
**File:** `tests/inductor/indirect_access/test_gather_production.py`

> Derived from development team files (`files-shared-by-development/test-cases-to-consider.md`, `vLLM/vllm_spyre_issue322.md`). These use actual tensor shapes from Q2/Q3 target models.
>
> **Indirect Access Validation Focus:** Validates exact production model shapes extracted from Q2/Q3 target models (gpt-oss-20b, Granite-3.3-8b, Ministral-3-14B, Mistral-Small-3.2-24B, Mistral multimodal), covering both the indirect write path (`index_copy_` KV cache fill at specific sequence positions) and read paths (`index_select` for RoPE cache, advanced indexing for last-token extraction, paged slot gather). These tests are the tightest regression targets: if any stride-map computation or SDSC encoding changes, these shapes — which have exact per-model stride tuples — will catch it before production deployment.

| ID | Test Name | Cache / Source Shape | Index / Position Shape | Model | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GMDL-01 | `test_index_copy_gpt20b_prefill` | cache (1,8,2048,64); key_states (1,8,11,64) | cache_pos (11,) | gpt-oss-20b | float16 | `index_copy_(2, cache_pos, key)` prefill; writes 11 tokens into position dim=2 of cache | Cache positions [0..10] contain key_states values; other positions unchanged | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_index_copy_gpt20b_prefill |
| GMDL-02 | `test_index_copy_gpt20b_decode` | cache (1,8,2048,64); key_states (1,8,1,64) | cache_pos (1,) | gpt-oss-20b | float16 | `index_copy_(2, cache_pos, key)` single decode step; one position written | Cache position `cache_pos[0]` on dim=2 updated; values match | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_index_copy_gpt20b_decode |
| GMDL-03 | `test_index_copy_granite_8b` | cache (1,8,2048,128); key_states (1,8,41,128) | cache_position (41,) | Granite-3.3-8b | float16 | Granite `index_copy_(2, cache_position, key_states)` from HF cache_utils; 41-token prefill | 41 cache positions updated correctly on dim=2 | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_index_copy_granite_8b |
| GMDL-04 | `test_index_copy_ministral_14b` | cache (1,8,2048,128); key_states (1,8,14,128) | cache_position (14,) | Ministral-3-14B | float16 | Ministral-3-14B-Instruct-2512 `index_copy_` pattern; 14-token prefill | 14 positions updated; values match key_states | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_index_copy_ministral_14b |
| GMDL-05 | `test_index_copy_mistral_24b` | cache (1,8,2048,128); key_states (1,8,855,128) | cache_position (855,) | Mistral-Small-3.2-24B | float16 | Mistral-Small-3.2-24B-Instruct-2506 `index_copy_`; 855-token long prefill | 855 cache positions updated; values match key_states | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_index_copy_mistral_24b |
| GMDL-06 | `test_masked_scatter_mistral_multimodal` | inputs_embeds (1,855,5120); image_features (266,5120) | mask (1,855,5120) bool | Mistral-Small multimodal | float16 | `inputs_embeds.masked_scatter(mask, image_features)` token injection; 266 image tokens injected | Post-scatter tensor has image_features at mask-True positions; rest unchanged | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_masked_scatter_mistral_multimodal |
| GMDL-07 | `test_rope_index_select` | cos_sin_cache (4096, 128) | positions (seq=128,) | SpyreRotaryEmbedding | float16 | `cos_sin_cache.index_select(0, positions)` → GATHER_OP_SPEC; resolves issue #1668 | Output (128,128) matches CPU; GATHER_OP_SPEC emitted | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_rope_index_select |
| GMDL-08 | `test_lm_head_logit_extraction` | hidden (B=12,S=1,D=4096) | last_pos_idx (B,) | LM head decode | float16 | `hidden[:, -1, :]` or `hidden[b, last_pos, :]` last-token extraction; advanced indexing | Output (12,4096) matches CPU last-token slice | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_lm_head_logit_extraction |
| GMDL-09 | `test_kv_cache_fill_granite` | cache (1,8,128,64) | seq positions (41,) | Granite prefill | float16 | Granite initial KV cache fill during prefill; write path; `index_copy_` on dim=2 | Cache updated at 41 positions; full read-back matches key_states | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_kv_cache_fill_granite |
| GMDL-10 | `test_kv_cache_decode_growing` | cache (1,8,2048,64) | position (step=1,) | Granite decode | float16 | Growing KV cache read: decode position increments each step; test step 0, 10, 100 | Output at each step position matches the value written there | exact | SENCORES=1 | TestGatherProductionInferencePatterns::test_kv_cache_decode_growing |
| GMDL-11 | `test_attention_qkv_split` | qkv_proj (B=1,S=1,3*4096) | — (split, no index) | QKVParallelLinear | float16 | `qkv.split([q_size, k_size, v_size], dim=-1)`; slice-based Q/K/V separation | Q,K,V slices match CPU split; no gather needed for split pattern | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_attention_qkv_split |
| GMDL-12 | `test_token_unpadding` | packed (total_tokens=512, D=4096) | position_ids (B=4,S=128) | Unpadded forward | float16 | Packed/unpadded token gather: each sequence position mapped from flat packed buffer | Output (4,128,4096) with each sequence's tokens in correct positions; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_token_unpadding |
| GMDL-13 | `test_token_repadding` | unpadded (total_tokens, D) | scatter_positions | Repadding | float16 | Repadding after unpadded forward: scatter tokens back to (B,S,D) padded format | Output (B,S,D) has tokens at correct positions; padded positions zeroed | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_token_repadding |
| GMDL-14 | `test_topk_moe_routing_top1` | expert_weights:(8,512,64) | top1_ids:(N_tokens,) from topk | Top-1 MoE routing | float16 | Top-1 expert selection via `scores.topk(1)`; expert_ids = argmax; gather selects single expert per token | Output (N_tokens,512,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_topk_moe_routing_top1 |
| GMDL-15 | `test_topk_moe_routing_top2` | expert_weights:(8,512,64) | top2_ids:(N_tokens,2) from topk | Top-2 MoE routing | float16 | Top-2 expert selection; two expert weight rows gathered per token; two GATHER_OP_SPECs | Output (N_tokens,2,512,64) matches CPU top-2 gather | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_topk_moe_routing_top2 |
| GMDL-16 | `test_topk_moe_weighted_routing` | expert_weights:(8,512,64) | top2_ids + top2_weights from topk | Weighted MoE aggregate | float16 | Top-2 routing with per-expert weights; gather + multiply by softmax weight + sum; weighted aggregation | Aggregated output matches CPU weighted expert sum | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_topk_moe_weighted_routing |
| GMDL-17 | `test_topk_speculative_candidate_embed` | embed_table:(32000,512) | candidate_ids:(B=4,K=5) topk candidates | Speculative decode embed | float16 | Speculative decode: gather embedding at K=5 candidate tokens per request; (4,5,512) output | Output matches CPU embedding at candidate positions | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_topk_speculative_candidate_embed |
| GMDL-18 | `test_topk_beam_search_kv_reorder` | kv_cache:(B=4,Lk,8,64) | beam_idx:(4,) from topk-beam | Beam KV reorder | float16 | Beam search KV reorder via gather: `kv_cache[beam_idx]` selects best-beam caches; `index_select` at batch dim | Output (4,Lk,8,64) reordered by beam matches CPU | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_topk_beam_search_kv_reorder |
| GMDL-19 | `test_multi_layer_shared_slot_index` | kv_pool:(32, pool,8,64) — 32 layers | slot_idxs:(B,Lk) shared | Multi-layer paged KV | float16 | Same slot_idxs tensor reused across 32 KV layers; 32 GATHER_OP_SPECs with one shared index | All 32 layer outputs match CPU; one index broadcast to all layers | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_multi_layer_shared_slot_index |
| GMDL-20 | `test_multi_layer_growing_decode_history` | kv_per_layer:(pool,8,64) × L layers | slot grows step-by-step | Growing decode KV history | float16 | Decode with growing context: at each decode step, gather covers one more KV position across all layers | All L gather outputs correct at each step; growing index handled | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_multi_layer_growing_decode_history |
| GMDL-21 | `test_multi_layer_4d_kv_per_layer` | 4D kv:(pool, kv_h, page_sz, D) × L layers | slot:(B,Lk) | Multi-layer 4D FMS KV | float16 | L layers of FMS 4D KV cache gathered with STL dim_order; all layers use same slot map | All layer outputs match CPU with 4D layout | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_multi_layer_4d_kv_per_layer |
| GMDL-22 | `test_multi_layer_bfloat16_kv` | kv:(pool,8,64) × L layers bfloat16 | slot:(B,Lk) | Multi-layer bf16 KV | bfloat16 | bfloat16 KV across L layers; SDSC wordLength=2 for each layer; correct bfloat16 output | All layer bfloat16 outputs match CPU | atol=2e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_multi_layer_bfloat16_kv |
| GMDL-23 | `test_multi_layer_parallel_k_v_per_layer` | kv_k:(pool,8,64), kv_v:(pool,8,64) × L | slot:(B,Lk) | Parallel K+V per layer | float16 | Separate K and V pool per layer; 2L GATHER_OP_SPECs; K and V gathered independently per layer | All K and V outputs correct; 2L SDSC bundles in compiled graph | atol=1e-2 | SENCORES=1 | TestGatherProductionInferencePatterns::test_multi_layer_parallel_k_v_per_layer |

---

## Section 14 — STL Paired Comparison Suite

**Test class:** `TestGatherSpyreTensorLayoutAnnotation`
**File:** `tests/inductor/indirect_access/test_gather_stl.py`

> Each test runs the same logical gather in two configurations — with and without explicit STL — and asserts the outputs are bit-exact. This validates that STL is transparent to numerical correctness and confirms the layout specification does not change results.
>
> **Indirect Access Validation Focus:** Validates the key invariant that `SpyreTensorLayout` is a layout hint only — it changes the SDSC `device_coordinates` encoding but must not change the numerical output. By running the same gather with and without explicit STL and asserting bit-exact agreement, these tests catch any regression where an STL annotation inadvertently changes which memory addresses are read. The SDSC `device_coordinates` field genuinely differs between the two forms, confirming STL has a real hardware-level effect on the hardware encoding while leaving numerical output unchanged.

| ID | Test Name | Input Shape | Index Shape | Comparison (A vs B) | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GSTLCMP-01 | `test_stl_vs_no_stl_2d_bare` | (32, 256) | (16,) | A: `x[idx]` with explicit STL  B: `x[:, idx, :]` no STL | float16 | STL transparent to numerical output; same gather result regardless of layout annotation | A output == B output (bit-exact) | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_vs_no_stl_2d_bare |
| GSTLCMP-02 | `test_stl_vs_no_stl_2d_exp` | (32, 256) | (16,) | A: `x[idx].exp()` with STL  B: without STL | float16 | Downstream op output identical with/without STL; fusion does not change numerics | exp(gathered) identical in both paths | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_vs_no_stl_2d_exp |
| GSTLCMP-03 | `test_stl_vs_no_stl_kv` | (512, 8, 64) | (4, 32) | A: KV gather with STL  B: KV gather without | float16 | Paged attention output equivalence; STL layout does not affect KV values | K and V outputs identical in both paths | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_vs_no_stl_kv |
| GSTLCMP-04 | `test_stl_vs_no_stl_embedding` | (512, 64) | (32,) | A: `weight[idx]` with STL  B: without | float16 | Embedding lookup output equivalence; embedding values identical | A embedding == B embedding | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_vs_no_stl_embedding |
| GSTLCMP-05 | `test_stl_vs_no_stl_bfloat16` | (32, 256) | (16,) | A: bfloat16 + STL  B: bfloat16, no STL | bfloat16 | Dtype-specific STL equivalence; bfloat16 precision identical both paths | A == B for bfloat16 gathered values | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_vs_no_stl_bfloat16 |
| GSTLCMP-06 | `test_stl_removal_regression` | (32, 256) | (16,) | A: STL removed from case where not required  B: no STL baseline | float16 | Removing unnecessary STL does not change output; no regression | A == B after removing spurious STL annotation | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_removal_regression |
| GSTLCMP-07 | `test_stl_api_equivalence` | (32, 256) | (16,) | A: `x[idx]` with STL  B: `torch.index_select(x,0,idx)` no STL | float16 | Different gather API forms produce identical output when logically equivalent | A == B (index_select and advanced indexing give same result) | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_api_equivalence |
| GSTLCMP-08 | `test_stl_vs_no_stl_moe` | (8, 512, 64) | (32,) | A: MoE expert gather with STL  B: without | float16 | MoE routing output equivalence across STL presence | Expert weight outputs identical in both paths | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_vs_no_stl_moe |
| GSTLCMP-09 | `test_stl_vs_no_stl_paged` | (512, 8, 64) | (4, 32) | A: full paged attention with STL  B: without | float16 | Full attention output equivalence; final attention result identical | Attention output identical in both paths | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_vs_no_stl_paged |

---

## Section 15 — Compiler Pipeline Validation

**Test class:** `TestGatherCompilerPassesMemoryAndDynamicShapes`
**File:** `tests/inductor/indirect_access/test_gather_compiler.py`

> Unit-level validation of the compiler passes and SDSC bundle encoding. These test the graph structure and metadata, not just numerical output.
>
> **Indirect Access Validation Focus:** Validates the compiler pipeline internals for indirect access: that `GATHER_OP_SPEC` / `DIRECT_OP_SPEC` / `NO_SPYRE_OP` are emitted for exactly the right operations, that SDSC bundle fields (`memOrg_`, `relatedIndirectAccessAlloc_`, `indirectAllocType_`, `computeOp`) are set to their specified values, that pass ordering (`propagate_layouts` before `detect_indirect_access`) is maintained, and that in-graph index producers (arange, topk, nonzero, where, argmax) propagate through the detection pass and still emit `GATHER_OP_SPEC`. Numerical correctness tests alone cannot catch op-spec misclassification or malformed SDSC fields — these structural tests close that gap.

| ID | Test Name | Operation Under Test | Input Shape | Index Shape | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|
| GCP-01 | `test_fx_no_spyre_op` | `F.embedding(idx, w)` | (4096, 64) | (32,) | `aten.embedding` → NO_SPYRE_OP; registered CPU fallback | NO_SPYRE_OP in op-spec; no GATHER_OP_SPEC for aten.embedding | N/A | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_fx_no_spyre_op |
| GCP-02 | `test_sdsc_index_in_hbm` | `x[idx]` | (32, 256) | (16,) | Index tensor `memOrg_` field = HBM (never placed in LX scratchpad) | SDSC bundle: idx buffer has `memOrg_=HBM`; no LX allocation | N/A | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_sdsc_index_in_hbm |
| GCP-03 | `test_sdsc_cross_links` | `x[idx]` | (32, 256) | (16,) | `relatedIndirectAccessAlloc_` cross-links: value↔index bidirectional | SDSC of x points to idx; SDSC of idx points to x | N/A | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_sdsc_cross_links |
| GCP-04 | `test_sdsc_two_independent_gathers` | `x[idx1]`; `y[idx2]` | (32,256) each | (16,) each | Two independent gathers → 2 SDSC bundles; cross-links do not mix | SDSC for x only links to idx1; SDSC for y only links to idx2 | N/A | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_sdsc_two_independent_gathers |
| GCP-05 | `test_sdsc_compute_op_routing` | `x[idx].exp()` | (32, 256) | (16,) | SDSC `computeOp` field routes to exp kernel; not a separate node | SDSC bundle: `computeOp=exp`; exp fused into gather bundle | N/A | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_sdsc_compute_op_routing |
| GCP-06 | `test_pass_order_layouts_first` | `x[idx]` | (32, 256) | (16,) | Compiler pass order: `propagate_layouts` runs before `detect_indirect_access`; pass order logged | Pass order log shows `propagate_layouts` before `detect_indirect_access` | N/A | TORCH_LOGS="+inductor" | TestGatherCompilerPassesMemoryAndDynamicShapes::test_pass_order_layouts_first |
| GCP-07 | `test_reshape_barrier` | `x.reshape(M,N)[idx]` | (32,256) reshaped | (16,) | Reshape breaks named-dim propagation; GATHER_OP_SPEC NOT generated after reshape | No GATHER_OP_SPEC in FX graph; falls back to CPU or DIRECT_OP_SPEC | N/A | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_reshape_barrier |
| GCP-08 | `test_bundle_symbolic_args` | `x[idx]` | (32, 256) | (16,) | `BUNDLE_SYMBOLIC_ARGS=1`; symbolic args serialized in bundle JSON | Bundle JSON contains symbolic arg entries | N/A | BUNDLE_SYMBOLIC_ARGS=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_bundle_symbolic_args |
| GCP-09 | `test_sdsc_indirect_alloc_type` | `x[idx]` | (32, 256) | (16,) | `indirectAllocType_` field: `index_tensor` on idx, `value_tensor` on x | SDSC for idx has `indirectAllocType_=index_tensor`; x has `value_tensor` | N/A | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_sdsc_indirect_alloc_type |
| GCP-10 | `test_two_different_indices_same_source` | x:(64, 128) | idx1:(16,), idx2:(8,) | — | float16 | Same source tensor, two distinct index tensors in one compiled graph; two independent GATHER_OP_SPECs emitted | Both x[idx1] and x[idx2] match CPU; two GATHER_OP_SPECs in compiled graph | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_two_different_indices_same_source |
| GCP-11 | `test_two_different_sources_different_indices` | x:(64,64), y:(32,64) | idx_x:(16,), idx_y:(12,) | — | float16 | Two distinct sources and two distinct indices in one graph; four independent GATHER_OP_SPECs | x[idx_x] and y[idx_y] both match CPU; four GATHER_OP_SPECs | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_two_different_sources_different_indices |
| GCP-12 | `test_arange_index_inside_graph` | (32, 256) | `torch.arange(16)` computed inside fn | In-graph index from `arange`; index computed inside compiled graph; GATHER_OP_SPEC handles in-graph index tensor | GATHER_OP_SPEC emitted; output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_arange_index_inside_graph |
| GCP-13 | `test_topk_index_inside_graph` | x:(32, 256), scores:(32,) | topk indices from `scores.topk(16)` | In-graph index from `topk`; top-16 row selection; index produced by topk inside compiled fn | GATHER_OP_SPEC emitted; top-16 rows match CPU | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_topk_index_inside_graph |
| GCP-14 | `test_nonzero_index_inside_graph` | (32, 256) | `mask.nonzero().squeeze()` inside fn | In-graph index from `nonzero`; data-dependent but statically known at trace; GATHER_OP_SPEC | Output rows at nonzero positions match CPU | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_nonzero_index_inside_graph |
| GCP-15 | `test_where_index_inside_graph` | (32, 256) | `torch.where(cond)` derived index | In-graph index from `where`; index derived from conditional inside compiled fn | GATHER_OP_SPEC emitted; selected rows match CPU | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_where_index_inside_graph |
| GCP-16 | `test_argmax_index_inside_graph` | x:(32, 256), logits:(32,) | `logits.argmax(dim=0, keepdim=True)` | Single-row gather: index = argmax of logit vector; P=1 special case from argmax; in-graph index | Best-row output matches CPU gather at argmax position | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_argmax_index_inside_graph |
| GCP-17 | `test_explicit_gather_dim0_lx` | `torch.gather(src, 0, index)` | src:(64, 128) | index:(32, 128) int64 | float16 | Explicit `aten::gather.out` at dim=0 with LX_PLANNING=1; first compiler-context test for the direct torch.gather API form (distinct from `x[idx]` aten::index path) | Output (32,128) matches CPU `torch.gather(src,0,idx)` | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_explicit_gather_dim0_lx |
| GCP-18 | `test_explicit_gather_dim1_lx` | `torch.gather(src, 1, index)` | src:(32, 64) | index:(32, 16) int64 | float16 | `aten::gather.out` at dim=1 with LX_PLANNING=1; non-leading gather through LX planner | Output (32,16) matches CPU `torch.gather(src,1,idx)` | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_explicit_gather_dim1_lx |
| GCP-19 | `test_explicit_gather_dim2_lx` | `torch.gather(src, 2, index)` | src:(8, 16, 64) | index:(8, 16, 32) int64 | float16 | `aten::gather.out` at dim=2 with LX_PLANNING=1; innermost-dim gather through LX planner | Output (8,16,32) matches CPU `torch.gather(src,2,idx)` | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_explicit_gather_dim2_lx |
| GCP-20 | `test_explicit_gather_dim0_bfloat16_lx` | `torch.gather(src, 0, index)` bfloat16 | src:(64, 128) | index:(32, 128) int64 | bfloat16 | `aten::gather.out` dim=0 bfloat16 with LX_PLANNING=1; wordLength=2 path through LX planner | Output (32,128) bfloat16 matches CPU `torch.gather` | atol=2e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_explicit_gather_dim0_bfloat16_lx |
| GCP-21 | `test_dim1_fancy_indexing_sencores_4` | `x[:, idx, :]` | x:(8, 32, 64) | idx:(16,) int64 | float16 | dim=1 fancy-index gather with SENCORES=4; non-leading-dim indirect access across 4 cores; unique angle vs. SENCORES=1 functional tests | Output (8,16,64) matches CPU `x[:,idx,:]` | atol=1e-2 | SENCORES=4 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dim1_fancy_indexing_sencores_4 |
| GCP-22 | `test_dim2_fancy_indexing_lx` | `x[:, :, idx]` | x:(8, 16, 64) | idx:(32,) int64 | float16 | dim=2 fancy-index gather with LX_PLANNING=1; innermost-dim indirect access through LX planner | Output (8,16,32) matches CPU `x[:,:,idx]` | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dim2_fancy_indexing_lx |
| GCP-23 | `test_dim1_explicit_gather_lx` | `torch.gather(src, 1, index)` | src:(4, 32, 64) | index:(4, 8, 64) int64 | float16 | `aten::gather.out` at dim=1 with LX_PLANNING=1; dim=1 KV-head reorder pattern in compiler context | Output (4,8,64) matches CPU `torch.gather(src,1,idx)` | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dim1_explicit_gather_lx |

---

## Section 16 — Memory, Layout, and Scale

**Test class:** `TestGatherCompilerPassesMemoryAndDynamicShapes`
**File:** `tests/inductor/indirect_access/test_gather_compiler.py`

> **Indirect Access Validation Focus:** Validates Spyre-specific memory layout constraints for indirect access: stick alignment (N=64 elements = 1 stick, non-aligned N=100, multi-stick N=256), non-contiguous value/index tensors, large tensor address bounds (128 MB, 256 MB approaching Spyre HBM limits), the HBM-only constraint that forbids index tensors from being placed in LX scratchpad, and the SDSC `stride_map` field value. These tests catch address overflow, padding corruption, and LX misplacement failures that are invisible in small-shape numerical tests.

| ID | Test Name | Input Shape | Index Shape | Memory Aspect | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GMEM-01 | `test_stick_aligned_exact` | (32, 64) | (16,) | N=64 = exactly 1 stick (64 fp16 elems) | float16 | Stick-aligned inner dim; no zero-padding needed; SDSC stride clean | Output (16,64) matches CPU; no padding artifacts | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_stick_aligned_exact |
| GMEM-02 | `test_non_contiguous_value` | (32, 256) transposed | (16,) | Non-contiguous value: `x.T` | float16 | Non-contiguous source tensor; Spyre sees column-major layout; gather still correct | Output matches CPU reference for transposed gather | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_non_contiguous_value |
| GMEM-03 | `test_non_contiguous_index` | (32, 256) | strided `idx[::2]` | Non-contiguous index (stride=2 view) | float16 | Strided index view; non-contiguous index tensor; address computation correct | Output matches CPU for strided index selection | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_non_contiguous_index |
| GMEM-04 | `test_large_tensor_256mb` | (32768, 4096) ≈256 MB | (64,) | Large tensor ~256 MB | float16 | 256 MB value tensor; near Spyre HBM limit; compile and run without OOM | Output (64,4096) matches CPU; no allocation failure | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_large_tensor_256mb |
| GMEM-05 | `test_lx_index_not_placed` | (32, 256) | (16,) | LX_PLANNING=1; index must be in HBM | float16 | Index tensor MUST NOT be placed in LX scratchpad (hardware constraint) | Index buffer allocation in HBM; no LX assignment for idx | N/A | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_index_not_placed |
| GMEM-06 | `test_lx_downstream_in_lx` | (32, 256) | (16,) | LX_PLANNING=1; downstream eligible | float16 | Downstream intermediate buffer CAN go in LX (not index); verify correct LX assignment | Downstream buffer assigned to LX; idx stays in HBM | N/A | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_downstream_in_lx |
| GMEM-07 | `test_sencores32_memory` | (64, 256) | (32,) | SENCORES=32 work-division | float16 | 32-core memory partitioning; each core gets correct slice of value tensor; no overlap | Output matches CPU; per-core slice boundaries correct | atol=1e-2 | SENCORES=32 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_sencores32_memory |

---

## Section 17 — Dynamic and Symbolic Shapes

**Test class:** `TestGatherCompilerPassesMemoryAndDynamicShapes`
**File:** `tests/inductor/indirect_access/test_gather_compiler.py`

> **Indirect Access Validation Focus:** Validates that the indirect access path works correctly with `torch.compile(dynamic=True)`: symbolic dims (P, B, Lk, pool size) must remain as symbolic expressions through compilation, be correctly encoded in the SDSC bundle's symbolic arg fields, and resolve to correct output at multiple distinct runtime shapes without triggering unnecessary recompilation. A regression here would mean Spyre gather requires static shapes for every inference call — eliminating the serving-time flexibility needed for variable batch sizes and context lengths.

| ID | Test Name | Input Shape | Index Shape | Symbolic Dim(s) | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GDYN-01 | `test_dynamic_index_size` | (32, 256) | (P,) dynamic | P | float16 | Compile once with symbolic P; run at P=16, P=32, P=64; no recompile at each size | Outputs match CPU at all three P values; compile count=1 | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_index_size |
| GDYN-02 | `test_dynamic_batch` | (B,S,D)=(B,128,256) | (B,Lk) dynamic | B | float16 | Variable batch B at inference; B=1,4,8 at runtime; one compiled graph | Outputs match CPU at each B; shape inference correct | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_batch |
| GDYN-03 | `test_dynamic_seq_len` | (pool,H,D)=(512,8,64) | (B,Lk) dynamic S | Lk | float16 | Variable key sequence length Lk; Lk=16,32,64 at runtime | Outputs match CPU at each Lk | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_seq_len |
| GDYN-04 | `test_dynamic_kv_pool_size` | (pool,8,64) dynamic pool | (4,32) | pool | float16 | Variable KV pool size; pool=256,512,1024; SDSC address range correct | Outputs match CPU at each pool size; no OOB | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_kv_pool_size |
| GDYN-05 | `test_dynamic_all_dims` | (M,N) all dynamic | (P,) dynamic | M, N, P | float16 | Fully symbolic gather: M, N, P all symbolic; most general case | Outputs match CPU at (M=32,N=256,P=16) and (M=64,N=128,P=32) | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_all_dims |
| GDYN-06 | `test_symbol_propagation` | (32, 256) | (P,) dynamic | P | float16 | Symbol P flows from index → output shape; output tensor symbolic shape (P,256) correct | Output shape is symbolic (P,256); at runtime resolves to correct size | N/A | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_symbol_propagation |
| GDYN-07 | `test_dynamic_embedding` | (vocab=32000, dim=512) | (seq,) dynamic | seq_len | float16 | Embedding with symbolic token sequence length; seq=32,64,128 at runtime | Embedding outputs match CPU at each seq | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_embedding |
| GDYN-08 | `test_dynamic_moe` | (8,512,64) | (tokens,) dynamic | token_count | float16 | MoE routing with symbolic active token count; tokens=8,16,32 | Expert gather outputs match CPU at each token count | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_moe |
| GDYN-09 | `test_dynamic_paged_kv` | (512,8,64) | (B=4,Lk) dynamic | Lk | float16 | Paged KV with symbolic key seq length; Lk=16,32,64 at runtime | KV outputs match CPU at each Lk | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_paged_kv |
| GDYN-10 | `test_dynamic_then_static_cache` | (32, 256) | (P,) | P dynamic → static | float16 | Compile dynamic first, then static at same shape; second compile is cache hit | Static compile returns same result; compile count stays at 1 after static call | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_dynamic_then_static_cache |

---

## Section 18 — Cross-Feature Dependencies

**Test class:** `TestGatherCompilerPassesMemoryAndDynamicShapes`
**File:** `tests/inductor/indirect_access/test_gather_compiler.py`

> **Indirect Access Validation Focus:** Validates that indirect access interacts correctly with every major Spyre compiler flag and optimization in combination — LX scratchpad planning, multi-core work division, dynamic shapes + STL + named-dims combined, chunked large tensor processing, global stick optimizer, symbolic arg bundling, and loop unrolling. The key invariant under all combinations is that the index tensor remains in HBM (never in LX) and the SDSC encoding is not disturbed by any optimization pass. A single broken interaction here could silently corrupt KV cache reads during production inference.

| ID | Test Name | Input Shape | Index Shape | Features Combined | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GCFG-01 | `test_gather_lx_planning` | (32, 256) | (16,) | gather + LX_PLANNING=1 | float16 | LX scratchpad planning active; downstream eligible for LX; index stays in HBM | Output matches CPU; idx in HBM; downstream in LX | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_lx_planning |
| GCFG-02 | `test_gather_work_division` | (64, 256) | (32,) | gather + SENCORES=4 | float16 | 4-core work-division with indirect access; partial gather merged correctly | Output matches CPU; SENCORES=4 partitioning | atol=1e-2 | SENCORES=4 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_work_division |
| GCFG-03 | `test_gather_symbolic_stl_named` | (32, 256) | (P,) dynamic | gather + dynamic + STL + named_dims | float16 | Combined: symbolic P, explicit STL, named dims; all three features interact | Compile succeeds; output matches CPU at P=16,32 | atol=1e-2 | SENCORES=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_symbolic_stl_named |
| GCFG-04 | `test_gather_chunk_large_tensors` | (16384, 4096) ≈128 MB | (64,) | gather + CHUNK_LARGE_TENSORS=1 | float16 | Large tensor processed in chunks; CHUNK_LARGE_TENSORS flag splits into sub-batches | Output matches CPU reference at full size | atol=1e-2 | CHUNK_LARGE_TENSORS=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_chunk_large_tensors |
| GCFG-05 | `test_gather_global_stick_opt` | (32, 256) | (16,) | gather + GLOBAL_STICK_OPTIMIZER=1 | float16 | Global stick optimizer active; indirect access not broken by stick re-ordering | Output matches CPU; no stick ordering side-effects | atol=1e-2 | GLOBAL_STICK_OPTIMIZER=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_global_stick_opt |
| GCFG-06 | `test_gather_bundle_symbolic` | (32, 256) | (16,) | gather + BUNDLE_SYMBOLIC_ARGS=1 | float16 | Symbolic arg bundling with gather; bundle JSON includes idx symbolic dimensions | Output matches CPU; bundle JSON contains symbolic args | atol=1e-2 | BUNDLE_SYMBOLIC_ARGS=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_bundle_symbolic |
| GCFG-07 | `test_gather_unroll_loops` | (32, 256) | (16,) | gather + UNROLL_LOOPS=1 | float16 | Loop unrolling does not corrupt indirect access address computation | Output matches CPU with loop unrolling | atol=1e-2 | UNROLL_LOOPS=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_unroll_loops |
| GCFG-08 | `test_gather_dxp_lx_frac_half` | (32, 256) | (16,) | gather + LX + DXP_LX_FRAC_AVAIL=0.5 | float16 | LX fractional allocation (50% of LX budget); gather and downstream compete for LX | Output matches CPU; no OOM; correct LX allocation fraction | atol=1e-2 | LX_PLANNING=1, DXP_LX_FRAC_AVAIL=0.5 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_dxp_lx_frac_half |
| GCFG-09 | `test_gather_co_opt_lx` | (32, 256) | (16,) | gather + CO_OPTIMIZING_LX_PLANNING=1 | float16 | Co-optimized LX with gather; STL 4-arg form; co-optimized planning does not misplace idx | Output matches CPU; idx not in LX | atol=1e-2 | CO_OPTIMIZING_LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_co_opt_lx |
| GCFG-10 | `test_gather_lx_sencores_combo` | (64, 256) | (32,) | gather + LX_PLANNING=1 + SENCORES=4 | float16 | Combined LX + 4-core; idx HBM constraint holds across multi-core partition | Output matches CPU; idx in HBM; LX used for eligible buffers | atol=1e-2 | LX_PLANNING=1, SENCORES=4 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_gather_lx_sencores_combo |

---

## Section 19 — Chained and Composed Gather

**Test class:** `TestGatherComposedChainsAndEndToEndPipelines`
**File:** `tests/inductor/indirect_access/test_gather_chained.py`

> Tests gather composed with other indirect access operations or with gather itself. The existing suite covers two-gather binary (xfail) and sequential gather+op chains. These cover new compositions.
>
> **Indirect Access Validation Focus:** Validates gather composed with other indirect access or data-movement operations in a single compiled graph — sequential gather + index_select, gather + scatter_add (read-modify-write pattern), two independent gathers from the same source tensor, and gather feeding downstream matmul or MoE aggregation. The key risk is SDSC cross-link confusion: when multiple GATHER_OP_SPECs share a source or output buffer, the `relatedIndirectAccessAlloc_` bidirectional link must not accidentally point to the wrong index tensor, causing the hardware to read using the wrong index.

| ID | Test Name | Source Shapes | Index Shapes | Composed Pattern | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GCH-01 | `test_gather_then_index_select` | x:(32,256), y:auto | idx1:(16,), idx2:(8,) | `y=x[idx1]; z=index_select(y,0,idx2)` | float16 | Sequential gather + index_select; two GATHER_OP_SPECs in one graph | z (8,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_then_index_select |
| GCH-02 | `test_gather_then_scatter_add` | x:(32,256), out:(32,256) | idx:(16,) | `read=x[idx]; out.scatter_add_(0, idx.unsqueeze(1).expand(-1,256), read)` | float16 | Gather then scatter_add accumulate; read-modify-write pattern | out matches CPU scatter_add result | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_then_scatter_add |
| GCH-03 | `test_gather_same_source_two_paths` | x:(32,256) | idx1:(8,), idx2:(8,) | `a=x[idx1]; b=x[idx2]` — two independent gathers from x | float16 | Two GATHER_OP_SPECs on same source; SDSC cross-links do not interfere | a and b both match CPU independently | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_same_source_two_paths |
| GCH-04 | `test_gather_concat_results` | x:(32,128), y:(32,128) | idx1:(8,), idx2:(8,) | `torch.cat([x[idx1], y[idx2]], dim=0)` | float16 | Two gather results concatenated; output (16,128); two SDSCs | Concatenated output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_concat_results |
| GCH-05 | `test_gather_add_direct_tensor` | x:(32,256), y:(16,256) | idx:(16,) | `x[idx] + y` where y is not gathered | float16 | Gather result element-wise added to non-gathered tensor; binary op after GATHER_OP_SPEC | Output = gathered rows + y; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_add_direct_tensor |
| GCH-06 | `test_gather_embedding_then_rope` | emb_table:(32000,128), cos/sin:(128,) | token_ids:(seq=64,) | `embed=table[tok]; out=embed*cos + rotate_half(embed)*sin` | float16 | Embedding lookup (GATHER_OP_SPEC) then RoPE element-wise multiply; two-stage compose | output (64,128) matches CPU RoPE application | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_embedding_then_rope |
| GCH-07 | `test_gather_kv_then_attention_score` | kv_cache:(512,8,64), q:(B=4,1,8,64) | slot:(4,32) | `k=kv_cache[slot]; attn=q@k.T/scale` | float16 | KV gather (GATHER_OP_SPEC) then matmul attention score; gather feeds matmul | Attention scores (4,1,8,32) match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_kv_then_attention_score |
| GCH-08 | `test_gather_moe_then_aggregate` | expert_w:(8,512,64) | expert_ids:(32,) | `out=expert_w[expert_ids]; agg+=out * gating_weight` | float16 | MoE gather (GATHER_OP_SPEC) then weighted aggregation; gating weight multiply + sum | Aggregated output (512,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_moe_then_aggregate |

---

## Section 20 — Edge Cases and Negative Tests

**Test class:** `TestGatherBoundaryNumericsAndAPIForms`
**File:** `tests/inductor/indirect_access/test_gather_edge_cases.py`

> **Indirect Access Validation Focus:** Validates the boundary conditions and known failure modes of the indirect access path: P=1 triggering `BackendCompilerFailed`, reshape barriers preventing `GATHER_OP_SPEC` from being emitted, float64 correctly excluded from indirect access, empty index (P=0) producing zero-element output without crash, 2D-shaped index producing documented divergence from CPU, dual-gather binary op triggering the known #2661 multi-SDSC bug, out-of-bounds behavior documented without crash, and negative dim normalization before `GATHER_OP_SPEC` dispatch. These tests define the exact error contract for each unsupported or partially supported indirect access scenario.

| ID | Test Name | Input Shape | Index Shape / Condition | What It Validates | Expected Outcome | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|
| GEC-01 | `test_single_row_index` | (32, 256) | idx:(1,) — P=1 | P=1 triggers BackendCompilerFailed; validates known error path | `BackendCompilerFailed` raised at compile; test catches exception | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_single_row_index |
| GEC-02 | `test_gather_after_reshape_barrier` | (32, 256) reshaped | idx:(16,) after reshape | Reshape before gather breaks named-dim chain; GATHER_OP_SPEC NOT emitted | No GATHER_OP_SPEC in FX graph; output computed via CPU fallback | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_gather_after_reshape_barrier |
| GEC-03 | `test_float64_value_not_gathered` | (32, 64) float64 | idx:(16,) int32 | float64 value must NOT produce GATHER_OP_SPEC; falls back to CPU computation | GATHER_OP_SPEC absent; output computed on CPU; values correct | atol=1e-10 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_float64_value_not_gathered |
| GEC-04 | `test_gate_off_no_gather` | (32, 256) | idx:(16,) | `x[idx]` numeric correctness; GATHER_OP_SPEC always emitted; output matches CPU reference | Output matches CPU reference; GATHER_OP_SPEC present in FX graph | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_gate_off_no_gather |
| GEC-05 | `test_empty_index` | (32, 256) | idx shape (0,) — empty | P=0 empty index produces zero-element output tensor; no crash | Output shape (0,256); no crash or NaN | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_empty_index |
| GEC-06 | `test_2d_index_divergence` | (16, 64) | idx shape (8,4) — 2D | 2D-shaped index compiles but device output diverges from CPU; document known behavior | Compile succeeds; output diverges from CPU (documented divergence, not a pass/fail check) | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_2d_index_divergence |
| GEC-07 | `test_dual_gather_binary_xfail` | (32, 256) | idx1:(16,), idx2:(16,) | `x[idx1] + x[idx2]` — two SDSC bundles in binary op; known issue #2661 | `BackendCompilerFailed` or multi-SDSC job plan error raised | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_dual_gather_binary_xfail |
| GEC-08 | `test_oob_index_behavior` | (32, 256) | idx contains M=32 (OOB value) | No runtime bounds check on Spyre; OOB behavior is undefined; document actual output | No crash; output documented (may contain garbage at OOB position) | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_oob_index_behavior |
| GEC-09 | `test_negative_dim_torch_gather` | (32, 256) | `torch.gather(x, -1, idx)` same-rank | Negative dim=-1 normalized to dim=1 before GATHER_OP_SPEC; gather on last axis | Output matches CPU `torch.gather(x,-1,idx)` | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_negative_dim_torch_gather |
| GEC-10 | `test_sin_cpu_fallback_split` | (32, 64) | idx:(16,) | `x[idx].sin()`: gather on Spyre, `sin` has no Spyre kernel → CPU fallback; two-device dispatch | Gather on Spyre; sin on CPU; combined output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_sin_cpu_fallback_split |
| GEC-11 | `test_index_fill_crash` | (32, 256) | idx:(16,), fill_value=0.0 | `x.index_fill_(0, idx, 0.0)` known crash: rank-0 constant codegen failure | `RuntimeError` or `BackendCompilerFailed` raised; crash documented | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_fill_crash |
| GEC-12 | `test_masked_scatter_crash` | (32, 256) | bool mask (32,256), src:(count, 256) | `out.masked_scatter(mask, src)` known crash on Spyre indirect path | `RuntimeError` raised; crash documented | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_scatter_crash |
| GEC-13 | `test_reshape_between_two_gathers` | (32, 256) | idx1:(16,), idx2:(8,) between reshape | `a=x[idx1].reshape(4,4,256); b=a[idx2]` — reshape between gathers; second gather may lose GATHER_OP_SPEC | First gather: GATHER_OP_SPEC; second: GATHER_OP_SPEC or CPU depending on reshape barrier | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_reshape_between_two_gathers |
| GEC-14 | `test_oob_negative_index` | (32, 256) | idx contains -1 | Negative index value with no Python bounds wrap on compiled path; device behavior documented | No crash; output at -1 position documented (likely wraps to last row or garbage) | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_oob_negative_index |
| GEC-15 | `test_gather_zero_stride_idx` | (32, 256) | idx: stride=0 expanded tensor | Zero-stride index (all elements the same address); no crash; all output rows = same input row | No crash; all output rows are identical; match CPU at zero-stride | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_gather_zero_stride_idx |
| GEC-16 | `test_empty_index_zero_output` | (32, 64) | idx:(0,) — P=0 | — | float16 | P=0 empty index; output shape must be (0, 64) with zero elements accessed; no out-of-bounds access | Output shape (0,64); no crash; matches CPU empty tensor | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_empty_index_zero_output |
| GEC-17 | `test_minimal_gather_m1_p1_d1` | (1, 1) | idx:(1,) | — | float16 | Absolute minimum: M=1, P=1, D=1; scalar gather from single-element source; degenerate shape | Output (1,1); value equals the single source element | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_minimal_gather_m1_p1_d1 |
| GEC-18 | `test_negative_index_int64` | (16, 64) | idx: [-1,-2,0,5,-1] int64 | — | float16 | Negative int64 index values; PyTorch defines `x[tensor([-1])]` as the last row (Python wrap semantics); validates Spyre matches CPU | Output rows at positions [14,13,0,5,14] match CPU; no undefined behavior | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_negative_index_int64 |
| GEC-19 | `test_negative_index_int64_downcast` | (32, 128) | idx: [-1,-4,-8,-16,0,3] int64 | — | float16 | Negative int64 index through int64→int32 downcast path; int32(-1) must be interpreted as M-1 (not as 4294967295 uint32 overflow); validates correct wrap in downcast path | Output rows at positions [31,28,24,16,0,3] match CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_negative_index_int64_downcast |

---

## Section 33 — SDPA with Paged KV Cache (FlashAttention Patterns)

**Test class:** `TestGatherPagedAttentionAndSDPA`
**File:** `tests/inductor/indirect_access/test_gather_attention.py`

> Full SDPA (scaled dot-product attention) patterns that combine paged KV-cache gather with on-device attention computation. Each test runs gather (GATHER_OP_SPEC) then matmul/softmax/weighted-V-sum on Spyre. This entire test class has no prior FVT coverage.
>
> **Indirect Access Validation Focus:** Validates the complete paged-attention pipeline as a single indirect access + compute unit: K and V tensors are each gathered from the paged KV pool via `GATHER_OP_SPEC`, then fed directly into Q@K^T/scale + softmax + @V on Spyre in a fused pass — with no host round-trip between gather and attention compute. A bug in the K or V SDSC `device_coordinates` here produces subtly wrong attention scores (plausible magnitude, wrong context) rather than NaN, making these tests critical for detecting silent accuracy regressions. Coverage includes GQA/MQA head ratios, causal and padding masks, online softmax accumulation (Flash-Attention style), chunked-prefill→decode phase transition, FMS 4D layout, and speculative-decode draft verification.

| ID | Test Name | KV Cache Shape | Slot Index Shape | Config | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GSDPA-01 | `test_sdpa_paged_decode_basic` | (pool, 8, 64) | (B, 1) decode | basic decode | float16 | Basic paged decode: gather K and V then compute Q@K^T/scale + softmax + @V; single decode step | Attention output matches CPU scaled-dot-product at decode | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_paged_decode_basic |
| GSDPA-02 | `test_sdpa_paged_prefill_full_context` | (pool, 8, 64) | (1, Lk) prefill | full prefill | float16 | Prefill with full context: gather K/V across Lk slots; online attention over full sequence | Attention output (1,Lq,8,Dv) matches CPU prefill | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_paged_prefill_full_context |
| GSDPA-03 | `test_sdpa_gqa_32q_8kv` | (pool, 8, 64) | (B, Lk) | GQA q=32, kv=8 | float16 | Group-query attention: 32 Q heads, 8 KV heads; KV gathered at 8 heads then broadcast to 32 Q heads | Attention output with head broadcast matches CPU GQA result | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_gqa_32q_8kv |
| GSDPA-04 | `test_sdpa_mqa_single_kv_head` | (pool, 1, 64) | (B, Lk) | MQA kv_heads=1 | float16 | Multi-query attention: single KV head shared by all Q heads; gathered at 1 head then broadcast | Attention output matches CPU MQA | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_mqa_single_kv_head |
| GSDPA-05 | `test_sdpa_bfloat16_paged_decode` | (pool, 8, 64) | (B, 1) | bfloat16 decode | bfloat16 | bfloat16 KV cache paged decode; SDSC wordLength=2; attention in bfloat16 | Attention output matches CPU bfloat16 decode | atol=2e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_bfloat16_paged_decode |
| GSDPA-06 | `test_sdpa_causal_mask_prefill` | (pool, 8, 64) | (1, Lk) | causal mask | float16 | Causal attention mask applied during prefill; gather K/V then mask future positions to -inf | Masked attention output matches CPU causal attention | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_causal_mask_prefill |
| GSDPA-07 | `test_sdpa_attention_bias_neg_inf_padding` | (pool, 8, 64) | (B, Lk) | bias padding | float16 | Attention bias with -inf for padded positions; bias added before softmax; padding ignored | Attention output at valid positions matches CPU; padded positions zero after softmax | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_attention_bias_neg_inf_padding |
| GSDPA-08 | `test_sdpa_batch_decode_b4` | (pool, 8, 64) | (4, 1) B=4 decode | batch decode | float16 | Batch decode B=4: 4 independent decode steps in one compiled graph; 4 slot indices | Attention output (4,1,8,Dv) matches CPU per-request decode | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_batch_decode_b4 |
| GSDPA-09 | `test_sdpa_sencores4_paged_decode` | (pool, 8, 64) | (B, Lk) | SENCORES=4 | float16 | 4-core paged decode attention; gather and matmul both executed with 4-core work division | Output matches CPU at SENCORES=4 | atol=1e-2 | SENCORES=4 | TestGatherPagedAttentionAndSDPA::test_sdpa_sencores4_paged_decode |
| GSDPA-10 | `test_sdpa_sencores32_paged_prefill` | (pool, 8, 64) | (1, Lk) | SENCORES=32 | float16 | Full-chip 32-core paged prefill attention; maximum parallelism | Output matches CPU at SENCORES=32 | atol=1e-2 | SENCORES=32 | TestGatherPagedAttentionAndSDPA::test_sdpa_sencores32_paged_prefill |
| GSDPA-11 | `test_sdpa_lx_planning_kv_gather` | (pool, 8, 64) | (B, Lk) | LX_PLANNING=1 | float16 | LX planning with KV gather inside SDPA; index stays in HBM; attention intermediates in LX | Attention output correct; idx in HBM; intermediates LX-eligible | atol=1e-2 | LX_PLANNING=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_lx_planning_kv_gather |
| GSDPA-12 | `test_sdpa_2d_slot_index_batched_decode` | (pool, 8, 64) | (B, Lk) 2D slot | 2D batched slot | float16 | 2D slot index (B, Lk=1) decode: batch dimension in slot index; per-request slot selection | Output (B,1,8,64) matches CPU; 2D index handled correctly | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_2d_slot_index_batched_decode |
| GSDPA-13 | `test_sdpa_rope_gather_then_attention` | cos/sin:(4096,64), kv:(pool,8,64) | pos:(seq,), slot:(B,Lk) | RoPE + paged KV | float16 | Two gathers: RoPE cos/sin cache + KV slots; apply RoPE then paged attention; two GATHER_OP_SPECs | Full RoPE+attention output matches CPU; two SDSC bundles | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_rope_gather_then_attention |
| GSDPA-14 | `test_sdpa_chunked_prefill_then_single_decode` | (pool, 8, 64) | (1, Lk=64) prefill; (B=4, 1) decode | prefill→decode | float16 | Two-phase: chunked prefill (gather 64 slots) then decode step (1 slot per request); phase transition | Both phases correct; KV cache consistent across prefill→decode | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_chunked_prefill_then_single_decode |
| GSDPA-15 | `test_sdpa_large_kv_pool_decode` | (32768, 8, 64) large pool | (12, 1) decode | large pool | float16 | Large pool (32768 pages) decode; address computation valid at high pool offsets; no overflow | Output (12,1,8,64) matches CPU; no address overflow at high pool idx | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_large_kv_pool_decode |
| GSDPA-16 | `test_sdpa_separate_k_v_caches_different_shapes` | k_pool:(pool,8,64), v_pool:(pool,8,32) | slot:(B,Lk) | k/v different D | float16 | K and V pools with different head dims (D_k=64, D_v=32); two independent GATHER_OP_SPECs with different strides | K and V outputs match CPU; no stride confusion between pools | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_separate_k_v_caches_different_shapes |
| GSDPA-17 | `test_sdpa_output_then_linear_projection` | (pool, 8, 64) | (B, Lk) | attn + linear | float16 | Full attention output (B,Lq,H,Dv) → reshape + linear projection (output projection); attn feeds linear | Final output (B,Lq,D_model) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_output_then_linear_projection |
| GSDPA-18 | `test_sdpa_beam_reorder_then_attend` | kv_cache:(B=4,Lk,8,64) | beam_idx:(4,) | beam reorder | float16 | Beam search: `kv_cache[beam_idx]` reorders KV to best beams then runs attention; gather at batch dim | Attention output with beam reorder matches CPU | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_beam_reorder_then_attend |
| GSDPA-19 | `test_sdpa_speculative_draft_verify` | kv_cache:(pool,8,64) | slot:(B, Ld=5) draft | speculative verify | float16 | Speculative decode verification: gather Ld=5 draft token KV slots; compute attention for all draft positions simultaneously | Draft verification output (B,5,8,Dv) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_speculative_draft_verify |
| GSDPA-20 | `test_sdpa_4d_kv_layout_paged` | (pages, kv_h, page_sz, D) 4D FMS | (B, Lk) | FMS 4D SDPA | float16 | FMS 4D KV layout gathered via STL dim_order then fed into SDPA; layout reorder before matmul | Attention output with FMS layout matches CPU attention | atol=1e-2 | SENCORES=1 | TestGatherPagedAttentionAndSDPA::test_sdpa_4d_kv_layout_paged |

---

## Section 34 — Non-Leading-Dim Gather (dim=1, dim=2 Production Patterns)

**Test class:** `TestGatherNonLeadingDimEagerCompile`
**File:** `tests/inductor/indirect_access/test_gather_functional.py`

> Gathers at non-outermost dimensions (dim=1, dim=2) for real model patterns: head selection in GQA/MQA, sequence position selection, and feature dimension indexing. `test_torch_gather_dim1` and `test_take_along_dim_dim1` are also partially covered in Section 24 (GAPI-23, GAPI-24). These 7 tests cover the remaining non-leading-dim scenarios not yet in any FVT section.
>
> **Indirect Access Validation Focus:** Validates gather at non-outermost tensor dimensions (dim=1 and dim=2), which require different SDSC `device_coordinates` encoding than the standard dim=0 gather — the compiler must account for the outer dimensions as a batch axis rather than as the gather axis. These patterns arise in GQA head selection (`x[:, head_idx, :]` at dim=1), sequence-position gather at dim=1 for sliding-window attention, feature-dimension selection at dim=2 for sparse feature retrieval, and `take_along_dim` at dim=2 for Flash-Attention top-k scoring. Incorrect handling of the outer batch dims in the SDSC for non-leading-dim gathers is a silent error — the output shape is correct but values are read from the wrong position within each batch element.

| ID | Test Name | Source Shape | Index Shape | dim | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GNLD-01 | `test_gather_dim1_gqa_head_select` | (B=4, 32, 64) — (B, heads, D) | (B=4, 8, 64) GQA kv_head idx | 1 | float16 | GQA head selection at dim=1: select 8 KV heads from 32 Q heads per batch; `torch.gather(x, 1, idx)` | Output (4,8,64) matches CPU; per-batch head gather | atol=1e-2 | SENCORES=1 | TestGatherNonLeadingDimEagerCompile::test_gather_dim1_gqa_head_select |
| GNLD-02 | `test_gather_dim1_sequence_position_select` | (B=2, S=128, D=256) | (B=2, Lk=32, D=256) position idx | 1 | float16 | Sequence position gather at dim=1: select Lk=32 positions from seq_len=128; `x[:, pos_ids, :]` form | Output (2,32,256) matches CPU; seq-pos gather | atol=1e-2 | SENCORES=1 | TestGatherNonLeadingDimEagerCompile::test_gather_dim1_sequence_position_select |
| GNLD-03 | `test_gather_dim1_kv_head_reorder` | (B=4, H=8, Lk=64, D=64) | (B=4, H=8, Lk=64, D=64) reorder idx | 1 | float16 | KV head reordering at dim=1 (beam reorder within head axis); `torch.gather(x, 1, idx)` on 4D | Output (4,8,64,64) beam-reordered correctly | atol=1e-2 | SENCORES=1 | TestGatherNonLeadingDimEagerCompile::test_gather_dim1_kv_head_reorder |
| GNLD-04 | `test_gather_dim2_feature_select` | (B=4, S=32, F=512) | (B=4, S=32, K=64) feature idx | 2 | float16 | Feature dimension select at dim=2: select K=64 features from F=512; sparse feature retrieval | Output (4,32,64) matches CPU; feature-dim gather | atol=1e-2 | SENCORES=1 | TestGatherNonLeadingDimEagerCompile::test_gather_dim2_feature_select |
| GNLD-05 | `test_gather_dim2_kv_slot_select` | (B=2, H=8, S=256, D=64) | (B=2, H=8, Lk=32, D=64) slot idx | 2 | float16 | KV slot select at dim=2: select Lk=32 positions from S=256 slots per head; `x[:, :, slots, :]` pattern | Output (2,8,32,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherNonLeadingDimEagerCompile::test_gather_dim2_kv_slot_select |
| GNLD-06 | `test_take_along_dim_dim2` | (B=4, H=8, S=64) — (B, H, S) | (B=4, H=8, Lk=16) topk indices at dim=2 | 2 | float16 | `torch.take_along_dim(x, idx, dim=2)`: top-k position selection at sequence dim; flash-topk attention pattern | Output (4,8,16) matches CPU take_along_dim at dim=2 | atol=1e-2 | SENCORES=1 | TestGatherNonLeadingDimEagerCompile::test_take_along_dim_dim2 |
| GNLD-07 | `test_gather_dim2_bfloat16` | (B=2, H=8, S=128, D=64) | (B=2, H=8, Lk=32, D=64) | 2 | bfloat16 | dim=2 gather in bfloat16; SDSC wordLength=2 at non-leading dim; gather of bfloat16 attention keys | Output (2,8,32,64) matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherNonLeadingDimEagerCompile::test_gather_dim2_bfloat16 |

---

## Summary

| Section | Category | Test IDs | Count |
|---|---|---|---|
| 1 | Core Functional Correctness | GFC-01..GFC-20 | 20 |
| 2 | Data Type Coverage | GDT-01..GDT-16 | 16 |
| 3 | Index Access Patterns | GIDX-01..GIDX-14 | 14 |
| 4 | SpyreTensorLayout (STL) | GSTL-01..GSTL-15 | 15 |
| 5 | Downstream Co-Scheduled Operations | GDS-01..GDS-17 | 17 |
| 6 | Named Dimensions | GNDM-01..GNDM-10 | 10 |
| 7 | Execution Modes | GEM-01..GEM-10 | 10 |
| 8 | Embedding Table Lookup | GEMB-01..GEMB-15 | 15 |
| 9 | MoE Expert Routing | GMOE-01..GMOE-10 | 10 |
| 10 | Paged Attention and KV Cache | GPA-01..GPA-26 | 26 |
| 11 | vLLM Serving Patterns | GVLLM-01..GVLLM-15 | 15 |
| 12 | FMS Serving Patterns | GFMS-01..GFMS-08 | 8 |
| 13 | Real Model Patterns | GMDL-01..GMDL-13 | 13 |
| 14 | STL Paired Comparison | GSTLCMP-01..GSTLCMP-09 | 9 |
| 15 | Compiler Pipeline Validation | GCP-01..GCP-23 | 23 |
| 16 | Memory, Layout, and Scale | GMEM-01..GMEM-07 | 7 |
| 17 | Dynamic and Symbolic Shapes | GDYN-01..GDYN-10 | 10 |
| 18 | Cross-Feature Dependencies | GCFG-01..GCFG-10 | 10 |
| 19 | Chained and Composed Gather | GCH-01..GCH-08 | 8 |
| 20 | Edge Cases and Negative Tests | GEC-01..GEC-19 | 19 |
| 21 | torch.masked_select and Mask-Based Gather | GMSK-01..GMSK-16 | 16 |
| 22 | torch.index_select — Dedicated Corner Cases | GIDXS-01..GIDXS-16 | 16 |
| 23 | EmbeddingBag and Lookup Table Advanced Coverage | GEMB2-01..GEMB2-15 | 15 |
| 24 | Python API Forms — Complete Coverage | GAPI-01..GAPI-25 | 25 |
| 25 | Non-Contiguous Source Patterns | GNCS-01..GNCS-10 | 10 |
| 26 | Known Bug Regression Cases | GBUG-01..GBUG-08 | 8 |
| 27 | Additional Corner Cases | GCOR-01..GCOR-14 | 14 |
| 28 | STL Advanced and Production Patterns | GSTLADV-01..GSTLADV-10 | 10 |
| 29 | Stride-Exact and Precision Real Model Cases | GSTR-01..GSTR-10 | 10 |
| 30 | Chained Gather with chunk, split, and RoPE Patterns | GCHNK-01..GCHNK-15 | 15 |
| 31 | E2E vLLM Decode and Prefill Inference Patterns | GE2E-01..GE2E-14 | 14 |
| 32 | LX Scratchpad and Compiler Optimization Depth | GLXD-01..GLXD-10 | 10 |
| 33 | SDPA with Paged KV Cache (FlashAttention Patterns) | GSDPA-01..GSDPA-20 | 20 |
| 34 | Non-Leading-Dim Gather (dim=1, dim=2 Production Patterns) | GNLD-01..GNLD-07 | 7 |
| — | **Total** | | **472** |

> **Note (v1.1):** Each test ID produces two independent pytest records via `@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])`. Total live pytest records = 822 (test_gather_compiler.py runs compiled-only with 50 tests; all other files run [eager]+[compiled]).
> Section 13 extended: GMDL-14..23 (10 new: TopK MoE + multi-layer KV). Section 15 extended: GCP-15..19 (5 new: in-graph index computation). All class names in "Implemented In" corrected to actual class names.

---

## Section 21 — torch.masked_select and Mask-Based Gather

**Test class:** `TestGatherBoundaryNumericsAndAPIForms`
**File:** `tests/inductor/indirect_access/test_gather_edge_cases.py`

> `torch.masked_select` is the P2 "Mask-as-Compaction" gather operation: output size is data-dependent (determined at runtime by the number of True values in the mask). Tests cover basic correctness, edge mask shapes, per-dtype behavior, mask derivation patterns, dynamic output handling, and chained usage. P2 path is expected to use CPU fallback or raise a controlled error on Spyre (data-dependent shape not yet supported); each case documents the expected classification.
>
> **Indirect Access Validation Focus:** Validates the P2 "Mask-as-Compaction" indirect access classification, where `x[bool_mask]` lowers to `nonzero() + index.Tensor` with a data-dependent output shape. On Spyre, data-dependent shapes are not yet supported on the indirect access path, so these tests verify that the CPU-fallback classification is applied consistently — all-True, all-False, comparison-derived, topk-derived, and broadcast masks all fall back cleanly without a crash. The `vs_nonzero_index` comparison validates API form equivalence; CPU fallback classification is invariant to environment configuration.

| ID | Test Name | Input Shape | Mask Source / Shape | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|
| GMSK-01 | `test_masked_select_1d_basic` | (64,) | Explicit bool (64,) ~50% True | float16 | 1D masked select; data-dependent output length; CPU fallback expected | Output is 1D flat of selected values; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_1d_basic |
| GMSK-02 | `test_masked_select_2d_basic` | (16, 64) | bool (16,64) ~50% True | float16 | 2D input flattened by selection; output length = True count | Output 1D with correct values at True positions; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_2d_basic |
| GMSK-03 | `test_masked_select_all_true` | (16, 64) | All-True mask (16,64) | float16 | All elements selected; output = x.flatten(); deterministic length | Output == x.flatten(); shape (1024,) | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_all_true |
| GMSK-04 | `test_masked_select_all_false` | (16, 64) | All-False mask (16,64) | float16 | No elements selected; output is empty tensor | Output shape (0,); no crash; dtype preserved | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_all_false |
| GMSK-05 | `test_masked_select_single_true` | (16, 64) | Only mask[i,j]=True | float16 | Single element selected; length=1 output; minimal selection | Output shape (1,); value == x[i,j]; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_single_true |
| GMSK-06 | `test_masked_select_comparison_mask` | (32, 64) | `mask = x > 0.5` (derived at runtime) | float16 | Most common real-model pattern: threshold-based selection | Output matches `x[x > 0.5]` CPU result | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_comparison_mask |
| GMSK-07 | `test_masked_select_topk_mask` | (32, 64) | mask from `topk` indices | float16 | TopK used as selection mask (GNN-style neighbor selection) | Output == x[topk_mask]; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_topk_mask |
| GMSK-08 | `test_masked_select_bfloat16` | (16, 64) | bool (16,64) ~50% True | bfloat16 | bfloat16 masked select; dtype preserved in output | Output matches CPU bfloat16 masked values | atol=2e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_bfloat16 |
| GMSK-09 | `test_masked_select_float32` | (16, 64) | bool (16,64) ~50% True | float32 | float32 masked select; higher precision output | Output matches CPU float32 masked values | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_float32 |
| GMSK-10 | `test_masked_select_broadcastable_mask` | (16, 64) | row mask (16,) broadcast to (16,64) | float16 | Broadcast mask (row vector) expands to 2D before selection | Output == x[mask.expand_as(x)]; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_broadcastable_mask |
| GMSK-11 | `test_masked_select_dynamic_output_shape` | (16, 64) | mask varies between runs | float16 | Data-dependent output length; expected CPU fallback for dynamic shape on Spyre | CPU fallback triggered; output matches CPU; no crash | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_dynamic_output_shape |
| GMSK-12 | `test_masked_select_chained_linear` | (16, 64) | bool ~50% True → linear(W:(64,32)) | float16 | Masked select output fed to linear layer; chained ops after gather | `linear(masked_values)` output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_chained_linear |
| GMSK-13 | `test_masked_select_attention_scores` | attn_scores (B=4, H=8, S=64, S=64) | `scores < -1e4` padding mask | float16 | Extract non-masked attention logits; padding mask application | Selected valid scores match CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_attention_scores |
| GMSK-14 | `test_masked_select_moe_expert_tokens` | tokens (N=512, D=256) | router_mask (N,) for expert e | float16 | Per-expert token selection in MoE: `x[router_mask]` for tokens assigned to expert e | Expert-token output matches CPU sparse selection | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_moe_expert_tokens |
| GMSK-15 | `test_masked_select_gate_off` | (16, 64) | bool ~50% True | float16 | masked_select always uses CPU fallback; no Spyre indirect access path for masked_select | CPU fallback used; output matches CPU reference; no error | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_gate_off |
| GMSK-16 | `test_masked_select_vs_nonzero_index` | (16, 64) | bool mask (16,64) | float16 | `x[mask]` vs `x[mask.nonzero()[:,0]]` produce identical results via different API paths | Both outputs identical (bit-exact) | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_masked_select_vs_nonzero_index |

---

## Section 22 — torch.index_select — Dedicated Corner Cases

**Test class:** `TestGatherBoundaryNumericsAndAPIForms`
**File:** `tests/inductor/indirect_access/test_gather_edge_cases.py`

> `torch.index_select` is covered at a basic level in Sections 1 and 3. These tests add dedicated corner cases: all dims, non-contiguous index, repeated indices, boundary values, multi-core behaviour, chained compositions, and cross-dtype combinations not yet validated.
>
> **Indirect Access Validation Focus:** Validates `torch.index_select` as a distinct API form of indirect read access — separate from `x[idx]` (advanced indexing) — across all gather dimensions (dim=0/1/2), 1D source tensors, repeated and boundary indices, int64 auto-downcast, method vs functional API equivalence (both must emit the same `GATHER_OP_SPEC`), downstream chaining with matmul and LayerNorm, multi-core partitioning, and LX planning. The dim=1 and dim=2 variants specifically exercise the non-outermost-dim gather path that requires STL, catching cases where `index_select` at a non-zero dim is incorrectly treated as an outermost-dim gather.

| ID | Test Name | Input Shape | Index Shape | dim | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GIDXS-01 | `test_index_select_dim0_basic` | (32, 256) | (16,) | 0 | float16 | `torch.index_select(x, 0, idx)` primary form; GATHER_OP_SPEC on dim=0 | Output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_dim0_basic |
| GIDXS-02 | `test_index_select_dim1` | (32, 256) | (16,) | 1 | float16 | `torch.index_select(x, 1, idx)` — column gather at dim=1; STL may be required | Output (32,16) matches CPU; STL in compiled graph if needed | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_dim1 |
| GIDXS-03 | `test_index_select_dim2` | (4, 32, 256) | (16,) | 2 | float16 | 3D source, `index_select` at dim=2 (inner); interior gather | Output (4,32,16) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_dim2 |
| GIDXS-04 | `test_index_select_1d_source` | (64,) | (16,) | 0 | float16 | 1D value tensor; each selected element is scalar; output shape (16,) | Output (16,) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_1d_source |
| GIDXS-05 | `test_index_select_all_rows` | (32, 256) | range(32) full | 0 | float16 | Complete row permutation; all rows accessed exactly once | Output is permuted version of input; all rows match CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_all_rows |
| GIDXS-06 | `test_index_select_repeated_rows` | (32, 256) | all-zeros (32,) | 0 | float16 | All idx = 0; repeated reads of first row; 32 identical output rows | All 32 output rows == input row 0; exact match | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_repeated_rows |
| GIDXS-07 | `test_index_select_boundary_first_last` | (32, 256) | [0, 31] | 0 | float16 | First and last rows only; boundary index values | Output (2,256): row 0 and row 31; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_boundary_first_last |
| GIDXS-08 | `test_index_select_bfloat16` | (32, 256) | (16,) | 0 | bfloat16 | bfloat16 source + int32 index via index_select | Output (16,256) matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_bfloat16 |
| GIDXS-09 | `test_index_select_float32` | (32, 256) | (16,) | 0 | float32 | float32 source + int32 index via index_select | Output (16,256) matches CPU float32 | atol=1e-5, rtol=1e-5 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_float32 |
| GIDXS-10 | `test_index_select_int64_index` | (32, 256) | (16,) int64 | 0 | float16 | int64 index triggers auto-downcast to int32; index_select still correct | Output matches CPU; downcast warning emitted | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_int64_index |
| GIDXS-11 | `test_index_select_method_form` | (32, 256) | (16,) | 0 | float16 | `x.index_select(0, idx)` method form produces same GATHER_OP_SPEC as functional | Output matches functional form output exactly | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_method_form |
| GIDXS-12 | `test_index_select_chained_matmul` | x:(32,256), W:(256,128) | (16,) | 0 | float16 | `x.index_select(0, idx) @ W` — gather then matmul; downstream co-scheduling | Output (16,128) matches CPU matmul after gather | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_chained_matmul |
| GIDXS-13 | `test_index_select_chained_layernorm` | (32, 256) | (16,) | 0 | float16 | `layer_norm(x.index_select(0, idx))` — gather then LayerNorm | Output (16,256) with LayerNorm matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_chained_layernorm |
| GIDXS-14 | `test_index_select_sencores4` | (32, 256) | (16,) | 0 | float16 | Multi-core index_select with SENCORES=4; 4-core work division | Output matches CPU at SENCORES=4 | atol=1e-2 | SENCORES=4 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_sencores4 |
| GIDXS-15 | `test_index_select_lx_planning` | (32, 256) | (16,) | 0 | float16 | index_select with LX scratchpad planning; index must stay in HBM | Output matches CPU; idx in HBM; downstream LX-eligible buffers in LX | atol=1e-2 | LX_PLANNING=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_lx_planning |
| GIDXS-16 | `test_index_select_dim2_on_4d` | (2, 4, 32, 64) | (16,) int64 | 2 | float16 | index_select at dim=2 on 4D tensor; head-level sequence position selection; non-outermost dim gather | Output shape (2,4,16,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_select_dim2_on_4d |

---

## Section 23 — EmbeddingBag and Lookup Table Advanced Coverage

**Test class:** `TestGatherEmbeddingTableLookupAndPooling`
**File:** `tests/inductor/indirect_access/test_gather_embedding.py`

> Section 8 covers `nn.Embedding` and `F.embedding` at the basic level. These tests add `nn.EmbeddingBag` modes, sparse weight patterns, production LLM token batching, GNN feature lookup, multi-table MoE routing lookup, and lookup chaining. All are inference-only; no backward or gradient tests.
>
> **Indirect Access Validation Focus:** Validates the broader "lookup table" indirect access pattern beyond basic embedding, including `nn.EmbeddingBag` aggregation modes (mean/sum/max via CPU fallback), variable-length bags with offsets, padding_idx exclusion, weight-tied embedding+LM-head shared matrix, GNN neighbor feature lookup (`node_features[neighbor_ids]` producing a 3D output), multi-table two-gather chains, and indirect access detection (confirming `GATHER_OP_SPEC` is always emitted for `weight[token_ids]` patterns). A key correctness property: weight tying must not cause the embedding lookup and LM-head projection to share or alias their SDSC bundles when both are in the same compiled graph.

| ID | Test Name | Embedding Shape (V, D) | Index / Input Shape | API Form | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GEMB2-01 | `test_embedding_bag_mean` | (4096, 64) | input:(32,), offsets:(4,) | `F.embedding_bag(..., mode="mean")` | float16 | EmbeddingBag mean reduction; CPU fallback path (bag ops not on Spyre); correct aggregation | Output (4,64) matches CPU mean | atol=2e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_bag_mean |
| GEMB2-02 | `test_embedding_bag_sum` | (4096, 64) | input:(32,), offsets:(4,) | `F.embedding_bag(..., mode="sum")` | float16 | EmbeddingBag sum pooling; bag reduction; CPU fallback | Output (4,64) matches CPU sum | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_bag_sum |
| GEMB2-03 | `test_embedding_bag_max` | (4096, 64) | input:(32,), offsets:(4,) | `F.embedding_bag(..., mode="max")` | float16 | EmbeddingBag max pooling; different reduction op; CPU fallback | Output (4,64) matches CPU max | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_bag_max |
| GEMB2-04 | `test_embedding_bag_with_offsets` | (4096, 64) | flat input:(32,), offsets:(4,) | `F.embedding_bag(..., offsets=offsets)` | float16 | Variable-length bags via offsets; each bag has different sequence length | Output (4,64) matches CPU per-bag aggregation | atol=2e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_bag_with_offsets |
| GEMB2-05 | `test_embedding_bag_padding_idx` | (4096, 64) | input:(32,) with zeros | `F.embedding_bag(..., padding_idx=0)` | float16 | Padding token (id=0) excluded from aggregation; padding mask behaviour | Output matches CPU with padding_idx=0 zeroed out | atol=2e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_bag_padding_idx |
| GEMB2-06 | `test_embedding_weight_tying` | (V=4096, D=64) | token_ids:(32,) + hidden:(32,64) | `E.weight` shared for embed + lm_head | float16 | Weight-tied embedding + LM head: same weight matrix used for both; no duplication | Embedding output and LM head output both correct; weight not copied | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_weight_tying |
| GEMB2-07 | `test_embedding_batch_gather` | (32000, 512) | token_ids:(B=4,S=128) | `weight[token_ids]` 2D advanced index | float16 | 2D batched embedding: each of 4 sequences × 128 tokens; output (4,128,512) | Output (4,128,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_batch_gather |
| GEMB2-08 | `test_embedding_gnn_node_features` | (N_nodes=1024, D=64) | neighbor_ids:(N_query=32,K=16) | `node_feats[neighbor_idx]` | float16 | GNN: each query node gathers features from K=16 neighbors; (32,16,64) output | Output (32,16,64) matches CPU neighbor feature gather | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_gnn_node_features |
| GEMB2-09 | `test_embedding_moe_table_lookup` | (E=8,D_in=512,D_out=64) | expert_ids:(32,) | `expert_weights[expert_ids]` advanced index | float16 | MoE routing weight table indexed by selected expert ids; GATHER_OP_SPEC on expert table | Output (32,512,64) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_moe_table_lookup |
| GEMB2-10 | `test_embedding_large_vocab_float16` | (50257, 768) | token_ids:(32,) | `weight[idx]` | float16 | GPT-2 full vocabulary (V=50257, D=768=12 sticks); large-table gather | Output (32,768) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_large_vocab_float16 |
| GEMB2-11 | `test_embedding_large_vocab_bfloat16` | (50257, 768) | token_ids:(32,) | `weight[idx]` | bfloat16 | Same large vocab table in bfloat16 | Output (32,768) matches CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_large_vocab_bfloat16 |
| GEMB2-12 | `test_embedding_chained_with_rope` | tok_table:(32000,128), pos_table:(4096,128) | token_ids:(seq,), positions:(seq,) | Two gathers: `tok_table[ids]` + `pos_table[pos]` | float16 | Two chained gathers in one graph: token embedding + RoPE position cache; two SDSC bundles | Both outputs match CPU; two GATHER_OP_SPECs | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_chained_with_rope |
| GEMB2-13 | `test_embedding_multi_table` | tab1:(8192,128), tab2:(4096,64) | id1:(32,), id2:(32,) | `e1=tab1[id1]; e2=tab2[id2]; cat([e1,e2])` | float16 | Two independent embedding tables concatenated; two SDSC bundles; no cross-link | cat output (32,192) matches CPU; tables separate | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_multi_table |
| GEMB2-14 | `test_embedding_gate_off` | (4096, 64) | token_ids:(32,) | `weight[token_ids]` | float16 | `weight[token_ids]` always emits GATHER_OP_SPEC; indirect access active for embedding lookup; output numerically matches CPU | GATHER_OP_SPEC emitted; output matches CPU reference | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embedding_gate_off |
| GEMB2-15 | `test_embeddingbag_max_float16` | weight:(64, 64) | input_ids:(8,) offsets:(2,) | max | float16 | EmbeddingBag max mode in float16; 2-byte element max reduction via gather; wordLength=2 | Max-pooled output per bag matches CPU at float16 tolerance | atol=1e-2 | SENCORES=1 | TestGatherEmbeddingTableLookupAndPooling::test_embeddingbag_max_float16 |

---

## Section 24 — Python API Forms — Complete Coverage

**Test class:** `TestGatherBoundaryNumericsAndAPIForms`  
**File:** `tests/inductor/indirect_access/test_gather_edge_cases.py`

> Every distinct Python-level API that exercises indirect gather access on Spyre. Tests confirm which form produces GATHER_OP_SPEC, which produces NO_SPYRE_OP (CPU fallback), and which produces DIRECT_OP_SPEC. Covers operator, method, functional, and module forms.
>
> **Indirect Access Validation Focus:** Validates the complete Python API surface for indirect access — confirming which form emits `GATHER_OP_SPEC` (the on-device indirect read path), which emits `NO_SPYRE_OP` (CPU fallback, e.g. `F.embedding_bag`, `F.embedding`), and which emits `DIRECT_OP_SPEC`. This is the definitive API classification table: `x[idx]`, `x.gather()`, `torch.gather()`, `torch.index_select()`, `take_along_dim()`, `take()`, `x[:, idx, :]`, bool/byte masks, repeat_interleave, `no_grad`/`inference_mode` contexts, `fullgraph=True`, and `nn.Module.forward` must all route through the documented op-spec. Any API form that silently falls back to CPU without emitting `GATHER_OP_SPEC` represents lost on-device performance.

| ID | Test Name | Input Shape | Index Shape | API Form | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GAPI-01 | `test_api_advanced_index_operator` | (32, 256) | (16,) | `x[idx]` — `__getitem__` | float16 | Primary API; `aten.index.Tensor` → GATHER_OP_SPEC | GATHER_OP_SPEC in graph; output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_advanced_index_operator |
| GAPI-02 | `test_api_gather_method` | (32, 256) | (16, 256) same-rank | `x.gather(0, idx)` | float16 | Method form of `torch.gather`; same-rank index; GATHER_OP_SPEC | Output (16,256) matches CPU `torch.gather` | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_gather_method |
| GAPI-03 | `test_api_gather_functional` | (32, 256) | (16, 256) same-rank | `torch.gather(x, 0, idx)` | float16 | Functional form; `aten.gather` → GATHER_OP_SPEC | Output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_gather_functional |
| GAPI-04 | `test_api_gather_out_buffer` | (32, 256) | (16, 256) | `torch.gather(x, 0, idx, out=buf)` | float16 | Pre-allocated output buffer; gather writes into `buf` in-place | `buf` contains correct gathered values; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_gather_out_buffer |
| GAPI-05 | `test_api_index_select_functional` | (32, 256) | (16,) | `torch.index_select(x, 0, idx)` | float16 | Functional index_select; 1D index; GATHER_OP_SPEC | Output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_index_select_functional |
| GAPI-06 | `test_api_index_select_method` | (32, 256) | (16,) | `x.index_select(0, idx)` | float16 | Method form of index_select; same as functional | Output identical to functional form | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_index_select_method |
| GAPI-07 | `test_api_take_along_dim` | (32, 256) | (16, 256) | `torch.take_along_dim(x, idx, dim=0)` | float16 | Same-rank index; topk-style selection; GATHER_OP_SPEC | Output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_take_along_dim |
| GAPI-08 | `test_api_take_flat` | (32, 64) | (128,) flat | `torch.take(x, flat_idx)` | float16 | Flat index into linearized tensor; gather by absolute position | Output (128,) matches CPU `torch.take` | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_take_flat |
| GAPI-09 | `test_api_mixed_slice_tensor` | (4, 32, 256) | (16,) on dim=1 | `x[:, idx, :]` — slice + tensor index | float16 | Interior dim gather via slice+tensor; no STL needed if outermost | Output (4,16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_mixed_slice_tensor |
| GAPI-10 | `test_api_const_plus_tensor_idx` | (4, 32, 256) | (16,) on dim=1 | `x[0, idx]` — constant first dim | float16 | First dim constant (=0), second dim tensor; lowers to index on selected slice | Output (16,256) matches CPU `x[0][idx]` | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_const_plus_tensor_idx |
| GAPI-11 | `test_api_setitem_assignment` | out:(32,256), src:(16,256) | (16,) | `out[idx] = src` — `__setitem__` | float16 | Python setitem → `aten.index_put_`; write path, not read path | `out[idx]` rows == `src` after assignment; rest unchanged | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_setitem_assignment |
| GAPI-12 | `test_api_bool_mask_read` | (16, 64) | bool (16,64) | `x[bool_mask]` — boolean mask indexing | float16 | Lowers to nonzero + index; data-dependent shape; CPU fallback expected | Output matches CPU `x[bool_mask]` | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_bool_mask_read |
| GAPI-13 | `test_api_byte_mask_read` | (16, 64) | byte mask (16,64) uint8 | `x[byte_mask]` — legacy byte mask | float16 | Legacy uint8 bool mask form; same path as bool mask | Output matches CPU `x[byte_mask.bool()]` | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_byte_mask_read |
| GAPI-14 | `test_api_bool_mask_write` | out:(16,64), val:(count,64) | bool mask (16,64) | `out[bool_mask] = val` — mask-based write | float16 | Boolean mask scatter write via `index_put_`; write path | `out[bool_mask]` rows == `val` values; rest unchanged | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_bool_mask_write |
| GAPI-15 | `test_api_multidim_advanced` | (16, 32) | idx1:(8,), idx2:(8,) | `x[idx1, idx2]` — two tensor indices | float16 | Multi-dimensional fancy indexing; gather by (row, col) pairs | Output (8,) matches CPU point-wise gather | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_multidim_advanced |
| GAPI-16 | `test_api_repeat_interleave` | (32, 64) | repeats=[2]*32 | `torch.repeat_interleave(x, repeats, dim=0)` | float16 | Decomposes to gather + expand; each row repeated twice | Output (64,64) matches CPU repeat_interleave | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_repeat_interleave |
| GAPI-17 | `test_api_embedding_weight_indexing` | weight:(32000,512) | token_ids:(32,) | `weight[token_ids]` — primary LLM embedding | float16 | Advanced index operator on 2D weight; primary LLM embedding pattern; GATHER_OP_SPEC | Output (32,512) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_embedding_weight_indexing |
| GAPI-18 | `test_api_embedding_bag` | weight:(4096,64) | input:(32,), offsets:(4,) | `F.embedding_bag(input, weight, offsets)` | float16 | EmbeddingBag aggregation; CPU fallback (bag ops not Spyre); correct output | CPU fallback used; output (4,64) matches CPU | atol=2e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_embedding_bag |
| GAPI-19 | `test_api_no_grad_context` | (32, 256) | (16,) | `with torch.no_grad(): x[idx]` | float16 | Gather inside `no_grad` context; no autograd tape; GATHER_OP_SPEC still emitted | Output (16,256) matches CPU; result.requires_grad=False | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_no_grad_context |
| GAPI-20 | `test_api_inference_mode` | (32, 256) | (16,) | `with torch.inference_mode(): x[idx]` | float16 | Inference mode context; lightest weight path; no autograd | Output matches CPU; inference_mode context active | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_inference_mode |
| GAPI-21 | `test_api_fullgraph_compile` | (32, 256) | (16,) | `torch.compile(fn, fullgraph=True)` | float16 | `fullgraph=True`; no graph breaks; gather must be in single graph | No graph break; output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_fullgraph_compile |
| GAPI-22 | `test_api_module_forward` | (32, 256) | (16,) | `nn.Module.forward` with `x[idx]` | float16 | Gather inside compiled `nn.Module`; `@torch.compile` on module | Output matches CPU; module-level compilation OK | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_api_module_forward |
| GAPI-23 | `test_torch_gather_dim1` | (8, 32, 64) | (8, 16, 64) int64 expanded | 1 | float16 | torch.gather at dim=1 on 3D tensor; selects sequence positions per batch-head; expanded index | Output shape (8,16,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_torch_gather_dim1 |
| GAPI-24 | `test_take_along_dim_dim1` | (4, 32, 64) | (4, 16, 64) int64 | 1 | float16 | torch.take_along_dim at dim=1; selects sequence positions per batch head | Output shape (4,16,64); values match CPU | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_take_along_dim_dim1 |
| GAPI-25 | `test_take_along_dim_none_flat` | (16, 32) → reshaped (512,) | flat_idx:(64,) int64 | dim=0 on flattened | float16 | take_along_dim flatten mode (dim=None semantics): x reshaped to 1D then gather; only untested API form; closes coverage gap | Output (64,) matches CPU gather from flattened x | atol=1e-2 | SENCORES=1 | TestGatherShapeRankAndIndexPatterns::test_take_along_dim_none_flat |

---

## Section 25 — Non-Contiguous Source Patterns

**Test class:** `TestGatherBoundaryNumericsAndAPIForms`
**File:** `tests/inductor/indirect_access/test_gather_edge_cases.py`

> The new FVT Section 16 covers transposed value tensor and strided index. These cases add the remaining non-contiguous patterns: row-stride, storage offset (slice), permuted 3D, and non-contiguous arising from real model operations.
>
> **Indirect Access Validation Focus:** Validates that the indirect access address computation accounts for the physical memory layout of non-contiguous tensors — row-strided sources (from `::2` slicing), column-sliced sources (non-zero storage offset), permuted 3D tensors, transposed sources, zero-stride broadcast-expanded tensors, K-slice from QKV split, and non-contiguous indices via `as_strided`. The SDSC `stride_map` field must reflect the actual storage strides, not the logical shape strides; a bug here produces reads from wrong physical addresses that may look plausible numerically but are displaced by the stride difference.

| ID | Test Name | Source Shape | Index Shape | Non-Contiguity Pattern | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GNCS-01 | `test_ncs_row_stride_2` | (64, 256) | (16,) | `src[::2, :]` row-stride=2 | float16 | Gather from row-strided source (non-contiguous via ::2); stride-aware address computation | Output (16,256) matches CPU gather from strided source | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_row_stride_2 |
| GNCS-02 | `test_ncs_storage_offset` | (32, 64) | (16,) | `src[:, 4:20]` column slice | float16 | Non-zero storage offset via column slice; gather from inner 16 cols of 64 | Output (16,16) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_storage_offset |
| GNCS-03 | `test_ncs_permuted_3d` | (4, 32, 256) | (4,) on dim=0 | `src.permute(1, 0, 2)` — (32,4,256) | float16 | Permuted 3D non-contiguous source; strides=(32*256, 256, 1) → rearranged | Output (4,4,256) matches CPU gather from permuted src | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_permuted_3d |
| GNCS-04 | `test_ncs_column_slice` | (32, 256) | (16,) | `src[:, ::2]` col-stride=2 | float16 | Every-other-column source; strides (256, 2) for even cols; gather from column-strided source | Output (16,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_column_slice |
| GNCS-05 | `test_ncs_after_transpose` | (32, 256) | (16,) on dim=0 | `src.T` — strides (1, 32) instead of (256, 1) | float16 | 2D transposed source; column-major layout; gather from non-contiguous transposed tensor | Output (16,32) matches CPU gather from src.T | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_after_transpose |
| GNCS-06 | `test_ncs_expand_broadcast` | (1, 256) broadcast | (16,) | `src.expand(32, 256)` — zero stride on dim=0 | float16 | Zero-stride broadcast expanded tensor; all rows share same memory; gather reads same row repeatedly | All 16 output rows == src[0]; exact match | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_expand_broadcast |
| GNCS-07 | `test_ncs_kv_after_qkv_split` | qkv:(B=1,S=128,3*256) | (16,) on B×S dim | `qkv[:, :, 256:512]` K-slice non-contiguous | float16 | K slice after QKV split (non-zero storage offset); real model pattern; gather from K with stride skip | K output (16,256) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_kv_after_qkv_split |
| GNCS-08 | `test_ncs_gather_then_permute` | (8, 32, 128) | (4,) | `x[idx].permute(0, 2, 1)` — permute downstream | float16 | Gather output becomes non-contiguous after permute; downstream sees non-contig input | Output (4,128,32) matches CPU permuted gather | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_gather_then_permute |
| GNCS-09 | `test_ncs_index_non_contig` | (32, 256) | strided: `base.as_strided((16,),(2,))` | Non-contiguous index via as_strided (stride=2) | float16 | Non-contiguous index tensor; address computation uses index strides correctly | Output (16,256) matches CPU with same strided index | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_index_non_contig |
| GNCS-10 | `test_ncs_output_contiguous_forced` | (32, 256) | (16,) | `x[idx].contiguous()` explicit | float16 | `.contiguous()` after gather; verify output is physically contiguous; no double-copy | Output is_contiguous()==True; values match CPU; no extra alloc | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_ncs_output_contiguous_forced |

---

## Section 26 — Known Bug Regression Cases

**Test class:** `TestGatherBoundaryNumericsAndAPIForms`
**File:** `tests/inductor/indirect_access/test_gather_edge_cases.py`

> Explicit regression tests for all four bugs identified in `files-shared-by-development/issues-identified-indirect-access.md`. Each test captures the exact failure mode, pins the expected behavior, and will serve as a pass indicator when the bug is fixed.
>
> **Indirect Access Validation Focus:** Pins the exact failure mode for each known active bug in the Spyre indirect access implementation: #2662 (SpyreCode parse error — "Device pointer out of program segment bounds" — on single SDSC gather, GBUG-01/02), #2661 (multi-SDSC job plan missing host-compute/transfer step for second SDSC address, GBUG-03), #2654 (`primaryDsInfo_` dimension order causing backend error on specific shapes, GBUG-04), #2653 (corelet splitting `DtException: A transfer cannot move more than one page at a time`, GBUG-05/06), a rank-0 constant codegen failure causing `index_fill_` to crash at codegen (GBUG-07), and a `masked_scatter_` CPU-fallback crash that surfaces before the indirect access path is reached (GBUG-08). Each test currently expects the specific failure and will flip to a clean pass when the corresponding bug is resolved, providing an unambiguous fix-verification signal for every open indirect access defect.

| ID | Test Name | Bug # | Input Shape | Index Shape | Operation | Expected Failure Mode | Fix Indicator | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|
| GBUG-01 | `test_bug2662_single_sdsc_parse_error` | #2662 | (32, 256) | (16,) | `x[idx].exp()` + downstream | `RuntimeError: Failed to parse SpyreCode command: Device pointer 0x34000 is out of program segment bounds` | Bug fixed when test passes without RuntimeError | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug2662_single_sdsc_parse_error |
| GBUG-02 | `test_bug2662_single_sdsc_bare_gather` | #2662 | (32, 256) | (16,) | `x[idx]` bare gather | Same SpyreCode parse error as GBUG-01; bare gather triggers it too | Bug fixed when bare gather runs without error | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug2662_single_sdsc_bare_gather |
| GBUG-03 | `test_bug2661_multi_sdsc_job_plan_missing` | #2661 | (32,256) each | (16,) each | `x[idx1] + x[idx2]` — two SDSCs | Host compute/transfer for SDSC address not generated in job plan; `BackendCompilerFailed` | Bug fixed when binary-gather op compiles and runs | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug2661_multi_sdsc_job_plan_missing |
| GBUG-04 | `test_bug2654_primary_ds_info_order` | #2654 | (32, 256) | (16,) | Compile gather; inspect `primaryDsInfo_` | `primaryDsInfo_` dimension order causes backend error on specific shapes | Bug fixed when correct order in `primaryDsInfo_` produces no error | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug2654_primary_ds_info_order |
| GBUG-05 | `test_bug2653_corelet_split_default` | #2653 | (32, 256) | (16,) | Gather with SENCORES=1 (default) | `DtException: A transfer cannot move more than one page at time` from corelet splitting | Bug fixed when default SENCORES works without SENCORELETS=1 workaround | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug2653_corelet_split_default |
| GBUG-06 | `test_bug2653_corelet_split_sencorelets1` | #2653 workaround | (32, 256) | (16,) | Gather with `SENCORELETS=1 SENCORES=1` | Workaround: with SENCORELETS=1, corelet splitting disabled → no crash | Workaround passes; test is PASS not XFAIL | SENCORELETS=1, SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug2653_corelet_split_sencorelets1 |
| GBUG-07 | `test_bug_index_fill_rank0_crash` | rank-0 codegen | (32, 256) | (16,) + fill_val=0.0 | `x.index_fill_(0, idx, 0.0)` | Crashes at codegen: rank-0 constant (`0.0`) codegen failure; `RuntimeError` | Bug fixed when index_fill_ compiles without crash | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug_index_fill_rank0_crash |
| GBUG-08 | `test_bug_masked_scatter_cpu_fallback_crash` | CPU fallback crash | out:(32,256), src:(count,256) | bool mask (32,256) | `out.masked_scatter_(mask, src)` | Crashes on CPU fallback path before indirect access path; documents exact RuntimeError | Bug fixed when masked_scatter_ works or falls back cleanly | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_bug_masked_scatter_cpu_fallback_crash |

---

## Section 27 — Additional Corner Cases

**Test class:** `TestGatherBoundaryNumericsAndAPIForms`
**File:** `tests/inductor/indirect_access/test_gather_edge_cases.py`

> Cases not in Sections 1–24: zero-size value tensor dimension, NaN/Inf in source, gate toggle between compilations, mixed-device index, special values through gather, and compilation context variants.
>
> **Indirect Access Validation Focus:** Validates that the indirect read path correctly handles special numeric values (NaN, ±Inf, fp16 subnormals) without flushing or corrupting them during SDSC address computation and DMA transfer, and that the compile-cache correctly isolates graphs compiled under different gate settings. The mixed-device tests (index on CPU, value on Spyre or vice versa) confirm that the device-placement guard fires before any SDSC encoding is attempted. The `torch.export` test validates that symbolic dynamic-shapes export of gather produces a valid `ExportedProgram` — a prerequisite for deployment workflows that pre-export models before serving.

| ID | Test Name | Input Shape | Index Shape / Condition | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|
| GCOR-01 | `test_zero_size_value_dim` | (0, 64) zero rows | (0,) empty | float16 | Empty source tensor; zero-row gather; no crash or NaN | Output shape (0,64); no exception | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_zero_size_value_dim |
| GCOR-02 | `test_nan_values_in_source` | (32, 64) with NaN rows | (8,) includes NaN rows | float16 | NaN values at gathered rows preserve NaN in output (bit-exact NaN) | NaN rows in output are NaN; non-NaN rows match CPU | exact (NaN-aware) | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_nan_values_in_source |
| GCOR-03 | `test_inf_values_in_source` | (32, 64) with ±Inf rows | (8,) includes Inf rows | float16 | ±Inf preserved exactly through gather; no clipping | Inf rows contain ±Inf; non-Inf rows match CPU | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_inf_values_in_source |
| GCOR-04 | `test_subnormal_float16` | (32, 64) with fp16 subnormals | (16,) | float16 | fp16 subnormal values (< 6e-8) gathered without flush-to-zero | Subnormal output values match CPU; no ftz corruption | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_subnormal_float16 |
| GCOR-05 | `test_all_zeros_source` | zeros(32, 64) | (16,) | float16 | Gather from all-zero tensor; output all-zero; GATHER_OP_SPEC still emitted | Output == zeros(16,64); GATHER_OP_SPEC present | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_all_zeros_source |
| GCOR-06 | `test_all_ones_source` | ones(32, 64) | (16,) | float16 | Gather from all-ones tensor; output all-ones | Output == ones(16,64) | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_all_ones_source |
| GCOR-07 | `test_gate_toggle_recompile` | (32, 256) | (16,) | float16 | Recompile with changed env (SENCORES change); second compile must not reuse first cached graph | Separate cache entries for each env configuration; both outputs match CPU reference | N/A | SENCORES=1 (recompile) | TestGatherBoundaryNumericsAndAPIForms::test_gate_toggle_recompile |
| GCOR-08 | `test_index_on_cpu_value_on_spyre` | x on Spyre (32, 256) | idx on CPU | float16 | Mixed-device: idx=CPU, x=Spyre; RuntimeError expected | `RuntimeError` about mixed device; clear message | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_on_cpu_value_on_spyre |
| GCOR-09 | `test_index_on_spyre_value_on_cpu` | x on CPU (32, 256) | idx on Spyre | float16 | Reversed mixed-device: idx=Spyre, x=CPU; RuntimeError expected | `RuntimeError` about mixed device | N/A | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_index_on_spyre_value_on_cpu |
| GCOR-10 | `test_no_grad_gather_output` | (32, 256) | (16,) | float16 | `no_grad` context; result.requires_grad=False; no autograd tape recorded | result.requires_grad == False; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_no_grad_gather_output |
| GCOR-11 | `test_inference_mode_gather` | (32, 256) | (16,) | float16 | `inference_mode` context; lightest inference path; no autograd overhead | Inference-mode tensor returned; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_inference_mode_gather |
| GCOR-12 | `test_export_dynamic_gather` | (32, 256) | (P,) dynamic | float16 | `torch.export` with dynamic_shapes `{idx: {0: sym_P}}`; export succeeds | Export produces ExportedProgram; dynamic dim annotated; eval at P=16 matches CPU | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_export_dynamic_gather |
| GCOR-13 | `test_integer_index_exact_boundary` | (32, 64) | [M-1=31] | float16 | idx = M-1 (last valid row); boundary access; no off-by-one | Output row matches input row 31 exactly | exact | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_integer_index_exact_boundary |
| GCOR-14 | `test_large_p_many_repeats_correctness` | (32, 64) | P=320 (10×M), all values 0..31 repeated 10× | float16 | Very large index with heavy repetition; P=10×M; correctness of all 320 output rows | All 320 output rows match their indexed source row | atol=1e-2 | SENCORES=1 | TestGatherBoundaryNumericsAndAPIForms::test_large_p_many_repeats_correctness |

---

## Section 28 — STL Advanced and Production Patterns

**Test class:** `TestGatherSpyreTensorLayoutAnnotation`
**File:** `tests/inductor/indirect_access/test_gather_stl.py`

> Covers the 4-argument SpyreTensorLayout form with `dim_order`, nested `spyre_hint` tiling as used in `test_paged.py`, and the gather→permute→contiguous→matmul chain critical for production paged attention.
>
> **Indirect Access Validation Focus:** Validates the most complex STL usage pattern — the 4-argument `SpyreTensorLayout(device_size, stride_map, dtype, dim_order)` form that reorders tensor axes before SDSC encoding — combined with nested 4-level `spyre_hint` tile hierarchies and the gather→permute→contiguous→matmul chain that is the exact production paged attention code path. The `dim_order=[1,0,2]` form is critical for paged attention: without it, the SDSC `device_coordinates` encodes the wrong outermost dimension, causing the hardware to page-miss into head-dim memory instead of cache-slot memory.

| ID | Test Name | KV Cache Shape | Slot / Index Shape | STL 4-arg Form | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GSTLADV-01 | `test_stl_4arg_dim_order` | (512, 8, 64) — (cache, H, D) | (4, 32) | `SpyreTensorLayout([512,8,64],[8*64,64,1],float16,[1,0,2])` | float16 | 4-arg STL with dim_order=[1,0,2]; cache becomes device outermost dim; SDSC device_coordinates reflect reorder | Output (4,32,8,64) matches CPU; STL dim_order=[1,0,2] in SDSC | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_4arg_dim_order |
| GSTLADV-02 | `test_stl_4arg_kv_exact_paged` | (32768, 32, 128) — production canonical | (12, 256) | `SpyreTensorLayout([32768,32,128],[32*128,128,1],torch.float16,[1,0,2])` | float16 | Exact canonical paged-attention STL from test_paged.py: cache_size=32768, H=32, D=128 | Output (12,256,32,128) matches CPU; STL fields match spec | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_4arg_kv_exact_paged |
| GSTLADV-03 | `test_stl_4arg_fms_4d` | (128, 8, 16, 128) — (pages, kv_h, page_sz, D) | (4, 16) | `SpyreTensorLayout([128,8,16,128],[8*16*128,16*128,128,1],float16,[0,2,1,3])` | float16 | FMS 4D layout with dim_order=[0,2,1,3]; kv_heads and page_sz reordered for device layout | Output (4,16,8,128) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_4arg_fms_4d |
| GSTLADV-04 | `test_gather_then_permute_contiguous` | (512, 8, 64) | (4, 32) slot | `keys[slot_idxs].permute(0,2,3,1).contiguous()` | float16 | Gather → permute → contiguous chain; production paged attention output layout transformation | Output (4,32,64,8) contiguous; values match CPU permuted gather | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_gather_then_permute_contiguous |
| GSTLADV-05 | `test_nested_spyre_hint_4levels` | (512, 8, 64) | (B=2, Lk=64) | 4-level `spyre_hint`: B→Lq→H→Lk tiles | float16 | 4-level nested tile hints from test_paged.py; tiling hierarchy; GATHER_OP_SPEC inside innermost tile | Compile succeeds with 4-level hint; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_nested_spyre_hint_4levels |
| GSTLADV-06 | `test_nested_spyre_hint_kv_gather` | (512, 8, 64) K and V | (B=2, Lk=64) | 4-level hint; `keys[slot]` and `values[slot]` inside tile | float16 | Two independent KV gathers inside 4-level nested hint; two SDSC bundles in tiled graph | Both K and V outputs match CPU; two SDSC bundles | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_nested_spyre_hint_kv_gather |
| GSTLADV-07 | `test_stl_dim_order_validation` | (32, 256) | (16,) | `dim_order=[0,2]` wrong length | float16 | Invalid `dim_order` (wrong length or out-of-range index) raises clear `ValueError` or `RuntimeError` | `ValueError` raised; error message mentions `dim_order` | N/A | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_dim_order_validation |
| GSTLADV-08 | `test_stl_4arg_with_named_dims` | (512, 8, 64) | (4, 32) | 4-arg STL + `name_tensor_dims` on all tensors | float16 | 4-arg STL dim_order + named dims co-exist; both annotations in compiled graph | STL dim_order and named dims both present; output matches CPU | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_4arg_with_named_dims |
| GSTLADV-09 | `test_full_paged_attention_pattern` | (512, 4, 64) mini | (B=2, Lk=64) | Full pattern: STL 4-arg + nested hint + online softmax + two gathers | float16 | Mini-scale replica of test_paged.py: B=2, H=4, Lq=64, Lk=64, D=64, cache=512; online softmax accumulation | Attention output (2,4,64,64) matches CPU full-attention | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_full_paged_attention_pattern |
| GSTLADV-10 | `test_stl_interior_vs_dim_order` | (512, 8, 64) | (4, 32) | A: `x[:,idx,:]` interior  B: 4-arg STL with dim_order=[1,0,2] | float16 | Both forms express same logical gather; verify numerical equivalence despite different SDSC encoding | A output == B output (bit-exact); SDSC device_coordinates differs | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_stl_interior_vs_dim_order |

---

## Section 29 — Stride-Exact and Precision Real Model Cases

**Test class:** `TestGatherSpyreTensorLayoutAnnotation`
**File:** `tests/inductor/indirect_access/test_gather_stl.py`

> Uses exact tensor shapes and strides extracted from Q2/Q3 model profiling (from `files-shared-by-development/test-cases-to-consider.md` and `vLLM/vllm_spyre_issue322.md`). Tests pin the specific stride values to catch address-computation regressions when strides change.
>
> **Indirect Access Validation Focus:** Validates SDSC `stride_map` correctness against exact per-model stride tuples extracted from production profiling — gpt-oss-20b strides `[65536, 8192, 64, 1]`, Granite-3.3-8b/Mistral strides `[2097152, 262144, 128, 1]`, and the Mistral-Small multimodal zero-stride broadcast mask `[855*5120, 5120, 0]`. Any change to the stride-map computation formula in `pass_utils.py` or `detect_indirect_access.py` that shifts the SDSC field by even one element width would be caught immediately by these pinned stride tests, before it reaches production models.

| ID | Test Name | Tensor Shape | Exact Strides | Index Shape | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GSTR-01 | `test_gpt20b_kv_strides_prefill` | (1, 8, 128, 64) | [65536, 8192, 64, 1] | (11,) on dim=2 | float16 | gpt-oss-20b small-cache exact strides; `index_copy_(2, cache_pos, key)` writes 11 positions | Written positions match key values; strides preserved in SDSC | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_gpt20b_kv_strides_prefill |
| GSTR-02 | `test_gpt20b_kv_strides_full` | (1, 8, 2048, 64) | [1048576, 131072, 64, 1] | (11,) on dim=2 | float16 | gpt-oss-20b full-cache strides; larger dim=2; SDSC stride_map matches [131072, 64, 1] for H×cache slice | Written positions match; no stride computation regression | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_gpt20b_kv_strides_full |
| GSTR-03 | `test_granite_kv_strides` | (1, 8, 2048, 128) | [2097152, 262144, 128, 1] | (41,) on dim=2 | float16 | Granite-3.3-8b / Ministral / Mistral-Small exact strides; 41-token prefill; SDSC stride map pin | 41 positions written; stride-map in SDSC = [262144,128,1] for dim slicing | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_granite_kv_strides |
| GSTR-04 | `test_granite_decode_single_step` | (1, 8, 2048, 128) | [2097152, 262144, 128, 1] | (1,) on dim=2 | float16 | Granite single-step decode: P=1 at write path (index_copy_); one position update | One cache position updated correctly; stride computation at P=1 | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_granite_decode_single_step |
| GSTR-05 | `test_mistral_24b_long_index` | (1, 8, 2048, 128) | [2097152, 262144, 128, 1] | (855,) on dim=2 | float16 | Mistral-Small-3.2-24B: 855-token long prefill; large write index; SDSC handles P=855 | All 855 positions correct; no stride overflow at large P | exact | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_mistral_24b_long_index |
| GSTR-06 | `test_granite_kv_read_gather` | (1, 8, 2048, 128) | [2097152, 262144, 128, 1] | (41,) on dim=2 read | float16 | Read path: `cache[:, :, idx, :]` gather (not write); exact strides used in GATHER_OP_SPEC address | Gathered positions match cache values; stride-exact read | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_granite_kv_read_gather |
| GSTR-07 | `test_masked_scatter_broadcast_stride` | inputs_embeds (1, 855, 5120) | bool mask strides [855*5120, 5120, 0] — zero stride in last dim | float16 | Mistral-Small multimodal: broadcast mask with zero stride in dim=2; `masked_scatter` reads mask with stride=0 | masked_scatter output matches CPU; zero-stride mask handled correctly | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_masked_scatter_broadcast_stride |
| GSTR-08 | `test_rope_cos_sin_cache_strides` | (4096, 128) | [128, 1] | positions (seq=128,) | float16 | RoPE cos/sin cache: contiguous [128,1] strides; `index_select(0, positions)` with GATHER_OP_SPEC | Output (128,128) matches CPU; strides correct in SDSC | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_rope_cos_sin_cache_strides |
| GSTR-09 | `test_lm_head_last_token_strides` | (1, 855, 5120) | [855*5120, 5120, 1] | (1,) at last S position | float16 | Last-token extraction from sequence: `hidden[:, -1, :]`; stride-aware dim=1 gather | Output (1,5120) matches CPU last-token slice | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_lm_head_last_token_strides |
| GSTR-10 | `test_precision_softmax_long_seq` | source (512, 4096) | (16,) — long inner dim | float16 | Gather 16 rows of 4096-dim then softmax(dim=-1); numerical stability at long seq | Softmax output (16,4096) rows sum to 1.0; matches CPU within tolerance | atol=1e-2 | SENCORES=1 | TestGatherSpyreTensorLayoutAnnotation::test_precision_softmax_long_seq |

---

## Section 30 — Chained Gather with chunk, split, and RoPE Patterns

**Test class:** `TestGatherComposedChainsAndEndToEndPipelines`
**File:** `tests/inductor/indirect_access/test_gather_chained.py`

> The `torch.chunk` and `torch.split` ops applied to a gathered result are critical production patterns: RoPE computes `cos_sin_cache.index_select(0, positions)` then `.chunk(2, dim=-1)` to unpack (cos, sin). These patterns are NOT covered by Section 5 (which only chains unary/reduction directly) or Section 8 (which covers the `index_select` step alone). All tests unique from existing TestGather/TestInternals suites.
>
> **Indirect Access Validation Focus:** Validates that `chunk` and `split` operations downstream of a gather are correctly co-scheduled or cleanly split from the `GATHER_OP_SPEC` bundle — the exact pattern in `SpyreRotaryEmbedding`: `cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)`. If the chunk/split is not wired to the gather output buffer correctly, the cos and sin tensors will alias or point to the wrong half of the gathered rows. Also validates SwiGLU gate/up split, QKV split, 4-way chunk, MoE per-expert token split, and GPT-J/NeoX interleaved RoPE rotation patterns — all using gather + structural split as a combined indirect access + data decomposition primitive.

| ID | Test Name | Source Shape | Index Shape | Composed Pattern | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GCHNK-01 | `test_index_select_then_chunk2` | cos_sin_cache (4096, 128) | positions (seq=64,) | `x.index_select(0, pos).chunk(2, dim=-1)` → cos:(64,64), sin:(64,64) | float16 | Exact SpyreRotaryEmbedding RoPE pattern; GATHER_OP_SPEC then chunk; two output tensors | cos and sin each (64,64); both match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_index_select_then_chunk2 |
| GCHNK-02 | `test_index_select_chunk_apply_rotation` | cache (4096, 128); q (64, 64) | positions (64,) | `cos,sin=cache[pos].chunk(2,-1); out=q*cos+rotate_half(q)*sin` | float16 | Full RoPE rotation: gather→chunk→dual elementwise multiply+add; complete SpyreRotaryEmbedding forward | out (64,64) matches CPU RoPE result | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_index_select_chunk_apply_rotation |
| GCHNK-03 | `test_gather_then_chunk4` | (32, 256) | (16,) | `x[idx].chunk(4, dim=-1)` → 4×(16,64) | float16 | 4-way equal chunk of gathered result; gather then split inner dim into 4 equal parts | All 4 chunks (16,64) match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_then_chunk4 |
| GCHNK-04 | `test_gather_then_split_qkv` | (seq=128, 3*256) | (16,) on seq dim | `qkv[idx].split([256,256,256], dim=-1)` → Q,K,V each (16,256) | float16 | QKV split after gather; non-uniform sizes (q_d, k_d, v_d); three output tensors | Q,K,V each (16,256); match CPU split | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_then_split_qkv |
| GCHNK-05 | `test_gather_then_chunk_then_matmul` | cache (4096, 128); q (64, 64) | positions (64,) | `cos,sin=cache[pos].chunk(2,-1); out=q*cos` | float16 | Gather→chunk→elementwise multiply; first half of RoPE forward; GATHER_OP_SPEC fused | out (64,64) = q * cos; matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_then_chunk_then_matmul |
| GCHNK-06 | `test_gather_then_view_then_chunk` | (32, 256) | (16,) | `x[idx].view(16,4,64).chunk(2,dim=-1)` | float16 | Reshape to head dims (H=4, D_h=64) then chunk each head; real model per-head decomposition | 2 chunks each (16,4,32); match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_then_view_then_chunk |
| GCHNK-07 | `test_gather_split_uniform` | (32, 256) | (16,) | `x[idx].split(64, dim=-1)` → 4×(16,64) | float16 | Uniform split by chunk-size=64; `torch.split(size, dim)` form | All 4 splits (16,64) match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_split_uniform |
| GCHNK-08 | `test_gather_chunk_named_dims` | cache (4096, 128) named `['pos','rot_dim']` | positions (64,) named `['seq']` | `cache[pos].chunk(2, dim=-1)` with named dims | float16 | Named dims propagate through gather and chunk; both output tensors carry annotations | Named dim annotations in both chunk outputs; values match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_chunk_named_dims |
| GCHNK-09 | `test_gather_chunk_bfloat16` | cos_sin_cache (4096, 128) | positions (64,) | `x.index_select(0, pos).chunk(2, dim=-1)` — bfloat16 | bfloat16 | bfloat16 RoPE cache gather + chunk; wordLength=2 in SDSC; bfloat16 precision correct | cos and sin (64,64) match CPU bfloat16 | atol=2e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_chunk_bfloat16 |
| GCHNK-10 | `test_embedding_then_chunk_gate` | embed_proj (V=4096, 2*256) | token_ids (seq=64,) | `gate,up=embed_proj[token_ids].chunk(2,dim=-1)` | float16 | SwiGLU: embedding table treated as gate/up projection; chunk into gate (64,256) and up (64,256) | gate and up each (64,256); match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_embedding_then_chunk_gate |
| GCHNK-11 | `test_gather_chunk_sencores4` | cache (4096, 128) | positions (64,) | `cache[pos].chunk(2, dim=-1)` at SENCORES=4 | float16 | Multi-core gather then chunk; SENCORES=4 work-division with downstream chunk split | cos and sin (64,64) match CPU; 4-core execution | atol=1e-2 | SENCORES=4 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_chunk_sencores4 |
| GCHNK-12 | `test_gather_split_moe_experts` | tokens (N=512, D=256) | expert_ids (N,) | `x[expert_ids].split([n_e0,n_e1,...,n_e7], dim=0)` — 8 expert splits | float16 | MoE: gather all expert-assigned tokens then split by per-expert token count; 8 output tensors | All 8 expert-token tensors match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_gather_split_moe_experts |
| GCHNK-13 | `test_rope_interleaved_gptj_style` | cos/sin:(2048, 64) | pos:(32,) | interleaved rotate | float16 | GPT-J style interleaved RoPE: gather cos/sin at positions then apply interleaved rotation on Q | Rotated Q matches CPU; interleaved even/odd channel pattern correct | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_rope_interleaved_gptj_style |
| GCHNK-14 | `test_rope_neox_qk_both` | cos/sin:(2048, 64) | pos:(32,) | rotate Q and K | float16 | NeoX-style RoPE applied to both Q and K after gather; two rotation ops in one compiled graph | Both rotated Q and K match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_rope_neox_qk_both |
| GCHNK-15 | `test_rope_interleaved_separate_cos_sin_caches` | cos:(2048,64), sin:(2048,64) | pos:(32,) | two gathers | float16 | Two separate cos/sin caches gathered at same positions; two independent GATHER_OP_SPECs; apply interleaved rotation | Both rotated output tensors match CPU; two gather ops verified | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_rope_interleaved_separate_cos_sin_caches |

---

## Section 31 — E2E vLLM Decode and Prefill Inference Patterns

**Test class:** `TestGatherComposedChainsAndEndToEndPipelines`
**File:** `tests/inductor/indirect_access/test_gather_chained.py`

> End-to-end patterns directly from `vllm_spyre_issue322.md` (spyre-inference#322) and `q2_q3_model_ops_test_scenarios.md`. Covers the full vLLM inference pipeline's indirect access points: token embedding, KV cache indexing, logit extraction, and the prefill→decode transition. Tests verify on-device execution (no CPU fallback) for each gather operation.
>
> Key reference patterns: `INDIRECT-001..006` from vLLM scenarios, FX graph `index_put + copy_` → `index_put_` transformation from `test-cases-to-consider.md`.
>
> **Indirect Access Validation Focus:** Validates full end-to-end indirect access pipelines extracted directly from vLLM serving: on-device token embedding (replacing the `aten.embedding` CPU fallback, issue #420), KV cache write via `index_copy_`, sequential vs paged KV read, growing decode cache, ragged-batch per-sample indexing, the FX `index_put + copy_` → `index_put_` graph transform that eliminates a redundant buffer copy, and the prefill→decode phase transition that must leave the KV cache in a consistent state. Each test verifies that a specific indirect access point executes on Spyre (`GATHER_OP_SPEC` in graph) rather than falling back to CPU — together they constitute the end-to-end inference latency regression suite.

| ID | Test Name | Tensor Shape | Index / Position | E2E Pattern | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GE2E-01 | `test_token_embedding_on_device` | vocab_table (128256, 4096) | token_ids (B=1,S=128) | `embedding_table[token_ids]` — issue #420 | float16 | Embedding fires GATHER_OP_SPEC (not CPU fallback `aten.embedding`); on-device execution | GATHER_OP_SPEC in graph; output (1,128,4096) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_token_embedding_on_device |
| GE2E-02 | `test_decode_logit_extraction` | hidden (B=12, S=1, D=4096) | logits_indices (12,) | `hidden_states[logits_indices]` — row extraction | float16 | Decode: extract hidden states at valid positions; indirect row selection | Output (12,4096) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_decode_logit_extraction |
| GE2E-03 | `test_last_token_gather_decode` | hidden (B=12, S=1, D=4096) | `[:, -1, :]` — last pos | `hidden_states[:, -1, :]` last-token extract | float16 | Last-token gather for autoregressive decode; dim=1 interior gather | Output (12,4096) matches CPU last-position slice | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_last_token_gather_decode |
| GE2E-04 | `test_kv_cache_append_position` | cache (1,8,2048,128) | cache_position (seq_len,) | `index_copy_(2, cache_position, key_states)` write | float16 | KV cache write via index_copy_ at specific position dim=2; write path | Written positions match key_states; rest of cache unchanged | exact | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_kv_cache_append_position |
| GE2E-05 | `test_kv_cache_read_prefill` | cache (1,8,2048,128) | `[:, :, :seq_len, :]` seq_len=512 | Full sequential KV read during prefill | float16 | Full-range KV read; sequential contiguous positions; prefill access | Output (1,8,512,128) matches cache slice | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_kv_cache_read_prefill |
| GE2E-06 | `test_kv_cache_read_decode` | cache (1,8,2048,128) | `[:, :, :cache_len+1, :]` growing | Incremental KV read; cache_len grows each step | float16 | Growing cache read at decode; verify position 0..cache_len included correctly | Output matches cache[:,:,:step+1,:] at each step | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_kv_cache_read_decode |
| GE2E-07 | `test_ragged_batch_cache_indexing` | cache (pool,8,64) | seq_lens (B=12,) per-sample | `cache[b, :, seq_lens[b], :]` per batch — vLLM INDIRECT-006 | float16 | Per-sample variable cache position; ragged batch indexing; each request at its own length | Each request output matches cache at its seq_lens[b] position | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_ragged_batch_cache_indexing |
| GE2E-08 | `test_qkv_split_view_source` | qkv (B=1,S=1,3*256) | — (split, no tensor index) | `_,_,v=qkv.split([256,256,256],-1); v.view(-1,8,32)` | float16 | vLLM QKVParallelLinear: non-contiguous v from split as gather source; strides from split preserved | v reshaped to (1,8,32) matches CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_qkv_split_view_source |
| GE2E-09 | `test_fx_index_put_copy_transform` | cache (1,8,2048,128) | cache_position (41,) | FX: `index_put + copy_` → `index_put_` single in-place op | float16 | FX graph transform from test-cases-to-consider.md; `index_put + copy_` eliminated into `index_put_` | FX graph has `index_put_` not the two-op pattern; cache updated correctly | exact | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_fx_index_put_copy_transform |
| GE2E-10 | `test_prefill_then_decode_transition` | cache (1,8,2048,128) | prefill:(512,) → decode:(1,) | Two-phase: prefill writes 512 positions then decode reads 1 | float16 | Phase transition: KV cache filled at prefill; decode reads from filled cache; cache state consistent | Decode-step KV matches prefill-written value at same position | exact | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_prefill_then_decode_transition |
| GE2E-11 | `test_rope_embed_index_select_chunk` | cos_sin_cache (max_pos=4096, 128) | positions (seq=128,) | `cos_sin=cos_sin_cache.index_select(0,pos); cos,sin=cos_sin.chunk(2,-1)` | float16 | Exact SpyreRotaryEmbedding (issue #1668): `index_select` → GATHER_OP_SPEC then `.chunk(2,-1)` unpack | cos (128,64) and sin (128,64) match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_rope_embed_index_select_chunk |
| GE2E-12 | `test_two_hop_gather_embed_kv` | vocab (128256,4096); kv_cache (pool,8,64) | token_ids (128,); slot_idxs (4,32) | `tok_emb=vocab[token_ids]; kv=kv_cache[slot_idxs]` — two independent gathers | float16 | Token embedding + KV gather in one compiled function; two GATHER_OP_SPECs; independent SDSC bundles | tok_emb (128,4096) and kv (4,32,8,64) both match CPU | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_two_hop_gather_embed_kv |
| GE2E-13 | `test_decode_symbolic_seq_len` | cache (1,8,2048,128) | slot (B,Lk) — Lk=1 symbolic | Decode with symbolic S=1; `torch.compile(dynamic=True)` | float16 | Symbolic decode step; Lk=1 at runtime; graph reused from prefill compilation | Output matches CPU decode step; compile count=1 across decode steps | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_decode_symbolic_seq_len |
| GE2E-14 | `test_lm_head_vocab_parallel_decode` | hidden (B=12,1,4096); lm_head (4096,32000) | `[:, 0, :]` last-token | `logits=F.linear(hidden,lm_head)[:,0,:]` — decode LM head | float16 | Last-token LM head logit: linear then index dim=1 at pos 0; gather at output of linear | logits (12,32000) matches CPU linear+slice | atol=1e-2 | SENCORES=1 | TestGatherComposedChainsAndEndToEndPipelines::test_lm_head_vocab_parallel_decode |

---

## Section 32 — LX Scratchpad and Compiler Optimization Depth

**Test class:** `TestGatherCompilerPassesMemoryAndDynamicShapes`
**File:** `tests/inductor/indirect_access/test_gather_compiler.py`

> Section 7 covers `LX_PLANNING=1` basic case and Section 18 covers `CO_OPTIMIZING_LX_PLANNING`, `DXP_LX_FRAC_AVAIL=0.5`, and `CHUNK_LARGE_TENSORS`. These tests add the missing depth: `DXP_LX_FRAC_AVAIL` at 0.1 and 0.8, `LAYOUT_SOLVER` variants, LX-LX overlap, and compound combinations. All are unique from existing TestInternals SDSC structural tests.
>
> **Indirect Access Validation Focus:** Validates the LX scratchpad planning system's interaction with indirect access at depth: index tensors must remain in HBM under all LX budget levels (tight 10% to generous 80%), LX-LX overlap scheduling for two concurrent gathers must not produce address aliasing, compute-data overlap can hide gather address computation behind a downstream matmul, and all three `LAYOUT_SOLVER` modes (firstfit, bestfit, greedy) must produce the same correct numerical output. The maximum-combination test (SENCORES=32 + CO_OPTIMIZING_LX_PLANNING=1 + LX_PLANNING=1) exercises the most complex multi-constraint allocation scenario that production serving will encounter.

| ID | Test Name | Input Shape | Index Shape | LX Config | dtype | What It Validates | Pass Criteria | Tol | Env | Implemented In |
|---|---|---|---|---|---|---|---|---|---|---|
| GLXD-01 | `test_lx_frac_tight` | (32, 256) | (16,) | DXP_LX_FRAC_AVAIL=0.1 (10% LX budget) | float16 | Tight LX budget; idx stays in HBM; downstream may be evicted from LX; no crash or OOM | Output (16,256) matches CPU; no LX allocation failure | atol=1e-2 | LX_PLANNING=1, DXP_LX_FRAC_AVAIL=0.1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_frac_tight |
| GLXD-02 | `test_lx_frac_generous` | (32, 256) | (16,) | DXP_LX_FRAC_AVAIL=0.8 (80% LX budget) | float16 | Generous LX (80%); downstream intermediate easily fits in LX; idx strictly in HBM | Output matches CPU; downstream buffer in LX; idx in HBM | atol=1e-2 | LX_PLANNING=1, DXP_LX_FRAC_AVAIL=0.8 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_frac_generous |
| GLXD-03 | `test_lx_overlap_two_gathers` | (32,256) x and y | (16,) each | LX_PLANNING=1; LX-LX overlap | float16 | Two gathers `a=x[idx1]; b=y[idx2]; out=a+b`; address computations overlap in scratchpad; DMA hiding | out (16,256) matches CPU; two SDSC bundles; LX-LX overlap scheduling | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_overlap_two_gathers |
| GLXD-04 | `test_lx_overlap_gather_matmul` | cache (512,8,64); q (4,1,8,64) | slots (4,32) | LX_PLANNING=1; compute-data overlap | float16 | `keys=cache[slots]; scores=q@keys.T`; KV gather address compute hidden behind matmul | scores (4,1,8,32) match CPU; no LX oversubscription | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_overlap_gather_matmul |
| GLXD-05 | `test_layout_solver_firstfit` | (32, 256) | (16,) | LAYOUT_SOLVER=firstfit | float16 | firstfit LX layout solver; first-available buffer assignment; verify correctness vs greedy default | Output matches CPU; no solver crash; same allocation count as greedy | atol=1e-2 | LX_PLANNING=1, LAYOUT_SOLVER=firstfit | TestGatherCompilerPassesMemoryAndDynamicShapes::test_layout_solver_firstfit |
| GLXD-06 | `test_layout_solver_bestfit` | (32, 256) | (16,) | LAYOUT_SOLVER=bestfit | float16 | bestfit LX solver; minimizes wasted space; verify correctness vs greedy | Output matches CPU; bestfit may reduce LX fragmentation | atol=1e-2 | LX_PLANNING=1, LAYOUT_SOLVER=bestfit | TestGatherCompilerPassesMemoryAndDynamicShapes::test_layout_solver_bestfit |
| GLXD-07 | `test_co_opt_lx_3d_gather` | (8, 32, 128) | (4,) | CO_OPTIMIZING_LX_PLANNING=1 | float16 | Co-optimized LX with 3D value tensor; co-opt planner integrates 3D gather into LX schedule | Output (4,32,128) matches CPU; co-opt LX allocation correct | atol=1e-2 | CO_OPTIMIZING_LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_co_opt_lx_3d_gather |
| GLXD-08 | `test_co_opt_lx_stl_4arg` | (512, 8, 64) — KV cache | (4, 32) | CO_OPTIMIZING_LX_PLANNING=1 + STL 4-arg dim_order=[1,0,2] | float16 | Most complex layout: co-opt LX + STL 4-arg form with dim_order; co-planner sees reordered layout | Output (4,32,8,64) matches CPU; no LX/STL conflict | atol=1e-2 | CO_OPTIMIZING_LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_co_opt_lx_stl_4arg |
| GLXD-09 | `test_lx_named_dims_combo` | (512, 8, 64) | (B=4,Lk=32) named dims | LX_PLANNING=1 + named dims B,Lk,H,D | float16 | LX planning with fully named tensors; planner respects named dim ordering when assigning LX | Named dims preserved; idx in HBM; output matches CPU | atol=1e-2 | LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_named_dims_combo |
| GLXD-10 | `test_lx_sencores32_co_opt` | (64, 256) | (32,) | LX_PLANNING=1 + SENCORES=32 + CO_OPTIMIZING_LX_PLANNING=1 | float16 | Maximum parallelism + co-opt LX: 32 cores, co-optimized LX, gather; all three features together | Output (32,256) matches CPU; no crash; 32-core LX co-opt scheduling | atol=1e-2 | LX_PLANNING=1, SENCORES=32, CO_OPTIMIZING_LX_PLANNING=1 | TestGatherCompilerPassesMemoryAndDynamicShapes::test_lx_sencores32_co_opt |

---

## FRS Requirements Coverage Matrix

| FRS Requirement | Sections | Test Count |
|---|---|---|
| FR-1 — Supported APIs | 1, 8, 9, 10, 11, 12, 13, 14, 19, 20, 21, 22, 23, 24, 27, 31 | ~140 |
| FR-2 — Shapes and dtypes | 1, 2, 3, 8, 9, 10, 11, 16, 17, 20, 21, 23, 25, 27, 29 | ~125 |
| FR-3 — Gather dim=0 constraint | 1, 4, 20, 22 | ~28 |
| FR-3.1 — SpyreTensorLayout | 4, 6, 10, 14, 26, 28, 29 | ~68 |
| FR-4 — Named dimensions | 6, 10, 12, 21, 26, 32 | ~20 |
| FR-5 — OpSpec encoding | 15, 24 | ~7 |
| FR-6 — SDSC bundle encoding | 2, 7, 15, 16, 18, 21, 24, 29, 32 | ~40 |
| FR-7 — Downstream co-scheduled ops | 5, 8, 13, 19, 21, 26, 27, 30, 31 | ~52 |
| FR-8 — Detection during layout propagation | 7, 15, 20, 21, 25, 32 | ~25 |
| FR-9 — E2E execution boundary | 7, 10, 12, 18, 22, 24, 25, 26, 31 | ~44 |

---

## Document History

| Version | Date | Author | Changes |
|---|---|---|---|
| 1.0 | 2026-07-19 | Mahalakshmi Pathivada | Initial document. 437 net-new test scenarios across 32 sections. All scenarios unique from existing TestGather (45), TestScatter (16), TestInternals (63), upstream YAML configs (580), and prior FVT plans. |
| 1.3 | 2026-08-08 | Mahalakshmi Pathivada | Dead-code removal: `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS` confirmed absent from torch-spyre source (not in `config.py` or any `torch_spyre/` file). Removed flag constant and all `os.environ[_ENABLE_FLAG]` set/pop calls from all 10 FVT test files and `conftest.py`. Updated gate=0 test docstrings to reflect actual behavior (numeric correctness). FVT document: removed `SPYRE_INDUCTOR_ENABLE_ADD_INDEX_TO_ADDRESS` from env variable legend; replaced all 433 `gate=1`/`gate=0` Env column values with `SENCORES=1`; updated 6 rows whose Purpose/Expected referenced gate semantics; updated 4 section blockquotes. Total test IDs unchanged: 482. |
| 1.4 | 2026-08-09 | Mahalakshmi Pathivada | Scenario-level duplicate removal. Confirmed 15 tests with zero scenario variation (identical shape/op/dtype/index/env/tolerance) and removed them: gcp02 `test_gather_gate_off_no_op_spec`, gcp03 `test_compute_op_routing`, gcp06 `test_pass_order_is_correct`, gcp09 `test_indirect_alloc_type`, gcp10 `test_fx_direct_op_spec`, gcp11 `test_compile_debug_artifacts`, cfg07 `test_lx_planning_off`, cfg08 `test_fusion_disabled_still_correct` (all were bare compare_mode clones of gcp01 with no structural assertions); api01 `test_api_basic_subscript` and bug01 `test_single_gather_plan_parses_correctly` from edge_cases (same scenario as gcp01); gem08 `test_gate_off_no_gather_op` and gem09 `test_gate_on_gather_op` from downstream (gate=0/gate=1 pair, now identical after flag removal); gstl13 `test_stl_gate_off`, gstl14 `test_stl_index_never_has_stl`, cmp06 `test_stl_sdsc_encoding_differs` from stl (no STL annotation injected, bare compare_mode). FVT rows GSTL-13, GSTL-14, GEM-08, GEM-09, GSTLCMP-06, GCP-02, GCP-12 removed. Total test IDs: 451 → 436. FVT spec rows: 479 → 472. |
| 1.6 | 2026-08-12 | Mahalakshmi Pathivada | Index dtype fix, new compiler tests, and duplicate removal. (1) **Index dtype**: changed all index tensor dtypes from int32 to int64 across 9 test files (320+ occurrences) to eliminate P0-B blocker (`InductorError: type conversion from torch.int32 to torch.int64` at `spyre_kernel.py:392`). `test_gather_dtypes.py` intentionally unchanged. Three int32 VALUE tensors preserved in `test_gather_edge_cases.py` (lines 163, 422, 1075). (2) **New GCP tests (GCP-17..GCP-23)**: Added 7 compiler-context tests to `test_gather_compiler.py` covering explicit `torch.gather(src, dim, index)` at dim=0/1/2 (LX_PLANNING=1), bfloat16 explicit gather, dim=1 fancy-index at SENCORES=4, dim=2 fancy-index (LX), and dim=1 explicit gather (LX). These are the first FVT compiler-context tests for `aten::gather.out` (distinct from `aten::index`). (3) **Duplicate removal**: removed 11 tests with identical scenarios across files: `test_stl_2d_with_scalar_div` (GSTL-06, removed — duplicated downstream `test_gather_scalar_div`); `test_float32_kv_cache_int32_index` renamed to `test_float32_kv_cache_access` (GPA-26, index dtype int32→int64); 3 tests from `test_gather_stl.py` (`test_stl_2d_with_scalar_div`, `test_stride_exact_gpt20b_shape`, `test_stride_exact_float32_kv`); 3 tests from `test_gather_edge_cases.py` (`test_ascending_indices`, `test_descending_indices`, `test_cor_gather_identity`); 5 tests from `test_gather_chained.py` (`test_chunked_kv_prefill`, `test_e2e_prefill_embed_lookup`, `test_e2e_batch_decode`, `test_e2e_rope_gather`, `test_e2e_beam_reorder`). FVT rows updated: GSTL-06 removed, GSTL-07..16 renumbered to GSTL-06..15. GCP section: +7 rows (GCP-17..GCP-23). Net test IDs: 466 → 472. Live records: 817 → 822. |
| 1.5 | 2026-08-12 | Mahalakshmi Pathivada | Compiler file cleanup. (1) Made `test_gather_compiler.py` compiled-only: removed class-level `@pytest.mark.parametrize("execution_mode", ["eager", "compiled"])`, removed all 12 `if execution_mode == "eager": pytest.skip(...)` blocks, hardcoded `"compiled"` in all 37 `compare_mode()` calls; 49 → 43 tests after further removal. (2) Removed 6 non-differentiating tests (`test_gather_op_spec_present`, `test_non_aligned_source`, `test_multi_stick_rows`, `test_large_tensor`, `test_sub_stick_width`, `test_4d_memory_layout`) — all used `lambda x, i: x[i]` with no env vars or structural assertions; indistinguishable from coverage in other files. FVT rows removed: GCP-01, GMEM-02, GMEM-03, GMEM-06, GMEM-11, GMEM-12 (pre-renumber IDs). (3) Renumbered all section IDs with gaps to consecutive: GSTL-13..16 (was 15..18), GEM-08..10 (was 10..12), GSTLCMP-06..09 (was 07..10), GCP-01..16 (was 03..19), GMEM-01..07 (was 01..10 with gaps). Test IDs: 436 → 430. FVT spec rows: 472 → 466. Live records: 872 → 817. |
| 1.2 | 2026-08-05 | Mahalakshmi Pathivada | Duplicate removal and gap closure. (1) Removed 2 true duplicates from test_gather_compiler.py: `test_gate_on_produces_result` (≡ test_gather_op_spec_present, gcp01) and `test_gate_off_cpu_fallback` (≡ test_gather_gate_off_no_op_spec, gcp02) — same shape/op/env in different class sections. (2) Added GAP-1 test: `test_take_along_dim_none_flat` in test_gather_functional.py — closes the only untested take_along_dim API form (flatten mode). FVT: GAPI-25. (3) Added GAP-2 tests: `test_negative_index_int64` and `test_negative_index_int64_downcast` in test_gather_edge_cases.py — validate PyTorch negative-index semantics (x[-1] → last row) and the int64→int32 downcast path for negative values. FVT: GEC-18, GEC-19. Total test IDs: 479 → 482. |
| 1.1 | 2026-08-03 | Mahalakshmi Pathivada | Gap-closure revision. (1) Corrected all "Implemented In" class names to match actual class names post-rename (TestGatherShapeRankAndIndexPatterns, TestGatherValueAndIndexDtypeCoverage, TestGatherSpyreTensorLayoutAnnotation, TestGatherFusedDownstreamOperations, TestGatherEmbeddingTableLookupAndPooling, TestGatherProductionInferencePatterns, TestGatherBoundaryNumericsAndAPIForms, TestGatherCompilerPassesMemoryAndDynamicShapes, TestGatherComposedChainsAndEndToEndPipelines). (2) Added Section 33 — SDPA with Paged KV Cache (GSDPA-01..20, 20 tests) for TestGatherPagedAttentionAndSDPA in test_gather_attention.py — previously completely unrepresented in FVT. (3) Added Section 34 — Non-Leading-Dim Gather (GNLD-01..07, 7 tests) for TestGatherNonLeadingDimEagerCompile dim=1/dim=2 patterns. (4) Extended Section 13 with GMDL-14..23 (10 new: 5 TopK MoE routing + 5 multi-layer KV patterns). (5) Extended Section 15 with GCP-15..19 (5 new: in-graph index computation via arange/topk/nonzero/where/argmax). (6) Added execution_mode parametrization note: each test ID now yields [eager] and [compiled] pytest records. Total: 437 → 479 test IDs; ~958 live records. |
