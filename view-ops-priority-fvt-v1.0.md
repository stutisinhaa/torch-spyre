# Functional Verification Test Plan — Priority View Operations (reshape, squeeze, unsqueeze, expand)

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Priority View Operations on torch-spyre: `reshape`, `squeeze`, `unsqueeze`, `expand` |
| **Version** | 1.0 |
| **Status** | Under Review |
| **Date** | 2025-07-11 |
| **Author** | Stuti Sinha |
| **Reviewers** | Kanaka Pathivada, Meena Soundararajan, Albin Joy |
| **Scope** | Inference-only compilation (`torch.compile`), static concrete dimensions (0D to 6D), eager CPU baseline vs compiled Spyre comparison |
| **Target Architecture** | Google Gemma-4 (`Gemma4ForConditionalGeneration`, `Gemma4TextModel`, `Gemma4VisionModel`, `Gemma4MoE`) |

---

## 1. Scope & Strategy

This document defines functional verification test cases for the four **priority view operations** — `reshape`, `squeeze`, `unsqueeze`, and `expand` — extracted from and inspired by the **Gemma-4-26B-A4B-it** architecture.

### Key Guidelines & Test Philosophy

1. **No Symbolic Dimensions**: All tests use concrete, static tensor shapes.
2. **Two Levels of Testing**:
   - **Module Level**: Whole Gemma-4 sub-module forward passes that rely on view ops internally. These mirror real model shapes from the 26B-A4B-it trace and validate compiled correctness of full view-op chains.
   - **Operation Level**: Direct standalone tests of each of the four ops in isolation, covering 0D–6D, all three floating-point dtypes, and a systematic range of aligned and unaligned stick scenarios.
3. **Four Priority Operators**: `reshape`, `squeeze`, `unsqueeze`, `expand`.
4. **Dimensionality Spectrum**: Explicit test coverage across tensor ranks **0D, 1D, 2D, 3D, 4D, 5D, and 6D**.
5. **Datatypes Covered**: `float16`, `bfloat16`, `float32`.
6. **Default Environment Variables Only**: All tests run at default env-var values. No SENCORES sweep, no LX_PLANNING variations.
7. **Test Classification**:
   - **`[GEMMA-SHAPE]`**: Shape taken directly from the Gemma-4-26B-A4B-it model trace.
   - **`[UNALIGNED-STICK]`**: Innermost dimension is not divisible by the per-dtype stick element count, or is smaller than one stick width.
   - **`[SIZE-1-DIM]`**: One or more tensor dimensions have size exactly 1 (the extreme unaligned case).
   - **`[COMPOSITE]`**: Multi-op chain test combining two or more view ops in sequence.

---

## 2. Stick Alignment Reference

The Spyre hardware stores tensors in 128-byte memory units called **sticks**. The **innermost (last) tensor dimension** maps to the stick dimension. The number of elements per stick varies by dtype:

| Dtype | Bytes per element | Elements per stick (128 B ÷ bytes/elem) |
|---|---|---|
| `float16` | 2 B | **64** |
| `bfloat16` | 2 B | **64** |
| `float32` | 4 B | **32** |

A tensor is **stick-aligned** when its innermost dimension size is divisible by the elements-per-stick number. It is **unaligned** in two ways:

- **Below one stick**: innermost dim < 64 (fp16/bf16) or < 32 (fp32). Examples: 1, 7, 13, 30.
- **Not divisible**: innermost dim is larger than one stick width but not a clean multiple. Examples: 65, 70, 72, 127, 33 (for fp16/bf16 where threshold is 64).

**Size-1 innermost dimensions** are the extreme unaligned case: 1 element fills one stick with 63 empty slots (for fp16/bf16). These arise naturally from `unsqueeze(-1)`, `reshape(..., 1)`, or `squeeze` on all-but-one dimensions.

To produce unaligned stick scenarios in practice: simply choose tensor shapes where the last dimension is not a multiple of 64 (fp16/bf16) or 32 (fp32). No special flags are needed — the hardware sees the unaligned boundary automatically.

**Aligned examples** (fp16/bf16): 64, 128, 256, 704, 1152, 2816

**Unaligned examples** (fp16/bf16): 1, 7, 13, 30, 33, 63, 65, 70, **72** *(Gemma vision head dim)*, 127

---

## 3. Gemma-4 Modules Under Test (Module Level)

The module-level tests target the following Gemma-4 sub-modules where the four priority ops are central:

| # | Module | Priority Ops Used | Key Shapes |
|---|---|---|---|
| 1 | `Gemma4TextAttention` | `reshape` (QKV split/merge), `unsqueeze`+`expand` (GQA KV broadcast) | `[1, 32, 4096]` → `[1, 32, 16, 256]` |
| 2 | `Gemma4TextMoE` | `reshape` (token flatten/unflatten, index flatten) | `[4, 32, 2816]` ↔ `[128, 2816]`, `[32, 8]` → `[256]` |
| 3 | `Gemma4RMSNorm` | `unsqueeze` (weight broadcast) | `[2816]` → `[1, 1, 2816]` over `[B, S, 2816]` |
| 4 | `Gemma4RotaryEmbedding` | `unsqueeze` (freq table head broadcast) | `[1, 32, 256]` → `[1, 32, 1, 256]` |
| 5 | `Gemma4VisionPatchEmbedder` | `reshape` (spatial patch flatten) | `[1, 1152, 16, 16]` → `[1, 256, 1152]` |
| 6 | `Gemma4VisionEncoderLayer` | `reshape` (QKV split, D=72 unaligned) | `[1, 280, 1152]` → `[1, 280, 16, 72]` |
| 7 | `Gemma4VisionProjector` | `reshape` (spatial pooling step) | `[1, 280, 1152]` → `[1, 70, 4, 1152]` |

---

## 4. Test Environment

| Variable | Value Used in This FVT |
|---|---|
| `SENCORES` | **1** (default) |
| `LX_PLANNING` | **0** (default) |
| `CO_OPTIMIZING_LX_PLANNING` | **0** (default) |
| `GLOBAL_STICK_OPTIMIZER` | **1** (default) |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | **1** (default) |

All test cases use default environment variable values. Multi-core and environment-variable sweep testing is out of scope for this FVT.

---

## 5. Table of Contents

| Section | Category | Test IDs | Count |
|---|---|---|---|
| **Section 1** | Module-Level End-to-End Tests | `MOD-01` .. `MOD-13` | 13 |
| **Section 2** | `reshape` — Operation Level | `RSH-01` .. `RSH-28` | 28 |
| **Section 3** | `squeeze` — Operation Level | `SQZ-01` .. `SQZ-26` | 26 |
| **Section 4** | `unsqueeze` — Operation Level | `UNS-01` .. `UNS-24` | 24 |
| **Section 5** | `expand` — Operation Level | `EXP-01` .. `EXP-28` | 28 |
| **Section 6** | Composite View-Op Chains | `CMP-01` .. `CMP-08` | 8 |
| — | **Total** | | **127** |

---

## Section 1 — Module-Level End-to-End Tests

Tests full forward execution of Gemma-4 sub-module forward passes that contain view ops. All tests use `bfloat16` (Gemma-4 production dtype) at default configuration shapes.

**Pass Criteria for all Module tests**: `compare_with_cpu`, `atol=1e-2`

| ID | Test Name | Module Under Test | Classification | View Ops Exercised | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus |
|---|---|---|---|---|---|---|---|
| `MOD-01` | `test_module_qkv_view_split_sliding` | `Gemma4TextAttention` | `[GEMMA-SHAPE]` | `reshape` / `view` | `[1, 32, 4096]` (`bfloat16`) | `[1, 32, 16, 256]` | 3D→4D QKV split for sliding attention (H=16, D=256) |
| `MOD-02` | `test_module_qkv_view_split_full` | `Gemma4TextAttention` | `[GEMMA-SHAPE]` | `reshape` / `view` | `[1, 32, 8192]` (`bfloat16`) | `[1, 32, 16, 512]` | 3D→4D QKV split for full attention (H=16, D=512) |
| `MOD-03` | `test_module_qkv_flatten_sliding` | `Gemma4TextAttention` | `[GEMMA-SHAPE]` | `reshape` | `[1, 32, 16, 256]` (`bfloat16`) | `[1, 32, 4096]` | 4D→3D head merge after attention output |
| `MOD-04` | `test_module_qkv_flatten_decode` | `Gemma4TextAttention` | `[GEMMA-SHAPE]` | `reshape` | `[1, 1, 16, 256]` (`bfloat16`) | `[1, 1, 4096]` | Single-token decode attention output flatten |
| `MOD-05` | `test_module_gqa_unsqueeze_expand_2x` | `Gemma4TextAttention` | `[GEMMA-SHAPE]` | `unsqueeze`, `expand`, `reshape` | `[1, 8, 32, 256]` (`bfloat16`) | `[1, 16, 32, 256]` | GQA KV broadcast 8→16 heads via `unsqueeze(2).expand(...).reshape(...)` |
| `MOD-06` | `test_module_gqa_unsqueeze_expand_8x` | `Gemma4TextAttention` | `[GEMMA-SHAPE]` | `unsqueeze`, `expand`, `reshape` | `[1, 2, 32, 512]` (`bfloat16`) | `[1, 16, 32, 512]` | GQA KV broadcast 2→16 heads (full attention, 8x expansion) |
| `MOD-07` | `test_module_moe_token_flatten` | `Gemma4TextMoE` | `[GEMMA-SHAPE]` | `reshape` | `[4, 32, 2816]` (`bfloat16`) | `[128, 2816]` | `[B, S, H]`→`[B*S, H]` token flatten for expert routing dispatch |
| `MOD-08` | `test_module_moe_token_unflatten` | `Gemma4TextMoE` | `[GEMMA-SHAPE]` | `reshape` | `[128, 2816]` (`bfloat16`) | `[4, 32, 2816]` | `[B*S, H]`→`[B, S, H]` reassembly after expert aggregation |
| `MOD-09` | `test_module_moe_index_flatten` | `Gemma4TextMoE` | `[GEMMA-SHAPE]` | `reshape` | `[32, 8]` (`int64`) | `[256]` | 2D top-k index array flatten to 1D dispatch array |
| `MOD-10` | `test_module_rmsnorm_weight_broadcast` | `Gemma4RMSNorm` | `[GEMMA-SHAPE]` | `unsqueeze` | `weight: [2816]`, `x: [4, 128, 2816]` (`bfloat16`) | `[4, 128, 2816]` | 1D weight broadcast over `[B, S, D]` via `unsqueeze(0).unsqueeze(0)` |
| `MOD-11` | `test_module_rope_freq_unsqueeze` | `Gemma4RotaryEmbedding` | `[GEMMA-SHAPE]` | `unsqueeze` | `[1, 32, 256]` (`bfloat16`) | `[1, 32, 1, 256]` | Frequency table `unsqueeze(2)` to broadcast across 16 attention heads |
| `MOD-12` | `test_module_vision_patch_flatten` | `Gemma4VisionPatchEmbedder` | `[GEMMA-SHAPE]` | `reshape` | `[1, 1152, 16, 16]` (`bfloat16`) | `[1, 256, 1152]` | 2D spatial grid `[B, D, H_p, W_p]`→`[B, N, D]` via `view + transpose` |
| `MOD-13` | `test_module_vision_qkv_split_d72` | `Gemma4VisionEncoderLayer` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `reshape` | `[1, 280, 1152]` (`bfloat16`) | `[1, 280, 16, 72]` | QKV split with D=72 (unaligned: 72 % 64 ≠ 0, non-power-of-2 vision head dim) |

---

## Section 2 — `reshape` Operation Level

Validates `reshape` and `view` across 0D–6D, all three dtypes, and the full aligned/unaligned/size-1 spectrum.

**Pass Criteria for all Section 2 tests**: `compare_with_cpu`, `atol=0.0` (reshape is a layout-only op, no numeric error expected)

| ID | Test Name | Classification | Input Dims & Dtype | Target Shape | Test Pattern | Stick Alignment Notes |
|---|---|---|---|---|---|---|
| `RSH-01` | `test_reshape_0d_scalar_to_1d` | `[SIZE-1-DIM]` | `[]` (all dtypes) | `[1]` | `x.reshape(1)` | 0D scalar; result innermost=1 (ultra-unaligned) |
| `RSH-02` | `test_reshape_0d_scalar_to_2d` | `[SIZE-1-DIM]` | `[]` (all dtypes) | `[1, 1]` | `x.reshape(1, 1)` | All dims = 1, innermost=1 (unaligned) |
| `RSH-03` | `test_reshape_1d_aligned_to_2d` | — | `[256]` (all dtypes) | `[16, 16]` | `x.reshape(16, 16)` | Inner 16: unaligned for fp16/bf16 (16 < 64) |
| `RSH-04` | `test_reshape_1d_gemma_hidden_aligned` | `[GEMMA-SHAPE]` | `[2816]` (all dtypes) | `[128, 22]` | `x.reshape(128, 22)` | Inner 22: unaligned (22 < 64); Gemma hidden dim 2816 |
| `RSH-05` | `test_reshape_1d_unaligned_to_2d` | `[UNALIGNED-STICK]` | `[63]` (all dtypes) | `[9, 7]` | `x.reshape(9, 7)` | Input and inner both unaligned (< 64) |
| `RSH-06` | `test_reshape_1d_size1_to_3d` | `[SIZE-1-DIM]` | `[1]` (all dtypes) | `[1, 1, 1]` | `x.reshape(1, 1, 1)` | All dims = 1 |
| `RSH-07` | `test_reshape_2d_aligned_flatten` | — | `[64, 64]` (all dtypes) | `[4096]` | `x.reshape(-1)` | Inner 4096: aligned (4096 % 64 = 0) |
| `RSH-08` | `test_reshape_2d_unaligned_inner` | `[UNALIGNED-STICK]` | `[32, 33]` (all dtypes) | `[1056]` | `x.reshape(-1)` | Inner 33: unaligned (33 % 64 ≠ 0, 33 < 64) |
| `RSH-09` | `test_reshape_2d_size1_col` | `[SIZE-1-DIM]` | `[128, 1]` (all dtypes) | `[128]` | `x.reshape(128)` | Input innermost = 1 (ultra-unaligned) |
| `RSH-10` | `test_reshape_2d_to_3d_split_aligned` | — | `[4, 2048]` (all dtypes) | `[4, 8, 256]` | `x.reshape(4, 8, 256)` | Inner 256: aligned (256 % 64 = 0) |
| `RSH-11` | `test_reshape_2d_to_3d_split_unaligned` | `[UNALIGNED-STICK]` | `[4, 70]` (all dtypes) | `[4, 2, 35]` | `x.reshape(4, 2, 35)` | Inner 35: unaligned (35 % 64 ≠ 0, 35 < 64) |
| `RSH-12` | `test_reshape_3d_flatten_last_two_aligned` | `[GEMMA-SHAPE]` | `[2, 16, 256]` (all dtypes) | `[2, 4096]` | `x.reshape(2, -1)` | Inner 4096: aligned |
| `RSH-13` | `test_reshape_3d_unaligned_seq` | `[GEMMA-SHAPE]` | `[3, 71, 2048]` (all dtypes) | `[3, 71, 8, 256]` | `x.reshape(3, 71, 8, 256)` | S=71 non-power-of-2; inner 256 aligned |
| `RSH-14` | `test_reshape_3d_unaligned_inner_d72` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[1, 280, 1152]` (all dtypes) | `[1, 280, 16, 72]` | `x.reshape(1, 280, 16, 72)` | D=72 unaligned (72 % 64 = 8); Gemma vision head dim |
| `RSH-15` | `test_reshape_3d_unaligned_inner_small` | `[UNALIGNED-STICK]` | `[2, 8, 13]` (all dtypes) | `[16, 13]` | `x.reshape(16, 13)` | Inner 13: < 64 (unaligned) |
| `RSH-16` | `test_reshape_3d_size1_trailing` | `[SIZE-1-DIM]` | `[4, 32, 1]` (all dtypes) | `[4, 32]` | `x.reshape(4, 32)` | Input innermost = 1 |
| `RSH-17` | `test_reshape_4d_head_flatten_aligned` | `[GEMMA-SHAPE]` | `[1, 16, 32, 256]` (all dtypes) | `[1, 32, 4096]` | `x.reshape(1, 32, 16 * 256)` | Inner 4096: aligned; post-attention head merge |
| `RSH-18` | `test_reshape_4d_head_flatten_d72_unaligned` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[1, 16, 280, 72]` (all dtypes) | `[1, 280, 1152]` | `x.reshape(1, 280, 16 * 72)` | Inner 1152: aligned, but input last dim was 72 (unaligned) |
| `RSH-19` | `test_reshape_4d_unaligned_inner_33` | `[UNALIGNED-STICK]` | `[2, 4, 8, 33]` (all dtypes) | `[2, 32, 33]` | `x.reshape(2, 32, 33)` | Inner 33: unaligned |
| `RSH-20` | `test_reshape_4d_size1_dims` | `[SIZE-1-DIM]` | `[1, 8, 1, 256]` (all dtypes) | `[8, 256]` | `x.reshape(8, 256)` | Two size-1 dims removed |
| `RSH-21` | `test_reshape_5d_flatten_aligned` | — | `[2, 4, 8, 16, 32]` (all dtypes) | `[2, 4, 4096]` | `x.reshape(2, 4, -1)` | fp32 inner 32: aligned; fp16/bf16 inner: unaligned (32 < 64) |
| `RSH-22` | `test_reshape_5d_unaligned_inner_7` | `[UNALIGNED-STICK]` | `[1, 2, 4, 8, 7]` (all dtypes) | `[1, 8, 56]` | `x.reshape(1, 8, 56)` | Inner 7: far below stick width (ultra-unaligned) |
| `RSH-23` | `test_reshape_5d_to_6d` | — | `[1, 2, 4, 8, 256]` (all dtypes) | `[1, 2, 4, 8, 4, 64]` | `x.reshape(1, 2, 4, 8, 4, 64)` | Inner 64: aligned |
| `RSH-24` | `test_reshape_6d_collapse` | — | `[1, 2, 4, 8, 4, 64]` (all dtypes) | `[1, 2, 4, 2048]` | `x.reshape(1, 2, 4, -1)` | Inner 2048: aligned |
| `RSH-25` | `test_reshape_6d_unaligned_inner` | `[UNALIGNED-STICK]` | `[1, 2, 4, 8, 4, 7]` (all dtypes) | `[1, 8, 224]` | `x.reshape(1, 8, 224)` | Input inner 7: unaligned; output inner 224: 224 % 64 = 32, unaligned |
| `RSH-26` | `test_reshape_6d_size1_trailing` | `[SIZE-1-DIM]` | `[2, 3, 4, 5, 6, 1]` (all dtypes) | `[2, 3, 4, 5, 6]` | `x.reshape(2, 3, 4, 5, 6)` | Trailing size-1 removed |
| `RSH-27` | `test_reshape_inferred_dim` | `[GEMMA-SHAPE]` | `[2, 32, 16, 256]` (all dtypes) | `[2, -1, 256]` | `x.reshape(2, -1, 256)` | `-1` inference (result inner 256: aligned) |
| `RSH-28` | `test_reshape_inferred_dim_unaligned` | `[UNALIGNED-STICK]` | `[3, 7, 6]` (all dtypes) | `[3, -1]` | `x.reshape(3, -1)` | `-1` inference; result inner 42: unaligned (42 % 64 ≠ 0) |

---

## Section 3 — `squeeze` Operation Level

Validates `squeeze` across 0D–6D inputs, all three dtypes, singleton and non-singleton dimensions, and unaligned stick innermost sizes.

**Pass Criteria for all Section 3 tests**: `compare_with_cpu`, `atol=0.0`

| ID | Test Name | Classification | Input Dims & Dtype | Op Pattern | Expected Output Dims | Stick Alignment Notes |
|---|---|---|---|---|---|---|
| `SQZ-01` | `test_squeeze_1d_size1_to_scalar` | `[SIZE-1-DIM]` | `[1]` (all dtypes) | `x.squeeze(0)` | `[]` (scalar) | 0D output |
| `SQZ-02` | `test_squeeze_1d_noop_aligned` | — | `[256]` (all dtypes) | `x.squeeze(0)` | `[256]` | No-op: dim-0 size=256, not 1; inner 256 aligned |
| `SQZ-03` | `test_squeeze_1d_noop_unaligned` | `[UNALIGNED-STICK]` | `[63]` (all dtypes) | `x.squeeze(0)` | `[63]` | No-op: dim-0 size=63; inner 63 < 64 (unaligned) |
| `SQZ-04` | `test_squeeze_2d_leading_aligned` | — | `[1, 256]` (all dtypes) | `x.squeeze(0)` | `[256]` | Remove batch; inner 256 aligned |
| `SQZ-05` | `test_squeeze_2d_leading_unaligned` | `[UNALIGNED-STICK]` | `[1, 33]` (all dtypes) | `x.squeeze(0)` | `[33]` | Remove batch; inner 33 < 64 (unaligned) |
| `SQZ-06` | `test_squeeze_2d_trailing_size1` | `[SIZE-1-DIM]` | `[128, 1]` (all dtypes) | `x.squeeze(-1)` | `[128]` | Trailing size-1 removed; new inner 128 aligned |
| `SQZ-07` | `test_squeeze_2d_all_singletons` | `[SIZE-1-DIM]` | `[1, 1]` (all dtypes) | `x.squeeze()` | `[]` (scalar) | All dims = 1; 0D result |
| `SQZ-08` | `test_squeeze_3d_batch_aligned` | `[GEMMA-SHAPE]` | `[1, 32, 2816]` (all dtypes) | `x.squeeze(0)` | `[32, 2816]` | Remove batch; inner 2816 aligned |
| `SQZ-09` | `test_squeeze_3d_seq_decode_aligned` | `[GEMMA-SHAPE]` | `[2, 1, 256]` (all dtypes) | `x.squeeze(1)` | `[2, 256]` | Remove seq=1 decode step; inner 256 aligned |
| `SQZ-10` | `test_squeeze_3d_seq_decode_unaligned` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[2, 1, 70]` (all dtypes) | `x.squeeze(1)` | `[2, 70]` | Remove seq=1 decode step; inner 70 unaligned (70 % 64 = 6) |
| `SQZ-11` | `test_squeeze_3d_noop_non_singleton` | — | `[1, 32, 2816]` (all dtypes) | `x.squeeze(1)` | `[1, 32, 2816]` | No-op: dim-1 size=32 ≠ 1 |
| `SQZ-12` | `test_squeeze_4d_decode_seq_aligned` | `[GEMMA-SHAPE]` | `[1, 1, 16, 256]` (all dtypes) | `x.squeeze(1)` | `[1, 16, 256]` | Single-token decode squeeze; inner 256 aligned |
| `SQZ-13` | `test_squeeze_4d_decode_seq_unaligned` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[1, 1, 16, 72]` (all dtypes) | `x.squeeze(1)` | `[1, 16, 72]` | Decode squeeze; D=72 unaligned (72 % 64 = 8) |
| `SQZ-14` | `test_squeeze_4d_all_singletons_aligned` | `[SIZE-1-DIM]` | `[1, 1, 32, 1]` (all dtypes) | `x.squeeze()` | `[32]` | All size-1 dims removed; result 1D |
| `SQZ-15` | `test_squeeze_4d_all_singletons_unaligned` | `[SIZE-1-DIM]` `[UNALIGNED-STICK]` | `[1, 1, 63, 1]` (all dtypes) | `x.squeeze()` | `[63]` | All size-1 dims removed; result inner 63 unaligned |
| `SQZ-16` | `test_squeeze_4d_tuple_dims` | — | `[1, 16, 1, 256]` (all dtypes) | `x.squeeze((0, 2))` | `[16, 256]` | Multi-dim tuple squeeze; inner 256 aligned |
| `SQZ-17` | `test_squeeze_4d_tuple_dims_unaligned` | `[UNALIGNED-STICK]` | `[1, 16, 1, 72]` (all dtypes) | `x.squeeze((0, 2))` | `[16, 72]` | Multi-dim tuple squeeze; inner 72 unaligned |
| `SQZ-18` | `test_squeeze_4d_negative_index` | — | `[32, 8, 64, 1]` (all dtypes) | `x.squeeze(-1)` | `[32, 8, 64]` | Trailing size-1; new inner 64 aligned |
| `SQZ-19` | `test_squeeze_4d_negative_index_unaligned` | `[UNALIGNED-STICK]` | `[32, 8, 30, 1]` (all dtypes) | `x.squeeze(-1)` | `[32, 8, 30]` | Trailing size-1; new inner 30 < 64 (unaligned) |
| `SQZ-20` | `test_squeeze_5d_all_singletons_aligned` | `[SIZE-1-DIM]` | `[1, 1, 32, 1, 256]` (all dtypes) | `x.squeeze()` | `[32, 256]` | Remove all size-1; inner 256 aligned |
| `SQZ-21` | `test_squeeze_5d_all_singletons_unaligned` | `[SIZE-1-DIM]` `[UNALIGNED-STICK]` | `[1, 1, 32, 1, 30]` (all dtypes) | `x.squeeze()` | `[32, 30]` | Remove all size-1; inner 30 < 64 (unaligned) |
| `SQZ-22` | `test_squeeze_6d_leading_aligned` | — | `[1, 2, 4, 1, 8, 256]` (all dtypes) | `x.squeeze(0)` | `[2, 4, 1, 8, 256]` | Remove leading batch; inner 256 aligned |
| `SQZ-23` | `test_squeeze_6d_leading_unaligned` | `[UNALIGNED-STICK]` | `[1, 2, 4, 1, 8, 13]` (all dtypes) | `x.squeeze(0)` | `[2, 4, 1, 8, 13]` | Remove leading batch; inner 13 < 64 (unaligned) |
| `SQZ-24` | `test_squeeze_6d_middle_singleton_aligned` | — | `[1, 2, 1, 4, 8, 256]` (all dtypes) | `x.squeeze(2)` | `[1, 2, 4, 8, 256]` | Remove middle size-1; inner 256 aligned |
| `SQZ-25` | `test_squeeze_6d_middle_singleton_unaligned` | `[UNALIGNED-STICK]` | `[1, 2, 1, 4, 8, 70]` (all dtypes) | `x.squeeze(2)` | `[1, 2, 4, 8, 70]` | Remove middle size-1; inner 70 unaligned (70 % 64 = 6) |
| `SQZ-26` | `test_squeeze_6d_noop_non_singleton` | — | `[2, 3, 4, 5, 6, 64]` (all dtypes) | `x.squeeze(0)` | `[2, 3, 4, 5, 6, 64]` | No-op: dim-0 size=2 ≠ 1; inner 64 aligned |

---

## Section 4 — `unsqueeze` Operation Level

Validates `unsqueeze` across 0D–5D inputs (producing up to 6D results), all three dtypes, positive/negative indices, and unaligned stick scenarios.

**Pass Criteria for all Section 4 tests**: `compare_with_cpu`, `atol=0.0`

| ID | Test Name | Classification | Input Dims & Dtype | Op Pattern | Expected Output Dims | Stick Alignment Notes |
|---|---|---|---|---|---|---|
| `UNS-01` | `test_unsqueeze_scalar_to_1d` | `[SIZE-1-DIM]` | `[]` (all dtypes) | `x.unsqueeze(0)` | `[1]` | 0D → 1D; result innermost = 1 (ultra-unaligned) |
| `UNS-02` | `test_unsqueeze_1d_front_aligned` | — | `[256]` (all dtypes) | `x.unsqueeze(0)` | `[1, 256]` | Leading dim insert; inner 256 aligned |
| `UNS-03` | `test_unsqueeze_1d_front_unaligned` | `[UNALIGNED-STICK]` | `[70]` (all dtypes) | `x.unsqueeze(0)` | `[1, 70]` | Leading dim insert; inner 70 unaligned (70 % 64 = 6) |
| `UNS-04` | `test_unsqueeze_1d_size1_input` | `[SIZE-1-DIM]` | `[1]` (all dtypes) | `x.unsqueeze(0)` | `[1, 1]` | Size-1 input, size-1 result inner |
| `UNS-05` | `test_unsqueeze_2d_dim0_aligned` | `[GEMMA-SHAPE]` | `[32, 2816]` (all dtypes) | `x.unsqueeze(0)` | `[1, 32, 2816]` | Batch insert; inner 2816 aligned |
| `UNS-06` | `test_unsqueeze_2d_dim0_unaligned` | `[UNALIGNED-STICK]` | `[32, 13]` (all dtypes) | `x.unsqueeze(0)` | `[1, 32, 13]` | Batch insert; inner 13 < 64 (unaligned) |
| `UNS-07` | `test_unsqueeze_2d_trailing_size1` | `[SIZE-1-DIM]` | `[32, 256]` (all dtypes) | `x.unsqueeze(-1)` | `[32, 256, 1]` | New trailing innermost = 1 (ultra-unaligned) |
| `UNS-08` | `test_unsqueeze_2d_attn_mask_double_aligned` | `[GEMMA-SHAPE]` | `[1, 32]` (all dtypes) | `x.unsqueeze(1).unsqueeze(2)` | `[1, 1, 1, 32]` | Attention mask broadcast alignment; inner 32 unaligned (< 64) |
| `UNS-09` | `test_unsqueeze_3d_dim2_rope_aligned` | `[GEMMA-SHAPE]` | `[1, 32, 256]` (all dtypes) | `x.unsqueeze(2)` | `[1, 32, 1, 256]` | RoPE freq broadcast across heads; inner 256 aligned |
| `UNS-10` | `test_unsqueeze_3d_dim2_rope_unaligned` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[1, 32, 70]` (all dtypes) | `x.unsqueeze(2)` | `[1, 32, 1, 70]` | RoPE freq broadcast; inner 70 unaligned (70 % 64 = 6) |
| `UNS-11` | `test_unsqueeze_3d_dim2_rope_d72` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[1, 32, 72]` (all dtypes) | `x.unsqueeze(2)` | `[1, 32, 1, 72]` | RoPE broadcast with D=72 unaligned vision head dim |
| `UNS-12` | `test_unsqueeze_3d_size1_trailing` | `[SIZE-1-DIM]` | `[4, 8, 1]` (all dtypes) | `x.unsqueeze(1)` | `[4, 1, 8, 1]` | Insert middle; trailing = 1 stays innermost (ultra-unaligned) |
| `UNS-13` | `test_unsqueeze_4d_dim2_gqa_aligned` | `[GEMMA-SHAPE]` | `[2, 8, 32, 256]` (all dtypes) | `x.unsqueeze(2)` | `[2, 8, 1, 32, 256]` | GQA head group dim insert before expand; inner 256 aligned |
| `UNS-14` | `test_unsqueeze_4d_dim2_gqa_unaligned` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[2, 8, 32, 72]` (all dtypes) | `x.unsqueeze(2)` | `[2, 8, 1, 32, 72]` | GQA insert; D=72 unaligned |
| `UNS-15` | `test_unsqueeze_4d_trailing_size1` | `[SIZE-1-DIM]` | `[1, 16, 32, 256]` (all dtypes) | `x.unsqueeze(-1)` | `[1, 16, 32, 256, 1]` | New trailing innermost = 1 (ultra-unaligned) |
| `UNS-16` | `test_unsqueeze_4d_negative_index` | — | `[1, 16, 32, 256]` (all dtypes) | `x.unsqueeze(-1)` | `[1, 16, 32, 256, 1]` | Same as UNS-15, confirming negative index handling |
| `UNS-17` | `test_unsqueeze_5d_to_6d_dim3_aligned` | — | `[1, 2, 4, 8, 16]` (all dtypes) | `x.unsqueeze(3)` | `[1, 2, 4, 1, 8, 16]` | 5D→6D; inner 16 < 64 (unaligned for fp16/bf16) |
| `UNS-18` | `test_unsqueeze_5d_to_6d_dim3_unaligned` | `[UNALIGNED-STICK]` | `[1, 2, 4, 8, 13]` (all dtypes) | `x.unsqueeze(3)` | `[1, 2, 4, 1, 8, 13]` | 5D→6D; inner 13 < 64 (unaligned) |
| `UNS-19` | `test_unsqueeze_dual_axis_aligned` | `[GEMMA-SHAPE]` | `[32, 256]` (all dtypes) | `x.unsqueeze(0).unsqueeze(3)` | `[1, 32, 256, 1]` | Dual unsqueeze; new inner = 1 (ultra-unaligned) |
| `UNS-20` | `test_unsqueeze_dual_axis_unaligned` | `[UNALIGNED-STICK]` | `[32, 70]` (all dtypes) | `x.unsqueeze(0).unsqueeze(3)` | `[1, 32, 70, 1]` | Dual unsqueeze; 70 unaligned, then 1 as new inner |
| `UNS-21` | `test_unsqueeze_3d_leading_batch_insert` | — | `[32, 16, 256]` (all dtypes) | `x.unsqueeze(0)` | `[1, 32, 16, 256]` | Batch insert at front; inner 256 aligned |
| `UNS-22` | `test_unsqueeze_2d_mask_broadcast_unaligned` | `[UNALIGNED-STICK]` | `[1, 33]` (all dtypes) | `x.unsqueeze(1).unsqueeze(2)` | `[1, 1, 1, 33]` | Attention mask alignment with unaligned seq (33 < 64) |
| `UNS-23` | `test_unsqueeze_1d_gemma_hidden_front` | `[GEMMA-SHAPE]` | `[2816]` (all dtypes) | `x.unsqueeze(0)` | `[1, 2816]` | Gemma hidden dim; inner 2816 aligned |
| `UNS-24` | `test_unsqueeze_4d_dim0_batch_insert` | `[GEMMA-SHAPE]` | `[16, 32, 256]` (all dtypes) | `x.unsqueeze(0)` | `[1, 16, 32, 256]` | Batch insert on 3D attention state; inner 256 aligned |

---

## Section 5 — `expand` Operation Level

Validates `expand` across 0D–5D inputs, all three dtypes, both aligned and unaligned stick scenarios, size-1 source dimensions, and `-1` (no-change) dimension markers.

**Pass Criteria for all Section 5 tests**: `compare_with_cpu`, `atol=0.0`

| ID | Test Name | Classification | Input Dims & Dtype | Op Pattern | Expected Output Dims | Stick Alignment Notes |
|---|---|---|---|---|---|---|
| `EXP-01` | `test_expand_scalar_to_4d_aligned` | `[SIZE-1-DIM]` | `[]` (all dtypes) | `x.expand(1, 16, 32, 256)` | `[1, 16, 32, 256]` | 0D scalar → 4D; inner 256 aligned |
| `EXP-02` | `test_expand_scalar_to_4d_unaligned` | `[SIZE-1-DIM]` `[UNALIGNED-STICK]` | `[]` (all dtypes) | `x.expand(1, 16, 32, 70)` | `[1, 16, 32, 70]` | 0D scalar → 4D; inner 70 unaligned (70 % 64 = 6) |
| `EXP-03` | `test_expand_1d_bias_to_3d_aligned` | `[GEMMA-SHAPE]` | `[2816]` (all dtypes) | `x.expand(2, 32, 2816)` | `[2, 32, 2816]` | 1D bias broadcast; inner 2816 aligned |
| `EXP-04` | `test_expand_1d_bias_to_3d_unaligned` | `[UNALIGNED-STICK]` | `[70]` (all dtypes) | `x.expand(2, 32, 70)` | `[2, 32, 70]` | 1D bias broadcast; inner 70 unaligned |
| `EXP-05` | `test_expand_1d_size1_to_3d_unaligned` | `[SIZE-1-DIM]` | `[1]` (all dtypes) | `x.expand(2, 32, 1)` | `[2, 32, 1]` | Size-1 vector broadcast; result inner = 1 (ultra-unaligned) |
| `EXP-06` | `test_expand_2d_batch_aligned` | — | `[1, 2816]` (all dtypes) | `x.expand(4, 2816)` | `[4, 2816]` | Batch broadcast; inner 2816 aligned |
| `EXP-07` | `test_expand_2d_batch_unaligned` | `[UNALIGNED-STICK]` | `[1, 33]` (all dtypes) | `x.expand(4, 33)` | `[4, 33]` | Batch broadcast; inner 33 < 64 (unaligned) |
| `EXP-08` | `test_expand_2d_size1_inner_to_aligned` | `[SIZE-1-DIM]` | `[32, 1]` (all dtypes) | `x.expand(32, 2816)` | `[32, 2816]` | Trailing size-1 expanded to aligned 2816 |
| `EXP-09` | `test_expand_3d_cls_token_aligned` | `[GEMMA-SHAPE]` | `[1, 1, 1152]` (all dtypes) | `x.expand(2, 1, 1152)` | `[2, 1, 1152]` | Vision CLS token broadcast; inner 1152 aligned |
| `EXP-10` | `test_expand_3d_cls_token_unaligned` | `[UNALIGNED-STICK]` | `[1, 1, 70]` (all dtypes) | `x.expand(2, 1, 70)` | `[2, 1, 70]` | CLS token broadcast; inner 70 unaligned, middle size-1 preserved |
| `EXP-11` | `test_expand_3d_minus_one_dims` | — | `[2, 1, 256]` (all dtypes) | `x.expand(-1, 16, -1)` | `[2, 16, 256]` | `-1` preserves existing dims; head broadcast only; inner 256 aligned |
| `EXP-12` | `test_expand_3d_minus_one_unaligned` | `[UNALIGNED-STICK]` | `[2, 1, 72]` (all dtypes) | `x.expand(-1, 16, -1)` | `[2, 16, 72]` | `-1` dims; inner 72 unaligned |
| `EXP-13` | `test_expand_4d_gqa_kv_2x_aligned` | `[GEMMA-SHAPE]` | `[1, 8, 1, 256]` (all dtypes) | `x.expand(1, 8, 2, 256)` | `[1, 8, 2, 256]` | GQA 2x head broadcast (Config A); inner 256 aligned |
| `EXP-14` | `test_expand_4d_gqa_kv_8x_aligned` | `[GEMMA-SHAPE]` | `[1, 2, 1, 512]` (all dtypes) | `x.expand(1, 2, 8, 512)` | `[1, 2, 8, 512]` | GQA 8x head broadcast (Config D); inner 512 aligned |
| `EXP-15` | `test_expand_4d_gqa_kv_unaligned` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[1, 2, 1, 72]` (all dtypes) | `x.expand(1, 2, 8, 72)` | `[1, 2, 8, 72]` | GQA broadcast; D=72 unaligned (vision heads) |
| `EXP-16` | `test_expand_4d_attn_mask_aligned` | `[GEMMA-SHAPE]` | `[1, 1, 32, 32]` (all dtypes) | `x.expand(4, 16, 32, 32)` | `[4, 16, 32, 32]` | Attention mask batch+head broadcast; inner 32 < 64 (unaligned) |
| `EXP-17` | `test_expand_4d_attn_mask_unaligned_seq` | `[UNALIGNED-STICK]` | `[1, 1, 33, 33]` (all dtypes) | `x.expand(2, 8, 33, 33)` | `[2, 8, 33, 33]` | Mask broadcast with S=33 (unaligned) |
| `EXP-18` | `test_expand_4d_rope_freqs_aligned` | `[GEMMA-SHAPE]` | `[1, 32, 1, 256]` (all dtypes) | `x.expand(1, 32, 16, 256)` | `[1, 32, 16, 256]` | RoPE freq broadcast over 16 heads; inner 256 aligned |
| `EXP-19` | `test_expand_4d_rope_freqs_unaligned` | `[GEMMA-SHAPE]` `[UNALIGNED-STICK]` | `[1, 32, 1, 72]` (all dtypes) | `x.expand(1, 32, 16, 72)` | `[1, 32, 16, 72]` | RoPE freq broadcast; D=72 unaligned |
| `EXP-20` | `test_expand_4d_size1_trailing_to_aligned` | `[SIZE-1-DIM]` | `[32, 8, 4, 1]` (all dtypes) | `x.expand(32, 8, 4, 2816)` | `[32, 8, 4, 2816]` | Trailing size-1 expanded to aligned 2816 |
| `EXP-21` | `test_expand_4d_odd_batch_heads` | `[UNALIGNED-STICK]` | `[1, 1, 33, 256]` (all dtypes) | `x.expand(3, 7, 33, 256)` | `[3, 7, 33, 256]` | Non-power-of-2 B=3, H=7, S=33 (unaligned); inner 256 aligned |
| `EXP-22` | `test_expand_4d_odd_batch_heads_unaligned_inner` | `[UNALIGNED-STICK]` | `[1, 1, 33, 70]` (all dtypes) | `x.expand(3, 7, 33, 70)` | `[3, 7, 33, 70]` | Non-power-of-2 B=3, H=7, S=33; inner 70 also unaligned |
| `EXP-23` | `test_expand_4d_mqa_16x` | `[GEMMA-SHAPE]` | `[1, 1, 1, 256]` (all dtypes) | `x.expand(1, 16, 32, 256)` | `[1, 16, 32, 256]` | MQA (KV=1, Q=16) 16x broadcast; inner 256 aligned |
| `EXP-24` | `test_expand_5d_block_cache_aligned` | — | `[16, 1, 4, 16, 256]` (all dtypes) | `x.expand(16, 8, 4, 16, 256)` | `[16, 8, 4, 16, 256]` | 5D paged KV cache head broadcast; inner 256 aligned |
| `EXP-25` | `test_expand_5d_block_cache_unaligned` | `[UNALIGNED-STICK]` | `[16, 1, 4, 16, 13]` (all dtypes) | `x.expand(16, 8, 4, 16, 13)` | `[16, 8, 4, 16, 13]` | 5D cache broadcast; inner 13 < 64 (unaligned) |
| `EXP-26` | `test_expand_5d_size1_inner_to_aligned` | `[SIZE-1-DIM]` | `[2, 4, 8, 16, 1]` (all dtypes) | `x.expand(2, 4, 8, 16, 256)` | `[2, 4, 8, 16, 256]` | 5D trailing size-1 expanded to aligned 256 |
| `EXP-27` | `test_expand_4d_size1_all_broadcast` | `[SIZE-1-DIM]` | `[1, 1, 1, 256]` (all dtypes) | `x.expand(4, 16, 32, 256)` | `[4, 16, 32, 256]` | All dims except inner are size-1 |
| `EXP-28` | `test_expand_4d_minus_one_inner_aligned` | — | `[2, 1, 64, 256]` (all dtypes) | `x.expand(-1, 16, -1, -1)` | `[2, 16, 64, 256]` | `-1` preserves all dims except head; inner 256 aligned |

---

## Section 6 — Composite View-Op Chains

Multi-step chains that mirror actual view sequences inside Gemma-4 sub-modules, testing interactions between two or more of the four priority ops compiled as a single kernel.

**Pass Criteria for all Section 6 tests**: `compare_with_cpu`, `atol=1e-2` (numerical tolerance applies when pointwise ops are in the chain)

| ID | Test Name | Classification | Input Dims & Dtype | Chain Pattern | Expected Output Dims | Validation Focus |
|---|---|---|---|---|---|---|
| `CMP-01` | `test_chain_gqa_broadcast_aligned` | `[GEMMA-SHAPE]` `[COMPOSITE]` | `[1, 8, 32, 256]` (all dtypes) | `unsqueeze(2) → expand(1, 8, 2, 32, 256) → reshape(1, 16, 32, 256)` | `[1, 16, 32, 256]` | Full GQA KV 2x broadcast chain; inner 256 aligned |
| `CMP-02` | `test_chain_gqa_broadcast_d72_unaligned` | `[GEMMA-SHAPE]` `[COMPOSITE]` `[UNALIGNED-STICK]` | `[1, 8, 32, 72]` (all dtypes) | `unsqueeze(2) → expand(1, 8, 2, 32, 72) → reshape(1, 16, 32, 72)` | `[1, 16, 32, 72]` | Full GQA KV 2x broadcast with D=72 (unaligned vision) |
| `CMP-03` | `test_chain_reshape_unsqueeze_expand` | `[GEMMA-SHAPE]` `[COMPOSITE]` | `[2, 32, 2048]` (all dtypes) | `reshape(2, 32, 8, 256) → unsqueeze(0) → expand(1, 2, 32, 8, 256)` | `[1, 2, 32, 8, 256]` | QKV split then broadcast for multi-batch inference; inner 256 aligned |
| `CMP-04` | `test_chain_squeeze_unsqueeze_expand` | `[COMPOSITE]` `[UNALIGNED-STICK]` | `[1, 32, 70]` (all dtypes) | `squeeze(0) → unsqueeze(0) → expand(3, 32, 70)` | `[3, 32, 70]` | Remove singleton batch, re-insert, then broadcast; inner 70 unaligned |
| `CMP-05` | `test_chain_view_split_transpose_aligned` | `[GEMMA-SHAPE]` `[COMPOSITE]` | `[1, 32, 4096]` (all dtypes) | `view(1, 32, 16, 256) → transpose(1, 2)` | `[1, 16, 32, 256]` | QKV split + head transpose (standard attention); inner 256 aligned |
| `CMP-06` | `test_chain_view_split_transpose_d72_unaligned` | `[GEMMA-SHAPE]` `[COMPOSITE]` `[UNALIGNED-STICK]` | `[1, 280, 1152]` (all dtypes) | `view(1, 280, 16, 72) → transpose(1, 2)` | `[1, 16, 280, 72]` | Vision QKV split + head transpose; D=72 unaligned |
| `CMP-07` | `test_chain_moe_flatten_unsqueeze_expand` | `[GEMMA-SHAPE]` `[COMPOSITE]` | `[4, 32, 2816]` (all dtypes) | `reshape(-1, 2816) → unsqueeze(-1) → expand(-1, 2816, 8)` | `[128, 2816, 8]` | MoE flatten + broadcast routing weight; inner 8 < 64 (unaligned) |
| `CMP-08` | `test_chain_squeeze_expand_size1_inner` | `[SIZE-1-DIM]` `[COMPOSITE]` | `[1, 32, 1]` (all dtypes) | `squeeze(0) → expand(32, 2816)` | `[32, 2816]` | Squeeze batch, then expand size-1 inner to aligned hidden dim |

---

## 7. Coverage Summary Matrix

| Op / Section | Dim Coverage | Dtypes | Aligned-Stick Tests | Unaligned-Stick Tests | Size-1-Dim Tests | Composite Tests | Total |
|---|---|---|---|---|---|---|---|
| **Module Level** (§1) | 3D, 4D, 5D | `bfloat16` | 6 | 4 | 3 | 0 | **13** |
| `reshape` (§2) | 0D–6D | all three | 9 | 12 | 7 | 0 | **28** |
| `squeeze` (§3) | 0D–6D | all three | 8 | 9 | 9 | 0 | **26** |
| `unsqueeze` (§4) | 0D–6D | all three | 8 | 9 | 7 | 0 | **24** |
| `expand` (§5) | 0D–5D | all three | 10 | 13 | 5 | 0 | **28** |
| **Composite** (§6) | 3D–5D | all three | 2 | 4 | 2 | 8 | **8** |
| **Total** | **0D–6D** | — | **43** | **51** | **33** | **8** | **127** |
