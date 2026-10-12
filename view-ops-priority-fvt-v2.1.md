# Functional Verification Test Plan — Priority View Operations
## `reshape` · `squeeze` · `unsqueeze` · `expand`

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Priority View Operations on torch-spyre: `reshape`, `squeeze`, `unsqueeze`, `expand` |
| **Version** | 2.1 |
| **Status** | Under Review |
| **Date** | 2025-07-11 |
| **Author** | Stuti Sinha |
| **Reviewers** | Kanaka Pathivada, Meena Soundararajan, Albin Joy |
| **Scope** | Text-only Gemma-4 modules (vision tower excluded). Inference-only `torch.compile`. Static concrete dimensions 0D–6D. CPU eager baseline vs compiled Spyre comparison. Default compiler env-var values only. Two model architecture variants (sliding-attention and full-attention) are covered as distinct shape regimes, not as env-var sweeps. |
| **Sources** | `gemma4_26b_module_operator_report_v2.html` (CPU trace, seq=32), `gemma4-view-ops-aten.html` (CPU ATen layer), `gemma4_fx_module_trace_hfadapter_aten.html` (Spyre compile trace, seq=64) |

---

## 1. Scope & Strategy

This document defines functional verification test cases for the four **priority view operations** — `reshape`/`view`, `squeeze`, `unsqueeze`, and `expand` — grounded in real ATen-level operator traces extracted from the **Gemma-4-26B-A4B-it** text stack.

### Key Guidelines

1. **Grounded in real traces.** Every module-level test case is derived from a confirmed ATen operation pattern in at least one of the three source HTML files. Op kinds and concrete shapes are taken directly from those traces; no shapes are invented.
2. **No vision modules.** The vision tower and all sub-modules under `model.vision_tower` had zero ATen activations in both traces (no image input was provided during tracing). They are excluded entirely.
3. **Two levels of testing:**
   - **Module Level (Section 1):** Full forward pass of a Gemma-4 text sub-module, tested as a unified unit. The view-op chain executes inside the module as it does in production.
   - **Operation Level (Sections 2–5):** Standalone tests of each of the four ops in isolation, covering 0D–6D, all three float dtypes, and a systematic range of aligned and unaligned-stick scenarios.
4. **Spyre lowering awareness.** On Spyre, `F.linear` is lowered to `permute → unsqueeze → expand → bmm` rather than `t → view → mm`. The `unsqueeze` and `expand` operation-level tests explicitly cover the weight-broadcast shapes that arise from this lowering.
5. **Default compiler environment variables only.** The two variables relevant to view-op compilation — `LX_PLANNING` and `GLOBAL_STICK_OPTIMIZER` — are kept at their defaults. `SENCORES` is kept at its default of 32. No env-var sweep is performed.
6. **Two model architecture variants, not user configs.** The Gemma-4-26B-A4B-it text stack has exactly two structural attention variants that produce different view-op shapes. Both are covered in module-level and operation-level tests. No further "downscaled CI" or "MQA" configuration sweep is performed — those are out of scope for this FVT. See Section 1.5 for reasoning.
7. **Three dtypes:** `bfloat16` (Gemma production dtype), `float16`, `float32`. Module-level tests use `bfloat16` only. Operation-level tests run all three.

---

## 2. ATen Op Inventory from Traces

The tables below summarise which of the four ops are present in each source, along with the concrete example shapes found. These shapes drive the module-level and operation-level test cases.

### 2.1 CPU Trace (`gemma4_26b_module_operator_report_v2.html`, seq=32)

| ATen Op | Modules Where Found | Representative Shapes |
|---|---|---|
| `aten.view.default` | `Gemma4TextDecoderLayer`, `Gemma4TextAttention`, `Linear` (all projections) | `[1,32,2816]→[32,2816]`; `[1,32,4096]→[1,32,16,256]`; `[32,8]→[256]`; `[256,2816]→[32,8,2816]` |
| `aten.unsqueeze.default` | `Gemma4Model` (mask build), `Gemma4TextAttention` (RoPE + GQA), `Gemma4TextMoE` (routing) | `[32]→[1,32]`; `[1,32,256]→[1,32,1,256]`; `[1,8,32,256]→[1,8,1,32,256]`; `[256]→[256,1]` |
| `aten.expand.default` | `Gemma4Model` (mask), `Gemma4TextAttention` (GQA + matmul prep) | `[1,1,32,32]→[1,-1,32,32]`; `[1,8,1,32,256]→[1,8,2,32,256]`; same-shape expands for bmm |
| `aten.reshape` / `aten.view` | Same modules (reshape decomposes to view or clone+view) | Same as `view` above |
| `aten.squeeze` | **Not observed** at ATen level in this trace | — (tested at op-level as standalone) |

### 2.2 Spyre Compile Trace (`gemma4_fx_module_trace_hfadapter_aten.html`, seq=64)

| ATen Op | Modules Where Found | Representative Shapes | Note |
|---|---|---|---|
| `aten.view.default` | `Gemma4Attention` (after bmm) | `[1,64,4096]→[1,64,16,256]`; `[1,64,2048]→[1,64,8,256]` | QKV split after batched matmul |
| `aten.unsqueeze.default` | All `Linear` sub-modules (q/k/v/o proj, gate/up/down proj) | `[2816,4096]→[1,2816,4096]`; `[2816,2048]→[1,2816,2048]`; `[4096,2816]→[1,4096,2816]` | Spyre lowering of `F.linear` weight: `permute→unsqueeze→expand→bmm` replaces `t→view→mm` |
| `aten.expand.default` | All `Linear` sub-modules | `[1,64,2816]→[1,64,2816]`; `[1,2816,4096]→[1,2816,4096]` | Same-shape expands for bmm operand alignment |
| `aten.reshape` / `aten.view` | — | — | Reshape not visible in Spyre trace at this level |
| `aten.squeeze` | **Not observed** | — | — |

---

## 1.5 On Model Configurations

The v1 FVT plan included five configuration regimes (A–E) sweeping `hidden_size`, `num_attention_heads`, `head_dim`, `num_experts`, and batch/sequence sizes. The question is whether those model parameters are in scope for this FVT.

### What model parameters change view-op shapes

Every model config parameter directly determines the tensor shapes that flow through `reshape`, `unsqueeze`, and `expand`:

| Parameter changed | View op affected | Shape that changes |
|---|---|---|
| `head_dim` (D) | `reshape` (QKV split), `unsqueeze`+`expand` (GQA) | `[B,S,H*D]→[B,S,H,D]`; `[B,KV,1,S,D]→[B,KV,r,S,D]` |
| `num_attention_heads` / `num_kv_heads` | `unsqueeze`+`expand` (GQA expansion factor) | repeat factor r = H/KV changes |
| `hidden_size` | `reshape` (token flatten), `expand` (bmm weight) | `[B,S,H]→[B*S,H]` |
| `num_experts` × `top_k` | `reshape` (MoE dispatch index) | `[B*S, K]→[B*S*K]` |
| `batch_size` B / `seq_len` S | All view ops | First two dims of every activation tensor |

### Decision: two architecture variants, no config sweep

The 26B-A4B-it model has exactly **two structural layer variants** that differ in their attention architecture:

| Variant | Layers | H | KV | D | GQA factor | Key view-op shapes |
|---|---|---|---|---|---|---|
| **Sliding-attention** | 0–4, 6–10, … (25 layers) | 16 | 8 | 256 | 2× | `[B,S,4096]→[B,S,16,256]`; `[B,8,1,S,256]→[B,8,2,S,256]` |
| **Full-attention** | 5, 11, 17, 23, 29 (5 layers) | 16 | 2 | 512 | 8× | `[B,S,8192]→[B,S,16,512]`; `[B,2,1,S,512]→[B,2,8,S,512]` |

**Both variants are tested** throughout this FVT because they produce distinct innermost dimensions (256 vs 512) and distinct GQA expansion factors (2× vs 8×). This is not a "config sweep" — it is covering the two shapes that actually exist in the production model.

**Additional configurations (downscaled CI, MQA, arbitrary batch/seq sweeps) are excluded** for this FVT because:
- They do not add new alignment scenarios beyond what the unaligned-stick section already covers systematically.
- The unaligned-stick tests (Section 2–5) already exercise all arithmetic combinations of inner-dim alignment that matter for the Spyre compiler, independently of which Gemma config produces them.
- A config sweep would multiply test count without adding compiler coverage.

If a future FVT explicitly targets multi-core scaling or serving-framework integration, SENCORES and batch/seq sweeps would be appropriate there.

---

## 3. Gemma-4 Text Modules Under Test

Only text-stack modules that actually produced view ops in the traces are included.

| # | Module | Class | Op(s) Used | Seq Len in Trace |
|---|---|---|---|---|
| 1 | `model` | `Gemma4Model` | `unsqueeze` (mask build: `[32]→[1,32]`, `[1,32]→[1,1,32]`, etc.), `expand` (mask broadcast), `clone` (position ids) | seq=32 |
| 2 | `model.language_model.layers.N` | `Gemma4TextDecoderLayer` | `view` (`[1,32,2816]→[32,2816]` + reverse) | seq=32 |
| 3 | `model.language_model.layers.N.self_attn` | `Gemma4TextAttention` / `Gemma4Attention` | `view` (QKV split), `unsqueeze` (RoPE freqs + GQA prep), `expand` (GQA repeat_kv + mask), `clone` (materialise expanded KV) | seq=32/64 |
| 4 | `model.language_model.layers.N.{q,k,v,o}_proj` | `Linear` (Spyre lowering) | `unsqueeze` (weight `[out,in]→[1,out,in]`), `expand` (input + weight for bmm) | seq=64 |
| 5 | `model.language_model.layers.N.mlp.{gate,up,down}_proj` | `Linear` (Spyre lowering) | `unsqueeze` (weight `[2816,2112]→[1,2816,2112]`), `expand` | seq=64 |
| 6 | `model.language_model.layers.N.moe_block` | `Gemma4TextMoE` | `view` (flatten/unflatten indices and outputs), `unsqueeze` (routing weight trailing dim) | seq=32 |
| 7 | `model.language_model.rotary_emb` | `Gemma4TextRotaryEmbedding` | `unsqueeze` (freq table shape prep), `expand` (batch broadcast), `view` (internal batching) | seq=32 |
| 8 | `lm_head` | `Linear` | `view` (`[1,32,2816]→[32,2816]`), implied `unsqueeze`+`expand` on weight `[262144,2816]` on Spyre | seq=32 |

---

## 4. Stick Alignment Reference

Spyre hardware stores tensors in 128-byte memory units called **sticks**. The **last (innermost) tensor dimension** maps to the stick dimension. Elements per stick by dtype:

| Dtype | Bytes/elem | Elements per stick |
|---|---|---|
| `float16` | 2 B | **64** |
| `bfloat16` | 2 B | **64** |
| `float32` | 4 B | **32** |

**Aligned:** innermost dim divisible by 64 (fp16/bf16) or 32 (fp32).
Examples: 32, 64, 128, 256, 512, 704, 2112, 2816, 4096.

**Unaligned — two sub-cases:**
- *Below one stick:* innermost dim < 64 (fp16/bf16) or < 32 (fp32). Examples: 1, 7, 13, 30.
- *Not divisible:* dim ≥ stick width but not a clean multiple. Examples: 33, 65, 70, 128+1=129, 256+1=257.

**Size-1 trailing dimension** — the extreme unaligned case: 1 element occupies one stick with 63 padding slots (fp16/bf16). Arises from `unsqueeze(-1)`, `reshape(...,1)`, or `squeeze`-all.

**How to produce unaligned scenarios:** choose tensor shapes where the last dimension is not a multiple of 64 (fp16/bf16) or 32 (fp32). No special flags are needed — the compiler sees the alignment condition automatically.

Aligned inner dims from model traces: **256**, **512**, **2816**, **4096**, **2048**, **2112**.
Unaligned inner dims of interest: **33**, **65**, **70**, **129**, **257**, **1** (size-1).

---

## 5. Test Environment

All values are taken from `torch_spyre/_inductor/config.py` (inductor config) and `torch_spyre/_inductor/fusion.py`. The "Default" column shows the value hard-coded in source; the "FVT Value" column confirms this FVT runs at that default.

| Variable | Source File | Default (from source) | FVT Value | Relevance to view ops |
|---|---|---|---|---|
| `SENCORES` | `config.py` | **32** | 32 | Core count; does not affect view-op shape or correctness, only work division |
| `LX_PLANNING` | `config.py` | **1 (True)** | 1 | LX scratchpad planning; affects kernel scheduling, not view-op decomposition |
| `CO_OPTIMIZING_LX_PLANNING` | `config.py` | **0 (False)** | 0 | Joint LX + work-division solver; no effect on view-op lowering |
| `GLOBAL_STICK_OPTIMIZER` | `config.py` | **1 (True)** | 1 | Stick optimizer for non-contiguous views; directly relevant — must be on |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | `fusion.py` | **True** | True | View + pointwise/reduction fusion; should be on for realistic compilation |

> **Note:** The previous version of this FVT listed SENCORES=1 and LX_PLANNING=0 as defaults. Both were incorrect. The correct defaults from source are SENCORES=32 and LX_PLANNING=1 (True). This FVT runs at those correct defaults.

---

## 6. Table of Contents

| Section | Category | Test IDs | Count |
|---|---|---|---|
| **Section 1** | Module-Level End-to-End Tests | `MOD-01` – `MOD-16` | 16 |
| **Section 2** | `reshape` / `view` — Operation Level | `RSH-01` – `RSH-30` | 30 |
| **Section 3** | `squeeze` — Operation Level | `SQZ-01` – `SQZ-24` | 24 |
| **Section 4** | `unsqueeze` — Operation Level | `UNS-01` – `UNS-28` | 28 |
| **Section 5** | `expand` — Operation Level | `EXP-01` – `EXP-28` | 28 |
| — | **Total** | | **126** |

---

## Section 1 — Module-Level End-to-End Tests

Each test wraps a Gemma-4 text sub-module forward pass so that the view-op chain executes exactly as it does in the model. All tests use `bfloat16`. Shapes are taken directly from the HTML traces.

**Default pass criteria:** `compare_with_cpu`, `atol=1e-2`

| ID | Test Name | Module / Class | Source Trace | ATen Ops Exercised | Input Shape(s) | Expected Output Shape | Notes |
|---|---|---|---|---|---|---|---|
| `MOD-01` | `test_mod_decoder_token_flatten` | `Gemma4TextDecoderLayer` | CPU trace, seq=32 | `view` | `[1, 32, 2816]` bf16 | `[32, 2816]` bf16 | `[B,S,H]→[B*S,H]` for router/expert input; direct `aten.view` from trace |
| `MOD-02` | `test_mod_decoder_token_unflatten` | `Gemma4TextDecoderLayer` | CPU trace, seq=32 | `view` | `[32, 2816]` bf16 | `[1, 32, 2816]` bf16 | Reverse of MOD-01; restores shape after expert aggregation |
| `MOD-03` | `test_mod_attn_qkv_view_split_sliding` | `Gemma4TextAttention` | CPU + Spyre trace | `view` | `[1, 32, 4096]` bf16 | `[1, 32, 16, 256]` bf16 | Q projection split (H=16, D=256, sliding layer); inner 256 aligned |
| `MOD-04` | `test_mod_attn_qkv_view_split_kv_sliding` | `Gemma4TextAttention` | CPU trace, seq=32 | `view` | `[1, 32, 2048]` bf16 | `[1, 32, 8, 256]` bf16 | KV projection split (KV_H=8, D=256); inner 256 aligned |
| `MOD-05` | `test_mod_attn_qkv_view_split_spyre_s64` | `Gemma4Attention` (Spyre) | Spyre trace, seq=64 | `view` | `[1, 64, 4096]` bf16 | `[1, 64, 16, 256]` bf16 | Same Q-split at Spyre seq=64; verifies compiled path |
| `MOD-06` | `test_mod_attn_head_merge` | `Gemma4TextAttention` | CPU trace | `view` / `reshape` | `[1, 32, 16, 256]` bf16 | `[1, 32, 4096]` bf16 | Merge heads back after attention output |
| `MOD-07` | `test_mod_attn_rope_unsqueeze_freq` | `Gemma4TextAttention` | CPU trace | `unsqueeze` | `[1, 32, 256]` bf16 | `[1, 32, 1, 256]` bf16 | Adds head axis to cos/sin for RoPE broadcast; from `aten.unsqueeze.default` |
| `MOD-08` | `test_mod_attn_gqa_unsqueeze_for_expand` | `Gemma4TextAttention` | CPU trace | `unsqueeze` | `[1, 8, 32, 256]` bf16 | `[1, 8, 1, 32, 256]` bf16 | `repeat_kv`: inserts repeat dim before expand; from `aten.unsqueeze.default` |
| `MOD-09` | `test_mod_attn_gqa_expand_2x` | `Gemma4TextAttention` | CPU trace | `expand` | `[1, 8, 1, 32, 256]` bf16 | `[1, 8, 2, 32, 256]` bf16 | `repeat_kv`: broadcast 8→16 heads (2x); no data copy |
| `MOD-10` | `test_mod_attn_gqa_expand_8x_full_attn` | `Gemma4TextAttention` | CPU trace | `expand` | `[1, 2, 1, 32, 512]` bf16 | `[1, 2, 8, 32, 512]` bf16 | `repeat_kv`: broadcast 2→16 heads (8x) for full-attention layers (D=512) |
| `MOD-11` | `test_mod_mask_unsqueeze_chain` | `Gemma4Model` | CPU trace | `unsqueeze` ×4 | `[32]` int64 | `[1, 1, 32, 1]` int64 | Position-id chain: `[32]→[1,32]→[1,1,32]→[1,1,32,1]`; from `aten.unsqueeze.default` ×3 |
| `MOD-12` | `test_mod_mask_expand` | `Gemma4Model` | CPU trace | `expand` | `[1, 1, 32, 32]` bf16 | `[1, 1, 32, 32]` bf16 | Same-shape expand for mask broadcast (`[1,-1,32,32]`); from `aten.expand.default` |
| `MOD-13` | `test_mod_moe_index_view_flatten` | `Gemma4TextMoE` | CPU trace | `view` | `[32, 8]` int64 | `[256]` int64 | Flatten top-k indices; from `aten.view.default`; int64 dtype |
| `MOD-14` | `test_mod_moe_output_view_unflatten` | `Gemma4TextMoE` | CPU trace | `view` | `[256, 2816]` bf16 | `[32, 8, 2816]` bf16 | Unflatten expert outputs back to `[tokens, top_k, hidden]` |
| `MOD-15` | `test_mod_moe_routing_unsqueeze` | `Gemma4TextMoE` | CPU trace | `unsqueeze` | `[256]` bf16 | `[256, 1]` bf16 | Adds trailing dim to per-slot routing weights for hidden-dim broadcast |
| `MOD-16` | `test_mod_linear_spyre_unsqueeze_weight` | `Linear` (Spyre lowering) | Spyre trace | `unsqueeze` | `[4096, 2816]` bf16 | `[1, 4096, 2816]` bf16 | Spyre lowering of `F.linear`: weight `permute→unsqueeze(0)→expand→bmm`; dim 0 insert |

---

## Section 2 — `reshape` / `view` Operation Level

Validates `reshape` and `view` across 0D–6D, all three dtypes, and the full aligned / unaligned / size-1 spectrum. Shapes include all forms observed in the model traces plus systematic dimension-coverage additions.

**Pass criteria:** `compare_with_cpu`, `atol=0.0` (reshape is layout-only; no numeric error expected)

### Aligned stick — innermost dim divisible by stick width

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `RSH-01` | `test_reshape_3d_to_4d_qsplit_sliding` | `[1, 32, 4096]` (all dtypes) | `x.view(1, 32, 16, 256)` | `[1, 32, 16, 256]` | **Gemma trace.** Inner 256: 256%64=0, aligned |
| `RSH-02` | `test_reshape_3d_to_4d_kvsplit_sliding` | `[1, 32, 2048]` (all dtypes) | `x.view(1, 32, 8, 256)` | `[1, 32, 8, 256]` | **Gemma trace.** Inner 256: aligned |
| `RSH-03` | `test_reshape_3d_to_4d_qsplit_full_attn` | `[1, 32, 8192]` (all dtypes) | `x.view(1, 32, 16, 512)` | `[1, 32, 16, 512]` | **Gemma trace.** Inner 512: aligned |
| `RSH-04` | `test_reshape_3d_to_4d_spyre_s64` | `[1, 64, 4096]` (all dtypes) | `x.view(1, 64, 16, 256)` | `[1, 64, 16, 256]` | **Spyre trace** seq=64. Inner 256: aligned |
| `RSH-05` | `test_reshape_3d_to_2d_token_flatten` | `[1, 32, 2816]` (all dtypes) | `x.view(-1, 2816)` | `[32, 2816]` | **Gemma trace** decoder. Inner 2816: aligned |
| `RSH-06` | `test_reshape_2d_to_3d_token_unflatten` | `[32, 2816]` (all dtypes) | `x.view(1, 32, 2816)` | `[1, 32, 2816]` | **Gemma trace** decoder. Inner 2816: aligned |
| `RSH-07` | `test_reshape_4d_to_3d_head_merge` | `[1, 32, 16, 256]` (all dtypes) | `x.reshape(1, 32, 4096)` | `[1, 32, 4096]` | **Gemma trace.** Inner 4096: aligned |
| `RSH-08` | `test_reshape_2d_moe_index_flatten` | `[32, 8]` (int64) | `x.view(-1)` | `[256]` | **Gemma trace** MoE. int64, inner 8: 8<64, unaligned — but test placed here as Gemma-exact |
| `RSH-09` | `test_reshape_2d_moe_output_unflatten` | `[256, 2816]` (all dtypes) | `x.view(32, 8, 2816)` | `[32, 8, 2816]` | **Gemma trace** MoE. Inner 2816: aligned |
| `RSH-10` | `test_reshape_1d_to_2d_aligned` | `[4096]` (all dtypes) | `x.reshape(64, 64)` | `[64, 64]` | Inner 64: aligned |
| `RSH-11` | `test_reshape_5d_to_3d_aligned` | `[1, 2, 4, 8, 256]` (all dtypes) | `x.reshape(1, 8, 2048)` | `[1, 8, 2048]` | Inner 2048: aligned; collapses 5D to 3D |
| `RSH-12` | `test_reshape_inferred_dim_aligned` | `[2, 32, 16, 256]` (all dtypes) | `x.reshape(2, -1, 256)` | `[2, 512, 256]` | `-1` inference; inner 256: aligned |

### Unaligned stick — innermost dim not a multiple of stick width

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `RSH-13` | `test_reshape_3d_to_2d_unaligned_33` | `[4, 8, 33]` (all dtypes) | `x.reshape(32, 33)` | `[32, 33]` | Inner 33: 33<64, unaligned |
| `RSH-14` | `test_reshape_2d_to_1d_unaligned_65` | `[8, 65]` (all dtypes) | `x.reshape(-1)` | `[520]` | Input inner 65: 65%64≠0, unaligned |
| `RSH-15` | `test_reshape_3d_unaligned_to_4d` | `[1, 32, 2080]` (all dtypes) | `x.reshape(1, 32, 8, 260)` | `[1, 32, 8, 260]` | Inner 260: 260%64=4, unaligned |
| `RSH-16` | `test_reshape_4d_head_merge_unaligned` | `[2, 8, 32, 70]` (all dtypes) | `x.reshape(2, 32, 560)` | `[2, 32, 560]` | Input inner 70: unaligned (70%64=6); output 560: 560%64=48, also unaligned |
| `RSH-17` | `test_reshape_3d_to_4d_inner_129` | `[1, 32, 1032]` (all dtypes) | `x.reshape(1, 32, 8, 129)` | `[1, 32, 8, 129]` | Inner 129: 129%64=1, unaligned |
| `RSH-18` | `test_reshape_2d_to_3d_inner_257` | `[4, 257]` (all dtypes) | `x.reshape(4, 1, 257)` | `[4, 1, 257]` | Inner 257: 257%64=1, unaligned; fp32 stick: 257%32=1 also unaligned |
| `RSH-19` | `test_reshape_inferred_dim_unaligned` | `[3, 7, 6]` (all dtypes) | `x.reshape(3, -1)` | `[3, 42]` | `-1` inference; inner 42: 42%64≠0, unaligned |
| `RSH-20` | `test_reshape_5d_to_3d_unaligned` | `[1, 2, 4, 8, 7]` (all dtypes) | `x.reshape(1, 8, 56)` | `[1, 8, 56]` | Input inner 7: 7<64; output 56: 56%64≠0, unaligned |

### Size-1 dimensions (extreme unaligned case)

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `RSH-21` | `test_reshape_0d_scalar_to_1d` | `[]` (all dtypes) | `x.reshape(1)` | `[1]` | 0D scalar; result inner=1, ultra-unaligned |
| `RSH-22` | `test_reshape_1d_size1_to_3d` | `[1]` (all dtypes) | `x.reshape(1, 1, 1)` | `[1, 1, 1]` | All dims=1 |
| `RSH-23` | `test_reshape_2d_size1_trailing` | `[128, 1]` (all dtypes) | `x.reshape(128)` | `[128]` | Remove trailing size-1; inner 1 → inner 128 aligned |
| `RSH-24` | `test_reshape_3d_size1_inner` | `[4, 32, 1]` (all dtypes) | `x.reshape(4, 32)` | `[4, 32]` | Input trailing size-1; new inner 32 < 64 (unaligned for fp16/bf16) |
| `RSH-25` | `test_reshape_4d_two_size1_dims` | `[1, 8, 1, 256]` (all dtypes) | `x.reshape(8, 256)` | `[8, 256]` | Two size-1 dims removed; inner 256 aligned |
| `RSH-26` | `test_reshape_6d_size1_trailing` | `[2, 3, 4, 5, 6, 1]` (all dtypes) | `x.reshape(2, 3, 4, 5, 6)` | `[2, 3, 4, 5, 6]` | Trailing size-1 collapsed; new inner 6 < 64, unaligned |

### High-rank (5D / 6D)

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `RSH-27` | `test_reshape_4d_to_5d_unfold` | `[1, 2, 4, 256]` (all dtypes) | `x.reshape(1, 2, 4, 4, 64)` | `[1, 2, 4, 4, 64]` | Split last dim; inner 64: aligned |
| `RSH-28` | `test_reshape_5d_to_6d_unfold` | `[1, 2, 4, 8, 256]` (all dtypes) | `x.reshape(1, 2, 4, 8, 4, 64)` | `[1, 2, 4, 8, 4, 64]` | 5D→6D; inner 64: aligned |
| `RSH-29` | `test_reshape_6d_collapse` | `[1, 2, 4, 8, 4, 64]` (all dtypes) | `x.reshape(1, 2, 4, -1)` | `[1, 2, 4, 2048]` | 6D→4D; inner 2048: aligned |
| `RSH-30` | `test_reshape_6d_unaligned_inner` | `[1, 2, 4, 8, 4, 7]` (all dtypes) | `x.reshape(1, 8, 224)` | `[1, 8, 224]` | Input inner 7 < 64; output 224: 224%64=32, unaligned |

---

## Section 3 — `squeeze` Operation Level

Validates `squeeze` across 0D–6D inputs, all three dtypes. `squeeze` was not observed in the Gemma ATen traces (it is not needed by the text stack for activation tensors) but is on the target op list and must be tested standalone.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### No-op cases (targeted dim is not size-1 — verifies correctness of no-op path)

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `SQZ-01` | `test_squeeze_1d_noop_aligned` | `[256]` (all dtypes) | `x.squeeze(0)` | `[256]` | dim-0 size=256≠1; no-op; inner 256 aligned |
| `SQZ-02` | `test_squeeze_1d_noop_unaligned` | `[63]` (all dtypes) | `x.squeeze(0)` | `[63]` | dim-0 size=63≠1; no-op; inner 63 < 64, unaligned |
| `SQZ-03` | `test_squeeze_3d_noop_middle` | `[1, 32, 2816]` (all dtypes) | `x.squeeze(1)` | `[1, 32, 2816]` | dim-1 size=32≠1; inner 2816 aligned |
| `SQZ-04` | `test_squeeze_4d_noop_all` | `[2, 16, 32, 256]` (all dtypes) | `x.squeeze()` | `[2, 16, 32, 256]` | No size-1 dims present; entire call is a no-op |

### Aligned inner dim after squeeze

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `SQZ-05` | `test_squeeze_2d_leading_batch_aligned` | `[1, 256]` (all dtypes) | `x.squeeze(0)` | `[256]` | Remove batch; inner 256 aligned |
| `SQZ-06` | `test_squeeze_3d_batch_aligned` | `[1, 32, 2816]` (all dtypes) | `x.squeeze(0)` | `[32, 2816]` | Remove batch; inner 2816 aligned (Gemma hidden dim) |
| `SQZ-07` | `test_squeeze_3d_seq_decode_aligned` | `[2, 1, 256]` (all dtypes) | `x.squeeze(1)` | `[2, 256]` | Remove seq=1 decode step; inner 256 aligned |
| `SQZ-08` | `test_squeeze_4d_decode_seq_aligned` | `[1, 1, 16, 256]` (all dtypes) | `x.squeeze(1)` | `[1, 16, 256]` | Single-token decode squeeze; inner 256 aligned |
| `SQZ-09` | `test_squeeze_4d_tuple_dims` | `[1, 16, 1, 256]` (all dtypes) | `x.squeeze((0, 2))` | `[16, 256]` | Multi-dim tuple squeeze; inner 256 aligned |
| `SQZ-10` | `test_squeeze_4d_negative_trailing` | `[32, 8, 64, 1]` (all dtypes) | `x.squeeze(-1)` | `[32, 8, 64]` | Trailing size-1; new inner 64 aligned |
| `SQZ-11` | `test_squeeze_5d_all_singletons_aligned` | `[1, 1, 32, 1, 256]` (all dtypes) | `x.squeeze()` | `[32, 256]` | Remove all size-1; inner 256 aligned |
| `SQZ-12` | `test_squeeze_6d_leading_aligned` | `[1, 2, 4, 1, 8, 256]` (all dtypes) | `x.squeeze(0)` | `[2, 4, 1, 8, 256]` | Remove leading batch; inner 256 aligned |

### Unaligned inner dim after squeeze

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `SQZ-13` | `test_squeeze_2d_leading_unaligned` | `[1, 33]` (all dtypes) | `x.squeeze(0)` | `[33]` | Remove batch; inner 33 < 64, unaligned |
| `SQZ-14` | `test_squeeze_3d_seq_decode_unaligned` | `[2, 1, 70]` (all dtypes) | `x.squeeze(1)` | `[2, 70]` | Remove seq=1; inner 70: 70%64=6, unaligned |
| `SQZ-15` | `test_squeeze_4d_decode_unaligned` | `[1, 1, 16, 65]` (all dtypes) | `x.squeeze(1)` | `[1, 16, 65]` | Decode squeeze; inner 65: 65%64=1, unaligned |
| `SQZ-16` | `test_squeeze_4d_tuple_dims_unaligned` | `[1, 16, 1, 33]` (all dtypes) | `x.squeeze((0, 2))` | `[16, 33]` | Multi-dim tuple; inner 33 unaligned |
| `SQZ-17` | `test_squeeze_4d_negative_unaligned` | `[32, 8, 30, 1]` (all dtypes) | `x.squeeze(-1)` | `[32, 8, 30]` | Trailing size-1; new inner 30 < 64, unaligned |
| `SQZ-18` | `test_squeeze_5d_all_singletons_unaligned` | `[1, 1, 32, 1, 30]` (all dtypes) | `x.squeeze()` | `[32, 30]` | Remove all size-1; inner 30 < 64, unaligned |
| `SQZ-19` | `test_squeeze_6d_leading_unaligned` | `[1, 2, 4, 1, 8, 13]` (all dtypes) | `x.squeeze(0)` | `[2, 4, 1, 8, 13]` | Remove batch; inner 13 < 64, unaligned |

### Size-1 dimension cases (ultra-unaligned — 1 element per stick)

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `SQZ-20` | `test_squeeze_1d_size1_to_scalar` | `[1]` (all dtypes) | `x.squeeze(0)` | `[]` (scalar) | 1→0D; input inner=1 (ultra-unaligned) |
| `SQZ-21` | `test_squeeze_2d_all_singletons` | `[1, 1]` (all dtypes) | `x.squeeze()` | `[]` (scalar) | All dims=1; 0D result |
| `SQZ-22` | `test_squeeze_4d_all_singletons_aligned` | `[1, 1, 32, 1]` (all dtypes) | `x.squeeze()` | `[32]` | Remove all size-1; inner 32: < 64 for fp16/bf16 (unaligned), aligned for fp32 |
| `SQZ-23` | `test_squeeze_4d_all_singletons_unaligned` | `[1, 1, 63, 1]` (all dtypes) | `x.squeeze()` | `[63]` | Remove all size-1; inner 63 < 64, unaligned for all dtypes |
| `SQZ-24` | `test_squeeze_2d_trailing_size1` | `[128, 1]` (all dtypes) | `x.squeeze(-1)` | `[128]` | Trailing size-1 removed; new inner 128 aligned |

---

## Section 4 — `unsqueeze` Operation Level

Validates `unsqueeze` across 0D–5D inputs (producing up to 6D), all three dtypes, positive and negative index positions, and all stick alignment scenarios. Both the activation-tensor patterns (Gemma attention / MoE) and the Spyre-lowering weight-tensor patterns (Linear projection) are covered.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### Aligned inner dim — Gemma activation patterns

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Source / Stick Note |
|---|---|---|---|---|---|
| `UNS-01` | `test_unsqueeze_1d_pos_ids_to_2d` | `[32]` int64 | `x.unsqueeze(0)` | `[1, 32]` | **Gemma trace** (`Gemma4Model`). Position ids; int64 |
| `UNS-02` | `test_unsqueeze_2d_rope_freq_head_dim` | `[1, 32, 256]` (all dtypes) | `x.unsqueeze(2)` | `[1, 32, 1, 256]` | **Gemma trace** (`Gemma4TextAttention`). RoPE cos/sin broadcast; inner 256 aligned |
| `UNS-03` | `test_unsqueeze_4d_gqa_repeat_dim` | `[1, 8, 32, 256]` (all dtypes) | `x.unsqueeze(2)` | `[1, 8, 1, 32, 256]` | **Gemma trace** (`Gemma4TextAttention`). repeat_kv insert dim before expand; inner 256 aligned |
| `UNS-04` | `test_unsqueeze_1d_routing_weight_trailing` | `[256]` (all dtypes) | `x.unsqueeze(-1)` | `[256, 1]` | **Gemma trace** (`Gemma4TextMoE`). Trailing dim for hidden-dim broadcast; new inner=1 (ultra-unaligned) |
| `UNS-05` | `test_unsqueeze_2d_mask_chain_step1` | `[1, 32]` int64 | `x.unsqueeze(1)` | `[1, 1, 32]` | **Gemma trace** (`Gemma4Model`) mask build chain step |
| `UNS-06` | `test_unsqueeze_3d_mask_chain_step2` | `[1, 1, 32]` int64 | `x.unsqueeze(3)` | `[1, 1, 32, 1]` | **Gemma trace** (`Gemma4Model`) mask build chain; new inner=1 (ultra-unaligned) |
| `UNS-07` | `test_unsqueeze_2d_batch_insert_2816` | `[32, 2816]` (all dtypes) | `x.unsqueeze(0)` | `[1, 32, 2816]` | Batch dim insert; inner 2816 aligned |
| `UNS-08` | `test_unsqueeze_3d_head_merge_5d` | `[2, 8, 32, 256]` (all dtypes) | `x.unsqueeze(2)` | `[2, 8, 1, 32, 256]` | GQA repeat_kv 8x; inner 256 aligned |
| `UNS-09` | `test_unsqueeze_4d_full_attn_gqa` | `[1, 2, 32, 512]` (all dtypes) | `x.unsqueeze(2)` | `[1, 2, 1, 32, 512]` | Full-attention KV repeat (D=512); inner 512 aligned |

### Aligned inner dim — Spyre Linear lowering patterns

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Source / Stick Note |
|---|---|---|---|---|---|
| `UNS-10` | `test_unsqueeze_spyre_linear_q_weight` | `[4096, 2816]` (all dtypes) | `x.unsqueeze(0)` | `[1, 4096, 2816]` | **Spyre trace** q_proj weight; inner 2816 aligned |
| `UNS-11` | `test_unsqueeze_spyre_linear_kv_weight` | `[2048, 2816]` (all dtypes) | `x.unsqueeze(0)` | `[1, 2048, 2816]` | **Spyre trace** k/v_proj weight; inner 2816 aligned |
| `UNS-12` | `test_unsqueeze_spyre_linear_o_weight` | `[2816, 4096]` (all dtypes) | `x.unsqueeze(0)` | `[1, 2816, 4096]` | **Spyre trace** o_proj weight; inner 4096 aligned |
| `UNS-13` | `test_unsqueeze_spyre_linear_mlp_weight` | `[2816, 2112]` (all dtypes) | `x.unsqueeze(0)` | `[1, 2816, 2112]` | **Spyre trace** MLP gate/up/down weight; inner 2112 aligned |

### Unaligned inner dim

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `UNS-14` | `test_unsqueeze_1d_front_unaligned` | `[70]` (all dtypes) | `x.unsqueeze(0)` | `[1, 70]` | Inner 70: 70%64=6, unaligned |
| `UNS-15` | `test_unsqueeze_2d_batch_unaligned` | `[32, 33]` (all dtypes) | `x.unsqueeze(0)` | `[1, 32, 33]` | Batch insert; inner 33 < 64, unaligned |
| `UNS-16` | `test_unsqueeze_3d_rope_freq_unaligned` | `[1, 32, 70]` (all dtypes) | `x.unsqueeze(2)` | `[1, 32, 1, 70]` | RoPE-style insert; inner 70 unaligned |
| `UNS-17` | `test_unsqueeze_3d_rope_freq_65` | `[1, 32, 65]` (all dtypes) | `x.unsqueeze(2)` | `[1, 32, 1, 65]` | Inner 65: 65%64=1, unaligned; just above one stick |
| `UNS-18` | `test_unsqueeze_4d_gqa_unaligned` | `[2, 8, 32, 65]` (all dtypes) | `x.unsqueeze(2)` | `[2, 8, 1, 32, 65]` | GQA-style insert; inner 65 unaligned |
| `UNS-19` | `test_unsqueeze_5d_to_6d_unaligned` | `[1, 2, 4, 8, 13]` (all dtypes) | `x.unsqueeze(3)` | `[1, 2, 4, 1, 8, 13]` | 5D→6D; inner 13 < 64, unaligned |
| `UNS-20` | `test_unsqueeze_2d_attn_mask_unaligned_seq` | `[1, 33]` (all dtypes) | `x.unsqueeze(1).unsqueeze(2)` | `[1, 1, 1, 33]` | Double unsqueeze; inner 33 unaligned |

### Size-1 dimension cases

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `UNS-21` | `test_unsqueeze_scalar_to_1d` | `[]` (all dtypes) | `x.unsqueeze(0)` | `[1]` | 0D→1D; result inner=1, ultra-unaligned |
| `UNS-22` | `test_unsqueeze_1d_size1_to_2d` | `[1]` (all dtypes) | `x.unsqueeze(0)` | `[1, 1]` | Size-1 input; result both size-1 |
| `UNS-23` | `test_unsqueeze_trailing_size1` | `[32, 256]` (all dtypes) | `x.unsqueeze(-1)` | `[32, 256, 1]` | New trailing inner=1, ultra-unaligned |
| `UNS-24` | `test_unsqueeze_3d_trailing_size1_unaligned` | `[4, 8, 70]` (all dtypes) | `x.unsqueeze(-1)` | `[4, 8, 70, 1]` | New inner=1; original last dim 70 was unaligned |
| `UNS-25` | `test_unsqueeze_dual_axis_size1` | `[32, 256]` (all dtypes) | `x.unsqueeze(0).unsqueeze(3)` | `[1, 32, 256, 1]` | Dual unsqueeze; leading+trailing size-1 |
| `UNS-26` | `test_unsqueeze_dual_axis_unaligned_inner` | `[32, 70]` (all dtypes) | `x.unsqueeze(0).unsqueeze(3)` | `[1, 32, 70, 1]` | New inner=1; preceding dim 70 was unaligned |
| `UNS-27` | `test_unsqueeze_5d_to_6d_aligned` | `[1, 2, 4, 8, 256]` (all dtypes) | `x.unsqueeze(3)` | `[1, 2, 4, 1, 8, 256]` | 5D→6D; inner 256 aligned |
| `UNS-28` | `test_unsqueeze_negative_index_middle` | `[1, 16, 32, 256]` (all dtypes) | `x.unsqueeze(-2)` | `[1, 16, 32, 1, 256]` | Negative index inserts before last; inner 256 aligned |

---

## Section 5 — `expand` Operation Level

Validates `expand` across 0D–5D inputs, all three dtypes. Both the activation-level patterns (GQA, attention mask, RoPE) and the Spyre Linear-lowering patterns (same-shape bmm-operand expands) are covered.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### Aligned inner dim — Gemma activation patterns

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Source / Stick Note |
|---|---|---|---|---|---|
| `EXP-01` | `test_expand_gqa_2x_sliding` | `[1, 8, 1, 32, 256]` (all dtypes) | `x.expand(1, 8, 2, 32, 256)` | `[1, 8, 2, 32, 256]` | **Gemma trace.** repeat_kv 8→16 heads (2x); inner 256 aligned |
| `EXP-02` | `test_expand_gqa_8x_full_attn` | `[1, 2, 1, 32, 512]` (all dtypes) | `x.expand(1, 2, 8, 32, 512)` | `[1, 2, 8, 32, 512]` | **Gemma trace.** repeat_kv 2→16 heads (8x, D=512); inner 512 aligned |
| `EXP-03` | `test_expand_mask_same_shape` | `[1, 1, 32, 32]` (all dtypes) | `x.expand(1, -1, 32, 32)` | `[1, 1, 32, 32]` | **Gemma trace.** Same-shape mask broadcast (`[1,-1,32,32]`); inner 32: 32<64, unaligned for fp16/bf16 |
| `EXP-04` | `test_expand_gqa_rope_freq_heads` | `[1, 32, 1, 256]` (all dtypes) | `x.expand(1, 32, 16, 256)` | `[1, 32, 16, 256]` | RoPE freq broadcast across 16 heads; inner 256 aligned |
| `EXP-05` | `test_expand_1d_bias_to_3d` | `[2816]` (all dtypes) | `x.expand(2, 32, 2816)` | `[2, 32, 2816]` | 1D bias/scale broadcast; inner 2816 aligned |
| `EXP-06` | `test_expand_minus_one_preserve_dims` | `[2, 1, 64, 256]` (all dtypes) | `x.expand(-1, 16, -1, -1)` | `[2, 16, 64, 256]` | `-1` keeps existing dims; head broadcast only; inner 256 aligned |
| `EXP-07` | `test_expand_batch_broadcast_aligned` | `[1, 32, 2816]` (all dtypes) | `x.expand(4, 32, 2816)` | `[4, 32, 2816]` | Batch broadcast 1→4; inner 2816 aligned |

### Aligned inner dim — Spyre Linear lowering patterns

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Source / Stick Note |
|---|---|---|---|---|---|
| `EXP-08` | `test_expand_spyre_input_same_shape` | `[1, 64, 2816]` (all dtypes) | `x.expand(1, 64, 2816)` | `[1, 64, 2816]` | **Spyre trace.** Same-shape expand on input for bmm; inner 2816 aligned |
| `EXP-09` | `test_expand_spyre_weight_q_same_shape` | `[1, 2816, 4096]` (all dtypes) | `x.expand(1, 2816, 4096)` | `[1, 2816, 4096]` | **Spyre trace.** Same-shape weight expand for q_proj bmm; inner 4096 aligned |
| `EXP-10` | `test_expand_spyre_weight_kv_same_shape` | `[1, 2816, 2048]` (all dtypes) | `x.expand(1, 2816, 2048)` | `[1, 2816, 2048]` | **Spyre trace.** Same-shape weight expand for k/v_proj bmm; inner 2048 aligned |
| `EXP-11` | `test_expand_spyre_weight_mlp_same_shape` | `[1, 2816, 2112]` (all dtypes) | `x.expand(1, 2816, 2112)` | `[1, 2816, 2112]` | **Spyre trace.** Same-shape weight expand for MLP gate/up bmm; inner 2112 aligned |

### Unaligned inner dim

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `EXP-12` | `test_expand_batch_unaligned_33` | `[1, 32, 33]` (all dtypes) | `x.expand(4, 32, 33)` | `[4, 32, 33]` | Batch broadcast; inner 33 < 64, unaligned |
| `EXP-13` | `test_expand_1d_bias_unaligned` | `[70]` (all dtypes) | `x.expand(2, 32, 70)` | `[2, 32, 70]` | 1D bias broadcast; inner 70: 70%64=6, unaligned |
| `EXP-14` | `test_expand_gqa_unaligned_65` | `[1, 8, 1, 32, 65]` (all dtypes) | `x.expand(1, 8, 2, 32, 65)` | `[1, 8, 2, 32, 65]` | GQA 2x; inner 65: 65%64=1, unaligned (just above one stick) |
| `EXP-15` | `test_expand_mask_unaligned_seq` | `[1, 1, 33, 33]` (all dtypes) | `x.expand(2, 8, 33, 33)` | `[2, 8, 33, 33]` | Mask broadcast with S=33; inner 33 unaligned |
| `EXP-16` | `test_expand_rope_freq_unaligned` | `[1, 32, 1, 70]` (all dtypes) | `x.expand(1, 32, 16, 70)` | `[1, 32, 16, 70]` | RoPE-style broadcast; inner 70 unaligned |
| `EXP-17` | `test_expand_odd_batch_heads` | `[1, 1, 33, 256]` (all dtypes) | `x.expand(3, 7, 33, 256)` | `[3, 7, 33, 256]` | Non-power-of-2 B=3, H=7; S=33 unaligned; inner 256 aligned |
| `EXP-18` | `test_expand_minus_one_unaligned` | `[2, 1, 70]` (all dtypes) | `x.expand(-1, 16, -1)` | `[2, 16, 70]` | `-1` dims; inner 70 unaligned |
| `EXP-19` | `test_expand_spyre_input_unaligned_seq` | `[1, 65, 2816]` (all dtypes) | `x.expand(1, 65, 2816)` | `[1, 65, 2816]` | Same-shape expand on input at S=65; middle dim unaligned, inner 2816 aligned |

### Size-1 dimension cases

| ID | Test Name | Input Shape & Dtype | Op Pattern | Output Shape | Stick Note |
|---|---|---|---|---|---|
| `EXP-20` | `test_expand_scalar_to_4d_aligned` | `[]` (all dtypes) | `x.expand(1, 16, 32, 256)` | `[1, 16, 32, 256]` | 0D scalar broadcast; inner 256 aligned |
| `EXP-21` | `test_expand_scalar_to_4d_unaligned` | `[]` (all dtypes) | `x.expand(1, 16, 32, 70)` | `[1, 16, 32, 70]` | 0D scalar broadcast; inner 70 unaligned |
| `EXP-22` | `test_expand_1d_size1_to_3d` | `[1]` (all dtypes) | `x.expand(2, 32, 1)` | `[2, 32, 1]` | Size-1 vector broadcast; result inner=1, ultra-unaligned |
| `EXP-23` | `test_expand_2d_size1_inner_to_aligned` | `[32, 1]` (all dtypes) | `x.expand(32, 2816)` | `[32, 2816]` | Trailing size-1 expanded to aligned 2816 |
| `EXP-24` | `test_expand_routing_weight_trailing` | `[32, 8, 1]` (all dtypes) | `x.expand(32, 8, 2816)` | `[32, 8, 2816]` | MoE routing weight trailing-1 broadcast (mirrors MOD-15 but as expand); inner 2816 aligned |
| `EXP-25` | `test_expand_5d_block_cache_aligned` | `[16, 1, 4, 16, 256]` (all dtypes) | `x.expand(16, 8, 4, 16, 256)` | `[16, 8, 4, 16, 256]` | 5D paged cache head broadcast; inner 256 aligned |
| `EXP-26` | `test_expand_5d_block_cache_unaligned` | `[16, 1, 4, 16, 13]` (all dtypes) | `x.expand(16, 8, 4, 16, 13)` | `[16, 8, 4, 16, 13]` | 5D cache; inner 13 < 64, unaligned |
| `EXP-27` | `test_expand_size1_all_broadcast_dims` | `[1, 1, 1, 256]` (all dtypes) | `x.expand(4, 16, 32, 256)` | `[4, 16, 32, 256]` | All dims except inner are size-1; inner 256 aligned |
| `EXP-28` | `test_expand_gqa_mqa_16x` | `[1, 1, 1, 256]` (all dtypes) | `x.expand(1, 16, 32, 256)` | `[1, 16, 32, 256]` | MQA (KV=1) 16x GQA broadcast; inner 256 aligned |

---

## 7. Coverage Summary Matrix

| Section | Op | Dim Range | Gemma-Traced Shapes | Spyre-Traced Shapes | Aligned-Stick | Unaligned-Stick | Size-1-Dim | Total |
|---|---|---|---|---|---|---|---|---|
| §1 Module Level | all 4 | 2D–5D | 14 | 2 | — | — | — | **16** |
| §2 `reshape`/`view` | `view`/`reshape` | 0D–6D | 9 | 1 | 12 | 8 | 6 | **30** |
| §3 `squeeze` | `squeeze` | 0D–6D | 0 | 0 | 12 | 7 | 5 | **24** |
| §4 `unsqueeze` | `unsqueeze` | 0D–6D | 9 | 4 | 9 | 7 | 8 | **28** |
| §5 `expand` | `expand` | 0D–5D | 7 | 4 | 11 | 8 | 9 | **28** |
| **Total** | | **0D–6D** | **39** | **11** | **44** | **30** | **28** | **126** |
