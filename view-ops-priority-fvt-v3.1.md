# Functional Verification Test Plan — Priority View Operations
## `reshape` · `squeeze` · `unsqueeze` · `expand`

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Priority View Operations on torch-spyre |
| **Version** | 3.1 |
| **Status** | Under Review |
| **Date** | 2025-07-14 |
| **Author** | Stuti Sinha |
| **Reviewers** | Kanaka Pathivada, Meena Soundararajan, Albin Joy |
| **Scope** | Text-only Gemma-4 modules (vision tower excluded). Inference-only `torch.compile`. Static concrete dimensions 0D–6D. Default compiler env-var values only. |
| **Sources** | `gemma4_26b_module_operator_report_v2.html` (CPU trace, seq=32), `gemma4-view-ops-aten.html` (CPU ATen layer), `gemma4_fx_module_trace_v2.html` (Spyre compile trace, seq=64) |

**Changes from v3.0:**
- Section 1 op subheadings removed; tests grouped by module only
- Section 2 restructured to match §x.1 CPU / §x.2 Spyre / §x.3 Unaligned / §x.4 Size-1 format (same as §4 and §5)
- Missing CPU trace module test added: bmm-operand `expand→view` pattern from `Gemma4TextAttention`
- All 0D–6D rank gaps filled for all four ops across all four subsection types
- Section title "Size-1 trailing dimension" corrected to "Tensors containing size-1 dimensions"
- §2.2 / §3.2 / §4.3 / §5.3 dissolved; chain information moved into the Chain column of the relevant tables
- EXP-09 (`expand→clone`) removed as duplicate of MOD-17/18; EXP-10/11 promoted into §5.1

---

## 1. Scope & Strategy

### Key Guidelines

1. **Grounded in real traces.** Every module-level test derives from a confirmed ATen op in at least one of the three source HTML files. Shapes are taken directly from those traces.
2. **No vision modules.** The vision tower had zero ATen activations in both traces (excluded entirely).
3. **Two levels of testing:**
   - **Section 1 (Module Level):** Each test reproduces the exact ATen op chain from a real Gemma-4 sub-module. All are `bfloat16`. The view-op chain executes exactly as in production.
   - **Sections 2–5 (Operation Level):** Standalone tests of each op across 0D–6D, all three float dtypes, and all stick-alignment scenarios. These test **chained sequences** of 2+ view ops — single-op coverage is already provided by `test_inductor_ops.py`.
4. **Default compiler environment variables only.**

| Variable | Source File | Default | Relevance to view ops |
|---|---|---|---|
| `SENCORES` | `config.py` | **32** | No view-op correctness impact |
| `LX_PLANNING` | `config.py` | **1 (True)** | Kernel scheduling |
| `CO_OPTIMIZING_LX_PLANNING` | `config.py` | **0 (False)** | No view-op impact |
| `GLOBAL_STICK_OPTIMIZER` | `config.py` | **1 (True)** | Stick optimizer for non-contiguous views — must be on |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | `fusion.py` | **True** | View + pointwise fusion |

> **Note:** Previous FVT versions listed SENCORES=1 and LX_PLANNING=0. Both were incorrect. Correct defaults from source are SENCORES=32 and LX_PLANNING=1.

5. **Two model architecture variants.** The 26B-A4B-it text stack has exactly two structural layer types producing distinct view-op shapes. Both are covered throughout.

| Variant | H | KV | D | GQA factor |
|---|---|---|---|---|
| **Sliding-attention** (25 layers) | 16 | 8 | 256 | 2× |
| **Full-attention** (5 layers) | 16 | 2 | 512 | 8× |

6. **Three dtypes:** `bfloat16` only for module-level. `float16`, `bfloat16`, `float32` for operation-level.
7. **Pass criteria for all tests:** `compare_with_cpu(fn, *inputs, run_eager=True)` — values must match CPU, not just shape.

---

## 2. Stick Alignment Reference

Spyre stores tensors in 128-byte memory units called **sticks**. The **innermost (last) tensor dimension** is the stick dimension.

| Dtype | Bytes/elem | Elements per stick |
|---|---|---|
| `float16` / `bfloat16` | 2 B | **64** |
| `float32` | 4 B | **32** |

**Aligned:** innermost dim divisible by 64 (fp16/bf16) or 32 (fp32).
Examples from traces: 32, 64, 128, 256, 512, 704, 2048, 2112, 2816, 4096.

**Unaligned — two sub-cases:**
- *Below one stick:* dim < 64 (fp16/bf16) or < 32 (fp32). Examples: 1, 7, 13, 30.
- *Not divisible:* dim ≥ stick width but not a clean multiple. Examples: 33, 65, 70, 129, 257.

**Size-1 innermost dimension** is the extreme unaligned case: 1 element per stick with 63 padding slots (fp16/bf16). Arises from `unsqueeze(-1)`, `reshape(...,1)`, or after `squeeze` removes all non-singleton dims.

**How to produce unaligned shapes:** choose tensor shapes with a non-multiple innermost dimension. No special flags are needed.

---

## 3. ATen Op Inventory from Traces

### 3.1 CPU Trace (`gemma4_26b_module_operator_report_v2.html`, seq=32)

| ATen Op | Module | Shapes (input→output) |
|---|---|---|
| `aten.view.default` | `Gemma4TextDecoderLayer` | `[1,32,2816]→[32,2816]` and reverse |
| `aten.view.default` | `Gemma4TextAttention` | `[1,32,4096]→[1,32,16,256]`; `[1,32,2048]→[1,32,8,256]`; `[1,32,16,256]→[1,32,4096]` |
| `aten.view.default` | `Gemma4TextAttention` (bmm prep) | `[1,16,32,256]→[16,32,256]`; `[1,16,256,32]→[16,256,32]`; `[1,16,32,32]→[16,32,32]` |
| `aten.view.default` | `Gemma4TextMoE` | `[32,8]→[256]`; `[256,2816]→[32,8,2816]` |
| `aten.unsqueeze.default` | `Gemma4Model` (mask chain) | `[32]→[1,32]→[1,1,32]→[1,1,32,1]` (4 steps) |
| `aten.unsqueeze.default` | `Gemma4TextAttention` (RoPE) | `[1,32,256]→[1,32,1,256]` (×2, for cos and sin) |
| `aten.unsqueeze.default` | `Gemma4TextAttention` (GQA) | `[1,8,32,256]→[1,8,1,32,256]` (×2, K and V) |
| `aten.unsqueeze.default` | `Gemma4TextMoE` (routing) | `[256]→[256,1]` |
| `aten.expand.default` | `Gemma4Model` (mask) | `[1,1,32,32]→expand(1,-1,32,32)→[1,1,32,32]` (same-shape) |
| `aten.expand.default` | `Gemma4TextAttention` (GQA) | `[1,8,1,32,256]→[1,8,2,32,256]→clone` (×2); `[1,2,1,32,512]→[1,2,8,32,512]→clone` (×2) |
| `aten.expand.default` | `Gemma4TextAttention` (bmm prep) | `[1,16,32,256]→expand(1,16,32,256)→view([16,32,256])`; `[1,16,256,32]→expand→view` |
| `aten.squeeze` | — | **Not observed** |

### 3.2 Spyre Compile Trace (`gemma4_fx_module_trace_v2.html`, seq=64)

| ATen Op | Module | Shapes (input→output) | Note |
|---|---|---|---|
| `aten.view.default` | `Gemma4Attention` (QKV split) | `[1,64,4096]→[1,64,16,256]`; `[1,64,2048]→[1,64,8,256]` | seq=64 |
| `aten.view.default` | `Gemma4TextRotaryEmbedding` | `[1,64,16,256]→[1,64,16,2,128]`; `[1,64,16,2,1,128]→[1,64,16,256]` | 4D→5D and back; 6D→4D |
| `aten.unsqueeze.default` | All `Linear` (q/k/v/o/gate/up/down proj) | `[2816,4096]→[1,2816,4096]`; `[2816,2048]→[1,2816,2048]`; `[2816,2112]→[1,2816,2112]` | Spyre lowering of `F.linear` |
| `aten.unsqueeze.default` | `Gemma4TextRotaryEmbedding` | `[1,64,16,2,128]→[1,64,16,1,2,128]` (dim=-3) | 5D→6D RoPE |
| `aten.unsqueeze.default` | `Gemma4TextMoE` | `[1,8]→[1,8,1]` (dim=2) then `expand→[1,8,64]` | MoE token dispatch |
| `aten.expand.default` | All `Linear` | `[1,64,2816]→expand(1,64,2816)`; `[1,2816,4096]→expand(1,2816,4096)` | Same-shape bmm operand alignment |
| `aten.expand.default` | `Gemma4TextMoE` | `[1,1,2816]→expand(1,8,2816)` | Token broadcast after `unsqueeze(1)` |

---

## 4. Gemma-4 Text Modules Under Test

| # | Module path | Class | Ops exercised | Seq in trace |
|---|---|---|---|---|
| 1 | `model` | `Gemma4Model` | `unsqueeze`×3 (mask chain), `expand` (mask same-shape) | CPU seq=32 |
| 2 | `model.language_model.layers.N` | `Gemma4TextDecoderLayer` | `view` (token flatten + unflatten) | CPU seq=32 |
| 3 | `model.language_model.layers.N.self_attn` | `Gemma4TextAttention` | `view` (QKV split, head merge), `unsqueeze`+`expand`+`clone` (GQA), `expand`+`view` (bmm prep) | CPU seq=32 |
| 4 | `model.language_model.layers.N.{q,k,v,o}_proj` | `Linear` (Spyre lowering) | `unsqueeze(0)` (weight), `expand` (input + weight for bmm) | Spyre seq=64 |
| 5 | `model.language_model.layers.N.mlp.{gate,up,down}_proj` | `Linear` (Spyre lowering) | `unsqueeze(0)`, `expand` | Spyre seq=64 |
| 6 | `model.language_model.layers.N.moe_block` | `Gemma4TextMoE` | `view` (index flatten/unflatten), `unsqueeze`+`expand` (routing weight + token dispatch) | CPU+Spyre |
| 7 | `model.language_model.rotary_emb` | `Gemma4TextRotaryEmbedding` | `view` (4D↔5D/6D), `unsqueeze` (5D→6D) | Spyre seq=64 |

---

## Section 1 — Module-Level Tests

Each test reproduces the exact ATen op chain from the named module. All use `bfloat16`.
**Pass criteria:** `compare_with_cpu(fn, *inputs, run_eager=True)`, `atol=1e-2`.

| ID | Test Name | Module | Source | ATen Chain | Input Shape(s) | Output Shape | Test Steps |
|---|---|---|---|---|---|---|---|
| `MOD-01` | `test_mod_decoder_token_flatten` | `Gemma4TextDecoderLayer` | CPU seq=32 | `view(-1,2816)` | `[1,32,2816]` bf16 | `[32,2816]` | `x=randn([1,32,2816],bf16)` → `compare_with_cpu(lambda t: t.view(-1,2816), x)` |
| `MOD-02` | `test_mod_decoder_token_unflatten` | `Gemma4TextDecoderLayer` | CPU seq=32 | `view(1,32,2816)` | `[32,2816]` bf16 | `[1,32,2816]` | `x=randn([32,2816],bf16)` → `compare_with_cpu(lambda t: t.view(1,32,2816), x)` |
| `MOD-03` | `test_mod_attn_qsplit_sliding` | `Gemma4TextAttention` | CPU seq=32 | `view(1,32,-1,256)` | `[1,32,4096]` bf16 | `[1,32,16,256]` | `compare_with_cpu(lambda t: t.view(1,32,-1,256), x)` |
| `MOD-04` | `test_mod_attn_kvsplit_sliding` | `Gemma4TextAttention` | CPU seq=32 | `view(1,32,-1,256)` | `[1,32,2048]` bf16 | `[1,32,8,256]` | `compare_with_cpu(lambda t: t.view(1,32,-1,256), x)` |
| `MOD-05` | `test_mod_attn_qsplit_spyre_s64` | `Gemma4Attention` | Spyre seq=64 | `view(1,64,-1,256)` | `[1,64,4096]` bf16 | `[1,64,16,256]` | `compare_with_cpu(lambda t: t.view(1,64,-1,256), x)` |
| `MOD-06` | `test_mod_attn_head_merge` | `Gemma4TextAttention` | CPU seq=32 | `reshape(1,32,-1)` | `[1,32,16,256]` bf16 | `[1,32,4096]` | `compare_with_cpu(lambda t: t.reshape(1,32,-1), x)` |
| `MOD-07` | `test_mod_attn_bmm_prep_expand_view` | `Gemma4TextAttention` | CPU seq=32 | `expand(1,16,32,256)→view([16,32,256])` | `[1,16,32,256]` bf16 | `[16,32,256]` | `compare_with_cpu(lambda t: t.expand(1,16,32,256).view(16,32,256), x)` |
| `MOD-08` | `test_mod_rope_view_chain` | `Gemma4TextRotaryEmbedding` | Spyre seq=64 | `view(5D)→unsqueeze(-3)→mul→view(4D)` | `[1,64,16,256]` bf16 | `[1,64,16,256]` | `x=randn([1,64,16,256],bf16); freq=randn([1,64,1,2,128],bf16)` → `compare_with_cpu(lambda t: t.view(1,64,16,2,128).unsqueeze(-3).mul(freq).view(1,64,16,256), x)` |
| `MOD-09` | `test_mod_moe_index_flatten` | `Gemma4TextMoE` | CPU seq=32 | `view(-1)` | `[32,8]` int64 | `[256]` | `x=torch.arange(256,dtype=torch.int64).view(32,8)` → `compare_with_cpu(lambda t: t.view(-1), x)` |
| `MOD-10` | `test_mod_moe_output_unflatten` | `Gemma4TextMoE` | CPU seq=32 | `view(32,8,2816)` | `[256,2816]` bf16 | `[32,8,2816]` | `compare_with_cpu(lambda t: t.view(32,8,2816), x)` |
| `MOD-11` | `test_mod_mask_unsqueeze_chain` | `Gemma4Model` | CPU seq=32 | `unsqueeze(0)→unsqueeze(1)→unsqueeze(3)` | `[32]` int64 | `[1,1,32,1]` | `x=torch.arange(32,dtype=torch.int64)` → `compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(1).unsqueeze(3), x)` |
| `MOD-12` | `test_mod_rope_unsqueeze_freq` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)` | `[1,32,256]` bf16 | `[1,32,1,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `MOD-13` | `test_mod_gqa_unsqueeze_sliding` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)` | `[1,8,32,256]` bf16 | `[1,8,1,32,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `MOD-14` | `test_mod_moe_routing_unsqueeze` | `Gemma4TextMoE` | CPU+Spyre | `unsqueeze(-1)` | `[256]` bf16 | `[256,1]` | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` |
| `MOD-15` | `test_mod_linear_weight_unsqueeze` | `Linear` (Spyre) | Spyre seq=64 | `unsqueeze(0)` | `[4096,2816]` bf16 | `[1,4096,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `MOD-16` | `test_mod_rope_unsqueeze_6d` | `Gemma4TextRotaryEmbedding` | Spyre seq=64 | `view(5D)→unsqueeze(-3)` | `[1,64,16,2,128]` bf16 | `[1,64,16,1,2,128]` | `compare_with_cpu(lambda t: t.unsqueeze(-3), x)` |
| `MOD-17` | `test_mod_mask_expand` | `Gemma4Model` | CPU seq=32 | `expand(1,-1,32,32)` (same-shape) | `[1,1,32,32]` bf16 | `[1,1,32,32]` | `compare_with_cpu(lambda t: t.expand(1,-1,32,32), x)` |
| `MOD-18` | `test_mod_gqa_expand_clone_sliding` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)→expand(1,8,2,32,256)→clone` | `[1,8,32,256]` bf16 | `[1,8,2,32,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,8,2,32,256).clone(), x)` |
| `MOD-19` | `test_mod_gqa_expand_clone_full_attn` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)→expand(1,2,8,32,512)→clone` | `[1,2,32,512]` bf16 | `[1,2,8,32,512]` | `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,2,8,32,512).clone(), x)` |
| `MOD-20` | `test_mod_linear_unsqueeze_expand_bmm` | `Linear` (Spyre) | Spyre seq=64 | `w.unsqueeze(0).expand(1,2816,4096)` then `bmm(x,w_e)` | weight `[2816,4096]` + input `[1,64,2816]` bf16 | `[1,64,4096]` | `w=randn([2816,4096],bf16); x=randn([1,64,2816],bf16)` → `compare_with_cpu(lambda w,x: torch.bmm(x,w.unsqueeze(0).expand(1,2816,4096)), w, x, atol=0.5, rtol=0.1)` |
| `MOD-21` | `test_mod_moe_token_unsqueeze_expand` | `Gemma4TextMoE` | Spyre seq=64 | `unsqueeze(1)→expand(1,8,2816)` | `[1,2816]` bf16 | `[1,8,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(1).expand(1,8,2816), x)` |

---

## Section 2 — `reshape` / `view` — Operation Level

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

> **Note on "aligned" / "unaligned" in subsection headings:** For `reshape`, what matters is the alignment of the **output** innermost dimension — the one the compiler will assign as the stick dimension after the reshape. The input innermost alignment only appears as a label in the Stick note column when it differs from the output. This is consistent with §3 (squeeze), §4 (unsqueeze), and §5 (expand), which all label by output alignment.

### 2.1 Gemma CPU-traced shapes (aligned)

Shapes taken directly from `gemma4_26b_module_operator_report_v2.html`.

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `RSH-01` | `test_reshape_qsplit_sliding_s32` | `[1,32,4096]` (all dtypes) | `view(1,32,-1,256)` | `[1,32,16,256]` | `compare_with_cpu(lambda t: t.view(1,32,-1,256), x)` |
| `RSH-02` | `test_reshape_kvsplit_sliding_s32` | `[1,32,2048]` (all dtypes) | `view(1,32,-1,256)` | `[1,32,8,256]` | `compare_with_cpu(lambda t: t.view(1,32,-1,256), x)` |
| `RSH-03` | `test_reshape_qsplit_full_attn` | `[1,32,8192]` (all dtypes) | `view(1,32,16,512)` | `[1,32,16,512]` | `compare_with_cpu(lambda t: t.view(1,32,16,512), x)` |
| `RSH-04` | `test_reshape_head_merge_sliding` | `[1,32,16,256]` (all dtypes) | `reshape(1,32,-1)` | `[1,32,4096]` | `compare_with_cpu(lambda t: t.reshape(1,32,-1), x)` |
| `RSH-05` | `test_reshape_token_flatten_3d_2d` | `[1,32,2816]` (all dtypes) | `view(-1,2816)` | `[32,2816]` | `compare_with_cpu(lambda t: t.view(-1,2816), x)` |
| `RSH-06` | `test_reshape_token_unflatten_2d_3d` | `[32,2816]` (all dtypes) | `view(1,32,2816)` | `[1,32,2816]` | `compare_with_cpu(lambda t: t.view(1,32,2816), x)` |
| `RSH-07` | `test_reshape_moe_index_2d_1d` | `[32,8]` int64 | `view(-1)` | `[256]` | `x=torch.arange(256,dtype=torch.int64).view(32,8)` → `compare_with_cpu(lambda t: t.view(-1), x)` |
| `RSH-08` | `test_reshape_moe_output_2d_3d` | `[256,2816]` (all dtypes) | `view(32,8,2816)` | `[32,8,2816]` | `compare_with_cpu(lambda t: t.view(32,8,2816), x)` |
| `RSH-09` | `test_reshape_round_trip_flatten_split` | `[1,32,16,256]` (all dtypes) | `reshape(1,32,-1).reshape(1,32,16,256)` | `[1,32,16,256]` | `compare_with_cpu(lambda t: t.reshape(1,32,-1).reshape(1,32,16,256), x)` — tests that stick layout is preserved through a round-trip |

### 2.2 Spyre-traced shapes (aligned)

Shapes taken directly from `gemma4_fx_module_trace_v2.html`.

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `RSH-10` | `test_reshape_qsplit_spyre_s64` | `[1,64,4096]` (all dtypes) | `view(1,64,-1,256)` | `[1,64,16,256]` | `compare_with_cpu(lambda t: t.view(1,64,-1,256), x)` |
| `RSH-11` | `test_reshape_kvsplit_spyre_s64` | `[1,64,2048]` (all dtypes) | `view(1,64,-1,256)` | `[1,64,8,256]` | `compare_with_cpu(lambda t: t.view(1,64,-1,256), x)` |
| `RSH-12` | `test_reshape_rope_4d_to_5d` | `[1,64,16,256]` (all dtypes) | `view(1,64,16,2,128)` | `[1,64,16,2,128]` | `compare_with_cpu(lambda t: t.view(1,64,16,2,128), x)` — RoPE internal split of head dim |
| `RSH-13` | `test_reshape_rope_6d_to_4d` | `[1,64,16,2,1,128]` (all dtypes) | `view(1,64,16,256)` | `[1,64,16,256]` | `compare_with_cpu(lambda t: t.view(1,64,16,256), x)` — RoPE 6D→4D back |
| `RSH-14` | `test_reshape_inferred_dim_aligned` | `[2,32,16,256]` (all dtypes) | `reshape(2,-1,256)` | `[2,512,256]` | `compare_with_cpu(lambda t: t.reshape(2,-1,256), x)` — `-1` inference |

### 2.3 Unaligned innermost dimension

Covers all ranks 1D–6D with non-multiple innermost dimensions.

| ID | Test Name | Input | Chain | Output | Stick note | Test Steps |
|---|---|---|---|---|---|---|
| `RSH-15` | `test_reshape_unaligned_1d_to_2d` | `[520]` (all dtypes) | `reshape(8,65)` | `[8,65]` | inner 65: 65%64=1 | `compare_with_cpu(lambda t: t.reshape(8,65), x)` |
| `RSH-16` | `test_reshape_unaligned_2d_to_1d` | `[8,65]` (all dtypes) | `reshape(-1)` | `[520]` | input inner 65 unaligned | `compare_with_cpu(lambda t: t.reshape(-1), x)` |
| `RSH-17` | `test_reshape_unaligned_3d_to_2d_33` | `[4,8,33]` (all dtypes) | `reshape(32,33)` | `[32,33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.reshape(32,33), x)` |
| `RSH-18` | `test_reshape_unaligned_3d_to_4d_260` | `[1,32,2080]` (all dtypes) | `reshape(1,32,8,260)` | `[1,32,8,260]` | output inner 260: 260%64=4 | `compare_with_cpu(lambda t: t.reshape(1,32,8,260), x)` |
| `RSH-19` | `test_reshape_unaligned_4d_head_merge_70` | `[2,8,32,70]` (all dtypes) | `reshape(2,32,560)` | `[2,32,560]` | input inner 70; output 560: 560%64=48 | `compare_with_cpu(lambda t: t.reshape(2,32,560), x)` |
| `RSH-20` | `test_reshape_unaligned_4d_to_3d_129` | `[1,32,8,129]` (all dtypes) | `reshape(1,32,1032)` | `[1,32,1032]` | inner 129: 129%64=1 | `compare_with_cpu(lambda t: t.reshape(1,32,1032), x)` |
| `RSH-21` | `test_reshape_unaligned_5d_to_3d` | `[1,2,4,8,7]` (all dtypes) | `reshape(1,8,56)` | `[1,8,56]` | input inner 7 < 64; output 56: 56%64≠0 | `compare_with_cpu(lambda t: t.reshape(1,8,56), x)` |
| `RSH-22` | `test_reshape_unaligned_6d_to_3d` | `[1,2,4,8,4,7]` (all dtypes) | `reshape(1,8,224)` | `[1,8,224]` | input inner 7; output 224: 224%64=32 | `compare_with_cpu(lambda t: t.reshape(1,8,224), x)` |

### 2.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Chain | Output | Notes | Test Steps |
|---|---|---|---|---|---|---|
| `RSH-23` | `test_reshape_0d_scalar_to_1d` | `[]` (all dtypes) | `reshape(1)` | `[1]` | 0D→1D; result inner=1 | `x=torch.tensor(1.0,dtype=dtype)` → `compare_with_cpu(lambda t: t.reshape(1), x)` |
| `RSH-24` | `test_reshape_1d_size1_to_3d` | `[1]` (all dtypes) | `reshape(1,1,1)` | `[1,1,1]` | All dims=1 | `compare_with_cpu(lambda t: t.reshape(1,1,1), x)` |
| `RSH-25` | `test_reshape_2d_remove_trailing_size1` | `[128,1]` (all dtypes) | `reshape(128)` | `[128]` | Remove trailing size-1; new inner 128 aligned | `compare_with_cpu(lambda t: t.reshape(128), x)` |
| `RSH-26` | `test_reshape_3d_trailing_size1_to_2d` | `[4,32,1]` (all dtypes) | `reshape(4,32)` | `[4,32]` | Remove trailing size-1; new inner 32 | `compare_with_cpu(lambda t: t.reshape(4,32), x)` |
| `RSH-27` | `test_reshape_4d_two_size1_to_2d` | `[1,8,1,256]` (all dtypes) | `reshape(8,256)` | `[8,256]` | Two non-trailing size-1 dims removed | `compare_with_cpu(lambda t: t.reshape(8,256), x)` |
| `RSH-28` | `test_reshape_4d_size1_trailing_to_3d` | `[2,8,32,1]` (all dtypes) | `reshape(2,8,32)` | `[2,8,32]` | Trailing size-1 removed; new inner 32 | `compare_with_cpu(lambda t: t.reshape(2,8,32), x)` |
| `RSH-29` | `test_reshape_5d_size1_inner_to_4d` | `[1,2,1,8,256]` (all dtypes) | `reshape(1,2,8,256)` | `[1,2,8,256]` | Middle size-1 removed | `compare_with_cpu(lambda t: t.reshape(1,2,8,256), x)` |
| `RSH-30` | `test_reshape_6d_size1_trailing_to_5d` | `[1,2,4,8,16,1]` (all dtypes) | `reshape(1,2,4,8,16)` | `[1,2,4,8,16]` | 6D trailing size-1 removed; new inner 16 | `compare_with_cpu(lambda t: t.reshape(1,2,4,8,16), x)` |

### 2.5 Restickification scenarios — view op as trigger

These tests exercise the cases where a view op causes the Spyre compiler to insert a **restickify** (a device-side data-movement operation) before the consuming op. A restickify is inserted when the layout propagated through the view is incompatible with what the next op's hardware constraint requires. Source: `propagate_layouts.py`, `insert_restickify.py`.

All four tests use `bfloat16` as the primary dtype; results are verified with `compare_with_cpu`.

| ID | Test Name | Restickify scenario | Chain | Input Shape(s) | Test Steps |
|---|---|---|---|---|---|
| `RSH-31` | `test_restickify_reshape_then_bmm` | **Scenario 1:** `reshape` changes which dim is K; matmul requires stick on K | `reshape(B*H, S, D) → bmm(reshaped, weight_T)` | x `[2,32,16,256]` bf16 + weight `[256,256]` bf16 | `x=randn([2,32,16,256],bf16); w=randn([256,256],bf16)` → `compare_with_cpu(lambda x,w: torch.bmm(x.reshape(64,16,256), w.unsqueeze(0).expand(64,256,256)), x, w, atol=0.5, rtol=0.1)` — restickify injected before bmm when reshape leaves stick on wrong dim |
| `RSH-32` | `test_restickify_reshape_then_add_incompatible_layout` | **Scenario 3:** two view-derived tensors with incompatible stick layouts fed into pointwise add | `x_reshaped + x_original` where reshape changed which dim is innermost | x `[4,128,64]` bf16 | `x=randn([4,128,64],bf16)` → `compare_with_cpu(lambda t: t.reshape(4,64,128) + t, x)` — `t.reshape(4,64,128)` has inner 128 while `t` has inner 64; compiler must unify stick layouts before add |
| `RSH-33` | `test_restickify_reshape_then_dtype_cast` | **Scenario 4:** view leaves non-offset-free layout; dtype cast across stick-width boundary triggers restickify | `reshape` then `.to(torch.float32)` on unaligned fp16 tensor | `[1,32,65]` fp16 | `x=randn([1,32,65],torch.float16)` → `compare_with_cpu(lambda t: t.reshape(32,65).to(torch.float32), x)` — 65%32=1 for fp32 stick; restickify required before cast |
| `RSH-34` | `test_restickify_reshape_then_layernorm` | **Scenario 5:** reshape leaves stick on wrong dim; layer norm requires stick on last (normalization) dim | `reshape(B*S, D) → F.layer_norm([D])` | `[1,32,2816]` bf16 | `x=randn([1,32,2816],bf16)` → `compare_with_cpu(lambda t: torch.nn.functional.layer_norm(t.reshape(32,2816), [2816]), x)` — if reshape leaves stick on non-last dim, restickify inserted before layer_norm |

---

## Section 3 — `squeeze` — Operation Level

`squeeze` was not observed in either Gemma ATen trace. All tests are standalone.
**Pass criteria:** `compare_with_cpu`, `atol=0.0`

> **Note on "aligned result" / "unaligned result" headings:** For `squeeze`, what matters is the alignment of the **output** innermost dimension — i.e., the dim that becomes the stick dimension after the op. When squeezing a **non-trailing** dim the innermost is unchanged (output = input innermost). When squeezing the **trailing** dim the output innermost is the second-to-last dim of the input. The headings "aligned result after squeeze" and "unaligned result after squeeze" capture this by labelling the output, not the input. The same logic applies to `unsqueeze` (§4) and `expand` (§5): both section headings describe the output innermost dimension. For `unsqueeze(-1)`, the output innermost is always 1 (ultra-unaligned) regardless of input, captured in §4.4. For `expand` of a size-1 innermost, the output innermost changes from 1 to the broadcast size, captured in §5.4.

### 3.1 No-op cases (dim is not size-1 — verifies no-op path)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `SQZ-01` | `test_squeeze_noop_1d_aligned` | `[256]` (all dtypes) | `squeeze(0)` | `[256]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` |
| `SQZ-02` | `test_squeeze_noop_1d_unaligned` | `[63]` (all dtypes) | `squeeze(0)` | `[63]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` |
| `SQZ-03` | `test_squeeze_noop_4d_no_singletons` | `[2,16,32,256]` (all dtypes) | `squeeze()` | `[2,16,32,256]` | `compare_with_cpu(lambda t: t.squeeze(), x)` |

### 3.2 Aligned result after squeeze

Covers 1D–6D inputs producing aligned innermost dim after squeeze.

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `SQZ-04` | `test_squeeze_2d_leading_to_1d_aligned` | `[1,256]` (all dtypes) | `squeeze(0)` | `[256]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` |
| `SQZ-05` | `test_squeeze_3d_batch_to_2d_aligned` | `[1,32,2816]` (all dtypes) | `squeeze(0)` | `[32,2816]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` — Gemma hidden dim |
| `SQZ-06` | `test_squeeze_3d_mid_decode_aligned` | `[2,1,256]` (all dtypes) | `squeeze(1)` | `[2,256]` | `compare_with_cpu(lambda t: t.squeeze(1), x)` — remove seq=1 decode step |
| `SQZ-07` | `test_squeeze_4d_decode_seq_aligned` | `[1,1,16,256]` (all dtypes) | `squeeze(1)` | `[1,16,256]` | `compare_with_cpu(lambda t: t.squeeze(1), x)` |
| `SQZ-08` | `test_squeeze_4d_tuple_dims_aligned` | `[1,16,1,256]` (all dtypes) | `squeeze((0,2))` | `[16,256]` | `compare_with_cpu(lambda t: t.squeeze((0,2)), x)` |
| `SQZ-09` | `test_squeeze_4d_trailing_to_3d_aligned` | `[32,8,64,1]` (all dtypes) | `squeeze(-1)` | `[32,8,64]` | `compare_with_cpu(lambda t: t.squeeze(-1), x)` |
| `SQZ-10` | `test_squeeze_5d_all_singletons_aligned` | `[1,1,32,1,256]` (all dtypes) | `squeeze()` | `[32,256]` | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-11` | `test_squeeze_6d_leading_to_5d_aligned` | `[1,2,4,1,8,256]` (all dtypes) | `squeeze(0)` | `[2,4,1,8,256]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` |

### 3.3 Unaligned result after squeeze

Covers 1D–6D inputs producing unaligned innermost dim after squeeze.

| ID | Test Name | Input | Chain | Output | Stick note | Test Steps |
|---|---|---|---|---|---|---|
| `SQZ-12` | `test_squeeze_2d_leading_unaligned_33` | `[1,33]` (all dtypes) | `squeeze(0)` | `[33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.squeeze(0), x)` |
| `SQZ-13` | `test_squeeze_3d_mid_unaligned_70` | `[2,1,70]` (all dtypes) | `squeeze(1)` | `[2,70]` | inner 70: 70%64=6 | `compare_with_cpu(lambda t: t.squeeze(1), x)` |
| `SQZ-14` | `test_squeeze_4d_decode_unaligned_65` | `[1,1,16,65]` (all dtypes) | `squeeze(1)` | `[1,16,65]` | inner 65: 65%64=1 | `compare_with_cpu(lambda t: t.squeeze(1), x)` |
| `SQZ-15` | `test_squeeze_4d_tuple_unaligned_33` | `[1,16,1,33]` (all dtypes) | `squeeze((0,2))` | `[16,33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.squeeze((0,2)), x)` |
| `SQZ-16` | `test_squeeze_5d_all_singletons_unaligned` | `[1,1,32,1,30]` (all dtypes) | `squeeze()` | `[32,30]` | inner 30 < 64 | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-17` | `test_squeeze_6d_leading_unaligned_13` | `[1,2,4,1,8,13]` (all dtypes) | `squeeze(0)` | `[2,4,1,8,13]` | inner 13 < 64 | `compare_with_cpu(lambda t: t.squeeze(0), x)` |

### 3.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Chain | Output | Notes | Test Steps |
|---|---|---|---|---|---|---|
| `SQZ-18` | `test_squeeze_1d_size1_to_scalar` | `[1]` (all dtypes) | `squeeze(0)` | `[]` scalar | 1D→0D | `compare_with_cpu(lambda t: t.squeeze(0), x)` |
| `SQZ-19` | `test_squeeze_2d_all_size1_to_scalar` | `[1,1]` (all dtypes) | `squeeze()` | `[]` scalar | All dims=1 | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-20` | `test_squeeze_2d_trailing_size1` | `[128,1]` (all dtypes) | `squeeze(-1)` | `[128]` | Remove trailing size-1; inner 128 aligned | `compare_with_cpu(lambda t: t.squeeze(-1), x)` |
| `SQZ-21` | `test_squeeze_3d_all_size1_to_1d` | `[1,1,256]` (all dtypes) | `squeeze()` | `[256]` | Two size-1 dims removed | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-22` | `test_squeeze_4d_mixed_size1_to_1d` | `[1,1,32,1]` (all dtypes) | `squeeze()` | `[32]` | Three size-1 removed; inner 32 | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-23` | `test_squeeze_5d_size1_inner` | `[2,1,32,1,256]` (all dtypes) | `squeeze()` | `[2,32,256]` | Two middle size-1 dims removed | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-24` | `test_squeeze_6d_size1_to_3d` | `[1,1,4,1,8,256]` (all dtypes) | `squeeze()` | `[4,8,256]` | Three size-1 removed from 6D | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-25` | `test_squeeze_chain_multi_then_reshape` | `[1,8,1,256]` (all dtypes) | `squeeze(0).squeeze(1).reshape(-1)` | `[2048]` | Multi-squeeze + reshape; tests stale-stride cleanup | `compare_with_cpu(lambda t: t.squeeze(0).squeeze(1).reshape(-1), x)` |

---

## Section 4 — `unsqueeze` — Operation Level

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### 4.1 Gemma CPU-traced shapes (aligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `UNS-01` | `test_unsqueeze_1d_pos_ids_to_2d` | `[32]` int64 | `unsqueeze(0)` | `[1,32]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` — `Gemma4Model` position ids |
| `UNS-02` | `test_unsqueeze_3d_rope_freq_to_4d` | `[1,32,256]` (all dtypes) | `unsqueeze(2)` | `[1,32,1,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` — RoPE cos/sin head axis |
| `UNS-03` | `test_unsqueeze_4d_gqa_sliding_to_5d` | `[1,8,32,256]` (all dtypes) | `unsqueeze(2)` | `[1,8,1,32,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` — GQA repeat_kv insert |
| `UNS-04` | `test_unsqueeze_4d_gqa_full_attn_to_5d` | `[1,2,32,512]` (all dtypes) | `unsqueeze(2)` | `[1,2,1,32,512]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` — full-attention D=512 |
| `UNS-05` | `test_unsqueeze_1d_moe_routing_trailing` | `[256]` (all dtypes) | `unsqueeze(-1)` | `[256,1]` | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` — MoE routing weight trailing dim; creates size-1 innermost |
| `UNS-06` | `test_unsqueeze_chain_gqa_expand_2x` | `[1,8,32,256]` (all dtypes) | `unsqueeze(2).expand(1,8,2,32,256)` | `[1,8,2,32,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,8,2,32,256), x)` — virtual expand, no clone |
| `UNS-07` | `test_unsqueeze_chain_gqa_expand_8x` | `[1,2,32,512]` (all dtypes) | `unsqueeze(2).expand(1,2,8,32,512)` | `[1,2,8,32,512]` | `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,2,8,32,512), x)` — virtual expand, no clone |

### 4.2 Spyre-traced shapes (aligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `UNS-08` | `test_unsqueeze_2d_linear_weight_q` | `[4096,2816]` (all dtypes) | `unsqueeze(0)` | `[1,4096,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` — q_proj weight |
| `UNS-09` | `test_unsqueeze_2d_linear_weight_kv` | `[2048,2816]` (all dtypes) | `unsqueeze(0)` | `[1,2048,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` — k/v_proj weight |
| `UNS-10` | `test_unsqueeze_2d_linear_weight_mlp` | `[2816,2112]` (all dtypes) | `unsqueeze(0)` | `[1,2816,2112]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` — MLP gate/up weight |
| `UNS-11` | `test_unsqueeze_5d_rope_to_6d` | `[1,64,16,2,128]` (all dtypes) | `unsqueeze(-3)` | `[1,64,16,1,2,128]` | `compare_with_cpu(lambda t: t.unsqueeze(-3), x)` — RoPE 5D→6D |
| `UNS-12` | `test_unsqueeze_chain_linear_expand_bmm` | weight `[2816,4096]`, input `[1,64,2816]` (all dtypes) | `w.unsqueeze(0).expand(1,2816,4096)` then `bmm(x,w_e)` | `[1,64,4096]` | `compare_with_cpu(lambda w,x: torch.bmm(x,w.unsqueeze(0).expand(1,2816,4096)), w, x, atol=0.5, rtol=0.1)` |
| `UNS-13` | `test_unsqueeze_chain_moe_token_broadcast` | `[1,2816]` (all dtypes) | `unsqueeze(1).expand(1,8,2816)` | `[1,8,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(1).expand(1,8,2816), x)` — MoE token dispatch |

### 4.3 Unaligned innermost dimension

Covers 0D–5D inputs (producing 1D–6D) with unaligned innermost dim.

| ID | Test Name | Input | Chain | Output | Stick note | Test Steps |
|---|---|---|---|---|---|---|
| `UNS-14` | `test_unsqueeze_0d_to_1d_unaligned` | `[]` (all dtypes) | `unsqueeze(0)` | `[1]` | 0D scalar; result inner=1 (ultra-unaligned) | `x=torch.tensor(1.5,dtype=dtype)` → `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-15` | `test_unsqueeze_1d_to_2d_unaligned_70` | `[70]` (all dtypes) | `unsqueeze(0)` | `[1,70]` | inner 70: 70%64=6 | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-16` | `test_unsqueeze_2d_to_3d_unaligned_33` | `[32,33]` (all dtypes) | `unsqueeze(0)` | `[1,32,33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-17` | `test_unsqueeze_3d_to_4d_rope_unaligned_65` | `[1,32,65]` (all dtypes) | `unsqueeze(2)` | `[1,32,1,65]` | inner 65: 65%64=1 | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `UNS-18` | `test_unsqueeze_4d_to_5d_gqa_unaligned_65` | `[2,8,32,65]` (all dtypes) | `unsqueeze(2)` | `[2,8,1,32,65]` | inner 65 | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `UNS-19` | `test_unsqueeze_5d_to_6d_unaligned_13` | `[1,2,4,8,13]` (all dtypes) | `unsqueeze(3)` | `[1,2,4,1,8,13]` | inner 13 < 64 | `compare_with_cpu(lambda t: t.unsqueeze(3), x)` |

### 4.4 Tensors containing size-1 dimensions

Covers 0D–5D inputs (producing 1D–6D). A trailing `unsqueeze(-1)` always creates a size-1 innermost dimension.

| ID | Test Name | Input | Chain | Output | Notes | Test Steps |
|---|---|---|---|---|---|---|
| `UNS-20` | `test_unsqueeze_0d_to_1d_aligned` | `[]` (all dtypes) | `unsqueeze(0)` | `[1]` | 0D→1D; result is size-1 | `x=torch.tensor(2.0,dtype=dtype)` → `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-21` | `test_unsqueeze_1d_size1_to_2d` | `[1]` (all dtypes) | `unsqueeze(0)` | `[1,1]` | Size-1 input stays size-1 | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-22` | `test_unsqueeze_2d_trailing_size1_aligned` | `[32,256]` (all dtypes) | `unsqueeze(-1)` | `[32,256,1]` | New innermost=1; preceding dim 256 aligned | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` |
| `UNS-23` | `test_unsqueeze_2d_trailing_size1_unaligned` | `[4,8,70]` (all dtypes) | `unsqueeze(-1)` | `[4,8,70,1]` | New innermost=1; preceding dim 70 unaligned | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` |
| `UNS-24` | `test_unsqueeze_3d_dual_axis` | `[32,256]` (all dtypes) | `unsqueeze(0).unsqueeze(3)` | `[1,32,256,1]` | Leading+trailing size-1 | `compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(3), x)` |
| `UNS-25` | `test_unsqueeze_mask_chain_full` | `[32]` int64 | `unsqueeze(0).unsqueeze(1).unsqueeze(3)` | `[1,1,32,1]` | Full Gemma mask build chain; trailing size-1 | `compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(1).unsqueeze(3), x)` |
| `UNS-26` | `test_unsqueeze_4d_size1_innermost_to_5d` | `[1,8,32,1]` (all dtypes) | `unsqueeze(2)` | `[1,8,1,32,1]` | Both new dim and innermost are size-1 | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `UNS-27` | `test_unsqueeze_5d_trailing_size1_to_6d` | `[1,2,4,8,256]` (all dtypes) | `unsqueeze(-1)` | `[1,2,4,8,256,1]` | 5D→6D; new innermost=1 | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` |
| `UNS-28` | `test_unsqueeze_5d_to_6d_aligned` | `[1,2,4,8,256]` (all dtypes) | `unsqueeze(3)` | `[1,2,4,1,8,256]` | 5D→6D; inserts size-1 at dim-3; innermost 256 aligned | `compare_with_cpu(lambda t: t.unsqueeze(3), x)` |

---

## Section 5 — `expand` — Operation Level

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### 5.1 Gemma CPU-traced shapes (aligned)

| ID | Test Name | Input | Chain | Output | Notes | Test Steps |
|---|---|---|---|---|---|---|
| `EXP-01` | `test_expand_gqa_2x_sliding_virtual` | `[1,8,1,32,256]` (all dtypes) | `expand(1,8,2,32,256)` | `[1,8,2,32,256]` | Virtual expand, no clone | `compare_with_cpu(lambda t: t.expand(1,8,2,32,256), x)` |
| `EXP-02` | `test_expand_gqa_8x_full_attn_virtual` | `[1,2,1,32,512]` (all dtypes) | `expand(1,2,8,32,512)` | `[1,2,8,32,512]` | Virtual expand, no clone | `compare_with_cpu(lambda t: t.expand(1,2,8,32,512), x)` |
| `EXP-03` | `test_expand_mask_same_shape` | `[1,1,32,32]` (all dtypes) | `expand(1,-1,32,32)` | `[1,1,32,32]` | Same-shape; inner 32 < 64 for fp16/bf16 | `compare_with_cpu(lambda t: t.expand(1,-1,32,32), x)` |
| `EXP-04` | `test_expand_rope_freq_heads` | `[1,32,1,256]` (all dtypes) | `expand(1,32,16,256)` | `[1,32,16,256]` | RoPE frequency broadcast across heads | `compare_with_cpu(lambda t: t.expand(1,32,16,256), x)` |
| `EXP-05` | `test_expand_bmm_prep_then_view` | `[1,16,32,256]` (all dtypes) | `expand(1,16,32,256).view(16,32,256)` | `[16,32,256]` | CPU trace bmm-operand pattern: same-shape expand then view; see §3.1 | `compare_with_cpu(lambda t: t.expand(1,16,32,256).view(16,32,256), x)` |
| `EXP-06` | `test_expand_chain_then_reshape_heads` | `[1,8,1,32,256]` (all dtypes) | `expand(1,8,2,32,256).reshape(1,16,32,256)` | `[1,16,32,256]` | Expand then flatten heads — forces clone internally | `compare_with_cpu(lambda t: t.expand(1,8,2,32,256).reshape(1,16,32,256), x)` |
| `EXP-07` | `test_expand_chain_bias_add` | `[2816]` + `[32,2816]` (all dtypes) | `b.expand(32,2816)+x` | `[32,2816]` | Bias broadcast then add | `b=randn([2816],dtype); x=randn([32,2816],dtype)` → `compare_with_cpu(lambda b,x: b.expand(32,2816)+x, b, x)` |

### 5.2 Spyre-traced shapes (aligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `EXP-08` | `test_expand_spyre_input_same_shape` | `[1,64,2816]` (all dtypes) | `expand(1,64,2816)` | `[1,64,2816]` | `compare_with_cpu(lambda t: t.expand(1,64,2816), x)` — input operand for bmm |
| `EXP-09` | `test_expand_spyre_weight_q` | `[1,2816,4096]` (all dtypes) | `expand(1,2816,4096)` | `[1,2816,4096]` | `compare_with_cpu(lambda t: t.expand(1,2816,4096), x)` — q_proj weight |
| `EXP-10` | `test_expand_spyre_weight_kv` | `[1,2816,2048]` (all dtypes) | `expand(1,2816,2048)` | `[1,2816,2048]` | `compare_with_cpu(lambda t: t.expand(1,2816,2048), x)` |
| `EXP-11` | `test_expand_spyre_weight_mlp` | `[1,2816,2112]` (all dtypes) | `expand(1,2816,2112)` | `[1,2816,2112]` | `compare_with_cpu(lambda t: t.expand(1,2816,2112), x)` |

### 5.3 Unaligned innermost dimension

Covers 0D–6D inputs with unaligned innermost dimension.

| ID | Test Name | Input | Chain | Output | Stick note | Test Steps |
|---|---|---|---|---|---|---|
| `EXP-12` | `test_expand_0d_scalar_to_4d_unaligned` | `[]` (all dtypes) | `expand(1,16,32,70)` | `[1,16,32,70]` | inner 70: 70%64=6 | `x=torch.tensor(1.0,dtype=dtype)` → `compare_with_cpu(lambda t: t.expand(1,16,32,70), x)` |
| `EXP-13` | `test_expand_1d_bias_unaligned_70` | `[70]` (all dtypes) | `expand(2,32,70)` | `[2,32,70]` | inner 70 | `compare_with_cpu(lambda t: t.expand(2,32,70), x)` |
| `EXP-14` | `test_expand_2d_batch_unaligned_33` | `[1,33]` (all dtypes) | `expand(4,33)` | `[4,33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.expand(4,33), x)` |
| `EXP-15` | `test_expand_3d_batch_unaligned_33` | `[1,32,33]` (all dtypes) | `expand(4,32,33)` | `[4,32,33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.expand(4,32,33), x)` |
| `EXP-16` | `test_expand_4d_gqa_unaligned_65` | `[1,8,1,32,65]` (all dtypes) | `expand(1,8,2,32,65)` | `[1,8,2,32,65]` | inner 65: 65%64=1 | `compare_with_cpu(lambda t: t.expand(1,8,2,32,65), x)` |
| `EXP-17` | `test_expand_4d_mask_unaligned_seq33` | `[1,1,33,33]` (all dtypes) | `expand(2,8,33,33)` | `[2,8,33,33]` | inner 33 | `compare_with_cpu(lambda t: t.expand(2,8,33,33), x)` |
| `EXP-18` | `test_expand_5d_gqa_unaligned_65` | `[1,8,1,32,65]` (all dtypes) | `expand(1,8,2,32,65)` | `[1,8,2,32,65]` | inner 65 | `compare_with_cpu(lambda t: t.expand(1,8,2,32,65), x)` |
| `EXP-19` | `test_expand_5d_minus_one_unaligned` | `[2,1,70]` (all dtypes) | `expand(-1,16,-1)` | `[2,16,70]` | `-1` preserves dim; inner 70 | `compare_with_cpu(lambda t: t.expand(-1,16,-1), x)` |
| `EXP-20` | `test_expand_6d_unaligned_inner` | `[1,2,1,4,1,13]` (all dtypes) | `expand(1,2,8,4,16,13)` | `[1,2,8,4,16,13]` | 6D; inner 13 < 64 | `compare_with_cpu(lambda t: t.expand(1,2,8,4,16,13), x)` |

### 5.4 Tensors containing size-1 dimensions

Covers 0D–6D. A size-1 innermost means expand can either keep it size-1 or broadcast it to aligned.

| ID | Test Name | Input | Chain | Output | Notes | Test Steps |
|---|---|---|---|---|---|---|
| `EXP-21` | `test_expand_0d_scalar_to_4d_aligned` | `[]` (all dtypes) | `expand(1,16,32,256)` | `[1,16,32,256]` | 0D broadcast; inner 256 aligned | `x=torch.tensor(1.0,dtype=dtype)` → `compare_with_cpu(lambda t: t.expand(1,16,32,256), x)` |
| `EXP-22` | `test_expand_1d_size1_keep_inner` | `[1]` (all dtypes) | `expand(2,32,1)` | `[2,32,1]` | Innermost stays size-1 | `compare_with_cpu(lambda t: t.expand(2,32,1), x)` |
| `EXP-23` | `test_expand_1d_size1_to_aligned` | `[1]` (all dtypes) | `expand(256)` | `[256]` | 1D size-1 broadcast to aligned | `compare_with_cpu(lambda t: t.expand(256), x)` |
| `EXP-24` | `test_expand_2d_size1_inner_to_2816` | `[32,1]` (all dtypes) | `expand(32,2816)` | `[32,2816]` | Size-1 innermost expanded to aligned | `compare_with_cpu(lambda t: t.expand(32,2816), x)` |
| `EXP-25` | `test_expand_3d_moe_trailing1_broadcast` | `[32,8,1]` (all dtypes) | `expand(32,8,2816)` | `[32,8,2816]` | MoE routing trailing-1 broadcast | `compare_with_cpu(lambda t: t.expand(32,8,2816), x)` |
| `EXP-26` | `test_expand_4d_all_size1_except_inner` | `[1,1,1,256]` (all dtypes) | `expand(4,16,32,256)` | `[4,16,32,256]` | All outer dims size-1; inner 256 aligned | `compare_with_cpu(lambda t: t.expand(4,16,32,256), x)` |
| `EXP-27` | `test_expand_4d_mqa_16x` | `[1,1,1,256]` (all dtypes) | `expand(1,16,32,256)` | `[1,16,32,256]` | MQA (KV=1) 16× broadcast | `compare_with_cpu(lambda t: t.expand(1,16,32,256), x)` |
| `EXP-28` | `test_expand_5d_size1_inner_aligned` | `[16,1,4,16,1]` (all dtypes) | `expand(16,8,4,16,256)` | `[16,8,4,16,256]` | 5D; size-1 innermost expanded to aligned 256 | `compare_with_cpu(lambda t: t.expand(16,8,4,16,256), x)` |
| `EXP-29` | `test_expand_5d_size1_inner_keep` | `[16,1,4,16,1]` (all dtypes) | `expand(16,8,4,16,1)` | `[16,8,4,16,1]` | 5D; inner stays size-1 | `compare_with_cpu(lambda t: t.expand(16,8,4,16,1), x)` |
| `EXP-30` | `test_expand_6d_size1_to_aligned` | `[1,1,1,4,1,256]` (all dtypes) | `expand(2,8,4,4,16,256)` | `[2,8,4,4,16,256]` | 6D; multiple size-1 broadcast; inner 256 aligned | `compare_with_cpu(lambda t: t.expand(2,8,4,4,16,256), x)` |
| `EXP-31` | `test_expand_6d_aligned` | `[1,2,1,4,1,256]` (all dtypes) | `expand(8,2,4,4,16,256)` | `[8,2,4,4,16,256]` | 6D aligned, no size-1 result | `compare_with_cpu(lambda t: t.expand(8,2,4,4,16,256), x)` |

---

## 6. Cross-Reference: Op-Level Chain Coverage vs Section 1

| Chain type | Op-level tests | In Section 1? | Notes |
|---|---|---|---|
| `view` standalone (Gemma aligned shapes) | RSH-01–08 | ✅ MOD-01–06 | Same shapes, multi-dtype in §2 |
| `view→view` round-trip | RSH-09 | ❌ | Section 1 tests each direction separately in different modules |
| `view` 4D→5D / 6D→4D (RoPE) | RSH-12, RSH-13 | ✅ MOD-08 tests full chain | §2.2 isolates the view step alone |
| `unsqueeze→expand` virtual (no clone) | UNS-06, UNS-07 | ❌ MOD-18/19 always `.clone()` | Tests the non-materialised path |
| `unsqueeze→expand→bmm` | UNS-12 | ✅ MOD-20 | Same chain, multi-dtype in §4 |
| `unsqueeze` multi-step mask chain | UNS-25 | ✅ MOD-11 | Same chain, multi-dtype in §4 |
| `expand→view` (bmm prep) | EXP-05 | ✅ MOD-07 | Multi-dtype in §5 |
| `expand→reshape` (flatten heads) | EXP-06 | ❌ | Not in any Section 1 test |
| `expand+add` (bias broadcast) | EXP-07 | ❌ | Not in any Section 1 test |
| `squeeze` any chain | All SQZ-* | ❌ | Zero Section 1 coverage; squeeze never appears in Gemma traces |
| Unaligned inner dim on any op | RSH-15–22, SQZ-12–17, UNS-15–19, EXP-12–20 | ❌ | Section 1 uses only Gemma shapes (all aligned) |
| Size-1 dims | RSH-23–30, SQZ-18–25, UNS-20–28, EXP-21–31 | Partial — MOD-14 (`[256]→[256,1]`) only | Extended to 0D–6D across all ops |
| 6D tensors | RSH-13, UNS-27/28, EXP-20/30/31, SQZ-24 | Partial — MOD-16 (6D for RoPE unsqueeze) | All ops now covered at 6D |
| `reshape→bmm` (stick on wrong K dim) | RSH-31 | ❌ | Restickify scenario 1; not in Section 1 |
| `reshape→add` (incompatible stick layouts from two views) | RSH-32 | ❌ | Restickify scenario 3 |
| `reshape→dtype cast` (cross-stick-width alignment) | RSH-33 | ❌ | Restickify scenario 4 |
| `reshape→layer_norm` (stick on wrong dim) | RSH-34 | ❌ | Restickify scenario 5 |

---

## 7. Rank Coverage Matrix

The table below confirms that every op has at least one test at every rank 0D–6D across every subsection type.

| Op | 0D | 1D | 2D | 3D | 4D | 5D | 6D |
|---|---|---|---|---|---|---|---|
| **reshape** (input rank) | RSH-23 | RSH-15,16,24,25 | RSH-07,17,25,26 | RSH-01–06,17,18,27 | RSH-04,19,20,28 | RSH-12,21,29 | RSH-22,30 |
| **squeeze** (input rank) | — (0D input; squeeze makes no sense) | SQZ-01,02,18 | SQZ-04,12,19,20 | SQZ-05,06,13,21 | SQZ-07–09,14,15,22 | SQZ-10,16,23 | SQZ-11,17,24 |
| **unsqueeze** (input rank → output) | UNS-14,20 (0D→1D) | UNS-01,05,21 (1D→2D) | UNS-02,08–10,22–24 (2D→3D) | UNS-03,04,17 (3D→4D) | UNS-11,18,26 (4D→5D) | UNS-11,19,27,28 (5D→6D) | — (6D input not applicable) |
| **expand** (input rank) | EXP-12,21 (0D) | EXP-13,22,23 (1D) | EXP-14,24 (2D) | EXP-15,19,25 (3D) | EXP-01–07,16,17,26,27 (4D–5D) | EXP-18,28,29 (5D) | EXP-20,30,31 (6D) |

---

## 8. Coverage Summary

| Section | Op | Dim range | CPU-traced | Spyre-traced | Aligned | Unaligned | Size-1 | Restickify | Total |
|---|---|---|---|---|---|---|---|---|---|
| §1 Module | all 4 | 2D–6D | 17 | 4 | — | — | — | — | **21** |
| §2 `reshape` | `view`/`reshape` | 0D–6D | 9 | 5 | 5 | 8 | 8 | 4 | **34** |
| §3 `squeeze` | `squeeze` | 1D–6D | 0 | 0 | 11 | 6 | 8 | — | **25** |
| §4 `unsqueeze` | `unsqueeze` | 0D–6D | 7 | 6 | 7 | 6 | 9 | — | **28** |
| §5 `expand` | `expand` | 0D–6D | 7 | 4 | 9 | 9 | 11 | — | **31** |
| **Total** | | **0D–6D** | **40** | **19** | **32** | **29** | **36** | **4** | **139** |
