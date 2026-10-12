# Functional Verification Test Plan — Priority View Operations
## `reshape` · `squeeze` · `unsqueeze` · `expand`

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Priority View Operations on torch-spyre: `reshape`, `squeeze`, `unsqueeze`, `expand` |
| **Version** | 3.0 |
| **Status** | Under Review |
| **Date** | 2025-07-14 |
| **Author** | Stuti Sinha |
| **Reviewers** | Kanaka Pathivada, Meena Soundararajan, Albin Joy |
| **Scope** | Text-only Gemma-4 modules (vision tower excluded). Inference-only `torch.compile`. Static concrete dimensions 0D–6D. CPU eager baseline vs compiled Spyre comparison using `compare_with_cpu`. Default compiler env-var values only. |
| **Sources** | `gemma4_26b_module_operator_report_v2.html` (CPU trace, seq=32), `gemma4-view-ops-aten.html` (CPU ATen layer), `gemma4_fx_module_trace_v2.html` (Spyre compile trace, seq=64) |

**Change from v2.1:** Every test case now includes a **Test Steps** column with the exact Python snippet to execute. New Spyre-v2-trace-only op patterns (RoPE reshape+unsqueeze, 6D view, MoE unsqueeze+expand) incorporated. Section cross-reference answers whether op-level chains are already covered in Section 1.

---

## 1. Scope & Strategy

### Key Guidelines

1. **Grounded in real traces.** Every module-level test derives from a confirmed ATen op pattern in at least one of the three source HTML files. Shapes are taken directly from those traces.
2. **No vision modules.** The vision tower had zero ATen activations in both traces (excluded entirely).
3. **Two levels of testing:**
   - **Section 1 (Module Level):** Each test wraps a minimal `nn.Module` that mirrors a real Gemma-4 sub-module forward pass. The view-op *chain* executes inside the module exactly as in production.
   - **Sections 2–5 (Operation Level):** Standalone tests of each op across 0D–6D, all three float dtypes, all stick-alignment scenarios. These are **not** single-op tests — they test **chained sequences** of 2+ view ops to exercise how the Spyre compiler threads stick-layout metadata through successive ops. Single-op coverage is already provided upstream by `test_inductor_ops.py`.
4. **Default compiler environment variables only.**

| Variable | Source File | Default | Relevance |
|---|---|---|---|
| `SENCORES` | `config.py` | **32** | Core count; no view-op correctness impact |
| `LX_PLANNING` | `config.py` | **1 (True)** | Kernel scheduling |
| `CO_OPTIMIZING_LX_PLANNING` | `config.py` | **0 (False)** | Joint solver; no view-op impact |
| `GLOBAL_STICK_OPTIMIZER` | `config.py` | **1 (True)** | Stick optimizer for non-contiguous views — must be on |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | `fusion.py` | **True** | View + pointwise fusion |

> **Note:** Previous FVT versions listed SENCORES=1 and LX_PLANNING=0. Both were incorrect. Correct values from source are SENCORES=32 and LX_PLANNING=1.

5. **Two model architecture variants, not a sweep.** The 26B-A4B-it text stack has exactly two structural layer types that produce distinct view-op shapes. Both are covered.

| Variant | H | KV | D | GQA factor |
|---|---|---|---|---|
| **Sliding-attention** (25 layers) | 16 | 8 | 256 | 2× |
| **Full-attention** (5 layers) | 16 | 2 | 512 | 8× |

6. **Three dtypes:** `bfloat16` (Gemma production), `float16`, `float32`. Module-level tests use `bfloat16` only. Operation-level tests run all three.
7. **Pass criteria for all tests:** `compare_with_cpu(fn, *inputs, run_eager=True)` — values must match CPU reference, not just shape.

---

## 2. Stick Alignment Reference

Spyre stores tensors in 128-byte memory units called **sticks**. The **innermost (last) tensor dimension** is the stick dimension.

| Dtype | Bytes/elem | Elements per stick |
|---|---|---|
| `float16` / `bfloat16` | 2 B | **64** |
| `float32` | 4 B | **32** |

**Aligned:** innermost dim divisible by 64 (fp16/bf16) or 32 (fp32).
Examples: 64, 128, 256, 512, 704, 2048, 2112, 2816, 4096.

**Unaligned — two sub-cases:**
- *Below one stick:* dim < 64 (fp16/bf16) or < 32 (fp32). Examples: 1, 7, 13, 30.
- *Not divisible:* dim ≥ stick width but not a clean multiple. Examples: 33, 65, 70, 129, 257.

**Size-1 trailing dimension** — the extreme unaligned case: 1 element per stick, 63 padding slots. Arises from `unsqueeze(-1)`, `reshape(...,1)`.

**How to produce unaligned shapes:** simply choose tensor shapes with non-multiple innermost dimensions. No special flags needed.

Aligned inner dims from traces: **128**, **256**, **512**, **704**, **2048**, **2112**, **2816**, **4096**.
Unaligned inner dims of interest: **1**, **7**, **13**, **30**, **33**, **65**, **70**, **129**, **257**.

---

## 3. ATen Op Inventory from Traces

### 3.1 CPU Trace (`gemma4_26b_module_operator_report_v2.html`, seq=32)

| ATen Op | Modules | Representative Shapes |
|---|---|---|
| `aten.view.default` | `Gemma4TextDecoderLayer`, `Gemma4TextAttention`, `Linear` projections | `[1,32,2816]→[32,2816]`; `[1,32,4096]→[1,32,16,256]`; `[32,8]→[256]`; `[256,2816]→[32,8,2816]` |
| `aten.unsqueeze.default` | `Gemma4Model` (mask), `Gemma4TextAttention` (RoPE+GQA), `Gemma4TextMoE` (routing) | `[32]→[1,32]`; `[1,32,256]→[1,32,1,256]`; `[1,8,32,256]→[1,8,1,32,256]`; `[256]→[256,1]` |
| `aten.expand.default` | `Gemma4Model` (mask), `Gemma4TextAttention` (GQA+bmm) | `[1,8,1,32,256]→[1,8,2,32,256]`; same-shape bmm expands |
| `aten.squeeze` | **Not observed** | — (tested op-level only) |

### 3.2 Spyre Compile Trace (`gemma4_fx_module_trace_v2.html`, seq=64)

| ATen Op | Modules | Representative Shapes | Note |
|---|---|---|---|
| `aten.view.default` | `Gemma4Attention` (QKV split, RoPE, output), MoE | `[1,64,4096]→[1,64,16,256]`; `[1,64,16,2,128]→[1,64,16,256]`; `[1,8,2,64,256]→[1,16,64,256]` | RoPE applies view to a 5D tensor to produce 6D; then view back after multiply |
| `aten.unsqueeze.default` | All `Linear` (q/k/v/o/gate/up/down proj), RoPE, MoE | `[2816,4096]→[1,2816,4096]`; `[1,64,16,2,128]→[1,64,16,1,2,128]`; `[1,8,2,64]→[1,8,2,64,1]` | Spyre lowering of `F.linear`; RoPE frequency prep; MoE routing weight trailing dim |
| `aten.expand.default` | All `Linear`, MoE token dispatch | `[1,64,2816]→[1,64,2816]`; `[1,8,1]→[1,8,64]`; `[1,1,2816]→[1,8,2816]` | Same-shape bmm operand expands; MoE token broadcast |

---

## 4. Gemma-4 Text Modules Under Test

| # | Module | Class | Ops Exercised | Trace |
|---|---|---|---|---|
| 1 | `model` | `Gemma4Model` | `unsqueeze`×3 (mask build: `[32]→[1,32]→[1,1,32]→[1,1,32,1]`), `expand` (mask broadcast) | CPU seq=32 |
| 2 | `model.language_model.layers.N` | `Gemma4TextDecoderLayer` | `view` (`[1,32,2816]→[32,2816]` + reverse) | CPU seq=32 |
| 3 | `model.language_model.layers.N.self_attn` | `Gemma4TextAttention` | `view` (QKV split + RoPE internal 5D/6D), `unsqueeze` (RoPE freq + GQA prep), `expand` (GQA), `clone` | CPU+Spyre |
| 4 | `model.language_model.layers.N.{q,k,v,o}_proj` | `Linear` (Spyre lowering) | `unsqueeze(0)` (weight), `expand` (input + weight for bmm) | Spyre seq=64 |
| 5 | `model.language_model.layers.N.mlp.{gate,up,down}_proj` | `Linear` (Spyre lowering) | `unsqueeze(0)`, `expand` | Spyre seq=64 |
| 6 | `model.language_model.layers.N.moe_block` | `Gemma4TextMoE` | `view` (flatten/unflatten), `unsqueeze` (routing weight trailing dim + token dispatch) | CPU+Spyre |
| 7 | `model.language_model.rotary_emb` | `Gemma4TextRotaryEmbedding` | `view` (4D→5D/6D for RoPE complex multiply), `unsqueeze` (freq table), `expand` | Spyre seq=64 |

---

## Section 1 — Module-Level Tests

Each test creates a minimal `nn.Module` whose `forward()` reproduces the exact ATen op chain from the trace. All use `bfloat16`. Pass criteria: `compare_with_cpu(fn, *inputs, run_eager=True)`.

> **On chains vs single ops:** Module-level tests naturally exercise **chains** (e.g., MOD-08+09 is `unsqueeze` then `expand`; MOD-16 is `unsqueeze` then `expand` then `bmm`). Sections 2–5 add *additional* chain combinations specifically chosen to exercise stick-layout propagation across op boundaries that are *not* covered here (see cross-reference note at each section header).

---

### reshape / view

| ID | Test Name | Module | Source | ATen Chain | Input | Output | Test Steps |
|---|---|---|---|---|---|---|---|
| `MOD-01` | `test_mod_decoder_token_flatten` | `Gemma4TextDecoderLayer` | CPU seq=32 | `view` | `[1, 32, 2816]` bf16 | `[32, 2816]` | `x = randn([1,32,2816], bf16)` → `compare_with_cpu(lambda t: t.view(-1, 2816), x)` |
| `MOD-02` | `test_mod_decoder_token_unflatten` | `Gemma4TextDecoderLayer` | CPU seq=32 | `view` | `[32, 2816]` bf16 | `[1, 32, 2816]` | `x = randn([32,2816], bf16)` → `compare_with_cpu(lambda t: t.view(1, 32, 2816), x)` |
| `MOD-03` | `test_mod_attn_qsplit_sliding` | `Gemma4TextAttention` | CPU+Spyre | `view` | `[1, 32, 4096]` bf16 | `[1, 32, 16, 256]` | `x = randn([1,32,4096], bf16)` → `compare_with_cpu(lambda t: t.view(1, 32, -1, 256), x)` |
| `MOD-04` | `test_mod_attn_kvsplit_sliding` | `Gemma4TextAttention` | CPU seq=32 | `view` | `[1, 32, 2048]` bf16 | `[1, 32, 8, 256]` | `x = randn([1,32,2048], bf16)` → `compare_with_cpu(lambda t: t.view(1, 32, -1, 256), x)` |
| `MOD-05` | `test_mod_attn_qsplit_spyre_s64` | `Gemma4Attention` | Spyre seq=64 | `view` | `[1, 64, 4096]` bf16 | `[1, 64, 16, 256]` | `x = randn([1,64,4096], bf16)` → `compare_with_cpu(lambda t: t.view(1, 64, -1, 256), x)` |
| `MOD-06` | `test_mod_attn_head_merge` | `Gemma4TextAttention` | CPU seq=32 | `reshape` | `[1, 32, 16, 256]` bf16 | `[1, 32, 4096]` | `x = randn([1,32,16,256], bf16)` → `compare_with_cpu(lambda t: t.reshape(1, 32, -1), x)` |
| `MOD-07` | `test_mod_rope_view_5d_to_6d` | `Gemma4TextRotaryEmbedding` | Spyre seq=64 | `view` (4D→5D) then `unsqueeze` then `view` back | `[1, 64, 16, 256]` bf16 | `[1, 64, 16, 256]` | `x = randn([1,64,16,256], bf16)`; `fn`: `t.view(1,64,16,2,128).unsqueeze(-3).mul(randn([1,64,1,2,128])).view(1,64,16,256)`; `compare_with_cpu(fn, x)` |
| `MOD-08` | `test_mod_moe_index_flatten` | `Gemma4TextMoE` | CPU seq=32 | `view` | `[32, 8]` int64 | `[256]` | `x = randint([32,8], int64)` → `compare_with_cpu(lambda t: t.view(-1), x)` |
| `MOD-09` | `test_mod_moe_output_unflatten` | `Gemma4TextMoE` | CPU seq=32 | `view` | `[256, 2816]` bf16 | `[32, 8, 2816]` | `x = randn([256,2816], bf16)` → `compare_with_cpu(lambda t: t.view(32, 8, 2816), x)` |

### unsqueeze

| ID | Test Name | Module | Source | ATen Chain | Input | Output | Test Steps |
|---|---|---|---|---|---|---|---|
| `MOD-10` | `test_mod_rope_unsqueeze_freq` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)` | `[1, 32, 256]` bf16 | `[1, 32, 1, 256]` | `x = randn([1,32,256], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `MOD-11` | `test_mod_gqa_unsqueeze_sliding` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)` | `[1, 8, 32, 256]` bf16 | `[1, 8, 1, 32, 256]` | `x = randn([1,8,32,256], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `MOD-12` | `test_mod_mask_unsqueeze_chain` | `Gemma4Model` | CPU seq=32 | `unsqueeze(0)→unsqueeze(1)→unsqueeze(3)` | `[32]` int64 | `[1, 1, 32, 1]` | `x = arange(32, int64)` → `compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(1).unsqueeze(3), x)` |
| `MOD-13` | `test_mod_moe_routing_unsqueeze` | `Gemma4TextMoE` | CPU+Spyre | `unsqueeze(-1)` | `[256]` bf16 | `[256, 1]` | `x = randn([256], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` |
| `MOD-14` | `test_mod_linear_weight_unsqueeze` | `Linear` (Spyre lowering) | Spyre seq=64 | `unsqueeze(0)` | `[4096, 2816]` bf16 | `[1, 4096, 2816]` | `x = randn([4096,2816], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `MOD-15` | `test_mod_moe_token_unsqueeze_expand` | `Gemma4TextMoE` | Spyre seq=64 | `unsqueeze(1)→expand(1,8,2816)` | `[1, 2816]` bf16 | `[1, 8, 2816]` | `x = randn([1,2816], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(1).expand(1,8,2816), x)` |
| `MOD-16` | `test_mod_rope_unsqueeze_6d` | `Gemma4TextRotaryEmbedding` | Spyre seq=64 | `view(5D)→unsqueeze(-3)` | `[1, 64, 16, 2, 128]` bf16 | `[1, 64, 16, 1, 2, 128]` | `x = randn([1,64,16,2,128], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(-3), x)` |

### expand

| ID | Test Name | Module | Source | ATen Chain | Input | Output | Test Steps |
|---|---|---|---|---|---|---|---|
| `MOD-17` | `test_mod_gqa_expand_2x_sliding` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)→expand→clone` | `[1, 8, 32, 256]` bf16 | `[1, 8, 2, 32, 256]` (clone) | `x = randn([1,8,32,256], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,8,2,32,256).clone(), x)` |
| `MOD-18` | `test_mod_gqa_expand_8x_full_attn` | `Gemma4TextAttention` | CPU seq=32 | `unsqueeze(2)→expand→clone` | `[1, 2, 32, 512]` bf16 | `[1, 2, 8, 32, 512]` (clone) | `x = randn([1,2,32,512], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,2,8,32,512).clone(), x)` |
| `MOD-19` | `test_mod_mask_expand` | `Gemma4Model` | CPU seq=32 | `expand` (same-shape) | `[1, 1, 32, 32]` bool | `[1, 1, 32, 32]` | `x = randn([1,1,32,32], bf16)` → `compare_with_cpu(lambda t: t.expand(1,-1,32,32), x)` |
| `MOD-20` | `test_mod_linear_spyre_unsqueeze_expand_bmm` | `Linear` (Spyre lowering) | Spyre seq=64 | `unsqueeze(0)→expand→bmm` | weight `[2816, 4096]` + input `[1, 64, 2816]` bf16 | `[1, 64, 4096]` | `w = randn([2816,4096], bf16); x = randn([1,64,2816], bf16)` → `compare_with_cpu(lambda w,x: torch.bmm(x, w.unsqueeze(0).expand(1,2816,4096)), w, x, atol=0.5, rtol=0.1)` |
| `MOD-21` | `test_mod_moe_routing_expand_token` | `Gemma4TextMoE` | Spyre seq=64 | `unsqueeze(1)→expand(1,-1,1)→clone` | `[1, 8]` bf16 | `[1, 8, 64]` | `x = randn([1,8], bf16)` → `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,8,64).clone(), x)` |

---

## Section 2 — `reshape` / `view` — Operation Level

**What Section 1 does NOT cover:** chains where the *stick dimension changes identity* across a reshape (e.g. flatten reduces rank; split increases rank with a new innermost). Section 1 tests only the exact shapes from the model. The chains below add 0D, synthetic unaligned, and high-rank coverage that is absent from Section 1.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### 2.1 Gemma-traced shapes (aligned, taken directly from traces)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `RSH-01` | `test_reshape_qsplit_sliding` | `[1, 32, 4096]` (all dtypes) | `view(1,32,-1,256)` | `[1,32,16,256]` | `x = randn(shape, dtype)` → `compare_with_cpu(lambda t: t.view(1,32,-1,256), x)` |
| `RSH-02` | `test_reshape_kvsplit_sliding` | `[1, 32, 2048]` (all dtypes) | `view(1,32,-1,256)` | `[1,32,8,256]` | `compare_with_cpu(lambda t: t.view(1,32,-1,256), x)` |
| `RSH-03` | `test_reshape_qsplit_full_attn` | `[1, 32, 8192]` (all dtypes) | `view(1,32,16,512)` | `[1,32,16,512]` | `compare_with_cpu(lambda t: t.view(1,32,16,512), x)` |
| `RSH-04` | `test_reshape_qsplit_spyre_s64` | `[1, 64, 4096]` (all dtypes) | `view(1,64,-1,256)` | `[1,64,16,256]` | `compare_with_cpu(lambda t: t.view(1,64,-1,256), x)` |
| `RSH-05` | `test_reshape_token_flatten` | `[1, 32, 2816]` (all dtypes) | `view(-1, 2816)` | `[32,2816]` | `compare_with_cpu(lambda t: t.view(-1, 2816), x)` |
| `RSH-06` | `test_reshape_token_unflatten` | `[32, 2816]` (all dtypes) | `view(1, 32, 2816)` | `[1,32,2816]` | `compare_with_cpu(lambda t: t.view(1, 32, 2816), x)` |
| `RSH-07` | `test_reshape_head_merge` | `[1, 32, 16, 256]` (all dtypes) | `reshape(1, 32, -1)` | `[1,32,4096]` | `compare_with_cpu(lambda t: t.reshape(1, 32, -1), x)` |
| `RSH-08` | `test_reshape_moe_index_flatten` | `[32, 8]` int64 | `view(-1)` | `[256]` | `x = torch.arange(256, dtype=torch.int64).view(32,8)` → `compare_with_cpu(lambda t: t.view(-1), x)` |
| `RSH-09` | `test_reshape_moe_output_unflatten` | `[256, 2816]` (all dtypes) | `view(32, 8, 2816)` | `[32,8,2816]` | `compare_with_cpu(lambda t: t.view(32, 8, 2816), x)` |

### 2.2 Chain: reshape → reshape (flatten + split, aligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `RSH-10` | `test_reshape_chain_flatten_then_split_aligned` | `[2, 16, 256]` (all dtypes) | `reshape(2,-1).reshape(2,16,256)` | `[2,16,256]` | `compare_with_cpu(lambda t: t.reshape(2,-1).reshape(2,16,256), x)` — tests round-trip stick layout identity |
| `RSH-11` | `test_reshape_chain_5d_collapse_aligned` | `[1, 2, 4, 8, 256]` (all dtypes) | `reshape(1,8,2048)` | `[1,8,2048]` | `compare_with_cpu(lambda t: t.reshape(1,8,2048), x)` — 5D→3D, inner 2048 aligned |
| `RSH-12` | `test_reshape_chain_inferred_dim` | `[2, 32, 16, 256]` (all dtypes) | `reshape(2,-1,256)` | `[2,512,256]` | `compare_with_cpu(lambda t: t.reshape(2,-1,256), x)` |

### 2.3 Chain: reshape with unaligned innermost dimension

| ID | Test Name | Input | Chain | Output | Stick note | Test Steps |
|---|---|---|---|---|---|---|
| `RSH-13` | `test_reshape_unaligned_3d_to_2d_33` | `[4, 8, 33]` (all dtypes) | `reshape(32, 33)` | `[32,33]` | inner 33 < 64 | `compare_with_cpu(lambda t: t.reshape(32, 33), x)` |
| `RSH-14` | `test_reshape_unaligned_2d_to_1d_65` | `[8, 65]` (all dtypes) | `reshape(-1)` | `[520]` | input inner 65: 65%64≠0 | `compare_with_cpu(lambda t: t.reshape(-1), x)` |
| `RSH-15` | `test_reshape_unaligned_to_4d_260` | `[1, 32, 2080]` (all dtypes) | `reshape(1,32,8,260)` | `[1,32,8,260]` | output inner 260: 260%64=4 | `compare_with_cpu(lambda t: t.reshape(1,32,8,260), x)` |
| `RSH-16` | `test_reshape_unaligned_head_merge_70` | `[2, 8, 32, 70]` (all dtypes) | `reshape(2,32,560)` | `[2,32,560]` | input inner 70; output 560: 560%64=48 | `compare_with_cpu(lambda t: t.reshape(2,32,560), x)` |
| `RSH-17` | `test_reshape_unaligned_inner_129` | `[1, 32, 1032]` (all dtypes) | `reshape(1,32,8,129)` | `[1,32,8,129]` | inner 129: 129%64=1 | `compare_with_cpu(lambda t: t.reshape(1,32,8,129), x)` |
| `RSH-18` | `test_reshape_unaligned_inferred_42` | `[3, 7, 6]` (all dtypes) | `reshape(3,-1)` | `[3,42]` | inner 42: 42%64≠0 | `compare_with_cpu(lambda t: t.reshape(3,-1), x)` |

### 2.4 Size-1 trailing dimension (extreme unaligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `RSH-19` | `test_reshape_0d_scalar_to_1d` | `[]` (all dtypes) | `reshape(1)` | `[1]` | `x = torch.tensor(1.0, dtype=dtype)` → `compare_with_cpu(lambda t: t.reshape(1), x)` |
| `RSH-20` | `test_reshape_1d_size1_to_3d` | `[1]` (all dtypes) | `reshape(1,1,1)` | `[1,1,1]` | `compare_with_cpu(lambda t: t.reshape(1,1,1), x)` |
| `RSH-21` | `test_reshape_2d_size1_trailing` | `[128, 1]` (all dtypes) | `reshape(128)` | `[128]` | `compare_with_cpu(lambda t: t.reshape(128), x)` — removes trailing size-1 |
| `RSH-22` | `test_reshape_3d_size1_trailing_to_2d` | `[4, 32, 1]` (all dtypes) | `reshape(4, 32)` | `[4,32]` | `compare_with_cpu(lambda t: t.reshape(4, 32), x)` |
| `RSH-23` | `test_reshape_4d_two_size1_dims` | `[1, 8, 1, 256]` (all dtypes) | `reshape(8, 256)` | `[8,256]` | `compare_with_cpu(lambda t: t.reshape(8, 256), x)` |

### 2.5 High-rank (5D / 6D)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `RSH-24` | `test_reshape_4d_to_5d_unfold` | `[1, 2, 4, 256]` (all dtypes) | `reshape(1,2,4,4,64)` | `[1,2,4,4,64]` | `compare_with_cpu(lambda t: t.reshape(1,2,4,4,64), x)` |
| `RSH-25` | `test_reshape_rope_5d_to_6d` | `[1, 64, 16, 2, 128]` (all dtypes) | `reshape(1,64,16,2,1,128)` | `[1,64,16,2,1,128]` | **Spyre trace** RoPE internal. `compare_with_cpu(lambda t: t.reshape(1,64,16,2,1,128), x)` |
| `RSH-26` | `test_reshape_6d_collapse_to_4d` | `[1, 2, 4, 8, 4, 64]` (all dtypes) | `reshape(1,2,4,-1)` | `[1,2,4,2048]` | `compare_with_cpu(lambda t: t.reshape(1,2,4,-1), x)` — 6D→4D, inner 2048 aligned |
| `RSH-27` | `test_reshape_6d_unaligned_inner` | `[1, 2, 4, 8, 4, 7]` (all dtypes) | `reshape(1,8,224)` | `[1,8,224]` | `compare_with_cpu(lambda t: t.reshape(1,8,224), x)` — input inner 7; output 224: 224%64=32 |

---

## Section 3 — `squeeze` — Operation Level

`squeeze` was not observed in either Gemma ATen trace (text stack does not produce singleton activation dims that need squeezing). Tests are standalone chain tests only. Section 1 has zero module-level squeeze coverage — these are entirely new.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### 3.1 No-op cases (targeted dim not size-1)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `SQZ-01` | `test_squeeze_noop_1d_aligned` | `[256]` (all dtypes) | `squeeze(0)` | `[256]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` — verifies no-op path doesn't corrupt layout |
| `SQZ-02` | `test_squeeze_noop_1d_unaligned` | `[63]` (all dtypes) | `squeeze(0)` | `[63]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` |
| `SQZ-03` | `test_squeeze_noop_4d_no_singletons` | `[2, 16, 32, 256]` (all dtypes) | `squeeze()` | `[2,16,32,256]` | `compare_with_cpu(lambda t: t.squeeze(), x)` |

### 3.2 Chain: squeeze → pointwise (aligned result)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `SQZ-04` | `test_squeeze_chain_2d_leading_then_add` | `[1, 256]` (all dtypes) | `squeeze(0)` then `+ bias[256]` | `[256]` | `x = randn([1,256],dtype); b = randn([256],dtype)` → `compare_with_cpu(lambda t: t.squeeze(0) + b, x)` |
| `SQZ-05` | `test_squeeze_chain_3d_batch_2816` | `[1, 32, 2816]` (all dtypes) | `squeeze(0)` | `[32, 2816]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` — Gemma hidden dim inner 2816 |
| `SQZ-06` | `test_squeeze_chain_4d_decode_seq_aligned` | `[1, 1, 16, 256]` (all dtypes) | `squeeze(1)` | `[1, 16, 256]` | `compare_with_cpu(lambda t: t.squeeze(1), x)` — single-token decode; inner 256 |
| `SQZ-07` | `test_squeeze_chain_tuple_dims` | `[1, 16, 1, 256]` (all dtypes) | `squeeze((0, 2))` | `[16, 256]` | `compare_with_cpu(lambda t: t.squeeze((0, 2)), x)` |
| `SQZ-08` | `test_squeeze_chain_negative_trailing` | `[32, 8, 64, 1]` (all dtypes) | `squeeze(-1)` | `[32, 8, 64]` | `compare_with_cpu(lambda t: t.squeeze(-1), x)` |
| `SQZ-09` | `test_squeeze_chain_5d_all_singletons` | `[1, 1, 32, 1, 256]` (all dtypes) | `squeeze()` | `[32, 256]` | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-10` | `test_squeeze_chain_6d_leading` | `[1, 2, 4, 1, 8, 256]` (all dtypes) | `squeeze(0)` | `[2,4,1,8,256]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` |

### 3.3 Unaligned result after squeeze

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `SQZ-11` | `test_squeeze_unaligned_2d_33` | `[1, 33]` (all dtypes) | `squeeze(0)` | `[33]` | `compare_with_cpu(lambda t: t.squeeze(0), x)` — inner 33 < 64 |
| `SQZ-12` | `test_squeeze_unaligned_3d_decode_70` | `[2, 1, 70]` (all dtypes) | `squeeze(1)` | `[2, 70]` | `compare_with_cpu(lambda t: t.squeeze(1), x)` |
| `SQZ-13` | `test_squeeze_unaligned_4d_65` | `[1, 1, 16, 65]` (all dtypes) | `squeeze(1)` | `[1, 16, 65]` | `compare_with_cpu(lambda t: t.squeeze(1), x)` — inner 65: 65%64=1 |
| `SQZ-14` | `test_squeeze_unaligned_tuple_33` | `[1, 16, 1, 33]` (all dtypes) | `squeeze((0, 2))` | `[16, 33]` | `compare_with_cpu(lambda t: t.squeeze((0, 2)), x)` |
| `SQZ-15` | `test_squeeze_unaligned_5d_30` | `[1, 1, 32, 1, 30]` (all dtypes) | `squeeze()` | `[32, 30]` | `compare_with_cpu(lambda t: t.squeeze(), x)` |

### 3.4 Size-1 cases and scalar results

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `SQZ-16` | `test_squeeze_1d_size1_to_scalar` | `[1]` (all dtypes) | `squeeze(0)` | `[]` scalar | `compare_with_cpu(lambda t: t.squeeze(0), x)` — 1D→0D |
| `SQZ-17` | `test_squeeze_2d_all_size1` | `[1, 1]` (all dtypes) | `squeeze()` | `[]` scalar | `compare_with_cpu(lambda t: t.squeeze(), x)` |
| `SQZ-18` | `test_squeeze_4d_to_1d` | `[1, 1, 32, 1]` (all dtypes) | `squeeze()` | `[32]` | `compare_with_cpu(lambda t: t.squeeze(), x)` — inner 32 < 64 for fp16/bf16 |
| `SQZ-19` | `test_squeeze_2d_trailing_size1` | `[128, 1]` (all dtypes) | `squeeze(-1)` | `[128]` | `compare_with_cpu(lambda t: t.squeeze(-1), x)` — removes trailing size-1, new inner 128 aligned |
| `SQZ-20` | `test_squeeze_chain_squeeze_then_reshape` | `[1, 8, 1, 256]` (all dtypes) | `squeeze(0).squeeze(1).reshape(-1)` | `[2048]` | `compare_with_cpu(lambda t: t.squeeze(0).squeeze(1).reshape(-1), x)` — multi-squeeze + reshape chain; tests stale-stride cleanup |

### 3.5 Chain: squeeze → expand (not in Section 1)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `SQZ-21` | `test_squeeze_chain_then_expand_aligned` | `[1, 32, 256]` (all dtypes) | `squeeze(0).unsqueeze(0).expand(4,32,256)` | `[4, 32, 256]` | `compare_with_cpu(lambda t: t.squeeze(0).unsqueeze(0).expand(4,32,256), x)` |
| `SQZ-22` | `test_squeeze_chain_then_expand_unaligned` | `[1, 32, 70]` (all dtypes) | `squeeze(0).unsqueeze(0).expand(4,32,70)` | `[4, 32, 70]` | `compare_with_cpu(lambda t: t.squeeze(0).unsqueeze(0).expand(4,32,70), x)` — inner 70 unaligned |

---

## Section 4 — `unsqueeze` — Operation Level

**What Section 1 does NOT cover:** chains from 0D or 1D inputs; dual-axis unsqueeze; unsqueeze producing size-1 innermost dimension (trailing unsqueeze); unsqueeze before a non-expand computation (RMS norm). Section 1 only covers the specific Gemma activation shapes.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### 4.1 Gemma activation patterns (aligned, direct from traces)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `UNS-01` | `test_unsqueeze_pos_ids_1d_to_2d` | `[32]` int64 | `unsqueeze(0)` | `[1, 32]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-02` | `test_unsqueeze_rope_freq_head_dim` | `[1, 32, 256]` (all dtypes) | `unsqueeze(2)` | `[1,32,1,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `UNS-03` | `test_unsqueeze_gqa_repeat_sliding` | `[1, 8, 32, 256]` (all dtypes) | `unsqueeze(2)` | `[1,8,1,32,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `UNS-04` | `test_unsqueeze_routing_weight_trailing` | `[256]` (all dtypes) | `unsqueeze(-1)` | `[256,1]` | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` — size-1 innermost |
| `UNS-05` | `test_unsqueeze_gqa_full_attn` | `[1, 2, 32, 512]` (all dtypes) | `unsqueeze(2)` | `[1,2,1,32,512]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` — full-attention (D=512) |

### 4.2 Spyre Linear lowering weight patterns (aligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `UNS-06` | `test_unsqueeze_linear_weight_q` | `[4096, 2816]` (all dtypes) | `unsqueeze(0)` | `[1,4096,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-07` | `test_unsqueeze_linear_weight_kv` | `[2048, 2816]` (all dtypes) | `unsqueeze(0)` | `[1,2048,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-08` | `test_unsqueeze_linear_weight_mlp` | `[2816, 2112]` (all dtypes) | `unsqueeze(0)` | `[1,2816,2112]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |

### 4.3 Chain: unsqueeze → expand (GQA + Linear lowering chains)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `UNS-09` | `test_unsqueeze_chain_gqa_2x` | `[1, 8, 32, 256]` (all dtypes) | `unsqueeze(2).expand(1,8,2,32,256)` | `[1,8,2,32,256]` | `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,8,2,32,256), x)` — virtual expand, no clone |
| `UNS-10` | `test_unsqueeze_chain_gqa_8x` | `[1, 2, 32, 512]` (all dtypes) | `unsqueeze(2).expand(1,2,8,32,512)` | `[1,2,8,32,512]` | `compare_with_cpu(lambda t: t.unsqueeze(2).expand(1,2,8,32,512), x)` |
| `UNS-11` | `test_unsqueeze_chain_weight_expand_bmm` | weight `[2816,4096]`, input `[1,64,2816]` (all dtypes) | `w.unsqueeze(0).expand(1,2816,4096)` then `bmm(x, w_e)` | `[1,64,4096]` | `compare_with_cpu(lambda w,x: torch.bmm(x, w.unsqueeze(0).expand(1,2816,4096)), w, x, atol=0.5, rtol=0.1)` |
| `UNS-12` | `test_unsqueeze_chain_token_broadcast` | `[1, 2816]` (all dtypes) | `unsqueeze(1).expand(1,8,2816)` | `[1,8,2816]` | `compare_with_cpu(lambda t: t.unsqueeze(1).expand(1,8,2816), x)` — MoE token dispatch |

### 4.4 Unaligned inner dim

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `UNS-13` | `test_unsqueeze_unaligned_1d_70` | `[70]` (all dtypes) | `unsqueeze(0)` | `[1, 70]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-14` | `test_unsqueeze_unaligned_2d_33` | `[32, 33]` (all dtypes) | `unsqueeze(0)` | `[1,32,33]` | `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-15` | `test_unsqueeze_unaligned_3d_rope_65` | `[1, 32, 65]` (all dtypes) | `unsqueeze(2)` | `[1,32,1,65]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` — inner 65: 65%64=1 |
| `UNS-16` | `test_unsqueeze_unaligned_4d_gqa_65` | `[2, 8, 32, 65]` (all dtypes) | `unsqueeze(2)` | `[2,8,1,32,65]` | `compare_with_cpu(lambda t: t.unsqueeze(2), x)` |
| `UNS-17` | `test_unsqueeze_unaligned_5d_to_6d` | `[1, 2, 4, 8, 13]` (all dtypes) | `unsqueeze(3)` | `[1,2,4,1,8,13]` | `compare_with_cpu(lambda t: t.unsqueeze(3), x)` — 5D→6D; inner 13 |

### 4.5 Size-1 cases (scalar to ND, trailing unsqueeze)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `UNS-18` | `test_unsqueeze_scalar_to_1d` | `[]` (all dtypes) | `unsqueeze(0)` | `[1]` | `x = torch.tensor(1.0, dtype=dtype)` → `compare_with_cpu(lambda t: t.unsqueeze(0), x)` |
| `UNS-19` | `test_unsqueeze_trailing_size1_aligned` | `[32, 256]` (all dtypes) | `unsqueeze(-1)` | `[32,256,1]` | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` — new innermost=1, ultra-unaligned |
| `UNS-20` | `test_unsqueeze_trailing_size1_unaligned` | `[4, 8, 70]` (all dtypes) | `unsqueeze(-1)` | `[4,8,70,1]` | `compare_with_cpu(lambda t: t.unsqueeze(-1), x)` — old innermost 70 also unaligned |
| `UNS-21` | `test_unsqueeze_dual_axis` | `[32, 256]` (all dtypes) | `unsqueeze(0).unsqueeze(3)` | `[1,32,256,1]` | `compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(3), x)` |
| `UNS-22` | `test_unsqueeze_dual_axis_unaligned` | `[32, 70]` (all dtypes) | `unsqueeze(0).unsqueeze(3)` | `[1,32,70,1]` | `compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(3), x)` |
| `UNS-23` | `test_unsqueeze_mask_chain_4_steps` | `[32]` int64 | `unsqueeze(0).unsqueeze(1).unsqueeze(3)` | `[1,1,32,1]` | `compare_with_cpu(lambda t: t.unsqueeze(0).unsqueeze(1).unsqueeze(3), x)` — full Gemma mask build chain |
| `UNS-24` | `test_unsqueeze_negative_middle` | `[1, 16, 32, 256]` (all dtypes) | `unsqueeze(-2)` | `[1,16,32,1,256]` | `compare_with_cpu(lambda t: t.unsqueeze(-2), x)` |

---

## Section 5 — `expand` — Operation Level

**What Section 1 does NOT cover:** scalar (0D) broadcasts; 1D bias-to-ND; expand with `-1` preserve dims; expand producing a size-1 innermost after reshape; 5D high-rank; unaligned head dims. Section 1 only covers the specific Gemma GQA and bmm-lowering shapes.

**Pass criteria:** `compare_with_cpu`, `atol=0.0`

### 5.1 Gemma GQA patterns (aligned, direct from traces)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `EXP-01` | `test_expand_gqa_2x_sliding` | `[1, 8, 1, 32, 256]` (all dtypes) | `expand(1,8,2,32,256)` | `[1,8,2,32,256]` | `compare_with_cpu(lambda t: t.expand(1,8,2,32,256), x)` — virtual, no materialization |
| `EXP-02` | `test_expand_gqa_8x_full_attn` | `[1, 2, 1, 32, 512]` (all dtypes) | `expand(1,2,8,32,512)` | `[1,2,8,32,512]` | `compare_with_cpu(lambda t: t.expand(1,2,8,32,512), x)` |
| `EXP-03` | `test_expand_mask_same_shape` | `[1, 1, 32, 32]` (all dtypes) | `expand(1,-1,32,32)` | `[1,1,32,32]` | `compare_with_cpu(lambda t: t.expand(1,-1,32,32), x)` |
| `EXP-04` | `test_expand_rope_freq_heads` | `[1, 32, 1, 256]` (all dtypes) | `expand(1,32,16,256)` | `[1,32,16,256]` | `compare_with_cpu(lambda t: t.expand(1,32,16,256), x)` |

### 5.2 Spyre Linear lowering patterns (aligned, direct from Spyre trace)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `EXP-05` | `test_expand_spyre_input_same_shape` | `[1, 64, 2816]` (all dtypes) | `expand(1,64,2816)` | `[1,64,2816]` | `compare_with_cpu(lambda t: t.expand(1,64,2816), x)` |
| `EXP-06` | `test_expand_spyre_weight_q` | `[1, 2816, 4096]` (all dtypes) | `expand(1,2816,4096)` | `[1,2816,4096]` | `compare_with_cpu(lambda t: t.expand(1,2816,4096), x)` |
| `EXP-07` | `test_expand_spyre_weight_kv` | `[1, 2816, 2048]` (all dtypes) | `expand(1,2816,2048)` | `[1,2816,2048]` | `compare_with_cpu(lambda t: t.expand(1,2816,2048), x)` |
| `EXP-08` | `test_expand_spyre_weight_mlp` | `[1, 2816, 2112]` (all dtypes) | `expand(1,2816,2112)` | `[1,2816,2112]` | `compare_with_cpu(lambda t: t.expand(1,2816,2112), x)` |

### 5.3 Chain: expand then materialize (clone / reshape forces copy)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `EXP-09` | `test_expand_chain_gqa_then_clone` | `[1, 8, 1, 32, 256]` (all dtypes) | `expand(1,8,2,32,256).clone()` | `[1,8,2,32,256]` contiguous | `compare_with_cpu(lambda t: t.expand(1,8,2,32,256).clone(), x)` — tests materialization path |
| `EXP-10` | `test_expand_chain_then_reshape` | `[1, 8, 1, 32, 256]` (all dtypes) | `expand(1,8,2,32,256).reshape(1,16,32,256)` | `[1,16,32,256]` | `compare_with_cpu(lambda t: t.expand(1,8,2,32,256).reshape(1,16,32,256), x)` — expand then flatten heads |
| `EXP-11` | `test_expand_chain_bias_then_add` | `[2816]` (all dtypes) | `expand(32,2816)` then `+ x` | `[32,2816]` | `b = randn([2816]); x = randn([32,2816])` → `compare_with_cpu(lambda b,x: b.expand(32,2816)+x, b, x)` |

### 5.4 Unaligned inner dim

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `EXP-12` | `test_expand_unaligned_batch_33` | `[1, 32, 33]` (all dtypes) | `expand(4,32,33)` | `[4,32,33]` | `compare_with_cpu(lambda t: t.expand(4,32,33), x)` |
| `EXP-13` | `test_expand_unaligned_bias_70` | `[70]` (all dtypes) | `expand(2,32,70)` | `[2,32,70]` | `compare_with_cpu(lambda t: t.expand(2,32,70), x)` |
| `EXP-14` | `test_expand_unaligned_gqa_65` | `[1, 8, 1, 32, 65]` (all dtypes) | `expand(1,8,2,32,65)` | `[1,8,2,32,65]` | `compare_with_cpu(lambda t: t.expand(1,8,2,32,65), x)` — inner 65 |
| `EXP-15` | `test_expand_unaligned_mask_seq33` | `[1, 1, 33, 33]` (all dtypes) | `expand(2,8,33,33)` | `[2,8,33,33]` | `compare_with_cpu(lambda t: t.expand(2,8,33,33), x)` |
| `EXP-16` | `test_expand_unaligned_minus_one` | `[2, 1, 70]` (all dtypes) | `expand(-1,16,-1)` | `[2,16,70]` | `compare_with_cpu(lambda t: t.expand(-1,16,-1), x)` — `-1` preserves dims |
| `EXP-17` | `test_expand_spyre_input_unaligned_seq` | `[1, 65, 2816]` (all dtypes) | `expand(1,65,2816)` | `[1,65,2816]` | `compare_with_cpu(lambda t: t.expand(1,65,2816), x)` — S=65 middle dim unaligned |

### 5.5 Size-1 and scalar inputs (extreme unaligned)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `EXP-18` | `test_expand_scalar_to_4d_aligned` | `[]` (all dtypes) | `expand(1,16,32,256)` | `[1,16,32,256]` | `x = torch.tensor(1.0, dtype=dtype)` → `compare_with_cpu(lambda t: t.expand(1,16,32,256), x)` |
| `EXP-19` | `test_expand_scalar_to_4d_unaligned` | `[]` (all dtypes) | `expand(1,16,32,70)` | `[1,16,32,70]` | `compare_with_cpu(lambda t: t.expand(1,16,32,70), x)` |
| `EXP-20` | `test_expand_1d_size1_to_3d` | `[1]` (all dtypes) | `expand(2,32,1)` | `[2,32,1]` | `compare_with_cpu(lambda t: t.expand(2,32,1), x)` — innermost stays size-1 |
| `EXP-21` | `test_expand_2d_size1_inner_to_2816` | `[32, 1]` (all dtypes) | `expand(32,2816)` | `[32,2816]` | `compare_with_cpu(lambda t: t.expand(32,2816), x)` — size-1 innermost expanded to aligned |
| `EXP-22` | `test_expand_routing_weight_trailing` | `[32, 8, 1]` (all dtypes) | `expand(32,8,2816)` | `[32,8,2816]` | `compare_with_cpu(lambda t: t.expand(32,8,2816), x)` — MoE trailing-1 broadcast |
| `EXP-23` | `test_expand_all_size1_except_inner` | `[1, 1, 1, 256]` (all dtypes) | `expand(4,16,32,256)` | `[4,16,32,256]` | `compare_with_cpu(lambda t: t.expand(4,16,32,256), x)` |
| `EXP-24` | `test_expand_mqa_16x` | `[1, 1, 1, 256]` (all dtypes) | `expand(1,16,32,256)` | `[1,16,32,256]` | `compare_with_cpu(lambda t: t.expand(1,16,32,256), x)` — MQA (KV=1) broadcast |

### 5.6 High-rank (5D)

| ID | Test Name | Input | Chain | Output | Test Steps |
|---|---|---|---|---|---|
| `EXP-25` | `test_expand_5d_cache_aligned` | `[16, 1, 4, 16, 256]` (all dtypes) | `expand(16,8,4,16,256)` | `[16,8,4,16,256]` | `compare_with_cpu(lambda t: t.expand(16,8,4,16,256), x)` |
| `EXP-26` | `test_expand_5d_cache_unaligned` | `[16, 1, 4, 16, 13]` (all dtypes) | `expand(16,8,4,16,13)` | `[16,8,4,16,13]` | `compare_with_cpu(lambda t: t.expand(16,8,4,16,13), x)` — inner 13 |

---

## 6. Cross-Reference: Are Op-Level Chain Patterns Covered by Section 1?

The table below maps each op-level chain type to whether an equivalent chain exists in Section 1.

| Chain Type | Representative op-level tests | Covered in Section 1? | Section 1 test if yes |
|---|---|---|---|
| `view` → (standalone, aligned) | RSH-01 to RSH-09 | **Yes** (exact Gemma shapes) | MOD-01 through MOD-09 |
| `view` → `view` (round-trip flatten/split) | RSH-10 | **No** — Section 1 only tests one direction per module | New in §2.2 |
| `view` (5D→6D RoPE internal) | RSH-25 | **Partial** — MOD-07 has the full RoPE chain but tests values through it, not the view step in isolation | New in §2.5 confirms view step alone |
| `unsqueeze` → `expand` (GQA, virtual) | UNS-09, UNS-10 | **Partial** — MOD-17/18 test `unsqueeze → expand → clone` (materialised). UNS-09/10 test the *virtual* (non-cloned) expand path | New in §4.3 |
| `unsqueeze` → `expand` → `bmm` | UNS-11 | **Yes** | MOD-20 |
| `unsqueeze` (multi-step mask chain) | UNS-23 | **Yes** | MOD-12 |
| `squeeze` (any chain) | All SQZ-* | **No** — squeeze is absent from all Section 1 module tests | Entirely new in §3 |
| `expand` → `clone` (materialise) | EXP-09 | **Yes** | MOD-17, MOD-18 |
| `expand` → `reshape` (flatten heads) | EXP-10 | **No** — Section 1 does not test the expand-then-reshape pattern | New in §5.3 |
| `expand` → `add` (bias broadcast) | EXP-11 | **No** | New in §5.3 |
| Unaligned inner dim on any op | RSH-13–18, SQZ-11–15, UNS-13–17, EXP-12–17 | **No** — Section 1 uses only exact Gemma shapes (all aligned) | Entirely new in §2–5 |
| Size-1 trailing dimension | RSH-19–23, SQZ-16–20, UNS-18–22, EXP-18–24 | **Partial** — UNS-04 (`[256]→[256,1]`) and MOD-13 cover this for MoE only | Extended coverage in §2–5 |
| 0D scalar input | RSH-19, UNS-18, EXP-18/19 | **No** | Entirely new in §2–5 |
| 5D/6D tensors | RSH-24–27, UNS-17, EXP-25/26 | **Partial** — MOD-16 has 6D for RoPE, MOD-17/18 have 5D for GQA | Extended coverage in §2–5 |

**Summary:** Section 1 covers the Gemma-traced chains with production shapes. Sections 2–5 add: `squeeze` (zero Section 1 coverage), `expand`+`reshape` chains, `expand`+`add` chains, all unaligned-dim variants, scalar inputs, and the virtual-expand (non-cloned) GQA path. These are genuinely not duplicated.

---

## 7. Coverage Summary Matrix

| Section | Op(s) | Dim Range | Gemma-Traced | Spyre-Traced | Aligned | Unaligned | Size-1 | Total |
|---|---|---|---|---|---|---|---|---|
| §1 Module Level | all 4 | 2D–6D | 18 | 3 | — | — | — | **21** |
| §2 `reshape`/`view` | `view`/`reshape` | 0D–6D | 9 | 2 | 12 | 6 | 5 | **27** |
| §3 `squeeze` | `squeeze` | 0D–6D | 0 | 0 | 10 | 5 | 7 | **22** |
| §4 `unsqueeze` | `unsqueeze` | 0D–6D | 8 | 3 | 12 | 5 | 7 | **24** |
| §5 `expand` | `expand` | 0D–5D | 6 | 4 | 11 | 6 | 9 | **26** |
| **Total** | | **0D–6D** | **41** | **12** | **45** | **22** | **28** | **120** |
