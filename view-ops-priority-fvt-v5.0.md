# Functional Verification Test Plan — Priority View Operations
## `reshape` · `squeeze` · `unsqueeze` · `expand`

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Priority View Operations on torch-spyre |
| **Version** | 4.0 |
| **Status** | Under Review |
| **Date** | 2025-07-15 |
| **Author** | Stuti Sinha |
| **Reviewers** | Kanaka Pathivada, Meena Soundararajan, Albin Joy |
| **Scope** | Text-only Gemma-4 modules (vision tower excluded). Inference-only `torch.compile`. Static concrete dimensions 0D–6D. Default compiler env-var values only. |
| **Sources** | `gemma4_26b_module_operator_report_v2.html` (CPU trace, seq=32), `gemma4-view-ops-aten.html` (CPU ATen layer), `gemma4_fx_module_trace_v2.html` (Spyre compile trace, seq=64) |

**What this document covers:**
- Section 1 (module-level): 21 tests reproducing exact Gemma-4 ATen chains, all `bfloat16`.
- Sections 2–5 (operation-level): per-op tests across 0D–6D, all three dtypes (`float16`, `bfloat16`, `float32`).
- §x.3/§x.4 tests use `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer(view_result, consumer)` so each test runs with both a `* 1.0` (pointwise) and a `.sum(dim=-1)` (sum_last) consumer. Effective combinations per §x.3/§x.4 method: **2 × 3 × 2 = 12**. This applies to all four ops: reshape (§2.3/§2.4), squeeze (§3.3/§3.4), unsqueeze (§4.3/§4.4), and expand (§5.3/§5.4).
- Tests producing 0D scalar output (SQZ-18, SQZ-19) use `CONSUMERS_SCALAR = ["pointwise"]` only — `sum(dim=-1)` is undefined on a 0D tensor.
- Op-level §2.1/§2.2/§4.1/§4.2/§5.1/§5.2 tests whose shape is already covered by a module-level test (MOD-xx) have been removed to avoid double-counting. The covering MOD-xx test is noted in the test file as a comment.
- RSH-19 and EXP-18 removed as duplicates at the same rank/bucket as RSH-20 and EXP-16 respectively.

---

## 1. Scope & Strategy

### Key Guidelines

1. **Grounded in real traces.** Every module-level test derives from a confirmed ATen op in at least one of the three source HTML files. Shapes are taken directly from those traces.
2. **No vision modules.** The vision tower had zero ATen activations in both traces (excluded entirely).
3. **Two levels of testing:**
   - **Section 1 (Module Level):** Each test reproduces the exact ATen op chain from a real Gemma-4 sub-module. All are `bfloat16`. The view-op chain executes exactly as in production.
   - **Sections 2–5 (Operation Level):** Each test is a **chain of view op(s) followed by one compute consumer** — either a pointwise (`* 1.0`) or reduction (`sum(dim=-1)`). The consumer is the minimum required to trigger `propagate_layouts.compute_layouts` and make stick-layout correctness observable.
4. **Default compiler environment variables only.**

| Variable | Source File | Default | Relevance to view ops |
|---|---|---|---|
| `SENCORES` | `config.py` | **32** | No view-op correctness impact |
| `LX_PLANNING` | `config.py` | **1 (True)** | Kernel scheduling |
| `CO_OPTIMIZING_LX_PLANNING` | `config.py` | **0 (False)** | No view-op impact |
| `GLOBAL_STICK_OPTIMIZER` | `config.py` | **1 (True)** | Stick optimizer for non-contiguous views — must be on |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | `fusion.py` | **True** | View + pointwise fusion |

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

### Why a compute consumer is required

View ops (`view`, `reshape`, `squeeze`, `unsqueeze`, `expand`) lower in the Spyre Inductor backend to `ReinterpretView` — a metadata-only alias that changes stride/shape without allocating a buffer. The scheduler wraps this in `NopKernelSchedulerNode` (`is_no_op() → True`). When `propagate_layouts` encounters a no-op, it assigns `generic_layout` and skips `compute_layouts` entirely. This means:

- No stick-layout decision is made for the view output.
- No restickify is considered or inserted.
- The compiled graph emits no device kernel for the view.
- `compare_with_cpu` passes trivially because the output is a strided alias of the input with identical element values — **layout propagation bugs are invisible**.

A downstream `ComputedBuffer` (`Pointwise` or `Reduction`) is required to make the compiler commit to a stick layout for the view's output and verify correctness. The two consumers used in this document are:

| Consumer | Code | When to use |
|---|---|---|
| **Pointwise** | `* 1.0` | View does not change which dimension is innermost (e.g. `squeeze(0)` on a leading dim, `unsqueeze(2)` on a middle dim). Forces `compute_layouts` to run; zero numerical error added. |
| **Reduction** | `.sum(dim=-1)` | View changes which dimension is innermost (e.g. `reshape` collapses/splits the last dim, `unsqueeze(-1)` adds a new innermost). The reduction enforces "stick must be on last dim"; if the compiler assigned it elsewhere, a restickify is triggered and any bug becomes a value mismatch. |

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
**No change from v3.1** — module chains already include real compute ops (mul, bmm, etc.) or produce graph outputs that require host copy-back.

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

**Pass criteria:** `compare_with_cpu`, `atol=0.0` unless noted.

> **Note on "aligned" / "unaligned" in subsection headings:** Labels describe the **output** innermost dimension — the stick dimension after the reshape.
>
> **Consumer parametrization for §2.3/§2.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `"pointwise"` (`* 1.0`) and `"sum_last"` (`.sum(dim=-1)`) consumers are run unconditionally for all shapes and inner-dim sizes.

### 2.1 Gemma CPU-traced shapes (aligned)

Shapes taken directly from `gemma4_26b_module_operator_report_v2.html`.
RSH-01/02/04/05/06/07/08 are covered by MOD-03/04/06/01/02/09/10 and are not repeated here.

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `RSH-03` | `test_reshape_qsplit_full_attn` | `[1,32,8192]` (all dtypes) | `view(1,32,16,512) * 1.0` | `[1,32,16,512]` | `* 1.0` | `compare_with_cpu(lambda t: t.view(1,32,16,512) * 1.0, x)` |
| `RSH-09` | `test_reshape_round_trip_flatten_split` | `[1,32,16,256]` (all dtypes) | `reshape(1,32,-1).reshape(1,32,16,256) * 1.0` | `[1,32,16,256]` | `* 1.0` | `compare_with_cpu(lambda t: t.reshape(1,32,-1).reshape(1,32,16,256) * 1.0, x)` |

### 2.2 Spyre-traced shapes (aligned)

Shapes taken directly from `gemma4_fx_module_trace_v2.html`.
RSH-10/12/13 are covered by MOD-05/08/08 and are not repeated here.

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `RSH-11` | `test_reshape_kvsplit_spyre_s64` | `[1,64,2048]` (all dtypes) | `view(1,64,-1,256) * 1.0` | `[1,64,8,256]` | `* 1.0` | `compare_with_cpu(lambda t: t.view(1,64,-1,256) * 1.0, x)` |
| `RSH-14` | `test_reshape_inferred_dim_aligned` | `[2,32,16,256]` (all dtypes) | `reshape(2,-1,256) * 1.0` | `[2,512,256]` | `* 1.0` | `compare_with_cpu(lambda t: t.reshape(2,-1,256) * 1.0, x)` |

### 2.3 Unaligned innermost dimension

Covers all ranks 1D–6D with non-multiple innermost dimensions.
Each test is parametrized with both `"pointwise"` and `"sum_last"` consumers.
RSH-19 is covered by RSH-20 (same rank/bucket) and is not repeated here.

| ID | Test Name | Input | Chain | Stick note |
|---|---|---|---|---|
| `RSH-15` | `test_reshape_unaligned_1d_to_2d` | `[520]` (all dtypes) | `reshape(8,65)` | inner 65: 65%64=1 |
| `RSH-16` | `test_reshape_unaligned_2d_to_1d` | `[8,65]` (all dtypes) | `reshape(-1)` | output inner 520; 1D output |
| `RSH-17` | `test_reshape_unaligned_3d_to_2d_33` | `[4,8,33]` (all dtypes) | `reshape(32,33)` | inner 33 < 64 |
| `RSH-18` | `test_reshape_unaligned_3d_to_4d_260` | `[1,32,2080]` (all dtypes) | `reshape(1,32,8,260)` | output inner 260: 260%64=4 |
| `RSH-20` | `test_reshape_unaligned_4d_to_3d_129` | `[1,32,8,129]` (all dtypes) | `reshape(1,32,1032)` | inner 1032: 1032%64=8 |
| `RSH-21` | `test_reshape_unaligned_5d_to_3d` | `[1,2,4,8,7]` (all dtypes) | `reshape(1,8,56)` | input inner 7; output 56: 56%64≠0 |
| `RSH-22` | `test_reshape_unaligned_6d_to_3d` | `[1,2,4,8,4,7]` (all dtypes) | `reshape(1,8,224)` | input inner 7; output 224: 224%64=32 |

### 2.4 Tensors containing size-1 dimensions

> **Consumer parametrization for §2.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `"pointwise"` (`* 1.0`) and `"sum_last"` (`.sum(dim=-1)`) consumers are run unconditionally for all shapes.

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `RSH-23` | `test_reshape_0d_scalar_to_1d` | `[]` (all dtypes) | `reshape(1)` | `[1]` | pointwise + sum_last | `x=torch.tensor(1.0,dtype=dtype)` → `compare_with_cpu(lambda t: _apply_consumer(t.reshape(1), consumer), x)` |
| `RSH-24` | `test_reshape_1d_size1_to_3d` | `[1]` (all dtypes) | `reshape(1,1,1)` | `[1,1,1]` | pointwise + sum_last | `compare_with_cpu(lambda t: _apply_consumer(t.reshape(1,1,1), consumer), x)` |
| `RSH-25` | `test_reshape_2d_remove_trailing_size1` | `[128,1]` (all dtypes) | `reshape(128)` | `[128]` | pointwise + sum_last | `compare_with_cpu(lambda t: _apply_consumer(t.reshape(128), consumer), x)` |
| `RSH-26` | `test_reshape_3d_trailing_size1_to_2d` | `[4,32,1]` (all dtypes) | `reshape(4,32)` | `[4,32]` | pointwise + sum_last | `compare_with_cpu(lambda t: _apply_consumer(t.reshape(4,32), consumer), x)` |
| `RSH-27` | `test_reshape_4d_two_size1_to_2d` | `[1,8,1,256]` (all dtypes) | `reshape(8,256)` | `[8,256]` | pointwise + sum_last | `compare_with_cpu(lambda t: _apply_consumer(t.reshape(8,256), consumer), x)` |
| `RSH-28` | `test_reshape_4d_size1_trailing_to_3d` | `[2,8,32,1]` (all dtypes) | `reshape(2,8,32)` | `[2,8,32]` | pointwise + sum_last | `compare_with_cpu(lambda t: _apply_consumer(t.reshape(2,8,32), consumer), x)` |
| `RSH-29` | `test_reshape_5d_size1_inner_to_4d` | `[1,2,1,8,256]` (all dtypes) | `reshape(1,2,8,256)` | `[1,2,8,256]` | pointwise + sum_last | `compare_with_cpu(lambda t: _apply_consumer(t.reshape(1,2,8,256), consumer), x)` |
| `RSH-30` | `test_reshape_6d_size1_trailing_to_5d` | `[1,2,4,8,16,1]` (all dtypes) | `reshape(1,2,4,8,16)` | `[1,2,4,8,16]` | pointwise + sum_last | `compare_with_cpu(lambda t: _apply_consumer(t.reshape(1,2,4,8,16), consumer), x)` |

### 2.5 Restickification scenarios — view op as trigger

**Unchanged from v3.1.** These tests already have real compute consumers (bmm, add, dtype cast, layer_norm) — that is the point of the tests.

| ID | Test Name | Restickify scenario | Chain | Input Shape(s) | Test Steps |
|---|---|---|---|---|---|
| `RSH-31` | `test_restickify_reshape_then_bmm` | Scenario 1: reshape changes K stick dim | `reshape(B*H,S,D) → bmm(reshaped, weight_T)` | x `[2,32,16,256]` bf16 + weight `[256,256]` bf16 | `compare_with_cpu(fn, x, w, atol=0.5, rtol=0.1)` |
| `RSH-32` | `test_restickify_reshape_then_add_incompatible_layout` | Scenario 3: two incompatible stick layouts into add | `t.reshape(4,64,128) + t.reshape(4,64,128)` | `[4,128,64]` bf16 | `compare_with_cpu(fn, x)` |
| `RSH-33` | `test_restickify_reshape_then_dtype_cast` | Scenario 4: cross-stick-width cast | `reshape(32,65).to(float32)` | `[1,32,65]` fp16 | `compare_with_cpu(fn, x)` |
| `RSH-34` | `test_restickify_reshape_then_layernorm` | Scenario 5: stick on wrong normalization dim | `reshape(32,2816) → F.layer_norm([2816])` | `[1,32,2816]` bf16 | `compare_with_cpu(fn, x)` |

---

## Section 3 — `squeeze` — Operation Level

`squeeze` was not observed in either Gemma ATen trace. All tests are standalone.
**Pass criteria:** `compare_with_cpu`, `atol=0.0` unless noted.

> **Consumer parametrization for §3.3/§3.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. `CONSUMERS = ["pointwise", "sum_last"]` — giving **2 × 3 × 2 = 12 combinations per method**. Tests producing a 0D scalar output (SQZ-18, SQZ-19) use `CONSUMERS_SCALAR = ["pointwise"]` only — `sum(dim=-1)` is undefined on a 0D tensor.

### 3.1 No-op cases (dim is not size-1)

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `SQZ-01` | `test_squeeze_noop_1d_aligned` | `[256]` (all dtypes) | `squeeze(0) * 1.0` | `[256]` | `* 1.0` | `compare_with_cpu(lambda t: t.squeeze(0) * 1.0, x)` |
| `SQZ-02` | `test_squeeze_noop_1d_unaligned` | `[63]` (all dtypes) | `squeeze(0) * 1.0` | `[63]` | `* 1.0` | `compare_with_cpu(lambda t: t.squeeze(0) * 1.0, x)` |
| `SQZ-03` | `test_squeeze_noop_4d_no_singletons` | `[2,16,32,256]` (all dtypes) | `squeeze() * 1.0` | `[2,16,32,256]` | `* 1.0` | `compare_with_cpu(lambda t: t.squeeze() * 1.0, x)` |

### 3.2 Aligned result after squeeze

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `SQZ-04` | `test_squeeze_2d_leading_to_1d_aligned` | `[1,256]` (all dtypes) | `squeeze(0) * 1.0` | `[256]` | `* 1.0` (non-trailing squeeze; inner unchanged) | `compare_with_cpu(lambda t: t.squeeze(0) * 1.0, x)` |
| `SQZ-05` | `test_squeeze_3d_batch_to_2d_aligned` | `[1,32,2816]` (all dtypes) | `squeeze(0) * 1.0` | `[32,2816]` | `* 1.0` (non-trailing squeeze; inner unchanged) | `compare_with_cpu(lambda t: t.squeeze(0) * 1.0, x)` |
| `SQZ-06` | `test_squeeze_3d_mid_decode_aligned` | `[2,1,256]` (all dtypes) | `squeeze(1) * 1.0` | `[2,256]` | `* 1.0` (non-trailing; inner unchanged) | `compare_with_cpu(lambda t: t.squeeze(1) * 1.0, x)` |
| `SQZ-07` | `test_squeeze_4d_decode_seq_aligned` | `[1,1,16,256]` (all dtypes) | `squeeze(1) * 1.0` | `[1,16,256]` | `* 1.0` | `compare_with_cpu(lambda t: t.squeeze(1) * 1.0, x)` |
| `SQZ-08` | `test_squeeze_4d_tuple_dims_aligned` | `[1,16,1,256]` (all dtypes) | `squeeze((0,2)) * 1.0` | `[16,256]` | `* 1.0` (neither dim is trailing) | `compare_with_cpu(lambda t: t.squeeze((0,2)) * 1.0, x)` |
| `SQZ-09` | `test_squeeze_4d_trailing_to_3d_aligned` | `[32,8,64,1]` (all dtypes) | `squeeze(-1) * 1.0` | `[32,8,64]` | `* 1.0` | `compare_with_cpu(lambda t: t.squeeze(-1) * 1.0, x)` |
| `SQZ-10` | `test_squeeze_5d_all_singletons_aligned` | `[1,1,32,1,256]` (all dtypes) | `squeeze() * 1.0` | `[32,256]` | `* 1.0` (innermost 256 unchanged) | `compare_with_cpu(lambda t: t.squeeze() * 1.0, x)` |
| `SQZ-11` | `test_squeeze_6d_leading_to_5d_aligned` | `[1,2,4,1,8,256]` (all dtypes) | `squeeze(0) * 1.0` | `[2,4,1,8,256]` | `* 1.0` (non-trailing squeeze; inner unchanged) | `compare_with_cpu(lambda t: t.squeeze(0) * 1.0, x)` |

### 3.3 Unaligned result after squeeze

| ID | Test Name | Input | Chain | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `SQZ-12` | `test_squeeze_2d_leading_unaligned_33` | `[1,33]` (all dtypes) | `squeeze(0)` | `[33]` | pointwise + sum_last | inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(0), consumer), x)` |
| `SQZ-13` | `test_squeeze_3d_mid_unaligned_70` | `[2,1,70]` (all dtypes) | `squeeze(1)` | `[2,70]` | pointwise + sum_last | inner 70: 70%64=6 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(1), consumer), x)` |
| `SQZ-14` | `test_squeeze_4d_decode_unaligned_65` | `[1,1,16,65]` (all dtypes) | `squeeze(1)` | `[1,16,65]` | pointwise + sum_last | inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(1), consumer), x)` |
| `SQZ-15` | `test_squeeze_4d_tuple_unaligned_33` | `[1,16,1,33]` (all dtypes) | `squeeze((0,2))` | `[16,33]` | pointwise + sum_last | inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze((0,2)), consumer), x)` |
| `SQZ-16` | `test_squeeze_5d_all_singletons_unaligned` | `[1,1,32,1,30]` (all dtypes) | `squeeze()` | `[32,30]` | pointwise + sum_last | inner 30 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(), consumer), x)` |
| `SQZ-17` | `test_squeeze_6d_leading_unaligned_13` | `[1,2,4,1,8,13]` (all dtypes) | `squeeze(0)` | `[2,4,1,8,13]` | pointwise + sum_last | inner 13 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(0), consumer), x)` |

### 3.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Chain | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `SQZ-18` | `test_squeeze_1d_size1_to_scalar` | `[1]` (all dtypes) | `squeeze(0)` | `[]` scalar | pointwise only (0D; sum undefined) | 1D→0D | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(0), consumer), x)` |
| `SQZ-19` | `test_squeeze_2d_all_size1_to_scalar` | `[1,1]` (all dtypes) | `squeeze()` | `[]` scalar | pointwise only (0D; sum undefined) | All dims=1 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(), consumer), x)` |
| `SQZ-20` | `test_squeeze_2d_trailing_size1` | `[128,1]` (all dtypes) | `squeeze(-1)` | `[128]` | pointwise + sum_last | Remove trailing size-1; inner changes: 1→128 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(-1), consumer), x)` |
| `SQZ-21` | `test_squeeze_3d_all_size1_to_1d` | `[1,1,256]` (all dtypes) | `squeeze()` | `[256]` | pointwise + sum_last | Two size-1 removed; inner unchanged: 256 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(), consumer), x)` |
| `SQZ-22` | `test_squeeze_4d_mixed_size1_to_1d` | `[1,1,32,1]` (all dtypes) | `squeeze()` | `[32]` | pointwise + sum_last | Three size-1 removed; inner changes 1→32 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(), consumer), x)` |
| `SQZ-23` | `test_squeeze_5d_size1_inner` | `[2,1,32,1,256]` (all dtypes) | `squeeze()` | `[2,32,256]` | pointwise + sum_last | Two middle size-1 removed; inner unchanged: 256 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(), consumer), x)` |
| `SQZ-24` | `test_squeeze_6d_size1_to_3d` | `[1,1,4,1,8,256]` (all dtypes) | `squeeze()` | `[4,8,256]` | pointwise + sum_last | Three size-1 removed from 6D; inner unchanged: 256 | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(), consumer), x)` |
| `SQZ-25` | `test_squeeze_chain_multi_then_reshape` | `[1,8,1,256]` (all dtypes) | `squeeze(0).squeeze(1).reshape(-1)` | `[2048]` | pointwise + sum_last | Multi-squeeze + reshape | `compare_with_cpu(lambda t: _apply_consumer(t.squeeze(0).squeeze(1).reshape(-1), consumer), x)` |

---

## Section 4 — `unsqueeze` — Operation Level

**Pass criteria:** `compare_with_cpu`, `atol=0.0` unless noted.

> **Consumer parametrization for §4.3/§4.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Both `* 1.0` (pointwise) and `.sum(dim=-1)` (sum_last) are applied. int64 tests in §4.1 have no consumer (integer ops).

### 4.1 Gemma CPU-traced shapes (aligned)

UNS-01/02/03/05/06/07 are covered by MOD-11/12/13/14/18/19 and are not repeated here.

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `UNS-04` | `test_unsqueeze_4d_gqa_full_attn_to_5d` | `[1,2,32,512]` (all dtypes) | `unsqueeze(2) * 1.0` | `[1,2,1,32,512]` | `* 1.0` (inner unchanged: 512) | `compare_with_cpu(lambda t: t.unsqueeze(2) * 1.0, x)` |

### 4.2 Spyre-traced shapes (aligned)

UNS-08/11/12/13 are covered by MOD-15/16/20/21 and are not repeated here.

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `UNS-09` | `test_unsqueeze_2d_linear_weight_kv` | `[2048,2816]` (all dtypes) | `unsqueeze(0) * 1.0` | `[1,2048,2816]` | `* 1.0` | `compare_with_cpu(lambda t: t.unsqueeze(0) * 1.0, x)` |
| `UNS-10` | `test_unsqueeze_2d_linear_weight_mlp` | `[2816,2112]` (all dtypes) | `unsqueeze(0) * 1.0` | `[1,2816,2112]` | `* 1.0` | `compare_with_cpu(lambda t: t.unsqueeze(0) * 1.0, x)` |

### 4.3 Unaligned innermost dimension

| ID | Test Name | Input | Chain | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `UNS-14` | `test_unsqueeze_0d_to_1d_unaligned` | `[]` (all dtypes) | `unsqueeze(0)` | `[1]` | pointwise + sum_last | 0D scalar; result inner=1 | `x=torch.tensor(1.5,dtype=dtype)` → `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)` |
| `UNS-15` | `test_unsqueeze_1d_to_2d_unaligned_70` | `[70]` (all dtypes) | `unsqueeze(0)` | `[1,70]` | pointwise + sum_last | inner 70: 70%64=6 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)` |
| `UNS-16` | `test_unsqueeze_2d_to_3d_unaligned_33` | `[32,33]` (all dtypes) | `unsqueeze(0)` | `[1,32,33]` | pointwise + sum_last | inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)` |
| `UNS-17` | `test_unsqueeze_3d_to_4d_rope_unaligned_65` | `[1,32,65]` (all dtypes) | `unsqueeze(2)` | `[1,32,1,65]` | pointwise + sum_last | inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)` |
| `UNS-18` | `test_unsqueeze_4d_to_5d_gqa_unaligned_65` | `[2,8,32,65]` (all dtypes) | `unsqueeze(2)` | `[2,8,1,32,65]` | pointwise + sum_last | inner 65 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)` |
| `UNS-19` | `test_unsqueeze_5d_to_6d_unaligned_13` | `[1,2,4,8,13]` (all dtypes) | `unsqueeze(3)` | `[1,2,4,1,8,13]` | pointwise + sum_last | inner 13 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(3), consumer), x)` |

### 4.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Chain | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `UNS-20` | `test_unsqueeze_0d_to_1d_aligned` | `[]` (all dtypes) | `unsqueeze(0)` | `[1]` | pointwise + sum_last | 0D→1D | `x=torch.tensor(2.0,dtype=dtype)` → `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)` |
| `UNS-21` | `test_unsqueeze_1d_size1_to_2d` | `[1]` (all dtypes) | `unsqueeze(0)` | `[1,1]` | pointwise + sum_last | Size-1 input stays size-1 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(0), consumer), x)` |
| `UNS-22` | `test_unsqueeze_2d_trailing_size1_aligned` | `[32,256]` (all dtypes) | `unsqueeze(-1)` | `[32,256,1]` | pointwise + sum_last | New innermost=1; preceding 256 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)` |
| `UNS-23` | `test_unsqueeze_2d_trailing_size1_unaligned` | `[4,8,70]` (all dtypes) | `unsqueeze(-1)` | `[4,8,70,1]` | pointwise + sum_last | New innermost=1; preceding 70 unaligned | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)` |
| `UNS-24` | `test_unsqueeze_3d_dual_axis` | `[32,256]` (all dtypes) | `unsqueeze(0).unsqueeze(3)` | `[1,32,256,1]` | pointwise + sum_last | Leading + trailing size-1 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(0).unsqueeze(3), consumer), x)` |
| `UNS-26` | `test_unsqueeze_4d_size1_innermost_to_5d` | `[1,8,32,1]` (all dtypes) | `unsqueeze(2)` | `[1,8,1,32,1]` | pointwise + sum_last | Both new dim and innermost size-1 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(2), consumer), x)` |
| `UNS-27` | `test_unsqueeze_5d_trailing_size1_to_6d` | `[1,2,4,8,256]` (all dtypes) | `unsqueeze(-1)` | `[1,2,4,8,256,1]` | pointwise + sum_last | 5D→6D; new innermost=1 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(-1), consumer), x)` |
| `UNS-28` | `test_unsqueeze_5d_to_6d_aligned` | `[1,2,4,8,256]` (all dtypes) | `unsqueeze(3)` | `[1,2,4,1,8,256]` | pointwise + sum_last | 5D→6D; size-1 at dim-3 | `compare_with_cpu(lambda t: _apply_consumer(t.unsqueeze(3), consumer), x)` |

---

## Section 5 — `expand` — Operation Level

**Pass criteria:** `compare_with_cpu`, `atol=0.0` unless noted.

> **Consumer parametrization for §5.3/§5.4:** Every test uses `@pytest.mark.parametrize("consumer", CONSUMERS)` with `_apply_consumer`. Since `expand` never changes the innermost dim, `sum(dim=-1)` reduces along the same (unchanged) inner dim in all cases — a valid and distinct code path. Exceptions: EXP-05, EXP-06, EXP-07 already have real consumers (view, reshape, add) that form the point of those tests — they are not consumer-parametrized.

### 5.1 Gemma CPU-traced shapes (aligned)

EXP-01/02/03 are covered by MOD-18/19/17 and are not repeated here.

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `EXP-04` | `test_expand_rope_freq_heads` | `[1,32,1,256]` (all dtypes) | `expand(1,32,16,256) * 1.0` | `[1,32,16,256]` | `* 1.0` | `compare_with_cpu(lambda t: t.expand(1,32,16,256) * 1.0, x)` |
| `EXP-05` | `test_expand_bmm_prep_then_view` | `[1,16,32,256]` (all dtypes) | `expand(1,16,32,256).view(16,32,256)` | `[16,32,256]` | **existing** (view) | `compare_with_cpu(lambda t: t.expand(1,16,32,256).view(16,32,256), x)` |
| `EXP-06` | `test_expand_chain_then_reshape_heads` | `[1,8,1,32,256]` (all dtypes) | `expand(1,8,2,32,256).reshape(1,16,32,256)` | `[1,16,32,256]` | **existing** (reshape) | `compare_with_cpu(lambda t: t.expand(1,8,2,32,256).reshape(1,16,32,256), x)` |
| `EXP-07` | `test_expand_chain_bias_add` | `[2816]` + `[32,2816]` (all dtypes) | `b.expand(32,2816)+x` | `[32,2816]` | **existing** (add) | `compare_with_cpu(lambda b,x: b.expand(32,2816)+x, b, x)` |

### 5.2 Spyre-traced shapes (aligned)

EXP-08/09 are covered by MOD-20 and are not repeated here.

| ID | Test Name | Input | Chain | Output | Consumer | Test Steps |
|---|---|---|---|---|---|---|
| `EXP-10` | `test_expand_spyre_weight_kv` | `[1,2816,2048]` (all dtypes) | `expand(1,2816,2048) * 1.0` | `[1,2816,2048]` | `* 1.0` | `compare_with_cpu(lambda t: t.expand(1,2816,2048) * 1.0, x)` |
| `EXP-11` | `test_expand_spyre_weight_mlp` | `[1,2816,2112]` (all dtypes) | `expand(1,2816,2112) * 1.0` | `[1,2816,2112]` | `* 1.0` | `compare_with_cpu(lambda t: t.expand(1,2816,2112) * 1.0, x)` |

### 5.3 Unaligned innermost dimension

| ID | Test Name | Input | Chain | Output | Consumer | Stick note | Test Steps |
|---|---|---|---|---|---|---|---|
| `EXP-12` | `test_expand_0d_scalar_to_4d_unaligned` | `[]` (all dtypes) | `expand(1,16,32,70)` | `[1,16,32,70]` | pointwise + sum_last | inner 70: 70%64=6 | `x=torch.tensor(1.0,dtype=dtype)` → `compare_with_cpu(lambda t: _apply_consumer(t.expand(1,16,32,70), consumer), x)` |
| `EXP-13` | `test_expand_1d_bias_unaligned_70` | `[70]` (all dtypes) | `expand(2,32,70)` | `[2,32,70]` | pointwise + sum_last | inner 70 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(2,32,70), consumer), x)` |
| `EXP-14` | `test_expand_2d_batch_unaligned_33` | `[1,33]` (all dtypes) | `expand(4,33)` | `[4,33]` | pointwise + sum_last | inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(4,33), consumer), x)` |
| `EXP-15` | `test_expand_3d_batch_unaligned_33` | `[1,32,33]` (all dtypes) | `expand(4,32,33)` | `[4,32,33]` | pointwise + sum_last | inner 33 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(4,32,33), consumer), x)` |
| `EXP-16` | `test_expand_4d_gqa_unaligned_65` | `[1,8,1,32,65]` (all dtypes) | `expand(1,8,2,32,65)` | `[1,8,2,32,65]` | pointwise + sum_last | inner 65: 65%64=1 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(1,8,2,32,65), consumer), x)` |
| `EXP-17` | `test_expand_4d_mask_unaligned_seq33` | `[1,1,33,33]` (all dtypes) | `expand(2,8,33,33)` | `[2,8,33,33]` | pointwise + sum_last | inner 33 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(2,8,33,33), consumer), x)` |
| `EXP-19` | `test_expand_5d_minus_one_unaligned` | `[2,1,70]` (all dtypes) | `expand(-1,16,-1)` | `[2,16,70]` | pointwise + sum_last | inner 70 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(-1,16,-1), consumer), x)` |
| `EXP-20` | `test_expand_6d_unaligned_inner` | `[1,2,1,4,1,13]` (all dtypes) | `expand(1,2,8,4,16,13)` | `[1,2,8,4,16,13]` | pointwise + sum_last | 6D; inner 13 < 64 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(1,2,8,4,16,13), consumer), x)` |

### 5.4 Tensors containing size-1 dimensions

| ID | Test Name | Input | Chain | Output | Consumer | Notes | Test Steps |
|---|---|---|---|---|---|---|---|
| `EXP-21` | `test_expand_0d_scalar_to_4d_aligned` | `[]` (all dtypes) | `expand(1,16,32,256)` | `[1,16,32,256]` | pointwise + sum_last | 0D broadcast; inner 256 aligned | `x=torch.tensor(1.0,dtype=dtype)` → `compare_with_cpu(lambda t: _apply_consumer(t.expand(1,16,32,256), consumer), x)` |
| `EXP-22` | `test_expand_1d_size1_keep_inner` | `[1]` (all dtypes) | `expand(2,32,1)` | `[2,32,1]` | pointwise + sum_last | Innermost stays size-1 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(2,32,1), consumer), x)` |
| `EXP-23` | `test_expand_1d_size1_to_aligned` | `[1]` (all dtypes) | `expand(256)` | `[256]` | pointwise + sum_last | 1D size-1 broadcast to aligned | `compare_with_cpu(lambda t: _apply_consumer(t.expand(256), consumer), x)` |
| `EXP-24` | `test_expand_2d_size1_inner_to_2816` | `[32,1]` (all dtypes) | `expand(32,2816)` | `[32,2816]` | pointwise + sum_last | Size-1 innermost → aligned | `compare_with_cpu(lambda t: _apply_consumer(t.expand(32,2816), consumer), x)` |
| `EXP-25` | `test_expand_3d_moe_trailing1_broadcast` | `[32,8,1]` (all dtypes) | `expand(32,8,2816)` | `[32,8,2816]` | pointwise + sum_last | MoE routing trailing-1 broadcast | `compare_with_cpu(lambda t: _apply_consumer(t.expand(32,8,2816), consumer), x)` |
| `EXP-26` | `test_expand_4d_all_size1_except_inner` | `[1,1,1,256]` (all dtypes) | `expand(4,16,32,256)` | `[4,16,32,256]` | pointwise + sum_last | All outer dims size-1; inner 256 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(4,16,32,256), consumer), x)` |
| `EXP-27` | `test_expand_4d_mqa_16x` | `[1,1,1,256]` (all dtypes) | `expand(1,16,32,256)` | `[1,16,32,256]` | pointwise + sum_last | MQA 16× broadcast | `compare_with_cpu(lambda t: _apply_consumer(t.expand(1,16,32,256), consumer), x)` |
| `EXP-28` | `test_expand_5d_size1_inner_aligned` | `[16,1,4,16,1]` (all dtypes) | `expand(16,8,4,16,256)` | `[16,8,4,16,256]` | pointwise + sum_last | 5D; size-1 innermost → aligned 256 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(16,8,4,16,256), consumer), x)` |
| `EXP-29` | `test_expand_5d_size1_inner_keep` | `[16,1,4,16,1]` (all dtypes) | `expand(16,8,4,16,1)` | `[16,8,4,16,1]` | pointwise + sum_last | 5D; inner stays size-1 | `compare_with_cpu(lambda t: _apply_consumer(t.expand(16,8,4,16,1), consumer), x)` |
| `EXP-30` | `test_expand_6d_size1_to_aligned` | `[1,1,1,4,1,256]` (all dtypes) | `expand(2,8,4,4,16,256)` | `[2,8,4,4,16,256]` | pointwise + sum_last | 6D; multiple size-1 broadcast | `compare_with_cpu(lambda t: _apply_consumer(t.expand(2,8,4,4,16,256), consumer), x)` |
| `EXP-31` | `test_expand_6d_aligned` | `[1,2,1,4,1,256]` (all dtypes) | `expand(8,2,4,4,16,256)` | `[8,2,4,4,16,256]` | pointwise + sum_last | 6D aligned, no size-1 result | `compare_with_cpu(lambda t: _apply_consumer(t.expand(8,2,4,4,16,256), consumer), x)` |

---

## 6. Cross-Reference: Op-Level Chain Coverage vs Section 1

Op-level §x.1/§x.2 tests whose base view op + input shape are covered by a module-level test have been removed to avoid double-counting; those MOD-xx tests are the single source of truth for those shapes.

| Chain type | Op-level tests | In Section 1? | Notes |
|---|---|---|---|
| `view` standalone (Gemma aligned shapes) | — (covered by MOD-01–06/09/10) | ✅ | RSH-01–08 are covered by MOD-01–06/09/10 and are not repeated here |
| `view→view` round-trip | RSH-09 | ❌ | No module-level test; kept in §2.1 |
| `view` 4D→5D / 6D→4D (RoPE) | — (covered by MOD-08) | ✅ | RSH-12/13 are covered by MOD-08 and are not repeated here |
| `view` K/V split seq=64 | RSH-11 | ❌ | No module test for this K/V shape; kept |
| `unsqueeze→expand` virtual (no clone) | — (covered by MOD-18/19) | ✅ | UNS-06/07 are covered by MOD-18/19 (with `.clone()`) and are not repeated here |
| `unsqueeze→expand→bmm` | — (covered by MOD-20) | ✅ | UNS-12 is covered by MOD-20 and is not repeated here |
| `unsqueeze` multi-step mask chain | — (covered by MOD-11) | ✅ | UNS-25 is covered by MOD-11 and is not repeated here |
| `unsqueeze` full-attn GQA 4D→5D | UNS-04 | ❌ | No module test for full-attn path; kept |
| `unsqueeze` K/V weight shapes | UNS-09, UNS-10 | ❌ | Unique shapes; kept |
| `expand→view` (bmm prep) | EXP-05 | ✅ MOD-07 | Kept — multi-dtype adds value beyond the module-level bfloat16-only test |
| `expand` GQA virtual | — (covered by MOD-18/19) | ✅ | EXP-01/02 are covered by MOD-18/19 and are not repeated here |
| `expand` mask same-shape | — (covered by MOD-17) | ✅ | EXP-03 is covered by MOD-17 and is not repeated here |
| `expand` same-shape bmm (Spyre) | — (covered by MOD-20) | ✅ | EXP-08/09 are covered by MOD-20 and are not repeated here |
| `expand→reshape` (flatten heads) | EXP-06 | ❌ | Not in any Section 1 test; kept |
| `expand+add` (bias broadcast) | EXP-07 | ❌ | Not in any Section 1 test; kept |
| `squeeze` any chain | All SQZ-* | ❌ | Zero Section 1 coverage |
| Unaligned inner dim on any op | RSH-15–22, SQZ-12–17, UNS-15–19, EXP-12–20 | ❌ | Section 1 uses only aligned Gemma shapes |
| Size-1 dims | RSH-23–30, SQZ-18–25, UNS-20–24/26–28, EXP-21–31 | Partial — MOD-14 only | Extended to 0D–6D across all ops |
| 6D tensors | UNS-27/28, EXP-20/30/31, SQZ-24 | Partial — MOD-16 only | All ops covered at 6D |
| `reshape→bmm` | RSH-31 | ❌ | Restickify scenario 1 |
| `reshape→add` | RSH-32 | ❌ | Restickify scenario 3 |
| `reshape→dtype cast` | RSH-33 | ❌ | Restickify scenario 4 |
| `reshape→layer_norm` | RSH-34 | ❌ | Restickify scenario 5 |

---

## 7. Rank Coverage Matrix

RSH-01/02/04–08/10/12/13, UNS-01–03/05–08/11–13/25, and EXP-01–03/08/09 are covered by MOD-xx module-level tests and are not repeated in the op-level sections.

| Op | 0D | 1D | 2D | 3D | 4D | 5D | 6D |
|---|---|---|---|---|---|---|---|
| **reshape** (input rank) | RSH-23 | RSH-15,16,24,25 | RSH-17,25,26 | RSH-17,18,27 | RSH-20,28 | RSH-21,29 | RSH-22,30 |
| **squeeze** (input rank) | — | SQZ-01,02,18 | SQZ-04,12,19,20 | SQZ-05,06,13,21 | SQZ-07–09,14,15,22 | SQZ-10,16,23 | SQZ-11,17,24 |
| **unsqueeze** (input rank → output) | UNS-14,20 (0D→1D) | UNS-21 (1D→2D) | UNS-09,10,22–24 (2D→3D) | UNS-04,17 (3D→4D) | UNS-18,26 (4D→5D) | UNS-19,27,28 (5D→6D) | — |
| **expand** (input rank) | EXP-12,21 (0D) | EXP-13,22,23 (1D) | EXP-14,24 (2D) | EXP-15,19,25 (3D) | EXP-04–07,16,17,26,27 (4D–5D) | EXP-28,29 (5D) | EXP-20,30,31 (6D) |

---

## 8. Coverage Summary

Active test counts (op-level tests only; module-level MOD-xx tests counted separately):

| Section | Op | Dim range | CPU-traced | Spyre-traced | Aligned | Unaligned | Size-1 | Restickify | Total active |
|---|---|---|---|---|---|---|---|---|---|
| §1 Module | all ops | 2D–6D | 17 | 4 | — | — | — | — | **21** |
| §2 `reshape` | `view`/`reshape` | 0D–6D | 2 | 2 | 2 | 7 | 8 | 4 | **23** |
| §3 `squeeze` | `squeeze` | 1D–6D | 0 | 0 | 11 | 6 | 8 | — | **25** |
| §4 `unsqueeze` | `unsqueeze` | 0D–6D | 1 | 2 | 3 | 6 | 8 | — | **17** |
| §5 `expand` | `expand` | 0D–6D | 4 | 2 | 6 | 8 | 11 | — | **25** |
| **Total** | | **0D–6D** | **24** | **10** | **22** | **27** | **35** | **4** | **110** |
