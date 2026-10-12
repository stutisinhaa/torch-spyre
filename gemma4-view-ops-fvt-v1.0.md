# Functional Verification Test Plan — Gemma-4 View and Data Movement Operations

---

| Field | Value |
|---|---|
| **Document Title** | FVT Plan — Gemma-4 View and Data Movement Operations on torch-spyre |
| **Version** | 1.1 |
| **Status** | Under Review |
| **Date** | 18-09-2026 |
| **Author** | Stuti Sinha |
| **Reviewers** | Kanaka Pathivada, Meena Soundararajan, Albin Joy |
| **Scope** | Inference-only compilation (`torch.compile`), static concrete dimensions (1D to 6D), eager CPU baseline vs compiled Spyre comparison |
| **Target Architecture** | Google Gemma-4 (`Gemma4ForConditionalGeneration`, `Gemma4TextModel`, `Gemma4VisionModel`, `Gemma4MoE`) |

---

## 1. Scope & Strategy

This document defines functional verification test cases for view, indexing, and data movement operations extracted from the **Gemma-4-26B-A4B-it** architecture.

### Key Guidelines & Test Philosophy:
1. **No Symbolic Dimensions**: All tests evaluate concrete, static tensor shapes across standard, scaled, and edge parameter regimes.
2. **Whole-Module Testing for Modules**: High-level architectural modules are tested end-to-end as unified `torch.nn.Module` units with instantiated weights and full forward passes.
3. **11 Covered Operators**:
   * Priority Operators: `reshape`, `squeeze`, `unsqueeze`, `expand`
   * Full View & Movement Suite: `view`, `transpose`, `clone`, `contiguous`, `cat`, `t` (2D transpose), `select`, `split` / `chunk`
4. **Dimensionality Spectrum**: Explicit test coverage across ranks **1D, 2D, 3D, 4D, 5D, and 6D**.
5. **Test Provenance / Classification Column**: Every test case explicitly declares its status:
   * **`[NET-NEW-SCENARIO]`**: Completely new architectural scenario not present anywhere in the codebase (e.g., GQA 8x expansion, MoE routing flatten, 5D/6D vision/paged caches, whole-module chains).
   * **`[GEMMA-PARAM-SWEEP]`**: Testing user-modified configuration scenarios (scaled heads, odd sequence lengths, downscaled experts, custom batch sizes).
   * **`[GEMMA-EDGE-SHAPE]`**: Non-power-of-2 hardware stick boundary test specifically targeting Gemma dimensions (D=72, H=2816, I=704).

---

## 2. The 7 Gemma-4 Modules with View & Data Movement Operations

The Gemma-4 architecture contains 7 critical sub-modules that heavily utilize view, reshape, slice, split, permute, and concatenation operations. These 7 modules are tested directly in Section 1:

| # | Module Name | Architectural Purpose | View & Data Ops Involved | Critical Spyre Verification Focus |
|---|---|---|---|---|
| **1** | **`Gemma4TextAttention`** | Self-Attention with GQA & KV-Cache | `view` (QKV projection split), `transpose(1, 2)` (seq-to-heads), `unsqueeze` + `expand` (GQA KV-broadcast: 8->16 or 2->16), `cat` (KV-cache append along seq dim), `reshape` (heads-to-seq output) | Multi-head tensor permutation layout propagation; non-contiguous attention score layouts; GQA asymmetric broadcasting. |
| **2** | **`Gemma4RotaryEmbedding`** | RoPE Complex Number Rotation | `split` (or `chunk`) into halves (dim=-1), `negate` (-x2), `cat` ([-x2, x1], dim=-1), `unsqueeze(2)` (freq table broadcast across heads) | Fusing split + negation + concatenation across non-trivial strides without copying buffer data; odd sequence RoPE alignment. |
| **3** | **`Gemma4TextMoE` / `Gemma4TextExperts`** | Sparse Mixture-of-Experts Layer | `topk` (top-8 expert selection), `reshape(-1)` (flatten token dispatch index array `[B*S, 8] -> [256]`), `t()` (expert weight transpose `[2816, 704] -> [704, 2816]`), `cat` / `reshape` (reassembling token outputs) | Int64 dispatch index tensor layout propagation (8 bytes/elem); non-power-of-2 intermediate dimension (I=704); batched GEMM layout compatibility. |
| **4** | **`Gemma4RMSNorm` + Residual** | Pre/Post Layer Normalization | `view` / `unsqueeze` (broadcasting 1D scaling weights across `[B, S, H]`), residual stream elementwise add | Normalization reduction layout fusion; residual stream dimension alignment; non-contiguous buffer preservation. |
| **5** | **`Gemma4VisionPatchEmbedder`** | Vision 2D Image Tokenizer | `Conv2d` (patch projection `[B, C, H, W] -> [B, D, H_p, W_p]`), `flatten(2)` (spatial grid collapse), `transpose(1, 2)` (`[B, D, N] -> [B, N, D]`), `cat` (CLS/prefix token concatenation) | Spatial 2D grid to 1D sequence token conversion; CLS token stitching; non-power-of-2 visual channel strides. |
| **6** | **`Gemma4VisionEncoderLayer`** | Vision Multi-Head Transformer Block | `view` (QKV split with non-power-of-2 head dim D=72), `transpose(1, 2)` (head permutation), `transpose(-2, -1)` (K^T MatMul), `contiguous` enforcement, `split` (MLP gate-up) | Stick boundary alignment for non-power-of-2 head dimension (D=72 = 144 bytes = 2.25 sticks); post-transpose contiguous copy elimination. |
| **7** | **`Gemma4VisionProjector` (Bridge)** | Vision-to-Text Multi-modal Connector | `view` / `reshape` (spatial 2D token grid downsampling), `cat` (combining vision tokens `[B, 280, 2816]` with prompt text tokens `[B, S, 2816]`) | Multi-modal sequence stitching; cross-module tensor dimension elevation (hidden_size 1152 -> 2816). |

---

## 3. Mathematical Consistency Rules for Valid User Configuration Changes

When users and downstream serving frameworks (such as vLLM, HuggingFace, or TGI) modify the Gemma-4 configuration file, tensor shapes at every layer change accordingly. To guarantee validity without triggering runtime shape mismatches, all parameter modifications must satisfy the following mathematical and architectural constraints:

### 3.1 Mathematical Constraints & Invariants

1. **Attention Head Divisibility**:
   `hidden_size = num_attention_heads * head_dim`
   *(or projected explicitly via dedicated W_q, W_k, W_v matrices where Q_dim = num_heads * head_dim).*
2. **GQA Group Divisibility (Grouped-Query Attention)**:
   `num_attention_heads % num_key_value_heads == 0`
   * Valid ratio examples:
     * 16 / 8 = 2x head expansion (Gemma-4 sliding attention default)
     * 16 / 2 = 8x head expansion (Gemma-4 full attention default)
     * 16 / 1 = 16x head expansion (Multi-Query Attention / MQA mode)
     * 8 / 4 = 2x head expansion (Downscaled mini configuration)
3. **MoE Top-K Capacity Constraint**:
   `top_k_experts <= num_experts`
   * Standard regime: top_k = 8, num_experts = 128
   * Downscaled CI regime: top_k = 2, num_experts = 8
4. **RoPE (Rotary Position Embedding) Constraints**:
   `rotary_dim <= head_dim` and `rotary_dim % 2 == 0`
   * Valid head dimensions: rotary_dim in {64, 128, 256, 512}.
5. **Vision Spatial Consistency**:
   `num_patches = (image_size / patch_size)^2`
   * For standard 256 x 256 image input with patch_size = 16: (256 / 16)^2 = 256 spatial patches (+ prefix/CLS tokens -> 280 total soft tokens).

---

### 3.2 The 5 Valid Configuration Regimes Tested in this FVT

The FVT test suite explicitly maps all test cases against these 5 configuration regimes to ensure comprehensive coverage of nominal and user-modified deployments:

| Configuration Regime | Purpose & Target Environment | Key Parameter Values | Resulting Tensor Shapes & Transformations |
|---|---|---|---|
| **Config A: Nominal 26B-A4B Baseline** | Standard production model validation | `hidden_size=2816`, `H=16`, `KV=8`, `D=256`, `E=128`, `K=8`, `moe_inter=704` | Standard sliding attention prefill `[1, 32, 2816]`, GQA 2x broadcast, MoE dispatch across 128 experts. |
| **Config B: Downscaled CI / Edge** | Fast CI regression, local debugging, edge deployment | `hidden_size=1024`, `H=8`, `KV=2`, `D=128`, `E=8`, `K=2`, `intermediate=512` | Reduced memory footprint, GQA 4x broadcast, 8-expert top-2 routing `[B*S, 8]`. |
| **Config C: Extreme GQA / MQA** | High-throughput serving optimization (MQA mode) | `hidden_size=4096`, `H=16`, `KV=1`, `D=256` | Single KV-head broadcast across all 16 attention heads (16x virtual expansion). |
| **Config D: Full Attention Wide Head Dim** | Global context full-attention layers | `hidden_size=2816`, `H=16`, `KV=2`, `D=512` (global), `vocab=262144` | Wide head projection (D=512), 8x GQA broadcast, large vocab softcapping `[B, S, 262144]`. |
| **Config E: Arbitrary Batch & Odd Sequence Sweeps** | Real-world dynamic serving & unaligned context lengths | Multi-batch: B in {3, 5, 7}; Odd sequence lengths: S in {1, 13, 33, 67, 71, 127} | Hardware stick-boundary alignment tests verifying non-power-of-2 and non-divisible spatial layouts. |

---

### 3.3 Parameter Sweep Ranges (`[GEMMA-PARAM-SWEEP]`)

To systematically validate dynamic serving workloads and user overrides, the following explicit ranges are evaluated across the test suite:

| Hyperparameter | Baseline (Config A) | Sweep Range Tested | Test Scenario Context |
|---|---|---|---|
| **Batch Size (B)** | `1` | `1, 2, 3, 4, 5, 7, 8` | Tests single-stream decode (B=1), power-of-2 batches (2, 4, 8), and odd prime batch sizes (3, 5, 7) across all view/permute axes. |
| **Sequence Length (S)** | `32` | `1, 13, 30, 32, 33, 64, 67, 71, 127, 128, 1024` | Single-token decode (S=1), unaligned/prime sequence lengths (13, 33, 67, 71, 127), standard prefill (32, 64, 128), and sliding window limit (1024). |
| **Attention Heads (H)** | `16` | `8, 16, 32` | Downscaled mini-model (H=8), standard Gemma (H=16), scaled wide attention (H=32). |
| **Key/Value Heads (KV_H)** | `8` (sliding) / `2` (full) | `1, 2, 4, 8, 16` | MQA mode (KV=1), extreme GQA (KV=2), standard GQA (KV=4, 8), and full MHA (KV=16). |
| **GQA Expansion Factor** | `2x` | `1x, 2x, 4x, 8x, 16x` | Multi-head key/value broadcasting factors (H / KV_H). |
| **Head Dimension (D)** | `256` | `64, 72, 128, 256, 512` | Downscaled (64, 128), vision irregular (72), standard sliding (256), and global full attention (512). |
| **MoE Total Experts (E)** | `128` | `8, 16, 64, 128` | Downscaled CI configurations (E=8, 16) up to full 26B model (E=128). |
| **MoE Top-K Active (K)** | `8` | `1, 2, 4, 8` | Single-expert top-1 routing (K=1), downscaled top-2 (K=2), standard top-8 (K=8). |
| **MoE Intermediate Size (I_moe)** | `704` | `256, 512, 704, 1408` | Non-power-of-2 Gemma baseline (704) and power-of-2 comparisons (256, 512, 1408). |

---

## 4. Test Environment Variables

| Variable | Default | Test Values | Purpose & Hardware Topology Mapping |
|---|---|---|---|
| `SENCORES` | 1 | 1, 2, 4, 8, 16, 32 | Number of Spyre cores enabled. `1` = Single-core compilation baseline; `4` = Quad-core cluster; `8` = Octa-core cluster; `16` = Half-chip topology; `32` = Full 32-core single-chip topology. |
| `LX_PLANNING` | 0 | 0, 1 | Enable LX scratchpad planning |
| `CO_OPTIMIZING_LX_PLANNING` | 0 | 0, 1 | Co-optimized LX layout placement |
| `GLOBAL_STICK_OPTIMIZER` | 1 | 0, 1 | Toggle stick optimizer for non-contiguous views |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | 1 | 0, 1 | View + pointwise/reduction kernel fusion verification |

---

## 4. Test Environment Variables

| Variable | Default | Test Values | Purpose & Hardware Topology Mapping |
|---|---|---|---|
| `SENCORES` | 1 | 1, 2, 4, 8, 16, 32 | Number of Spyre cores enabled. `1` = Single-core compilation baseline; `4` = Quad-core cluster; `8` = Octa-core cluster; `16` = Half-chip topology; `32` = Full 32-core single-chip topology. |
| `LX_PLANNING` | 0 | 0, 1 | Enable LX scratchpad planning |
| `CO_OPTIMIZING_LX_PLANNING` | 0 | 0, 1 | Co-optimized LX layout placement |
| `GLOBAL_STICK_OPTIMIZER` | 1 | 0, 1 | Toggle stick optimizer for non-contiguous views |
| `SPYRE_INDUCTOR_ENABLE_FUSION` | 1 | 0, 1 | View + pointwise/reduction kernel fusion verification |

---

## 5. Table of Contents

| Section | Category | Test IDs | Count |
|---|---|---|---|
| **Section 1** | Module-Level End-to-End Gemma Tests | `GMOD-01` .. `GMOD-15` | 15 |
| **Section 2** | `reshape` & `view` Tensor Transformations | `GRSH-01` .. `GRSH-20` | 20 |
| **Section 3** | `squeeze` & `unsqueeze` Broadcast & Dimension Alignment | `GSQ-01` .. `GSQ-16` | 16 |
| **Section 4** | `expand` & Multi-Head Broadcast Patterns | `GEXP-01` .. `GEXP-14` | 14 |
| **Section 5** | `transpose` & `t` Multidimensional Permutations | `GTRN-01` .. `GTRN-16` | 16 |
| **Section 6** | `split` & `chunk` Projection Decompositions | `GSPL-01` .. `GSPL-14` | 14 |
| **Section 7** | `cat` Sequence Extension & Rotary Stitching | `GCAT-01` .. `GCAT-14` | 14 |
| **Section 8** | `select` & Slice Decimations | `GSEL-01` .. `GSEL-12` | 12 |
| **Section 9** | `clone` & `contiguous` Layout Enforcement | `GCNT-01` .. `GCNT-12` | 12 |
| **Section 10** | High-Rank 5D & 6D Advanced Vision/Paged KV Workloads | `GHRK-01` .. `GHRK-10` | 10 |
| — | **Total Test Cases** | | **143** |

---

## Section 1 — Module-Level End-to-End Gemma Tests

Tests the complete forward execution of entire Gemma-4 sub-modules compiled end-to-end on Spyre.

| ID | Test Name | Module Under Test | Config Mapping | Classification | Module Details & Specification | Input Tensor Dimensions & Dtype | Expected Output Dimensions | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|---|
| `GMOD-01` | `test_module_text_attention_sliding_prefill` | `Gemma4TextAttention` | **Config A** | `[NET-NEW-SCENARIO]` | Sliding attention (H=16, KV=8, D=256) | `hidden_states`: `[1, 32, 2816]` (`bfloat16`), `attn_mask`: `[1, 1, 32, 32]` | `attn_output`: `[1, 32, 2816]`, `kv_cache`: `[1, 8, 32, 256]` | Multi-head QKV view, head transpose, GQA expand, score view, output linear reshape | `compare_with_cpu`, `atol=1e-2` |
| `GMOD-02` | `test_module_text_attention_full_decode` | `Gemma4TextAttention` | **Config D** | `[NET-NEW-SCENARIO]` | Full global attention (H=16, KV=2, D=512) | `hidden_states`: `[1, 1, 2816]` (`bfloat16`), `past_kv`: `[1, 2, 128, 512]` | `attn_output`: `[1, 1, 2816]`, `kv_cache`: `[1, 2, 129, 512]` | Single-token decode view, KV-cache concatenation along sequence dim, 8x head broadcast | `compare_with_cpu`, `atol=1e-2` |
| `GMOD-03` | `test_module_text_attention_batched_odd_seq` | `Gemma4TextAttention` | **Config B / E** | `[GEMMA-PARAM-SWEEP]` | Custom downscaled (H=8, KV=4, D=128) | `hidden_states`: `[3, 67, 1024]` (`bfloat16`) | `attn_output`: `[3, 67, 1024]` | Non-aligned batch (B=3) and odd sequence length (S=67) handling through attention view-transpose chains | `compare_with_cpu`, `atol=1e-2` |
| `GMOD-04` | `test_module_rotary_embedding_prefill` | `Gemma4RotaryEmbedding` | **Config A** | `[NET-NEW-SCENARIO]` | Rotary position embedding (dim=64, max_pos=4096) | `x`: `[1, 16, 32, 256]`, `position_ids`: `[1, 32]` (`int64`) | `rotated_x`: `[1, 16, 32, 256]` | RoPE split half, unsqueeze freq broadcast, negate, cat rotation, contiguous enforcement | `compare_with_cpu`, `atol=1e-3` |
| `GMOD-05` | `test_module_rotary_embedding_decode` | `Gemma4RotaryEmbedding` | **Config A / E** | `[NET-NEW-SCENARIO]` | Rotary decode step (dim=64, max_pos=4096) | `x`: `[4, 8, 1, 256]`, `position_ids`: `[4, 1]` (`int64`) | `rotated_x`: `[4, 8, 1, 256]` | Single-token decode frequency broadcast, unsqueeze at dim=2 (B=4) | `compare_with_cpu`, `atol=1e-3` |
| `GMOD-06` | `test_module_moe_router_topk` | `Gemma4TextMoE` / Router | **Config A** | `[NET-NEW-SCENARIO]` | MoE gating router (hidden_size=2816, num_exp=128, top_k=8) | `hidden_states`: `[32, 2816]` (`bfloat16`) | `router_weights`: `[32, 8]`, `selected_experts`: `[32, 8]` | Linear projection, softmax, top-k selection, weight normalization div | `compare_with_cpu`, `atol=1e-3` |
| `GMOD-07` | `test_module_moe_experts_scatter_gather` | `Gemma4TextMoE` / Experts | **Config A** | `[NET-NEW-SCENARIO]` | Batched expert MLP (E=128, K=8, hidden=2816, inter=704) | `hidden_states`: `[32, 2816]`, `topk_idx`: `[32, 8]`, `topk_wt`: `[32, 8]` | `output`: `[32, 2816]` (`bfloat16`) | Flatten indices `reshape(-1)`, token indexing `x[idx]`, batched expert MLP, weight mul & sum | `compare_with_cpu`, `atol=2e-2` |
| `GMOD-08` | `test_module_dense_mlp` | `Gemma4MLP` | **Config A** | `[NET-NEW-SCENARIO]` | Dense non-MoE MLP (hidden_size=2816, intermediate=2112) | `x`: `[2, 32, 2816]` (`bfloat16`) | `out`: `[2, 32, 2816]` (`bfloat16`) | Gate/Up linear projection, GeLU tanh activation, elementwise mul, down projection | `compare_with_cpu`, `atol=1e-2` |
| `GMOD-09` | `test_module_vision_patch_embedder` | `Gemma4VisionPatchEmbedder` | **Config A** | `[NET-NEW-SCENARIO]` | Vision 2D patch embedder (patch=16, hidden=1152) | `pixel_values`: `[1, 3, 256, 256]` (`bfloat16`) | `patches`: `[1, 256, 1152]` (`bfloat16`) | Conv2d patch projection, flattening 2D spatial patches `[B, C, H, W] -> [B, N, D]`, transpose | `compare_with_cpu`, `atol=1e-2` |
| `GMOD-10` | `test_module_vision_encoder_layer` | `Gemma4VisionEncoderLayer` | **Config A** | `[NET-NEW-SCENARIO]` | Vision transformer block (hidden=1152, H=16, D=72) | `hidden_states`: `[1, 280, 1152]` (`bfloat16`) | `layer_out`: `[1, 280, 1152]` (`bfloat16`) | Complete vision self-attention with D=72, layer-norm, post-attn norm, MLP block | `compare_with_cpu`, `atol=1e-2` |
| `GMOD-11` | `test_module_text_decoder_layer_sliding` | `Gemma4TextDecoderLayer` | **Config A** | `[NET-NEW-SCENARIO]` | Sliding decoder layer with integrated MoE | `hidden_states`: `[1, 32, 2816]` (`bfloat16`) | `layer_out`: `[1, 32, 2816]` (`bfloat16`) | End-to-end decoder layer chaining attention views + RMSNorm + MoE routing + residual adds | `compare_with_cpu`, `atol=2e-2` |
| `GMOD-12` | `test_module_text_decoder_layer_full` | `Gemma4TextDecoderLayer` | **Config D** | `[NET-NEW-SCENARIO]` | Full attention decoder layer with dense MLP | `hidden_states`: `[1, 32, 2816]` (`bfloat16`) | `layer_out`: `[1, 32, 2816]` (`bfloat16`) | Full attention decoder layer with global head dims and non-MoE MLP | `compare_with_cpu`, `atol=2e-2` |
| `GMOD-13` | `test_module_rms_norm` | `Gemma4RMSNorm` | **Config A** | `[NET-NEW-SCENARIO]` | Pre/Post Layer RMSNorm (dim=2816, eps=1e-6) | `x`: `[4, 128, 2816]` (`bfloat16`) | `out`: `[4, 128, 2816]` (`bfloat16`) | Reduction along last dim, rsqrt, unsqueeze broadcast, scale mul | `compare_with_cpu`, `atol=1e-3` |
| `GMOD-14` | `test_module_token_embedding` | `Gemma4Embedding` | **Config A** | `[NET-NEW-SCENARIO]` | Token embedding table lookup (vocab=262144, hidden=2816) | `input_ids`: `[2, 64]` (`int64`) | `embeds`: `[2, 64, 2816]` (`bfloat16`) | Table lookup indexing followed by embedding multiplier scaling | `compare_with_cpu`, `atol=0.0` |
| `GMOD-15` | `test_module_vision_pooling_and_projection` | `Gemma4VisionProjector` | **Config A** | `[NET-NEW-SCENARIO]` | Multi-modal connector bridge (pool=3, hidden=1152 -> 2816) | `features`: `[1, 280, 1152]` (`bfloat16`) | `pooled`: `[1, 94, 1152]` (`bfloat16`) | 1D adaptive average pooling across patch tokens, reshape, and linear projection | `compare_with_cpu`, `atol=1e-2` |

---

## Section 2 — `reshape` & `view` Tensor Transformations

Validates memory layout reinterpretations, tensor folding, unfolding, and multidimensional flattening across 1D to 6D.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GRSH-01` | `test_view_qkv_split_standard` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(B, S, H, D)` | `[1, 32, 4096]` (`bfloat16`) | `[1, 32, 16, 256]` | 3D to 4D unflattening for standard sliding Q projection | `compare_with_cpu`, `atol=0.0` |
| `GRSH-02` | `test_view_qkv_split_full_head` | **Config D** | `[NET-NEW-SCENARIO]` | `x.view(B, S, H, D)` | `[1, 32, 8192]` (`bfloat16`) | `[1, 32, 16, 512]` | 3D to 4D unflattening for full global head dim (D=512) | `compare_with_cpu`, `atol=0.0` |
| `GRSH-03` | `test_view_qkv_split_kv_heads` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(B, S, KV_H, D)` | `[1, 32, 2048]` (`bfloat16`) | `[1, 32, 8, 256]` | 3D to 4D unflattening for GQA KV projection | `compare_with_cpu`, `atol=0.0` |
| `GRSH-04` | `test_view_qkv_split_global_kv` | **Config D** | `[NET-NEW-SCENARIO]` | `x.view(B, S, KV_H, D)` | `[1, 32, 1024]` (`bfloat16`) | `[1, 32, 2, 512]` | 3D to 4D unflattening for 2-head global KV projection | `compare_with_cpu`, `atol=0.0` |
| `GRSH-05` | `test_reshape_attn_out_flatten` | **Config A** | `[NET-NEW-SCENARIO]` | `x.reshape(B, S, H * D)` | `[1, 32, 16, 256]` (`bfloat16`) | `[1, 32, 4096]` | 4D to 3D flattening post attention computation | `compare_with_cpu`, `atol=0.0` |
| `GRSH-06` | `test_reshape_attn_out_non_contiguous` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.transpose(1, 2).reshape(B, S, H * D)` | `[2, 16, 64, 256]` (`bfloat16`) | `[2, 64, 4096]` | Reshape requiring implicit copy/layout conversion on non-contiguous tensor | `compare_with_cpu`, `atol=0.0` |
| `GRSH-07` | `test_view_moe_token_flatten` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.view(-1, hidden_size)` | `[4, 32, 2816]` (`bfloat16`) | `[128, 2816]` | Batch-sequence flattening `[B, S, H] -> [B*S, H]` for expert routing (B=4) | `compare_with_cpu`, `atol=0.0` |
| `GRSH-08` | `test_view_moe_token_unflatten` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.view(B, S, hidden_size)` | `[128, 2816]` (`bfloat16`) | `[4, 32, 2816]` | Unflattening combined tokens back to `[B, S, H]` post expert aggregation | `compare_with_cpu`, `atol=0.0` |
| `GRSH-09` | `test_reshape_moe_topk_flatten` | **Config A** | `[NET-NEW-SCENARIO]` | `x.reshape(-1)` | `[32, 8]` (`int64`) | `[256]` | 2D to 1D flattening for expert dispatch index array | `compare_with_cpu`, `atol=0.0` |
| `GRSH-10` | `test_view_vision_spatial_to_sequence` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(B, C, -1).transpose(1, 2)` | `[2, 1152, 16, 16]` (`bfloat16`) | `[2, 256, 1152]` | Vision patch 2D grid flattening into 1D sequence tokens | `compare_with_cpu`, `atol=0.0` |
| `GRSH-11` | `test_view_vision_sequence_to_spatial` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(B, H_p, W_p, C)` | `[2, 256, 1152]` (`bfloat16`) | `[2, 16, 16, 1152]` | 3D token sequence unflattening back to 4D spatial representation | `compare_with_cpu`, `atol=0.0` |
| `GRSH-12` | `test_view_1d_to_2d` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.view(128, 22)` | `[2816]` (`bfloat16`) | `[128, 22]` | 1D flat buffer folding into 2D matrix (2816 = 128 * 22) | `compare_with_cpu`, `atol=0.0` |
| `GRSH-13` | `test_view_2d_to_1d` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.view(-1)` | `[64, 44]` (`bfloat16`) | `[2816]` | 2D matrix collapse into 1D vector | `compare_with_cpu`, `atol=0.0` |
| `GRSH-14` | `test_view_odd_batch_and_seq` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.view(3, 71, 8, 256)` | `[3, 71, 2048]` (`bfloat16`) | `[3, 71, 8, 256]` | Non-power-of-2 batch (B=3) and prime sequence length (S=71) unflattening | `compare_with_cpu`, `atol=0.0` |
| `GRSH-15` | `test_reshape_decode_single_token` | **Config A** | `[NET-NEW-SCENARIO]` | `x.reshape(1, 1, 16, 256)` | `[1, 1, 4096]` (`bfloat16`) | `[1, 1, 16, 256]` | Single-token decode projection reshape | `compare_with_cpu`, `atol=0.0` |
| `GRSH-16` | `test_view_vision_heads_split` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.view(B, N, 16, 72)` | `[1, 280, 1152]` (`bfloat16`) | `[1, 280, 16, 72]` | Vision attention QKV split with non-power-of-2 head dim (D=72) | `compare_with_cpu`, `atol=0.0` |
| `GRSH-17` | `test_reshape_chained_multiview` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(1, 32, 16, 256).reshape(1, 512, 256)` | `[1, 32, 4096]` (`bfloat16`) | `[1, 512, 256]` | Chained multi-step view-to-reshape transformations | `compare_with_cpu`, `atol=0.0` |
| `GRSH-18` | `test_view_inferred_dimension` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.view(2, -1, 256)` | `[2, 32, 16, 256]` (`bfloat16`) | `[2, 512, 256]` | Negative index `-1` dimension inference on Spyre inductor | `compare_with_cpu`, `atol=0.0` |
| `GRSH-19` | `test_view_5d_paged_cache_block` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(16, 8, 16, 16, 16)` | `[16, 8, 256, 256]` (`bfloat16`) | `[16, 8, 16, 16, 16]` | 4D KV-cache partitioned into 5D blocked structure | `compare_with_cpu`, `atol=0.0` |
| `GRSH-20` | `test_reshape_scalar_to_1d` | **Config A** | `[NET-NEW-SCENARIO]` | `x.reshape(1)` | `[]` (0D scalar, `bfloat16`) | `[1]` | 0D scalar rank elevation to 1D tensor | `compare_with_cpu`, `atol=0.0` |

---

## Section 3 — `squeeze` & `unsqueeze` Broadcast & Dimension Alignment

Validates singleton dimension insertion, removal, and rank-changing operations across all axes.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GSQ-01` | `test_unsqueeze_rope_freq_dim2` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(2)` | `[1, 32, 256]` (`bfloat16`) | `[1, 32, 1, 256]` | RoPE frequency broadcast across attention head dimension | `compare_with_cpu`, `atol=0.0` |
| `GSQ-02` | `test_unsqueeze_attn_mask_dim1_dim2` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(1).unsqueeze(2)` | `[1, 32]` (`bfloat16`) | `[1, 1, 1, 32]` | Attention mask broadcast alignment for multi-head SDPA | `compare_with_cpu`, `atol=0.0` |
| `GSQ-03` | `test_unsqueeze_moe_expert_mask_dim_neg1` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(-1)` | `[256]` (`bool`) | `[256, 1]` | Boolean expert capacity mask alignment along trailing dimension | `compare_with_cpu`, `atol=0.0` |
| `GSQ-04` | `test_unsqueeze_leading_dim0` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(0)` | `[32, 2816]` (`bfloat16`) | `[1, 32, 2816]` | Batch dimension insertion at leading position | `compare_with_cpu`, `atol=0.0` |
| `GSQ-05` | `test_unsqueeze_scalar_to_1d` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(0)` | `[]` (scalar `bfloat16`) | `[1]` | 0D scalar to 1D unsqueeze | `compare_with_cpu`, `atol=0.0` |
| `GSQ-06` | `test_unsqueeze_4d_to_5d` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(2)` | `[2, 8, 32, 256]` (`bfloat16`) | `[2, 8, 1, 32, 256]` | 4D attention state elevated to 5D for group-query partitioning | `compare_with_cpu`, `atol=0.0` |
| `GSQ-07` | `test_squeeze_all_singletons` | **Config A** | `[NET-NEW-SCENARIO]` | `x.squeeze()` | `[1, 1, 32, 1, 256]` (`bfloat16`) | `[32, 256]` | Removal of all singleton dimensions in a 5D tensor | `compare_with_cpu`, `atol=0.0` |
| `GSQ-08` | `test_squeeze_specific_dim0` | **Config A** | `[NET-NEW-SCENARIO]` | `x.squeeze(0)` | `[1, 32, 2816]` (`bfloat16`) | `[32, 2816]` | Squeezing batch dimension at dim 0 | `compare_with_cpu`, `atol=0.0` |
| `GSQ-09` | `test_squeeze_specific_dim1` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.squeeze(1)` | `[2, 1, 16, 256]` (`bfloat16`) | `[2, 16, 256]` | Squeezing intermediate singleton head/sequence dimension | `compare_with_cpu`, `atol=0.0` |
| `GSQ-10` | `test_squeeze_trailing_dim_neg1` | **Config A** | `[NET-NEW-SCENARIO]` | `x.squeeze(-1)` | `[32, 8, 1]` (`float32`) | `[32, 8]` | Squeezing trailing reduction dimension post sum/top-k | `compare_with_cpu`, `atol=0.0` |
| `GSQ-11` | `test_squeeze_noop_non_singleton` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.squeeze(1)` | `[1, 32, 2816]` (`bfloat16`) | `[1, 32, 2816]` | Verifying no-op behavior when targeted dim size != 1 | `compare_with_cpu`, `atol=0.0` |
| `GSQ-12` | `test_squeeze_decode_seq_dim1` | **Config A** | `[NET-NEW-SCENARIO]` | `x.squeeze(1)` | `[1, 1, 16, 256]` (`bfloat16`) | `[1, 16, 256]` | Decode single-token step squeezing sequence dimension | `compare_with_cpu`, `atol=0.0` |
| `GSQ-13` | `test_unsqueeze_multiple_axes` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.unsqueeze(0).unsqueeze(3)` | `[32, 256]` (`bfloat16`) | `[1, 32, 256, 1]` | Dual-axis unsqueeze at boundary positions | `compare_with_cpu`, `atol=0.0` |
| `GSQ-14` | `test_squeeze_multiple_tuple_dims` | **Config A** | `[NET-NEW-SCENARIO]` | `x.squeeze((0, 2))` | `[1, 16, 1, 256]` (`bfloat16`) | `[16, 256]` | Multi-dimension tuple squeeze | `compare_with_cpu`, `atol=0.0` |
| `GSQ-15` | `test_unsqueeze_vision_token_pooling` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(1)` | `[2, 280, 1152]` (`bfloat16`) | `[2, 1, 280, 1152]` | Vision token tensor unsqueeze for 2D pooling broadcast | `compare_with_cpu`, `atol=0.0` |
| `GSQ-16` | `test_unsqueeze_6d_tensor` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(3)` | `[1, 2, 4, 8, 16]` (`bfloat16`) | `[1, 2, 4, 1, 8, 16]` | High-rank 5D to 6D unsqueeze transformation | `compare_with_cpu`, `atol=0.0` |

---

## Section 4 — `expand` & Multi-Head Broadcast Patterns

Validates zero-copy virtual broadcasting of singleton dimensions across standard and asymmetric configurations.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GEXP-01` | `test_expand_gqa_kv_heads_sliding` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(2).expand(B, KV_H, 2, S, D).reshape(B, H, S, D)` | `[1, 8, 32, 256]` (`bfloat16`) | `[1, 16, 32, 256]` | GQA key/value head broadcasting (8 -> 16 heads, factor=2) | `compare_with_cpu`, `atol=0.0` |
| `GEXP-02` | `test_expand_gqa_kv_heads_full` | **Config D** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(2).expand(B, KV_H, 8, S, D).reshape(B, H, S, D)` | `[1, 2, 32, 512]` (`bfloat16`) | `[1, 16, 32, 512]` | GQA broadcasting for full layer (2 -> 16 heads, factor=8) | `compare_with_cpu`, `atol=0.0` |
| `GEXP-03` | `test_expand_attn_mask_batch_and_heads` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.expand(B, H, S, S)` | `[1, 1, 32, 32]` (`bfloat16`) | `[4, 16, 32, 32]` | Attention mask 2D broadcast across batch size (4) and heads (16) | `compare_with_cpu`, `atol=0.0` |
| `GEXP-04` | `test_expand_rope_freqs_heads` | **Config A** | `[NET-NEW-SCENARIO]` | `x.expand(B, S, H, D)` | `[1, 32, 1, 256]` (`bfloat16`) | `[1, 32, 16, 256]` | RoPE rotary frequencies expanded across 16 attention heads | `compare_with_cpu`, `atol=0.0` |
| `GEXP-05` | `test_expand_decode_single_token_heads` | **Config A** | `[NET-NEW-SCENARIO]` | `x.expand(1, 16, 1, 256)` | `[1, 1, 1, 256]` (`bfloat16`) | `[1, 16, 1, 256]` | Single-token decode broadcasting across heads | `compare_with_cpu`, `atol=0.0` |
| `GEXP-06` | `test_expand_moe_routing_weights` | **Config A** | `[NET-NEW-SCENARIO]` | `x.expand(32, 8, 2816)` | `[32, 8, 1]` (`float32`) | `[32, 8, 2816]` | Gating weights broadcast across hidden channels for expert blending | `compare_with_cpu`, `atol=0.0` |
| `GEXP-07` | `test_expand_1d_bias_to_3d` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.expand(2, 32, 2816)` | `[2816]` (`bfloat16`) | `[2, 32, 2816]` | 1D bias/scaling vector broadcast across batch and sequence dims | `compare_with_cpu`, `atol=0.0` |
| `GEXP-08` | `test_expand_odd_batch_and_heads` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.expand(3, 7, 33, 256)` | `[1, 1, 33, 256]` (`bfloat16`) | `[3, 7, 33, 256]` | Non-power-of-2 broadcast (B=3, H=7, S=33) | `compare_with_cpu`, `atol=0.0` |
| `GEXP-09` | `test_expand_minus_one_unmodified_dims` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.expand(-1, 16, -1, -1)` | `[2, 1, 64, 256]` (`bfloat16`) | `[2, 16, 64, 256]` | Expand utilizing `-1` for preserving existing dimension sizes | `compare_with_cpu`, `atol=0.0` |
| `GEXP-10` | `test_expand_vision_class_token` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.expand(2, 1, 1152)` | `[1, 1, 1152]` (`bfloat16`) | `[2, 1, 1152]` | Vision CLS token broadcast across batch size | `compare_with_cpu`, `atol=0.0` |
| `GEXP-11` | `test_expand_sliding_window_local_mask` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.expand(2, 16, 1024, 1024)` | `[1, 1, 1024, 1024]` (`bfloat16`) | `[2, 16, 1024, 1024]` | 1024-token sliding window causal mask broadcast | `compare_with_cpu`, `atol=0.0` |
| `GEXP-12` | `test_expand_followed_by_pointwise_add` | **Config A** | `[NET-NEW-SCENARIO]` | `(x.expand(1, 16, 32, 256) + y)` | `x`: `[1, 1, 32, 256]`, `y`: `[1, 16, 32, 256]` | `[1, 16, 32, 256]` | Broadcaster followed immediately by compiled pointwise addition | `compare_with_cpu`, `atol=1e-3` |
| `GEXP-13` | `test_expand_5d_kv_paged_blocks` | **Config A** | `[NET-NEW-SCENARIO]` | `x.expand(16, 8, 4, 16, 16)` | `[16, 1, 4, 16, 16]` (`bfloat16`) | `[16, 8, 4, 16, 16]` | 5D block cache broadcast across KV heads | `compare_with_cpu`, `atol=0.0` |
| `GEXP-14` | `test_expand_scalar_to_4d` | **Config A** | `[NET-NEW-SCENARIO]` | `x.expand(1, 16, 32, 32)` | `[]` (scalar `bfloat16`) | `[1, 16, 32, 32]` | 0D scalar constant broadcast across 4D attention score grid | `compare_with_cpu`, `atol=0.0` |

---

## Section 5 — `transpose` & `t` Multidimensional Permutations

Validates stride manipulations, 2D matrix transpositions, and ND axes reorderings.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GTRN-01` | `test_transpose_mha_qkv_seq_heads` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(1, 2)` | `[1, 32, 16, 256]` (`bfloat16`) | `[1, 16, 32, 256]` | Attention head transposition `[B, S, H, D] -> [B, H, S, D]` | `compare_with_cpu`, `atol=0.0` |
| `GTRN-02` | `test_transpose_mha_key_for_score_gemm` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(-2, -1)` | `[1, 16, 32, 256]` (`bfloat16`) | `[1, 16, 256, 32]` | Key matrix transposition K^T for Q x K^T matrix multiplication | `compare_with_cpu`, `atol=0.0` |
| `GTRN-03` | `test_transpose_mha_out_heads_seq` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(1, 2)` | `[1, 16, 32, 256]` (`bfloat16`) | `[1, 32, 16, 256]` | Attention score output transpose back to sequence-major before linear projection | `compare_with_cpu`, `atol=0.0` |
| `GTRN-04` | `test_transpose_vision_qkv_seq_heads` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.transpose(1, 2)` | `[1, 280, 16, 72]` (`bfloat16`) | `[1, 16, 280, 72]` | Vision attention head transpose with non-power-of-2 head dim (D=72) | `compare_with_cpu`, `atol=0.0` |
| `GTRN-05` | `test_transpose_vision_key_gemm` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.transpose(-2, -1)` | `[1, 16, 280, 72]` (`bfloat16`) | `[1, 16, 72, 280]` | Vision key matrix transposition K^T | `compare_with_cpu`, `atol=0.0` |
| `GTRN-06` | `test_t_2d_linear_weight_gemm` | **Config A** | `[NET-NEW-SCENARIO]` | `x.t()` | `[2816, 4096]` (`bfloat16`) | `[4096, 2816]` | 2D weight matrix transposition for GEMM kernel execution | `compare_with_cpu`, `atol=0.0` |
| `GTRN-07` | `test_t_2d_moe_expert_weight` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.t()` | `[2816, 704]` (`bfloat16`) | `[704, 2816]` | 2D MoE expert up/gate projection weight transposition (704 x 2816) | `compare_with_cpu`, `atol=0.0` |
| `GTRN-08` | `test_transpose_moe_batched_expert_weights` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(-2, -1)` | `[128, 1408, 2816]` (`bfloat16`) | `[128, 2816, 1408]` | Batched 3D expert weight transposition across 128 experts | `compare_with_cpu`, `atol=0.0` |
| `GTRN-09` | `test_transpose_batched_odd_seq_decode` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.transpose(1, 2)` | `[3, 1, 8, 256]` (`bfloat16`) | `[3, 8, 1, 256]` | Decode phase batched single-token head transpose (B=3) | `compare_with_cpu`, `atol=0.0` |
| `GTRN-10` | `test_transpose_leading_batch_and_seq` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.transpose(0, 1)` | `[2, 64, 2816]` (`bfloat16`) | `[64, 2, 2816]` | Batch and sequence dimension permutation `[B, S, H] -> [S, B, H]` | `compare_with_cpu`, `atol=0.0` |
| `GTRN-11` | `test_transpose_5d_tensor` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(2, 3)` | `[2, 4, 8, 16, 32]` (`bfloat16`) | `[2, 4, 16, 8, 32]` | 5D tensor intermediate axis swap | `compare_with_cpu`, `atol=0.0` |
| `GTRN-12` | `test_transpose_6d_tensor` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(4, 5)` | `[1, 2, 3, 4, 8, 16]` (`bfloat16`) | `[1, 2, 3, 4, 16, 8]` | 6D tensor trailing axis swap | `compare_with_cpu`, `atol=0.0` |
| `GTRN-13` | `test_transpose_chained_double_swap` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(1, 2).transpose(2, 3)` | `[1, 32, 16, 256]` (`bfloat16`) | `[1, 16, 256, 32]` | Compound chained transpose sequence | `compare_with_cpu`, `atol=0.0` |
| `GTRN-14` | `test_transpose_followed_by_bmm` | **Config A / E** | `[NET-NEW-SCENARIO]` | `torch.bmm(x.transpose(1, 2), y)` | `x`: `[4, 32, 64]`, `y`: `[4, 32, 128]` | `[4, 64, 128]` (`bfloat16`) | Non-contiguous transposed tensor fed directly into batched GEMM | `compare_with_cpu`, `atol=1e-2` |
| `GTRN-15` | `test_transpose_identity_same_dim` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.transpose(1, 1)` | `[2, 16, 32, 256]` (`bfloat16`) | `[2, 16, 32, 256]` | Identity transpose on matching dimension indices | `compare_with_cpu`, `atol=0.0` |
| `GTRN-16` | `test_t_2d_asymmetric_aspect_ratio` | **Config D** | `[NET-NEW-SCENARIO]` | `x.t()` | `[16, 262144]` (`bfloat16`) | `[262144, 16]` | Extreme aspect ratio 2D matrix transpose (vocab logit softcapping) | `compare_with_cpu`, `atol=0.0` |

---

## Section 6 — `split` & `chunk` Projection Decompositions

Validates tensor partitioning along hidden, head, and channel axes into multiple views.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GSPL-01` | `test_split_fused_qkv_equal` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.split(x, [4096, 2048, 2048], dim=-1)` | `[1, 32, 8192]` (`bfloat16`) | 3 tensors: `[1, 32, 4096]`, `[1, 32, 2048]`, `[1, 32, 2048]` | Fused QKV linear projection split for Gemma GQA (Q=16x256, K=8x256, V=8x256) | `compare_with_cpu`, `atol=0.0` |
| `GSPL-02` | `test_split_fused_qkv_global_heads` | **Config D** | `[NET-NEW-SCENARIO]` | `torch.split(x, [8192, 1024, 1024], dim=-1)` | `[1, 32, 10240]` (`bfloat16`) | 3 tensors: `[1, 32, 8192]`, `[1, 32, 1024]`, `[1, 32, 1024]` | Fused QKV split for full attention layer (Q=16x512, K=2x512, V=2x512) | `compare_with_cpu`, `atol=0.0` |
| `GSPL-03` | `test_split_moe_gate_up_fused` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `torch.split(x, [704, 704], dim=-1)` | `[32, 1408]` (`bfloat16`) | 2 tensors: `[32, 704]`, `[32, 704]` | Fused Gate-Up MLP projection split in MoE expert block (704 * 2 = 1408) | `compare_with_cpu`, `atol=0.0` |
| `GSPL-04` | `test_split_dense_mlp_gate_up_fused` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `torch.split(x, [2112, 2112], dim=-1)` | `[1, 32, 4224]` (`bfloat16`) | 2 tensors: `[1, 32, 2112]`, `[1, 32, 2112]` | Fused Gate-Up projection split in dense MLP block | `compare_with_cpu`, `atol=0.0` |
| `GSPL-05` | `test_chunk_rope_halves_dim_neg1` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.chunk(x, 2, dim=-1)` | `[1, 16, 32, 256]` (`bfloat16`) | 2 tensors: `[1, 16, 32, 128]`, `[1, 16, 32, 128]` | RoPE split head into first and second halves for complex rotation | `compare_with_cpu`, `atol=0.0` |
| `GSPL-06` | `test_chunk_rope_halves_global_head` | **Config D** | `[NET-NEW-SCENARIO]` | `torch.chunk(x, 2, dim=-1)` | `[1, 16, 32, 512]` (`bfloat16`) | 2 tensors: `[1, 16, 32, 256]`, `[1, 16, 32, 256]` | RoPE split for 512-dim full attention head | `compare_with_cpu`, `atol=0.0` |
| `GSPL-07` | `test_chunk_sequence_chunks_dim1` | **Config A / E** | `[GEMMA-PARAM-SWEEP]` | `torch.chunk(x, 4, dim=1)` | `[1, 128, 2816]` (`bfloat16`) | 4 tensors: each `[1, 32, 2816]` | Long sequence chunking into fixed context windows | `compare_with_cpu`, `atol=0.0` |
| `GSPL-08` | `test_chunk_batch_dim0` | **Config A / E** | `[GEMMA-PARAM-SWEEP]` | `torch.chunk(x, 2, dim=0)` | `[4, 32, 2816]` (`bfloat16`) | 2 tensors: each `[2, 32, 2816]` | Batch-parallel chunking (B=4) | `compare_with_cpu`, `atol=0.0` |
| `GSPL-09` | `test_split_uneven_sequence` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `torch.split(x, [30, 2], dim=1)` | `[1, 32, 2816]` (`bfloat16`) | 2 tensors: `[1, 30, 2816]`, `[1, 2, 2816]` | Prefix tokens vs trailing generated tokens partition | `compare_with_cpu`, `atol=0.0` |
| `GSPL-10` | `test_split_1d_tensor` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.split(x, [128, 128], dim=0)` | `[256]` (`int64`) | 2 tensors: each `[128]` | 1D index array splitting | `compare_with_cpu`, `atol=0.0` |
| `GSPL-11` | `test_chunk_vision_patches_dim1` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.chunk(x, 2, dim=1)` | `[1, 280, 1152]` (`bfloat16`) | 2 tensors: each `[1, 140, 1152]` | Vision token patch sequence split | `compare_with_cpu`, `atol=0.0` |
| `GSPL-12` | `test_chunk_5d_tensor_dim2` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.chunk(x, 2, dim=2)` | `[2, 8, 4, 16, 16]` (`bfloat16`) | 2 tensors: each `[2, 8, 2, 16, 16]` | 5D block tensor chunking | `compare_with_cpu`, `atol=0.0` |
| `GSPL-13` | `test_split_single_size_arg` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `torch.split(x, 64, dim=-1)` | `[1, 16, 32, 256]` (`bfloat16`) | 4 tensors: each `[1, 16, 32, 64]` | Integer split size parameter dividing tensor into equal partitions | `compare_with_cpu`, `atol=0.0` |
| `GSPL-14` | `test_split_followed_by_linear` | **Config A** | `[NET-NEW-SCENARIO]` | `q, k, v = torch.split(x, [256, 128, 128], dim=-1)` -> `F.linear(q, W)` | `x`: `[1, 32, 512]`, `W`: `[256, 256]` | `[1, 32, 256]` | Sliced chunk feeding directly into downstream linear projection | `compare_with_cpu`, `atol=1e-2` |

---

## Section 7 — `cat` Sequence Extension & Rotary Stitching

Validates tensor concatenations across sequence, hidden, head, and batch dimensions.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GCAT-01` | `test_cat_kv_cache_append_decode` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([past_k, new_k], dim=2)` | `past_k`: `[1, 8, 127, 256]`, `new_k`: `[1, 8, 1, 256]` (`bfloat16`) | `[1, 8, 128, 256]` | KV cache sequence extension during autoregressive single-token decode | `compare_with_cpu`, `atol=0.0` |
| `GCAT-02` | `test_cat_kv_cache_prefill_chunk` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([past_k, new_k], dim=2)` | `past_k`: `[1, 8, 64, 256]`, `new_k`: `[1, 8, 32, 256]` (`bfloat16`) | `[1, 8, 96, 256]` | Multi-token chunk KV cache extension | `compare_with_cpu`, `atol=0.0` |
| `GCAT-03` | `test_cat_rope_rotated_halves` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([-x2, x1], dim=-1)` | `x1`: `[1, 16, 32, 128]`, `x2`: `[1, 16, 32, 128]` (`bfloat16`) | `[1, 16, 32, 256]` | Rotary embedding recombination of negated and rotated halves | `compare_with_cpu`, `atol=0.0` |
| `GCAT-04` | `test_cat_rope_rotated_halves_global` | **Config D** | `[NET-NEW-SCENARIO]` | `torch.cat([-x2, x1], dim=-1)` | `x1`: `[1, 16, 32, 256]`, `x2`: `[1, 16, 32, 256]` (`bfloat16`) | `[1, 16, 32, 512]` | RoPE recombination for 512-dim full attention head | `compare_with_cpu`, `atol=0.0` |
| `GCAT-05` | `test_cat_vision_and_text_embeddings` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([vision_embeds, text_embeds], dim=1)` | `vision`: `[1, 280, 2816]`, `text`: `[1, 32, 2816]` (`bfloat16`) | `[1, 312, 2816]` | Multi-modal sequence concatenation along token sequence axis | `compare_with_cpu`, `atol=0.0` |
| `GCAT-06` | `test_cat_moe_expert_outputs` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([e0, e1, e2, e3], dim=0)` | 4 tensors: each `[8, 2816]` (`bfloat16`) | `[32, 2816]` | Assembling processed expert tokens back along batch dimension | `compare_with_cpu`, `atol=0.0` |
| `GCAT-07` | `test_cat_1d_position_ids` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([past_pos, new_pos], dim=0)` | `past_pos`: `[127]`, `new_pos`: `[1]` (`int64`) | `[128]` | 1D sequence position id extension | `compare_with_cpu`, `atol=0.0` |
| `GCAT-08` | `test_cat_attn_heads_dim1` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `torch.cat([h1, h2], dim=1)` | `h1`: `[1, 8, 32, 256]`, `h2`: `[1, 8, 32, 256]` (`bfloat16`) | `[1, 16, 32, 256]` | Concatenating partitioned head groups along head dimension | `compare_with_cpu`, `atol=0.0` |
| `GCAT-09` | `test_cat_batched_odd_seq_kv` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `torch.cat([k_past, k_new], dim=2)` | `k_past`: `[3, 8, 71, 256]`, `k_new`: `[3, 8, 1, 256]` (`bfloat16`) | `[3, 8, 72, 256]` | Multi-batch (B=3) odd sequence KV concatenation | `compare_with_cpu`, `atol=0.0` |
| `GCAT-10` | `test_cat_non_contiguous_inputs` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([x.transpose(1, 2), y.transpose(1, 2)], dim=2)` | `x`: `[1, 32, 8, 256]`, `y`: `[1, 16, 8, 256]` (`bfloat16`) | `[1, 8, 48, 256]` | Concatenating transposed non-contiguous tensors | `compare_with_cpu`, `atol=0.0` |
| `GCAT-11` | `test_cat_5d_paged_blocks` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([b1, b2], dim=0)` | `b1`: `[8, 8, 4, 16, 16]`, `b2`: `[8, 8, 4, 16, 16]` (`bfloat16`) | `[16, 8, 4, 16, 16]` | 5D block cache concatenation along block axis | `compare_with_cpu`, `atol=0.0` |
| `GCAT-12` | `test_cat_6d_tensors_dim3` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([t1, t2], dim=3)` | `t1`: `[1, 2, 2, 4, 8, 16]`, `t2`: `[1, 2, 2, 4, 8, 16]` (`bfloat16`) | `[1, 2, 2, 8, 8, 16]` | 6D tensor concatenation along middle axis | `compare_with_cpu`, `atol=0.0` |
| `GCAT-13` | `test_cat_three_way_qkv` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([q, k, v], dim=-1)` | `q`: `[1, 32, 4096]`, `k`: `[1, 32, 2048]`, `v`: `[1, 32, 2048]` | `[1, 32, 8192]` | Fusing individual Q, K, V activations into combined buffer | `compare_with_cpu`, `atol=0.0` |
| `GCAT-14` | `test_cat_followed_by_rmsnorm` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([a, b], dim=-1)` -> `RMSNorm` | `a`: `[1, 32, 1408]`, `b`: `[1, 32, 1408]` (`bfloat16`) | `[1, 32, 2816]` | Concatenation output feeding directly into fused normalization | `compare_with_cpu`, `atol=1e-3` |

---

## Section 8 — `select` & Slice Decimations

Validates coordinate indexing, single-token extractions, and rank-reducing selection operations.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GSEL-01` | `test_select_decode_last_token` | **Config A** | `[NET-NEW-SCENARIO]` | `x.select(dim=1, index=-1)` | `[1, 32, 2816]` (`bfloat16`) | `[1, 2816]` | Extracting final sequence token for autoregressive logit generation | `compare_with_cpu`, `atol=0.0` |
| `GSEL-02` | `test_select_single_batch` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.select(dim=0, index=0)` | `[4, 32, 2816]` (`bfloat16`) | `[32, 2816]` | Extracting single batch item from 3D tensor (B=4) | `compare_with_cpu`, `atol=0.0` |
| `GSEL-03` | `test_select_specific_head` | **Config A** | `[NET-NEW-SCENARIO]` | `x.select(dim=1, index=3)` | `[1, 16, 32, 256]` (`bfloat16`) | `[1, 32, 256]` | Selecting single attention head from 4D tensor | `compare_with_cpu`, `atol=0.0` |
| `GSEL-04` | `test_select_top1_expert_index` | **Config A** | `[NET-NEW-SCENARIO]` | `x.select(dim=-1, index=0)` | `[32, 8]` (`int64`) | `[32]` | Selecting primary top-1 expert assignment from top-k matrix | `compare_with_cpu`, `atol=0.0` |
| `GSEL-05` | `test_select_1d_element` | **Config A** | `[NET-NEW-SCENARIO]` | `x.select(dim=0, index=5)` | `[128]` (`float32`) | `[]` (scalar `float32`) | Extracting scalar element from 1D array | `compare_with_cpu`, `atol=0.0` |
| `GSEL-06` | `test_slice_sequence_window` | **Config A** | `[NET-NEW-SCENARIO]` | `x[:, :16, :]` | `[1, 32, 2816]` (`bfloat16`) | `[1, 16, 2816]` | Sliding window sequence sub-slice | `compare_with_cpu`, `atol=0.0` |
| `GSEL-07` | `test_slice_head_dim_half` | **Config A** | `[NET-NEW-SCENARIO]` | `x[:, :, :, :128]` | `[1, 16, 32, 256]` (`bfloat16`) | `[1, 16, 32, 128]` | Head dimension slice for partial RoPE application (50% rotary factor) | `compare_with_cpu`, `atol=0.0` |
| `GSEL-08` | `test_slice_vision_tokens_without_cls` | **Config A** | `[NET-NEW-SCENARIO]` | `x[:, 1:, :]` | `[1, 281, 1152]` (`bfloat16`) | `[1, 280, 1152]` | Stripping CLS token from vision sequence | `compare_with_cpu`, `atol=0.0` |
| `GSEL-09` | `test_select_5d_block_idx` | **Config A** | `[NET-NEW-SCENARIO]` | `x.select(dim=0, index=2)` | `[16, 8, 4, 16, 16]` (`bfloat16`) | `[8, 4, 16, 16]` | Slicing specific cache block from 5D paged buffer | `compare_with_cpu`, `atol=0.0` |
| `GSEL-10` | `test_select_6d_frame_idx` | **Config A** | `[NET-NEW-SCENARIO]` | `x.select(dim=1, index=0)` | `[2, 4, 16, 16, 16, 8]` (`bfloat16`) | `[2, 16, 16, 16, 8]` | Slicing first frame from 6D video-vision tensor | `compare_with_cpu`, `atol=0.0` |
| `GSEL-11` | `test_select_negative_index_head` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.select(dim=1, index=-2)` | `[2, 8, 64, 256]` (`bfloat16`) | `[2, 64, 256]` | Negative coordinate selection on head axis | `compare_with_cpu`, `atol=0.0` |
| `GSEL-12` | `test_select_followed_by_matmul` | **Config A / E** | `[NET-NEW-SCENARIO]` | `x.select(dim=1, index=-1) @ W` | `x`: `[2, 32, 2816]`, `W`: `[2816, 2816]` | `[2, 2816]` (`bfloat16`) | Selected decode token directly feeding final linear output projection | `compare_with_cpu`, `atol=1e-2` |

---

## Section 9 — `clone` & `contiguous` Layout Enforcement

Validates memory isolation and contiguous physical layout restoration on permuted, non-contiguous tensors.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GCNT-01` | `test_contiguous_after_head_transpose` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(1, 2).contiguous()` | `[1, 32, 16, 256]` (`bfloat16`) | `[1, 16, 32, 256]` | Enforcing contiguous layout post `[B, S, H, D] -> [B, H, S, D]` transpose before SDPA | `compare_with_cpu`, `atol=0.0` |
| `GCNT-02` | `test_contiguous_after_key_transpose` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(-2, -1).contiguous()` | `[1, 16, 32, 256]` (`bfloat16`) | `[1, 16, 256, 32]` | Enforcing contiguous layout on transposed key matrix before MatMul | `compare_with_cpu`, `atol=0.0` |
| `GCNT-03` | `test_clone_non_contiguous_attention_scores` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(1, 2).clone()` | `[1, 32, 16, 256]` (`bfloat16`) | `[1, 16, 32, 256]` | Deep-copy cloning of non-contiguous attention buffer | `compare_with_cpu`, `atol=0.0` |
| `GCNT-04` | `test_clone_memory_format_contiguous` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.clone(memory_format=torch.contiguous_format)` | `[2, 64, 8, 256]` (`bfloat16`) | `[2, 64, 8, 256]` | Explicit `contiguous_format` clone memory allocation | `compare_with_cpu`, `atol=0.0` |
| `GCNT-05` | `test_contiguous_after_strided_slice` | **Config A** | `[NET-NEW-SCENARIO]` | `x[:, ::2, :].contiguous()` | `[1, 64, 2816]` (`bfloat16`) | `[1, 32, 2816]` | Subsampling strided slice converted to contiguous physical memory | `compare_with_cpu`, `atol=0.0` |
| `GCNT-06` | `test_contiguous_already_contiguous_noop` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.contiguous()` | `[1, 32, 2816]` (`bfloat16`) | `[1, 32, 2816]` | Verifying zero-overhead no-op when tensor is already physically contiguous | `compare_with_cpu`, `atol=0.0` |
| `GCNT-07` | `test_contiguous_vision_patch_transpose` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.transpose(1, 2).contiguous()` | `[1, 280, 16, 72]` (`bfloat16`) | `[1, 16, 280, 72]` | Vision patch attention contiguous conversion with D=72 head dimension | `compare_with_cpu`, `atol=0.0` |
| `GCNT-08` | `test_clone_1d_tensor` | **Config D** | `[NET-NEW-SCENARIO]` | `x.clone()` | `[262144]` (`bfloat16`) | `[262144]` | Large 1D buffer clone (vocabulary projection weights) | `compare_with_cpu`, `atol=0.0` |
| `GCNT-09` | `test_contiguous_5d_tensor_after_permute` | **Config A** | `[NET-NEW-SCENARIO]` | `x.permute(0, 2, 1, 3, 4).contiguous()` | `[2, 4, 8, 16, 16]` (`bfloat16`) | `[2, 8, 4, 16, 16]` | 5D block cache layout restoration | `compare_with_cpu`, `atol=0.0` |
| `GCNT-10` | `test_contiguous_6d_tensor_after_permute` | **Config A** | `[NET-NEW-SCENARIO]` | `x.permute(0, 2, 1, 3, 4, 5).contiguous()` | `[1, 2, 4, 8, 16, 8]` (`bfloat16`) | `[1, 4, 2, 8, 16, 8]` | 6D tensor permutation layout restoration | `compare_with_cpu`, `atol=0.0` |
| `GCNT-11` | `test_clone_preserves_grad_fn_none_inference` | **Config A** | `[NET-NEW-SCENARIO]` | `x.clone()` | `[32, 2816]` (`bfloat16`) | `[32, 2816]` | Verifying clean inference-only allocation with zero autograd overhead | `compare_with_cpu`, `atol=0.0` |
| `GCNT-12` | `test_contiguous_batched_odd_aspect` | **Config E** | `[GEMMA-PARAM-SWEEP]` | `x.transpose(1, 2).contiguous()` | `[3, 71, 8, 256]` (`bfloat16`) | `[3, 8, 71, 256]` | Non-aligned batch (3) and odd sequence (71) contiguous conversion | `compare_with_cpu`, `atol=0.0` |

---

## Section 10 — High-Rank 5D & 6D Advanced Vision/Paged KV Workloads

Validates complex multidimensional data manipulation patterns representative of paged attention caches, video patch hierarchies, and chunked context trees.

| ID | Test Name | Config Mapping | Classification | Test Steps / Pattern | Input Tensor Dims & Dtype | Expected Output Dims | Validation Focus | Pass Criteria |
|---|---|---|---|---|---|---|---|---|
| `GHRK-01` | `test_5d_paged_kv_view_and_reshape` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(num_blocks, num_heads, block_size, head_dim)` | `[16, 8, 4096]` (`bfloat16`) | `[16, 8, 16, 256]` | 3D flat pool unflattening into 4D blocked cache | `compare_with_cpu`, `atol=0.0` |
| `GHRK-02` | `test_5d_paged_kv_head_broadcast` | **Config A** | `[NET-NEW-SCENARIO]` | `x.unsqueeze(1).expand(16, 8, 16, 16, 16)` | `[16, 1, 16, 16, 16]` (`bfloat16`) | `[16, 8, 16, 16, 16]` | 5D block cache head broadcasting across 8 KV heads | `compare_with_cpu`, `atol=0.0` |
| `GHRK-03` | `test_5d_paged_kv_block_cat` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.cat([cache_a, cache_b], dim=0)` | `cache_a`: `[16, 8, 16, 256]`, `cache_b`: `[16, 8, 16, 256]` | `[32, 8, 16, 256]` | 5D KV-cache block pool concatenation along block dimension | `compare_with_cpu`, `atol=0.0` |
| `GHRK-04` | `test_5d_vision_spatial_patch_grid` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(B, num_patches_h, num_patches_w, num_heads, head_dim)` | `[1, 16, 16, 1152]` (`bfloat16`) | `[1, 16, 16, 16, 72]` | 4D vision spatial map unfolded into 5D multi-head representation (D=72) | `compare_with_cpu`, `atol=0.0` |
| `GHRK-05` | `test_5d_vision_grid_transpose` | **Config A** | `[GEMMA-EDGE-SHAPE]` | `x.transpose(1, 2)` | `[1, 16, 16, 16, 72]` (`bfloat16`) | `[1, 16, 16, 16, 72]` | Spatial grid transposition across height and width patch axes | `compare_with_cpu`, `atol=0.0` |
| `GHRK-06` | `test_6d_video_multimodal_unflatten` | **Config A** | `[NET-NEW-SCENARIO]` | `x.view(B, T, H_p, W_p, num_heads, head_dim)` | `[1, 4, 256, 1152]` (`bfloat16`) | `[1, 4, 16, 16, 16, 72]` | 4D video sequence unflattening into 6D spatio-temporal head structure | `compare_with_cpu`, `atol=0.0` |
| `GHRK-07` | `test_6d_video_temporal_transpose` | **Config A** | `[NET-NEW-SCENARIO]` | `x.transpose(1, 4)` | `[1, 4, 16, 16, 16, 72]` (`bfloat16`) | `[1, 16, 16, 16, 4, 72]` | Permuting temporal and attention head axes in 6D tensor | `compare_with_cpu`, `atol=0.0` |
| `GHRK-08` | `test_6d_video_flatten_to_3d` | **Config A** | `[NET-NEW-SCENARIO]` | `x.reshape(B, T * H_p * W_p, num_heads * head_dim)` | `[1, 4, 16, 16, 16, 72]` (`bfloat16`) | `[1, 1024, 1152]` | 6D tensor complete collapse back to 3D flattened token sequence | `compare_with_cpu`, `atol=0.0` |
| `GHRK-09` | `test_6d_chunk_along_frames` | **Config A** | `[NET-NEW-SCENARIO]` | `torch.chunk(x, 2, dim=1)` | `[1, 4, 16, 16, 16, 72]` (`bfloat16`) | 2 tensors: each `[1, 2, 16, 16, 16, 72]` | 6D tensor partitioning along temporal frame dimension | `compare_with_cpu`, `atol=0.0` |
| `GHRK-10` | `test_6d_contiguous_after_permute` | **Config A** | `[NET-NEW-SCENARIO]` | `x.permute(0, 4, 1, 2, 3, 5).contiguous()` | `[1, 4, 16, 16, 16, 72]` (`bfloat16`) | `[1, 16, 4, 16, 16, 72]` | Full 6D multi-axis permutation and contiguous layout restoration | `compare_with_cpu`, `atol=0.0` |

---

## 5. Coverage Summary Matrix

| Operator / Suite | Primary Architectural Context in Gemma-4 | Dimension Coverage | Net-New Scenarios | Param Sweeps | Edge Shapes | Total Tests |
|---|---|---|---|---|---|---|
| **Whole Modules** | Attention, MoE Router, MoE Experts, Vision Tower, RMSNorm, Decoder Layers | 1D, 2D, 3D, 4D | **14** | **1** | **0** | **15** |
| `reshape` / `view` | QKV Split, MoE Flatten/Unflatten, Patch Unfold, Decode Reshape | 1D, 2D, 3D, 4D, 5D | **15** | **2** | **3** | **20** |
| `squeeze` / `unsqueeze` | RoPE Freq Broadcast, Mask Alignment, MoE Capacity, Single-token Decode | 0D, 1D, 2D, 3D, 4D, 5D, 6D | **12** | **4** | **0** | **16** |
| `expand` | GQA KV-Head Broadcast (8->16, 2->16), 2D Mask Broadcast, Bias Broadcaster | 0D, 1D, 2D, 3D, 4D, 5D | **12** | **2** | **0** | **14** |
| `transpose` / `t` | Multi-Head Attention QKV (B, S, H, D <-> B, H, S, D), K^T MatMul, 2D GEMM | 2D, 3D, 4D, 5D, 6D | **10** | **3** | **3** | **16** |
| `split` / `chunk` | Fused QKV Decomposition, Gate-Up MLP Split, RoPE Split Half-Rotations | 1D, 2D, 3D, 4D, 5D | **8** | **4** | **2** | **14** |
| `cat` | KV-Cache Append (Decode/Prefill), RoPE Rotation Stitching, Multi-modal Sequence | 1D, 2D, 3D, 4D, 5D, 6D | **12** | **2** | **0** | **14** |
| `select` / Slicing | Decode Last Token Extractions, Top-1 Expert Selection, Head Selection | 1D, 2D, 3D, 4D, 5D, 6D | **10** | **2** | **0** | **12** |
| `clone` / `contiguous` | Post-Transpose Layout Enforcement for GEMM/SDPA/SDSC Kernels | 1D, 2D, 3D, 4D, 5D, 6D | **7** | **4** | **1** | **12** |
| **High-Rank (5D/6D)** | Paged Attention Cache Blocks, Video-Vision Hierarchical Patches | 5D, 6D | **9** | **0** | **1** | **10** |
| **Total** | | | **109** | **24** | **10** | **143** |
