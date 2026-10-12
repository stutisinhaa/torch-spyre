# Copyright 2025 The Torch-Spyre Authors.
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

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from utils_inductor import compare_with_cpu, cached_randn

class TestGemmaViews(unittest.TestCase):
    def setUp(self):
        super().setUp()
        torch.manual_seed(0xAFFE)

    # 1 & 2. view / reshape tests
    def test_gemma_projection_view_and_reshape(self):
        # Scenario: Splits a flattened tensor [B, S, H * D] to [B, S, H, D]
        # and flattens [B, S, H, D] back to [B, S, H * D].
        # We test both standard Gemma shapes (aligned) and non-aligned sequence lengths/batches (padding check).
        shapes = [
            # (B, S, H, D)
            (1, 32, 16, 256),   # Standard sliding layer prefill
            (1, 1, 16, 256),    # Standard sliding layer decode
            (2, 67, 16, 256),   # Non-aligned sequence, multi-batch
            (1, 32, 16, 512),   # Full layer head dim (512)
            (3, 71, 8, 256),    # Custom small config, non-aligned seq and batch
        ]

        for B, S, H, D in shapes:
            # Test view split
            x = cached_randn((B, S, H * D), dtype=torch.float16)
            compare_with_cpu(lambda t: t.view(B, S, H, D), x, run_eager=True)

            # Test reshape flatten
            y = cached_randn((B, S, H, D), dtype=torch.float16)
            compare_with_cpu(lambda t: t.reshape(B, S, H * D), y, run_eager=True)

    # 3. transpose tests
    def test_gemma_attention_transpose(self):
        # Scenario: Transpose multi-head attention dims [B, S, H, D] -> [B, H, S, D]
        # and key transpose for attention matrix multiplication [B, H, S, D] -> [B, H, D, S].
        shapes = [
            # (B, S, H, D)
            (1, 32, 16, 256),
            (2, 67, 8, 256),
            (1, 1, 16, 256),
            (1, 512, 16, 256),
        ]

        for B, S, H, D in shapes:
            # 1. Head transpose
            x = cached_randn((B, S, H, D), dtype=torch.float16)
            compare_with_cpu(lambda t: t.transpose(1, 2), x, run_eager=True)

            # 2. Key transpose
            y = cached_randn((B, H, S, D), dtype=torch.float16)
            compare_with_cpu(lambda t: t.transpose(-2, -1), y, run_eager=True)

    # 4 & 5. clone / contiguous tests
    def test_gemma_contiguity_enforcement(self):
        # Scenario: After transposing [B, S, H, D] -> [B, H, S, D], force contiguous layout
        # using clone(memory_format=torch.contiguous_format) or contiguous() before next ops.
        shapes = [
            (1, 32, 16, 256),
            (2, 67, 8, 256),
        ]

        for B, S, H, D in shapes:
            x = cached_randn((B, S, H, D), dtype=torch.float16)
            
            # Test contiguous()
            def fn_cont(t):
                return t.transpose(1, 2).contiguous()
            compare_with_cpu(fn_cont, x, run_eager=True)

            # Test clone(memory_format=torch.contiguous_format)
            def fn_clone(t):
                return t.transpose(1, 2).clone(memory_format=torch.contiguous_format)
            compare_with_cpu(fn_clone, x, run_eager=True)

    # 6. unsqueeze tests
    def test_gemma_unsqueeze(self):
        # Scenario: Gemma RMSNorm or GQA uses unsqueeze to insert broadcast dimensions.
        # Adding dimension: [B, KVH, S, D] -> [B, KVH, 1, S, D]
        shapes = [
            (1, 8, 32, 256),
            (2, 2, 67, 512),
            (1, 1, 1, 256),
        ]

        for B, KVH, S, D in shapes:
            x = cached_randn((B, KVH, S, D), dtype=torch.float16)
            compare_with_cpu(lambda t: t.unsqueeze(2), x, run_eager=True)

    # 7. cat tests
    def test_gemma_concatenation(self):
        # Scenario 1: RoPE sine/cosine cat and half rotation cat.
        # Scenario 2: KV-cache appending along sequence dimension.
        scenarios = [
            # (B, H, S, D_half)
            (1, 16, 32, 128),
            (2, 8, 67, 128),
        ]

        for B, H, S, D_half in scenarios:
            x1 = cached_randn((B, H, S, D_half), dtype=torch.float16)
            x2 = cached_randn((B, H, S, D_half), dtype=torch.float16)
            compare_with_cpu(lambda a, b: torch.cat([a, b], dim=-1), x1, x2, run_eager=True)

        # KV-cache append scenarios: past_k of shape [B, KVH, past_len, D] + new_k of shape [B, KVH, seq_len, D]
        kv_scenarios = [
            # (B, KVH, past_len, seq_len, D)
            (1, 8, 512, 1, 256),    # Sliding decode append
            (2, 2, 128, 32, 512),   # Full attention prefill append
            (1, 8, 1000, 67, 256),  # Non-aligned past and new seq length
        ]

        for B, KVH, past_len, seq_len, D in kv_scenarios:
            past_k = cached_randn((B, KVH, past_len, D), dtype=torch.float16)
            new_k = cached_randn((B, KVH, seq_len, D), dtype=torch.float16)
            compare_with_cpu(lambda p, n: torch.cat([p, n], dim=2), past_k, new_k, run_eager=True)

    # 8. expand tests
    def test_gemma_gqa_expansion(self):
        # Scenario: Expand key/value tensors along a singleton dimension for GQA.
        # [B, KVH, 1, S, D] -> [B, KVH, exp_factor, S, D]
        scenarios = [
            # (B, KVH, exp_factor, S, D)
            (1, 8, 2, 32, 256),    # 8 KV heads -> 16 Q heads
            (1, 2, 8, 32, 512),    # 2 KV heads -> 16 Q heads
            (2, 4, 4, 67, 256),    # Non-aligned seq length
        ]

        for B, KVH, exp_factor, S, D in scenarios:
            x = cached_randn((B, KVH, 1, S, D), dtype=torch.float16)
            compare_with_cpu(lambda t: t.expand(B, KVH, exp_factor, S, D), x, run_eager=True)

    # 9. t tests
    def test_gemma_weight_transpose(self):
        # Scenario: Weights transpose in MoE expert GEMM or standard Linear.
        shapes = [
            (704, 2816),   # MoE intermediate to hidden (704 is stick-aligned for fp16: 704 / 64 = 11)
            (2112, 2816),  # Dense intermediate to hidden
            (256, 128),    # Custom smaller weight
        ]

        for out_features, in_features in shapes:
            w = cached_randn((out_features, in_features), dtype=torch.float16)
            compare_with_cpu(lambda t: t.t(), w, run_eager=True)

    # 10. select & narrow (slice) tests
    def test_gemma_slice_and_select(self):
        # Scenario 1: RoPE partial dimension select: .narrow(-1, 0, rot_dim)
        # Scenario 2: Sliding window cache cropping: .narrow(2, kv_len - win, win)
        # Scenario 3: MoE expert gating selection: select gating scores .select(dim, index)
        
        # 1. RoPE narrow
        x = cached_randn((1, 16, 32, 256), dtype=torch.float16)
        compare_with_cpu(lambda t: t.narrow(-1, 0, 64), x, run_eager=True)

        # 2. Sliding window narrow
        # Window size = 128, sequence length = 150 -> narrow(2, 22, 128)
        y = cached_randn((1, 8, 150, 256), dtype=torch.float16)
        compare_with_cpu(lambda t: t.narrow(2, 22, 128), y, run_eager=True)

        # 3. MoE select
        # Select expert gating weight: shape [B*S, K] -> [B*S] at expert index e
        z = cached_randn((32, 8), dtype=torch.float16)
        for e in [0, 3, 7]:
            compare_with_cpu(lambda t, idx=e: t.select(1, idx), z, run_eager=True)

    # 11. split tests
    def test_gemma_joint_qkv_split(self):
        # Scenario: Splits a joint QKV projection output into Q, K, and V tensors.
        # Shape of QKV output: [B, S, Q_dim + K_dim + V_dim]
        # In Gemma sliding layer: Q = 16*256 = 4096, K = 8*256 = 2048, V = 8*256 = 2048. Total = 8192
        # Splitting size list: [4096, 2048, 2048]
        scenarios = [
            # (B, S, [split_sizes])
            (1, 32, [4096, 2048, 2048]),  # Standard sliding QKV split
            (2, 67, [2048, 1024, 1024]),  # Custom non-aligned, joint QKV split
            (1, 15, [512, 256, 256]),     # Small joint split
        ]

        for B, S, splits in scenarios:
            total_dim = sum(splits)
            qkv = cached_randn((B, S, total_dim), dtype=torch.float16)
            compare_with_cpu(lambda t, sizes=splits: torch.split(t, sizes, dim=-1), qkv, run_eager=True)

    # Composite: Module-level Gemma Attention view combinations
    def test_gemma_attention_module_views(self):
        # Scenario: Complete attention head projection, split, transpose, GQA expansion,
        # self-attention matrix-multiply, and reassemble back.
        # This tests how multiple contiguous views propagate and interact in Compiled mode!
        B, S, H, KVH, D = 1, 32, 16, 8, 256
        exp = H // KVH  # 2

        # 1. Inputs
        q_proj = cached_randn((B, S, H * D), dtype=torch.float16)
        k_proj = cached_randn((B, S, KVH * D), dtype=torch.float16)
        v_proj = cached_randn((B, S, KVH * D), dtype=torch.float16)

        def attn_view_flow(q_in, k_in, v_in):
            # Split Q, K, V heads
            q = q_in.view(B, S, H, D).transpose(1, 2)     # [B, H, S, D]
            k = k_in.view(B, S, KVH, D).transpose(1, 2)   # [B, KVH, S, D]
            v = v_in.view(B, S, KVH, D).transpose(1, 2)   # [B, KVH, S, D]

            # GQA expand Key and Value
            k = k.unsqueeze(2).expand(-1, -1, exp, -1, -1).reshape(B, H, S, D)
            v = v.unsqueeze(2).expand(-1, -1, exp, -1, -1).reshape(B, H, S, D)

            # Transpose key for matrix multiply
            # Attention weights: Softmax(Q @ K.T / sqrt(D))
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(D)
            attn_weights = F.softmax(scores, dim=-1)

            # Attention output
            out = torch.matmul(attn_weights, v)   # [B, H, S, D]

            # Reassemble
            out = out.transpose(1, 2).clone(memory_format=torch.contiguous_format)
            return out.reshape(B, S, H * D)

        compare_with_cpu(attn_view_flow, q_proj, k_proj, v_proj, run_eager=True)
