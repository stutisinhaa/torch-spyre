"""
capture_stage1.py

Traces the REAL Gemma 4 26B-A4B-it model via torch.compile + a custom
Dynamo backend that walks the FX graph and records every view operation
with its output shape.

Requirements:
  pip install transformers accelerate
  huggingface-cli login        # must accept model license first at:
  # https://huggingface.co/google/gemma-4-26B-A4B-it

IMPORTANT: model ID is case-sensitive — capital B and A4B are required:
  CORRECT: google/gemma-4-26B-A4B-it
  WRONG:   google/gemma-4-26b-it      (404 — does not exist)

This script downloads only the model config + tokenizer on first run
(a few MB). The full weights (~50 GB) are needed for the actual forward
pass. To skip the forward pass and only verify the config, set:
  CONFIG_ONLY=1 python3 capture_stage1.py
"""

import os
import sys
import torch
from collections import defaultdict

MODEL = "google/gemma-4-26B-A4B-it"

VIEW_OPS = {
    "aten.reshape", "aten.view", "aten._unsafe_view",
    "aten.transpose", "aten.permute", "aten.clone",
    "aten.squeeze", "aten.unsqueeze", "aten.flatten",
    "aten.cat", "aten.stack", "aten.repeat",
    "aten.unbind", "aten.unfold", "aten.flip",
    "aten.split", "aten.expand", "aten.narrow", "aten.select",
    "aten.contiguous",
}

# ── Step 0: verify transformers is available ──────────────────────────────────

try:
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
except ImportError:
    print("ERROR: transformers not installed.")
    print("  pip install transformers accelerate")
    sys.exit(1)

# ── Step 1: load and print real config (no weights needed) ────────────────────

print("=" * 72)
print(f"  Loading config for {MODEL}")
print("=" * 72)

try:
    config = AutoConfig.from_pretrained(MODEL)
except Exception as e:
    print(f"\nERROR loading config: {e}")
    print("\nMost likely causes:")
    print("  1. Not authenticated — run:  huggingface-cli login")
    print("  2. License not accepted — visit:")
    print(f"     https://huggingface.co/google/gemma-4-26B-A4B-it")
    print("  3. Wrong model ID — must be exactly: google/gemma-4-26B-A4B-it")
    sys.exit(1)

tc = config.text_config
print(f"""
  Text decoder config:
    hidden_size                = {tc.hidden_size}
    num_attention_heads        = {tc.num_attention_heads}
    num_key_value_heads        = {tc.num_key_value_heads}
    num_global_key_value_heads = {getattr(tc, 'num_global_key_value_heads', 'N/A')}
    head_dim                   = {getattr(tc, 'head_dim', 'N/A')}
    global_head_dim            = {getattr(tc, 'global_head_dim', 'N/A')}
    num_hidden_layers          = {tc.num_hidden_layers}
    num_experts                = {getattr(tc, 'num_experts', 'N/A')}
    top_k_experts              = {getattr(tc, 'top_k_experts', 'N/A')}
    moe_intermediate_size      = {getattr(tc, 'moe_intermediate_size', 'N/A')}
    intermediate_size          = {tc.intermediate_size}
    sliding_window             = {getattr(tc, 'sliding_window', 'N/A')}
    layer_types (first 6)      = {getattr(tc, 'layer_types', ['N/A'])[:6]}
""")

if os.environ.get("CONFIG_ONLY"):
    print("CONFIG_ONLY=1 set — stopping before model load.")
    sys.exit(0)

# ── Step 2: load full model (needs ~50 GB RAM or disk for offload) ────────────

print("  Loading model weights (this will take a while for 26B)...")
print("  Tip: set CONFIG_ONLY=1 to skip this and only see the config.\n")

model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    torch_dtype=torch.float16,
    device_map="cpu",          # CPU only — no GPU/Spyre needed for tracing
)
model.eval()
print("  Model loaded.\n")

# ── Step 3: Dynamo backend that records view ops ───────────────────────────────

found = defaultdict(list)   # op_name → list of output shapes
graph_breaks = []           # track where Dynamo gave up

def spyre_view_inspector(gm: torch.fx.GraphModule, example_inputs):
    """
    Custom Dynamo backend — called once per contiguous subgraph.
    Walks the FX graph, records every view op + output shape.
    Returns gm.forward without compiling anything.
    """
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        op_name = str(node.target)
        # "aten.view.default" → "aten.view"
        base = ".".join(op_name.split(".")[:2])
        if base not in VIEW_OPS:
            continue
        shape = None
        val = node.meta.get("val")
        if val is not None:
            if isinstance(val, torch.Tensor):
                shape = tuple(val.shape)
            elif isinstance(val, (list, tuple)):
                shape = [tuple(v.shape) for v in val if isinstance(v, torch.Tensor)]
        found[base].append(shape)
    return gm.forward   # do not compile — just run eagerly

# Count graph breaks by wrapping the backend
_subgraph_count = [0]
def counting_backend(gm, example_inputs):
    _subgraph_count[0] += 1
    return spyre_view_inspector(gm, example_inputs)

# ── Step 4: tokenize and run ──────────────────────────────────────────────────

tokenizer = AutoTokenizer.from_pretrained(MODEL)
inputs = tokenizer("Hello world", return_tensors="pt")

compiled = torch.compile(model, backend=counting_backend, fullgraph=False)

print("  Running forward pass (triggers Dynamo tracing)...")
with torch.no_grad():
    compiled(**inputs)

# ── Step 5: print results ─────────────────────────────────────────────────────

print(f"\n  Dynamo captured {_subgraph_count[0]} subgraph(s).")
if _subgraph_count[0] > 1:
    print(f"  NOTE: {_subgraph_count[0]-1} graph break(s) detected.")
    print("        Each break is a place Dynamo couldn't compile (likely MoE routing).\n")

print("\n" + "=" * 72)
print("  VIEW OPS FOUND IN GEMMA 4 (Dynamo FX graph — ground truth)")
print("=" * 72)

for op in sorted(found):
    shapes = found[op]
    # deduplicate shapes for display
    seen, unique = set(), []
    for s in shapes:
        k = str(s)
        if k not in seen:
            seen.add(k); unique.append(s)
    print(f"\n  {op}  ({len(shapes)} total calls, {len(unique)} unique shapes)")
    for s in unique[:5]:
        print(f"      {s}")
    if len(unique) > 5:
        print(f"      ... and {len(unique)-5} more unique shapes")

# ops from the target list that were NOT found
print("\n" + "=" * 72)
print("  OPS NOT FOUND (not present in this model/input):")
print("=" * 72)
for op in sorted(VIEW_OPS):
    if op not in found:
        print(f"  {op}")

print()
