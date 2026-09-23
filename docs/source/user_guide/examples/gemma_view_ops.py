# capture_stage1.py
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from collections import defaultdict

VIEW_OPS = {
    "aten.reshape", "aten.view", "aten._unsafe_view",
    "aten.transpose", "aten.permute", "aten.clone",
    "aten.squeeze", "aten.unsqueeze", "aten.flatten",
    "aten.cat", "aten.stack", "aten.repeat",
    "aten.unbind", "aten.unfold", "aten.flip",
    "aten.split", "aten.expand", "aten.narrow", "aten.select",
    "aten.contiguous",
}

found = defaultdict(list)  # op_name -> list of output shapes

def spyre_view_inspector(gm: torch.fx.GraphModule, example_inputs):
    """Runs instead of compiling — just records view ops."""
    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        op_name = str(node.target)
        # Normalise: "aten.view.default" → "aten.view"
        base = ".".join(op_name.split(".")[:2])
        if base in VIEW_OPS:
            shape = None
            val = node.meta.get("val")
            if val is not None:
                shape = tuple(val.shape)
            found[base].append(shape)
    return gm.forward  # return callable without compiling

# Load model (CPU, no Spyre needed)
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-4-26B-A4B-it",
    torch_dtype=torch.float16,
    device_map="cpu",
)
model.eval()
compiled = torch.compile(model, backend=spyre_view_inspector, fullgraph=False)

tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-26B-A4B-it")
inputs = tokenizer("Hello world", return_tensors="pt")

with torch.no_grad():
    compiled(**inputs)  # triggers tracing → calls spyre_view_inspector

print("\n=== VIEW OPS FOUND IN GEMMA 4 (Dynamo FX graph) ===")
for op, shapes in sorted(found.items()):
    print(f"\n{op}  ({len(shapes)} occurrences)")
    for s in shapes[:5]:  # first 5 unique shapes
        print(f"    {s}")
    if len(shapes) > 5:
        print(f"    ... and {len(shapes)-5} more")
