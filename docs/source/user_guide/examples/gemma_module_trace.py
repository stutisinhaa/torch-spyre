#!/usr/bin/env python3
"""
generate_gemma4_module_report.py

Traces google/gemma-4-26B-A4B-it on CPU (eagerly, no torch.compile) and
writes gemma4_26b_module_operator_report.html — a per-module operator report
in the same style as granite_3_3_8b_module_operator_report.html.

Strategy
--------
- TorchDispatchMode intercepts every real ATen kernel call at runtime.
- A scope stack (maintained by forward hooks) attributes each op to the
  innermost active nn.Module — works correctly even across MoE graph breaks
  that would confuse Dynamo's nn_module_stack approach.

Usage:
    python3 generate_gemma4_module_report.py

Env vars:
    SEQ_LEN=32       input sequence length (default: 32)
    OUTPUT=path      output HTML file (default: gemma4_26b_module_operator_report.html)
    MODEL=...        HuggingFace model ID override

Requirements:
    pip install transformers accelerate
    huggingface-cli login   # + accept license at hf.co/google/gemma-4-26B-A4B-it

Memory: ~50 GB RAM for full 26B weights (device_map="cpu").
"""

import html
import json
import os
import sys
from collections import OrderedDict, defaultdict

MODEL_ID = os.environ.get("MODEL",  "google/gemma-4-26B-A4B-it")
SEQ_LEN  = int(os.environ.get("SEQ_LEN", "32"))
OUTPUT   = os.environ.get("OUTPUT", "gemma4_26b_module_operator_report.html")

# ── imports ───────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils._python_dispatch import TorchDispatchMode
except ImportError:
    print("ERROR: PyTorch not found.", file=sys.stderr); sys.exit(1)

try:
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    print("ERROR: pip install transformers accelerate", file=sys.stderr); sys.exit(1)

# ── shape formatters ──────────────────────────────────────────────────────────

def _fmt(x):
    if isinstance(x, torch.Tensor):
        return f"{list(x.shape)} ({x.dtype})"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in x) + "]"
    return repr(x)


def _fmt_args(args):
    parts = []
    for a in args:
        if isinstance(a, torch.Tensor):
            parts.append(_fmt(a))
        elif isinstance(a, (list, tuple)) and any(isinstance(v, torch.Tensor) for v in a):
            parts.append("[" + ", ".join(_fmt(v) if isinstance(v, torch.Tensor) else repr(v) for v in a) + "]")
        else:
            parts.append(repr(a))
    return "[" + ", ".join(parts) + "]"


def _fmt_out(out):
    if out is None:
        return "None"
    if isinstance(out, torch.Tensor):
        return _fmt(out)
    if isinstance(out, (list, tuple)):
        return "[" + ", ".join(_fmt(v) if isinstance(v, torch.Tensor) else repr(v) for v in out) + "]"
    return repr(out)

# ── recording state ───────────────────────────────────────────────────────────

_module_meta = OrderedDict()     # path → {"type", "inputs", "output"}
_module_ops  = defaultdict(list) # path → [{"op", "inputs", "output"}, ...]
_scope_stack = []                # innermost active module path at the top


def _register_hooks(model: nn.Module):
    handles = []

    def _pre(name, mod, args, kwargs):
        _scope_stack.append(name)
        inp_strs = [_fmt(a) for a in args if isinstance(a, torch.Tensor)]
        if name not in _module_meta:
            _module_meta[name] = {"type": type(mod).__name__, "inputs": inp_strs, "output": ""}

    def _post(name, mod, args, output):
        if _scope_stack and _scope_stack[-1] == name:
            _scope_stack.pop()
        if name in _module_meta:
            _module_meta[name]["output"] = _fmt(output)

    for path, mod in model.named_modules():
        label = path or "<root>"
        h1 = mod.register_forward_pre_hook(
            lambda m, a, kw, _n=label: _pre(_n, m, a, kw), with_kwargs=True)
        h2 = mod.register_forward_hook(
            lambda m, a, out, _n=label: _post(_n, m, a, out))
        handles += [h1, h2]
        if label not in _module_meta:
            _module_meta[label] = {"type": type(mod).__name__, "inputs": [], "output": ""}

    return handles


class _OpRecorder(TorchDispatchMode):
    def __torch_dispatch__(self, op, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        out = op(*args, **kwargs)
        path = _scope_stack[-1] if _scope_stack else "<root>"
        _module_ops[path].append({
            "op":     str(op),
            "inputs": _fmt_args(args),
            "output": _fmt_out(out),
        })
        return out

# ── HTML renderer ─────────────────────────────────────────────────────────────

CSS = """\
:root{
  --bg:#F4F6F7;--panel:#FFF;--sunk:#FAFBFC;--ink:#0D1418;
  --ink2:#4A5760;--ink3:#8B959B;--rule:#DBE2E5;--rule2:#EDF1F3;
  --accent:#0F5C72;--accent2:#E3EFF2;--purple:#75297F;
  --sans:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;
  --mono:ui-monospace,SFMono-Regular,"SF Mono",Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--sans);
     font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}
.wrap{max-width:1120px;margin:0 auto;padding:2.2rem 1.4rem 5rem}
h1{font-size:1.55rem;margin:0 0 .25rem;letter-spacing:-.02em;font-weight:730;line-height:1.2}
h2{font-size:1rem;margin:2.1rem 0 .55rem;font-weight:690;letter-spacing:-.01em}
p{margin:0 0 .85rem;max-width:90ch;color:var(--ink2)}
.kick{font-family:var(--mono);font-size:.65rem;letter-spacing:.17em;
      text-transform:uppercase;color:var(--accent);margin:0 0 .6rem;font-weight:600}
.sub{font-family:var(--mono);font-size:.71rem;color:var(--ink3);margin-top:.15rem}
.stats{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
       gap:1px;background:var(--rule);border:1px solid var(--rule);
       border-radius:6px;overflow:hidden;margin:1rem 0}
.stat{background:var(--panel);padding:.8rem .95rem}
.stat .n{font-size:1.35rem;font-weight:700;display:block;line-height:1;letter-spacing:-.02em}
.stat .l{font-size:.69rem;color:var(--ink3);display:block;margin-top:.38rem;line-height:1.3}
.spec{background:var(--panel);border:1px solid var(--rule);border-radius:6px;
      overflow:hidden;margin:.9rem 0}
.spec div{display:flex;gap:1rem;padding:.48rem 1rem;border-bottom:1px solid var(--rule2);font-size:.84rem}
.spec div:last-child{border-bottom:0}
.spec .k{flex:0 0 12rem;color:var(--ink3)}
.spec .v{color:var(--ink);font-family:var(--mono);font-size:.79rem;word-break:break-word}
.note{background:var(--accent2);border-left:3px solid var(--accent);padding:.65rem .9rem;
      border-radius:0 5px 5px 0;margin:.9rem 0;font-size:.85rem;color:#123F4D}
details.mod{border:1px solid var(--rule);background:var(--panel);border-radius:5px}
details.mod+details.mod{margin-top:3px}
details.mod summary{list-style:none;cursor:pointer;padding:.55rem .72rem;
      display:flex;align-items:center;gap:.6rem;font-size:.84rem;flex-wrap:wrap}
details.mod summary::-webkit-details-marker{display:none}
details.mod summary:hover{background:var(--sunk)}
.arrow{color:var(--ink3);font-size:.57rem;transition:transform .12s}
details.mod[open] .arrow{transform:rotate(90deg)}
.type{font-family:var(--mono);font-size:.68rem;color:var(--accent);
      background:var(--accent2);border-radius:3px;padding:.1rem .4rem}
.cnt{font-family:var(--mono);font-size:.69rem;color:var(--ink3);
     margin-left:auto;white-space:nowrap}
.module-body{padding:.1rem .72rem .8rem;border-top:1px solid var(--rule2)}
.module-body h4{margin:.85rem 0 .45rem;font-size:.76rem;
     text-transform:uppercase;letter-spacing:.05em;color:var(--ink3)}
.scroll{overflow-x:auto}
.tbl{border:1px solid var(--rule);border-radius:5px;overflow:hidden;margin:.4rem 0}
table{border-collapse:collapse;width:100%;font-size:.76rem;background:var(--panel)}
th,td{text-align:left;padding:.45rem .65rem;border-bottom:1px solid var(--rule2);vertical-align:top}
thead th{font-size:.65rem;text-transform:uppercase;letter-spacing:.05em;
         color:var(--ink3);background:var(--sunk)}
tbody tr:last-child td{border-bottom:0}
td.m{font-family:var(--mono);font-size:.72rem;white-space:nowrap}
td.d{font-family:var(--mono);font-size:.69rem;color:var(--ink2);word-break:break-word}
@media(max-width:680px){
  .wrap{padding:1.5rem 1rem 3rem}h1{font-size:1.3rem}
  .spec div{flex-direction:column;gap:.08rem}.spec .k{flex:none}.cnt{margin-left:auto}
}"""


def _e(s):
    return html.escape(str(s))


def render_html(module_meta, module_ops, seq_len):
    total_modules = len(module_meta)
    total_ops     = sum(len(v) for v in module_ops.values())
    unique_ops    = len({r["op"] for ops in module_ops.values() for r in ops})

    out = []
    out.append(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma-4-26B-A4B-it \u2014 CPU Module &amp; Operator Trace</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<p class="kick">Torch-Spyre &middot; Gemma 4 model analysis &middot; CPU trace</p>
<h1>Gemma-4-26B-A4B-it \u2014 Module &amp; Operator Report</h1>
<p class="sub">One forward pass recorded on CPU (seq_len={seq_len}). Expand any module below to inspect its tensor shapes and operations.</p>

<div class="stats">
  <div class="stat"><span class="n">{total_modules}</span><span class="l">executed modules captured</span></div>
  <div class="stat"><span class="n">{total_ops}</span><span class="l">recorded operation entries</span></div>
  <div class="stat"><span class="n">{unique_ops}</span><span class="l">unique ATen operations</span></div>
  <div class="stat"><span class="n">bfloat16</span><span class="l">model execution dtype</span></div>
</div>

<div class="note">
  <b>How to use this report:</b>
  click a module row to expand it. The expanded section shows the module's
  input/output tensors and every ATen operation with its input and output shapes.
</div>

<h2>Run configuration</h2>
<div class="spec">
  <div><span class="k">Model</span><span class="v">google/gemma-4-26B-A4B-it</span></div>
  <div><span class="k">Device</span><span class="v">CPU (eager, no compile)</span></div>
  <div><span class="k">Dtype</span><span class="v">bfloat16</span></div>
  <div><span class="k">Input IDs</span><span class="v">input_ids  shape=[1, {seq_len}]  dtype=torch.int64</span></div>
</div>

<h2>Modules</h2>
<p>Expand a module to see its tensor shapes and operations.</p>
""")

    for path, meta in module_meta.items():
        ops = module_ops.get(path, [])
        n   = len(ops)
        cnt = f"{n} operation{'s' if n != 1 else ''}"

        out.append(f"""
<details class="mod">
  <summary>
    <span class="arrow">&#9654;</span>
    <b>{_e(path)}</b>
    <span class="type">{_e(meta['type'])}</span>
    <span class="cnt">{cnt}</span>
  </summary>
  <div class="module-body">
    <div class="spec">
      <div><span class="k">Module type</span><span class="v">{_e(meta['type'])}</span></div>
      <div><span class="k">Input tensor</span><span class="v">{_e(meta['inputs'])}</span></div>
      <div><span class="k">Output tensor</span><span class="v">{_e(meta['output'])}</span></div>
    </div>
    <h4>Operations</h4>
    <div class="tbl scroll">
      <table>
        <thead><tr><th>Operation</th><th>Input shape / arguments</th><th>Output shape</th></tr></thead>
        <tbody>""")

        for rec in ops:
            out.append(f"""
          <tr>
            <td class="m">{_e(rec['op'])}</td>
            <td class="d">{_e(rec['inputs'])}</td>
            <td class="d">{_e(rec['output'])}</td>
          </tr>""")

        out.append("""
        </tbody>
      </table>
    </div>
  </div>
</details>""")

    out.append("""
</div>
</body>
</html>
""")
    return "".join(out)

# ── main ──────────────────────────────────────────────────────────────────────

def main():
    print(f"[1/4] Loading config for {MODEL_ID}...", file=sys.stderr, flush=True)
    try:
        config = AutoConfig.from_pretrained(MODEL_ID)
    except Exception as e:
        print(f"ERROR: {e}\n  → run: huggingface-cli login", file=sys.stderr)
        sys.exit(1)

    tc = config.text_config

    def _tc(attr, fallback=None):
        try:
            return getattr(tc, attr)
        except Exception:
            pass
        try:
            return getattr(tc.per_layer_config[0], attr, fallback)
        except Exception:
            return fallback

    vocab_size = _tc("vocab_size")
    print(f"     hidden_size={_tc('hidden_size')}  layers={_tc('num_hidden_layers')}  "
          f"experts={_tc('num_experts')}  vocab={vocab_size}", file=sys.stderr, flush=True)

    print("[2/4] Loading model weights (~50 GB, may take a while)...", file=sys.stderr, flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        _attn_implementation="eager",
    )
    model.eval()
    print("     Model loaded.", file=sys.stderr, flush=True)

    print("[3/4] Running forward pass with op recorder...", file=sys.stderr, flush=True)
    handles  = _register_hooks(model)
    dummy    = torch.randint(0, vocab_size, (1, SEQ_LEN), dtype=torch.long)

    with torch.no_grad(), _OpRecorder():
        model(input_ids=dummy)

    for h in handles:
        h.remove()

    total_ops  = sum(len(v) for v in _module_ops.values())
    unique_ops = len({r["op"] for ops in _module_ops.values() for r in ops})
    print(f"     Modules: {len(_module_meta)}  |  Ops: {total_ops}  |  Unique ATen ops: {unique_ops}",
          file=sys.stderr, flush=True)

    print(f"[4/4] Writing HTML to {OUTPUT}...", file=sys.stderr, flush=True)
    page = render_html(_module_meta, _module_ops, SEQ_LEN)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"     Done → {OUTPUT}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

