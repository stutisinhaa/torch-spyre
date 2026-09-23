#!/usr/bin/env python3
"""
generate_gemma4_module_report_v2.py

Traces google/gemma-4-26B-A4B-it on CPU (eagerly, no torch.compile) and
writes a per-module operator report showing TWO layers per module:

  1. "PyTorch API calls"  -- torch.*/Tensor.* calls as literally invoked by
     the model's Python source, captured via TorchFunctionMode.
  2. "ATen operations"    -- the decomposed kernels those calls lower to,
     captured via TorchDispatchMode (same as the original script).

Both modes are active simultaneously and attributed to the same module-scope
stack, so you can see e.g. "torch.nn.functional.linear" (layer 1) immediately
followed by the "aten.t.default" + "aten.mm.default" (layer 2) it decomposes
into, inside the same module's row.

Usage / env vars: same as the original script.
    SEQ_LEN=32   OUTPUT=path   MODEL=...

Requirements:
    pip install transformers accelerate
"""

import dataclasses
import html
import os
import sys
from collections import OrderedDict, defaultdict

MODEL_ID = os.environ.get("MODEL",  "google/gemma-4-26B-A4B-it")
SEQ_LEN  = int(os.environ.get("SEQ_LEN", "32"))
OUTPUT   = os.environ.get("OUTPUT", "gemma4_26b_module_operator_report_v2.html")

# ── imports ───────────────────────────────────────────────────────────────────

try:
    import torch
    import torch.nn as nn
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.overrides import TorchFunctionMode
except ImportError:
    print("ERROR: PyTorch not found (or too old for TorchFunctionMode).", file=sys.stderr)
    sys.exit(1)

try:
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:
    print("ERROR: pip install transformers accelerate", file=sys.stderr); sys.exit(1)

# ── shape formatters ──────────────────────────────────────────────────────────
#
# IMPORTANT: never call Python's built-in repr()/str() on an arbitrary object
# here. Some environments monkey-patch Tensor.__repr__ (e.g. torch_spyre's
# spyre_aware_repr) with a signature that isn't compatible with being
# re-dispatched through handle_torch_function() while a TorchFunctionMode is
# active on the mode stack -- which is exactly what happens whenever a
# dataclass/ModelOutput's default __repr__ recurses into a Tensor field
# during formatting here. Walking known shapes explicitly (Tensor, list/
# tuple/dict, dataclasses) and falling back to just the type name for
# anything else avoids ever invoking an object's own __repr__/__str__ except
# for plain, known-safe scalars.

_SAFE_SCALAR_TYPES = (int, float, bool, str, bytes, type(None))


def _fmt(x, _depth=0):
    if isinstance(x, torch.Tensor):
        return f"{list(x.shape)} ({x.dtype})"
    if isinstance(x, _SAFE_SCALAR_TYPES):
        return repr(x)
    if _depth > 4:
        return f"<{type(x).__name__} (truncated)>"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(_fmt(v, _depth + 1) for v in x) + "]"
    if isinstance(x, dict):
        return "{" + ", ".join(f"{k!r}: {_fmt(v, _depth + 1)}" for k, v in x.items()) + "}"
    if dataclasses.is_dataclass(x) and not isinstance(x, type):
        parts = []
        for f in dataclasses.fields(x):
            try:
                v = getattr(x, f.name)
            except Exception:
                continue
            if v is None:
                continue
            parts.append(f"{f.name}={_fmt(v, _depth + 1)}")
        return f"{type(x).__name__}(" + ", ".join(parts) + ")"
    # Cache objects and other library types expose a rich-but-unsafe-to-repr
    # __repr__; describe them by type only rather than risk invoking it.
    return f"<{type(x).__name__}>"


def _fmt_args(args):
    return "[" + ", ".join(_fmt(a) for a in args) + "]"


def _fmt_out(out):
    return _fmt(out)


def _func_name(func):
    """Best-effort readable name for a torch-function-level callable."""
    mod = getattr(func, "__module__", None)
    qn = getattr(func, "__qualname__", None) or getattr(func, "__name__", None)
    if mod and qn:
        return f"{mod}.{qn}"
    return str(func)


# Torch-function calls that fire constantly as incidental bookkeeping
# (property-style shape/dtype checks, dunder repr/format, etc.) rather than
# real tensor-shape or math operations. Filtered out of the "PyTorch API
# calls" table so it stays readable; tune this set as needed.
_FUNC_NOISE = {
    "shape", "dtype", "device", "ndim", "requires_grad", "is_cuda", "layout",
    "is_leaf", "grad", "grad_fn", "data", "dim", "size", "numel",
    "element_size", "is_floating_point", "get_device", "type",
    "__repr__", "__format__", "__len__", "__bool__", "is_contiguous",
    "is_complex", "is_conj",
}


def _is_noise(name: str) -> bool:
    short = name.rsplit(".", 1)[-1]
    return short in _FUNC_NOISE

# ── recording state ───────────────────────────────────────────────────────────

_module_meta     = OrderedDict()      # path -> {"type", "inputs", "output"}
_module_ops      = defaultdict(list)  # path -> [{"op", "inputs", "output"}, ...]   (ATen layer)
_module_func_ops = defaultdict(list)  # path -> [{"op", "inputs", "output"}, ...]   (torch-function layer)
_scope_stack     = []                 # innermost active module path at the top


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
    """Layer 2: ATen dispatch level (decomposed kernels)."""
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


class _FuncRecorder(TorchFunctionMode):
    """Layer 1: Python/torch-API level (torch.view, torch.cat, F.linear, ...)."""
    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        name = _func_name(func)
        out = func(*args, **kwargs)
        if not _is_noise(name):
            path = _scope_stack[-1] if _scope_stack else "<root>"
            _module_func_ops[path].append({
                "op":     name,
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
.type2{font-family:var(--mono);font-size:.68rem;color:var(--purple);
      background:#F3E9F4;border-radius:3px;padding:.1rem .4rem}
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


def _render_table(ops):
    if not ops:
        return "<p style=\"font-size:.78rem;color:var(--ink3)\">(none)</p>"
    rows = []
    for rec in ops:
        rows.append(f"""
          <tr>
            <td class="m">{_e(rec['op'])}</td>
            <td class="d">{_e(rec['inputs'])}</td>
            <td class="d">{_e(rec['output'])}</td>
          </tr>""")
    return f"""
    <div class="tbl scroll">
      <table>
        <thead><tr><th>Operation</th><th>Input shape / arguments</th><th>Output shape</th></tr></thead>
        <tbody>{"".join(rows)}
        </tbody>
      </table>
    </div>"""


def render_html(module_meta, module_ops, module_func_ops, seq_len):
    total_modules   = len(module_meta)
    total_ops       = sum(len(v) for v in module_ops.values())
    unique_ops      = len({r["op"] for ops in module_ops.values() for r in ops})
    total_func_ops  = sum(len(v) for v in module_func_ops.values())
    unique_func_ops = len({r["op"] for ops in module_func_ops.values() for r in ops})

    out = []
    out.append(f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma-4-26B-A4B-it \u2014 CPU Module &amp; Operator Trace (2-layer)</title>
<style>{CSS}</style>
</head>
<body>
<div class="wrap">

<p class="kick">Torch-Spyre &middot; Gemma 4 model analysis &middot; CPU trace &middot; 2-layer</p>
<h1>Gemma-4-26B-A4B-it \u2014 Module &amp; Operator Report</h1>
<p class="sub">One forward pass recorded on CPU (seq_len={seq_len}). Each module shows both the
PyTorch-API calls it made (torch-function layer) and the ATen kernels those calls decomposed
into (dispatch layer).</p>

<div class="stats">
  <div class="stat"><span class="n">{total_modules}</span><span class="l">executed modules captured</span></div>
  <div class="stat"><span class="n">{total_func_ops}</span><span class="l">PyTorch API call entries</span></div>
  <div class="stat"><span class="n">{unique_func_ops}</span><span class="l">unique torch-function ops</span></div>
  <div class="stat"><span class="n">{total_ops}</span><span class="l">ATen operation entries</span></div>
  <div class="stat"><span class="n">{unique_ops}</span><span class="l">unique ATen operations</span></div>
</div>

<div class="note">
  <b>How to use this report:</b>
  click a module row to expand it. "PyTorch API calls" is what the model's Python
  source literally invoked (torch.view, F.linear, ...); "ATen operations" is what those
  calls decomposed into at runtime. Noise entries (plain .shape/.dtype/.size checks etc.)
  are filtered from the PyTorch-API table.
</div>

<h2>Run configuration</h2>
<div class="spec">
  <div><span class="k">Model</span><span class="v">google/gemma-4-26B-A4B-it</span></div>
  <div><span class="k">Device</span><span class="v">CPU (eager, no compile)</span></div>
  <div><span class="k">Dtype</span><span class="v">bfloat16</span></div>
  <div><span class="k">Input IDs</span><span class="v">input_ids  shape=[1, {seq_len}]  dtype=torch.int64</span></div>
</div>

<h2>Modules</h2>
<p>Expand a module to see its tensor shapes and operations, at both layers.</p>
""")

    for path, meta in module_meta.items():
        func_ops = module_func_ops.get(path, [])
        aten_ops = module_ops.get(path, [])
        n_func, n_aten = len(func_ops), len(aten_ops)
        cnt = f"{n_func} api / {n_aten} aten"

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
    <h4>PyTorch API calls</h4>
    {_render_table(func_ops)}
    <h4>ATen operations</h4>
    {_render_table(aten_ops)}
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
        print(f"ERROR: {e}\n  -> run: huggingface-cli login", file=sys.stderr)
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

    print("[3/4] Running forward pass with op recorder (2 layers)...", file=sys.stderr, flush=True)
    handles  = _register_hooks(model)
    dummy    = torch.randint(0, vocab_size, (1, SEQ_LEN), dtype=torch.long)

    # _FuncRecorder (torch-function / Python API layer) wraps _OpRecorder
    # (ATen dispatch layer). Both are active for the same call, so each
    # Python-level call and its ATen decomposition land in the same module's
    # scope.
    with torch.no_grad(), _FuncRecorder(), _OpRecorder():
        model(input_ids=dummy)

    for h in handles:
        h.remove()

    total_ops       = sum(len(v) for v in _module_ops.values())
    unique_ops      = len({r["op"] for ops in _module_ops.values() for r in ops})
    total_func_ops  = sum(len(v) for v in _module_func_ops.values())
    unique_func_ops = len({r["op"] for ops in _module_func_ops.values() for r in ops})
    print(f"     Modules: {len(_module_meta)}  |  ATen ops: {total_ops} ({unique_ops} unique)  |  "
          f"PyTorch API calls: {total_func_ops} ({unique_func_ops} unique)",
          file=sys.stderr, flush=True)

    print(f"[4/4] Writing HTML to {OUTPUT}...", file=sys.stderr, flush=True)
    page = render_html(_module_meta, _module_ops, _module_func_ops, SEQ_LEN)
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(page)
    print(f"     Done -> {OUTPUT}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()

