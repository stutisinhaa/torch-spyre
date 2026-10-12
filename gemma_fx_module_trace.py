#!/usr/bin/env python3
"""
gemma_fx_module_trace.py

Records nn.Module → ATen op attribution for Gemma-4 running through
AutoSpyreModelForCausalLM with torch.compile, by reading
node.meta["nn_module_stack"] from the Dynamo FX graph.

No eager pre-pass, no forward hooks. Dynamo populates nn_module_stack on
every call_function node before it hands the graph to the backend. A thin
wrapper intercepts each subgraph, reads that metadata + node.meta["val"]
shapes, records the attribution, then delegates to the real Spyre backend.

Usage:
    python gemma_fx_module_trace.py

Env vars:
    MODEL    HuggingFace model ID  (default: google/gemma-4-26B-A4B-it)
    SEQ_LEN  Input sequence length (default: 64)
    OUTPUT   Output HTML path      (default: gemma4_fx_module_trace.html)
"""

import html as html_lib
import os
import sys
from collections import OrderedDict, defaultdict

MODEL_ID = os.environ.get("MODEL", "google/gemma-4-26B-A4B-it")
SEQ_LEN  = int(os.environ.get("SEQ_LEN", "64"))
OUTPUT   = os.environ.get("OUTPUT", "gemma4_fx_module_trace_hfadapter_aten.html")

try:
    import torch
    import torch.nn as nn
except ImportError:
    sys.exit("ERROR: PyTorch not installed")

try:
    from hf_adapters import AutoSpyreModelForCausalLM
    from transformers import AutoTokenizer
except ImportError:
    sys.exit("ERROR: pip install hf-adapters transformers")

# ── recording state ───────────────────────────────────────────────────────────

_records: dict = defaultdict(list)   # module_path → [{op, inputs, output}, ...]
_module_classes: dict = {}            # module_path → class name string


def _shape(val) -> str:
    """Format node.meta['val'] (fake tensor or tuple thereof)."""
    if val is None:
        return "?"
    if isinstance(val, torch.Tensor):
        return "%s (%s)" % (list(val.shape), val.dtype)
    if isinstance(val, (list, tuple)):
        parts = [_shape(v) for v in val if isinstance(v, torch.Tensor)]
        return "[%s]" % ", ".join(parts) if parts else "[]"
    return type(val).__name__


def _cls_name(cls) -> str:
    return cls if isinstance(cls, str) else cls.__name__


def _arg_shape(a) -> str:
    if isinstance(a, torch.fx.Node):
        return _shape(a.meta.get("val"))
    if isinstance(a, (list, tuple)):
        parts = [_arg_shape(v) for v in a if isinstance(v, torch.fx.Node)]
        return "[%s]" % ", ".join(parts) if parts else ""
    if isinstance(a, (int, float, bool)):
        return repr(a)
    if isinstance(a, str):
        return repr(a)
    return ""


# ── inspecting backend wrapper ────────────────────────────────────────────────

def _make_inspecting_backend(real_backend):
    """
    Wraps a Dynamo backend callable. Reads nn_module_stack + val from every
    call_function node, records attribution, then forwards ALL arguments
    (including config_patches and any other kwargs) to the real backend.
    """
    def inspecting_backend(gm: torch.fx.GraphModule, example_inputs, **kwargs):
        for node in gm.graph.nodes:
            if node.op != "call_function":
                continue

            # nn_module_stack: OrderedDict key → (module_path_str, cls)
            # Items are ordered outermost → innermost.
            stack = node.meta.get("nn_module_stack") or {}

            out_val   = node.meta.get("val")
            out_shape = _shape(out_val)
            in_shapes = "[%s]" % ", ".join(
                filter(None, (_arg_shape(a) for a in node.args))
            )
            op_str = str(node.target)

            if stack:
                items = list(stack.values())   # [(path, cls), ...] outermost first

                # Build breadcrumb once — shared by all levels of this node's stack
                breadcrumb = " &rsaquo; ".join(
                    "<b>%s</b> <span style='color:var(--ink3)'>(%s)</span>"
                    % (html_lib.escape(_cls_name(c)), html_lib.escape(p))
                    for p, c in items
                )

                # Record the op under EVERY level of the stack so that parent
                # modules (Gemma4Attention, Gemma4MoEBlock, etc.) accumulate all
                # ops dispatched by their descendants, exactly like the v2 CPU report.
                for mod_path, cls in items:
                    cls_str = _cls_name(cls)
                    _module_classes.setdefault(mod_path, cls_str)
                    _records[mod_path].append({
                        "op":        op_str,
                        "inputs":    in_shapes,
                        "output":    out_shape,
                        "breadcrumb": breadcrumb,
                    })
            else:
                _records["<outside_module>"].append({
                    "op":        op_str,
                    "inputs":    in_shapes,
                    "output":    out_shape,
                    "breadcrumb": "",
                })

        # Forward ALL kwargs (config_patches, etc.) to the real backend
        return real_backend(gm, example_inputs, **kwargs)

    return inspecting_backend


# ── HTML renderer — styled to match gemma4_26b_module_operator_report_v2 ─────

_CSS = """\
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
.bc{font-size:.76rem;color:var(--ink3);margin:.25rem 0 .75rem;line-height:1.5}
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


def _e(s) -> str:
    return html_lib.escape(str(s))


def _render_table(rows):
    if not rows:
        return "<p style='font-size:.78rem;color:var(--ink3)'>(none)</p>"
    trs = "".join(
        "<tr><td class='m'>%s</td><td class='d'>%s</td><td class='d'>%s</td></tr>"
        % (_e(r["op"]), _e(r["inputs"]), _e(r["output"]))
        for r in rows
    )
    return (
        "<div class='tbl scroll'><table>"
        "<thead><tr><th>Operation</th><th>Input shape / arguments</th>"
        "<th>Output shape</th></tr></thead>"
        "<tbody>%s</tbody></table></div>" % trs
    )


def _render_html() -> str:
    n_modules  = len(_records)
    total_ops  = sum(len(v) for v in _records.values())
    unique_ops = len({r["op"] for rows in _records.values() for r in rows})

    parts = []
    parts.append("""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Gemma-4-26B-A4B-it &mdash; Spyre Module &amp; ATen Op Trace</title>
<style>%s</style>
</head>
<body>
<div class="wrap">

<p class="kick">Torch-Spyre &middot; Gemma 4 26B-A4B-it &middot; HF-Adapter &middot; Spyre hardware &middot; nn_module_stack trace</p>
<h1>Gemma-4-26B-A4B-it &mdash; Module &amp; ATen Op Report (Spyre, torch.compile)</h1>
<p class="sub">ATen ops attributed to nn.Module subclasses via node.meta["nn_module_stack"] in the Dynamo FX graph.
Shapes from node.meta["val"]. Real Spyre compilation runs underneath.</p>

<div class="stats">
  <div class="stat"><span class="n">%d</span><span class="l">module paths recorded</span></div>
  <div class="stat"><span class="n">%d</span><span class="l">ATen operation entries</span></div>
  <div class="stat"><span class="n">%d</span><span class="l">unique ATen operations</span></div>
  <div class="stat"><span class="n">%d</span><span class="l">input sequence length</span></div>
</div>

<div class="note">
  <b>How to use this report:</b> click a module row to expand it.
  "ATen operations" are the kernels dispatched directly by that module (not its children),
  attributed via <code>node.meta["nn_module_stack"]</code>.
  Input/output shapes come from <code>node.meta["val"]</code> (Dynamo fake tensors &mdash; same shapes the Spyre compiler sees).
  <br><br>
  <b>Not included:</b> ops injected by the Spyre Inductor backend after graph handoff
  (<code>spyre.copy_forced</code>, restickify, etc.) and ops outside compiled regions
  (adapter Python bookkeeping). For those, use <code>TORCH_LOGS=aot_graphs</code>.
</div>

<h2>Run configuration</h2>
<div class="spec">
  <div><span class="k">Model</span><span class="v">%s</span></div>
  <div><span class="k">Adapter</span><span class="v">AutoSpyreModelForCausalLM</span></div>
  <div><span class="k">Device</span><span class="v">spyre:0 (hardware)</span></div>
  <div><span class="k">Dtype</span><span class="v">bfloat16</span></div>
  <div><span class="k">Input IDs</span><span class="v">input_ids  shape=[1, %d]  dtype=torch.int64</span></div>
  <div><span class="k">Trace method</span><span class="v">node.meta["nn_module_stack"] from Dynamo FX graph via custom backend wrapper</span></div>
</div>

<h2>Modules</h2>
<p>Expand a module to see its nesting breadcrumb and every ATen operation it dispatched directly.</p>
""" % (_CSS, n_modules, total_ops, unique_ops, SEQ_LEN, _e(MODEL_ID), SEQ_LEN))

    for mod_path, rows in _records.items():
        cls_name   = _module_classes.get(mod_path, "")
        breadcrumb = rows[0]["breadcrumb"] if rows else ""
        n          = len(rows)
        cnt_str    = "%d operation%s" % (n, "s" if n != 1 else "")

        parts.append("""
<details class="mod">
  <summary>
    <span class="arrow">&#9654;</span>
    <b>%s</b>
    <span class="type">%s</span>
    <span class="cnt">%s</span>
  </summary>
  <div class="module-body">
    <div class="spec">
      <div><span class="k">Module type</span><span class="v">%s</span></div>
      <div><span class="k">Module path</span><span class="v">%s</span></div>
    </div>
    <p class="bc">%s</p>
    <h4>ATen operations</h4>
    %s
  </div>
</details>""" % (
            _e(mod_path or "&lt;root&gt;"),
            _e(cls_name),
            cnt_str,
            _e(cls_name),
            _e(mod_path or "<root>"),
            breadcrumb,          # already HTML-escaped + tagged in the inspector
            _render_table(rows),
        ))

    parts.append("""
</div>
<footer style="margin-top:3rem;padding-top:1rem;border-top:1px solid #e5e7eb;
               text-align:center;font-size:.72rem;color:#57606a">Made with IBM Bob</footer>
</body>
</html>""")
    return "".join(parts)


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    print("[1/4] Loading %s ..." % MODEL_ID, flush=True)
    model = AutoSpyreModelForCausalLM.from_pretrained(MODEL_ID)
    tok   = AutoTokenizer.from_pretrained(MODEL_ID)

    # Patch compile_fx_inner — the post-AOT-Autograd callback that receives
    # the fully decomposed graph where every call_function node has val set.
    # nn_module_stack is preserved through AOT so attribution still works.
    # compile_fx (the outer function) receives the pre-decomposition Dynamo
    # graph where val is not reliably set on all nodes.
    import torch._inductor.compile_fx as _cfx
    _real_compile_fx = _cfx.compile_fx_inner

    _cfx.compile_fx_inner = _make_inspecting_backend(_real_compile_fx)

    print("[2/4] Tokenising input (seq_len=%d) ..." % SEQ_LEN, flush=True)
    inputs = tok(["Hello world"], return_tensors="pt", padding=True)

    ids = inputs["input_ids"]
    if ids.shape[1] < SEQ_LEN:
        pad = torch.zeros(1, SEQ_LEN - ids.shape[1], dtype=ids.dtype)
        ids = torch.cat([pad, ids], dim=1)
    else:
        ids = ids[:, :SEQ_LEN]
    inputs["input_ids"] = ids

    if "attention_mask" in inputs:
        mask = inputs["attention_mask"]
        if mask.shape[1] < SEQ_LEN:
            pad = torch.zeros(1, SEQ_LEN - mask.shape[1], dtype=mask.dtype)
            mask = torch.cat([pad, mask], dim=1)
        else:
            mask = mask[:, :SEQ_LEN]
        inputs["attention_mask"] = mask

    print("[3/4] Running model.generate() — first call triggers torch.compile + inspection ...",
          flush=True)
    try:
        model.generate(**inputs, max_new_tokens=1)
    finally:
        _cfx.compile_fx_inner = _real_compile_fx

    total  = sum(len(v) for v in _records.values())
    unique = len({r["op"] for rows in _records.values() for r in rows})
    print("     Recorded %d ATen op calls (%d unique) across %d module paths"
          % (total, unique, len(_records)), flush=True)

    print("[4/4] Writing HTML to %s ..." % OUTPUT, flush=True)
    page = _render_html()
    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(page)
    print("     Done -> %s" % OUTPUT, flush=True)

    # Stdout summary
    print("\n%s" % ("─" * 72))
    print("  MODULE SUMMARY")
    print("─" * 72)
    for mod_path, rows in _records.items():
        cls     = _module_classes.get(mod_path, "")
        unique_ops = sorted({r["op"] for r in rows})
        label   = (mod_path or "<root>")[:55]
        print("  %-55s [%s]  %d calls" % (label, cls, len(rows)))
        print("    %s%s" % (
            ", ".join(unique_ops[:5]),
            " ..." if len(unique_ops) > 5 else "",
        ))


if __name__ == "__main__":
    main()
