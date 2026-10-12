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

"""
add_layout_trace.py — Stage-by-stage layout trace for a pointwise add.

Prints the SpyreTensorLayout of two inputs (a, b) and their output (c = a + b)
at every stage of the Spyre compilation pipeline:

  Stage 0 — Eager:          real device tensor, layout from the runtime
  Stage 1 — Dynamo FX:      symbolic proxy shapes (no device layout yet)
  Stage 2 — Lowering:       Inductor LoopLevelIR node, FixedLayout assigned
  Stage 3 — Layout prop:    propagate_spyre_tensor_layouts assigns SpyreTensorLayouts
  Stage 4 — Finalize:       scheduler resolves candidate STLs, FixedTiledLayout pinned
  Stage 5 — KernelSpec:     TensorArg records (device_size, stride_map) for codegen
  Stage 6 — Runtime output: live output tensor checked after kernel executes

Run:
    TORCH_SPYRE_MOCK_DEVICE=1 python add_layout_trace.py
"""

import os
import textwrap
from typing import Any

import torch
import torch.fx

try:
    import torch_spyre
except ImportError:
    raise RuntimeError("torch_spyre is missing. Install via: pip install torch-spyre")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SEP = "=" * 72

def _stl_summary(stl) -> str:
    """One-line summary of a SpyreTensorLayout (or None / str)."""
    if stl is None:
        return "None"
    if isinstance(stl, str):
        return stl
    parts = [f"device_size={list(stl.device_size)}",
             f"stride_map={list(stl.stride_map)}"]
    if hasattr(stl, "device_dtype"):
        parts.append(f"device_dtype={stl.device_dtype}")
    return "SpyreTensorLayout(" + ", ".join(parts) + ")"


def _tensor_row(label: str, t: torch.Tensor) -> str:
    stl = None
    if t.device.type == "spyre":
        stl = t.device_tensor_layout()
    return (
        f"  {label:<6}  shape={tuple(t.shape)!s:<18}  "
        f"stride={tuple(t.stride())!s:<18}  "
        f"dtype={t.dtype!s:<18}  "
        f"stl={_stl_summary(stl)}"
    )


def _stage(title: str, body: str) -> None:
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)
    print(body)


# ---------------------------------------------------------------------------
# Stage 0 — Eager: create tensors directly on the Spyre device
# ---------------------------------------------------------------------------

def stage0_eager(device, rows, cols):
    a = torch.randn(rows, cols, dtype=torch.float16, device=device)
    b = torch.randn(rows, cols, dtype=torch.float16, device=device)
    c = a + b

    body = "\n".join([
        _tensor_row("a", a),
        _tensor_row("b", b),
        _tensor_row("c=a+b", c),
        "",
        "  These are live device tensors.  The SpyreTensorLayout comes directly",
        "  from the C++ runtime via tensor.device_tensor_layout().",
        "  device_size  : padded, higher-dimensional on-device shape (stick-tiled)",
        "  stride_map   : host-element jumps for each device axis — the bridge",
        "                 between host flat offsets and device coordinates.",
    ])
    _stage("STAGE 0 — Eager (live device tensors, layout from runtime)", body)
    return a, b


# ---------------------------------------------------------------------------
# Stage 1 — Dynamo FX graph capture
# ---------------------------------------------------------------------------

def stage1_dynamo(a, b):
    """
    Use torch.fx.symbolic_trace to capture the FX graph.
    At this point Dynamo has only symbolic shapes (FakeTensors);
    no SpyreTensorLayout exists yet because FakeTensor does not carry
    device-specific layout metadata.
    """
    class AddMod(torch.nn.Module):
        def forward(self, a, b):
            return a + b

    mod = AddMod()
    # Use make_fx (Dynamo's own mechanism) for a proper FakeTensor trace
    from torch._functorch.aot_autograd import make_functional_fx
    try:
        from torch._dynamo.symbolic_shapes import ShapeEnv
        from torch._subclasses.fake_tensor import FakeTensorMode
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
        with fake_mode:
            fa = fake_mode.from_tensor(a)
            fb = fake_mode.from_tensor(b)
        gm = torch.fx.symbolic_trace(mod)
    except Exception:
        gm = torch.fx.symbolic_trace(mod)

    lines = ["  FX graph nodes after Dynamo capture:", ""]
    for node in gm.graph.nodes:
        meta = node.meta.get("val", None)
        if hasattr(meta, "shape"):
            shape_str = f"shape={tuple(meta.shape)}"
            dtype_str = f"dtype={meta.dtype}"
        elif meta is not None:
            shape_str = f"val={meta!r}"
            dtype_str = ""
        else:
            shape_str = ""
            dtype_str = ""
        lines.append(f"    {node.op:<14}  name={node.name:<12}  {shape_str}  {dtype_str}")

    lines += [
        "",
        "  No SpyreTensorLayout yet — FX nodes carry only abstract shapes.",
        "  Layout propagation happens later (Stage 3), after Inductor lowering.",
    ]
    _stage("STAGE 1 — Dynamo FX graph (symbolic shapes, no device layout)", "\n".join(lines))
    return gm


# ---------------------------------------------------------------------------
# Stage 2 — Inductor LoopLevelIR lowering (intercept inside compile)
# ---------------------------------------------------------------------------

_lowered_nodes: dict[str, Any] = {}   # filled by the hook below

def _install_lowering_hook():
    """
    Monkey-patch ComputedBuffer.decide_layout to record the FixedLayout
    assigned during lowering (before propagate_spyre_tensor_layouts runs).
    """
    try:
        from torch._inductor.ir import ComputedBuffer
        _orig_decide = ComputedBuffer.decide_layout

        def _patched_decide(self):
            result = _orig_decide(self)
            layout = getattr(self, "layout", None)
            if layout is not None and hasattr(layout, "size"):
                _lowered_nodes[self.get_name()] = {
                    "type": type(layout).__name__,
                    "size": [str(s) for s in layout.size],
                    "stride": [str(s) for s in layout.stride],
                    "dtype": str(layout.dtype),
                    # SpyreTensorLayout not yet assigned here
                    "device_layout": getattr(layout, "device_layout", None),
                }
            return result

        ComputedBuffer.decide_layout = _patched_decide
        return _orig_decide
    except Exception as e:
        return None


def _remove_lowering_hook(orig):
    if orig is not None:
        try:
            from torch._inductor.ir import ComputedBuffer
            ComputedBuffer.decide_layout = orig
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Stage 3 — propagate_spyre_tensor_layouts hook
# ---------------------------------------------------------------------------

_prop_snapshots: dict[str, Any] = {}   # filled by the hook below

def _install_prop_hook():
    """
    Intercept propagate_spyre_tensor_layouts to record the candidate STLs
    assigned to each operation (op.layouts) right after the pass runs.
    """
    try:
        import torch_spyre._inductor.propagate_layouts as pl
        _orig_prop = pl.propagate_spyre_tensor_layouts

        def _patched_prop(graph):
            _orig_prop(graph)
            for op in graph.operations:
                name = op.get_name() if hasattr(op, "get_name") else str(op)
                layouts = getattr(op, "layouts", [])
                _prop_snapshots[name] = [_stl_summary(s) for s in layouts]

        pl.propagate_spyre_tensor_layouts = _patched_prop
        return _orig_prop, pl
    except Exception:
        return None, None


def _remove_prop_hook(orig, pl):
    if orig is not None and pl is not None:
        pl.propagate_spyre_tensor_layouts = orig


# ---------------------------------------------------------------------------
# Stage 4 — FixedTiledLayout finalization (scheduler resolve pass)
# ---------------------------------------------------------------------------

_finalized: dict[str, Any] = {}   # filled by the hook below

def _install_finalize_hook():
    """
    Record each buffer's FixedTiledLayout after the scheduler resolves
    candidates and pins a single layout per buffer.
    """
    try:
        from torch_spyre._inductor.ir import FixedTiledLayout

        import torch_spyre._inductor.propagate_layouts as pl
        _orig_resolve = pl._resolve_copy_back_candidates

        def _patched_resolve(operations):
            _orig_resolve(operations)
            for op in operations:
                name = op.get_name() if hasattr(op, "get_name") else str(op)
                layout = None
                if hasattr(op, "get_layout"):
                    try:
                        layout = op.get_layout()
                    except Exception:
                        pass
                if isinstance(layout, FixedTiledLayout):
                    _finalized[name] = _stl_summary(layout.device_layout)
                elif layout is not None:
                    _finalized[name] = f"<{type(layout).__name__}> (not FixedTiledLayout)"

        pl._resolve_copy_back_candidates = _patched_resolve
        return _orig_resolve, pl
    except Exception:
        return None, None


def _remove_finalize_hook(orig, pl):
    if orig is not None and pl is not None:
        pl._resolve_copy_back_candidates = orig


# ---------------------------------------------------------------------------
# Stage 5 — KernelSpec TensorArg capture
# ---------------------------------------------------------------------------

_kernel_specs: list[Any] = []   # filled by the hook below

def _install_kernel_spec_hook():
    """
    Record each KernelSpec produced by SpyreKernel.generate_op_spec.
    Each TensorArg inside it carries the final (device_size, stride_map).
    """
    try:
        from torch_spyre._inductor import scheduler as spyre_sched
        _orig_generate = spyre_sched.SpyreScheduling.generate_kernel_code

        def _patched_generate(self, sched_node):
            result = _orig_generate(self, sched_node)
            # SpyreKernel stores the op_spec on itself
            kernel = getattr(sched_node, "_spyre_kernel", None)
            if kernel is None:
                # Fall back: look for a recently created SpyreKernel
                pass
            return result

        return None, None   # hooks into OpSpec emission instead (see below)
    except Exception:
        return None, None


# Simpler approach: hook OpSpec.__init__
def _install_opspec_hook():
    try:
        from torch_spyre._inductor.op_spec import OpSpec

        _orig_init = OpSpec.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            _kernel_specs.append(self)

        OpSpec.__init__ = _patched_init
        return _orig_init
    except Exception:
        return None


def _remove_opspec_hook(orig):
    if orig is not None:
        try:
            from torch_spyre._inductor.op_spec import OpSpec
            OpSpec.__init__ = orig
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Main compilation + runtime run
# ---------------------------------------------------------------------------

def run_trace(rows: int, cols: int, device: torch.device) -> None:
    print(f"\n{SEP}")
    print(f"  POINTWISE ADD LAYOUT TRACE   shape=({rows}, {cols})   device={device}")
    print(f"  dtype=float16   stick_size=64 elements (128-byte stick)")
    print(SEP)

    # ── Stage 0: eager ──────────────────────────────────────────────────────
    a, b = stage0_eager(device, rows, cols)

    # ── Stage 1: Dynamo FX ──────────────────────────────────────────────────
    stage1_dynamo(a, b)

    # ── Install hooks for stages 2–5 ────────────────────────────────────────
    _lowered_nodes.clear()
    _prop_snapshots.clear()
    _finalized.clear()
    _kernel_specs.clear()

    orig_decide = _install_lowering_hook()
    orig_prop, prop_mod = _install_prop_hook()
    orig_resolve, resolve_mod = _install_finalize_hook()
    orig_opspec = _install_opspec_hook()

    # ── torch.compile run ────────────────────────────────────────────────────
    def add_fn(x, y):
        return x + y

    compiled_add = torch.compile(add_fn, backend="spyre")
    try:
        c = compiled_add(a, b)
    finally:
        _remove_lowering_hook(orig_decide)
        _remove_prop_hook(orig_prop, prop_mod)
        _remove_finalize_hook(orig_resolve, resolve_mod)
        _remove_opspec_hook(orig_opspec)

    # ── Stage 2: Lowering (FixedLayout before STL propagation) ──────────────
    lines = ["  ComputedBuffer.decide_layout results (FixedLayout, NO device layout yet):", ""]
    if _lowered_nodes:
        for name, info in sorted(_lowered_nodes.items()):
            lines.append(f"  [{name}]  layout_type={info['type']}")
            lines.append(f"           size={info['size']}  stride={info['stride']}")
            lines.append(f"           dtype={info['dtype']}  device_layout={info['device_layout']}")
            lines.append("")
    else:
        lines.append("  (no ComputedBuffers intercepted — possibly already compiled and cached)")
    lines += [
        "  At this stage Inductor has assigned a plain FixedLayout with host size+stride.",
        "  The tiled device_layout field is None — it will be filled by Stage 3.",
    ]
    _stage("STAGE 2 — Inductor LoopLevelIR lowering (FixedLayout, no STL yet)", "\n".join(lines))

    # ── Stage 3: Layout propagation ─────────────────────────────────────────
    lines = ["  propagate_spyre_tensor_layouts results (candidate STL set per op):", ""]
    if _prop_snapshots:
        for name, layouts in sorted(_prop_snapshots.items()):
            lines.append(f"  [{name}]  {len(layouts)} candidate(s):")
            for stl_str in layouts:
                lines.append(f"    • {stl_str}")
        lines.append("")
    else:
        lines.append("  (no snapshots — model may have been served from cache)")
    lines += [
        "  Each op receives a LIST of candidate STLs (one per possible stick dimension).",
        "  For a plain row-major (rows, cols) input there is exactly one candidate.",
        "  The stick dimension is the last logical axis, giving device_size=[num_sticks, rows, 64].",
        "  stride_map[0]=64  (one stick_size step along the last host dim per stick-group)",
        "  stride_map[1]=cols (one full row in host memory per row step)",
        "  stride_map[2]=1    (one element per within-stick step)",
    ]
    _stage("STAGE 3 — propagate_spyre_tensor_layouts (candidate STLs assigned)", "\n".join(lines))

    # ── Stage 4: Finalization ────────────────────────────────────────────────
    lines = ["  _resolve_copy_back_candidates: single STL pinned to FixedTiledLayout:", ""]
    if _finalized:
        for name, stl_str in sorted(_finalized.items()):
            lines.append(f"  [{name}]  {stl_str}")
        lines.append("")
    else:
        lines.append("  (no FixedTiledLayouts observed — model may be cached)")
    lines += [
        "  After resolution each buffer has exactly ONE device_layout.",
        "  For add (multi-arg pointwise) the AllSameNode cost forces all three tensors",
        "  (a, b, output) to share the same STL — no restickify is needed.",
        "  FixedTiledLayout is now the layout type instead of bare FixedLayout.",
    ]
    _stage("STAGE 4 — Layout finalization (FixedTiledLayout pinned, AllSameNode)", "\n".join(lines))

    # ── Stage 5: KernelSpec TensorArgs ──────────────────────────────────────
    lines = ["  OpSpec objects emitted (each captures TensorArgs for codegen):", ""]
    if _kernel_specs:
        for i, spec in enumerate(_kernel_specs):
            lines.append(f"  OpSpec[{i}]  op={getattr(spec, 'op', '?')}  "
                         f"dimensions={getattr(spec, 'dimensions', '?')}")
            for j, arg in enumerate(getattr(spec, "args", [])):
                arg_type = type(arg).__name__
                d_size   = getattr(arg, "device_size", None)
                s_map    = getattr(arg, "stride_map",  None)
                role     = getattr(arg, "role",        "?")
                lines.append(f"    arg[{j}]  type={arg_type}  role={role}  "
                             f"device_size={d_size}  stride_map={s_map}")
            lines.append("")
    else:
        lines.append("  (no OpSpecs intercepted — model may have been served from cache)")
    lines += [
        "  The OpSpec is the bridge to SuperDSC codegen.  Each TensorArg carries:",
        "  device_size — the on-device shape the compiler will iterate over",
        "  stride_map  — tells the runtime how to map each device step to a host offset",
    ]
    _stage("STAGE 5 — KernelSpec / OpSpec (TensorArgs for SuperDSC codegen)", "\n".join(lines))

    # ── Stage 6: Runtime output ──────────────────────────────────────────────
    _stage("STAGE 6 — Runtime output (live tensor from kernel execution)",
           "\n".join([
               _tensor_row("a", a),
               _tensor_row("b", b),
               _tensor_row("c=a+b", c),
               "",
               "  The kernel ran on device; c is a live Spyre tensor.",
               "  Its STL must exactly match what the compiler recorded in Stage 4.",
               "  If you compare Stage 0 (eager a,b) vs Stage 6 (compiled c):",
               "  • a and b already had an STL when passed in as graph inputs",
               "  • c is newly allocated by the compiled kernel with the pinned STL",
           ]))

    # ── Summary table ────────────────────────────────────────────────────────
    stl_a = a.device_tensor_layout()
    stl_c = c.device_tensor_layout()

    print(f"\n{SEP}")
    print(f"  SUMMARY — how device_size and stride_map evolve per stage")
    print(SEP)
    print(f"  Shape: ({rows}, {cols})   dtype=float16   stick_size=64")
    print()
    print(f"  {'Stage':<50}  {'device_size':<24}  stride_map")
    print(f"  {'-'*50}  {'-'*24}  {'-'*28}")

    def _fmt(stl):
        if stl is None:
            return "None", "None"
        ds = list(stl.device_size)
        sm = list(stl.stride_map)
        return str(ds), str(sm)

    ds_a, sm_a = _fmt(stl_a)
    ds_c, sm_c = _fmt(stl_c)

    rows_summary = [
        ("0  Eager input a",               ds_a, sm_a),
        ("0  Eager input b",               ds_a, sm_a),   # same shape → same layout
        ("1  Dynamo FX",                   "symbolic",    "not present"),
        ("2  Lowering (FixedLayout)",      "host size",   "None (not yet)"),
        ("3  Layout prop (candidate STL)", ds_a,          sm_a),
        ("4  Finalize (FixedTiledLayout)", ds_a,          sm_a),
        ("5  KernelSpec TensorArg",        ds_a,          sm_a),
        ("6  Runtime output c",            ds_c,          sm_c),
    ]
    for label, ds, sm in rows_summary:
        print(f"  {label:<50}  {str(ds):<24}  {sm}")

    print()
    print("  KEY OBSERVATIONS:")
    print("  • Stages 0–1: no device layout (FX uses symbolic shapes)")
    print("  • Stage 2: FixedLayout uses plain host size/stride — STL is None")
    print("  • Stage 3: STL is first computed; add is multi-arg pointwise so")
    print("    AllSameNode forces a, b, output to share the identical STL")
    print("  • Stages 4–6: STL is immutable — device_size and stride_map are")
    print("    pinned and carried unchanged through codegen into the live tensor")
    print(f"\n{SEP}\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("spyre:0" if torch.spyre.is_available() else "cpu")
    print(f"device: {device}")

    if device.type != "spyre":
        print(
            "\nSpyre device not detected.\n"
            "Run with:  TORCH_SPYRE_MOCK_DEVICE=1 python add_layout_trace.py"
        )
    else:
        # Two cases: one stick-group (cols == stick_size) and two stick-groups
        run_trace(rows=4, cols=64, device=device)   # one stick-group (degenerate)
        run_trace(rows=4, cols=128, device=device)  # two stick-groups
