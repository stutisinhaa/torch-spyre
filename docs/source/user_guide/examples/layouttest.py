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
layouttest.py

Part 1 — Raw layout read
    Reads and prints every attribute directly from the SpyreTensorLayout object.
    No calculation. No simulation. Just what the runtime gives you.

Part 2 — Two formulae, same device coordinates
    Uses the values just read to show:

      device_offset = dot(device_coords, device_stride)
      host_offset   = dot(device_coords, stride_map)

    device_stride is the ONE value not stored by the runtime — it is derived
    from device_size (row-major strides of that shape). Everything else is a
    direct read.  The two formulae give different numbers for the same coords,
    which is the core of why stride_map exists.

Run:
    python layouttest.py                             # real hardware
    TORCH_SPYRE_MOCK_DEVICE=1 python layouttest.py  # mock device
"""

import torch

try:
    import torch_spyre
except ImportError:
    raise RuntimeError("torch_spyre is missing. Install via pip install torch-spyre.")


# ---------------------------------------------------------------------------
# Part 1 — Raw layout read: zero calculation, just attribute access
# ---------------------------------------------------------------------------

def print_raw_layout(tensor: torch.Tensor, label: str = ""):
    """
    Reads and prints every attribute directly from the SpyreTensorLayout object.
    No calculation. No simulation. Just raw values from the runtime.
    """
    print(f"\n{'='*62}")
    print(f"RAW LAYOUT  {label}  shape={tuple(tensor.shape)}")
    print(f"{'='*62}")

    # Standard PyTorch tensor attributes
    print(f"[tensor]  shape           = {tuple(tensor.shape)}")
    print(f"[tensor]  stride()        = {tuple(tensor.stride())}")
    print(f"[tensor]  dtype           = {tensor.dtype}")
    print(f"[tensor]  device          = {tensor.device}")
    print(f"[tensor]  is_contiguous() = {tensor.is_contiguous()}")
    print()

    # SpyreTensorLayout object — returned by the runtime
    layout = tensor.device_tensor_layout()
    if layout is None:
        print("  layout = None (FakeTensor or non-Spyre tensor)")
        print(f"{'='*62}\n")
        return

    # Every attribute stored directly on the layout object.
    # These come from the C++ runtime (or mock stub) — we only read them.
    for attr in ["device_size", "stride_map", "dim_map",
                 "device_dtype", "data_format", "dim_order",
                 "element_arrangement"]:
        val = getattr(layout, attr, "<not present>")
        print(f"[layout]  {attr:22s} = {val}")

    # elems_per_stick() is a method on the dtype, not a stored int
    dtype_obj = getattr(layout, "device_dtype", None)
    if dtype_obj is not None and hasattr(dtype_obj, "elems_per_stick"):
        print(f"[layout]  device_dtype.elems_per_stick() "
              f"= {dtype_obj.elems_per_stick()}")

    print(f"{'='*62}\n")


# ---------------------------------------------------------------------------
# Part 2 — Two formulae using the values you just read
# ---------------------------------------------------------------------------

def show_two_formulae(tensor: torch.Tensor, label: str = ""):
    """
    Shows the two offset formulae side by side for a fixed set of elements.

    Values read directly from the runtime:
      stride_map    ← layout.stride_map       (stored on layout)
      device_size   ← layout.device_size      (stored on layout)
      logical strides ← tensor.stride()       (stored on tensor)

    One derived value (NOT stored anywhere in the runtime):
      device_stride ← row-major strides of device_size
                      (the hardware computes this internally; Python must too)

    The two formulae, using the same device_coords:
      host_offset   = dot(device_coords, stride_map)    [reads stride_map]
      device_offset = dot(device_coords, device_stride) [uses derived value]
    """
    if tensor.dim() != 2:
        print(f"show_two_formulae: skipping '{label}' (2-D tensors only)\n")
        return

    layout = tensor.device_tensor_layout()
    if layout is None:
        print("No layout available.\n")
        return

    # ── direct reads from the runtime ───────────────────────────────────────
    device_size  = list(int(x) for x in layout.device_size)   # read from layout
    stride_map   = list(int(x) for x in layout.stride_map)    # read from layout
    log_strides  = list(int(x) for x in tensor.stride())      # read from tensor
    stick_size   = device_size[-1]                             # last entry of device_size

    # ── the ONE derived value ────────────────────────────────────────────────
    # device_stride is NOT stored by the runtime.
    # It is always the row-major strides of device_size — computed here
    # the same way PyTorch computes strides from a shape.
    s = 1
    device_stride = []
    for d in reversed(device_size):
        device_stride.insert(0, s)
        s *= d

    # ── print what we have ───────────────────────────────────────────────────
    rows, cols = int(tensor.shape[0]), int(tensor.shape[1])
    W = 76

    sample = []
    for (row, col) in [
        (0, 0),
        (0, stick_size - 1),
        (0, stick_size),
        (1, 0),
        (1, stick_size + 5),
        (rows - 1, cols - 1),
    ]:
        if row < rows and col < cols:
            sg  = col // stick_size
            pos = col % stick_size
            sample.append((row, col, sg, pos))

    # ── header ───────────────────────────────────────────────────────────────
    print(f"\n{'='*W}")
    print(f"  THREE FORMULAE  {label}  shape=({rows}, {cols})")
    print(f"{'='*W}")
    print(f"  Values read directly from the runtime:")
    print(f"    log_strides  = {log_strides}  ← tensor.stride()")
    print(f"    stride_map   = {stride_map}  ← layout.stride_map")
    print(f"    device_size  = {device_size}  ← layout.device_size")
    print(f"  One derived value (NOT stored by runtime):")
    print(f"    device_stride = {device_stride}  ← row-major strides of device_size")
    print()

    # ════════════════════════════════════════════════════════════════════════
    # FORMULA 1 — CPU / host
    # offset = dot(logical_coords, log_strides)
    # ════════════════════════════════════════════════════════════════════════
    print(f"  {'─'*W}")
    print(f"  FORMULA 1 (CPU)   offset = dot([row, col], log_strides={log_strides})")
    print(f"  {'─'*W}")
    print(f"  {'[row,col]':>10} | logical_coords | dot product expanded              | offset")
    print(f"  {'-'*10}-+---------------+-----------------------------------+-------")
    for (row, col, sg, pos) in sample:
        offset = row * log_strides[0] + col * log_strides[1]
        expanded = f"{row}×{log_strides[0]} + {col}×{log_strides[1]}"
        print(f"  [{row:2d},{col:3d}]      | ({row:2d}, {col:3d})      | {expanded:35s} | {offset}")
    print()

    # ════════════════════════════════════════════════════════════════════════
    # FORMULA 2 — Spyre device
    # device_offset = dot(device_coords, device_stride)
    # ════════════════════════════════════════════════════════════════════════
    print(f"  {'─'*W}")
    print(f"  FORMULA 2 (Spyre) device_offset = dot(device_coords, device_stride={device_stride})")
    print(f"  {'─'*W}")
    print(f"  {'[row,col]':>10} | device_coords     | dot product expanded                    | device_offset")
    print(f"  {'-'*10}-+-------------------+----------------------------------------+--------------")
    for (row, col, sg, pos) in sample:
        dc = (sg, row, pos)
        dev_off = sg * device_stride[0] + row * device_stride[1] + pos * device_stride[2]
        expanded = f"{sg}×{device_stride[0]} + {row}×{device_stride[1]} + {pos}×{device_stride[2]}"
        print(f"  [{row:2d},{col:3d}]      | {str(dc):17s} | {expanded:42s} | {dev_off}")
    print()
    print(f"  ^ These numbers live in the device's slot grid — unrelated to host offsets above.")
    print()

    # ════════════════════════════════════════════════════════════════════════
    # FORMULA 3 — the bridge / unifier
    # offset = dot(device_coords, stride_map)
    # ════════════════════════════════════════════════════════════════════════
    print(f"  {'─'*W}")
    print(f"  FORMULA 3 (bridge) offset = dot(device_coords, stride_map={stride_map})")
    print(f"  Same device_coords as Formula 2, but stride_map instead of device_stride.")
    print(f"  Result must equal Formula 1.  This is what stride_map was built to guarantee.")
    print(f"  {'─'*W}")
    print(f"  {'[row,col]':>10} | device_coords     | dot product expanded                    | offset | F1==F3?")
    print(f"  {'-'*10}-+-------------------+----------------------------------------+--------+-------")
    all_ok = True
    for (row, col, sg, pos) in sample:
        dc      = (sg, row, pos)
        f1      = row * log_strides[0]  + col * log_strides[1]
        f3      = sg  * stride_map[0]   + row * stride_map[1]  + pos * stride_map[2]
        expanded = f"{sg}×{stride_map[0]} + {row}×{stride_map[1]} + {pos}×{stride_map[2]}"
        ok      = "✓" if f1 == f3 else "✗ BUG"
        if f1 != f3:
            all_ok = False
        print(f"  [{row:2d},{col:3d}]      | {str(dc):17s} | {expanded:42s} | {f3:6d} | {ok}")

    print()
    if all_ok:
        print(f"  ✓ Formula 1 == Formula 3 for every element.")
        print(f"    stride_map successfully unites the CPU and Spyre address spaces.")
    else:
        print(f"  ✗ Mismatch detected — stride_map is incorrect for this layout.")
    print(f"{'='*W}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = torch.device("spyre:0" if torch.spyre.is_available() else "cpu")
    print(f"device: {device}\n")

    if device.type != "spyre":
        print("Run with:  TORCH_SPYRE_MOCK_DEVICE=1 python layouttest.py")
    else:
        # ── Part 1: raw reads ────────────────────────────────────────────────
        print("━" * 62)
        print("PART 1 — Raw layout attributes (direct reads, no calculation)")
        print("━" * 62)

        print_raw_layout(
            torch.arange(65, dtype=torch.float16, device=device),
            "1D (65,)")

        t2 = torch.randn(32, 100, dtype=torch.float16, device=device)
        print_raw_layout(t2, "2D (32, 100)")

        base = torch.randn(64, 128, dtype=torch.float16, device=device)
        print_raw_layout(base[::2, :], "non-contiguous (64,128)[::2,:]")

        # ── Part 2: two formulae ─────────────────────────────────────────────
        print("━" * 62)
        print("PART 2 — Two formulae using the values above")
        print("━" * 62)

        show_two_formulae(t2, "(32, 100)")

        t4 = torch.randn(4, 128, dtype=torch.float16, device=device)
        show_two_formulae(t4, "(4, 128)")

