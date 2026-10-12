#!/usr/bin/env python3
"""
trace_gemma4_view_ops.py

Runs a minimal Gemma 4 26B-A4B-it forward pass eagerly on CPU and records
every view operation that fires, with exact tensor shapes.

Real config values from google/gemma-4-26B-A4B-it/config.json:
  hidden_size=2816, num_attention_heads=16, num_key_value_heads=8 (sliding),
  num_global_key_value_heads=2 (full), head_dim=256, global_head_dim=512,
  num_hidden_layers=30, num_experts=128, top_k_experts=8,
  moe_intermediate_size=704, intermediate_size=2112,
  sliding_window=1024, partial_rotary_factor=0.25 (full attn only),
  layer_types=[sliding x5, full] x5
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import defaultdict

# ── Config ────────────────────────────────────────────────────────────────────

CFG = {
    "hidden_size":                 2816,
    "num_attention_heads":         16,
    "num_key_value_heads":         8,    # sliding attention
    "num_global_key_value_heads":  2,    # full attention
    "head_dim":                    256,  # sliding attention
    "global_head_dim":             512,  # full attention
    "num_experts":                 128,
    "top_k_experts":               8,
    "moe_intermediate_size":       704,
    "intermediate_size":           2112,
    "sliding_window":              1024,
    "partial_rotary_factor":       0.25, # full attention RoPE only
    "layer_types": (["sliding_attention"] * 5 + ["full_attention"]) * 5,
}

# ── Hook infrastructure ───────────────────────────────────────────────────────

_log = defaultdict(list)

def _s(t):
    return tuple(t.shape) if isinstance(t, torch.Tensor) else None

def _record(op, ins, out, note=""):
    _log[op].append({"in": ins, "out": out, "note": note})

# Save originals before installing hooks
_orig_view       = torch.Tensor.view
_orig_reshape    = torch.Tensor.reshape
_orig_permute    = torch.Tensor.permute
_orig_transpose  = torch.Tensor.transpose
_orig_clone      = torch.Tensor.clone
_orig_unsqueeze  = torch.Tensor.unsqueeze
_orig_squeeze    = torch.Tensor.squeeze
_orig_expand     = torch.Tensor.expand
_orig_flatten    = torch.Tensor.flatten
_orig_narrow     = torch.Tensor.narrow
_orig_select     = torch.Tensor.select
_orig_split      = torch.Tensor.split
_orig_cat        = torch.cat
_orig_stack      = torch.stack
_orig_contiguous = torch.Tensor.contiguous

def _hook_view(self, *shape):
    r = _orig_reshape(self, *shape)
    _record("aten.view", [_s(self)], _s(r))
    return r

def _hook_reshape(self, *shape):
    r = _orig_reshape(self, *shape)
    _record("aten.reshape", [_s(self)], _s(r))
    return r

def _hook_permute(self, *dims):
    d = dims[0] if len(dims)==1 and isinstance(dims[0],(list,tuple)) else dims
    r = _orig_permute(self, d)
    _record("aten.permute", [_s(self)], _s(r), str(d))
    return r

def _hook_transpose(self, d0, d1):
    r = _orig_transpose(self, d0, d1)
    _record("aten.transpose", [_s(self)], _s(r), f"({d0},{d1})")
    return r

def _hook_clone(self, **kwargs):
    mf = kwargs.get("memory_format", torch.preserve_format)
    r = _orig_clone(self, memory_format=mf)
    _record("aten.clone", [_s(self)], _s(r), str(mf))
    return r

def _hook_unsqueeze(self, dim):
    r = _orig_unsqueeze(self, dim)
    _record("aten.unsqueeze", [_s(self)], _s(r), f"dim={dim}")
    return r

def _hook_squeeze(self, *args):
    r = _orig_squeeze(self, *args)
    _record("aten.squeeze", [_s(self)], _s(r))
    return r

def _hook_expand(self, *sizes):
    sz = sizes[0] if len(sizes)==1 and isinstance(sizes[0],(list,tuple)) else sizes
    r = _orig_expand(self, sz)
    _record("aten.expand", [_s(self)], _s(r), str(sz))
    return r

def _hook_flatten(self, start=0, end=-1):
    r = _orig_flatten(self, start, end)
    _record("aten.flatten", [_s(self)], _s(r), f"{start}..{end}")
    return r

def _hook_narrow(self, dim, start, length):
    r = _orig_narrow(self, dim, start, length)
    _record("aten.narrow", [_s(self)], _s(r), f"dim={dim}")
    return r

def _hook_select(self, dim, idx):
    r = _orig_select(self, dim, idx)
    _record("aten.select", [_s(self)], _s(r), f"dim={dim}")
    return r

def _hook_split(self, split_size, dim=0):
    r = _orig_split(self, split_size, dim)
    _record("aten.split", [_s(self)], [_s(x) for x in r], f"size={split_size},dim={dim}")
    return r

def _hook_contiguous(self, **kwargs):
    r = _orig_contiguous(self, **kwargs)
    _record("aten.contiguous", [_s(self)], _s(r))
    return r

def _hook_cat(tensors, dim=0, **kw):
    r = _orig_cat(tensors, dim, **kw)
    _record("aten.cat", [_s(t) for t in tensors if isinstance(t, torch.Tensor)], _s(r), f"dim={dim}")
    return r

def _hook_stack(tensors, dim=0, **kw):
    r = _orig_stack(tensors, dim, **kw)
    _record("aten.stack", [_s(t) for t in tensors if isinstance(t, torch.Tensor)], _s(r), f"dim={dim}")
    return r

def install():
    torch.Tensor.view        = _hook_view
    torch.Tensor.reshape     = _hook_reshape
    torch.Tensor.permute     = _hook_permute
    torch.Tensor.transpose   = _hook_transpose
    torch.Tensor.clone       = _hook_clone
    torch.Tensor.unsqueeze   = _hook_unsqueeze
    torch.Tensor.squeeze     = _hook_squeeze
    torch.Tensor.expand      = _hook_expand
    torch.Tensor.flatten     = _hook_flatten
    torch.Tensor.narrow      = _hook_narrow
    torch.Tensor.select      = _hook_select
    torch.Tensor.split       = _hook_split
    torch.Tensor.contiguous  = _hook_contiguous
    torch.cat                = _hook_cat
    torch.stack              = _hook_stack

def uninstall():
    torch.Tensor.view        = _orig_view
    torch.Tensor.reshape     = _orig_reshape
    torch.Tensor.permute     = _orig_permute
    torch.Tensor.transpose   = _orig_transpose
    torch.Tensor.clone       = _orig_clone
    torch.Tensor.unsqueeze   = _orig_unsqueeze
    torch.Tensor.squeeze     = _orig_squeeze
    torch.Tensor.expand      = _orig_expand
    torch.Tensor.flatten     = _orig_flatten
    torch.Tensor.narrow      = _orig_narrow
    torch.Tensor.select      = _orig_select
    torch.Tensor.split       = _orig_split
    torch.Tensor.contiguous  = _orig_contiguous
    torch.cat                = _orig_cat
    torch.stack              = _orig_stack

# ── Minimal model ─────────────────────────────────────────────────────────────

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d))
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6) * self.w


def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x.narrow(-1, h, h), x.narrow(-1, 0, h)], dim=-1)


def rope(x, seq_len, rot_dim):
    # cos/sin: [1, 1, S, rot_dim]  broadcasts over batch and heads
    half = rot_dim // 2
    inv_freq = 1.0 / (10000 ** (torch.arange(0, half).float() / half))
    t = torch.arange(seq_len).float()
    f = torch.outer(t, inv_freq)           # [S, half]
    c = torch.cat([f.cos(), f.cos()], -1).unsqueeze(0).unsqueeze(0)  # [1,1,S,rot_dim]
    s = torch.cat([f.sin(), f.sin()], -1).unsqueeze(0).unsqueeze(0)
    return x * c + rotate_half(x) * s


class Attention(nn.Module):
    def __init__(self, cfg, is_global):
        super().__init__()
        self.is_global = is_global
        self.H   = cfg["num_attention_heads"]
        self.KVH = cfg["num_global_key_value_heads"] if is_global else cfg["num_key_value_heads"]
        self.D   = cfg["global_head_dim"] if is_global else cfg["head_dim"]
        self.W   = cfg["sliding_window"]
        self.prf = cfg["partial_rotary_factor"] if is_global else 1.0
        H, KVH, D, hidden = self.H, self.KVH, self.D, cfg["hidden_size"]
        self.q_proj = nn.Linear(hidden, H*D,   bias=False)
        self.k_proj = nn.Linear(hidden, KVH*D, bias=False)
        self.v_proj = nn.Linear(hidden, KVH*D, bias=False)
        self.o_proj = nn.Linear(H*D, hidden,   bias=False)
        self.q_norm = RMSNorm(D)
        self.k_norm = RMSNorm(D)

    def forward(self, x, past_kv=None):
        B, S, _ = x.shape
        H, KVH, D = self.H, self.KVH, self.D

        q = self.q_proj(x).view(B, S, H,   D).permute(0,2,1,3)   # VIEW + PERMUTE
        k = self.k_proj(x).view(B, S, KVH, D).permute(0,2,1,3)   # VIEW + PERMUTE
        v = self.v_proj(x).view(B, S, KVH, D).permute(0,2,1,3)   # VIEW + PERMUTE

        q = self.q_norm(q)
        k = self.k_norm(k)

        # RoPE: rotate first rot_dim dims of head_dim
        rot_dim = int(D * self.prf)
        if self.prf < 1.0:
            q_r, q_p = q.narrow(-1,0,rot_dim), q.narrow(-1,rot_dim,D-rot_dim)  # NARROW x4
            k_r, k_p = k.narrow(-1,0,rot_dim), k.narrow(-1,rot_dim,D-rot_dim)
            q = torch.cat([rope(q_r,S,rot_dim), q_p], dim=-1)  # CAT
            k = torch.cat([rope(k_r,S,rot_dim), k_p], dim=-1)  # CAT
        else:
            q = rope(q, S, D)
            k = rope(k, S, D)

        # KV cache append
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)   # CAT
            v = torch.cat([past_kv[1], v], dim=2)   # CAT

        # GQA expansion
        exp = H // KVH
        if exp > 1:
            k = k.unsqueeze(2).expand(-1,-1,exp,-1,-1).flatten(1,2)  # UNSQUEEZE+EXPAND+FLATTEN
            v = v.unsqueeze(2).expand(-1,-1,exp,-1,-1).flatten(1,2)

        # Sliding window crop
        kv_len = k.shape[2]
        if not self.is_global and kv_len > self.W:
            k = k.narrow(2, kv_len-self.W, self.W)   # NARROW
            v = v.narrow(2, kv_len-self.W, self.W)

        # Attention
        out = torch.matmul(F.softmax(torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(D), dim=-1), v)
        # Reassemble
        out = out.transpose(1,2).clone(memory_format=torch.contiguous_format)  # TRANSPOSE + CLONE
        return self.o_proj(out.reshape(B, S, H*D))   # RESHAPE


class MoELayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, E, I = cfg["hidden_size"], cfg["num_experts"], cfg["moe_intermediate_size"]
        self.router = nn.Linear(H, E, bias=False)
        self.gate   = nn.Parameter(torch.empty(E, I, H)); nn.init.normal_(self.gate, std=0.01)
        self.up     = nn.Parameter(torch.empty(E, I, H)); nn.init.normal_(self.up,   std=0.01)
        self.down   = nn.Parameter(torch.empty(E, H, I)); nn.init.normal_(self.down, std=0.01)
        self.K = cfg["top_k_experts"]

    def forward(self, x):
        B, S, H = x.shape
        xf = x.view(B*S, H)                               # VIEW
        scores = F.softmax(self.router(xf), dim=-1)
        out = torch.zeros_like(xf)
        for e in range(self.K):                            # loop over top-k only (static shapes)
            w = scores.narrow(1, e, 1)                     # NARROW
            out = out + w * (F.silu(xf @ self.gate[e].t()) * (xf @ self.up[e].t())) @ self.down[e].t()
        return out.view(B, S, H)                           # VIEW


class DenseMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        H, I = cfg["hidden_size"], cfg["intermediate_size"]
        self.gate = nn.Linear(H, I, bias=False)
        self.up   = nn.Linear(H, I, bias=False)
        self.down = nn.Linear(I, H, bias=False)
    def forward(self, x):
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


class Layer(nn.Module):
    def __init__(self, cfg, idx):
        super().__init__()
        is_global = cfg["layer_types"][idx] == "full_attention"
        self.attn = Attention(cfg, is_global)
        self.mlp  = DenseMLP(cfg) if is_global else MoELayer(cfg)
        self.n1   = RMSNorm(cfg["hidden_size"])
        self.n2   = RMSNorm(cfg["hidden_size"])
    def forward(self, x, past_kv=None):
        x = x + self.attn(self.n1(x), past_kv=past_kv)
        x = x + self.mlp(self.n2(x))
        return x

# ── Trace runner ──────────────────────────────────────────────────────────────

def trace(seq_len, past_kv=None):
    global _log
    _log = defaultdict(list)
    # layer 0 = sliding+MoE, layer 5 = full+dense  — covers all patterns
    layers = [Layer(CFG, 0).eval(), Layer(CFG, 5).eval()]
    x = torch.randn(1, seq_len, CFG["hidden_size"])
    install()
    try:
        with torch.no_grad():
            for i, layer in enumerate(layers):
                pkv = past_kv[i] if past_kv else None
                x = layer(x, past_kv=pkv)
    finally:
        uninstall()
    return dict(_log)

# ── Pretty print ──────────────────────────────────────────────────────────────

def show(found, title):
    W = 72
    print(f"\n{'='*W}\n  {title}\n{'='*W}")
    print(f"\n  {'OP':<22} {'CALLS':>5}   UNIQUE SHAPES  (input → output)  [note]")
    print(f"  {'-'*68}")
    for op in sorted(found):
        entries = found[op]
        seen, uniq = set(), []
        for e in entries:
            k = (str(e["in"]), str(e["out"]))
            if k not in seen:
                seen.add(k); uniq.append(e)
        print(f"\n  {op:<22} {len(entries):>4}x")
        for e in uniq[:4]:
            ins = " + ".join(str(s) for s in e["in"] if s)
            print(f"    in : {ins}")
            print(f"    out: {e['out']}" + (f"   [{e['note']}]" if e["note"] else ""))
        if len(uniq) > 4:
            print(f"    … {len(uniq)-4} more unique shapes")

    print(f"\n  {'─'*W}")
    print(f"  {'OP':<22} {'FOUND?':>8}   {'TOTAL CALLS':>12}   SPYRE NOTE")
    print(f"  {'─'*W}")
    ROWS = [
        ("aten.view",        "reshape_via_cpu if layout incompatible"),
        ("aten.reshape",     "reshape_via_cpu if layout incompatible"),
        ("aten.permute",     "restickify (data copy) injected before bmm"),
        ("aten.transpose",   "zero-copy ReinterpretView when stick-aligned"),
        ("aten.clone",       "dedicated Spyre lowering in lowering.py:757"),
        ("aten.unsqueeze",   "zero-cost metadata op"),
        ("aten.expand",      "stride-0 → materialized before bmm"),
        ("aten.flatten",     "decomposes to reshape"),
        ("aten.narrow",      "check offset % 64 == 0 for fp16"),
        ("aten.cat",         "Spyre lowering; concat offset must be % 64"),
        ("aten.split",       "upstream SliceView; split point must be % 64"),
        ("aten.contiguous",  "decomposes to clone(contiguous_format)"),
        ("aten.squeeze",     "zero-cost metadata op"),
        ("aten.select",      "upstream SliceView; data-dep shapes in MoE"),
        ("aten.stack",       "decomposes to unsqueeze+cat"),
        ("aten.repeat",      "CPU fallback registered in fallbacks.py"),
        ("aten.unfold",      "CPU fallback via spyre::unfold"),
        ("aten.flip",        "no Spyre registration; upstream fallback"),
        ("aten.unbind",      "no Spyre registration"),
    ]
    for op, note in ROWS:
        cnt = len(found.get(op, []))
        status = "FOUND" if cnt > 0 else "  --"
        print(f"  {op:<22} {status:>8}   {cnt:>12}   {note}")
    print()

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("="*72)
    print("  Gemma 4 26B-A4B-it — View Operations (eager trace, real config)")
    print("="*72)
    print(f"""
  Config (text decoder, google/gemma-4-26B-A4B-it/config.json):
    hidden_size={CFG['hidden_size']}, heads={CFG['num_attention_heads']}Q/{CFG['num_key_value_heads']}KV(slide)/{CFG['num_global_key_value_heads']}KV(full)
    head_dim={CFG['head_dim']}(slide)/{CFG['global_head_dim']}(full), layers=30, experts={CFG['num_experts']}, top_k={CFG['top_k_experts']}
    GQA expansion: sliding={CFG['num_attention_heads']//CFG['num_key_value_heads']}x  full={CFG['num_attention_heads']//CFG['num_global_key_value_heads']}x
    sliding_window={CFG['sliding_window']}, partial_rotary_factor={CFG['partial_rotary_factor']}(full attn only)
    Tracing layer 0 (sliding+MoE) + layer 5 (full+dense) — all patterns covered
    """)

    # Prefill: full sequence, no cache
    print("  [1/2] Prefill  (seq_len=512, no KV cache)...")
    prefill = trace(seq_len=512)
    show(prefill, "PREFILL — seq_len=512, no KV cache")

    # Decode: 1 new token appended to existing cache of 512 tokens
    print("  [2/2] Decode   (seq_len=1, past_len=512 in KV cache)...")
    past = [
        (torch.randn(1, 8, 512, 256), torch.randn(1, 8, 512, 256)),   # sliding layer
        (torch.randn(1, 2, 512, 512), torch.randn(1, 2, 512, 512)),   # full layer
    ]
    decode = trace(seq_len=1, past_kv=past)
    show(decode, "DECODE  — seq_len=1, past_len=512 in KV cache")

if __name__ == "__main__":
    main()
