#!/usr/bin/env python3
"""
Validate the MLA absorption fold for the GLM-5.2 -> ds4 port.

ds4 runs *absorbed* MLA: each head's query is a single vector in the KV latent
space, and there is no per-head kv_b up-projection at inference time. GLM-5.2
ships *un-absorbed* weights (q_b_proj, kv_b_proj, o_proj). The converter must fold
kv_b into q_b (keys) and into o_proj (values). This script proves that fold is
mathematically exact, on real layer-0 weights, two independent ways:

  (1) activation-level: native un-absorbed attention == absorbed attention
  (2) weight-level: the pre-folded attn_q_b matrix reproduces the absorbed query

It does NOT need the GPU or much RAM (a few layer-0 matrices), so it can run with
the production server up. RoPE is included as an (unrotated) decoupled channel,
which is neutral to the absorption algebra.

Usage: python3 check_mla_absorption.py [--hf /Volumes/4TB-1/glm-5.2] [--layer 0]
"""
import argparse, glob, json, struct
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--hf", default="/Volumes/4TB-1/glm-5.2")
ap.add_argument("--layer", type=int, default=0)
args = ap.parse_args()

# GLM-5.2 attention dims (config.json)
H, QL, NH = 6144, 2048, 64
NOPE, ROPE, KVL, VH = 192, 64, 512, 256
EPS = 1e-5
L = args.layer

names = {
    "q_a":    f"model.layers.{L}.self_attn.q_a_proj.weight",
    "q_a_ln": f"model.layers.{L}.self_attn.q_a_layernorm.weight",
    "q_b":    f"model.layers.{L}.self_attn.q_b_proj.weight",
    "kv_a":   f"model.layers.{L}.self_attn.kv_a_proj_with_mqa.weight",
    "kv_a_ln":f"model.layers.{L}.self_attn.kv_a_layernorm.weight",
    "kv_b":   f"model.layers.{L}.self_attn.kv_b_proj.weight",
    "o":      f"model.layers.{L}.self_attn.o_proj.weight",
}
want = set(names.values())

index = {}
for f in sorted(glob.glob(args.hf + "/model-*.safetensors")):
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        hdr = json.loads(fh.read(n))
    base = 8 + n
    for k, v in hdr.items():
        if k in want:
            index[k] = (f, base, v)
    if want <= set(index):
        break
missing = want - set(index)
if missing:
    raise SystemExit("missing tensors: " + ", ".join(sorted(missing)))


def read(name):
    f, base, meta = index[name]
    b0, b1 = meta["data_offsets"]
    with open(f, "rb") as fh:
        fh.seek(base + b0)
        raw = fh.read(b1 - b0)
    dt = meta["dtype"]
    if dt == "BF16":
        u = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16
        a = u.view(np.float32)
    elif dt == "F32":
        a = np.frombuffer(raw, dtype=np.float32)
    else:
        raise SystemExit("unsupported dtype " + dt)
    return a.reshape(meta["shape"]).astype(np.float64)


W = {k: read(v) for k, v in names.items()}


def rms(x, w):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + EPS) * w


def softmax_causal(s):  # s[t,s,i], mask s>t
    T = s.shape[0]
    m = np.triu(np.ones((T, T), bool), 1)
    s = np.where(m[:, :, None], -1e30, s)
    s = s - s.max(1, keepdims=True)
    e = np.exp(s)
    return e / e.sum(1, keepdims=True)


T = 5
rng = np.random.default_rng(0)
h = rng.standard_normal((T, H)) * 0.1
scale = 1.0 / np.sqrt(NOPE + ROPE)

# ---- shared projections ----
q_a = rms(h @ W["q_a"].T, W["q_a_ln"])                 # [T,2048]
q = (q_a @ W["q_b"].T).reshape(T, NH, NOPE + ROPE)      # [T,64,256]
q_nope, q_rope = q[..., :NOPE], q[..., NOPE:]
kva = h @ W["kv_a"].T                                   # [T,576]
ckv = rms(kva[:, :KVL], W["kv_a_ln"])                   # [T,512]
k_rope = kva[:, KVL:]                                   # [T,64] (shared across heads)

# ---- native (un-absorbed) ----
kv = (ckv @ W["kv_b"].T).reshape(T, NH, NOPE + VH)      # [T,64,448]
k_nope, v = kv[..., :NOPE], kv[..., NOPE:]
S = (np.einsum("tih,sih->tsi", q_nope, k_nope) +
     np.einsum("tih,sh->tsi", q_rope, k_rope)) * scale
A = softmax_causal(S)
o_native = np.einsum("tsi,sih->tih", A, v).reshape(T, NH * VH) @ W["o"].T

# ---- absorbed (the ds4 form) ----
kvb = W["kv_b"].reshape(NH, NOPE + VH, KVL)
W_UK = kvb[:, :NOPE, :]                                 # [64,192,512] c_kv -> k_nope
W_UV = kvb[:, NOPE:, :]                                 # [64,256,512] c_kv -> v
q_nope_abs = np.einsum("tih,ihl->til", q_nope, W_UK)    # [T,64,512]
S2 = (np.einsum("til,sl->tsi", q_nope_abs, ckv) +
      np.einsum("tih,sh->tsi", q_rope, k_rope)) * scale
A2 = softmax_causal(S2)
ctx = np.einsum("tsi,sl->til", A2, ckv)                 # [T,64,512]
o_abs = np.einsum("til,iml->tim", ctx, W_UV).reshape(T, NH * VH) @ W["o"].T

# ---- weight-level fold (the actual converter formula for attn_q_b) ----
qb = W["q_b"].reshape(NH, NOPE + ROPE, QL)
qb_nope, qb_rope = qb[:, :NOPE, :], qb[:, NOPE:, :]      # [64,192,2048],[64,64,2048]
# per head: [ W_UK_i^T @ qb_nope_i  (512x2048) ; qb_rope_i (64x2048) ] -> [576,2048]
fold_nope = np.einsum("ihl,ihq->ilq", W_UK, qb_nope)    # [64,512,2048]
attn_q_b = np.concatenate([fold_nope, qb_rope], axis=1)  # [64,576,2048]
q_abs2 = np.einsum("tq,ilq->til", q_a, attn_q_b)         # [T,64,576]
q_nope_abs2, q_rope2 = q_abs2[..., :KVL], q_abs2[..., KVL:]


def rel(a, b):
    return np.abs(a - b).max() / (np.abs(b).max() + 1e-12)


print(f"layer {L}: activation absorption  max|native-abs|={np.abs(o_native-o_abs).max():.3e}  rel={rel(o_abs,o_native):.3e}")
print(f"layer {L}: attention weights      max|A-A2|     ={np.abs(A-A2).max():.3e}")
print(f"layer {L}: weight-fold q_nope_abs max|.|        ={np.abs(q_nope_abs2-q_nope_abs).max():.3e}  rel={rel(q_nope_abs2,q_nope_abs):.3e}")
print(f"layer {L}: weight-fold q_rope     max|.|        ={np.abs(q_rope2-q_rope).max():.3e}")
ok = (rel(o_abs, o_native) < 1e-9 and rel(q_nope_abs2, q_nope_abs) < 1e-9)
print("RESULT:", "PASS - absorption fold is exact" if ok else "FAIL - fold mismatch, do not proceed")
