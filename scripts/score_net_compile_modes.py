"""On-box: torch.compile mode sweep for the score net (A100 §6b, deeper perf).

    PYTHONPATH=~/causal_bench CUDA_VISIBLE_DEVICES=0 python scripts/score_net_compile_modes.py

Builds on #152 (bf16 ~1.8x at real width). Compares torch.compile modes on top of
bf16 at that scale: 'default' (fuse), 'max-autotune' (kernel search, slow warmup),
'reduce-overhead' (CUDA graphs — kills per-step launch overhead for the fixed-shape
training loop). Warmup is excluded from timing (it also triggers the graph
capture / autotune search).
"""
from __future__ import annotations
import time
import numpy as np
import torch

from causal_bench.generative.vpsde import Schedule
from causal_bench.generative.score_net import ScoreMLP, train_score

assert torch.cuda.is_available()
print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

DIM, HIDDEN, BATCH = 768, 4096, 16384          # the scale where bf16 pays off
sch = Schedule(n_steps=100)
X = np.random.default_rng(0).standard_normal((BATCH, DIM)).astype(np.float32)


def bench(precision, compile_, mode, epochs=40, warm=6):
    torch.manual_seed(0)
    m = ScoreMLP(DIM, HIDDEN)
    train_score(m, X, sch, epochs=warm, device="cuda", precision=precision,
                compile=compile_, compile_mode=mode)          # warmup / graph capture
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_score(m, X, sch, epochs=epochs, device="cuda", precision=precision,
                compile=compile_, compile_mode=mode)
    torch.cuda.synchronize()
    return epochs / (time.perf_counter() - t0)


print(f"\n[compile-mode sweep] ScoreMLP dim={DIM} hidden={HIDDEN} batch={BATCH}, A100")
rows = [
    ("fp32",                 "fp32", False, "default"),
    ("bf16",                 "bf16", False, "default"),
    ("bf16+compile:default", "bf16", True,  "default"),
    ("bf16+compile:max-auto","bf16", True,  "max-autotune"),
    ("bf16+compile:cudagraph","bf16", True, "reduce-overhead"),
]
base = None
print(f"  {'config':>24s} {'steps/s':>9s} {'speedup':>8s}")
for name, prec, comp, mode in rows:
    try:
        sps = bench(prec, comp, mode)
        if base is None:
            base = sps
        print(f"  {name:>24s} {sps:9.1f} {sps/base:7.2f}x")
    except Exception as e:
        print(f"  {name:>24s}   FAILED: {type(e).__name__}: {str(e)[:60]}")

print("\n[note] max-autotune trades a long one-time warmup for steady-state kernels; "
      "reduce-overhead (CUDA graphs) removes per-step launch overhead but needs "
      "static shapes (the training loop qualifies). Pick per run length: graphs for "
      "long fixed-shape training, default for short/dynamic.")
