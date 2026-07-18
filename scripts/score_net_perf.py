"""On-box score-net perf layer (A100 spec §6b): bf16 / Tensor Cores / torch.compile.

    PYTHONPATH=~/causal_bench CUDA_VISIBLE_DEVICES=0 python scripts/score_net_perf.py

Validates bf16 forward ~= fp32 (within bf16 tolerance) and profiles training
throughput fp32 vs tf32 vs bf16 vs bf16+compile at real-embedding scale (the
dim=3 test MLP is memory-bound and shows nothing; a wide net is compute-bound
and exercises the Tensor Cores).  fp32 stays the default in code; this measures
what the opt-in buys.
"""
from __future__ import annotations
import copy, time
import numpy as np
import torch

from causal_bench.generative.vpsde import Schedule
from causal_bench.generative.score_net import ScoreMLP, train_score, make_torch_score_fn

assert torch.cuda.is_available()
print("device:", torch.cuda.get_device_name(0), "| torch", torch.__version__)

sch = Schedule(n_steps=100)

# ---- correctness: bf16 forward ~= fp32 (same weights), ABS diff ------------
torch.manual_seed(0)
DIMc = 256
Xc = np.random.default_rng(0).standard_normal((8192, DIMc)).astype(np.float32)
probe = np.random.default_rng(1).standard_normal((512, DIMc)).astype(np.float32)
m = ScoreMLP(DIMc, 1024)
train_score(m, Xc, sch, epochs=3, device="cuda", precision="fp32")
s_fp32 = make_torch_score_fn(copy.deepcopy(m), sch, device="cuda", precision="fp32")(probe, t=10)
s_bf16 = make_torch_score_fn(copy.deepcopy(m), sch, device="cuda", precision="bf16")(probe, t=10)
absd = np.abs(s_bf16 - s_fp32).max()
scale = np.abs(s_fp32).mean()
print(f"[correctness] bf16 vs fp32 forward: max abs diff = {absd:.3e} "
      f"(mean|score|={scale:.2e}) -> {absd/scale*100:.1f}% of scale  "
      f"{'OK' if absd < 0.05*scale + 1e-3 else 'HIGH'}")

# ---- throughput sweep: does bf16/Tensor Cores win as the net grows? --------
def bench(dim, hidden, batch, precision, compile_, epochs=30):
    torch.manual_seed(0)
    X = np.random.default_rng(0).standard_normal((batch, dim)).astype(np.float32)
    mm = ScoreMLP(dim, hidden)
    train_score(mm, X, sch, epochs=3, device="cuda", precision=precision, compile=compile_)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    train_score(mm, X, sch, epochs=epochs, device="cuda", precision=precision, compile=compile_)
    torch.cuda.synchronize()
    return epochs / (time.perf_counter() - t0)

print(f"\n[throughput] steps/s and bf16 speedup vs fp32 (A100)")
print(f"  {'dim':>5s} {'hidden':>6s} {'batch':>6s} {'fp32':>8s} {'tf32':>8s} "
      f"{'bf16':>8s} {'bf16+cmp':>9s} {'best/fp32':>10s}")
for dim, hidden, batch in [(256, 1024, 8192), (768, 4096, 16384), (1024, 8192, 16384)]:
    r = {}
    for name, prec, comp in [("fp32", "fp32", False), ("tf32", "tf32", False),
                             ("bf16", "bf16", False), ("cmp", "bf16", True)]:
        r[name] = bench(dim, hidden, batch, prec, comp)
    best = max(r["tf32"], r["bf16"], r["cmp"])
    print(f"  {dim:5d} {hidden:6d} {batch:6d} {r['fp32']:8.1f} {r['tf32']:8.1f} "
          f"{r['bf16']:8.1f} {r['cmp']:9.1f} {best/r['fp32']:9.2f}x")

print("\n[note] fp32 remains the code default (spec §6b); precision='bf16' + "
      "compile=True are opt-in. bf16 uses A100 Tensor Cores; tf32 accelerates "
      "fp32 matmuls; torch.compile fuses the MLP. Gains grow with width/batch "
      "(the score net is 'maximize-FLOPs' side of the roofline).")
