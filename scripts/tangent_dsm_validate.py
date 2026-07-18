"""Validate the tangent-space-penalty DSM (#99 Layer-2) on a synthetic manifold.

    PYTHONPATH=~/causal_bench CUDA_VISIBLE_DEVICES=0 python scripts/tangent_dsm_validate.py

Train the same score net with and without the tangent penalty on a swiss roll
(2D in R^3); the penalized net's on-manifold score should be MORE tangent (smaller
normal-component fraction). No real data.
"""
from __future__ import annotations
import numpy as np
import torch

from causal_bench.generative.vpsde import Schedule
from causal_bench.generative.score_net import ScoreMLP
from causal_bench.geometry.tangent_dsm import (
    estimate_local_normals, train_score_tangent, normal_component_fraction,
)


def swiss_roll(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    X = np.c_[t * np.cos(t), rng.uniform(0, 21, n), t * np.sin(t)].astype(np.float32)
    return (X - X.mean(0)) / X.std(0)                        # standardize (like ZCA-ish)


if __name__ == "__main__":
    print("device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
    X = swiss_roll()
    sch = Schedule(n_steps=100)
    normals = estimate_local_normals(X, k=15, intrinsic_dim=2)          # (N, 1, 3)
    print(f"swiss roll N={len(X)} in R^3, intrinsic dim 2 -> {normals.shape[1]} normal/pt")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0); base = ScoreMLP(3, 128)
    train_score_tangent(base, X, sch, normals, lam=0.0, epochs=300,
                        rng=np.random.default_rng(1), device=dev)
    torch.manual_seed(0); pen = ScoreMLP(3, 128)
    train_score_tangent(pen, X, sch, normals, lam=3.0, epochs=300,
                        rng=np.random.default_rng(1), device=dev)

    nb = normal_component_fraction(base, X, normals, sch, t=5, device=dev)
    npn = normal_component_fraction(pen, X, normals, sch, t=5, device=dev)
    print(f"\non-manifold score normal-component fraction (0 = fully tangent):")
    print(f"  baseline (lam=0):   {nb:.3f}")
    print(f"  tangent-penalized:  {npn:.3f}   ({(1-npn/nb)*100:.0f}% less off-manifold)")
    ok = npn < nb
    print(f"\nTANGENT-DSM VALIDATION: {'PASS' if ok else 'FAIL'} "
          f"(penalized score is {'more' if ok else 'NOT more'} tangent)")
    assert ok
