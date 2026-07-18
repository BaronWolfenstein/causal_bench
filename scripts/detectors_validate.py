"""Validate the #99 geometry detectors (the gate) on synthetic regimes with
KNOWN structure — the detector must FIRE on curved / stratified and stay QUIET on
flat.  No real data.

    PYTHONPATH=~/causal_bench python scripts/detectors_validate.py
"""
from __future__ import annotations
import numpy as np

from causal_bench.geometry import detectors as D


def make_flat(n=3000, dim=10, seed=0):
    return np.random.default_rng(seed).standard_normal((n, dim))          # isotropic R^dim


def make_curved(n=3000, seed=1):
    rng = np.random.default_rng(seed)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    h = rng.uniform(0, 21, n)
    return np.c_[t * np.cos(t), h, t * np.sin(t)]                          # swiss roll: 2D in R^3


def make_stratified(n=3000, dim=10, seed=2):
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((n // 2, dim)) * 0.3
    b = rng.standard_normal((n - n // 2, dim)) * 0.3
    b[:, 0] += 8.0                                                          # two disjoint sheets
    return np.vstack([a, b])


def report(name, X, expect):
    s = D.screen(X)
    print(f"\n[{name}]  (expect: {expect})")
    print(f"  intrinsic_dim = {s['intrinsic_dim']:.1f} / ambient {s['ambient_dim']:.0f}"
          f"   geodesic_growth = {s['geodesic_growth']:.2f} (frac_finite {s['geodesic_frac_finite']:.2f})")
    print(f"  n_components = {s['n_components']}   algebraic_connectivity = {s['algebraic_connectivity']:.2e}")
    print(f"  -> curved={s['curved']}  stratified={s['stratified']}  "
          f"flat_approx_fails={s['flat_approx_fails']}")
    return s


if __name__ == "__main__":
    print("=== #99 detector gate — curved/stratified must fire, flat must not ===")
    sf = report("FLAT (Gaussian R^10)", make_flat(), "no failure")
    sc = report("CURVED (swiss roll)", make_curved(), "curved")
    ss = report("STRATIFIED (2 sheets)", make_stratified(), "stratified")

    ok = (not sf["flat_approx_fails"]) and sc["curved"] and ss["stratified"]
    print(f"\nGATE VALIDATION: {'PASS' if ok else 'FAIL'}  "
          f"(flat quiet={not sf['flat_approx_fails']}, curved fires={sc['curved']}, "
          f"stratified fires={ss['stratified']})")
    assert ok
