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


def make_weakly_bridged(n=3000, dim=10, seed=3, n_bridge=25):
    """Two sheets joined by a thin bridge of a few transitional points: the graph
    is CONNECTED (1 exact zero) but strongly 2-sheeted — the eigengap must catch
    it where the exact-zero component count (=1) misses it."""
    rng = np.random.default_rng(seed)
    m = (n - n_bridge) // 2
    a = rng.standard_normal((m, dim)) * 0.3
    b = rng.standard_normal((n - n_bridge - m, dim)) * 0.3; b[:, 0] += 8.0
    br = rng.standard_normal((n_bridge, dim)) * 0.15
    br[:, 0] = np.linspace(0.5, 7.5, n_bridge)                             # the bridge
    return np.vstack([a, b, br])


def report(name, X, expect):
    s = D.screen(X)
    print(f"\n[{name}]  (expect: {expect})")
    print(f"  intrinsic_dim = {s['intrinsic_dim']:.1f} / ambient {s['ambient_dim']:.0f}"
          f"   geodesic_growth = {s['geodesic_growth']:.2f}")
    print(f"  n_components (exact) = {s['n_components']}   n_sheets (eigengap) = {s['n_sheets']}"
          f"   gap_significance = {s['gap_significance']:.1f}")
    print(f"  -> curved={s['curved']}  stratified={s['stratified']}  "
          f"flat_approx_fails={s['flat_approx_fails']}")
    return s


if __name__ == "__main__":
    print("=== #99 detector gate — curved/stratified must fire, flat must not ===")
    sf = report("FLAT (Gaussian R^10)", make_flat(), "no failure")
    sc = report("CURVED (swiss roll)", make_curved(), "curved")
    ss = report("STRATIFIED (2 disjoint sheets)", make_stratified(), "stratified")
    sw = report("WEAKLY-BRIDGED (2 sheets + bridge)", make_weakly_bridged(),
                "stratified via eigengap (exact count misses it)")

    ok = ((not sf["flat_approx_fails"]) and sc["curved"] and ss["stratified"]
          and sw["stratified"] and sw["n_components"] == 1)     # bridged: 1 component, 2 sheets
    print(f"\nGATE VALIDATION: {'PASS' if ok else 'FAIL'}")
    print(f"  flat quiet={not sf['flat_approx_fails']}  curved fires={sc['curved']}  "
          f"disjoint fires={ss['stratified']}  bridged fires={sw['stratified']} "
          f"(exact-count={sw['n_components']}, eigengap={sw['n_sheets']})")
    assert ok
