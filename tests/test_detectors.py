"""Off-box tests for the #99 detector gate: it must fire on curved / stratified
synthetic geometry and stay quiet on flat.  This is the gate whose *validity* has
to hold before it's ever pointed at a real embedding manifold.
"""
import numpy as np
import pytest

pytest.importorskip("sklearn")
pytest.importorskip("scipy")

from causal_bench.geometry import detectors as D


def _flat(n=1500, dim=8):
    return np.random.default_rng(0).standard_normal((n, dim))


def _curved(n=1500):
    rng = np.random.default_rng(1)
    t = rng.uniform(1.5 * np.pi, 4.5 * np.pi, n)
    return np.c_[t * np.cos(t), rng.uniform(0, 21, n), t * np.sin(t)]   # swiss roll (2D in R^3)


def _stratified(n=1500, dim=8):
    rng = np.random.default_rng(2)
    a = rng.standard_normal((n // 2, dim)) * 0.3
    b = rng.standard_normal((n - n // 2, dim)) * 0.3
    b[:, 0] += 8.0
    return np.vstack([a, b])


def test_flat_does_not_fire():
    s = D.screen(_flat())
    assert not s["flat_approx_fails"]
    assert s["intrinsic_dim"] > 0.85 * s["ambient_dim"]     # ID ~ ambient
    assert s["n_components"] == 1


def test_curved_fires_via_intrinsic_dim():
    s = D.screen(_curved())
    assert s["curved"] and s["flat_approx_fails"]
    assert s["intrinsic_dim"] < 2.6                          # swiss roll is 2D
    assert s["n_components"] == 1 and s["n_sheets"] == 1     # connected, one sheet


def test_disjoint_fires_via_components():
    s = D.screen(_stratified())
    assert s["stratified"] and s["flat_approx_fails"]
    assert s["n_components"] >= 2 and s["n_sheets"] >= 2     # disjoint sheets


def _weakly_bridged(n=1500, dim=8, seed=3, n_bridge=20):
    rng = np.random.default_rng(seed)
    m = (n - n_bridge) // 2
    a = rng.standard_normal((m, dim)) * 0.3
    b = rng.standard_normal((n - n_bridge - m, dim)) * 0.3; b[:, 0] += 8.0
    br = rng.standard_normal((n_bridge, dim)) * 0.15
    br[:, 0] = np.linspace(0.5, 7.5, n_bridge)              # thin bridge
    return np.vstack([a, b, br])


def test_eigengap_catches_weakly_bridged():
    """One connected component but two sheets: exact-count misses it (==1); the
    eigengap must catch it (>=2) — the whole point of adding the eigengap."""
    s = D.screen(_weakly_bridged())
    assert s["n_components"] == 1                            # connected via the bridge
    assert s["n_sheets"] >= 2 and s["stratified"]            # eigengap fires anyway
