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
    assert s["n_components"] == 1                            # connected


def test_stratified_fires_via_components():
    s = D.screen(_stratified())
    assert s["stratified"] and s["flat_approx_fails"]
    assert s["n_components"] >= 2                            # disjoint sheets
