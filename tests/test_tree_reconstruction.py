"""Broadcast-on-tree reconstruction: the statistical-mechanics-faithful version of
the #122 phase transition (real BP density evolution, KS threshold, finite-size
scaling). Synthetic; no patient data."""
import numpy as np

from causal_bench.diagnostics.tree_reconstruction import (
    ks_threshold, bp_magnetization, reconstruction_scan, finite_size_widths,
)


def test_ks_threshold_values():
    assert abs(ks_threshold(2) - 0.5 * (1 - 1 / np.sqrt(2))) < 1e-9
    assert abs(ks_threshold(4) - 0.25) < 1e-9          # b=4 → (1-1/2)/2


def test_reconstruction_possible_below_ks_impossible_above():
    # b=3: KS ε* ≈ 0.211. Below → magnetization survives; above → decays to 0.
    m_below = bp_magnetization(depth=14, branching=3, epsilon=0.08, pop=3000, seed=0)
    m_above = bp_magnetization(depth=14, branching=3, epsilon=0.33, pop=3000, seed=0)
    assert m_below > 0.3
    assert m_above < 0.05


def test_magnetization_non_increasing_in_noise():
    eps = np.linspace(0.03, 0.40, 10)
    m = [bp_magnetization(depth=11, branching=3, epsilon=e, pop=2500, seed=1) for e in eps]
    assert np.all(np.diff(m) <= 1e-6)


def test_susceptibility_peak_locates_ks_threshold():
    r = reconstruction_scan(3, depth=12, eps_grid=np.linspace(0.02, 0.40, 22),
                            pop=3000, seed=2)
    assert abs(r["eps_star"] - ks_threshold(3)) < 0.05
    # the reduced control parameter b·λ² ≈ 1 at the transition
    assert abs(r["control_at_star"] - 1.0) < 0.4


def test_finite_size_scaling_sharpens_the_transition():
    w = finite_size_widths(3, depths=(3, 6, 12),
                           eps_grid=np.linspace(0.02, 0.40, 22), pop=2500, seed=3)
    assert w["widths"][-1] < w["widths"][0]            # deeper ⇒ narrower ⇒ sharper
