"""Canonical KS diagnostic: fixed-point order parameter + linear stability."""
import numpy as np

from causal_bench.diagnostics.tree_reconstruction import (
    ks_threshold, bp_fixed_point_magnetization, reconstruction_threshold,
    linear_stability_multiplier,
)


def test_fixed_point_positive_below_ks_zero_above():
    eps_c = ks_threshold(3)                               # ≈ 0.211
    assert bp_fixed_point_magnetization(3, 0.5 * eps_c, pop=4000, seed=0) > 0.3
    assert bp_fixed_point_magnetization(3, min(0.49, 1.6 * eps_c), pop=4000, seed=0) < 0.05


def test_bisection_threshold_matches_ks():
    # binary symmetric + small branching ⇒ reconstruction threshold = KS. The
    # residual is finite-pop + critical slowing-down near the continuous transition
    # (the linear-stability multiplier below is the sharper, bias-free locator).
    for b in (2, 4):
        eps_c = reconstruction_threshold(b, pop=6000, seed=1)
        assert abs(eps_c - ks_threshold(b)) < 0.03       # canonical self-consistency


def test_linear_multiplier_equals_b_lambda_squared():
    for b, eps in ((2, 0.10), (3, 0.15), (4, 0.05)):
        mult = linear_stability_multiplier(b, eps, pop=40000, seed=2)
        expected = b * (1 - 2 * eps) ** 2
        assert abs(mult - expected) < 0.15 * expected    # multiplier ≈ b·λ²


def test_multiplier_crosses_one_at_ks():
    b = 3
    eps_c = ks_threshold(b)
    assert linear_stability_multiplier(b, 0.5 * eps_c, pop=40000, seed=3) > 1.0
    assert linear_stability_multiplier(b, 1.5 * eps_c, pop=40000, seed=3) < 1.0
