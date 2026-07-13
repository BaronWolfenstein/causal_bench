"""RHM grammar + exact rule-BP — canonical SFW Part B (#131). The grammar produces
the genuine class-overlap transition the symmetric broadcast could not. Synthetic,
exact BP."""
import numpy as np

from causal_bench.diagnostics.rhm_grammar import (
    make_rhm, rhm_class_overlap, rhm_transition_scan, rhm_finite_size,
    rhm_bp_density_evolution, rhm_fss_collapse, rhm_density_evolution_threshold,
    make_lowrank_corruption, _sample_corruption, structured_corruption_shift,
)
from causal_bench.diagnostics.tree_reconstruction import diffusion_class_overlap


def test_grammar_has_expected_shape():
    rules = make_rhm(8, 2, 3, seed=0)
    assert rules.shape == (8, 3, 2) and rules.min() >= 0 and rules.max() < 8


def test_rhm_shows_a_class_overlap_transition():
    hi = rhm_class_overlap(8, 2, 2, 7, 1.0, n_trees=300, seed=1)   # clean
    lo = rhm_class_overlap(8, 2, 2, 7, 0.3, n_trees=300, seed=1)   # heavily corrupted
    assert hi > 0.5                                                # class recoverable clean
    assert lo < 0.15                                              # destroyed under corruption
    assert hi - lo > 0.5                                          # a genuine transition


def test_overlap_monotone_non_decreasing_in_theta():
    th = [0.3, 0.5, 0.7, 0.9, 1.0]
    m = [rhm_class_overlap(8, 2, 2, 7, t, n_trees=250, seed=2) for t in th]
    assert np.all(np.diff(m) >= -0.03)                            # less corruption ⇒ ≥ overlap


def test_grammar_transition_vs_broadcast_flatness():
    # THE canonical-SFW contrast: the grammar has a large-range transition; the
    # symmetric broadcast is flat (no transition) at the same corruption sweep.
    g_hi = rhm_class_overlap(8, 2, 2, 7, 1.0, n_trees=250, seed=3)
    g_lo = rhm_class_overlap(8, 2, 2, 7, 0.3, n_trees=250, seed=3)
    b_hi = diffusion_class_overlap(12, 3, 8, 0.85, 1.0, pop=4000, seed=3)
    b_lo = diffusion_class_overlap(12, 3, 8, 0.85, 0.3, pop=4000, seed=3)
    assert (g_hi - g_lo) > 0.5                                    # grammar: real transition
    assert abs(b_hi - b_lo) < 0.10                               # broadcast: flat


def test_transition_scan_locates_a_susceptibility_peak():
    r = rhm_transition_scan(8, 2, 2, 7, n_trees=300, seed=1)
    assert r["overlap"][-1] > r["overlap"][0] + 0.5              # rises across the sweep
    assert 0.3 < r["theta_star"] < 0.9                           # peak in the transition band


def test_finite_size_transition_sharpens_with_depth():
    # FSS: a genuine transition sharpens (width = 1/max-susceptibility shrinks) as
    # depth grows — the hallmark of a real phase transition vs a smooth crossover.
    r = rhm_finite_size(8, 2, 2, depths=(3, 5, 7), n_trees=300, seed=1)
    assert r["widths"][-1] < r["widths"][0]                      # deeper ⇒ sharper


def test_density_evolution_predicts_the_empirical_transition():
    # The rule-BP density-evolution (population dynamics) predictor transitions at the
    # same θ as exact BP on sampled trees — the recursion's fixed point matches the
    # empirical threshold (no full trees sampled).
    de_lo = rhm_bp_density_evolution(8, 2, 2, 20, 0.4, pop=4000, seed=3)
    de_hi = rhm_bp_density_evolution(8, 2, 2, 20, 0.7, pop=4000, seed=3)
    assert de_lo < 0.3                                            # below transition: collapsed
    assert de_hi > 0.5                                            # above transition: recovered
    emp = rhm_transition_scan(8, 2, 2, 8, n_trees=300, seed=2)["theta_star"]
    assert 0.35 < emp < 0.75                                     # empirical θ* in the DE band


def test_density_evolution_threshold_is_stable_across_seeds():
    # The FSS anchor's whole point is stability where theta_star(L) extrapolation
    # is not: population-dynamics cost is additive in pop, not exponential in depth.
    vals = [rhm_density_evolution_threshold(8, 2, 2, seed=s) for s in (0, 1, 2, 3)]
    assert max(vals) - min(vals) < 0.1
    assert all(0.3 < t < 0.6 for t in vals)


def test_fss_collapse_recovers_a_stable_theta_c_and_small_residual():
    # theta_c is anchored from density evolution (stable), not extrapolated from the
    # noisy finite-tree theta_star(L) (which can land outside [0, 1] with few depths).
    r = rhm_fss_collapse(8, 2, 2, (3, 5, 7), n_trees=200, n_reps=3, seed=10)
    assert 0.3 < r["theta_c"] < 0.6                              # matches DE anchor band
    assert r["nu"] > 0                                           # finite positive exponent
    assert r["collapse_residual"] < 0.05                         # curves genuinely collapse


def test_fss_collapse_theta_c_override():
    # An explicit theta_c bypasses the density-evolution anchor.
    r = rhm_fss_collapse(8, 2, 2, (3, 5, 7), n_trees=200, n_reps=2, seed=11, theta_c=0.5)
    assert r["theta_c"] == 0.5


# ─── structured (low-rank) corruption channel (#138) ─────────────────────────
def test_lowrank_corruption_is_valid_zero_diagonal_row_stochastic():
    C, U = make_lowrank_corruption(8, 4, beta=6.0, seed=0)
    assert C.shape == (8, 8)
    assert np.allclose(C.sum(1), 1.0)                            # row-stochastic
    assert np.allclose(np.diag(C), 0.0)                         # never replace with self
    assert (C >= 0).all()


def test_lowrank_corruption_beta_zero_is_uniform_over_others():
    C, _ = make_lowrank_corruption(8, 4, beta=0.0, seed=1)
    off = C[~np.eye(8, dtype=bool)]
    assert np.allclose(off, 1.0 / 7.0)                          # uniform over v-1 others


def test_lowrank_corruption_beta_positive_concentrates():
    C, _ = make_lowrank_corruption(8, 4, beta=6.0, seed=2)
    assert C.max(1).mean() > 3.0 / 7.0                          # far more peaked than uniform


def test_sample_corruption_never_returns_self_and_stays_in_support():
    C, _ = make_lowrank_corruption(8, 4, beta=6.0, seed=3)
    rng = np.random.default_rng(0)
    leaves = rng.integers(0, 8, 500)
    repl = _sample_corruption(leaves, C, rng)
    assert (repl != leaves).all()                               # zero diagonal ⇒ always changes
    assert repl.min() >= 0 and repl.max() < 8


def test_uniform_corruption_matrix_still_shows_a_transition():
    # The corruption-matrix code path (beta=0, uniform-over-others) must still
    # reproduce the SFW transition — overlap rises with theta.
    C, _ = make_lowrank_corruption(8, 4, beta=0.0, seed=0)
    r = rhm_transition_scan(8, 2, 2, 7, corruption=C, n_trees=250, seed=1)
    assert r["overlap"][-1] - r["overlap"][0] > 0.5


def test_concentrated_corruption_washes_out_the_transition():
    # THE #138 finding: a concentrated (low-rank) channel keeps corrupted leaves
    # informative, so the root class survives even under heavy corruption — the
    # overlap floor (at heaviest corruption) is lifted from ~0 toward the clean
    # value, washing out the transition. Robust in sign and magnitude across seeds.
    for sd in (0, 1, 2):
        r = structured_corruption_shift(8, 2, 2, 7, r=4, beta=6.0, n_trees=250, seed=sd)
        assert r["overlap_floor_uniform"] < 0.1                 # uniform: class destroyed
        assert r["overlap_floor_structured"] > 0.5             # structured: class survives
        assert r["floor_lift"] > 0.5                           # large, positive, consistent
