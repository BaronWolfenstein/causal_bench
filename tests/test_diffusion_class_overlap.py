"""Exact-BP diffusion class-overlap on a symmetric broadcast tree (#131 Part B
stepping-stone). The key result is a NEGATIVE one: the broadcast is robust to leaf
corruption (no SFW transition) — canonical SFW needs the RHM grammar."""
import numpy as np

from causal_bench.diagnostics.tree_reconstruction import (
    diffusion_class_overlap, qary_bp_magnetization, qary_ks_threshold,
)


def test_clean_diffusion_matches_planted_reconstruction():
    m = diffusion_class_overlap(16, 3, 4, 0.85, 1.0, pop=5000, seed=0)
    planted = qary_bp_magnetization(16, 3, 4, 0.85, init="planted", pop=5000, seed=0)
    assert abs(m - planted) < 0.05                       # θ_diff=1 ⇒ clean reconstruction


def test_overlap_non_decreasing_in_theta_diff():
    td = [0.2, 0.4, 0.6, 0.8, 1.0]
    m = [diffusion_class_overlap(14, 3, 4, 0.85, t, pop=4000, seed=1) for t in td]
    assert np.all(np.diff(m) >= -0.02)


def test_supercritical_broadcast_is_robust_to_leaf_corruption():
    # THE FINDING: a supercritical tree amplifies any positive leaf info, so heavy
    # leaf corruption barely moves the overlap — there is NO diffusion transition.
    tg = 0.85                                            # ≫ θ_KS = 1/√3 ≈ 0.577
    m_clean = diffusion_class_overlap(16, 3, 4, tg, 1.0, pop=5000, seed=2)
    m_noised = diffusion_class_overlap(16, 3, 4, tg, 0.4, pop=5000, seed=2)
    assert abs(m_clean - m_noised) < 0.05                # flat ⇒ robust, no transition


def test_better_generation_gives_higher_overlap():
    m = [diffusion_class_overlap(16, 3, 4, tg, 0.4, pop=5000, seed=3)
         for tg in (0.70, 0.85, 0.95)]
    assert m[0] < m[1] < m[2]                            # stronger tree ⇒ more class info
