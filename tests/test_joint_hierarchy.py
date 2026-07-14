"""Joint hierarchical DGP — prerequisite A for #144. Couples canonical rule-BP
identifiability (representation side) with per-level effect heterogeneity (effect side)
on one product-grammar object, with a coupling knob (w_group vs w_member)."""
import numpy as np

from causal_bench.dgp.joint_hierarchy import (
    make_joint_hierarchy, sample_joint_cohort, true_tau_by_level, joint_reconstruction_scan,
)


def test_spec_structure_and_effect_tables():
    spec = make_joint_hierarchy(4, 3, 2, 2, seed=0)
    assert spec["rules"].shape == (12, 2, 2)                    # product grammar over v=12
    assert (spec["groups"] == np.arange(12) // 3).all()
    assert spec["group_effect"].shape == (4,) and spec["member_effect"].shape == (3,)


def test_true_tau_tracks_the_coupling_knob():
    # tau at each level is set by its weight, NOT by identifiability (the #144 crux).
    coarse = make_joint_hierarchy(4, 3, 2, 2, w_group=2.0, w_member=0.2, seed=0)
    fine = make_joint_hierarchy(4, 3, 2, 2, w_group=0.2, w_member=2.0, seed=0)
    tc, tf = true_tau_by_level(coarse), true_tau_by_level(fine)
    assert tc["tau_group"] > tc["tau_member"]                  # effect at coarse level
    assert tf["tau_member"] > tf["tau_group"]                  # effect at fine level


def test_cohort_is_coherent_and_effect_variance_matches_tau():
    spec = make_joint_hierarchy(4, 3, 2, 2, w_group=1.5, w_member=0.3, seed=0)
    coh = sample_joint_cohort(spec, 4000, depth=6, seed=1)
    assert len(coh["Y"]) == 4000 and np.isfinite(coh["Y"]).all()
    assert set(np.unique(coh["group"])) <= set(range(4))
    assert abs(coh["A"].mean() - 0.5) < 0.05                    # randomized treatment
    # empirical between-group effect SD ≈ the true tau_group
    grp_means = [coh["effect"][coh["group"] == k].mean() for k in np.unique(coh["group"])]
    assert abs(np.std(grp_means) - true_tau_by_level(spec)["tau_group"]) < 0.15


def test_canonical_rule_bp_coarse_survives_more_corruption_than_fine():
    # THE point: identifiability measured by EXACT rule-BP (not the embedding crossover).
    # The coarse (group) coordinate is recoverable under more corruption than the fine
    # (member) one → its half-clean threshold is lower.
    spec = make_joint_hierarchy(4, 3, 2, 2, seed=0)
    scan = joint_reconstruction_scan(spec, depth=7, n_trees=200, seed=2)
    assert scan["theta_c_group"] <= scan["theta_c_member"]     # coarse ≥ robust
    assert scan["group_overlap"][-1] - scan["group_overlap"][0] > 0.5   # a real transition
    assert scan["member_overlap"][-1] - scan["member_overlap"][0] > 0.5


def test_reconstruction_overlaps_rise_with_retention():
    spec = make_joint_hierarchy(4, 3, 2, 2, seed=0)
    scan = joint_reconstruction_scan(spec, depth=6, n_trees=200, seed=3)
    assert np.all(np.diff(scan["group_overlap"]) > -0.05)      # ~monotone in theta
    assert np.all(np.diff(scan["member_overlap"]) > -0.05)


# ─── BP-decoded labels at θ₀ (the #144 label-observation model) ───────────────
def test_decode_labels_near_perfect_at_theta0_one():
    from causal_bench.dgp.joint_hierarchy import decode_cohort_labels
    spec = make_joint_hierarchy(4, 3, 2, 2, seed=0)
    coh = sample_joint_cohort(spec, 1500, depth=7, seed=1)
    d = decode_cohort_labels(spec, coh, theta0=1.0, seed=2)
    assert d["group_decode_acc"] > 0.85 and d["member_decode_acc"] > 0.85
    assert set(np.unique(d["group_decoded"])) <= set(range(4))
    assert set(np.unique(d["member_decoded"])) <= set(range(3))


def test_decode_accuracy_declines_with_corruption():
    from causal_bench.dgp.joint_hierarchy import decode_cohort_labels
    spec = make_joint_hierarchy(4, 3, 2, 2, seed=0)
    coh = sample_joint_cohort(spec, 1500, depth=7, seed=1)
    hi = decode_cohort_labels(spec, coh, theta0=0.9, seed=2)
    lo = decode_cohort_labels(spec, coh, theta0=0.5, seed=2)
    assert hi["group_decode_acc"] > lo["group_decode_acc"]      # less corruption ⇒ better


def test_coarse_coordinate_is_more_learnable_than_fine():
    # In the informative regime, the coarse (group) coordinate is decoded at least as
    # accurately as the fine (member) one — the learnability ordering that makes
    # identifiability bite on the fine-level effect (the #144 license).
    from causal_bench.dgp.joint_hierarchy import decode_cohort_labels
    spec = make_joint_hierarchy(4, 3, 2, 2, seed=0)
    coh = sample_joint_cohort(spec, 2000, depth=7, seed=1)
    d = decode_cohort_labels(spec, coh, theta0=0.7, seed=2)
    assert d["group_decode_acc"] >= d["member_decode_acc"] - 0.03
    assert d["group_decode_acc"] > 1.0 / 4 and d["member_decode_acc"] > 1.0 / 3  # above chance
