"""Borrowing-calibration fidelity engine (exp41 / #144 step C). Pure-numpy helpers run
anywhere; the end-to-end engine test needs the 3.12 [bayes] (PyMC) stack and skips
without it."""
import numpy as np
import pytest

from causal_bench.validation.joint_fidelity import (
    population_effect, make_null_spec, _subgroup_estimates, _policy_tau_sd, joint_fidelity,
)
from causal_bench.dgp.joint_hierarchy import make_joint_hierarchy, decode_cohort_labels, sample_joint_cohort


def test_population_effect_matches_weighted_table_means():
    spec = make_joint_hierarchy(4, 3, 2, 2, w_group=1.5, w_member=0.3, seed=0)
    expected = 1.5 * spec["group_effect"].mean() + 0.3 * spec["member_effect"].mean()
    assert abs(population_effect(spec) - expected) < 1e-12


def test_make_null_spec_has_zero_mu_and_target_tau():
    # heterogeneous null: μ = 0 at the level, between-subgroup SD = tau_scale
    spec = make_null_spec(4, 3, 2, 2, level="group", tau_scale=0.4, seed=0)
    assert abs(population_effect(spec)) < 1e-9                  # centered ⇒ μ = 0
    from causal_bench.dgp.joint_hierarchy import true_tau_by_level
    assert abs(true_tau_by_level(spec)["tau_group"] - 0.4) < 1e-9
    # global null: tau_scale=0 ⇒ μ = τ = 0
    g0 = make_null_spec(4, 3, 2, 2, level="group", tau_scale=0.0, seed=0)
    assert true_tau_by_level(g0)["tau_group"] == 0.0


def test_subgroup_estimates_recover_mean_difference_and_drop_sparse():
    Y = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 5.0])
    A = np.array([1, 1, 1, 0, 0, 0, 1])
    sub = np.array([0, 0, 0, 0, 0, 0, 1])                       # subgroup 1 has 1 unit (dropped)
    th, se = _subgroup_estimates(Y, A, sub, n_sub=2, min_per_arm=3)
    assert len(th) == 1 and abs(th[0] - 1.0) < 1e-9            # only subgroup 0 survives, effect=1


def test_policy_tau_sd_flat_oracle_canonical():
    spec = make_joint_hierarchy(4, 3, 2, 2, w_group=1.5, w_member=0.3, seed=0)
    dec = {"group_decode_acc": 0.92, "member_decode_acc": 0.75}
    flat = _policy_tau_sd("flat", "group", spec, dec, flat_tau_sd=0.5, tau_sd_min=0.05, tau_sd_max=1.0)
    orc = _policy_tau_sd("oracle", "group", spec, dec, flat_tau_sd=0.5, tau_sd_min=0.05, tau_sd_max=1.0)
    can = _policy_tau_sd("canonical", "group", spec, dec, flat_tau_sd=0.5, tau_sd_min=0.05, tau_sd_max=1.0)
    assert flat == 0.5
    from causal_bench.dgp.joint_hierarchy import true_tau_by_level
    assert abs(orc - true_tau_by_level(spec)["tau_group"]) < 1e-9
    assert 0.05 <= can <= 1.0 and can > 0.5                     # high group acc ⇒ weak pooling


def test_joint_fidelity_runs_and_global_null_is_not_inflated():
    pytest.importorskip("pymc")
    spec = make_null_spec(4, 3, 2, 2, level="group", tau_scale=0.0, seed=0)   # μ=τ=0
    r = joint_fidelity(spec, level="group", policy="canonical", theta0=0.7,
                       n_reps=4, n_units=2500, draws=200, tune=200, seed=1)
    assert r["mu_true"] == 0.0
    assert r["n_used"] >= 1
    # a global-null Type-I should not be grossly inflated (loose bound for a tiny run)
    assert np.isnan(r["reject_rate"]) or r["reject_rate"] <= 0.5


def test_make_scenario_spec_sets_mu_and_tau():
    from causal_bench.validation.joint_fidelity import make_scenario_spec
    from causal_bench.dgp.joint_hierarchy import true_tau_by_level
    spec = make_scenario_spec(4, 3, 2, 2, level="member", mu=0.5, tau=0.3, seed=0)
    assert abs(population_effect(spec) - 0.5) < 1e-9            # μ = mu
    assert abs(true_tau_by_level(spec)["tau_member"] - 0.3) < 1e-9   # τ = tau
    assert true_tau_by_level(spec)["tau_group"] == 0.0         # other level carries none
