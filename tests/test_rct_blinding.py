"""RCT-blinding validation for the OC-sim / synthetic comparator (#139). The guard must
recover the RCT (unconfounded) effect from a properly-adjusted OC-sim fit on confounded
observational data, and FLAG a naive or unmeasured-confounded comparator that can't."""
import numpy as np

from causal_bench.validation.rct_blinding import (
    make_confounded_cohorts, fit_oc_sim, oc_sim_rmst, rct_blinding_recovery,
)


def test_cohorts_rct_randomized_observational_confounded():
    coh = make_confounded_cohorts(n_obs=5000, n_rct=5000, confound=3.0, seed=0)
    obs, rct = coh["obs"], coh["rct"]
    # RCT: treatment ~ 0.5 and ⊥ prognosis; observational: strongly selected on prognosis
    assert abs(rct["A"].mean() - 0.5) < 0.05
    beta = coh["beta"]                                         # true prognostic direction
    g_obs = (obs["X"] * beta).sum(1)
    g_rct = (rct["X"] * beta).sum(1)
    assert abs(np.corrcoef(rct["A"], g_rct)[0, 1]) < 0.05      # randomized
    assert abs(np.corrcoef(obs["A"], g_obs)[0, 1]) > 0.3       # strongly confounded
    assert coh["true_ate"] > 0.1                               # a real effect exists


def test_adjusted_oc_sim_recovers_the_rct_effect():
    for sd in range(3):
        r = rct_blinding_recovery(confound=3.0, adjust=True, seed=sd)
        assert abs(r["ate_bias_vs_rct"]) < 0.05                # recovers held-out RCT
        assert r["recovered"] is True


def test_naive_oc_sim_is_flagged_under_confounding():
    # THE guard: a confounded (naive) synthetic comparator must NOT silently pass.
    for sd in range(3):
        r = rct_blinding_recovery(confound=3.0, adjust=False, seed=sd)
        assert abs(r["ate_bias_vs_rct"]) > 0.2                 # grossly biased
        assert r["recovered"] is False                         # flagged


def test_unmeasured_confounding_defeats_even_the_adjusted_oc_sim():
    # Adjustment can only fix MEASURED confounding; the guard still catches residual bias.
    r = rct_blinding_recovery(confound=3.0, adjust=True, unmeasured=1.5, seed=0)
    assert abs(r["ate_bias_vs_rct"]) > 0.2
    assert r["recovered"] is False


def test_rct_is_a_valid_ground_truth_and_control_branch_recovers():
    # The held-out RCT (scored against, never the obs fit) tracks the analytic truth,
    # and the adjusted OC-sim's control branch matches the RCT control curve.
    r = rct_blinding_recovery(confound=3.0, adjust=True, seed=1)
    assert abs(r["ate_rct"] - r["ate_true"]) < 0.05
    assert abs(r["rmst_control_ocsim"] - r["rmst_control_rct"]) < 0.05
    assert r["control_curve_l1"] < 0.05


def test_g_computation_rmst_is_bounded_by_horizon():
    coh = make_confounded_cohorts(n_obs=2000, n_rct=2000, horizon=2.0, seed=0)
    model = fit_oc_sim(coh["obs"]["X"], coh["obs"]["A"], coh["obs"]["T"], adjust=True)
    rmst = oc_sim_rmst(model, coh["obs"]["X"], 0.0, 2.0, seed=0)
    assert 0.0 < rmst <= 2.0
