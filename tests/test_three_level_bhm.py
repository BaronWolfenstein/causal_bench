"""Tests for the two-vs-three-level OC fidelity harness (#40).

The conjugate **two-level** kernel ignores subgroup heterogeneity (τ); the
**three-level** BHM models it via MCMC. This suite pins: the three-level fit
recovers the effect and reports MCMC diagnostics (R-hat / bulk-ESS / tail-ESS),
the tail-ESS gate flags weak fits, the two-level kernel is over-confident when
τ>0 (smaller SE), and the fidelity harness produces both kernels' OC + deltas.

MCMC-backed tests skip without pymc (the 3.12 `[bayes]` stack). Cheap draws keep
the suite fast; assertions are structural + directional, not precise OC values
(those need the 200–500-replicate real run).
"""
import numpy as np
import pytest

pytest.importorskip("pymc")

from causal_bench.estimators.three_level_bhm import (
    simulate_three_level, fit_two_level_conjugate, fit_three_level_bhm,
    tail_ess_ok, run_fidelity,
)

# a fast MCMC config for tests
FAST = dict(draws=300, tune=300, chains=2, seed=0)


def test_simulate_has_subgroup_structure():
    d = simulate_three_level(n_subgroups=8, n_per=30, true_effect=0.5, tau=0.4, seed=1)
    assert d["y"].shape == (8 * 30,)
    assert set(np.unique(d["subgroup"])) == set(range(8))
    assert d["true_effect"] == 0.5


def test_three_level_recovers_effect_and_reports_diagnostics():
    d = simulate_three_level(n_subgroups=10, n_per=40, true_effect=0.6, tau=0.3, seed=2)
    fit = fit_three_level_bhm(d, **FAST)
    assert abs(fit["effect"] - 0.6) < 0.25                 # posterior mean ~ truth
    for key in ("r_hat", "bulk_ess", "tail_ess", "se", "ci_lo", "ci_hi", "rejects_null"):
        assert key in fit
    assert np.isfinite(fit["tail_ess"]) and fit["tail_ess"] > 0
    assert fit["r_hat"] < 1.1                               # converged at these draws


def test_two_level_conjugate_is_overconfident_under_heterogeneity():
    # ignoring τ underestimates the SE — the anti-conservative approximation #40
    # exists to quantify. Same data, two-level SE < three-level SE.
    d = simulate_three_level(n_subgroups=12, n_per=30, true_effect=0.5, tau=0.6, seed=3)
    two = fit_two_level_conjugate(d)
    three = fit_three_level_bhm(d, **FAST)
    assert abs(two["effect"] - 0.5) < 0.3                   # both roughly unbiased for the mean
    assert two["se"] < three["se"]                          # but two-level is over-confident


def test_tail_ess_gate():
    good = {"tail_ess": 400.0}
    weak = {"tail_ess": 40.0}
    assert tail_ess_ok(good, threshold=100.0) is True
    assert tail_ess_ok(weak, threshold=100.0) is False


def test_fidelity_harness_reports_both_kernels_and_deltas():
    res = run_fidelity(n_reps=4, tau=0.5, effect_alt=0.6, tail_ess_threshold=100.0,
                       **FAST)
    for kernel in ("two_level", "three_level"):
        for q in ("type_i", "power", "coverage"):
            assert 0.0 <= res[kernel][q] <= 1.0
    for q in ("type_i", "power", "coverage"):
        assert res["delta"][q] == pytest.approx(
            res["two_level"][q] - res["three_level"][q], abs=1e-9)
    assert "n_mcmc_fits" in res and res["n_mcmc_fits"] == 4 * 2   # null + alt reps
    assert "n_tail_ess_flagged" in res


# ---- three-level meta-analysis over subgroup summaries + exp19 real-DGP wiring ----
from causal_bench.estimators.three_level_bhm import fit_three_level_meta


def test_three_level_meta_recovers_effect_from_summaries():
    rng = np.random.default_rng(0)
    mu_true, tau_true = -0.12, 0.15
    theta = rng.normal(mu_true, tau_true, 8)
    se = np.full(8, 0.10)
    theta_hat = theta + rng.normal(0, se)
    fit = fit_three_level_meta(theta_hat, se, true_effect=mu_true, **FAST)
    assert abs(fit["effect"] - mu_true) < 0.15                 # μ recovered from summaries
    for k in ("r_hat", "bulk_ess", "tail_ess", "se", "rejects_null", "covers_truth"):
        assert k in fit
    assert fit["r_hat"] < 1.1


def test_fit_meta_lognormal_tau_prior_runs_and_recovers():
    # the empirical (van Zwet) policy needs a LogNormal τ prior, not a HalfNormal
    # scale. Passing tau_prior=("lognormal", (mu_log, sigma_log)) must build that
    # prior and still recover μ.
    rng = np.random.default_rng(1)
    mu_true, tau_true = -0.10, 0.12
    theta = rng.normal(mu_true, tau_true, 8)
    se = np.full(8, 0.10)
    fit = fit_three_level_meta(theta + rng.normal(0, se), se, true_effect=mu_true,
                               tau_prior=("lognormal", (-1.82, 0.90)), **FAST)
    assert abs(fit["effect"] - mu_true) < 0.15
    assert np.isfinite(fit["tail_ess"]) and fit["r_hat"] < 1.1


def test_fit_meta_halfnormal_tau_prior_equals_tau_sd():
    # back-compat: tau_prior=("halfnormal", (s,)) builds the SAME model as the
    # legacy tau_sd=s, so with a shared seed the decision is identical.
    rng = np.random.default_rng(2)
    theta = rng.normal(0.0, 0.2, 6)
    se = np.full(6, 0.12)
    theta_hat = theta + rng.normal(0, se)
    a = fit_three_level_meta(theta_hat, se, tau_sd=0.3, **FAST)
    b = fit_three_level_meta(theta_hat, se, tau_prior=("halfnormal", (0.3,)), **FAST)
    assert a["effect"] == pytest.approx(b["effect"], abs=1e-9)
    assert a["se"] == pytest.approx(b["se"], abs=1e-9)


def test_fit_meta_unknown_tau_prior_family_raises():
    with pytest.raises(ValueError):
        fit_three_level_meta(np.zeros(4), np.ones(4), tau_prior=("cauchy", (0.5,)), **FAST)


def test_exp19_bridge_extracts_summaries_and_both_kernels():
    # the real registry DGP → 4 subgroup summaries + exp19's ESS-weighted two-level.
    from experiments.exp36_three_level_fidelity import exp19_subgroup_summaries
    s = exp19_subgroup_summaries(phi=0.7, conflict=0.0, scenario="alternative", seed=1)
    assert s["theta_hat"].shape == s["se"].shape and len(s["theta_hat"]) == 4
    assert np.isfinite(s["theta_hat"]).all() and (s["se"] > 0).all()
    assert hasattr(s["two_level"], "ate_posterior")            # exp19 BorrowingResult
    # the three-level meta runs on those real summaries and reports diagnostics
    three = fit_three_level_meta(s["theta_hat"], s["se"], true_effect=s["true_ate"], **FAST)
    assert np.isfinite(three["tail_ess"]) and three["tail_ess"] > 0
