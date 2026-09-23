"""Kish-ESS overlap diagnostic for the uplift targeting policy value (exp52). Pure Python.

policy_value_ess is the off-policy-EVALUATION analogue of the PPO reuse gate ESS(pi_theta/pi_old): it measures
how usable the logged batch is for a *different* target policy pi. These tests pin its bounds and the overlap
ordering (a policy that leans on low-propensity units has a smaller ESS)."""
import numpy as np

from causal_bench.validation.uplift_policy import (
    sim_uplift, dr_learner_cate, policy_value_ess, qini,
)


def _nuis(n=3000, seed=0):
    d = sim_uplift(n, seed=seed)
    tau_hat, nuis = dr_learner_cate(d, folds=2, seed=seed)
    return d, tau_hat, nuis


class TestPolicyValueESS:
    def test_bounds(self):
        d, _, nuis = _nuis()
        n = len(d["A"])
        for pi in (np.zeros(n, int), np.ones(n, int), (d["W1"] > 0).astype(int)):
            ess, essf = policy_value_ess(nuis, pi)
            assert np.isfinite(ess) and 0.0 < ess <= n + 1e-6
            assert 0.0 < essf <= 1.0 + 1e-9

    def test_deterministic(self):
        d, _, nuis = _nuis()
        pi = (d["W1"] > 0).astype(int)
        assert policy_value_ess(nuis, pi) == policy_value_ess(nuis, pi)

    def test_overlap_ordering(self):
        # e(W)=expit(0.8*W1): treating where W1>0 leans on HIGH-propensity treated units (small 1/e weights →
        # good ESS); treating where W1<0 forces LOW-propensity treated units (large 1/e weights → worse ESS).
        d, _, nuis = _nuis(n=6000)
        pi_good = (d["W1"] > 0).astype(int)
        pi_bad = (d["W1"] < 0).astype(int)
        _, essf_good = policy_value_ess(nuis, pi_good)
        _, essf_bad = policy_value_ess(nuis, pi_bad)
        assert essf_good > essf_bad


class TestQiniESS:
    def test_qini_reports_ess(self):
        d, tau_hat, nuis = _nuis()
        q = qini(tau_hat, nuis, d)
        ess = q["ess_frac_dr"]
        assert ess.shape == q["ks"].shape
        assert np.all(ess > 0.0) and np.all(ess <= 1.0 + 1e-9)
        assert q["min_ess_frac_dr"] == float(ess.min())
