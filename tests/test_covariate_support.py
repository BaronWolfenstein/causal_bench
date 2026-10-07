"""Covariate-support (positivity-in-W) diagnostic — the regime-B complement to the weight-dispersion ESS.

exp55 regime B (#224): a near-deterministic targeting rule segregates the treated/control covariate
distributions, so the DR contrast is badly biased (outcome model extrapolates) while the weight-dispersion
Kish ESS stays HEALTHY. `covariate_support` must flag that support gap where `ate_overlap_ess` cannot."""
import numpy as np

from causal_bench.validation.uplift_policy import (
    sim_uplift, dr_learner_cate, aipw_policy_value, ate_overlap_ess, covariate_support, true_values,
)

_TRUTH = (lambda tv: tv["v_all"] - tv["v_none"])(true_values(curv=0.7))   # treat-all − none, curv-misspecified


def _support_metrics(d, seed):
    _, nuis = dr_learner_cate(d, seed=seed)
    W, A, Y, e, _, _ = nuis
    Wmat = np.column_stack([d["W1"], d["W2"]])
    return ate_overlap_ess(A, e)[1], covariate_support(Wmat, A, e)


def _dr_bias(d, seed):
    _, nuis = dr_learner_cate(d, seed=seed); n = len(d["A"])
    contrast = aipw_policy_value(nuis, np.ones(n, int)) - aipw_policy_value(nuis, np.zeros(n, int))
    return abs(contrast - _TRUTH)


def test_good_overlap_reads_healthy():
    # smooth, mild confounding: good covariate support ⇒ high overlap, ~no extrapolation, no false alarm
    _, sup = _support_metrics(sim_uplift(4000, gamma=0.3, curv=0.7, seed=0), seed=0)
    assert sup["overlap_coef"] > 0.7
    assert sup["common_support"] > 0.9
    assert sup["extrapolation"] < 0.1


def test_support_gap_flagged_where_ess_is_blind():
    # near-deterministic rule with a steep cut: the DR bias is CATASTROPHIC while the weight ESS does NOT
    # collapse to match (the false-green trap) — and the covariate-support check fires strongly.
    d = sim_uplift(4000, threshold=2.2, curv=0.7, seed=0)
    ess_frac, sup = _support_metrics(d, seed=0)
    bias = _dr_bias(d, seed=0)
    assert bias > 1.0                          # catastrophic bias (outcome model extrapolating)
    assert ess_frac > 0.3                      # ...yet the weight ESS is NOT collapsed — disproportionate = blind
    assert sup["overlap_coef"] < 0.2           # the covariate supports barely overlap
    assert sup["common_support"] < 0.3
    assert sup["extrapolation"] > 0.8          # most units are in pure-extrapolation regions (the real signal)


def test_extrapolation_monotone_in_support_gap():
    # as the threshold steepens the support gap worsens monotonically (the signal the ESS misses)
    ex = [covariate_support(np.column_stack([(d := sim_uplift(4000, threshold=t, curv=0.7, seed=1))["W1"], d["W2"]]),
                            d["A"])["extrapolation"] for t in (0.0, 1.5, 2.2)]
    assert ex[0] < ex[1] < ex[2]


def test_1d_covariate_accepted():
    d = sim_uplift(1000, gamma=1.0, seed=2)
    out = covariate_support(d["W1"], d["A"])           # 1-D W must be accepted
    assert set(out) == {"overlap_coef", "common_support", "knn_cross_frac", "extrapolation"}


if __name__ == "__main__":
    import pytest, sys
    sys.exit(pytest.main([__file__, "-v"]))
