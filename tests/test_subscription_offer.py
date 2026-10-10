"""Causal monthly RMST: independent finite-population identities and recovery."""
from itertools import product

import numpy as np
import pandas as pd
import pytest

from causal_bench.estimators.monthly_offer import (
    arm_pseudo_outcome, censoring_rows, estimate_offer,
)
from causal_bench.validation.subscription_offer import population_truth, simulate_offers


def enumerated_population():
    """Enumerate A,T,C exactly, not a Monte Carlo check of the same implementation."""
    h = np.array([[.22, .31, .18, .25], [.11, .16, .13, .17]])
    q = np.array([.17, .23])
    p = .65
    rows, probability = [], []
    for a, t, c in product(range(2), range(1, 6), range(1, 6)):
        mass_t = np.prod(1 - h[a, :t-1]) * (h[a, t-1] if t <= 4 else 1)
        mass_c = (1 - q[a]) ** (c - 1) * (q[a] if c <= 4 else 1)
        ct = c - .5 if c <= 4 else 4
        rows.append({"A": a, "observed_time": min(t, ct, 4), "event": int(t <= min(ct, 4))})
        probability.append((p if a else 1 - p) * mass_t * mass_c)
    return pd.DataFrame(rows), np.array(probability), h, q, p


@pytest.mark.parametrize("correct", ["all", "outcome", "weights"])
def test_exact_double_robust_identity_with_censoring(correct):
    df, probability, true_h, q, p = enumerated_population()
    assert probability.sum() == pytest.approx(1)
    for a in (0, 1):
        h = true_h[a] if correct != "weights" else np.array([.4, .07, .35, .2])
        g = (1 - q[a]) ** np.arange(1, 5) if correct != "outcome" else .9 ** np.arange(1, 5)
        propensity = p if correct != "outcome" else .4
        pseudo = arm_pseudo_outcome(df, np.tile(h, (len(df), 1)), np.tile(g, (len(df), 1)),
                                    np.full(len(df), propensity), a, 4)
        # Independent definition: sum min(t,H) * complete event-time PMF.
        truth = sum(min(t, 4) * np.prod(1 - true_h[a, :t-1])
                    * (true_h[a, t-1] if t <= 4 else 1) for t in range(1, 6))
        assert probability @ pseudo == pytest.approx(truth, abs=1e-12)


def test_two_wrong_nuisance_groups_need_not_be_unbiased():
    df, probability, *_ = enumerated_population()
    pseudo = arm_pseudo_outcome(df, np.full((len(df), 4), .4),
                                np.tile(.9 ** np.arange(1, 5), (len(df), 1)),
                                np.full(len(df), .4), 1, 4)
    true_h = np.array([.11, .16, .13, .17])
    truth = 1 + np.cumprod(1 - true_h)[:-1].sum()
    assert abs(probability @ pseudo - truth) > .05


def test_censor_completion_boundary_and_administrative_horizon():
    df = pd.DataFrame({"subscription_id": [0, 1, 2, 3], "cohort": 0,
                       "channel": 0, "plan": 1, "engaged": 1, "A": [0, 1, 0, 1],
                       "observed_time": [2.4, 2., 2., 4.], "event": [0, 1, 0, 0]})
    rows = censoring_rows(df, 4)
    assert rows.loc[rows.subscription_id == 0, "event"].tolist() == [0, 0, 1]
    assert rows.loc[rows.subscription_id == 1, "age"].tolist() == [1, 2]
    assert rows.loc[rows.subscription_id == 2, "event"].tolist() == [0, 0, 1]
    assert rows.loc[rows.subscription_id == 3, "event"].tolist() == [0, 0, 0, 0]


def test_horizon_one_is_exactly_one_active_month_in_both_arms():
    df, *_ = enumerated_population()
    p = arm_pseudo_outcome(df, np.full((len(df), 1), .2), np.full((len(df), 1), .8),
                          np.full(len(df), .5), 1, 1)
    np.testing.assert_array_equal(p, 1)


def test_shape_guard_rejects_broadcasting_and_nonmonotone_G():
    df, *_ = enumerated_population()
    with pytest.raises(ValueError, match="nuisance"):
        arm_pseudo_outcome(df, np.full((len(df), 4), .2), np.full((len(df), 4), .8),
                           np.full((len(df), 1), .5), 1, 4)
    with pytest.raises(ValueError, match="nuisance"):
        arm_pseudo_outcome(df, np.full((len(df), 4), .2), np.tile([.8, .9, .7, .6], (len(df), 1)),
                           np.full(len(df), .5), 1, 4)


def test_null_and_population_oracle_units():
    assert population_truth(effect=0)["effect_months"] == 0
    truth = population_truth()
    assert 0 < truth["effect_months"] < 12
    assert truth["arm_rmst_months"][1] - truth["arm_rmst_months"][0] == truth["effect_months"]


def test_nuisance_predictions_are_on_held_out_subscribers(monkeypatch):
    import causal_bench.estimators.monthly_offer as module
    original = module.fit_monthly_hazard
    calls = []

    def wrapped(rows, spec):
        trained = set(rows["subscription_id"])
        fitted = original(rows, spec)

        class Guard:
            def predict_survival(self, check):
                assert not trained.intersection(check["subscription_id"])
                calls.append(len(check))
                return fitted.predict_survival(check)
        return Guard()

    monkeypatch.setattr(module, "fit_monthly_hazard", wrapped)
    df, _ = simulate_offers(1600, seed=7)
    estimate_offer(df, folds=2, seed=0)
    assert len(calls) == 8  # two folds, two nuisance fits, two interventions


@pytest.mark.parametrize("assignment,effect", [("confounded", -.55), ("randomized", 0.)])
def test_estimated_nuisance_causal_recovery(assignment, effect):
    df, truth = simulate_offers(12000, seed=11, assignment=assignment, effect=effect)
    estimate = estimate_offer(df, seed=20, randomized_probability=.5 if assignment == "randomized" else None)
    assert abs(estimate["estimate_months"] - truth["population"]["effect_months"]) < 3 * estimate["standard_error"]
    assert estimate["diagnostics"]["propensity_clipped"] == 0
    assert estimate["diagnostics"]["censor_survival_clipped"] == 0


def test_rejects_delayed_entry_and_nonbinary_assignment():
    df, _ = simulate_offers(100)
    with pytest.raises(ValueError, match="delayed entry"):
        estimate_offer(df.assign(entry_age=np.minimum(1, np.floor(df.observed_time))))
    with pytest.raises(ValueError, match="Binary"):
        estimate_offer(df.assign(A=2))


def test_strict_concrete_horizon_and_option_forwarding(monkeypatch):
    import causal_bench.estimators.concrete_rmst as module
    df = pd.DataFrame({"Delta": [1, 0, 1], "T_obs": [2., 12., 15.]})
    with pytest.raises(ValueError, match="horizon_policy"):
        module.ConcreteRMSTEstimator(horizon_policy="guess")
    est = module.ConcreteRMSTEstimator(rmst_contrast=True, horizon_policy="strict")
    with pytest.raises(ValueError, match="no silent capping"):
        est.estimate(df, horizon=15.)
    monkeypatch.setattr(module, "_concrete_available", lambda: True)
    seen = {}

    def bridge(kw, timeout):
        seen.update(kw)
        return .6, .1

    monkeypatch.setattr(module, "_run_concrete_isolated", bridge)
    result = est.estimate(df, horizon=12.)[0]
    assert seen["horizon_policy"] == "strict" and seen["rmst_contrast"] is True
    assert result.point_estimate == .6
    assert result.ci_lower == pytest.approx(.404)


def test_oracle_columns_never_enter_observed_data():
    observed, truth = simulate_offers(100)
    assert not {"event_time", "hazard", "survival", "propensity", "population"}.intersection(observed.columns)
    assert "population" in truth


def test_deliberate_misspecification_does_not_claim_valid_if_intervals():
    df, _ = simulate_offers(1600, seed=7)
    result = estimate_offer(df, folds=2, propensity_adjusted=False)
    assert result["ci95"] is None and result["standard_error"] is None
