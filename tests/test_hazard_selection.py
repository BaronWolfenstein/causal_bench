"""Built-in selection bias of the hazard ratio (depletion of susceptibles).

Treatment is RANDOMIZED and the conditional (given frailty U) hazard ratio is constant
by construction, so there is no confounding of any kind. The Cox HR is nevertheless
biased, because the hazard at t conditions on survival to t and survival is a collider
between A and U (A -> S(t) <- U -> event). Cumulative-risk estimands (risk difference,
RMST) do not condition on survival and stay unbiased on the SAME replicates.
"""
import numpy as np
import pytest

from causal_bench.validation.hazard_selection import (
    simulate_frailty_survival, marginal_survival, marginal_hazard_ratio,
    true_risk_difference, true_rmst_difference,
    cox_hazard_ratio, km_risk_difference, km_rmst_difference, windowed_hazard_ratios,
)

PROTECTIVE = -0.7                                   # conditional log-HR (exp = 0.497)
NO_FRAILTY = 1e6                                    # Gamma(k,k) with k huge -> U ~ 1


def test_marginal_hr_starts_at_conditional_and_attenuates_toward_one():
    kw = dict(log_hr=PROTECTIVE, frailty_shape=1.0, baseline_rate=1.0)
    hr0 = marginal_hazard_ratio(0.0, **kw)
    assert hr0 == pytest.approx(np.exp(PROTECTIVE), rel=1e-9)   # t=0: no depletion yet
    ts = np.array([0.0, 0.5, 1.0, 2.0, 5.0, 50.0])
    hrs = np.array([marginal_hazard_ratio(t, **kw) for t in ts])
    assert np.all(np.diff(hrs) > 0)                             # monotone drift toward 1
    assert hrs[-1] > 0.9                                        # and it really does approach 1
    assert np.all(hrs < 1.0)                                    # never overshoots


def test_without_frailty_the_marginal_hr_is_flat():
    # U degenerate at 1 -> no depletion effect -> marginal HR == conditional HR for all t.
    kw = dict(log_hr=PROTECTIVE, frailty_shape=NO_FRAILTY, baseline_rate=1.0)
    for t in (0.0, 1.0, 5.0):
        assert marginal_hazard_ratio(t, **kw) == pytest.approx(np.exp(PROTECTIVE), rel=1e-3)


def test_marginal_survival_matches_simulation():
    d = simulate_frailty_survival(60000, log_hr=PROTECTIVE, frailty_shape=1.0,
                                  horizon=1.5, seed=0)
    for a in (0, 1):
        m = d["A"] == a
        emp = np.mean(d["T_event"][m] > 1.0)                    # uncensored event times
        ana = marginal_survival(1.0, a, log_hr=PROTECTIVE, frailty_shape=1.0)
        assert abs(emp - ana) < 0.01


def test_cox_recovers_the_conditional_hr_when_there_is_no_frailty():
    d = simulate_frailty_survival(20000, log_hr=PROTECTIVE, frailty_shape=NO_FRAILTY,
                                  horizon=2.0, seed=1)
    hr = cox_hazard_ratio(d["A"], d["T_obs"], d["Delta"])
    assert hr == pytest.approx(np.exp(PROTECTIVE), rel=0.06)


def test_cox_hr_is_attenuated_under_frailty_despite_randomization():
    # THE demonstration: randomized A, constant conditional HR, yet Cox is biased toward
    # the null purely from conditioning on survival.
    d = simulate_frailty_survival(20000, log_hr=PROTECTIVE, frailty_shape=1.0,
                                  horizon=2.0, seed=2)
    hr = cox_hazard_ratio(d["A"], d["T_obs"], d["Delta"])
    assert np.exp(PROTECTIVE) < hr < 1.0                        # strictly between, attenuated
    assert hr > np.exp(PROTECTIVE) + 0.05                       # a materially large bias


def test_cumulative_risk_estimands_stay_unbiased_under_the_same_frailty():
    d = simulate_frailty_survival(20000, log_hr=PROTECTIVE, frailty_shape=1.0,
                                  horizon=2.0, seed=2)
    rd = km_risk_difference(d["A"], d["T_obs"], d["Delta"], horizon=1.0)
    rmst = km_rmst_difference(d["A"], d["T_obs"], d["Delta"], horizon=1.0)
    rd_true = true_risk_difference(1.0, log_hr=PROTECTIVE, frailty_shape=1.0)
    rmst_true = true_rmst_difference(1.0, log_hr=PROTECTIVE, frailty_shape=1.0)
    assert abs(rd - rd_true) < 0.02                             # unbiased for the marginal truth
    assert abs(rmst - rmst_true) < 0.02


def test_late_window_hr_is_more_attenuated_than_early_window():
    # Depletion accumulates: among survivors past t_split the arms' U-distributions have
    # diverged further, so the conditional-on-survival contrast is closer to null.
    d = simulate_frailty_survival(40000, log_hr=PROTECTIVE, frailty_shape=1.0,
                                  horizon=4.0, seed=3)
    early, late = windowed_hazard_ratios(d["A"], d["T_obs"], d["Delta"], t_split=1.0)
    assert early < late                                          # later window closer to 1
    assert early == pytest.approx(np.exp(PROTECTIVE), abs=0.12)  # early ~ conditional HR
