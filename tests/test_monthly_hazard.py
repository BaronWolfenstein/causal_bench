"""Independent likelihood identities, censoring boundaries and simulation truth."""
import numpy as np
import pandas as pd
import pytest
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from causal_bench.estimators.monthly_hazard import (
    HazardSpec, binomial_mle, design, fit_monthly_hazard, person_months,
)
from causal_bench.validation.subscription_retention import (
    cohort_split, evaluate, run, simulate_subscriptions,
)


def test_censoring_boundary_and_delayed_entry():
    df = pd.DataFrame({"subscription_id": [0, 1, 2, 3, 4], "cohort": [0]*5,
                       "observed_time": [2.4, 3, 2, 0.3, 4], "event": [0, 1, 0, 0, 1],
                       "entry_age": [0, 0, 0, 0, 2]})
    rows = person_months(df, 6)
    assert rows.groupby("subscription_id")["age"].apply(list).to_dict() == {
        0: [1, 2], 1: [1, 2, 3], 2: [1, 2], 4: [3, 4]}
    assert rows.loc[rows.event == 1, ["subscription_id", "age"]].values.tolist() == [[1, 3], [4, 4]]
    assert (rows.calendar_month == rows.cohort + rows.age - 1).all()


@pytest.mark.parametrize("column,value", [("event", 2), ("observed_time", -1),
                                          ("observed_time", 2.5), ("cohort", 1.3),
                                          ("entry_age", 3), ("cohort", np.nan)])
def test_bad_subscription_metadata_rejected(column, value):
    df = pd.DataFrame({"subscription_id": [0], "cohort": [0], "observed_time": [2],
                       "event": [1], "entry_age": [0]})
    df[column] = value
    with pytest.raises(ValueError):
        person_months(df, 12)


def test_duplicate_subscription_ids_rejected():
    df, _ = simulate_subscriptions(20)
    df.loc[1, "subscription_id"] = df.loc[0, "subscription_id"]
    with pytest.raises(ValueError, match="One row"):
        person_months(df, 12)


def test_binomial_mle_matches_analytic_saturated_cells():
    n = np.array([100, 200, 50])
    events = np.array([20, 130, 5])
    p = events / n
    beta, cov, ll, _ = binomial_mle(np.eye(3), events, n)
    np.testing.assert_allclose(beta, np.log(p / (1 - p)), atol=1e-8)
    np.testing.assert_allclose(cov, np.diag(1 / (n * p * (1-p))), atol=1e-8)
    assert ll == pytest.approx(np.sum(events*np.log(p) + (n-events)*np.log1p(-p)))


@pytest.mark.parametrize("kind", ["rank", "separation", "nonfinite", "counts", "shape"])
def test_invalid_or_unidentifiable_binomial_fits_fail(kind):
    x, y, n = np.eye(2), np.array([2., 3.]), np.array([10., 10.])
    if kind == "rank":
        x = np.ones((2, 2))
    elif kind == "separation":
        y = np.zeros(2)
    elif kind == "nonfinite":
        x[0, 0] = np.nan
    elif kind == "counts":
        y[0] = 11
    else:
        y = y[:, None]
    with pytest.raises(ValueError):
        binomial_mle(x, y, n)


def test_row_and_cell_fits_match_independent_logistic_reference():
    df, _ = simulate_subscriptions(6000, seed=12)
    rows = person_months(df, 12)
    spec = HazardSpec(age_interactions=("plan",))
    cell = fit_monthly_hazard(rows, spec)
    row = fit_monthly_hazard(rows, spec, aggregate=False)
    np.testing.assert_allclose(cell.coefficients, row.coefficients, atol=1e-8)
    np.testing.assert_allclose(cell.covariance, row.covariance, atol=1e-8)
    assert cell.log_likelihood == pytest.approx(row.log_likelihood, abs=1e-7)
    assert cell.fit_rows < row.fit_rows / 10
    x, _ = design(rows, spec)
    # Separate sklearn solver, no penalty/intercept, same full age indicator design.
    reference = LogisticRegression(penalty=None, fit_intercept=False, solver="newton-cg",
                                   tol=1e-9, max_iter=200).fit(x, rows.event)
    np.testing.assert_allclose(cell.coefficients, reference.coef_[0], atol=2e-5)


def test_aggregation_does_not_hide_nan_predictor():
    df, _ = simulate_subscriptions(1000)
    rows = person_months(df, 12)
    rows.loc[0, "plan"] = np.nan
    with pytest.raises(ValueError, match="Nonfinite"):
        fit_monthly_hazard(rows, HazardSpec())


def test_discrete_lifetime_and_survival_have_correct_endpoint_convention():
    from causal_bench.estimators.monthly_hazard import MonthlyHazardFit
    h, p = 4, 0.2
    fit = MonthlyHazardFit(HazardSpec(h, (), False), np.full(h, np.log(p/(1-p))),
                          np.eye(h), tuple(f"age_{k}" for k in range(1, h+1)), 0, 1, 1, 0)
    subscribers = pd.DataFrame({"cohort": [0, 12]})
    expected = (1 - p) ** np.arange(h+1)
    np.testing.assert_allclose(fit.predict_survival(subscribers), np.tile(expected, (2, 1)))
    np.testing.assert_allclose(fit.restricted_lifetime(subscribers), expected[:-1].sum())
    # Endpoint cancellation time T=1,2,3,4, or >4, independently enumerated.
    mass = p*(1-p)**np.arange(h)
    lifetime = np.dot(np.arange(1, h+1), mass) + h*(1-p)**h
    assert lifetime == pytest.approx(expected[:-1].sum())


def test_chronological_splits_and_oracle_isolation():
    df, truth = simulate_subscriptions(4000, seed=3)
    split = cohort_split(df)
    assert set(df.columns).isdisjoint(truth)
    assert df.iloc[split["train"]].cohort.max() < df.iloc[split["validation"]].cohort.min()
    assert df.iloc[split["validation"]].cohort.max() < df.iloc[split["test"]].cohort.min()
    ids = [set(df.iloc[v].subscription_id) for v in split.values()]
    assert not (ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
    assert sum(map(len, ids)) == len(df)
    assert df.iloc[split["train"]].cohort.max() + 12 < df.iloc[split["validation"]].cohort.min()
    assert df.iloc[split["validation"]].cohort.max() + 12 < df.iloc[split["test"]].cohort.min()


def test_known_hazard_simulation_and_censor_adjusted_brier():
    df, truth = simulate_subscriptions(60000, seed=17)
    # Before fitting anything, expected curves match complete cancellation outcomes.
    empirical = np.array([(truth["event_time"] > k).mean() for k in range(13)])
    np.testing.assert_allclose(empirical, truth["survival"].mean(axis=0), atol=0.009)
    from causal_bench.estimators.monthly_hazard import MonthlyHazardFit
    spec = HazardSpec(age_interactions=("plan",))
    beta = np.r_[-2.7 - .065*np.arange(12), .35, -.45, -.4, .4, .2, .06]
    _, names = design(person_months(df.iloc[:50], 12), spec)
    fit = MonthlyHazardFit(spec, beta, np.eye(len(beta)), names, 0, 1, 1, 0)
    np.testing.assert_allclose(fit.predict_survival(df), truth["survival"], atol=1e-14)
    metrics = evaluate(fit, df, truth)
    np.testing.assert_allclose(metrics["ipcw_brier_by_month_known_G"],
                               metrics["brier_by_month_complete_oracle"], atol=.006)


def test_benchmark_recovers_future_cohort_truth():
    result = run(n=12000, seed=2)
    m = result["metrics"]["calendar_interaction"]["test"]
    assert m["survival_rmse_vs_oracle"] < .035
    assert abs(m["restricted_active_months"] - m["oracle_restricted_active_months"]) < .2
    assert result["aggregation_equivalence"]["max_coefficient_error"] < 1e-7
    assert result["subscriber_bootstrap"]["requested"] == 0


def test_test_labels_cannot_change_model_selection(monkeypatch):
    import causal_bench.validation.subscription_retention as module
    original = module.simulate_subscriptions
    first = run(n=4000, seed=4)

    def changed_test_labels(*args, **kwargs):
        df, truth = original(*args, **kwargs)
        m = df.cohort >= 54
        df.loc[m, "observed_time"] = 1
        df.loc[m, "event"] = 1
        return df, truth

    monkeypatch.setattr(module, "simulate_subscriptions", changed_test_labels)
    second = run(n=4000, seed=4)
    assert first["selected_using_validation_only"] == second["selected_using_validation_only"]
    assert first["selected_coefficients"] == second["selected_coefficients"]
    assert first["validation_selection_log_loss"] == second["validation_selection_log_loss"]


def test_bootstrap_objective_roundoff_does_not_reject_converged_fit():
    # Regression for replicate 51 of seed-0 bootstrap: the final Newton decrease
    # was below objective precision. Keep the gradient tolerance unchanged.
    df, _ = simulate_subscriptions(24000)
    train = df.iloc[cohort_split(df)["train"]]
    rng = np.random.default_rng(100000)
    for _ in range(52):
        index = rng.integers(0, len(train), len(train))
    sample = train.iloc[index].copy()
    sample["subscription_id"] = np.arange(len(sample))
    fit = fit_monthly_hazard(person_months(sample, 12), HazardSpec(age_interactions=("plan",)))
    x, _ = design(person_months(sample, 12), fit.spec)
    score = x.T @ (expit(x @ fit.coefficients) - person_months(sample, 12).event.to_numpy()) / len(x)
    assert np.max(np.abs(score)) < 1e-10
