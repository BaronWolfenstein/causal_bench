"""Known-truth monthly retention benchmark; no R or concrete dependency.

    S(k|x,cohort) = product_{j=1}^k (1-h(j,x,calendar_month)).

Only acquisition-time binary predictors are modeled. Calendar seasonality is
periodic, avoiding unrestricted age/period/cohort fixed-effect aliasing. Future
engagement histories, competing exits, and intervention effects are out of scope.
"""
from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from scipy.special import expit

from causal_bench.estimators.monthly_hazard import (
    HazardSpec, fit_monthly_hazard, person_months,
)


def simulate_subscriptions(n=24000, *, horizon=12, seed=0, interaction=0.06):
    """Observed subscriptions and SEPARATE oracle truth, never fed to a fitter.

    Cohorts train 0..23, validation 36..41, test 54..59; administrative
    cutoffs 36/54/72. Maturation gaps prevent future-label leakage: all train
    outcomes are available before validation entry, and all validation outcomes
    before test entry. All cohorts can attain H<=12. Gap cohorts are not sampled.
    Loss to observation is exponential, independent of cancellation conditional
    on baseline channel/plan. Its known survival is used ONLY for evaluation.
    Cancellation is absorbing and occurs at a month end. No reactivation.
    """
    if n < 1 or not isinstance(horizon, int) or not 1 <= horizon <= 12:
        raise ValueError("Positive n and horizon in 1..12 required")
    rng = np.random.default_rng(seed)
    allowed_cohorts = np.r_[np.arange(24), np.arange(36, 42), np.arange(54, 60)]
    cohort = allowed_cohorts[rng.integers(0, len(allowed_cohorts), n)]
    channel, plan, engaged = rng.binomial(1, [0.45, 0.5, 0.6], (n, 3)).T
    age = np.arange(1, horizon + 1)
    month = (cohort[:, None] + age - 1) % 12
    eta = (-2.7 - 0.065 * (age - 1) + 0.35 * channel[:, None]
           - 0.45 * plan[:, None] - 0.4 * engaged[:, None]
           + 0.4 * np.sin(2 * np.pi * month / 12)
           + 0.2 * np.cos(2 * np.pi * month / 12)
           + interaction * plan[:, None] * (age - 1))
    hazard = expit(eta)
    failures = rng.random((n, horizon)) < hazard
    event_time = np.where(failures.any(axis=1), failures.argmax(axis=1) + 1, horizon + 1)
    censor_scale = 28 * np.exp(0.6 * channel - 0.4 * plan)
    censor_time = rng.exponential(censor_scale)
    cutoff = np.where(cohort < 24, 36, np.where(cohort < 42, 54, 72))
    observation_limit = np.minimum(np.minimum(censor_time, cutoff - cohort), horizon)
    event = (event_time <= observation_limit).astype(int)
    observed_time = np.minimum(event_time, observation_limit)
    observed = pd.DataFrame({
        "subscription_id": np.arange(n), "cohort": cohort, "channel": channel,
        "plan": plan, "engaged": engaged, "observed_time": observed_time, "event": event,
    })
    survival = np.column_stack([np.ones(n), np.cumprod(1 - hazard, axis=1)])
    truth = {"hazard": hazard, "survival": survival, "event_time": event_time,
             "censor_scale": censor_scale, "restricted_lifetime": survival[:, :-1].sum(axis=1)}
    return observed, truth


def cohort_split(subscriptions):
    """Split BEFORE expansion, with outcome-maturation gaps between cohorts."""
    c = subscriptions["cohort"].to_numpy()
    allowed = np.r_[np.arange(24), np.arange(36, 42), np.arange(54, 60)]
    if not np.isin(c, allowed).all():
        raise ValueError("Benchmark cohorts must be in train/validation/test ranges")
    return {"train": np.flatnonzero(c <= 23),
            "validation": np.flatnonzero((c >= 36) & (c <= 41)),
            "test": np.flatnonzero(c >= 54)}


def observed_log_loss(fit, rows):
    p = np.clip(fit.predict_hazard(rows), 1e-12, 1 - 1e-12)
    y = rows["event"].to_numpy()
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log1p(-p)))


def evaluate(fit, subscriptions, truth):
    """Simulation-oracle curve errors and known-G IPCW Brier, not training labels.

    Horvitz-Thompson Brier divides by subscribers, not sum of observed weights.
    Real data requires a validated censoring estimator; known G is an oracle here.
    """
    survival = fit.predict_survival(subscriptions)
    h = fit.spec.horizon
    t = subscriptions["observed_time"].to_numpy()
    d = subscriptions["event"].to_numpy()
    ipcw, oracle = [], []
    for k in range(1, h + 1):
        failed = (d == 1) & (t <= k)
        known_alive = t >= k
        g = np.exp(-np.minimum(t, k) / truth["censor_scale"])
        w = np.where(failed | known_alive, 1 / g, 0)
        y = failed.astype(float)
        risk = 1 - survival[:, k]
        ipcw.append(float(np.mean(w * (y - risk)**2)))
        oracle.append(float(np.mean(((truth["event_time"] <= k) - risk)**2)))
    lifetime = survival[:, :-1].sum(axis=1)
    risk = 1 - survival[:, -1]
    failed = (d == 1) & (t <= h)
    event_weight = failed * np.exp(np.minimum(t, h) / truth["censor_scale"])
    calibration = []
    bins = np.minimum((risk * 10).astype(int), 9)
    for b in range(10):
        m = bins == b
        if m.any():
            calibration.append({"bin": [b / 10, (b + 1) / 10], "n": int(m.sum()),
                                "predicted_risk": float(risk[m].mean()),
                                "observed_risk_ipcw_known_G": float(event_weight[m].mean()),
                                "oracle_risk": float((1 - truth["survival"][m, -1]).mean())})
    return {
        "survival_rmse_vs_oracle": float(np.sqrt(np.mean((survival[:, 1:] - truth["survival"][:, 1:])**2))),
        "retention_at_horizon": float(survival[:, -1].mean()),
        "oracle_retention_at_horizon": float(truth["survival"][:, -1].mean()),
        "restricted_active_months": float(lifetime.mean()),
        "oracle_restricted_active_months": float(truth["restricted_lifetime"].mean()),
        "restricted_lifetime_mae_vs_oracle": float(np.mean(np.abs(lifetime - truth["restricted_lifetime"]))),
        "ipcw_brier_by_month_known_G": ipcw, "brier_by_month_complete_oracle": oracle,
        "horizon_risk_calibration": calibration,
        "observed_person_month_log_loss": observed_log_loss(fit, person_months(subscriptions, h)),
    }


def run(*, n=24000, seed=0, horizon=12, bootstrap_reps=0):
    subscriptions, truth = simulate_subscriptions(n, horizon=horizon, seed=seed)
    split = cohort_split(subscriptions)
    if any(len(ids) == 0 for ids in split.values()):
        raise ValueError("Each chronological partition must contain subscriptions")
    train = subscriptions.iloc[split["train"]]
    validation = subscriptions.iloc[split["validation"]]
    train_rows = person_months(train, horizon)
    val_rows = person_months(validation, horizon)
    specs = {
        "tenure_only": HazardSpec(horizon, (), False),
        "tenure_covariates": HazardSpec(horizon, seasonality=False),
        "calendar_main": HazardSpec(horizon),
        "calendar_interaction": HazardSpec(horizon, age_interactions=("plan",)),
    }
    fits = {name: fit_monthly_hazard(train_rows, spec) for name, spec in specs.items()}
    validation_loss = {name: observed_log_loss(fit, val_rows) for name, fit in fits.items()}
    selected = min(validation_loss, key=validation_loss.get)  # locked before test evaluation
    row_fit = fit_monthly_hazard(train_rows, specs[selected], aggregate=False)
    fit = fits[selected]
    equivalence = {
        "max_coefficient_error": float(np.max(np.abs(fit.coefficients - row_fit.coefficients))),
        "max_covariance_error": float(np.max(np.abs(fit.covariance - row_fit.covariance))),
        "log_likelihood_error_per_observation": abs(fit.log_likelihood - row_fit.log_likelihood) / len(train_rows),
        "person_months": len(train_rows), "aggregated_cells": fit.fit_rows,
    }
    if (equivalence["max_coefficient_error"] > 1e-7 or equivalence["max_covariance_error"] > 1e-7
            or equivalence["log_likelihood_error_per_observation"] > 1e-10):
        raise ValueError("Row/cell likelihood equivalence failed")
    results = {}
    for name, fitted in fits.items():
        results[name] = {}
        for partition in ("validation", "test"):
            ids = split[partition]
            results[name][partition] = evaluate(
                fitted, subscriptions.iloc[ids], {key: value[ids] for key, value in truth.items()})
    test = subscriptions.iloc[split["test"]]
    boot, bootstrap_failures = [], []
    if bootstrap_reps < 0:
        raise ValueError("bootstrap_reps must be nonnegative")
    rng = np.random.default_rng(seed + 100_000)
    for _ in range(bootstrap_reps):
        # Resample subscriptions, never person-months. Give duplicate draws new IDs.
        sample = train.iloc[rng.integers(0, len(train), len(train))].copy()
        sample["subscription_id"] = np.arange(len(sample))
        try:
            bf = fit_monthly_hazard(person_months(sample, horizon), specs[selected])
            boot.append(float(bf.restricted_lifetime(test).mean()))
        except ValueError as exc:
            bootstrap_failures.append({"replicate": len(boot) + len(bootstrap_failures), "reason": str(exc)})
    bootstrap = {"requested": bootstrap_reps, "successful": len(boot),
                 "failed": bootstrap_reps - len(boot),
                 "failures": bootstrap_failures,
                 "scope": "training-subscriber bootstrap; selected spec and test covariates fixed; selection uncertainty excluded",
                 "restricted_active_months_percentile_95": (
                     np.quantile(boot, [0.025, 0.975]).tolist() if len(boot) >= 20 else None)}
    return {
        "configuration": {"n": n, "seed": seed, "horizon": horizon, "interaction": 0.06},
        "partition_sizes": {k: len(v) for k, v in split.items()},
        "partition_cohorts": {"train": [0, 23], "validation": [36, 41], "test": [54, 59]},
        "as_of_cutoffs": {"train": 36, "validation": 54, "test": 72},
        "selected_using_validation_only": selected, "validation_selection_log_loss": validation_loss,
        "specifications": {k: asdict(v) for k, v in specs.items()}, "aggregation_equivalence": equivalence,
        "selected_coefficients": dict(zip(fit.names, fit.coefficients.tolist())),
        "selected_model_based_se": dict(zip(fit.names, np.sqrt(np.diag(fit.covariance)).tolist())),
        "metrics": results, "subscriber_bootstrap": bootstrap,
    }
