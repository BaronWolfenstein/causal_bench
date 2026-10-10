"""Baseline-offer causal RMST for month-end cancellation.

Cross-fitted one-step/AIPW estimator; this is not TMLE or a concrete wrapper.
Identification requires consistency, baseline treatment exchangeability, overlap,
and independent observation censoring given baseline X and offer A. No delayed
entry, repeated offers, competing exits, or post-offer covariates are supported.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from causal_bench.estimators.monthly_hazard import (
    HazardSpec, binomial_mle, fit_monthly_hazard, person_months,
)

BASELINE = ("channel", "plan", "engaged")


def propensity_design(df):
    angle = 2 * np.pi * df["cohort"].to_numpy(float) / 12
    return np.column_stack([np.ones(len(df)), df[list(BASELINE)].to_numpy(float),
                            np.sin(angle), np.cos(angle)])


def validate(df, horizon):
    if "entry_age" in df and np.any(df["entry_age"].to_numpy() != 0):
        raise ValueError("Baseline offer estimator does not support delayed entry")
    person_months(df, horizon)  # validates IDs, times, events and month-end convention
    if "A" not in df or not np.isin(df["A"], [0, 1]).all():
        raise ValueError("Binary baseline offer assignment A required")
    if (not np.isfinite(propensity_design(df)).all()
            or np.any(df["observed_time"].to_numpy() > horizon)):
        raise ValueError("Finite baseline predictors and times within horizon required")


def censoring_rows(df, horizon):
    """Observation dropout BEFORE each month-end outcome, conditional on prior survival.

    A subscriber censored at 2.4 is a dropout in interval 3, not a churn-negative
    outcome in interval 3. At an integer tie, cancellation takes precedence.
    Administrative censoring at H is not an observation dropout at H.
    """
    validate(df, horizon)
    t = df["observed_time"].to_numpy(float)
    d = df["event"].to_numpy(int)
    age = np.arange(1, horizon + 1)
    risk = (t[:, None] >= age - 1) & ~((d[:, None] == 1) & (t[:, None] <= age - 1))
    i, j = np.where(risk)
    rows = df.iloc[i].copy().reset_index(drop=True)
    rows["age"] = age[j]
    rows["event"] = (t[i] < age[j]).astype(int)
    return rows


def arm_pseudo_outcome(df, hazard, censor_survival, propensity, arm, horizon):
    """Per-subscriber RMST one-step pseudo-outcome for one intervention.

    For S(k)=prod_{l<=k}(1-h_l), residual coefficient for month j is
    q_j=sum_{k=j}^{H-1} S(k)/S(j). Compute this backward, avoiding division by a
    tiny survival probability. Only months 1..H-1 affect E[min(T,H)].
    E[pseudo]=RMST if h is correct OR both g and G are correct.
    Arrays must be out-of-fold estimates evaluated under the specified arm.
    """
    n = len(df)
    h, g, p = map(lambda x: np.asarray(x, float), (hazard, censor_survival, propensity))
    if (h.shape != (n, horizon) or g.shape != h.shape or p.shape != (n,)
            or arm not in (0, 1) or not all(np.isfinite(x).all() for x in (h, g, p))
            or np.any((h <= 0) | (h >= 1)) or np.any((g <= 0) | (g > 1))
            or np.any((p <= 0) | (p >= 1)) or np.any(np.diff(g, axis=1) > 1e-12)):
        raise ValueError("Invalid nuisance shapes, probabilities or censor survival")
    survival = np.column_stack([np.ones(n), np.cumprod(1 - h, axis=1)])
    plug = survival[:, :-1].sum(axis=1)
    q = np.zeros_like(h)
    for j in range(horizon - 2, -1, -1):
        q[:, j] = 1 + (1 - h[:, j + 1]) * q[:, j + 1]
    age = np.arange(1, horizon + 1)
    t, d = df["observed_time"].to_numpy(), df["event"].to_numpy()
    observed_risk = t[:, None] >= age
    event = (d[:, None] == 1) & (t[:, None] == age)
    residual = observed_risk * (event - h)
    arm_probability = p if arm == 1 else 1 - p
    correction = ((df["A"].to_numpy() == arm) / arm_probability
                  * np.sum(q * residual / g, axis=1))
    return plug - correction


def estimate_offer(df, *, horizon=12, folds=3, seed=0, randomized_probability=None,
                   outcome_adjusted=True, censor_adjusted=True, propensity_adjusted=True,
                   positivity_floor=0.01):
    """Same-population treated-minus-control active months with Wald IF interval.

    Nuisance specifications/folds are fixed in advance; no outcome-driven choice.
    The explicit lower bounds are reported with their clipping counts. These
    parametric models are a baseline, not protection against hidden confounding.
    """
    validate(df, horizon)
    if (not isinstance(folds, int) or folds < 2 or folds > len(df)
            or not 0 < positivity_floor < 0.5):
        raise ValueError("Valid fold count and positivity floor required")
    if randomized_probability is not None and not 0 < randomized_probability < 1:
        raise ValueError("Randomization probability must lie in (0,1)")
    df = df.reset_index(drop=True)
    rng = np.random.default_rng(seed)
    fold = np.empty(len(df), int)
    for arm in (0, 1):
        indices = np.flatnonzero(df["A"].to_numpy() == arm)
        if len(indices) < folds:
            raise ValueError("Each offer arm needs at least one subscriber per fold")
        fold[rng.permutation(indices)] = np.arange(len(indices)) % folds
    out_spec = HazardSpec(horizon, ("A", *BASELINE) if outcome_adjusted else ("A",),
                          outcome_adjusted, ("plan", "A") if outcome_adjusted else ())
    censor_spec = HazardSpec(horizon, ("A", *BASELINE) if censor_adjusted else ("A",), False)
    pseudo, plug = np.empty((len(df), 2)), np.empty((len(df), 2))
    diagnostics = {"propensity_clipped": 0, "censor_survival_clipped": 0,
                   "minimum_raw_arm_probability": 1.0, "minimum_raw_censor_survival": 1.0,
                   "positivity_floor": positivity_floor, "folds": folds}
    for f in range(folds):
        train, check = df.iloc[fold != f], df.iloc[fold == f]
        outcome = fit_monthly_hazard(person_months(train, horizon), out_spec)
        censor = fit_monthly_hazard(censoring_rows(train, horizon), censor_spec)
        if randomized_probability is not None:
            p = np.full(len(check), randomized_probability)
        else:
            x = propensity_design(train) if propensity_adjusted else np.ones((len(train), 1))
            b, *_ = binomial_mle(x, train["A"].to_numpy(), np.ones(len(train)))
            xc = propensity_design(check) if propensity_adjusted else np.ones((len(check), 1))
            p = expit(xc @ b)
        diagnostics["minimum_raw_arm_probability"] = min(
            diagnostics["minimum_raw_arm_probability"], float(np.minimum(p, 1 - p).min()))
        diagnostics["propensity_clipped"] += int(((p < positivity_floor) | (p > 1 - positivity_floor)).sum())
        p = np.clip(p, positivity_floor, 1 - positivity_floor)
        for arm in (0, 1):
            intervention = check.assign(A=arm)
            s = outcome.predict_survival(intervention)
            h = 1 - s[:, 1:] / s[:, :-1]
            g = censor.predict_survival(intervention)[:, 1:]
            diagnostics["minimum_raw_censor_survival"] = min(
                diagnostics["minimum_raw_censor_survival"], float(g.min()))
            diagnostics["censor_survival_clipped"] += int((g < positivity_floor).sum())
            g = np.maximum(g, positivity_floor)
            pseudo[fold == f, arm] = arm_pseudo_outcome(check, h, g, p, arm, horizon)
            plug[fold == f, arm] = s[:, :-1].sum(axis=1)
    difference = pseudo[:, 1] - pseudo[:, 0]
    point = float(difference.mean())
    se = float(difference.std(ddof=1) / np.sqrt(len(df)))
    # Double robustness of the point estimate does not imply double robustness
    # of a naive plug-in IF variance when one nuisance group is misspecified.
    inference = outcome_adjusted and censor_adjusted and (propensity_adjusted or randomized_probability is not None)
    return {"estimate_months": point, "standard_error": se if inference else None,
            "ci95": [point - 1.96 * se, point + 1.96 * se] if inference else None,
            "arm_rmst_months": pseudo.mean(axis=0).tolist(),
            "g_computation_months": float(np.mean(plug[:, 1] - plug[:, 0])),
            "diagnostics": diagnostics,
            "estimand": "population E[min(T^offer,H)-min(T^no_offer,H)]",
            "horizon_months": horizon, "n": len(df),
            "uncertainty": ("cross-fitted influence-function Wald; requires consistent nuisances and regularity; iid subscribers; no account clustering"
                            if inference else "not reported for deliberately misspecified nuisance specifications")}
