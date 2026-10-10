"""Unpenalized monthly logistic hazards, independent of R/concrete.

This is a prediction baseline, not a causal treatment-effect estimator. Reuses
exp34's interval-intercept/cumulative-product design without its IPCW nuisance
fit or subgroup targeting. Model-based covariance is inverse observed information;
it is not a cluster-robust covariance estimate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit


@dataclass(frozen=True)
class HazardSpec:
    horizon: int = 12
    covariates: tuple[str, ...] = ("channel", "plan", "engaged")
    seasonality: bool = True
    age_interactions: tuple[str, ...] = ()

    def __post_init__(self):
        if not isinstance(self.horizon, int) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if len(set(self.covariates)) != len(self.covariates):
            raise ValueError("Duplicate covariates")
        if (len(set(self.age_interactions)) != len(self.age_interactions)
                or not set(self.age_interactions) <= set(self.covariates)):
            raise ValueError("Interactions must be distinct modeled covariates")


def person_months(subscriptions: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Expand one row/subscription to observed monthly outcomes.

    Required: subscription_id, cohort (integer acquisition month), observed_time
    (months since acquisition), event (1=cancellation at an integer month end).
    Optional entry_age is an integer number of elapsed months at delayed entry.
    Censoring at 2.4 contributes months 1 and 2, never a false zero for month 3.
    A cancellation at month 3 contributes a one for month 3 and no later rows.
    Covariates must be measured at acquisition; this API does not accept histories.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    required = {"subscription_id", "cohort", "observed_time", "event"}
    if not required <= set(subscriptions):
        raise ValueError(f"Missing subscription columns: {required - set(subscriptions)}")
    if subscriptions.empty or subscriptions["subscription_id"].isna().any():
        raise ValueError("Nonempty subscriptions with nonmissing IDs required")
    if subscriptions["subscription_id"].duplicated().any():
        raise ValueError("One row per subscription required")
    t = subscriptions["observed_time"].to_numpy(dtype=float)
    d = subscriptions["event"].to_numpy(dtype=float)
    cohort = subscriptions["cohort"].to_numpy(dtype=float)
    entry = (subscriptions["entry_age"].to_numpy(dtype=float)
             if "entry_age" in subscriptions else np.zeros(len(t)))
    if not all(np.isfinite(v).all() for v in (t, d, cohort, entry)):
        raise ValueError("Nonfinite observation metadata")
    if (np.any(t < 0) or not np.isin(d, [0, 1]).all()
            or np.any(cohort != np.floor(cohort))
            or np.any(entry < 0) or np.any(entry != np.floor(entry))
            or np.any(t < entry)):
        raise ValueError("Invalid time, cohort, entry_age, or event")
    if np.any((d == 1) & ((t != np.floor(t)) | (t <= entry))):
        raise ValueError("Cancellation must be at an observed integer month end after entry")
    age = np.arange(1, horizon + 1)
    eligible = (age[None, :] <= t[:, None]) & (age[None, :] > entry[:, None])
    i, k = np.where(eligible)
    rows = subscriptions.iloc[i].copy().reset_index(drop=True)
    rows["age"] = age[k]
    rows["calendar_month"] = cohort[i].astype(int) + age[k] - 1
    rows["month_of_year"] = rows["calendar_month"] % 12
    rows["event"] = ((d[i] == 1) & (t[i] == age[k])).astype(int)
    return rows


def design(rows: pd.DataFrame, spec: HazardSpec) -> tuple[np.ndarray, tuple[str, ...]]:
    age = rows["age"].to_numpy(dtype=float)
    if (not np.isfinite(age).all() or np.any(age != np.floor(age))
            or np.any(age < 1) or np.any(age > spec.horizon)):
        raise ValueError("Age outside fitted monthly support")
    parts = [np.eye(spec.horizon)[age.astype(int) - 1]]
    names = [f"age_{k}" for k in range(1, spec.horizon + 1)]
    for col in spec.covariates:
        parts.append(rows[[col]].to_numpy(dtype=float))
        names.append(col)
    if spec.seasonality:
        month = rows["month_of_year"].to_numpy(dtype=float)
        if not np.isin(month, np.arange(12)).all():
            raise ValueError("month_of_year must be in 0..11")
        angle = 2 * np.pi * month / 12
        parts.extend([np.sin(angle)[:, None], np.cos(angle)[:, None]])
        names.extend(["calendar_sin", "calendar_cos"])
    for col in spec.age_interactions:
        parts.append((rows[col].to_numpy(dtype=float) * (age - 1))[:, None])
        names.append(f"{col}:age")
    x = np.column_stack(parts)
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite covariates")
    return x, tuple(names)


def binomial_mle(x: np.ndarray, events: np.ndarray, n: np.ndarray):
    """Newton fit to sum(events*eta - n*log(1+exp(eta))).

    Omits binomial combinatorial constants, so the reported likelihood matches
    the expanded Bernoulli likelihood, including after aggregation. No penalty,
    class balancing, or censoring weights. Fail loudly on unsupported fits.
    """
    x, y, n = np.asarray(x, float), np.asarray(events, float), np.asarray(n, float)
    if (x.ndim != 2 or y.shape != (len(x),) or n.shape != y.shape
            or len(x) == 0 or not all(np.isfinite(v).all() for v in (x, y, n))
            or np.any(n <= 0) or np.any(y < 0) or np.any(y > n)
            or np.any(n != np.floor(n)) or np.any(y != np.floor(y))):
        raise ValueError("Invalid binomial design/counts")
    if np.linalg.matrix_rank(x) != x.shape[1]:
        raise ValueError("Hazard design is rank deficient; pool unsupported ages/cells")
    if y.sum() == 0 or y.sum() == n.sum():
        raise ValueError("Both cancellations and non-events required")
    beta = np.zeros(x.shape[1])
    scale = n.sum()

    def loss(b):
        eta = x @ b
        return float(np.sum(n * np.logaddexp(0, eta) - y * eta) / scale)

    for iteration in range(100):
        p = expit(x @ beta)
        score = x.T @ (n * p - y) / scale
        info = x.T @ ((n * p * (1 - p))[:, None] * x) / scale
        if np.max(np.abs(score)) < 1e-10:
            if (np.linalg.cond(info) > 1e12 or np.linalg.eigvalsh(info).min() < 1e-9
                    or np.max(np.abs(beta)) > 20):
                raise ValueError("Unstable/separated hazard fit; pool or regularize explicitly")
            return beta, np.linalg.inv(info) / scale, -loss(beta) * scale, iteration
        try:
            step = np.linalg.solve(info, score)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Singular hazard information") from exc
        before = loss(beta)
        fraction = 1.0
        while fraction > 2**-25:
            candidate = beta - fraction * step
            # Near the optimum the objective improvement is below floating-point
            # resolution. Allow only rounding-sized slack; the score convergence
            # requirement above is unchanged (no relaxed statistical tolerance).
            slack = 8 * np.finfo(float).eps * max(1.0, abs(before))
            if loss(candidate) <= before - 1e-4 * fraction * float(score @ step) + slack:
                beta = candidate
                break
            fraction *= 0.5
        else:
            raise ValueError("Hazard fit line search failed")
    raise ValueError("Hazard fit did not converge (possibly separation)")


@dataclass
class MonthlyHazardFit:
    spec: HazardSpec
    coefficients: np.ndarray
    covariance: np.ndarray
    names: tuple[str, ...]
    log_likelihood: float
    fit_rows: int
    observations: int
    iterations: int

    def predict_hazard(self, rows):
        x, names = design(rows, self.spec)
        if names != self.names:
            raise ValueError("Prediction design changed")
        return expit(x @ self.coefficients)

    def predict_survival(self, subscriptions):
        """S[:, k]=P(T>k), k=0..H, forecasting from acquisition (not entry)."""
        cohort = subscriptions["cohort"].to_numpy(dtype=float)
        if not np.isfinite(cohort).all() or np.any(cohort != np.floor(cohort)):
            raise ValueError("Integer acquisition cohort required")
        n, h = len(subscriptions), self.spec.horizon
        rows = subscriptions.iloc[np.repeat(np.arange(n), h)].copy()
        rows["age"] = np.tile(np.arange(1, h + 1), n)
        rows["month_of_year"] = (rows["cohort"].to_numpy() + rows["age"].to_numpy() - 1) % 12
        hazard = self.predict_hazard(rows).reshape(n, h)
        return np.column_stack([np.ones(n), np.cumprod(1 - hazard, axis=1)])

    def restricted_lifetime(self, subscriptions):
        """E[min(T,H)] = sum_{k=0}^{H-1} S(k), in active months.

        Cancellation is at month ends. This is a discrete sum, not trapezoidal
        integration of a continuous event-time curve, and is not lifetime LTV.
        """
        return self.predict_survival(subscriptions)[:, :-1].sum(axis=1)


def fit_monthly_hazard(rows: pd.DataFrame, spec: HazardSpec, *, aggregate=True):
    # Validate before grouping: NaNs must not disappear through pandas groupby.
    design(rows, spec)
    if not np.isin(rows["event"].to_numpy(), [0, 1]).all():
        raise ValueError("Person-month events must be binary")
    if aggregate:
        keys = ["age", *spec.covariates]
        if spec.seasonality:
            keys.append("month_of_year")
        cells = rows.groupby(keys, sort=True, dropna=False)["event"].agg(["sum", "size"]).reset_index()
        x, names = design(cells, spec)
        events, n = cells["sum"].to_numpy(), cells["size"].to_numpy()
    else:
        x, names = design(rows, spec)
        events, n = rows["event"].to_numpy(), np.ones(len(rows))
    beta, cov, ll, iterations = binomial_mle(x, events, n)
    return MonthlyHazardFit(spec, beta, cov, names, ll, len(x), int(n.sum()), iterations)
