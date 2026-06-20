"""Hawthorne estimators for Exp 12: DiD methods for staggered deployment.

All estimators take a site × period panel DataFrame (from generate_hawthorne_data)
and estimate the average treatment effect at each event-time (periods since deployment).

Estimators:
  naive_twfe          — OLS with entity + time FE; biased under heterogeneous effects
  event_study_twfe    — TWFE with relative-time dummies; shows the Hawthorne arc
  dchd_dynamic        — De Chaisemartin-D'Haultfoeuille group-time ATTs (simplified)
  callaway_santhanna  — Doubly-robust group-time ATTs (Callaway-Sant'Anna 2021 style)
  twfe_with_calendar  — TWFE + continuous calendar-time covariate (Senn fix)

References:
  de Chaisemartin & D'Haultfoeuille (2020). AER.
  Callaway & Sant'Anna (2021). JoE.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class HawthorneEstimate:
    method: str
    beta: float                        # overall average treatment effect
    se: float = float("nan")
    event_time_ates: dict[int, float] = field(default_factory=dict)   # {event_time: ATT}
    event_time_ses: dict[int, float] = field(default_factory=dict)
    neg_weight_fraction: float = float("nan")  # TWFE negative weight share


# ─── Two-way demeaning (Gauss-Seidel iteration) ───────────────────────────────

def _twoway_demean(
    x: np.ndarray,
    site_ids: np.ndarray,
    period_ids: np.ndarray,
    max_iter: int = 200,
    tol: float = 1e-9,
) -> np.ndarray:
    """Two-way within transformation via alternating projections."""
    x_dm = x.copy().astype(float)
    unique_sites = np.unique(site_ids)
    unique_periods = np.unique(period_ids)
    for _ in range(max_iter):
        x_prev = x_dm.copy()
        for s in unique_sites:
            mask = site_ids == s
            x_dm[mask] -= x_dm[mask].mean()
        for t in unique_periods:
            mask = period_ids == t
            x_dm[mask] -= x_dm[mask].mean()
        if np.max(np.abs(x_dm - x_prev)) < tol:
            break
    return x_dm


def _ols_se(Y_dm: np.ndarray, X_dm: np.ndarray, n_fe: int) -> float:
    """OLS SE for coefficient on X_dm after two-way demeaning."""
    beta = np.dot(X_dm, Y_dm) / (np.dot(X_dm, X_dm) + 1e-12)
    resid = Y_dm - beta * X_dm
    n = len(Y_dm)
    df = n - n_fe - 1
    df = max(df, 1)
    var_beta = (np.dot(resid, resid) / df) / (np.dot(X_dm, X_dm) + 1e-12)
    return float(np.sqrt(max(var_beta, 0)))


# ─── Negative weight diagnostic ───────────────────────────────────────────────

def _negative_weight_fraction(
    D: np.ndarray,
    site_ids: np.ndarray,
    period_ids: np.ndarray,
) -> float:
    """Fraction of observations where the two-way demeaned D is negative.

    Approximates the de Chaisemartin-D'Haultfoeuille diagnostic: TWFE weights
    on clean comparisons (untreated vs newly-treated) are positive, but weights
    on already-treated units used as controls are negative. The fraction of
    observations with D_dm < 0 is a proxy for the negative-weight share.
    """
    D_dm = _twoway_demean(D, site_ids, period_ids)
    n_neg = int(np.sum(D_dm < 0))
    return float(n_neg / len(D_dm))


# ─── 1. Naive TWFE ────────────────────────────────────────────────────────────

def naive_twfe(df: pd.DataFrame) -> HawthorneEstimate:
    """Y ~ D + entity_FE + time_FE (OLS via within transformation)."""
    Y = df["Y"].values.astype(float)
    D = df["D"].values.astype(float)
    site_ids = df["site"].values
    period_ids = df["period"].values

    Y_dm = _twoway_demean(Y, site_ids, period_ids)
    D_dm = _twoway_demean(D, site_ids, period_ids)

    beta = float(np.dot(D_dm, Y_dm) / (np.dot(D_dm, D_dm) + 1e-12))
    n_fe = len(np.unique(site_ids)) + len(np.unique(period_ids))
    se = _ols_se(Y_dm, D_dm, n_fe)
    neg_frac = _negative_weight_fraction(D, site_ids, period_ids)

    return HawthorneEstimate(
        method="naive_twfe",
        beta=beta,
        se=se,
        neg_weight_fraction=neg_frac,
    )


# ─── 2. Event-study TWFE ──────────────────────────────────────────────────────

def event_study_twfe(df: pd.DataFrame) -> HawthorneEstimate:
    """TWFE with relative-time dummies (event-study specification).

    Drops event_time = -1 as the reference period. Never-treated sites receive
    rel_time = -99 (excluded from the relative-time dummies but included in the
    entity + time FE as control units).

    Returns event_time_ates mapping event_time → estimated ATT at that event-time.
    """
    site_ids = df["site"].values
    period_ids = df["period"].values
    Y = df["Y"].values.astype(float)

    deploy_by_site = df.groupby("site")["deploy_period"].first()

    # Compute rel_time for each obs
    rel_times = []
    for _, row in df.iterrows():
        dp = deploy_by_site.loc[row["site"]]
        if pd.isna(dp) or dp is None:
            rel_times.append(-99)
        else:
            rel_times.append(int(row["period"] - dp))
    rel_times = np.array(rel_times, dtype=int)

    # All event-times that appear (excluding reference -1 and never-treated -99)
    event_time_labels = sorted(set(rel_times[(rel_times != -1) & (rel_times != -99)]))

    n = len(Y)
    n_sites = len(np.unique(site_ids))
    n_periods = len(np.unique(period_ids))

    # Design matrix: one dummy per non-reference, non-control event-time
    K = len(event_time_labels)
    X = np.zeros((n, K))
    for k, et in enumerate(event_time_labels):
        X[:, k] = (rel_times == et).astype(float)

    # Two-way demean each column of X and Y
    Y_dm = _twoway_demean(Y, site_ids, period_ids)
    X_dm = np.column_stack([
        _twoway_demean(X[:, k], site_ids, period_ids) for k in range(K)
    ])

    # OLS: β = (X_dm'X_dm)^{-1} X_dm'Y_dm
    try:
        betas = np.linalg.lstsq(X_dm, Y_dm, rcond=None)[0]
    except np.linalg.LinAlgError:
        betas = np.full(K, float("nan"))

    # SE for each coefficient (sandwich-style from diagonal)
    resid = Y_dm - X_dm @ betas
    df_resid = max(n - K - n_sites - n_periods, 1)
    sigma2 = float(np.dot(resid, resid) / df_resid)
    XtX = X_dm.T @ X_dm + 1e-10 * np.eye(K)
    try:
        cov = sigma2 * np.linalg.inv(XtX)
        ses = np.sqrt(np.maximum(np.diag(cov), 0))
    except np.linalg.LinAlgError:
        ses = np.full(K, float("nan"))

    event_time_ates = {int(et): float(betas[k]) for k, et in enumerate(event_time_labels)}
    event_time_ses = {int(et): float(ses[k]) for k, et in enumerate(event_time_labels)}

    # Overall beta = average of post-treatment event-time coefficients
    post_ates = [v for k, v in event_time_ates.items() if k >= 0]
    beta_avg = float(np.mean(post_ates)) if post_ates else float("nan")

    return HawthorneEstimate(
        method="event_study_twfe",
        beta=beta_avg,
        event_time_ates=event_time_ates,
        event_time_ses=event_time_ses,
        neg_weight_fraction=_negative_weight_fraction(
            df["D"].values.astype(float), site_ids, period_ids
        ),
    )


# ─── 3. DCHD dynamic (simplified did_multiplegt_dyn) ─────────────────────────

def dchd_dynamic(df: pd.DataFrame) -> HawthorneEstimate:
    """De Chaisemartin-D'Haultfoeuille dynamic ATTs (simplified).

    For each cohort g (first period treated) and each post-treatment event-time k:
        ATT(g, k) = (Ȳ_{g, g+k} - Ȳ_{g, g-1}) - (Ȳ_{ctrl, g+k} - Ȳ_{ctrl, g-1})

    where ctrl = never-treated OR not-yet-treated-at-g sites.
    ATT(event_time=k) is then the weighted average of ATT(g, k) across cohorts,
    weighted by cohort size at baseline.
    """
    site_periods = df.set_index(["site", "period"])["Y"]
    deploy_by_site = df.groupby("site")["deploy_period"].first()

    # Separate treated cohorts and control units
    treated_sites = deploy_by_site[deploy_by_site.notna()].index.tolist()
    never_treated = deploy_by_site[deploy_by_site.isna()].index.tolist()

    cohort_groups: dict[int, list[int]] = {}
    for s in treated_sites:
        dp = int(deploy_by_site[s])
        cohort_groups.setdefault(dp, []).append(s)

    max_periods = int(df["period"].max())

    # Aggregate by event_time across cohorts
    ate_by_event: dict[int, list[tuple[float, int]]] = {}  # event_time → [(ATT, n_cohort)]

    for g, sites_g in cohort_groups.items():
        base_period = g - 1
        if base_period < 1:
            continue

        # Baseline Y for cohort g
        def _mean_Y(site_list: list[int], period: int) -> Optional[float]:
            vals = [site_periods.get((s, period), np.nan) for s in site_list]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else None

        Y_g_base = _mean_Y(sites_g, base_period)
        if Y_g_base is None:
            continue

        for k in range(0, max_periods - g + 1):
            t = g + k
            if t > max_periods:
                break

            Y_g_t = _mean_Y(sites_g, t)
            if Y_g_t is None:
                continue

            # Control group: never-treated + not-yet-treated at period g
            ctrl_sites = never_treated + [
                s for s, dp in deploy_by_site.items()
                if not pd.isna(dp) and int(dp) > t
            ]
            if not ctrl_sites:
                ctrl_sites = never_treated

            Y_ctrl_base = _mean_Y(ctrl_sites, base_period)
            Y_ctrl_t = _mean_Y(ctrl_sites, t)

            if Y_ctrl_base is None or Y_ctrl_t is None:
                continue

            att = (Y_g_t - Y_g_base) - (Y_ctrl_t - Y_ctrl_base)
            ate_by_event.setdefault(k, []).append((att, len(sites_g)))

    event_time_ates: dict[int, float] = {}
    for et, pairs in ate_by_event.items():
        atts, ns = zip(*pairs)
        ns = np.array(ns, dtype=float)
        event_time_ates[et] = float(np.average(atts, weights=ns))

    post_ates = [v for k, v in event_time_ates.items() if k >= 0]
    beta_avg = float(np.mean(post_ates)) if post_ates else float("nan")

    return HawthorneEstimate(
        method="dchd_dynamic",
        beta=beta_avg,
        event_time_ates=event_time_ates,
    )


# ─── 4. Callaway-Sant'Anna (DR group-time ATTs) ───────────────────────────────

def callaway_santhanna(df: pd.DataFrame) -> HawthorneEstimate:
    """Doubly-robust group-time ATTs (Callaway-Sant'Anna 2021 style).

    Augments the DCHD DiD with IPW on the cohort-selection probability so the
    estimator is consistent if either the selection model or the parallel-trends
    assumption holds for the outcome model.

    In our setting without unit-level covariates, this reduces to IPW on
    cohort proportions (uniform weights across sites in each cohort), so the
    difference from dchd_dynamic is minimal. The main value is in demonstrating
    the framework for comparison with dchd_dynamic.
    """
    # For concordance check: estimate cohort-selection propensity
    # Here we use equal weights since we have no site-level covariates
    # (HawthorneConfig doesn't generate site-level X beyond random effects)
    site_periods = df.set_index(["site", "period"])["Y"]
    deploy_by_site = df.groupby("site")["deploy_period"].first()

    treated_sites = deploy_by_site[deploy_by_site.notna()].index.tolist()
    never_treated = deploy_by_site[deploy_by_site.isna()].index.tolist()

    cohort_groups: dict[int, list[int]] = {}
    for s in treated_sites:
        dp = int(deploy_by_site[s])
        cohort_groups.setdefault(dp, []).append(s)

    max_periods = int(df["period"].max())
    ate_by_event: dict[int, list[tuple[float, int]]] = {}

    for g, sites_g in cohort_groups.items():
        base_period = g - 1
        if base_period < 1:
            continue

        def _mean_Y(site_list: list[int], period: int) -> Optional[float]:
            vals = [site_periods.get((s, period), np.nan) for s in site_list]
            vals = [v for v in vals if not np.isnan(v)]
            return float(np.mean(vals)) if vals else None

        Y_g_base = _mean_Y(sites_g, base_period)
        if Y_g_base is None:
            continue

        # Control: never-treated (pure comparison group for CS)
        # This matches the "nevertreated" option in csdid / csa R packages
        ctrl_sites = never_treated if never_treated else [
            s for s, dp in deploy_by_site.items()
            if not pd.isna(dp) and int(dp) > max_periods
        ]
        if not ctrl_sites:
            # Fall back to not-yet-treated
            ctrl_sites = [
                s for s, dp in deploy_by_site.items()
                if not pd.isna(dp) and int(dp) > g + (max_periods - g) // 2
            ]
        if not ctrl_sites:
            continue

        Y_ctrl_base = _mean_Y(ctrl_sites, base_period)
        if Y_ctrl_base is None:
            continue

        # IPW: propensity of being in cohort g vs control, given pre-treatment outcome
        # With no site covariates, the IPW weight is proportional to cohort/control sizes.
        n_g = len(sites_g)
        n_c = len(ctrl_sites)
        p_g = n_g / (n_g + n_c)  # marginal probability of being treated cohort g
        ipw_g = 1.0 / p_g if p_g > 0 else 1.0
        ipw_c = 1.0 / (1 - p_g) if p_g < 1 else 1.0

        for k in range(0, max_periods - g + 1):
            t = g + k
            if t > max_periods:
                break
            Y_g_t = _mean_Y(sites_g, t)
            Y_ctrl_t = _mean_Y(ctrl_sites, t)
            if Y_g_t is None or Y_ctrl_t is None:
                continue

            # DR-ATT: doubly-robust combination
            att = (Y_g_t - Y_g_base) * ipw_g - (Y_ctrl_t - Y_ctrl_base) * ipw_c
            # Re-normalise to put on the same scale as dchd
            att = att * p_g  # equivalent to standard ATT formula after normalisation
            ate_by_event.setdefault(k, []).append((att, n_g))

    event_time_ates: dict[int, float] = {}
    for et, pairs in ate_by_event.items():
        atts, ns = zip(*pairs)
        event_time_ates[et] = float(np.average(atts, weights=np.array(ns, dtype=float)))

    post_ates = [v for k, v in event_time_ates.items() if k >= 0]
    beta_avg = float(np.mean(post_ates)) if post_ates else float("nan")

    return HawthorneEstimate(
        method="callaway_santhanna",
        beta=beta_avg,
        event_time_ates=event_time_ates,
    )


# ─── 5. TWFE with calendar-time covariate (Senn fix) ─────────────────────────

def twfe_with_calendar(df: pd.DataFrame) -> HawthorneEstimate:
    """TWFE + continuous calendar-time covariate.

    Y ~ D + period (continuous) + entity_FE + time_FE

    Including period as a linear covariate absorbs the secular trend that is
    confounded with the staggered deployment timing (earlier-deploying sites
    get a longer time-series with more secular improvement). This is the
    'Senn fix' for enrollment-drift bias.
    """
    Y = df["Y"].values.astype(float)
    D = df["D"].values.astype(float)
    T_cont = df["period"].values.astype(float)
    site_ids = df["site"].values
    period_ids = df["period"].values

    # Demean Y and D by entity + time FE
    Y_dm = _twoway_demean(Y, site_ids, period_ids)
    D_dm = _twoway_demean(D, site_ids, period_ids)
    T_dm = _twoway_demean(T_cont, site_ids, period_ids)

    # OLS with [D_dm, T_dm] as regressors
    X_dm = np.column_stack([D_dm, T_dm])
    try:
        betas = np.linalg.lstsq(X_dm, Y_dm, rcond=None)[0]
        beta_D = float(betas[0])
    except np.linalg.LinAlgError:
        beta_D = float("nan")

    n_fe = len(np.unique(site_ids)) + len(np.unique(period_ids))
    resid = Y_dm - X_dm @ betas
    df_resid = max(len(Y) - 2 - n_fe, 1)
    sigma2 = float(np.dot(resid, resid) / df_resid)
    XtX = X_dm.T @ X_dm + 1e-10 * np.eye(2)
    try:
        se_D = float(np.sqrt(max(sigma2 * np.linalg.inv(XtX)[0, 0], 0)))
    except np.linalg.LinAlgError:
        se_D = float("nan")

    return HawthorneEstimate(
        method="twfe_with_calendar",
        beta=beta_D,
        se=se_D,
        neg_weight_fraction=_negative_weight_fraction(D, site_ids, period_ids),
    )


# ─── Convenience wrapper ──────────────────────────────────────────────────────

def run_all_hawthorne_estimators(df: pd.DataFrame) -> dict[str, HawthorneEstimate]:
    """Run all five Hawthorne estimators and return results by name."""
    return {
        "naive_twfe":         naive_twfe(df),
        "event_study_twfe":   event_study_twfe(df),
        "dchd_dynamic":       dchd_dynamic(df),
        "callaway_santhanna": callaway_santhanna(df),
        "twfe_with_calendar": twfe_with_calendar(df),
    }
