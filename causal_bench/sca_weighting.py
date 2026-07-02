"""Propensity weighting + region-R positivity map for the SCA balance pipeline.

Replaces the synthetic-demo stand-ins in exp29 with the production seam:

- ``propensity_scores`` — P(Target | X) via the production nuisance learner
  (HAL, the primary; logistic as a fast fallback / test path). HAL's càdlàg /
  bounded-variation rate is what licenses the downstream doubly-robust inference
  (see the HAL-vs-GP decision issue); this is where that learner enters.
- ``odds_weights`` — the ATT-style Baseline reweighting (odds of the propensity).
- ``region_r_from_positivity`` — the sparse region R as an **output** of the
  fitted positivity map, not a hardcoded covariate cutoff: R is the tail of the
  propensity where the Baseline Cohort's effective support (Kish ESS) falls
  below a floor. On an engineered DGP this coincides with the severity tail, but
  it is discovered from the weights, so it transfers to real covariates where no
  single cutoff defines the sparse region.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _kish(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if w.size else 0.0


def propensity_scores(target: pd.DataFrame, baseline: pd.DataFrame, covs: list[str],
                      method: str = "hal", seed: int = 0):
    """Estimate P(Target | X) for both groups. Returns ``(e_target, e_baseline)``.

    method='hal' uses the production HAL classifier (requires the hal9001 R
    package); method='logistic' is the fast standardized-logistic fallback used
    in tests and where HAL is unavailable.
    """
    Xt, Xb = target[covs].to_numpy(float), baseline[covs].to_numpy(float)
    X = np.vstack([Xt, Xb])
    y = np.r_[np.ones(len(Xt)), np.zeros(len(Xb))]
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd

    if method == "hal":
        from causal_bench.hal import HALClassifier
        model = HALClassifier().fit(Xs, y)
        p = model.predict_proba(Xs)[:, 1]
    elif method == "logistic":
        p = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, y).predict_proba(Xs)[:, 1]
    else:
        raise ValueError(f"unknown propensity method: {method!r}")
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return p[:len(Xt)], p[len(Xt):]


def odds_weights(e_baseline: np.ndarray) -> np.ndarray:
    """ATT-style Baseline weights: odds of the propensity, normalized to mean 1."""
    e = np.clip(np.asarray(e_baseline, dtype=float), 1e-6, 1 - 1e-6)
    w = e / (1 - e)
    return w * len(w) / w.sum()


def region_r_from_positivity(target: pd.DataFrame, baseline: pd.DataFrame,
                             e_target: np.ndarray, e_baseline: np.ndarray,
                             ess_floor: float = 40.0) -> dict:
    """Define the sparse region R from the positivity map, not a covariate cutoff.

    R = {records with propensity ≥ q*}, where q* is the smallest threshold at
    which the Baseline Cohort's Kish ESS among ``e ≥ q`` drops to ``ess_floor``
    — i.e. the propensity tail where Baseline support becomes thin. Returned as
    boolean masks over the Target and Baseline rows plus the ESS map.
    """
    e_baseline = np.asarray(e_baseline, dtype=float)
    e_target = np.asarray(e_target, dtype=float)
    w_all = odds_weights(e_baseline)

    # scan candidate thresholds (Target propensity quantiles, high→low); pick the
    # highest q whose Baseline ESS in {e≥q} has just fallen to the floor.
    qs = np.quantile(e_target, np.linspace(0.5, 0.99, 50))
    q_star = qs[-1]
    for q in qs:  # ascending: ESS in {e>=q} is non-increasing in q
        ess_q = _kish(w_all[e_baseline >= q])
        if ess_q <= ess_floor:
            q_star = q
            break

    in_r_b = e_baseline >= q_star
    in_r_t = e_target >= q_star
    return {
        "q_star": float(q_star),
        "in_R_target": in_r_t,
        "in_R_baseline": in_r_b,
        "ess_baseline_global": _kish(w_all),
        "ess_baseline_R": _kish(w_all[in_r_b]),
        "n_baseline_R": int(in_r_b.sum()),
        "n_target_R": int(in_r_t.sum()),
    }
