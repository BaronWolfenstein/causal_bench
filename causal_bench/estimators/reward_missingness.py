"""Reward estimands under dialogue turn-missingness (#47).

Trajectory reward = mean per-turn utility u. Under MCAR the observed-only mean is
unbiased; under MAR it is biased but IPW-on-observables corrects it; under MNAR
(missingness driven by the latent state) IPW-on-observables cannot correct, and a
noisy latent proxy only partially recovers it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def true_reward(df: pd.DataFrame) -> float:
    """Complete-data reward: mean u over ALL turns."""
    return float(df["u"].mean())


def naive_reward(df: pd.DataFrame) -> float:
    """Observed-only reward: mean u over observed turns."""
    return float(df.loc[df["observed"], "u"].mean())


def ipw_reward(df: pd.DataFrame, feature_cols: list[str]) -> float:
    """Hájek-normalized inverse-probability-of-observation weighted reward.

    P(observed | features) from logistic regression on ``feature_cols``. With
    observable features this corrects MAR; it cannot correct MNAR (the features do
    not include the latent state driving the missingness)."""
    X = df[feature_cols].to_numpy()
    y = df["observed"].astype(int).to_numpy()
    p = LogisticRegression(max_iter=1000).fit(X, y).predict_proba(X)[:, 1]
    obs = df["observed"].to_numpy()
    w = np.where(obs, 1.0 / np.clip(p, 1e-3, 1.0), 0.0)
    u = df["u"].to_numpy()
    return float(np.sum(w * u) / np.sum(w))


def proxy_reward(df: pd.DataFrame, proxy_col: str = "z_proxy") -> float:
    """IPW using a noisy proxy for the latent state in the propensity model.

    Recovers *part* of the MNAR bias, in proportion to proxy quality; a nonzero
    residual remains (the honest endpoint — no proxy fully corrects MNAR)."""
    return ipw_reward(df, [proxy_col])
