"""Reward estimands under dialogue turn-missingness (#47).

Trajectory reward = mean per-turn utility u. Under MCAR the observed-only mean is
unbiased; under MAR it is biased but IPW-on-observables corrects it; under MNAR
(missingness driven by the latent state) IPW-on-observables cannot correct, and a
noisy latent proxy only partially recovers it.
"""
from __future__ import annotations

import pandas as pd


def true_reward(df: pd.DataFrame) -> float:
    """Complete-data reward: mean u over ALL turns."""
    return float(df["u"].mean())


def naive_reward(df: pd.DataFrame) -> float:
    """Observed-only reward: mean u over observed turns."""
    return float(df.loc[df["observed"], "u"].mean())
