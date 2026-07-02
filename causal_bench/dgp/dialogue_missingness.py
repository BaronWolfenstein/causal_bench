"""Dialogue turn-missingness layer (#47) — exp13's censoring grid for dialogue.

Annotates any long-format trajectory frame (columns trajectory_id, t, z, u, a) with
an ``observed`` mask, a turn-lapse ``dt``, and a noisy latent proxy ``z_proxy``,
under one of three missingness mechanisms (MCAR / MAR / MNAR). ``z`` is the hidden
latent state: mechanisms may read it, but observation-model estimators may not.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _miss_prob(df: pd.DataFrame, mechanism: str, severity: float) -> np.ndarray:
    if mechanism == "mcar":
        return np.full(len(df), severity)
    if mechanism == "mar":
        # Observable-driven: the prior turn's footprint u_{t-1} (MAR — IPW-correctable).
        # u_prev is observed AND correlates with the reward (via z autocorrelation), so
        # it induces a real, correctable reward bias — unlike an observable orthogonal to u.
        u_prev = df.groupby("trajectory_id")["u"].shift(1)
        x = u_prev.fillna(u_prev.mean()).to_numpy()
        return 1.0 / (1.0 + np.exp(-severity * (x - np.nanmean(x))))
    if mechanism == "mnar":
        # Latent-driven: low z (frustrated) drops more; not a function of observables.
        z = df["z"].to_numpy()
        return 1.0 / (1.0 + np.exp(-severity * (-(z - z.mean()))))
    raise ValueError(f"unknown mechanism {mechanism!r}")


def apply_turn_missingness(traj_df: pd.DataFrame, mechanism: str, severity: float,
                           seed: int, proxy_noise_sd: float = 0.0) -> pd.DataFrame:
    """Annotate turns with ``observed`` / ``dt`` / ``z_proxy`` under a mechanism.

    - MCAR: ``P(miss) = severity``, i.i.d.
    - ``dt``: turns since the previous observed turn within a trajectory (≥1).
    - ``z_proxy``: ``z + N(0, proxy_noise_sd)`` — a noisy observable stand-in for
      the latent state, used by the proxy-corrected estimator.
    """
    rng = np.random.default_rng(seed)
    df = traj_df.sort_values(["trajectory_id", "t"]).copy()
    p = np.clip(_miss_prob(df, mechanism, severity), 0.0, 1.0)
    df["observed"] = rng.random(len(df)) >= p
    df["z_proxy"] = df["z"] + rng.normal(0.0, proxy_noise_sd, len(df))

    dt = []
    for _, g in df.groupby("trajectory_id"):
        gap, out = 0, []
        for obs in g["observed"]:
            gap += 1
            out.append(gap)
            if obs:
                gap = 0
        dt.extend(out)
    df["dt"] = dt
    return df
