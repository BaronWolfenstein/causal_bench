"""Adaptation metrics for the Q2 three-arm contrast (#46).

Post-shock tracking error and time-to-recover per arm, and the headline
marginal-capture ratio (design spec §4): the fraction of achievable adaptation
(naive → oracle) that the imperfect detector's flag captures. "Flag beats
naive" alone is near-tautological and is NOT the reported result.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _turns_since_last_shock(e: np.ndarray) -> np.ndarray:
    """For each turn, turns elapsed since the most recent shock fired (the shock
    at turn i moves z entering turn i+1). NaN before any shock."""
    out = np.full(len(e), np.nan)
    last = None
    for i, e_i in enumerate(e):
        if last is not None:
            out[i] = i - last
        if e_i == 1:
            last = i
    return out


def tracking_metrics(filtered_df: pd.DataFrame, window: int = 4) -> dict:
    """Post-shock tracking error, quiet-step error, and mean time-to-recover.

    post_shock_err: mean |ẑ−z| over turns 1..window after each shock.
    quiet_err: mean |ẑ−z| over all other turns.
    time_to_recover: per shock, first k in 1..window with |ẑ−z| ≤ 1.5·quiet_err
    (censored at window if never, or if the trajectory ends first).
    """
    d = filtered_df.sort_values(["trajectory_id", "t"]).copy()
    d["abs_err"] = (d["z_hat"] - d["z"]).abs()

    post_errs, quiet_errs = [], []
    for _, g in d.groupby("trajectory_id", sort=False):
        e = g["e"].to_numpy()
        ae = g["abs_err"].to_numpy()
        since = _turns_since_last_shock(e)
        in_window = (since >= 1) & (since <= window)
        post_errs.append(ae[in_window])
        quiet_errs.append(ae[~in_window])
    post = np.concatenate(post_errs) if post_errs else np.array([])
    quiet = np.concatenate(quiet_errs) if quiet_errs else np.array([])
    quiet_err = float(quiet.mean()) if len(quiet) else float("nan")

    tol = 1.5 * quiet_err
    recoveries = []
    for _, g in d.groupby("trajectory_id", sort=False):
        e = g["e"].to_numpy()
        ae = g["abs_err"].to_numpy()
        for i in np.flatnonzero(e == 1):
            rec = window
            for k in range(1, window + 1):
                if i + k < len(ae) and ae[i + k] <= tol:
                    rec = k
                    break
            recoveries.append(rec)

    return {
        "post_shock_err": float(post.mean()) if len(post) else float("nan"),
        "quiet_err": quiet_err,
        "time_to_recover": float(np.mean(recoveries)) if recoveries else float("nan"),
        "n_shock_turns": int(len(post)),
    }


def marginal_capture(err_naive: float, err_flag: float, err_oracle: float) -> float:
    """Headline (spec §4): fraction of achievable adaptation the detector captures,
    (naive − flag) / (naive − oracle). NaN when there is no achievable gap."""
    denom = err_naive - err_oracle
    if not np.isfinite(denom) or denom <= 1e-12:
        return float("nan")
    return float((err_naive - err_flag) / denom)
