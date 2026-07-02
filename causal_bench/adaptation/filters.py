"""Belief filters for Q2 adaptation (#46) — the agent-side model the DGP lacks.

Information structure (design spec §1): the filter consumes ONLY the
agent-observable footprint (u, a) plus an optional per-turn shock flag. The
negative control n and the ground truth z, e are never read here — shift
information reaches the belief only via the flag channel. The three arms share
this one filter and differ only in what the flag is: nothing (naive), the NC
detector's flag, or the true shock indicator (oracle ceiling).

The filter is an EKF on the sigmoid emission u = σ(β·z) + ε. It is given the
true model parameters (gamma, beta_emit, emit_noise_sd) — model knowledge, not
shift information: it has no shock term, so an exogenous jump in z is exactly
what it cannot predict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns the belief filter may read — the agent-observable footprint only.
_OBSERVABLE = ["trajectory_id", "t", "u", "a"]


def run_belief_filter(
    traj_df: pd.DataFrame,
    *,
    gamma: float,
    beta_emit: float = 1.0,
    emit_noise_sd: float = 0.2,
    z0_mean: float = 0.0,
    z0_sd: float = 1.0,
    q_process: float = 0.05,
    inflate_var: float = 4.0,
    flag=None,
) -> pd.DataFrame:
    """Filter ẑ_t from the footprint; returns rows sorted by (trajectory_id, t)
    with added columns ``z_hat`` and ``z_hat_var``.

    Per turn: predict ẑ⁻ = ẑ + γ·tanh(a_prev), P⁻ = P + q_process; if the
    aligned ``flag`` entry is truthy, admit an exogenous jump by adding
    ``inflate_var`` to P⁻ so the next emission dominates the update; then an
    EKF measurement update on u_t under h(z) = σ(β·z).
    """
    df = traj_df.sort_values(["trajectory_id", "t"]).reset_index(drop=True)
    obs = df[_OBSERVABLE]
    flag_arr = (np.zeros(len(df), dtype=bool) if flag is None
                else np.asarray(flag, dtype=bool))
    if len(flag_arr) != len(df):
        raise ValueError("flag must align with traj_df rows")
    R = emit_noise_sd**2
    z_hats = np.empty(len(df))
    p_vars = np.empty(len(df))
    i = 0
    for _, g in obs.groupby("trajectory_id", sort=False):
        z_hat, P = z0_mean, z0_sd**2
        a_prev = None
        for u_t, a_t in zip(g["u"], g["a"]):
            if a_prev is not None:
                z_hat = z_hat + gamma * np.tanh(a_prev)
                P = P + q_process
            if flag_arr[i]:
                P = P + inflate_var
            s = 1.0 / (1.0 + np.exp(-beta_emit * z_hat))
            H = beta_emit * s * (1.0 - s)
            S = H * H * P + R
            K = P * H / S
            z_hat = z_hat + K * (u_t - s)
            P = (1.0 - K * H) * P
            z_hats[i], p_vars[i] = z_hat, P
            a_prev = a_t
            i += 1
    out = df.copy()
    out["z_hat"] = z_hats
    out["z_hat_var"] = p_vars
    return out


def oracle_flags(traj_df: pd.DataFrame) -> np.ndarray:
    """Ceiling arm's flag: the TRUE shock indicator, shifted to the turn where the
    jumped latent state is first observable. Reads ground-truth ``e`` — permitted
    only here, for the oracle ceiling (design spec §4)."""
    d = traj_df.sort_values(["trajectory_id", "t"])
    return (d.groupby("trajectory_id")["e"].shift(1).fillna(0) == 1).to_numpy()


def nc_flags(traj_df: pd.DataFrame, threshold: float) -> np.ndarray:
    """Detector arm's flag: |nc_residual| > threshold, aligned to sorted rows.
    NaN residuals (first turns) never flag."""
    from causal_bench.detectors.exogenous import negative_control_residual

    scored = negative_control_residual(traj_df)   # sorted by (trajectory_id, t)
    resid = scored["nc_residual"].abs().to_numpy()
    return np.nan_to_num(resid, nan=-np.inf) > threshold
