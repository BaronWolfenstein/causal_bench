"""Exogenous-shift detection from a user simulator's footprint (#46).

The agent cannot see the exogenous shock e_t; it only sees the behavioral footprint.
The negative control n_t has zero action-effect by construction, so its one-step
prediction under the no-shock model is exact absent a shock — a large residual flags
that the latent state moved for a reason the agent did not cause.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Columns a detector is allowed to read — the agent-observable footprint only.
_OBSERVABLE = ["trajectory_id", "t", "u", "a", "n"]


def negative_control_residual(traj_df: pd.DataFrame) -> pd.DataFrame:
    """Per-turn residual of the negative control vs its no-shock one-step prediction.

    Uses only agent-observable columns. `n̂_t = n_{t-1} + γ̂·tanh(a_{t-1})`, where γ̂
    is the endogenous NC drift estimated by OLS of Δn on tanh(a_{t-1}). A large
    `|nc_residual|` flags an exogenous shift (the first turn of each trajectory has
    no predecessor → NaN residual).
    """
    df = traj_df[_OBSERVABLE].sort_values(["trajectory_id", "t"]).copy()
    df["n_prev"] = df.groupby("trajectory_id")["n"].shift(1)
    df["a_prev"] = df.groupby("trajectory_id")["a"].shift(1)

    step = df.dropna(subset=["n_prev", "a_prev"])
    dn = (step["n"] - step["n_prev"]).to_numpy()
    x = np.tanh(step["a_prev"].to_numpy())
    gamma_hat = float(np.polyfit(x, dn, 1)[0])

    pred = df["n_prev"] + gamma_hat * np.tanh(df["a_prev"])
    df["nc_residual"] = df["n"] - pred
    return df
