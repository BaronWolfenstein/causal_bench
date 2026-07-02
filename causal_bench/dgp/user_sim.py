"""User-simulator trajectory DGP (#46) — simulator-validation workstream.

A sequential latent-state user simulator for instrumenting the Collinear "cigarette"
problem: a latent user state ``z_t`` emits an agent-visible footprint ``u_t`` and a
zero-action-effect negative control ``n_t``, evolves endogenously under the agent
action ``a_t``, and is perturbed by an exogenous, agent-unobservable shock ``e_t``
that enters ONLY the transition (``e_t`` ⊥ the agent's action).

Ground-truth columns ``z`` and ``e`` are emitted for scoring detectors but are the
hidden state a detector may not read (see ``detectors/exogenous.py``).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field


class UserSimConfig(BaseModel):
    model_config = {"extra": "forbid"}

    n_trajectories: int = Field(200, ge=1)
    n_turns: int = Field(20, ge=2)
    z0_mean: float = 0.0
    z0_sd: float = 1.0
    beta_emit: float = 1.0
    gamma_action: float = 0.3
    shock_rate: float = Field(0.1, ge=0.0, le=1.0)   # λ — exogenous shock frequency
    shock_delta: float = 0.0                          # δ — shock magnitude (swept)
    emit_noise_sd: float = Field(0.2, ge=0.0)
    nc_noise_sd: float = Field(0.2, ge=0.0)
    nc_coupling: float = Field(1.0, ge=0.0)   # how strongly the neg. control reads z; <1 = weaker/indirect


def generate_user_sim_trajectories(config: UserSimConfig, seed: int) -> pd.DataFrame:
    """Generate ``n_trajectories`` sequential trajectories of length ``n_turns``.

    Returns a long-format DataFrame, one row per (trajectory, turn), with columns
    ``[trajectory_id, t, z, u, a, n, e]``.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for traj in range(config.n_trajectories):
        z = rng.normal(config.z0_mean, config.z0_sd)
        for t in range(config.n_turns):
            a = float(rng.normal(0.0, 1.0))            # agent action (placeholder policy)
            u = float(
                1.0 / (1.0 + np.exp(-config.beta_emit * z))
                + rng.normal(0.0, config.emit_noise_sd)
            )
            e = int(rng.random() < config.shock_rate)
            # Negative control: driven by the latent state, with NO term in a_t —
            # the agent's action has zero causal path to n, so any shift in n is exogenous.
            n = float(config.nc_coupling * z + rng.normal(0.0, config.nc_noise_sd))
            rows.append(
                {"trajectory_id": traj, "t": t, "z": z, "u": u, "a": a, "n": n, "e": e}
            )
            # Endogenous transition + exogenous shock (shock enters only here).
            z = z + config.gamma_action * np.tanh(a) + (config.shock_delta if e else 0.0)
    return pd.DataFrame(rows)
