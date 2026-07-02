"""Exp 26: exogenous-shock detection in a user simulator (#46).

Sweeps shock magnitude δ; reports how well a negative-control residual detects the
agent-unobservable e_t from its footprint (ROC/power).

Detection only (the post's Q1). The agent-adaptation contrast — endogenous-
continuation vs an NC-flag agent that re-plans on detection (Q2) — is NOT built
here; it is tracked as follow-up in #46. Detection here is also measured under a
near-direct negative control (n = z + small noise); the degraded-observability
regime that actually stresses the method is likewise tracked in #46.
"""
from pathlib import Path

import pandas as pd

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.detectors.exogenous import negative_control_residual
from causal_bench.detectors.metrics import detection_roc

OUT_DIR = Path("results/exp26_user_sim")


def run_detection_sweep(deltas, n_trajectories: int = 400, seed: int = 7) -> pd.DataFrame:
    rows = []
    for i, delta in enumerate(deltas):
        cfg = UserSimConfig(n_trajectories=n_trajectories, n_turns=8, shock_rate=0.15,
                            shock_delta=float(delta), nc_noise_sd=0.3, gamma_action=0.3)
        d = generate_user_sim_trajectories(cfg, seed=seed + i)
        scored = negative_control_residual(d)
        e_prev = (d.sort_values(["trajectory_id", "t"])
                    .groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy())
        roc = detection_roc(scored, e_prev)
        rows.append({"shock_delta": float(delta), **roc})
    return pd.DataFrame(rows)


def run_observability_sweep(couplings, shock_delta: float = 2.0,
                            n_trajectories: int = 400, seed: int = 7) -> pd.DataFrame:
    """At fixed δ, sweep the negative control's coupling to the latent state.

    The v1 control (coupling≈1) is a near-direct latent sensor, so detection
    saturates; this curve shows how detection degrades as the control becomes a
    weaker, more indirect signal — the regime that transfers to the real problem.
    """
    rows = []
    for i, coupling in enumerate(couplings):
        cfg = UserSimConfig(n_trajectories=n_trajectories, n_turns=8, shock_rate=0.15,
                            shock_delta=float(shock_delta), nc_noise_sd=0.3,
                            nc_coupling=float(coupling), gamma_action=0.3)
        d = generate_user_sim_trajectories(cfg, seed=seed + i)
        scored = negative_control_residual(d)
        e_prev = (d.sort_values(["trajectory_id", "t"])
                    .groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy())
        roc = detection_roc(scored, e_prev)
        rows.append({"nc_coupling": float(coupling), "shock_delta": float(shock_delta), **roc})
    return pd.DataFrame(rows)


def run(n_trajectories: int = 400, seed: int = 7):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = run_detection_sweep([0.0, 0.5, 1.0, 2.0, 3.0, 4.0], n_trajectories, seed)
    tbl.to_parquet(OUT_DIR / "detection_sweep.parquet", index=False)
    print(tbl.to_string(index=False))
    return tbl


if __name__ == "__main__":
    run()
