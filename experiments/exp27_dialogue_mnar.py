"""Exp 27: MNAR turn-missingness in dialogue (#47), exp13 sibling.

Sweeps missingness mechanism × severity over user-simulator trajectories; reports
how biased a trajectory-reward estimand is under naive / IPW-on-observables /
proxy-corrected estimators. MNAR is uncorrectable by IPW-on-observables and only
partially recovered by an observable proxy for the latent state.
"""
from pathlib import Path

import pandas as pd

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.dgp.dialogue_missingness import apply_turn_missingness
from causal_bench.estimators.reward_missingness import (
    true_reward, naive_reward, ipw_reward, proxy_reward,
)

OUT_DIR = Path("results/exp27_dialogue_mnar")


def _base(seed: int) -> pd.DataFrame:
    cfg = UserSimConfig(n_trajectories=800, n_turns=10, shock_rate=0.0, emit_noise_sd=0.1)
    return generate_user_sim_trajectories(cfg, seed=seed)


def _mechanism_severity(mechanism: str, intensity: float) -> float:
    """Map an abstract sweep intensity to a mechanism-appropriate parameter.

    MCAR's severity IS a drop probability, so cap it in a non-degenerate range;
    MAR/MNAR's severity is a logistic slope, used directly.
    """
    if mechanism == "mcar":
        return min(0.15 * intensity, 0.6)
    return intensity


def run_missingness_sweep(mechanisms, severities, seed: int = 12) -> pd.DataFrame:
    base = _base(seed)
    rows = []
    for mech in mechanisms:
        for sev in severities:
            param = _mechanism_severity(mech, float(sev))
            d = apply_turn_missingness(base, mech, param, seed=seed + 1, proxy_noise_sd=0.3)
            # observable prior footprint — the MAR-driving covariate for IPW
            u_prev = d.groupby("trajectory_id")["u"].shift(1)
            d["u_prev"] = u_prev.fillna(u_prev.mean())
            t = true_reward(d)
            rows.append({
                "mechanism": mech, "severity": float(sev),
                "naive_bias": naive_reward(d) - t,
                "ipw_bias": ipw_reward(d, ["u_prev"]) - t,
                "proxy_bias": proxy_reward(d, "z_proxy") - t,
            })
    return pd.DataFrame(rows)


def run(seed: int = 12):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tbl = run_missingness_sweep(["mcar", "mar", "mnar"], [1.0, 2.0, 3.0], seed)
    tbl.to_parquet(OUT_DIR / "reward_bias.parquet", index=False)
    print(tbl.to_string(index=False))
    return tbl


if __name__ == "__main__":
    run()
