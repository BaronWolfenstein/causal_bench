"""Exp 28: Q2 three-arm adaptation contrast (#46; design spec 2026-07-02).

Given the eₜ detector (exp26/Q1), how much does acting on its flag help the
agent's belief re-track the user's latent state after an unobserved shock?
Three arms sharing one EKF belief filter over the footprint (u, a):

- naive: self-correcting filter, no flag (NOT a strawman — it measurement-
  updates on u every turn, so it partially recovers on its own);
- nc_flag: same filter; on |nc_residual| > c it inflates belief variance so
  the next emission dominates (c calibrated at a target FPR on a separate
  draw — the tie-back to exp26's ROC);
- oracle: same filter conditioned on the true shock indicator — the ceiling.

Headline: marginal capture (naive − nc_flag)/(naive − oracle), and its
degradation as the negative control weakens (nc_coupling ↓). "Flag beats
naive" alone is near-tautological and is not the reported result.

v1 limit (spec §5): this measures belief-tracking (A1+B1), a proxy for
adaptation, not task outcome; the act-on-belief reward loop (B2) is deferred.
"""
from pathlib import Path

import pandas as pd

from causal_bench.adaptation.filters import nc_flags, oracle_flags, run_belief_filter
from causal_bench.adaptation.metrics import marginal_capture, tracking_metrics
from causal_bench.detectors.exogenous import negative_control_residual
from causal_bench.detectors.metrics import threshold_at_fpr
from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories

OUT_DIR = Path("results/exp28_q2_adaptation")


def _make_cfg(shock_delta, nc_coupling, n_trajectories, n_turns):
    # shock_rate=0.08, not exp26's 0.15: shocks are same-sign, so at 0.15 over 12
    # turns z accumulates into the sigmoid emission's saturated range, where u is
    # uninformative about magnitude — no arm can re-track there and the achievable
    # gap (naive − oracle) collapses. 0.08 keeps post-shock turns mostly in the
    # informative range; detection itself is unaffected (the NC is linear in z).
    return UserSimConfig(n_trajectories=n_trajectories, n_turns=n_turns,
                         shock_rate=0.08, shock_delta=float(shock_delta),
                         nc_noise_sd=0.3, nc_coupling=float(nc_coupling),
                         gamma_action=0.3)


def calibrate_threshold(cfg: UserSimConfig, seed: int, target_fpr: float = 0.1) -> float:
    """Detection cutoff at target FPR from a separate calibration draw (Q1 tie-back)."""
    d = generate_user_sim_trajectories(cfg, seed=seed)
    scored = negative_control_residual(d)
    e_prev = (d.sort_values(["trajectory_id", "t"])
                .groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy())
    return threshold_at_fpr(scored, e_prev, target_fpr=target_fpr)


def run_three_arm(shock_delta: float = 2.0, nc_coupling: float = 1.0,
                  n_trajectories: int = 400, n_turns: int = 12, seed: int = 11,
                  target_fpr: float = 0.1, window: int = 4) -> pd.DataFrame:
    cfg = _make_cfg(shock_delta, nc_coupling, n_trajectories, n_turns)
    c = calibrate_threshold(cfg, seed=seed + 1000, target_fpr=target_fpr)
    d = generate_user_sim_trajectories(cfg, seed=seed)
    kw = dict(gamma=cfg.gamma_action, beta_emit=cfg.beta_emit,
              emit_noise_sd=cfg.emit_noise_sd, z0_mean=cfg.z0_mean, z0_sd=cfg.z0_sd)
    arms = {"naive": None, "nc_flag": nc_flags(d, threshold=c), "oracle": oracle_flags(d)}
    rows = []
    for arm, fl in arms.items():
        m = tracking_metrics(run_belief_filter(d, flag=fl, **kw), window=window)
        rows.append({"arm": arm, "shock_delta": float(shock_delta),
                     "nc_coupling": float(nc_coupling), "threshold": c, **m})
    tbl = pd.DataFrame(rows)
    err = tbl.set_index("arm")["post_shock_err"]
    tbl["capture"] = marginal_capture(err["naive"], err["nc_flag"], err["oracle"])
    return tbl


def run_capture_vs_observability(couplings, shock_delta: float = 2.0,
                                 n_trajectories: int = 400, n_turns: int = 12,
                                 seed: int = 11, target_fpr: float = 0.1,
                                 window: int = 4) -> pd.DataFrame:
    tables = [run_three_arm(shock_delta, coupling, n_trajectories, n_turns,
                            seed=seed + i, target_fpr=target_fpr, window=window)
              for i, coupling in enumerate(couplings)]
    return pd.concat(tables, ignore_index=True)


def run(n_trajectories: int = 400, seed: int = 11):
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    three = run_three_arm(n_trajectories=n_trajectories, seed=seed)
    three.to_parquet(OUT_DIR / "three_arm.parquet", index=False)
    print(three.to_string(index=False))
    sweep = run_capture_vs_observability([1.0, 0.7, 0.5, 0.3, 0.1],
                                         n_trajectories=n_trajectories, seed=seed)
    sweep.to_parquet(OUT_DIR / "capture_vs_observability.parquet", index=False)
    print(sweep[["nc_coupling", "arm", "post_shock_err", "time_to_recover", "capture"]]
          .to_string(index=False))
    return three, sweep


if __name__ == "__main__":
    run()
