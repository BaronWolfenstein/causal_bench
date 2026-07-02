"""Tests for the negative-control exogenous-shift detector (#46)."""
import numpy as np

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.detectors.exogenous import negative_control_residual
from causal_bench.detectors.metrics import detection_roc


def test_residual_spikes_at_shock_turns():
    cfg = UserSimConfig(n_trajectories=300, n_turns=8, shock_rate=0.15, shock_delta=3.0,
                        nc_noise_sd=0.1, emit_noise_sd=0.1, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=5)
    scored = negative_control_residual(d)
    # residual magnitude is larger on the step AFTER a shock (z jumped) than on quiet steps
    d_shift = d.copy()
    d_shift["e_prev"] = d_shift.groupby("trajectory_id")["e"].shift(1).fillna(0)
    merged = scored.assign(e_prev=d_shift["e_prev"].values)
    post_shock = merged.loc[merged.e_prev == 1, "nc_residual"].abs().mean()
    quiet = merged.loc[merged.e_prev == 0, "nc_residual"].abs().mean()
    assert post_shock > 2 * quiet


def test_auc_increases_with_shock_magnitude():
    def auc_for(delta):
        cfg = UserSimConfig(n_trajectories=400, n_turns=8, shock_rate=0.15,
                            shock_delta=delta, nc_noise_sd=0.3, gamma_action=0.3)
        d = generate_user_sim_trajectories(cfg, seed=6)
        scored = negative_control_residual(d)
        e_prev = (d.sort_values(["trajectory_id", "t"])
                    .groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy())
        return detection_roc(scored, e_prev)["auc"]
    assert auc_for(0.5) < auc_for(3.0)          # bigger shocks are easier to detect
    assert auc_for(3.0) > 0.75                   # large shocks are clearly detectable


def test_sweep_returns_monotone_auc_table():
    from experiments.exp26_user_sim_detection import run_detection_sweep
    tbl = run_detection_sweep(deltas=[0.0, 1.0, 3.0], n_trajectories=300, seed=7)
    assert list(tbl["shock_delta"]) == [0.0, 1.0, 3.0]
    # δ=0 → no signal (AUC ~0.5 or NaN); δ=3 → strong
    assert tbl.loc[tbl.shock_delta == 3.0, "auc"].iloc[0] > 0.75


def test_detection_degrades_with_observability():
    """At fixed δ, weakening the negative control's coupling to the latent state
    lowers detection AUC — gracefully, not a cliff. This is the transferable result:
    detection under a realistic (weak, indirect) control, not a near-direct sensor."""
    from experiments.exp26_user_sim_detection import run_observability_sweep
    tbl = run_observability_sweep(couplings=[1.0, 0.5, 0.2], shock_delta=2.0,
                                  n_trajectories=400, seed=7)
    assert list(tbl["nc_coupling"]) == [1.0, 0.5, 0.2]
    aucs = tbl.set_index("nc_coupling")["auc"]
    assert aucs[1.0] > aucs[0.2]        # weaker signal → lower AUC
    assert aucs[1.0] > 0.9              # near-direct control detects well
