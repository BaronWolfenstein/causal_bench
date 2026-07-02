"""Tests for the negative-control exogenous-shift detector (#46)."""
import numpy as np

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.detectors.exogenous import negative_control_residual


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
