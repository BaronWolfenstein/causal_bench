"""Tests for the Q2 three-arm belief filter and tracking metrics (#46).

Arms differ only in the shock-flag channel: naive (no flag), NC-flag (detector),
oracle (true e). The filter reads only the agent-observable footprint (u, a).
"""
import numpy as np

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories
from causal_bench.adaptation.filters import run_belief_filter


def _filter_kwargs(cfg):
    return dict(gamma=cfg.gamma_action, beta_emit=cfg.beta_emit,
                emit_noise_sd=cfg.emit_noise_sd, z0_mean=cfg.z0_mean, z0_sd=cfg.z0_sd)


def test_filter_output_shape_and_columns():
    cfg = UserSimConfig(n_trajectories=5, n_turns=6)
    d = generate_user_sim_trajectories(cfg, seed=0)
    f = run_belief_filter(d, **_filter_kwargs(cfg))
    assert len(f) == len(d)
    assert {"z_hat", "z_hat_var"} <= set(f.columns)
    assert (f["z_hat_var"] > 0).all()
    assert f.equals(f.sort_values(["trajectory_id", "t"]).reset_index(drop=True))


def test_filter_tracks_latent_state_without_shocks():
    cfg = UserSimConfig(n_trajectories=300, n_turns=12, shock_rate=0.0,
                        emit_noise_sd=0.2, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=1)
    f = run_belief_filter(d, **_filter_kwargs(cfg))
    late = f[f["t"] >= 4]
    err = (late["z_hat"] - late["z"]).abs().mean()
    prior_err = (late["z"] - cfg.z0_mean).abs().mean()
    assert err < 0.6 * prior_err    # far better than never updating the prior


def test_naive_filter_partially_self_corrects_after_shock():
    """Anti-strawman check (spec §4): naive measurement-updates on u, so its
    post-shock error DECREASES over the turns after a shock."""
    cfg = UserSimConfig(n_trajectories=500, n_turns=12, shock_rate=0.08,
                        shock_delta=2.0, emit_noise_sd=0.2, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=2)
    f = run_belief_filter(d, **_filter_kwargs(cfg))
    f = f.copy()
    f["abs_err"] = (f["z_hat"] - f["z"]).abs()
    # turns since the most recent shock, per trajectory
    errs_by_k = {}
    for _, g in f.groupby("trajectory_id", sort=False):
        e = g["e"].to_numpy()
        ae = g["abs_err"].to_numpy()
        last = None
        for i in range(len(e)):
            if last is not None:
                k = i - last
                errs_by_k.setdefault(k, []).append(ae[i])
            if e[i] == 1:
                last = i
    assert np.mean(errs_by_k[4]) < 0.8 * np.mean(errs_by_k[1])
