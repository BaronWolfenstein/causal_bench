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


def test_oracle_and_nc_flags_align_and_fire():
    from causal_bench.adaptation.filters import oracle_flags, nc_flags
    from causal_bench.detectors.exogenous import negative_control_residual
    from causal_bench.detectors.metrics import threshold_at_fpr
    cfg = UserSimConfig(n_trajectories=200, n_turns=8, shock_rate=0.15,
                        shock_delta=2.0, nc_noise_sd=0.3, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=3)
    ds = d.sort_values(["trajectory_id", "t"]).reset_index(drop=True)

    ofl = oracle_flags(d)
    expected = (ds.groupby("trajectory_id")["e"].shift(1).fillna(0) == 1).to_numpy()
    assert ofl.dtype == bool and (ofl == expected).all()

    scored = negative_control_residual(d)
    e_prev = ds.groupby("trajectory_id")["e"].shift(1).fillna(0).to_numpy()
    c = threshold_at_fpr(scored, e_prev, target_fpr=0.1)
    nfl = nc_flags(d, threshold=c)
    assert nfl.dtype == bool and len(nfl) == len(d)
    assert not nfl[ds["t"] == 0].any()          # NaN residual on first turns → no flag
    # detector flags fire mostly where the oracle does (δ=2 is well-detectable)
    assert nfl[ofl].mean() > 0.5
    assert nfl[~ofl].mean() < 0.15


def test_oracle_arm_beats_naive_post_shock():
    from causal_bench.adaptation.filters import oracle_flags
    cfg = UserSimConfig(n_trajectories=500, n_turns=12, shock_rate=0.08,
                        shock_delta=2.0, emit_noise_sd=0.2, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=4)
    kw = _filter_kwargs(cfg)
    f_naive = run_belief_filter(d, **kw)
    f_oracle = run_belief_filter(d, flag=oracle_flags(d), **kw)

    def post_shock_err(f):
        f = f.copy()
        f["abs_err"] = (f["z_hat"] - f["z"]).abs()
        e_prev = f.groupby("trajectory_id")["e"].shift(1).fillna(0)
        return f.loc[e_prev == 1, "abs_err"].mean()

    assert post_shock_err(f_oracle) < 0.75 * post_shock_err(f_naive)
