"""Tests for the user-simulator trajectory DGP (#46).

A sequential latent-state simulator: latent z_t emits an agent-visible footprint
u_t and a zero-action-effect negative control n_t, evolves endogenously under the
agent action a_t, and is perturbed by an exogenous, agent-unobservable shock e_t
that enters only the transition.
"""
import numpy as np

from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories


def test_trajectory_shape_and_columns():
    cfg = UserSimConfig(n_trajectories=10, n_turns=5)
    df = generate_user_sim_trajectories(cfg, seed=0)
    assert set(df.columns) == {"trajectory_id", "t", "z", "u", "a", "n", "e"}
    assert len(df) == 10 * 5
    assert df["t"].min() == 0 and df["t"].max() == 4
    assert df["trajectory_id"].nunique() == 10


def test_shock_enters_only_transition_and_is_exogenous():
    # δ=0: z evolves by the endogenous rule alone (no jumps)
    cfg0 = UserSimConfig(n_trajectories=1, n_turns=4, shock_rate=0.0, shock_delta=0.0,
                         emit_noise_sd=0.0, gamma_action=0.3)
    d0 = generate_user_sim_trajectories(cfg0, seed=1).sort_values("t").reset_index(drop=True)
    for t in range(3):
        expected = d0.loc[t, "z"] + 0.3 * np.tanh(d0.loc[t, "a"])
        assert d0.loc[t + 1, "z"] == expected  # endogenous only


def test_shock_shifts_next_state_by_delta():
    cfg = UserSimConfig(n_trajectories=200, n_turns=6, shock_rate=1.0, shock_delta=2.0,
                        emit_noise_sd=0.0, gamma_action=0.3)
    d = generate_user_sim_trajectories(cfg, seed=2)
    # with shock_rate=1 every step fires: z_{t+1} = z_t + 0.3 tanh(a_t) + 2.0
    one = d[d.trajectory_id == 0].sort_values("t").reset_index(drop=True)
    step = one.loc[1, "z"] - (one.loc[0, "z"] + 0.3 * np.tanh(one.loc[0, "a"]))
    assert abs(step - 2.0) < 1e-9
