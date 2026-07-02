"""Tests for the user-simulator trajectory DGP (#46).

A sequential latent-state simulator: latent z_t emits an agent-visible footprint
u_t and a zero-action-effect negative control n_t, evolves endogenously under the
agent action a_t, and is perturbed by an exogenous, agent-unobservable shock e_t
that enters only the transition.
"""
from causal_bench.dgp.user_sim import UserSimConfig, generate_user_sim_trajectories


def test_trajectory_shape_and_columns():
    cfg = UserSimConfig(n_trajectories=10, n_turns=5)
    df = generate_user_sim_trajectories(cfg, seed=0)
    assert set(df.columns) == {"trajectory_id", "t", "z", "u", "a", "n", "e"}
    assert len(df) == 10 * 5
    assert df["t"].min() == 0 and df["t"].max() == 4
    assert df["trajectory_id"].nunique() == 10
