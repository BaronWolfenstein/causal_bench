"""Tests for the reward-under-missingness estimators (#47)."""
import numpy as np

from causal_bench.dgp.dialogue_missingness import apply_turn_missingness
from causal_bench.estimators.reward_missingness import (
    true_reward, naive_reward, ipw_reward,
)
from tests.test_dialogue_missingness import _traj


def _with_obs_features(df):
    df = df.copy()
    u_prev = df.groupby("trajectory_id")["u"].shift(1)
    df["u_prev"] = u_prev.fillna(u_prev.mean())
    return df


def test_mcar_naive_unbiased_mnar_biased():
    base = _traj(n_traj=600, seed=6)
    mcar = apply_turn_missingness(base, "mcar", severity=0.4, seed=7)
    mnar = apply_turn_missingness(base, "mnar", severity=2.5, seed=7)
    assert abs(naive_reward(mcar) - true_reward(mcar)) < 0.01      # MCAR: unbiased
    assert naive_reward(mnar) - true_reward(mnar) > 0.03           # MNAR: dropping low-u turns inflates reward


def test_ipw_corrects_mar_but_not_mnar():
    base = _traj(n_traj=800, seed=8)
    mar = _with_obs_features(apply_turn_missingness(base, "mar", severity=2.0, seed=9))
    mnar = _with_obs_features(apply_turn_missingness(base, "mnar", severity=2.5, seed=9))
    t_mar, t_mnar = true_reward(mar), true_reward(mnar)
    # MAR: IPW on the observable prior footprint closes most of the naive bias
    assert abs(ipw_reward(mar, ["u_prev"]) - t_mar) < abs(naive_reward(mar) - t_mar) * 0.5
    # MNAR: IPW on observables does NOT close the bias (still substantial)
    assert abs(ipw_reward(mnar, ["u_prev"]) - t_mnar) > 0.02
