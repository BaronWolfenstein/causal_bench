"""Tests for the reward-under-missingness estimators (#47)."""
import numpy as np

from causal_bench.dgp.dialogue_missingness import apply_turn_missingness
from causal_bench.estimators.reward_missingness import true_reward, naive_reward
from tests.test_dialogue_missingness import _traj


def test_mcar_naive_unbiased_mnar_biased():
    base = _traj(n_traj=600, seed=6)
    mcar = apply_turn_missingness(base, "mcar", severity=0.4, seed=7)
    mnar = apply_turn_missingness(base, "mnar", severity=2.5, seed=7)
    assert abs(naive_reward(mcar) - true_reward(mcar)) < 0.01      # MCAR: unbiased
    assert naive_reward(mnar) - true_reward(mnar) > 0.03           # MNAR: dropping low-u turns inflates reward
