"""Tests for the dialogue turn-missingness layer (#47), exp13 sibling."""
import numpy as np
import pandas as pd

from causal_bench.dgp.dialogue_missingness import apply_turn_missingness


def _traj(n_traj=200, n_turns=10, seed=0):
    """Small hand-built trajectory frame (self-contained; no #46 dependency)."""
    rng = np.random.default_rng(seed)
    rows = []
    for j in range(n_traj):
        z = rng.normal(0, 1)
        for t in range(n_turns):
            a = rng.normal(0, 1)
            z = z + 0.2 * np.tanh(a)
            rows.append({"trajectory_id": j, "t": t, "z": z, "u": 1 / (1 + np.exp(-z)), "a": a})
    return pd.DataFrame(rows)


def test_mcar_drops_independent_fraction_and_sets_dt():
    df = apply_turn_missingness(_traj(), mechanism="mcar", severity=0.3, seed=1)
    assert {"observed", "dt", "z_proxy"}.issubset(df.columns)
    frac = 1 - df["observed"].mean()
    assert abs(frac - 0.3) < 0.03                      # ~severity dropped
    assert df["dt"].min() >= 1                          # turn-lapse ≥ 1
    corr = np.corrcoef(df["z"], df["observed"].astype(float))[0, 1]
    assert abs(corr) < 0.05                            # MCAR: drop ⊥ latent state


def test_mar_depends_on_observable_not_latent_given_it():
    df = apply_turn_missingness(_traj(seed=2), mechanism="mar", severity=2.0, seed=3)
    df["a_prev_abs"] = df.groupby("trajectory_id")["a"].shift(1).abs()
    sub = df.dropna(subset=["a_prev_abs"])
    c_obs = np.corrcoef(sub["a_prev_abs"], (~sub["observed"]).astype(float))[0, 1]
    assert c_obs > 0.1                                 # missingness tracks observable |a_prev|
