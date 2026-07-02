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


def test_mar_depends_on_observable_prior_footprint():
    df = apply_turn_missingness(_traj(seed=2), mechanism="mar", severity=4.0, seed=3)
    df["u_prev"] = df.groupby("trajectory_id")["u"].shift(1)
    sub = df.dropna(subset=["u_prev"])
    c_obs = np.corrcoef(sub["u_prev"], sub["observed"].astype(float))[0, 1]
    assert c_obs < -0.1                                # higher prior footprint → more likely missing


def test_mnar_low_latent_state_drops_more():
    df = apply_turn_missingness(_traj(seed=4), mechanism="mnar", severity=2.0, seed=5)
    c = np.corrcoef(df["z"], (~df["observed"]).astype(float))[0, 1]
    assert c < -0.1                                    # lower z (frustrated) → more missing
