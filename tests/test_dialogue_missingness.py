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
