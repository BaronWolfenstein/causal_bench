"""Bias amplification (#174). Load-bearing: including the instrument amplifies the
residual unmeasured-confounding bias beyond both screening and no adjustment, and
the outcome-adaptive guard (screened on the treatment-free outcome model) drops
the instrument and recovers the safe bias."""
import numpy as np

from causal_bench.dgp.bias_amplification import (
    BiasAmpConfig, draw_bias_amplification, regression_adjustment_ate,
    outcome_adaptive_screen)
from experiments.exp40_bias_amplification import run


def test_including_instrument_amplifies_bias():
    r = run(n=2000, reps=150)
    ab = r["abs_bias"]
    # the whole point: adjusting for {X,Z} is WORSE than {X} (amplification)...
    assert ab["include {X,Z}"] > ab["screen {X}"] + 0.05
    # ...and worse even than adjusting for nothing
    assert ab["include {X,Z}"] > ab["none (crude)"]
    # residual unmeasured-confounding bias is present in every set (U is unmeasured)
    assert ab["screen {X}"] > 0.2


def test_outcome_adaptive_guard_recovers_screen():
    r = run(n=2000, reps=150)
    ab = r["abs_bias"]
    # the guard's bias tracks the screened set, not the amplified one
    assert abs(ab["outcome-adaptive"] - ab["screen {X}"]) < 0.1
    assert ab["outcome-adaptive"] < ab["include {X,Z}"]
    # and it drops the instrument in the large majority of reps
    kept_only_X = r["adaptive_kept"].get(("X",), 0)
    assert kept_only_X > 0.8 * sum(r["adaptive_kept"].values())


def test_guard_does_not_condition_on_treatment():
    """Direct check of the collider guard: screening WITH A conditioned in would
    keep Z (collider Z→A←U→Y); the treatment-free screen drops it."""
    df = draw_bias_amplification(4000, seed=0)
    kept = outcome_adaptive_screen(df, ["X", "Z"])
    assert "X" in kept and "Z" not in kept


def test_null_effect_estimate_is_pure_bias():
    """tau=0 → any nonzero estimate is bias; sanity that the DGP is a clean null."""
    df = draw_bias_amplification(5000, seed=1, config=BiasAmpConfig(tau=0.0))
    # oracle adjustment for BOTH confounders (X and the truly-unmeasured U is not
    # available) — here we confirm the crude estimate is nonzero (confounded)
    assert abs(regression_adjustment_ate(df, ["X"])) > 0.2
