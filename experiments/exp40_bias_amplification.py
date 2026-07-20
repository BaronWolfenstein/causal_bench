"""Exp 40: Bias amplification — conditioning on an instrument amplifies residual
unmeasured-confounding bias (causal_bench #174, ENCIRCLE).

The propensity/adjustment set is the knob. With an unmeasured confounder U and a
strong instrument Z (predicts treatment, not outcome), adjusting for {X, Z} leaves
a LARGER bias than adjusting for {X} alone — and larger even than adjusting for
NOTHING. "Adjust for everything you observed" is actively harmful when the
observed set contains instruments and confounding is unmeasured — the exact risk
of a naive embedding-propensity, whose high-dim frozen-encoder features are
instrument-rich. The `outcome_adaptive_screen` guard recovers the safe set.

Collider caveat (baked into the guard): the screen must use the covariate–outcome
association, NOT condition on treatment — conditioning on A opens Z→A←U→Y and would
keep the instrument. Run: `PYTHONPATH=. python experiments/exp40_bias_amplification.py`.
"""
from __future__ import annotations

import numpy as np

from causal_bench.dgp.bias_amplification import (
    BiasAmpConfig, draw_bias_amplification, regression_adjustment_ate,
    outcome_adaptive_screen, true_tau)

ADJUSTMENT_SETS = {
    "none (crude)": [],
    "screen {X}": ["X"],
    "include {X,Z}": ["X", "Z"],
}


def run(n: int = 2000, reps: int = 200, config: BiasAmpConfig = BiasAmpConfig()) -> dict:
    """Mean |bias| of the ATE per adjustment set + the outcome-adaptive guard."""
    tau = true_tau(config)
    bias = {k: [] for k in ADJUSTMENT_SETS}
    bias["outcome-adaptive"] = []
    kept: dict = {}
    for seed in range(reps):
        df = draw_bias_amplification(n, seed, config)
        for name, cols in ADJUSTMENT_SETS.items():
            bias[name].append(regression_adjustment_ate(df, cols) - tau)
        keep = outcome_adaptive_screen(df, ["X", "Z"])
        kept[tuple(keep)] = kept.get(tuple(keep), 0) + 1
        bias["outcome-adaptive"].append(regression_adjustment_ate(df, keep) - tau)
    return {"abs_bias": {k: float(abs(np.mean(v))) for k, v in bias.items()},
            "adaptive_kept": kept, "tau": tau}


def main() -> None:
    r = run()
    print(f"Bias amplification (true ATE = {r['tau']:.1f}; estimate == bias):")
    for name, b in r["abs_bias"].items():
        print(f"  {name:18s} |bias| = {b:.3f}")
    ab = r["abs_bias"]
    print(f"\n  including the instrument amplifies: {ab['include {X,Z}']:.3f} "
          f"> screen {ab['screen {X}']:.3f}  (and > crude {ab['none (crude)']:.3f})")
    print(f"  outcome-adaptive guard kept: {r['adaptive_kept']}  (drops Z → recovers screen)")


if __name__ == "__main__":
    main()
