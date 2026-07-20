"""Exp 23: Immortal-time bias — the estimator-proof honest null (causal_bench #21,
ENCIRCLE).

A device has NO effect (true RD = 0), yet classifying patients as "device" from
eligibility — before they survive to implant — makes it look protective. Covariate
adjustment does NOT fix it (the bias is mis-aligned time-zero, not confounding);
only the DESIGN fix (a landmark here; clone-censor-weight for a grace period)
recovers the null. This is the case ENCIRCLE's single-arm-vs-external-control
design must prevent by construction, not estimate away.

Run: `PYTHONPATH=. python experiments/exp23_immortal_time.py`.
"""
from __future__ import annotations

import numpy as np

from causal_bench.dgp.immortal_time import (
    ImmortalTimeConfig, draw_immortal_time, naive_risk_difference,
    adjusted_effect, landmark_risk_difference, grace_period_naive_rd,
    ccw_risk_difference)


def run(n: int = 4000, reps: int = 300, config: ImmortalTimeConfig = ImmortalTimeConfig()) -> dict:
    naive, adj, lm, gnaive, ccw = [], [], [], [], []
    for seed in range(reps):
        df = draw_immortal_time(n, seed, config)
        naive.append(naive_risk_difference(df))
        adj.append(adjusted_effect(df))
        lm.append(landmark_risk_difference(df, config))
        gnaive.append(grace_period_naive_rd(df, config))
        ccw.append(ccw_risk_difference(df, config))
    return {"true_effect": 0.0,
            "naive_immortal": float(np.mean(naive)),
            "adjusted_for_X": float(np.mean(adj)),
            "landmark_design_fix": float(np.mean(lm)),
            "grace_naive": float(np.mean(gnaive)),
            "clone_censor_weight": float(np.mean(ccw))}


def main() -> None:
    r = run()
    print(f"Immortal-time bias (TRUE effect = {r['true_effect']:.1f}, honest null):")
    print(f"  naive (time-zero at eligibility) RD = {r['naive_immortal']:+.3f}  "
          f"← spurious 'device protective'")
    print(f"  adjusted for X (confounder)      RD = {r['adjusted_for_X']:+.3f}  "
          f"← estimator does NOT remove it")
    print(f"  landmark (design fix)            RD = {r['landmark_design_fix']:+.3f}  "
          f"← recovers the null (exact for point-implant)")
    print(f"\n  Grace-period strategy ('implant by G'):")
    print(f"  grace-period per-protocol (naive) RD = {r['grace_naive']:+.3f}  "
          f"← immortal-time biased")
    print(f"  clone-censor-weight (CCW)         RD = {r['clone_censor_weight']:+.3f}  "
          f"← removes the bulk via cloning + IPCW")
    print("\n  Immortal-time bias is design-level and estimator-proof: fix time-zero "
          "(landmark / clone-censor-weight), not the estimator.")


if __name__ == "__main__":
    main()
