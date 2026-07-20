"""Exp 42: Calendar-time (era) confounding and the embedding-laundering trap
(causal_bench #173, ENCIRCLE).

Era drives both membership (concurrent trial vs historical control) and outcome
(secular trend), so it is a confounder. Adjusting for a patient-STATE proxy that
only imperfectly mirrors era (what a frozen-encoder embedding gives you)
**launders** the calendar confounding — the bias is roughly halved but far from
gone. Putting era in EXPLICITLY recovers the null. Lesson for the #99
embedding-propensity: calendar/era must be a first-class covariate, not left to
leak through the state embedding.

Companion to exp37 (#82, unmeasured confounding × enrollment drift): exp37 sweeps
the compounding grid; this isolates the state-vs-era laundering mechanism.
Run: `PYTHONPATH=. python experiments/exp42_calendar_confounding.py`.
"""
from __future__ import annotations

import numpy as np

from causal_bench.dgp.calendar_confounding import (
    CalendarConfig, draw_calendar, adjusted_effect, true_tau)

ADJUSTMENT_SETS = {
    "naive (Y~A)": [],
    "state-proxy {S,X}": ["S", "X"],
    "era-explicit {E,X}": ["E", "X"],
}


def run(n: int = 3000, reps: int = 300, config: CalendarConfig = CalendarConfig()) -> dict:
    tau = true_tau(config)
    bias = {k: [] for k in ADJUSTMENT_SETS}
    for seed in range(reps):
        df = draw_calendar(n, seed, config)
        for name, cols in ADJUSTMENT_SETS.items():
            bias[name].append(adjusted_effect(df, cols) - tau)
    return {"tau": tau, "abs_bias": {k: float(abs(np.mean(v))) for k, v in bias.items()}}


def main() -> None:
    r = run()
    print(f"Calendar-time confounding (true ATE = {r['tau']:.1f}; estimate == bias):")
    for name, b in r["abs_bias"].items():
        print(f"  {name:20s} |bias| = {b:.3f}")
    ab = r["abs_bias"]
    print(f"\n  state-proxy launders era: |bias| {ab['state-proxy {S,X}']:.3f} still large "
          f"vs era-explicit {ab['era-explicit {E,X}']:.3f} (~null).")
    print("  Lesson: era must be an EXPLICIT propensity covariate, not laundered "
          "through the patient-state embedding.")


if __name__ == "__main__":
    main()
