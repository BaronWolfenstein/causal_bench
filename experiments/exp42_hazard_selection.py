"""Exp 42: the hazard ratio's built-in selection bias vs cumulative-risk estimands.

Demonstrates, rather than asserts, why this benchmark's primary estimands are cumulative
risk measures (ENCIRCLE's KM 1-year composite rate; the cloglog-of-cumulative-incidence
borrowing scale; RMST in the OC-sim blinding harness) and not a hazard ratio.

Setup (`causal_bench.validation.hazard_selection`): treatment is RANDOMIZED, there is no
baseline confounding of any kind, and the conditional (given frailty U) hazard ratio is
CONSTANT by construction. The only extra ingredient is an unobserved Gamma(k, k) frailty.
Because the hazard at t is defined only among those still event-free at t, and survival is
a collider between A and U (``A -> S(t) <- U -> event``), conditioning on survival opens a
non-causal path and biases the HR anyway — the depletion-of-susceptibles effect.

With Gamma(k, k) frailty everything has a closed form, so the experiment is
self-validating: the marginal HR is ``(c1/c0)(1 + c0 t/k)/(1 + c1 t/k)``, equal to the
conditional HR at t=0 and provably attenuating to 1. The k -> infinity row is the control
(no frailty ⇒ no bias), which shows the drift is the frailty and not the estimator.

Read-out: the Cox HR is biased toward the null, monotonically in frailty variance, and
drifts between an early and a late analysis window even though the conditional effect is
constant. The risk difference and RMST difference — which never condition on survival —
stay unbiased for their marginal truths on the SAME replicates.

Caveat kept explicit: even without selection bias the HR is **non-collapsible**, so a
hazard-scale contrast is not directly comparable to a risk-difference/RMST contrast in the
same table. `estimators/cox.py` is retained in the bake-off as a deliberately naive
comparator; this experiment is what licenses reading it that way.

Run: python -m experiments.exp42_hazard_selection
"""
from pathlib import Path

from causal_bench.validation.hazard_selection import (
    hazard_selection_report, marginal_hazard_ratio,
)

OUT_DIR = Path("results/exp42_hazard_selection")
# Gamma(k, k) frailty: variance 1/k, so smaller k = stronger unmeasured heterogeneity.
FRAILTY = [("none (k=1e6)", 1e6), ("moderate (k=2)", 2.0),
           ("strong (k=1)", 1.0), ("severe (k=0.5)", 0.5)]
LOG_HR = -0.7                                       # conditional HR = 0.497, protective


def run(*, n=20000, n_reps=20, horizon=2.0, eval_at=1.0, t_split=1.0, seed=0):
    rows = []
    for label, k in FRAILTY:
        r = hazard_selection_report(log_hr=LOG_HR, frailty_shape=k, n=n, n_reps=n_reps,
                                    horizon=horizon, eval_at=eval_at, t_split=t_split,
                                    seed=seed)
        rows.append({"frailty": label, "k": k, **r})
    return rows


def report(rows) -> str:
    lines = [
        "| frailty | Cox HR | cond. truth | HR bias | early HR | late HR | drift |"
        " RD bias | RMST bias |",
        "|---------|--------|-------------|---------|----------|---------|-------|"
        "---------|-----------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['frailty']} | {r['cox_hr']:.3f} | {r['hr_conditional_truth']:.3f} | "
            f"{r['cox_hr_bias_vs_conditional']:+.3f} | {r['hr_early']:.3f} | "
            f"{r['hr_late']:.3f} | {r['hr_drift']:+.3f} | "
            f"{r['risk_difference_bias']:+.4f} | {r['rmst_difference_bias']:+.4f} |")
    lines.append("")
    lines.append("Analytic marginal HR(t) (conditional HR is constant at "
                 f"{__import__('numpy').exp(LOG_HR):.3f}):")
    lines.append("")
    lines.append("| frailty | HR(0) | HR(0.5) | HR(1) | HR(2) | HR(5) |")
    lines.append("|---------|-------|---------|-------|-------|-------|")
    for label, k in FRAILTY:
        hrs = [marginal_hazard_ratio(t, log_hr=LOG_HR, frailty_shape=k)
               for t in (0.0, 0.5, 1.0, 2.0, 5.0)]
        lines.append(f"| {label} | " + " | ".join(f"{h:.3f}" for h in hrs) + " |")
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 42: built-in selection bias of the HR")
    p.add_argument("--n", type=int, default=20000)
    p.add_argument("--n-reps", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()

    rows = run(n=a.n, n_reps=a.n_reps, seed=a.seed)
    rep = report(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)
    print(f"\nSaved → {OUT_DIR}")
    print("\nRead-out: with NO frailty the Cox HR is unbiased and does not drift — the "
          "control. As frailty variance grows the Cox HR attenuates toward the null "
          "monotonically and the early→late window gap widens, purely from conditioning "
          "on survival (treatment is randomized; the conditional HR is constant). The "
          "risk difference and RMST difference stay unbiased throughout. This is the "
          "evidence for preferring cumulative-incidence / RMST estimands.")


if __name__ == "__main__":
    main()
