"""Exp 49: probabilistic QBA for unmeasured confounding — calibration + misspecification.

Fox/MacLehose/Lash (IJE 2023) probabilistic quantitative bias analysis, closed-form v1
(the n-independent omitted-variable form). A measured covariate set + an UNMEASURED
binary confounder U biases the naive effect; QBA samples the bias parameters from
priors and reports systematic- and total-error intervals
(`causal_bench.validation.qba_confounding`).

The experiment (beyond a point bias-adjustment) is the CALIBRATION study: does the
total-error interval cover the truth under CORRECT priors, and how does coverage
degrade under bias-parameter MISSPECIFICATION? Read-out: correct priors recover the
effect and cover; underestimating U's effect ("misspec_half") undercovers; assuming U
harmless ("misspec_null") stays at the naive bias with ~0 coverage. Same discipline as
exp41 borrowing-calibration -- a bias tool is only trustworthy if its interval is
calibrated, and QBA's is exactly as good as its priors.

v2 (gated, not here): record-level, estimator-agnostic QBA via an influence-function
one-step update, to certify QBA for the nonlinear TMLE/AIPW estimators at scale.

Run: python -m experiments.exp49_qba_confounding
"""
import json
from pathlib import Path

from causal_bench.validation.qba_confounding import report_rows

OUT_DIR = Path("results/exp49_qba_confounding")


def run(*, n=1500, n_reps=200, seed=0):
    return report_rows(n=n, n_reps=n_reps, seed=seed)


def report(rows) -> str:
    lines = ["Exp 49: QBA for unmeasured confounding — calibration (true ATE = 1.0)",
             "",
             f"  {'prior mode':>14} {'naive_bias':>11} {'adj_bias':>9} {'cov_syst':>9} {'cov_total':>10}",
             "  " + "-" * 58]
    for r in rows:
        lines.append(f"  {r['mode']:>14} {r['naive_bias']:>+11.3f} {r['adjusted_bias']:>+9.3f} "
                     f"{r['coverage_syst']:>9.2f} {r['coverage_total']:>10.2f}")
    lines += ["",
              "Read: correct priors recover the effect (adj_bias ~ 0) and the total-error",
              "interval covers; underestimating U ('misspec_half') undercovers; assuming U",
              "harmless ('misspec_null') stays at the naive bias with ~0 coverage. QBA is",
              "trustworthy only to the extent its bias-parameter priors are."]
    return "\n".join(lines)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 49: QBA for unmeasured confounding")
    p.add_argument("--n", type=int, default=1500)
    p.add_argument("--n_reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rows = run(n=a.n, n_reps=a.n_reps, seed=a.seed)
    print(report(rows))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "results.json").write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
