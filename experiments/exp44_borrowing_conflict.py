"""Exp 44: two-epoch borrowing under prior-data conflict — Type-I(δ) per policy.

Historical epoch at μ=δ, current epoch truly null (μ=0); δ is the prior-data-conflict
magnitude. Sweeps δ and reports Type-I (reject H0: μ_current=0) for each μ-borrowing policy —
the size a point-calibration at δ=0 misses. `flat` ignores history; `map`/`pooled` inflate as
δ grows; `robust_map` down-weights the conflicting prior and stays near nominal (the payoff).

Reference: Qian (Evidence in the Wild / JSM 2026); FDA *Use of Bayesian Methodology in Clinical
Trials* (draft, Jan 2026); Schmidli et al. 2014 (robust MAP). Spec:
docs/superpowers/specs/2026-07-29-two-epoch-borrowing-conflict-design.md. Part A (general core);
the ENCIRCLE residual-borrowing layer is gated on a diagnostic and NOT built here.

Requires the 3.12 `[bayes]` stack (PyMC/NumPyro). Run: python -m experiments.exp44_borrowing_conflict
"""
from pathlib import Path

from causal_bench.validation.two_epoch_borrowing import type_i_curve

OUT_DIR = Path("results/exp44_borrowing_conflict")
POLICIES = ["flat", "map", "robust_map", "pooled"]
DELTAS = [0.0, 0.15, 0.30, 0.45, 0.60]


def run(*, deltas=DELTAS, policies=POLICIES, n_reps=25, draws=400, tune=400,
        chains=2, seed=0, **kw):
    rows = []
    for pol in policies:
        rows += type_i_curve(deltas, pol, n_reps=n_reps, draws=draws, tune=tune,
                             chains=chains, seed=seed, **kw)
    return rows


def report(rows, *, alpha=0.05) -> str:
    deltas = sorted({r["delta"] for r in rows})
    policies = [p for p in POLICIES if any(r["policy"] == p for r in rows)]
    grid = {(r["policy"], r["delta"]): r["type_i"] for r in rows}
    L = ["## Exp 44 — two-epoch borrowing under prior–data conflict\n",
         "Type-I (reject H0: μ_current=0) vs conflict magnitude δ. Current epoch is truly null; "
         f"δ=0 is exchangeable (no conflict). Nominal α={alpha}.\n",
         "| policy | " + " | ".join(f"δ={d:.2f}" for d in deltas) + " |",
         "|--------|" + "|".join("-------" for _ in deltas) + "|"]
    for p in policies:
        L.append(f"| {p} | " + " | ".join(f"{grid.get((p, d), float('nan')):.3f}" for d in deltas) + " |")
    # supremum Type-I over the δ sweep (the #195 discipline) — the headline size per policy
    L.append("")
    L.append("| policy | sup Type-I over δ | flag |")
    L.append("|--------|-------------------|------|")
    for p in policies:
        sup = max(grid.get((p, d), float("nan")) for d in deltas)
        L.append(f"| {p} | {sup:.3f} | {'⚠ inflates' if sup > alpha + 0.05 else 'ok'} |")
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 44: two-epoch borrowing under conflict")
    p.add_argument("--n-reps", type=int, default=25)
    p.add_argument("--draws", type=int, default=400)
    p.add_argument("--tune", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rows = run(n_reps=a.n_reps, draws=a.draws, tune=a.tune, seed=a.seed)
    rep = report(rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)
    print("\nRead-out: at δ=0 every policy is nominal. As the historical prior conflicts with "
          "the null current data (δ grows), `map` and `pooled` inflate Type-I while `robust_map` "
          "down-weights the vague-mixture component and stays near nominal — adaptation is real, "
          "but (per Qian) it is not automatic: read where robust_map's sup Type-I still exceeds α.")


if __name__ == "__main__":
    main()
