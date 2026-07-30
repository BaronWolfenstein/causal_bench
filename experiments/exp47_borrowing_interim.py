"""Exp 47 (#196): borrowing × interim-looks composition.

Composes external-control borrowing (a historical prior, possibly in conflict δ) with repeated
interim analyses (`sequential.py`: naive / O'Brien-Fleming α-spending / Howard confidence
sequence). Under the current NULL, reports **cumulative** Type-I (reject at ANY of K looks) per
(borrowing policy × monitoring rule × δ).

The demonstration: neither component test catches the composition. `sequential.py` monitors a
*non-borrowing* statistic; exp41/exp44 are *single-analysis*. Borrowing centers the per-look
statistic at δ and breaks the independent-increments assumption OBF is built on, so
borrowing × monitoring inflates cumulative Type-I beyond either alone — up to 1.0 under EVERY
monitoring rule at strong conflict. Discounting (`power`) mitigates but does not rescue.

Reference: Qian (Evidence in the Wild / JSM 2026); FDA *Use of Bayesian Methodology in Clinical
Trials* (draft, Jan 2026). CED-relevant: a borrowed SCA + CMS milestone-triggered re-analyses.

Run: python -m experiments.exp47_borrowing_interim
"""
from pathlib import Path

from causal_bench.validation.borrowing_interim import composition_type_i

OUT_DIR = Path("results/exp47_borrowing_interim")
POLICIES = ["flat", "map", "power"]
DELTAS = [0.0, 0.15, 0.30]
METHODS = ["naive", "obf", "confidence_sequence"]


def run(*, K=5, n_reps=3000, n_per_look=100, seed=0):
    return composition_type_i(DELTAS, POLICIES, K=K, n_reps=n_reps, n_per_look=n_per_look, seed=seed)


def report(rows, *, alpha=0.05, K=5) -> str:
    grid = {(r["policy"], r["delta"], r["method"]): r["type_i"] for r in rows}
    deltas = sorted({r["delta"] for r in rows})
    L = [f"## Exp 47 — borrowing × interim-looks composition (cumulative Type-I over K={K} looks)\n",
         f"Current cohort truly null; historical prior at μ=δ (δ=0 = no conflict). Nominal α={alpha}.\n",
         "| policy | monitoring | " + " | ".join(f"δ={d:.2f}" for d in deltas) + " |",
         "|--------|------------|" + "|".join("------" for _ in deltas) + "|"]
    for pol in POLICIES:
        for m in METHODS:
            cells = " | ".join(f"{grid.get((pol, d, m), float('nan')):.3f}" for d in deltas)
            L.append(f"| {pol} | {m} | {cells} |")
    L.append("")
    L.append("Read-out: **flat + naive** shows the pure multiplicity inflation that OBF / "
             "confidence-sequence correctly control. **map under conflict** drives cumulative "
             "Type-I toward 1.0 under EVERY monitoring rule — borrowing centers the statistic at "
             "δ, so boundaries built for a null-centered z are meaningless: the composition "
             "breaks all three monitors, which no single-component test reveals. **power** "
             "(discounted borrow) mitigates but does not rescue (OBF still inflated at δ>0).")
    return "\n".join(L)


def main():
    import argparse
    p = argparse.ArgumentParser(description="Exp 47: borrowing × interim-looks composition")
    p.add_argument("--n-reps", type=int, default=3000)
    p.add_argument("--K", type=int, default=5)
    a = p.parse_args()
    rows = run(K=a.K, n_reps=a.n_reps)
    rep = report(rows, K=a.K)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "summary.md").write_text(rep + "\n")
    print(rep)


if __name__ == "__main__":
    main()
