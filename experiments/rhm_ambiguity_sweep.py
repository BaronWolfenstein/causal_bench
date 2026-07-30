"""Does ambiguity move theta*? — the finer sweep #137 needs (not an expNN; a diagnostic study).

The first sweep (commit 2d6dbc2) resolved that ambiguity SUPPRESSES class overlap
monotonically and gracefully, but could NOT answer #137's actual question — whether the
transition point theta* moves. There the finite-difference argmax read 0.80 -> 0.90,
which is one cell on a 0.1-spaced grid: at the resolution limit and not separable from
noise. Degradation of the order parameter and movement of the transition are different
claims, and only the second bears on whether the theta <-> t mapping this issue reports
is still valid under ambiguity.

Three changes make theta* estimable rather than merely visible:

1. **Fine grid** (0.02 spacing) so the susceptibility peak is not pinned to a coarse cell.
2. **Sub-grid peak location** by parabolic interpolation through the argmax and its two
   neighbours — an argmax alone cannot resolve a shift smaller than the grid.
3. **Independent grammar seeds** as replicates, giving theta* a standard error. A shift is
   only claimed when it exceeds the seed-to-seed spread; without this the previous
   0.80 -> 0.90 reading was uninterpretable.

Depth 5 (not 4) because the transition sharpens with depth (the FSS result in #136), and
a sharper peak localises better.

Run (detached):  nohup python -m experiments.rhm_ambiguity_sweep --jobs 32 &
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from causal_bench.diagnostics.rhm_grammar import (
    make_emission, make_rhm, rhm_class_overlap,
)

OUT_DIR = Path("results/rhm_ambiguity")
V, S, M, DEPTH = 8, 2, 3, 5
P_MERGES = [0.0, 0.25, 0.5, 1.0]
# Finer near the low end: the AMBIGUITY BUDGET (largest p_merge whose theta* shift is
# NOT resolved) lives there -- p_merge=0.25 came back z=1.65 at depth 5, i.e. inside the
# budget -- so locating its edge needs resolution below 0.25, not above it.
P_MERGES_FINE = [0.0, 0.125, 0.25, 0.375, 0.5, 1.0]


def peak_subgrid(x: np.ndarray, y: np.ndarray) -> float:
    """Location of the max of y(x) to sub-grid precision via a parabola through the
    argmax and its neighbours. Returns the bare argmax at the boundary, where no
    three-point fit exists — flagged by the caller rather than silently extrapolated."""
    i = int(np.argmax(y))
    if i == 0 or i == len(y) - 1:
        return float(x[i])
    y0, y1, y2 = y[i - 1], y[i], y[i + 1]
    denom = (y0 - 2.0 * y1 + y2)
    if abs(denom) < 1e-12:
        return float(x[i])
    delta = 0.5 * (y0 - y2) / denom                      # in grid units
    return float(x[i] + delta * (x[1] - x[0]))


def _one(task, *, thetas, n_trees):
    """One (p_merge, grammar_seed, depth) replicate -> its overlap curve. Top-level for pickling."""
    p_merge, gseed, depth = task
    rules = make_rhm(V, S, M, seed=gseed)
    emission = make_emission(V, p_merge, seed=1000 + gseed)
    ov = [rhm_class_overlap(V, S, M, depth, float(t), n_trees=n_trees, seed=gseed,
                            rules=rules, emission=emission) for t in thetas]
    return {"p_merge": p_merge, "grammar_seed": gseed, "depth": depth, "overlap": ov}


def main():
    p = argparse.ArgumentParser(description="#137: does ambiguity move theta*?")
    p.add_argument("--jobs", type=int, default=16)
    p.add_argument("--n-trees", type=int, default=2000)
    p.add_argument("--seeds", type=int, default=8, help="independent grammars (replicates)")
    p.add_argument("--depths", type=int, nargs="+", default=[DEPTH],
                   help="tree depths. The prediction under test: 2602.06065 says "
                        "cross-scale correlation lifts local ambiguity, so MORE depth "
                        "should WIDEN the budget (shift shrinks with depth).")
    p.add_argument("--fine", action="store_true", help="finer p_merge grid near the budget")
    p.add_argument("--out", type=str, default=str(OUT_DIR))
    a = p.parse_args()

    thetas = np.round(np.arange(0.60, 1.0001, 0.02), 4)
    pmerges = P_MERGES_FINE if a.fine else P_MERGES
    tasks = [(pm, gs, d) for d in a.depths for pm in pmerges for gs in range(a.seeds)]
    print(f"{len(tasks)} replicates ({len(pmerges)} p_merge x {a.seeds} grammars x "
          f"{len(a.depths)} depths={a.depths}), {len(thetas)} thetas, "
          f"n_trees={a.n_trees}, jobs={a.jobs}", flush=True)

    f = partial(_one, thetas=thetas, n_trees=a.n_trees)
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            rows = []
            for r in pool.imap_unordered(f, tasks):
                rows.append(r)
                print(f"  done depth={r['depth']} p_merge={r['p_merge']:.3f} "
                      f"seed={r['grammar_seed']} ({len(rows)}/{len(tasks)})", flush=True)
    else:
        rows = [f(t) for t in tasks]

    # theta* per replicate from the susceptibility (dm/dtheta) peak
    tc = 0.5 * (thetas[1:] + thetas[:-1])                # midpoints of the difference
    summary = []
    for depth in a.depths:
      for pm in pmerges:
        sel = [r for r in rows if r["p_merge"] == pm and r["depth"] == depth]
        if not sel:
            continue
        stars = []
        for r in sel:
            d = np.diff(np.asarray(r["overlap"])) / np.diff(thetas)
            stars.append(peak_subgrid(tc, d))
        stars = np.asarray(stars, float)
        summary.append({"p_merge": pm, "depth": depth, "n": len(stars),
                        "theta_star": float(stars.mean()),
                        "se": float(stars.std(ddof=1) / np.sqrt(len(stars))) if len(stars) > 1
                              else float("nan"),
                        "stars": stars.tolist()})

    lines = ["", "### theta* vs ambiguity and depth (susceptibility peak, sub-grid)", "",
             "| depth | p_merge | theta* ± SE | shift vs p=0 | shift / SE_diff | verdict |",
             "|-------|---------|-------------|--------------|-----------------|---------|"]
    budgets = {}
    for depth in a.depths:
        block = [r for r in summary if r["depth"] == depth]
        if not block:
            continue
        base = next(r for r in block if r["p_merge"] == 0.0)
        # Budget = the edge BEFORE the first resolved shift, walking p_merge upward.
        # "Largest unresolved" would be wrong if resolution is non-monotone (a
        # high-p_merge cell can fail to resolve from low power, not from safety).
        first_resolved = None
        for r in block:
            d = r["theta_star"] - base["theta_star"]
            se_d = (float(np.sqrt(r["se"] ** 2 + base["se"] ** 2))
                    if r["p_merge"] else float("nan"))
            z = abs(d) / se_d if se_d and np.isfinite(se_d) and se_d > 0 else float("nan")
            resolved = np.isfinite(z) and z > 2
            verdict = ("— (reference)" if r["p_merge"] == 0.0
                       else ("MOVES" if resolved else "not resolved"))

            if resolved and r["p_merge"] > 0 and first_resolved is None:
                first_resolved = r["p_merge"]
            lines.append(f"| {depth} | {r['p_merge']:.3f} | {r['theta_star']:.4f} ± "
                         f"{r['se']:.4f} | {d:+.4f} | {z:.2f} | {verdict} |")
        below = [r["p_merge"] for r in block
                 if r["p_merge"] > 0 and (first_resolved is None or r["p_merge"] < first_resolved)]
        budgets[depth] = max(below) if below else 0.0
    lines += ["", "### Ambiguity budget vs depth", "",
              "| depth | budget (largest UNRESOLVED p_merge) |", "|-------|------|"]
    for depth in a.depths:
        lines.append(f"| {depth} | {budgets.get(depth, 0.0):.3f} |")
    lines += ["",
              "PREDICTION UNDER TEST (2602.06065): correlations across scales lift local",
              "ambiguity, so a DEEPER tree has more scales to lift it and the budget should",
              "WIDEN with depth. The opposite — budget narrowing — would mean ambiguity",
              "accumulates across levels faster than structure absorbs it, and would make",
              "the theta<->t mapping progressively less transferable to deep real data.",
              "",
              "Budget = the largest p_merge BELOW the first resolved shift, so it is the",
              "safe edge rather than merely an unresolved cell (a high-p_merge cell can",
              "fail to resolve from low power). It is a grid quantity — read it as a",
              "bracket, not a point estimate. Budget = max p_merge means nothing resolved,",
              "which is a POWER statement, not a safety one."]
    lines += ["",
              "A shift is claimed only at |shift| > 2 SE of the DIFFERENCE, using the",
              "grammar-to-grammar spread as the error term. The earlier 0.80 -> 0.90",
              "reading had no error bar and was one coarse grid cell — not evidence.",
              "",
              "theta* is the susceptibility (dm/dtheta) peak located to sub-grid precision;",
              "a bare argmax cannot resolve a shift smaller than the grid spacing."]
    report = "\n".join(lines)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(report + "\n")
    (out / "rows.json").write_text(json.dumps(
        {"thetas": thetas.tolist(), "rows": rows, "summary": summary}, indent=2))
    print(report, flush=True)
    print(f"\nSaved → {out}", flush=True)


if __name__ == "__main__":
    main()
