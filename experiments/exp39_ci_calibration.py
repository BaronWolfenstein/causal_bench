"""exp39 calibration — size and power of the zero-flow CI test over a DGP grid (#85).

Supersedes the single-cell version of this driver, which measured one point
(n=300, dim Z=1, 200 reps) and reported a normal-approximation band. That was a smoke
test being asked to carry a calibration claim: at 200 reps an FPR near 0.05 carries a
95% interval of roughly +/-0.03, so a true size of 0.05 and one of 0.08 are
indistinguishable. We could say the test was not wildly mis-sized; we could not say it
was correctly sized.

That matters more than usual here because of the collider caveat (#103, exp22): the
zero-flow test both *detects* colliders and *is fooled by* them, and the estimand-side
discipline leans on its behaviour under H0. Its size is load-bearing.

WHAT DRIVES SIZE. Not `n_perm` -- that only sets the p-value's Monte-Carlo resolution
(at n_perm=100 the finest attainable p is ~0.01, adequate for alpha=0.05 but not for
tail work). Size is driven by **residualization quality**: the test compares X and Y
after regressing both on Z, so any Z-signal the nuisance learner fails to absorb leaks
into the statistic and inflates the false-positive rate. The default nuisance learner is
a random forest, so the axes that stress it are:

  - **dim(Z)** -- the curse. At fixed n, more nuisance dimensions means worse
    residualization. This is the axis most likely to break size.
  - **n** -- smaller n, worse residualization, and a shorter permutation null.
  - **nonlinearity** -- the RF is *meant* to absorb nonlinear Z effects; whether it
    actually does at small n and high dim is the question, not an assumption.

Reported with **Wilson** intervals (reused from validation/joint_fidelity), which stay
valid near 0 and 1 where a normal SE collapses -- power cells sit near 1 routinely.

A null cell is flagged MIS-SIZED only when its Wilson interval **excludes** alpha, so a
flag means the grid resolved a real distortion rather than noise.

Run:
    python -m experiments.exp39_ci_calibration                   # local, modest grid
    python -m experiments.exp39_ci_calibration --reps 2000 --jobs 64 --full
"""
from __future__ import annotations

import argparse
import json
from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from causal_bench.detectors.zero_flow_ci import zero_flow_ci_test
from causal_bench.validation.joint_fidelity import wilson_ci

OUT_DIR = Path("results/exp39_ci_calibration")

# (n, dim_Z) cells. The full grid crosses small n with high-dim Z, which is where
# residualization is expected to fail if it fails anywhere.
CELLS_QUICK = [(300, 1), (300, 3)]
CELLS_FULL = [(150, 1), (150, 3), (150, 8),
              (300, 1), (300, 3), (300, 8),
              (600, 1), (600, 3), (600, 8)]


def _z_signal(Z, nonlinear: bool):
    """Aggregate Z into a scalar driver. The nonlinear variant is what the RF must absorb."""
    if nonlinear:
        return (np.sin(1.5 * Z[:, 0]) + 0.5 * Z[:, 0] ** 2
                + 0.4 * Z[:, 1:].sum(axis=1))
    return Z.sum(axis=1) / np.sqrt(Z.shape[1])


def make_dgp(kind: str, dim_z: int, nonlinear: bool):
    """H0: X indep Y | Z (both driven by Z). H1: X -> Y on top of the shared Z."""
    def dgp(n, rng):
        Z = rng.standard_normal((n, dim_z))
        s = _z_signal(Z, nonlinear)
        X = s + 0.5 * rng.standard_normal(n)
        Y = (s + 0.5 * rng.standard_normal(n) if kind == "null"
             else 0.6 * X + s + 0.5 * rng.standard_normal(n))
        return X, Y, Z
    return dgp


def _one(r, *, kind, dim_z, nonlinear, n, n_perm, alpha, seed0):
    """One replicate. Top-level so multiprocessing can pickle it."""
    rng = np.random.default_rng(seed0 + r)
    dgp = make_dgp(kind, dim_z, nonlinear)
    return zero_flow_ci_test(*dgp(n, rng), n_perm=n_perm, alpha=alpha, rng=rng).verdict


def run_cell(*, kind, dim_z, nonlinear, n, reps, n_perm, alpha, seed0, pool=None):
    f = partial(_one, kind=kind, dim_z=dim_z, nonlinear=nonlinear, n=n,
                n_perm=n_perm, alpha=alpha, seed0=seed0)
    verdicts = pool.map(f, range(reps)) if pool else [f(r) for r in range(reps)]
    # "underpowered" (n < min_n) is a distinct verdict and must not be silently counted
    # as a non-rejection -- that would deflate the apparent size.
    n_under = sum(v == "underpowered" for v in verdicts)
    usable = [v for v in verdicts if v != "underpowered"]
    k, m = sum(v == "refutes" for v in usable), len(usable)
    rate = k / m if m else float("nan")
    lo, hi = wilson_ci(k, m) if m else (float("nan"), float("nan"))
    return {"kind": kind, "n": n, "dim_z": dim_z, "nonlinear": nonlinear,
            "reps": m, "n_underpowered": n_under, "rate": rate, "ci": [lo, hi],
            "mis_sized": bool(kind == "null" and m and not (lo <= alpha <= hi))}


def run(*, reps, n_perm, alpha, cells, jobs, seed0=0):
    rows, pool = [], (Pool(jobs) if jobs > 1 else None)
    try:
        i = 0
        for nonlinear in (False, True):
            for (n, dim_z) in cells:
                for kind in ("null", "alt"):
                    i += 1
                    rows.append(run_cell(kind=kind, dim_z=dim_z, nonlinear=nonlinear,
                                         n=n, reps=reps, n_perm=n_perm, alpha=alpha,
                                         seed0=seed0 + i * 1_000_003, pool=pool))
                    r = rows[-1]
                    print(f"  {r['kind']:>4} n={n:<4} dimZ={dim_z:<2} "
                          f"{'nonlin' if nonlinear else 'linear'} -> "
                          f"{r['rate']:.3f} [{r['ci'][0]:.3f},{r['ci'][1]:.3f}]"
                          f"{'  MIS-SIZED' if r['mis_sized'] else ''}", flush=True)
    finally:
        if pool:
            pool.close(); pool.join()
    return rows


def report(rows, *, alpha, n_perm) -> str:
    lines = [f"### Zero-flow CI test — size and power (alpha={alpha}, n_perm={n_perm})", "",
             "| Z effect | n | dim Z | reps | H0 size [95% CI] | H1 power [95% CI] |"
             " size verdict |",
             "|----------|---|-------|------|------------------|-------------------|"
             "--------------|"]
    by = {(r["nonlinear"], r["n"], r["dim_z"], r["kind"]): r for r in rows}
    for (nl, n, dz) in sorted({(r["nonlinear"], r["n"], r["dim_z"]) for r in rows}):
        h0, h1 = by.get((nl, n, dz, "null")), by.get((nl, n, dz, "alt"))
        if not h0:
            continue
        pw = f"{h1['rate']:.3f} [{h1['ci'][0]:.2f},{h1['ci'][1]:.2f}]" if h1 else "—"
        lines.append(
            f"| {'nonlinear' if nl else 'linear'} | {n} | {dz} | {h0['reps']} | "
            f"{h0['rate']:.3f} [{h0['ci'][0]:.3f},{h0['ci'][1]:.3f}] | {pw} | "
            f"{'**MIS-SIZED**' if h0['mis_sized'] else 'ok'} |")
    lines += [
        "",
        "Intervals are **Wilson**. `MIS-SIZED` means the H0 interval EXCLUDES alpha — a",
        "flag is a resolved distortion, not noise. Absence of a flag does NOT certify a",
        "cell as correctly sized; it means the cell is consistent with alpha at this rep",
        "count.",
        "",
        f"`n_perm={n_perm}` bounds the attainable p-value near {1.0/(n_perm+1):.3f} — fine",
        f"for alpha={alpha}, but it would need raising to probe further into the tail.",
        "",
        "Size is driven by residualization quality, not by n_perm. Read the dim-Z axis",
        "first: that is where the nuisance learner is expected to fail if it fails at all.",
    ]
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description="exp39: zero-flow CI size/power grid")
    p.add_argument("--reps", type=int, default=300)
    p.add_argument("--n-perm", type=int, default=100)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--full", action="store_true", help="9-cell (n, dim Z) grid")
    p.add_argument("--out", type=str, default=str(OUT_DIR))
    a = p.parse_args()

    cells = CELLS_FULL if a.full else CELLS_QUICK
    print(f"{len(cells)} (n,dimZ) cells x 2 dgp x 2 nonlinearity, "
          f"reps={a.reps}, jobs={a.jobs}")
    rows = run(reps=a.reps, n_perm=a.n_perm, alpha=a.alpha, cells=cells, jobs=a.jobs)
    rep = report(rows, alpha=a.alpha, n_perm=a.n_perm)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(rep + "\n")
    (out / "rows.json").write_text(json.dumps(rows, indent=2))
    print("\n" + rep)
    print(f"\nSaved → {out}")


if __name__ == "__main__":
    main()
