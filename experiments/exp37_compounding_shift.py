"""Exp 37: Compounding covariate shift — unmeasured confounding × enrollment
drift (LENS-inspired, causal_bench #82).

Motivated by Seegmiller & Preum, "LENS: Measuring Distribution Shift in User
Prompts" (ACL 2026): moderate covariate shift alone costs ~73% average
performance loss, and degradation is WORST when shift is driven by a latent
group AND correlated with time — i.e. super-additive (compounding), not just
the sum of the two individual effects.

exp3 (unmeasured confounding) and exp6 (enrollment drift) each sweep ONE axis
independently. This experiment crosses them: unmeasured confounding IS latent-
group structure (an unobserved subpopulation driving treatment/outcome);
enrollment drift IS the temporal axis. The question: does bias(U, drift)
exceed what you'd predict from bias(U, 0) and bias(0, drift) alone (additive),
or does it compound super-additively the way LENS found?

Grid: unmeasured_confounding_strength in {0.0, 0.27, 0.53, 0.8} (exp3's
endpoints, 4 points) x enrollment_drift in {0.0, 0.17, 0.33, 0.5} (exp6's
endpoints, 4 points) = 16 cells. Estimators restricted to a decision-relevant
subset spanning no-adjustment -> parametric-adjustment -> flexible DR, to
keep the 2D grid tractable: naive (no adjustment), cox (parametric covariate
adjustment, exp6's fix for drift alone), aipw, tmle_ipcw (doubly robust).

Compounding metric: for each estimator, at each (u, d) cell,
    excess_bias(u,d) = bias(u,d) - [bias(u,0) + bias(0,d) - bias(0,0)]
the two-way-interaction term from a 2x2-style decomposition. excess_bias > 0
means compounding (worse than additive, the LENS finding); <= 0 means the
axes are additive or protective in combination.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_bench.dgp.config import DGPConfig
from causal_bench.runner import run_simulation

UC_GRID = [0.0, 0.27, 0.53, 0.8]        # exp3's endpoints, 4 points
DRIFT_GRID = [0.0, 0.17, 0.33, 0.5]     # exp6's endpoints, 4 points
ESTIMATORS = ["naive", "cox", "aipw", "tmle_ipcw"]
N_SIMS = 100  # smaller than exp3/exp6's 200: this is a 16-cell grid, not a 6-cell sweep
N_MAIN = 500

OUT_DIR = Path(__file__).parent.parent / "results" / "exp37_compounding_shift"


def run_grid(n_sims: int = N_SIMS, seed: int = 42) -> pd.DataFrame:
    """Run the full (unmeasured_confounding_strength x enrollment_drift) grid
    for every estimator in ESTIMATORS. Returns one row per (estimator, u, d)
    cell with the bias at that cell."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    records = []
    n_cells = len(UC_GRID) * len(DRIFT_GRID)
    cell_num = 0

    for u, d in product(UC_GRID, DRIFT_GRID):
        cell_num += 1
        print(f"  Cell {cell_num}/{n_cells}: U={u:.2f}, drift={d:.2f}", flush=True)
        config = DGPConfig(n=N_MAIN, unmeasured_confounding_strength=u,
                          enrollment_drift=d, true_tau=-0.5)
        cell_seed = seed + int(u * 1000) * 100 + int(d * 1000)
        sim_results = run_simulation(config, ESTIMATORS, n_sim=n_sims,
                                     seed=cell_seed, debug_first_replicate=False)
        for name in ESTIMATORS:
            sr = sim_results.get(name)
            if sr is None:
                continue
            bias = float(np.mean(sr.estimates) - sr.true_value)
            records.append({
                "estimator": name, "U": u, "drift": d,
                "bias": bias, "true_value": sr.true_value,
                "n_sim": sr.n_sim,
            })
    return pd.DataFrame(records)


def compounding_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each estimator, compute excess_bias(u,d) = bias(u,d) -
    [bias(u,0) + bias(0,d) - bias(0,0)] at every (u,d) cell with u>0 and d>0.
    Positive = compounding (super-additive, the LENS finding); <=0 = additive
    or protective."""
    rows = []
    for est, sub in df.groupby("estimator"):
        pivot = sub.pivot(index="U", columns="drift", values="bias")
        bias_00 = pivot.loc[0.0, 0.0]
        for u in UC_GRID:
            if u == 0.0:
                continue
            for d in DRIFT_GRID:
                if d == 0.0:
                    continue
                bias_ud = pivot.loc[u, d]
                bias_u0 = pivot.loc[u, 0.0]
                bias_0d = pivot.loc[0.0, d]
                additive_prediction = bias_u0 + bias_0d - bias_00
                excess = bias_ud - additive_prediction
                rows.append({
                    "estimator": est, "U": u, "drift": d,
                    "bias_observed": bias_ud,
                    "additive_prediction": additive_prediction,
                    "excess_bias": excess,
                    "compounding": excess > 0,
                })
    return pd.DataFrame(rows)


def plot_heatmaps(df: pd.DataFrame, out_path: Path) -> None:
    """One |bias| heatmap per estimator, U x drift."""
    n_est = len(ESTIMATORS)
    fig, axes = plt.subplots(1, n_est, figsize=(4.5 * n_est, 4))
    if n_est == 1:
        axes = [axes]

    for ax, est in zip(axes, ESTIMATORS):
        sub = df[df["estimator"] == est]
        pivot = sub.pivot(index="U", columns="drift", values="bias").reindex(
            index=UC_GRID, columns=DRIFT_GRID)
        im = ax.imshow(np.abs(pivot.values), aspect="auto", origin="lower",
                       cmap="Reds")
        ax.set_xticks(range(len(DRIFT_GRID)))
        ax.set_xticklabels([f"{d:.2f}" for d in DRIFT_GRID])
        ax.set_yticks(range(len(UC_GRID)))
        ax.set_yticklabels([f"{u:.2f}" for u in UC_GRID])
        ax.set_xlabel("enrollment_drift")
        if est == ESTIMATORS[0]:
            ax.set_ylabel("unmeasured_confounding_strength")
        ax.set_title(est)
        fig.colorbar(im, ax=ax, label="|bias|")

    fig.suptitle(
        "Exp 37: |bias| under unmeasured confounding x enrollment drift\n"
        "(does bias compound faster than either axis alone?)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


def main() -> None:
    print(f"\nExp 37: Compounding covariate shift (unmeasured confounding x drift)")
    print(f"  Grid: U {UC_GRID} x drift {DRIFT_GRID} ({len(UC_GRID)*len(DRIFT_GRID)} cells)")
    print(f"  Estimators: {ESTIMATORS}, n_sims={N_SIMS}\n")

    df = run_grid()
    df.to_csv(OUT_DIR / "bias_grid.csv", index=False)
    print(f"\nBias grid saved: {OUT_DIR / 'bias_grid.csv'}")

    comp = compounding_table(df)
    comp.to_csv(OUT_DIR / "compounding_table.csv", index=False)
    print("\nCompounding table (excess_bias > 0 => super-additive, the LENS finding):")
    print(comp.to_string(index=False))

    plot_heatmaps(df, OUT_DIR / "bias_heatmaps.png")
    print("\nDone.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 37: Compounding covariate shift")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = run_grid(n_sims=args.n_sims, seed=args.seed)
    df.to_csv(OUT_DIR / "bias_grid.csv", index=False)
    comp = compounding_table(df)
    comp.to_csv(OUT_DIR / "compounding_table.csv", index=False)
    print(comp.to_string(index=False))
    plot_heatmaps(df, OUT_DIR / "bias_heatmaps.png")
