"""Exp 24: Site clustering in registry comparator — undercoverage demonstration.

The TVT Registry is site-clustered (TEER SAP: Makkar et al., JAMA 2023, GEE
for site clustering). This experiment shows that independence-assuming SEs on
clustered registry data produce anticonservative CIs; cluster-robust (bootstrap)
SEs recover nominal coverage.

Sweeps ICC ∈ {0.0, 0.05, 0.10, 0.20, 0.30} × n_sites ∈ {5, 10, 20}
Estimand: ATE (mean difference) in the main registry comparator arm.
Variance: independence SE vs cluster-bootstrap SE (B=200).
Reports: empirical coverage of 95% CI for each (ICC, n_sites) cell.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
from causal_bench.estimators.hierarchical import summarise_registry

# ── Configuration ─────────────────────────────────────────────────────────────

ICC_GRID    = [0.0, 0.05, 0.10, 0.20, 0.30]
NSITES_GRID = [5, 10, 20]
N_REPS      = 200
BOOTSTRAP_B = 200
N_MAIN      = 700
ALPHA       = 0.05

OUT_DIR = Path(__file__).parent.parent / "results" / "exp24_site_clustering"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Main simulation loop ──────────────────────────────────────────────────────

def _compute_estimand(icc: float, n_sites: int, n_large: int = 50_000) -> float:
    """Compute the large-sample limit of the difference-in-means ATE estimator.

    The registry DGP has mild confounding (W1, W2 affect both A and Y), so the
    difference-in-means E[Y|A=1] − E[Y|A=0] does not equal cfg.true_ate_main.
    We estimate the true *estimand* of the naive estimator via a large-n draw at
    the same (icc, n_sites) settings so site-level effects are properly averaged out.
    """
    cfg = RegistryConfig(n_main=n_large, n_sites=n_sites, icc=icc, seed=999_999)
    main_df, _, _, _ = generate_registry_data(cfg)
    t = main_df[main_df["A"] == 1]["Y"].mean()
    c = main_df[main_df["A"] == 0]["Y"].mean()
    return float(t - c)


def run_coverage_experiment() -> pd.DataFrame:
    """Sweep ICC × n_sites grid, compute empirical coverage for both SE estimators.

    For each (icc, n_sites) cell:
      - Compute the true *estimand* of the difference-in-means via a large-n draw
        (the DGP has mild confounding, so the estimand ≠ cfg.true_ate_main)
      - Draw N_REPS independent registry datasets from RegistryConfig(n_main=N_MAIN, ...)
      - Compute ATE + independence SE → 95% CI → check if estimand is covered
      - Compute ATE + cluster-bootstrap SE → 95% CI → check coverage
      - Record mean coverage for both estimators in the cell

    At ICC=0, both CIs should achieve ~95% coverage (valid variance estimators).
    At ICC>0, the independence CI undercovers while cluster-bootstrap recovers coverage.
    """
    from scipy.stats import norm
    z = norm.ppf(1.0 - ALPHA / 2.0)

    records = []
    n_cells = len(ICC_GRID) * len(NSITES_GRID)
    cell_num = 0

    for icc, n_sites in product(ICC_GRID, NSITES_GRID):
        cell_num += 1
        print(f"  Cell {cell_num}/{n_cells}: icc={icc:.2f}, n_sites={n_sites}  ", end="", flush=True)

        # True estimand of the difference-in-means at this (icc, n_sites) setting
        true_estimand = _compute_estimand(icc=icc, n_sites=n_sites)

        cover_indep  = 0
        cover_robust = 0
        ate_vals     = []

        for rep in range(N_REPS):
            seed = rep * 1000 + int(icc * 100) * 10 + n_sites
            cfg = RegistryConfig(
                n_main=N_MAIN,
                n_sites=n_sites,
                icc=icc,
                seed=seed,
            )
            main_df, _, _, _ = generate_registry_data(cfg)

            # Independence SE
            s_i = summarise_registry(main_df, true_estimand, "main", cluster_robust=False)
            ate = s_i.ate_hat
            ci_lo_i = ate - z * s_i.se_hat
            ci_hi_i = ate + z * s_i.se_hat
            if ci_lo_i <= true_estimand <= ci_hi_i:
                cover_indep += 1

            # Cluster-bootstrap SE
            s_r = summarise_registry(
                main_df, true_estimand, "main",
                cluster_robust=True,
                bootstrap_B=BOOTSTRAP_B,
                bootstrap_rng=np.random.default_rng(seed + 1),
            )
            ci_lo_r = ate - z * s_r.se_hat
            ci_hi_r = ate + z * s_r.se_hat
            if ci_lo_r <= true_estimand <= ci_hi_r:
                cover_robust += 1

            ate_vals.append(ate)

        cov_i = cover_indep  / N_REPS
        cov_r = cover_robust / N_REPS
        bias  = float(np.mean(ate_vals)) - true_estimand
        print(f"estimand={true_estimand:+.4f}  indep={cov_i:.3f}  robust={cov_r:.3f}  bias={bias:+.5f}")

        records.append({
            "icc":              icc,
            "n_sites":          n_sites,
            "n_main":           N_MAIN,
            "n_reps":           N_REPS,
            "true_estimand":    true_estimand,
            "coverage_indep":   cov_i,
            "coverage_robust":  cov_r,
            "ate_bias":         bias,
        })

    return pd.DataFrame(records)


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_coverage(df: pd.DataFrame, out_path: Path) -> None:
    """Coverage vs ICC for both estimators, one line per n_sites."""
    fig, axes = plt.subplots(1, len(NSITES_GRID), figsize=(5 * len(NSITES_GRID), 4),
                             sharey=True)
    if len(NSITES_GRID) == 1:
        axes = [axes]

    colors = {"coverage_indep": "#d62728", "coverage_robust": "#1f77b4"}
    labels = {"coverage_indep": "Independence SE", "coverage_robust": "Cluster-bootstrap SE"}

    for ax, n_sites in zip(axes, NSITES_GRID):
        sub = df[df["n_sites"] == n_sites].sort_values("icc")
        for col, color in colors.items():
            ax.plot(sub["icc"], sub[col], marker="o", color=color,
                    label=labels[col], linewidth=2, markersize=6)
        ax.axhline(0.95, color="k", linestyle="--", linewidth=1, label="Nominal 95%")
        ax.set_title(f"n_sites = {n_sites}", fontsize=13)
        ax.set_xlabel("ICC", fontsize=11)
        if n_sites == NSITES_GRID[0]:
            ax.set_ylabel("Empirical 95% CI coverage", fontsize=11)
        ax.set_ylim(0.4, 1.02)
        ax.set_xlim(-0.02, max(ICC_GRID) + 0.02)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Exp 24: Coverage vs ICC — {N_MAIN} main-registry patients, {N_REPS} reps\n"
        "Independence SE undercovers under site clustering; cluster-bootstrap recovers nominal",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    print(f"\nExp 24: Site clustering undercoverage demonstration")
    print(f"  Grid: ICC ∈ {ICC_GRID} × n_sites ∈ {NSITES_GRID}")
    print(f"  N_REPS={N_REPS}, N_MAIN={N_MAIN}, BOOTSTRAP_B={BOOTSTRAP_B}\n")

    df = run_coverage_experiment()

    csv_path = OUT_DIR / "coverage_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nCoverage table saved: {csv_path}")

    print("\nSummary table (coverage_indep vs coverage_robust):")
    pivot = df.pivot_table(
        index="icc", columns="n_sites",
        values=["coverage_indep", "coverage_robust"],
    )
    print(pivot.to_string())

    plot_path = OUT_DIR / "coverage_vs_icc.png"
    plot_coverage(df, plot_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
