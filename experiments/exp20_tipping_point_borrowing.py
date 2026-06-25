"""Exp 20: Tipping-point sweep × borrowing strength.

Asks: for a fixed decision target (does the posterior CI exclude zero?),
how much comparator bias can the conclusion absorb at each level of
borrowing strength, and what effective sample size does the prior contribute?

This is the regulatory-grade sensitivity analysis that complements exp19's
OC study. Exp19 characterises OC breakdown as φ (embedding fidelity) and
conflict vary — it asks "when does borrowing break?" Exp20 holds the
decision target fixed and maps the 2D surface:

    (comparator_bias × borrowing_strength) → P(conclude)

so a reviewer can read off: "if the registry is biased at level δ, does
the conclusion hold at this borrowing level, and how many effective
registry patients does the prior contribute?"

Grid:
  tau_prior_sd      ∈ {0.02, 0.05, 0.10, 0.20, 0.40}
                      smaller = more informative prior = stronger borrowing
  conflict_strength ∈ {0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
                      0 = registry ATE matches main; 1 = opposite sign

Outputs (results/exp20_tipping_point_borrowing/):
  tipping_surface_{teer,mac}.png  — 2D heatmap: red = conclusion flips
  ess_prior_curve.png             — ESS_prior vs tau_prior_sd at zero conflict
  results.parquet                 — full cell-level aggregates
"""
from __future__ import annotations

from itertools import product
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
from causal_bench.estimators.hierarchical import (
    BorrowingResult,
    compute_oc_metrics,
    population_level_borrow,
    summarise_registry,
)

OUT_DIR = Path("results/exp20_tipping_point_borrowing")
N_REPS  = 100

TAU_GRID      = [0.02, 0.05, 0.10, 0.20, 0.40]
CONFLICT_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
TARGETS       = ["teer", "mac"]


def _run_cell(
    tau_prior_sd: float,
    conflict_strength: float,
    target: str,
    n_reps: int,
    base_seed: int,
) -> list[BorrowingResult]:
    results = []
    for rep in range(n_reps):
        cfg = RegistryConfig(
            conflict_strength=conflict_strength,
            tau_prior_sd=tau_prior_sd,
            seed=base_seed + rep,
        )
        main_df, teer_df, mac_df, _ = generate_registry_data(cfg)
        target_df  = teer_df if target == "teer" else mac_df
        target_ate = cfg.true_ate_teer if target == "teer" else cfg.true_ate_mac

        main_summ   = summarise_registry(main_df,   cfg.true_ate_main, "main")
        target_summ = summarise_registry(target_df, target_ate,        target)

        result = population_level_borrow(
            main_summary=main_summ,
            target_summary=target_summ,
            tau_prior_sd=tau_prior_sd,
            robust_weight=cfg.robust_weight,
            vague_sd=cfg.vague_sd,
        )
        results.append(result)
    return results


def run(n_reps: int = N_REPS, seed: int = 42) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    total = len(TAU_GRID) * len(CONFLICT_GRID) * len(TARGETS)
    done  = 0

    for tau_prior_sd, conflict_strength, target in product(
        TAU_GRID, CONFLICT_GRID, TARGETS
    ):
        results = _run_cell(tau_prior_sd, conflict_strength, target, n_reps, seed)
        oc = compute_oc_metrics(results, null_scenario=False)
        done += 1
        print(
            f"[{done:3d}/{total}] tau={tau_prior_sd:.2f}  "
            f"conflict={conflict_strength:.1f}  target={target}  "
            f"conclude={oc.power:.2f}  "
            f"ESS_prior={oc.ess_prior_mean:.1f}  "
            f"map_w={oc.map_weight_mean:.2f}"
        )
        rows.append({
            "tau_prior_sd":      tau_prior_sd,
            "conflict_strength": conflict_strength,
            "target":            target,
            "conclude_rate":     oc.power,
            "ess_prior_mean":    oc.ess_prior_mean,
            "ess_prior_sd":      oc.ess_prior_sd,
            "ess_total_mean":    oc.ess_total_mean,
            "map_weight_mean":   oc.map_weight_mean,
            "coverage":          oc.coverage,
            "mde":               oc.mde,
        })

    df = pd.DataFrame(rows)
    out_parquet = OUT_DIR / "results.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\nSaved → {out_parquet}")

    _plot_tipping_surfaces(df)
    _plot_ess_curve(df)
    return df


def _plot_tipping_surfaces(df: pd.DataFrame) -> None:
    for target in TARGETS:
        sub = df[df["target"] == target]
        pivot = sub.pivot(
            index="tau_prior_sd",
            columns="conflict_strength",
            values="conclude_rate",
        )
        fig, ax = plt.subplots(figsize=(8, 5))
        im = ax.imshow(
            pivot.values,
            aspect="auto",
            origin="lower",
            vmin=0.0,
            vmax=1.0,
            cmap="RdYlGn",
        )
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f"{v:.1f}" for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
        ax.set_xlabel("Comparator bias (conflict_strength)")
        ax.set_ylabel("tau_prior_sd (smaller = stronger borrowing)")
        ax.set_title(
            f"Exp 20: Tipping-point surface — {target.upper()}\n"
            "P(conclude | H₁) — green = robust, red = conclusion flips"
        )
        for i in range(len(pivot.index)):
            for j in range(len(pivot.columns)):
                val = pivot.values[i, j]
                ax.text(
                    j, i, f"{val:.2f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if 0.25 < val < 0.75 else "white",
                )
        plt.colorbar(im, ax=ax, label="P(conclude)")
        plt.tight_layout()
        save_path = OUT_DIR / f"tipping_surface_{target}.png"
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        print(f"Saved → {save_path}")


def _plot_ess_curve(df: pd.DataFrame) -> None:
    """ESS_prior vs tau_prior_sd at zero conflict."""
    zero_conf = df[df["conflict_strength"] == 0.0]
    fig, ax = plt.subplots(figsize=(6, 4))
    for target in TARGETS:
        sub = zero_conf[zero_conf["target"] == target].sort_values("tau_prior_sd")
        ax.errorbar(
            sub["tau_prior_sd"],
            sub["ess_prior_mean"],
            yerr=sub["ess_prior_sd"],
            marker="o",
            label=target.upper(),
            capsize=4,
        )
    ax.set_xlabel("tau_prior_sd (smaller = more informative prior)")
    ax.set_ylabel("ESS_prior (patients)")
    ax.set_title(
        "Exp 20: ESS of the borrowed prior vs borrowing strength\n"
        "(at zero comparator conflict)"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    save_path = OUT_DIR / "ess_prior_curve.png"
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 20: Tipping-point × borrowing strength")
    p.add_argument("--n-reps", type=int, default=N_REPS)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_reps=args.n_reps, seed=args.seed)
