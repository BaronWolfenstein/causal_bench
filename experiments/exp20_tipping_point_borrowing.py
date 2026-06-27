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

Additionally sweeps robust_weight (w axis, issue #22 item 2):
  robust_weight ∈ {0.1, 0.3, 0.5, 0.7, 0.9} — weight on the vague (robustifying)
  component; MAP component weight = 1 − robust_weight. Higher robust_weight = less
  borrowing from the informative prior.
  flip_robust_weight: minimum robust_weight at which the conclusion flips.
  Under conflict: flip happens at LOW robust_weight (prior props up a conclusion the
  data opposes; small down-weighting of prior lets data flip it immediately).
  Under concordance: flip happens at HIGH robust_weight or NaN (prior and data agree;
  conclusion is robust to prior down-weighting).

Outputs (results/exp20_tipping_point_borrowing/):
  tipping_surface_{teer,mac}.png     — 2D heatmap: red = conclusion flips
  ess_prior_curve.png                — ESS_prior vs tau_prior_sd at zero conflict
  flip_robust_weight_{teer,mac}.png  — flip_robust_weight vs conflict per τ
  results.parquet                    — full cell-level aggregates (τ×conflict×target)
  w_sweep_results.parquet            — robust_weight sweep cell-level aggregates
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

# Mixture-weight (w) sweep: MAP component weight = 1 - robust_weight (issue #22 item 2)
# Using a regular grid (model is analytical and fast; no QMC needed).
W_GRID             = [0.1, 0.3, 0.5, 0.7, 0.9]  # MAP component weight values
ROBUST_WEIGHT_GRID = [0.9, 0.7, 0.5, 0.3, 0.1]  # corresponding robust_weight values


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


def _run_cell_w(
    tau_prior_sd: float,
    conflict_strength: float,
    robust_weight: float,
    target: str,
    n_reps: int,
    base_seed: int,
) -> list[BorrowingResult]:
    """Run one cell of the robust_weight sweep (issue #22 item 2)."""
    results = []
    for rep in range(n_reps):
        cfg = RegistryConfig(
            conflict_strength=conflict_strength,
            tau_prior_sd=tau_prior_sd,
            robust_weight=robust_weight,
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
            robust_weight=robust_weight,
            vague_sd=cfg.vague_sd,
        )
        results.append(result)
    return results


def run_w_sweep(n_reps: int = N_REPS, seed: int = 42) -> pd.DataFrame:
    """Sweep robust_weight across the (τ, conflict, target) grid.

    For each (τ, conflict, target) cell, also finds the flip_robust_weight:
    the minimum robust_weight at which the conclusion rate drops below 0.5
    (i.e., the 95% CrI no longer reliably excludes null).
    """
    rows = []
    total = len(TAU_GRID) * len(CONFLICT_GRID) * len(TARGETS) * len(ROBUST_WEIGHT_GRID)
    done = 0

    for tau_prior_sd, conflict_strength, target in product(
        TAU_GRID, CONFLICT_GRID, TARGETS
    ):
        conclude_by_rw: dict[float, float] = {}
        for robust_weight in ROBUST_WEIGHT_GRID:
            results = _run_cell_w(tau_prior_sd, conflict_strength, robust_weight, target, n_reps, seed)
            oc = compute_oc_metrics(results, null_scenario=False)
            done += 1
            conclude_rate = oc.power if np.isfinite(oc.power) else float("nan")
            conclude_by_rw[robust_weight] = conclude_rate
            rows.append({
                "tau_prior_sd":      tau_prior_sd,
                "conflict_strength": conflict_strength,
                "target":            target,
                "robust_weight":     robust_weight,
                "map_component_w":   1.0 - robust_weight,
                "conclude_rate":     conclude_rate,
                "ess_prior_mean":    oc.ess_prior_mean,
                "map_weight_mean":   oc.map_weight_mean,
            })
            if done % 10 == 0:
                print(f"  [w-sweep {done:3d}/{total}] tau={tau_prior_sd:.2f}  "
                      f"conflict={conflict_strength:.1f}  target={target}  "
                      f"rw={robust_weight:.1f}  conclude={conclude_rate:.2f}")

    df = pd.DataFrame(rows)

    # Compute flip_robust_weight per (τ, conflict, target)
    flip_rows = []
    for tau_prior_sd, conflict_strength, target in product(TAU_GRID, CONFLICT_GRID, TARGETS):
        sub = df[
            (df["tau_prior_sd"] == tau_prior_sd)
            & (df["conflict_strength"] == conflict_strength)
            & (df["target"] == target)
        ].sort_values("robust_weight")
        # Find minimum robust_weight where conclude_rate < 0.5
        flipped = sub[sub["conclude_rate"] < 0.5]
        flip_rw = float(flipped["robust_weight"].min()) if not flipped.empty else float("nan")
        flip_rows.append({
            "tau_prior_sd":      tau_prior_sd,
            "conflict_strength": conflict_strength,
            "target":            target,
            "flip_robust_weight": flip_rw,
        })

    flip_df = pd.DataFrame(flip_rows)
    return df, flip_df


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
        cal_z_str = f"{oc.calibrated_z_mean:.3f}" if np.isfinite(oc.calibrated_z_mean) else "n/a"
        r_str     = f"{oc.r_ratio_mean:.2f}"      if np.isfinite(oc.r_ratio_mean)     else "n/a"
        print(
            f"[{done:3d}/{total}] tau={tau_prior_sd:.2f}  "
            f"conflict={conflict_strength:.1f}  target={target}  "
            f"conclude={oc.power:.2f}  "
            f"ESS_prior={oc.ess_prior_mean:.1f}  "
            f"map_w={oc.map_weight_mean:.2f}  "
            f"cal_z={cal_z_str}  r={r_str}"
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
            "calibrated_z_mean": oc.calibrated_z_mean,
            "r_ratio_mean":      oc.r_ratio_mean,
        })

    df = pd.DataFrame(rows)
    out_parquet = OUT_DIR / "results.parquet"
    df.to_parquet(out_parquet, index=False)
    print(f"\nSaved → {out_parquet}")

    _plot_tipping_surfaces(df)
    _plot_ess_curve(df)

    # ── Robust-weight sweep (issue #22 item 2) ──
    print("\nRunning robust_weight sweep...")
    w_df, flip_df = run_w_sweep(n_reps=n_reps, seed=seed)
    w_parquet = OUT_DIR / "w_sweep_results.parquet"
    w_df.to_parquet(w_parquet, index=False)
    print(f"Saved → {w_parquet}")

    flip_parquet = OUT_DIR / "flip_robust_weight.parquet"
    flip_df.to_parquet(flip_parquet, index=False)
    print(f"Saved → {flip_parquet}")

    _plot_flip_robust_weight(flip_df)

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


def _plot_flip_robust_weight(flip_df: pd.DataFrame) -> None:
    """Plot flip_robust_weight vs conflict_strength, per tau, for each target.

    flip_robust_weight is the minimum robust_weight at which the conclusion
    flips (conclude_rate < 0.5).  Higher flip_robust_weight = more robust
    conclusion (holds even when the prior is almost all vague).
    """
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(TAU_GRID)))
    for target in TARGETS:
        fig, ax = plt.subplots(figsize=(7, 5))
        sub = flip_df[flip_df["target"] == target]
        for tau, color in zip(TAU_GRID, colors):
            cell = sub[sub["tau_prior_sd"] == tau].sort_values("conflict_strength")
            ax.plot(
                cell["conflict_strength"],
                cell["flip_robust_weight"],
                "o-",
                color=color,
                label=f"τ={tau:.2f}",
            )
        ax.set_xlabel("Conflict strength")
        ax.set_ylabel("flip_robust_weight (min rw where conclusion flips)")
        ax.set_title(
            f"Exp 20: Mixture-weight tipping point — {target.upper()}\n"
            "Higher value = conclusion robust to less informative prior"
        )
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="rw=0.5 reference")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        save_path = OUT_DIR / f"flip_robust_weight_{target}.png"
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
