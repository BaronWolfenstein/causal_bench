"""Exp 19: Hierarchical borrowing operating-characteristics study.

Answers: "Does patient-level similarity-weighted borrowing deliver real precision
gains, or does it import bias when embedding proximity ≠ effect similarity?"

The decisive lever is φ (embedding_fidelity) ∈ [0, 1]:
  φ = 1  embedding proximity perfectly proxies true CATE similarity → borrowing ideal
  φ = 0  embedding proximity is uninformative → patient-level borrowing is pure noise

Three borrowing levels evaluated at each φ:
  population  — single τ² shrinkage, robust MAP prior; no embedding dependency
  subgroup    — CATE-stratified K-means subgroups, per-subgroup MAP prior (CMS layer)
  patient     — continuous similarity-weighted borrowing; φ-dependent (internal layer)

Scenario grid (fully crossed):
  φ ∈ {0.0, 0.25, 0.50, 0.75, 1.0}       — embedding fidelity sweep (key)
  conflict ∈ {0.0, 0.5, 1.0}              — prior-data conflict strength
  scenario ∈ {"alternative", "null"}       — H₁ (true ATE = −0.12) vs H₀ (true ATE = 0)

Outputs (one plot per decisive question):
  type_m_type_s.png   — Type M / Type S vs φ, by level and conflict
  mde_by_level.png    — MDE vs φ, population vs patient
  ess_conflict.png    — ESS prior mean and MAP weight collapse under conflict
  coverage.png        — 95% CI coverage vs φ (should be ≥ 0.95)
  power_curve.png     — power vs φ, alternative scenario

The study quantifies the φ regime where patient-level borrowing is defensible.
Below the crossover point (patient Type M > population Type M), fall back to
population-level as the honest claimable MDE ceiling.

The φ proxy diagnostic (cross-validated embedding-predicts-effect concordance)
should be estimated from the main cohort before relying on patient-level
borrowing for anything regulatory. See hierarchical.py for ESS collapse as
the runtime conflict diagnostic.
"""
from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import warnings

from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
from causal_bench.estimators.hierarchical import (
    summarise_registry,
    population_level_borrow,
    patient_level_borrow,
    compute_oc_metrics,
    BorrowingResult,
    OCMetrics,
)
from causal_bench.estimators.subgroup import (
    discover_subgroups,
    subgroup_level_borrow,
    SubgroupBorrowingResult,
)

OUT_DIR = Path("results/exp19_hierarchical_oc")
N_REPS = 200   # increase to 1000+ for final OC report (rare cohort: high-variance)
N_SUBGROUPS = 4  # target subgroups for subgroup-level borrowing

# Scenario grid
PHI_VALUES       = (0.0, 0.25, 0.50, 0.75, 1.0)
CONFLICT_VALUES  = (0.0, 0.5, 1.0)
SCENARIOS        = ("alternative", "null")
LEVELS           = ("population", "subgroup", "patient")
TARGET_REGISTRIES = ("teer", "mac")


def _base_config(
    phi: float,
    conflict: float,
    scenario: str,
    seed: int,
) -> RegistryConfig:
    true_ate = -0.12 if scenario == "alternative" else 0.0
    return RegistryConfig(
        true_ate_main=true_ate,
        conflict_strength=conflict,
        embedding_fidelity=phi,
        seed=seed,
    )


def _run_one_rep(
    phi: float,
    conflict: float,
    scenario: str,
    seed: int,
) -> dict[str, dict[str, BorrowingResult]]:
    """One replication: generate data, run all levels × all target registries."""
    cfg = _base_config(phi, conflict, scenario, seed)
    main_df, teer_df, mac_df, embeddings = generate_registry_data(cfg)

    true_ate = cfg.true_ate_main
    main_sum  = summarise_registry(main_df,  true_ate,                   "main")
    teer_sum  = summarise_registry(teer_df,  cfg.true_ate_teer,          "teer")  # type: ignore[arg-type]
    mac_sum   = summarise_registry(mac_df,   cfg.true_ate_mac,           "mac")   # type: ignore[arg-type]

    results: dict[str, dict[str, BorrowingResult]] = {
        level: {} for level in LEVELS
    }

    # Subgroup model: fit once per rep from main cohort, shared across targets
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        subgroup_model = discover_subgroups(
            main_df, embeddings["main"],
            n_subgroups=N_SUBGROUPS,
            classifier="knn",
        )

    for target_name, target_sum, target_df, target_emb, target_true in [
        ("teer", teer_sum, teer_df, embeddings["teer"], cfg.true_ate_teer),
        ("mac",  mac_sum,  mac_df,  embeddings["mac"],  cfg.true_ate_mac),
    ]:
        results["population"][target_name] = population_level_borrow(
            main_summary=main_sum,
            target_summary=target_sum,
            tau_prior_sd=cfg.tau_prior_sd,
            robust_weight=cfg.robust_weight,
            vague_sd=cfg.vague_sd,
        )

        # Subgroup level: aggregate across subgroups to a single population-style
        # BorrowingResult via ESS-weighted mean for OC comparisons.
        sub_results = subgroup_level_borrow(
            main_df, target_df,
            embeddings["main"], target_emb,
            subgroup_model,
            target_true_ate=target_true,
            tau_prior_sd=cfg.tau_prior_sd,
            robust_weight=cfg.robust_weight,
            vague_sd=cfg.vague_sd,
        )
        results["subgroup"][target_name] = _aggregate_subgroup_results(
            sub_results, target_name, target_true,
        )

        results["patient"][target_name] = patient_level_borrow(
            main_df=main_df,
            target_df=target_df,
            main_emb=embeddings["main"],
            target_emb=target_emb,
            target_true_ate=target_true,
            config=cfg,
        )

    return results


def _aggregate_subgroup_results(
    sub_results: list[SubgroupBorrowingResult],
    target_registry: str,
    true_ate: float,
    alpha: float = 0.05,
) -> BorrowingResult:
    """ESS-weighted aggregate of per-subgroup BorrowingResults.

    Aggregation rule: weight each subgroup's posterior by its ESS_total
    (information content). Pooled ATE is ESS-weighted mean; pooled variance
    is ESS-weighted mean of posterior variances (conservative — ignores
    between-subgroup heterogeneity for the OC comparison metric).
    """
    from scipy.stats import norm as _norm

    if not sub_results:
        # All subgroups degenerate — fall back to NaN BorrowingResult
        return BorrowingResult(
            level="subgroup", target_registry=target_registry,
            ate_posterior=float("nan"), se_posterior=float("nan"),
            ci_lower=float("nan"), ci_upper=float("nan"),
            ess_prior=0.0, ess_data=0.0, ess_total=0.0,
            map_weight=float("nan"),
            rejects_null=False, covers_truth=False, true_ate=true_ate,
        )

    weights = np.array([r.borrowing.ess_total for r in sub_results])
    weights = np.maximum(weights, 1e-8)
    w_norm  = weights / weights.sum()

    ate_pooled = float(np.sum(w_norm * [r.borrowing.ate_posterior for r in sub_results]))
    var_pooled = float(np.sum(w_norm * [r.borrowing.se_posterior ** 2 for r in sub_results]))
    se_pooled  = float(np.sqrt(max(var_pooled, 1e-12)))

    z = _norm.ppf(1.0 - alpha / 2.0)
    ci_lo = ate_pooled - z * se_pooled
    ci_hi = ate_pooled + z * se_pooled

    ess_prior = sum(r.borrowing.ess_prior for r in sub_results)
    ess_data  = sum(r.borrowing.ess_data  for r in sub_results)
    map_w_avg = float(np.mean([r.borrowing.map_weight for r in sub_results]))

    return BorrowingResult(
        level="subgroup",
        target_registry=target_registry,
        ate_posterior=ate_pooled,
        se_posterior=se_pooled,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ess_prior=ess_prior,
        ess_data=ess_data,
        ess_total=ess_prior + ess_data,
        map_weight=map_w_avg,
        rejects_null=bool(abs(ate_pooled / max(se_pooled, 1e-12)) > z),
        covers_truth=bool(ci_lo <= true_ate <= ci_hi),
        true_ate=true_ate,
    )


def run_scenario_cell(
    phi: float,
    conflict: float,
    scenario: str,
    n_reps: int,
    base_seed: int = 0,
) -> dict[str, dict[str, list[BorrowingResult]]]:
    """Run n_reps replications for one (φ, conflict, scenario) cell.

    Returns dict[level][target_registry] → list of BorrowingResult.
    """
    collected: dict[str, dict[str, list[BorrowingResult]]] = {
        level: {reg: [] for reg in TARGET_REGISTRIES} for level in LEVELS  # population/subgroup/patient
    }
    rng = np.random.default_rng(base_seed + int(phi * 1000) + int(conflict * 100))
    seeds = [int(rng.integers(0, 2**31)) for _ in range(n_reps)]

    for seed in seeds:
        try:
            rep = _run_one_rep(phi, conflict, scenario, seed)
            for level in LEVELS:
                for reg in TARGET_REGISTRIES:
                    if reg in rep[level]:
                        collected[level][reg].append(rep[level][reg])
        except Exception:
            pass  # skip failed reps (rare cohort edge cases)

    return collected


# ─── Plots ────────────────────────────────────────────────────────────────────

_LEVEL_STYLE = {
    "population": ("tab:blue",   "o-",  "Population-level"),
    "subgroup":   ("tab:green",  "^-",  "Subgroup-level (CMS)"),
    "patient":    ("tab:orange", "s--", "Patient-level"),
}
_CONFLICT_ALPHA = {0.0: 1.0, 0.5: 0.6, 1.0: 0.3}


def plot_type_m_type_s(
    oc_grid: dict,
    target: str,
    save_path: str,
) -> None:
    """Type M and Type S vs φ, per level and conflict (alternative scenario)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for level, (color, ls, label) in _LEVEL_STYLE.items():
        for conflict in CONFLICT_VALUES:
            alpha_val = _CONFLICT_ALPHA[conflict]
            phi_vals, type_m_vals, type_s_vals = [], [], []
            for phi in PHI_VALUES:
                key = (phi, conflict, "alternative")
                if key not in oc_grid:
                    continue
                m = oc_grid[key][level][target]
                if np.isnan(m.type_m):
                    continue
                phi_vals.append(phi)
                type_m_vals.append(m.type_m)
                type_s_vals.append(m.type_s)

            lbl = f"{label} (conflict={conflict:.1f})"
            axes[0].plot(phi_vals, type_m_vals, ls, color=color, alpha=alpha_val, label=lbl)
            axes[1].plot(phi_vals, type_s_vals, ls, color=color, alpha=alpha_val, label=lbl)

    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle=":")
    axes[0].set_xlabel("Embedding fidelity φ")
    axes[0].set_ylabel("Type M (exaggeration ratio)")
    axes[0].set_title(f"Type M error vs φ — {target.upper()} registry")
    axes[0].legend(fontsize=7)
    axes[0].grid(alpha=0.3)

    axes[1].axhline(0.0, color="black", linewidth=0.8, linestyle=":")
    axes[1].set_xlabel("Embedding fidelity φ")
    axes[1].set_ylabel("Type S (sign error rate | significant)")
    axes[1].set_title(f"Type S error vs φ — {target.upper()} registry")
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)

    fig.suptitle(
        f"Exp 19: Type M / Type S — where does patient-level borrowing bias exceed population-level?",
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_mde(oc_grid: dict, target: str, save_path: str) -> None:
    """MDE vs φ by borrowing level (alternative scenario, no conflict)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    conflict = 0.0

    for level, (color, ls, label) in _LEVEL_STYLE.items():
        phi_vals, mde_vals = [], []
        for phi in PHI_VALUES:
            key = (phi, conflict, "alternative")
            if key not in oc_grid:
                continue
            m = oc_grid[key][level][target]
            if not np.isnan(m.mde):
                phi_vals.append(phi)
                mde_vals.append(m.mde)
        ax.plot(phi_vals, mde_vals, ls, color=color, label=label)

    ax.set_xlabel("Embedding fidelity φ")
    ax.set_ylabel("MDE (80% power)")
    ax.set_title(
        f"Exp 19: MDE vs φ — {target.upper()} registry (no conflict)\n"
        "Honest claimable MDE bounded by level where borrowing assumption holds"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_ess_conflict(oc_grid: dict, target: str, save_path: str) -> None:
    """ESS prior mean and MAP weight as conflict increases (φ=1.0, alternative)."""
    fig, (ax_ess, ax_mw) = plt.subplots(1, 2, figsize=(11, 4))
    phi = 1.0

    for level, (color, ls, label) in _LEVEL_STYLE.items():
        conf_vals, ess_vals, mw_vals = [], [], []
        for conflict in CONFLICT_VALUES:
            key = (phi, conflict, "alternative")
            if key not in oc_grid:
                continue
            m = oc_grid[key][level][target]
            conf_vals.append(conflict)
            ess_vals.append(m.ess_prior_mean)
            mw_vals.append(m.map_weight_mean)
        ax_ess.plot(conf_vals, ess_vals, ls, color=color, label=label, marker="o")
        ax_mw.plot(conf_vals, mw_vals,  ls, color=color, label=label, marker="o")

    ax_ess.set_xlabel("Prior-data conflict strength")
    ax_ess.set_ylabel("Mean ESS contributed by prior")
    ax_ess.set_title(f"ESS collapse under conflict — {target.upper()} (φ=1)")
    ax_ess.legend(fontsize=8)
    ax_ess.grid(alpha=0.3)

    ax_mw.axhline(0.10, color="gray", linestyle=":", linewidth=0.8, label="Robust weight floor")
    ax_mw.set_xlabel("Prior-data conflict strength")
    ax_mw.set_ylabel("MAP component weight (posterior)")
    ax_mw.set_title("Robust MAP auto-discount under conflict")
    ax_mw.legend(fontsize=8)
    ax_mw.grid(alpha=0.3)

    fig.suptitle("Exp 19: ESS and MAP weight collapse — robust prior behavior under conflict", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_coverage(oc_grid: dict, target: str, save_path: str) -> None:
    """95% CI coverage vs φ, by level and conflict."""
    fig, ax = plt.subplots(figsize=(8, 5))

    for level, (color, ls, label) in _LEVEL_STYLE.items():
        for conflict in CONFLICT_VALUES:
            a = _CONFLICT_ALPHA[conflict]
            phi_vals, cov_vals = [], []
            for phi in PHI_VALUES:
                key = (phi, conflict, "alternative")
                if key not in oc_grid:
                    continue
                m = oc_grid[key][level][target]
                if not np.isnan(m.coverage):
                    phi_vals.append(phi)
                    cov_vals.append(m.coverage)
            ax.plot(phi_vals, cov_vals, ls, color=color, alpha=a,
                    label=f"{label} (conflict={conflict:.1f})")

    ax.axhline(0.95, color="black", linewidth=0.8, linestyle="--", label="Nominal 95%")
    ax.set_ylim(0.7, 1.02)
    ax.set_xlabel("Embedding fidelity φ")
    ax.set_ylabel("95% CI coverage")
    ax.set_title(f"Exp 19: Coverage vs φ — {target.upper()} registry")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_power(oc_grid: dict, target: str, save_path: str) -> None:
    """Power vs φ, no conflict (alternative scenario)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    conflict = 0.0

    for level, (color, ls, label) in _LEVEL_STYLE.items():
        phi_vals, power_vals = [], []
        for phi in PHI_VALUES:
            key = (phi, conflict, "alternative")
            if key not in oc_grid:
                continue
            m = oc_grid[key][level][target]
            if not np.isnan(m.power):
                phi_vals.append(phi)
                power_vals.append(m.power)
        ax.plot(phi_vals, power_vals, ls, color=color, label=label, marker="o")

    ax.axhline(0.80, color="gray", linestyle=":", linewidth=0.8, label="80% target")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Embedding fidelity φ")
    ax.set_ylabel("Power")
    ax.set_title(
        f"Exp 19: Power vs φ — {target.upper()} registry (no conflict)\n"
        "Patient-level should exceed population-level when φ > crossover"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(n_reps: int = N_REPS, seed: int = 42) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Exp 19: Hierarchical borrowing OC study")
    print(f"  grid: {len(PHI_VALUES)} φ × {len(CONFLICT_VALUES)} conflict × "
          f"{len(SCENARIOS)} scenarios × {len(TARGET_REGISTRIES)} registries")
    print(f"  n_reps={n_reps} per cell  (total cells: "
          f"{len(PHI_VALUES)*len(CONFLICT_VALUES)*len(SCENARIOS)})")

    # Run grid
    oc_grid: dict = {}   # (phi, conflict, scenario) → {level: {registry: OCMetrics}}

    total_cells = len(PHI_VALUES) * len(CONFLICT_VALUES) * len(SCENARIOS)
    cell_idx = 0
    for phi, conflict, scenario in product(PHI_VALUES, CONFLICT_VALUES, SCENARIOS):
        cell_idx += 1
        print(f"  [{cell_idx}/{total_cells}] φ={phi:.2f}  conflict={conflict:.1f}  "
              f"scenario={scenario}", end="  ", flush=True)

        collected = run_scenario_cell(phi, conflict, scenario, n_reps, base_seed=seed)

        oc_by_level: dict[str, dict[str, OCMetrics]] = {}
        for level in LEVELS:
            oc_by_level[level] = {}
            for reg in TARGET_REGISTRIES:
                rlist = collected[level][reg]
                oc_by_level[level][reg] = compute_oc_metrics(
                    rlist,
                    null_scenario=(scenario == "null"),
                )

        oc_grid[(phi, conflict, scenario)] = oc_by_level

        # Quick summary for this cell
        teer_pop = oc_by_level["population"]["teer"]
        teer_pat = oc_by_level["patient"]["teer"]
        stat = "type1" if scenario == "null" else "power"
        pop_val = teer_pop.type1_error if scenario == "null" else teer_pop.power
        pat_val = teer_pat.type1_error if scenario == "null" else teer_pat.power
        print(f"TEER {stat}: pop={pop_val:.2f}  pat={pat_val:.2f}  "
              f"ESS_prior(pop)={teer_pop.ess_prior_mean:.1f}")

    # ── Print summary tables ──
    print("\n── Type M summary (alternative, TEER, no conflict) ──")
    print(f"  {'φ':>5}  {'pop_TypeM':>10}  {'pat_TypeM':>10}  {'pop_MDE':>10}  {'pat_MDE':>10}")
    for phi in PHI_VALUES:
        key = (phi, 0.0, "alternative")
        if key not in oc_grid:
            continue
        pop = oc_grid[key]["population"]["teer"]
        pat = oc_grid[key]["patient"]["teer"]
        print(f"  {phi:5.2f}  {pop.type_m:10.3f}  {pat.type_m:10.3f}  "
              f"{pop.mde:10.3f}  {pat.mde:10.3f}")

    print("\n── ESS collapse under conflict (φ=1.0, TEER, alternative) ──")
    print(f"  {'conflict':>8}  {'pop_ESS_prior':>14}  {'pat_ESS_prior':>14}  "
          f"{'pop_MAP_w':>10}  {'pat_MAP_w':>10}")
    for conflict in CONFLICT_VALUES:
        key = (1.0, conflict, "alternative")
        if key not in oc_grid:
            continue
        pop = oc_grid[key]["population"]["teer"]
        pat = oc_grid[key]["patient"]["teer"]
        print(f"  {conflict:8.1f}  {pop.ess_prior_mean:14.1f}  {pat.ess_prior_mean:14.1f}  "
              f"{pop.map_weight_mean:10.3f}  {pat.map_weight_mean:10.3f}")

    # ── Plots ──
    for target in TARGET_REGISTRIES:
        sfx = target
        plot_type_m_type_s(oc_grid, target, str(OUT_DIR / f"type_m_type_s_{sfx}.png"))
        plot_mde(oc_grid, target,           str(OUT_DIR / f"mde_by_level_{sfx}.png"))
        plot_ess_conflict(oc_grid, target,  str(OUT_DIR / f"ess_conflict_{sfx}.png"))
        plot_coverage(oc_grid, target,      str(OUT_DIR / f"coverage_{sfx}.png"))
        plot_power(oc_grid, target,         str(OUT_DIR / f"power_curve_{sfx}.png"))

    print(f"\nAll outputs → {OUT_DIR}/")
    return oc_grid


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 19: Hierarchical borrowing OC study")
    p.add_argument("--n-reps", type=int, default=N_REPS)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_reps=args.n_reps, seed=args.seed)
