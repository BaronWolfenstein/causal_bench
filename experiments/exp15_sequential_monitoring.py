"""Exp 15: Sequential CED monitoring — anytime-valid vs alpha-spending vs naive.

Simulates post-coverage surveillance: data accumulates annually (400 patients/year
over 5 years, edwards_realistic parameters), and the treatment effect is
re-estimated at each annual look. Three treatment effect trajectories test
whether each monitoring approach correctly maintains or relinquishes rejection:

  stable      — constant ATE -0.12 throughout follow-up
  degrading   — ATE shrinks 3pp/year (-0.12, -0.09, -0.06, -0.03, 0.00)
  step_change — ATE vanishes at year 3 (-0.12, -0.12, 0.00, 0.00, 0.00)
  null        — true ATE = 0 throughout (type I error calibration)

Three monitoring approaches:
  naive              — TMLE+IPCW at alpha=0.05 at each look, no correction
  obf                — O'Brien-Fleming alpha spending (conservative early)
  confidence_sequence — Howard et al. (2021) mixture martingale (anytime-valid)

This is the differentiating capability concrete doesn't have:
concrete = point-in-time ("what is the treatment effect?")
Exp 15  = sequential ("is the treatment effect STILL what we thought?")
CED requires annual updates; naive re-testing inflates false positives.

Reference: Howard et al. (2021). Time-uniform, nonparametric, nonasymptotic
confidence sequences. Annals of Statistics 49(2): 1055-1083.
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from causal_bench.dgp.config import DGPConfig, CovariateDependentCensoringConfig
from causal_bench.dgp.survival import generate_data, compute_true_effects
from causal_bench.dgp.scenarios import get_scenario
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator
from causal_bench.sequential import apply_sequential_methods, SequentialResult

N_PER_YEAR = 400
N_YEARS = 5
OUT_DIR = Path("results/exp15_sequential_monitoring")
N_SIMS = 200  # increase to 500 for publication

_EDWARDS_BASE = dict(
    censoring=CovariateDependentCensoringConfig(informativeness=0.6), censoring_rate=0.25,
    positivity_severity=1.5, crossover_rate=0.05,
    unmeasured_confounding_strength=0.2,
    collider_strength=0.4, enrollment_drift=0.15,
    outcome_nonlinearity=0.5, effect_heterogeneity=0.3,
)

_TAU_SEQUENCES = {
    "stable":      [-0.12] * N_YEARS,
    "degrading":   [-0.12 + 0.03 * k for k in range(N_YEARS)],
    "step_change": [-0.12 if k < 3 else 0.0 for k in range(N_YEARS)],
    "null":        [0.0] * N_YEARS,
}

_ESTIMATOR = TMLEIPCWEstimator(use_compliance=False)


def _true_ate(tau: float) -> float:
    cfg = DGPConfig(n=50_000, seed=0, true_tau=tau, **_EDWARDS_BASE)
    return compute_true_effects(cfg)["ATE"]


def _run_one_sim(trajectory: str, sim_seed: int) -> Optional[dict[str, SequentialResult]]:
    tau_seq = _TAU_SEQUENCES[trajectory]
    rng = np.random.default_rng(sim_seed)
    cohorts = []
    for year_idx, tau in enumerate(tau_seq):
        cfg = DGPConfig(
            n=N_PER_YEAR,
            seed=int(rng.integers(0, 2**31)),
            true_tau=tau,
            **_EDWARDS_BASE,
        )
        df = generate_data(cfg)
        df = df.copy()
        df["enrollment_year"] = year_idx + 1
        cohorts.append(df)

    estimates, ses = [], []
    for k in range(1, N_YEARS + 1):
        combined = pd.concat(cohorts[:k], ignore_index=True)
        try:
            results = _ESTIMATOR.estimate(combined, horizon=1.0)
            if not results:
                return None
            r = results[0]
            estimates.append(r.point_estimate)
            ses.append(r.standard_error if np.isfinite(r.standard_error) and r.standard_error > 0 else 0.1)
        except Exception:
            return None

    true_values = [_true_ate(tau) for tau in tau_seq]
    return apply_sequential_methods(
        estimates=estimates,
        ses=ses,
        true_values=true_values,
        trajectory=trajectory,
        K=N_YEARS,
    )


def _aggregate(sim_results: list[dict[str, SequentialResult]]) -> dict:
    """Aggregate across sims: coverage by look, rejection rate, false rejection."""
    methods = ("naive", "obf", "confidence_sequence")
    agg = {m: {
        "coverage_by_look": [[] for _ in range(N_YEARS)],
        "reject_by_look": [[] for _ in range(N_YEARS)],
        "ever_rejected": [],
        "false_rejection": [],
        "first_rejection_look": [],
    } for m in methods}

    for sim in sim_results:
        if sim is None:
            continue
        for m in methods:
            r = sim[m]
            for k in range(N_YEARS):
                covered = r.ci_lowers[k] <= r.true_values[k] <= r.ci_uppers[k]
                agg[m]["coverage_by_look"][k].append(covered)
                agg[m]["reject_by_look"][k].append(r.rejects_null[k])
            agg[m]["ever_rejected"].append(r.ever_rejected)
            agg[m]["false_rejection"].append(r.false_rejection)
            if r.first_rejection_look is not None:
                agg[m]["first_rejection_look"].append(r.first_rejection_look)

    summary = {}
    for m in methods:
        a = agg[m]
        summary[m] = {
            "coverage_by_look": [np.mean(v) for v in a["coverage_by_look"]],
            "reject_rate_by_look": [np.mean(v) for v in a["reject_by_look"]],
            "ever_rejected_rate": np.mean(a["ever_rejected"]) if a["ever_rejected"] else float("nan"),
            "false_rejection_rate": np.mean(a["false_rejection"]) if a["false_rejection"] else float("nan"),
            "median_first_rejection": (float(np.median(a["first_rejection_look"]))
                                       if a["first_rejection_look"] else float("nan")),
        }
    return summary


def _plot_coverage(agg_by_traj: dict, save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    looks = list(range(1, N_YEARS + 1))
    colors = {"naive": "tab:red", "obf": "tab:blue", "confidence_sequence": "tab:green"}
    labels = {"naive": "Naive", "obf": "OBF", "confidence_sequence": "Conf. Sequence"}

    for ax, traj in zip(axes, ("stable", "null")):
        agg = agg_by_traj[traj]
        for m, color in colors.items():
            cov = agg[m]["coverage_by_look"]
            ax.plot(looks, cov, marker="o", color=color, label=labels[m])
        ax.axhline(0.95, color="black", linestyle="--", linewidth=0.8, label="0.95 target")
        ax.set_xlabel("Annual look")
        ax.set_ylabel("Coverage")
        ax.set_title(f"Coverage by look — {traj} trajectory")
        ax.set_ylim(0.7, 1.02)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / "coverage_by_look.png", dpi=150)
    plt.close(fig)


def _plot_detection(agg_by_traj: dict, save_dir: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    looks = list(range(1, N_YEARS + 1))
    colors = {"naive": "tab:red", "obf": "tab:blue", "confidence_sequence": "tab:green"}
    labels = {"naive": "Naive", "obf": "OBF", "confidence_sequence": "Conf. Sequence"}

    for ax, traj in zip(axes, ("stable", "degrading", "step_change")):
        agg = agg_by_traj[traj]
        for m, color in colors.items():
            rej = agg[m]["reject_rate_by_look"]
            ax.plot(looks, rej, marker="o", color=color, label=labels[m])
        ax.axhline(0.80, color="gray", linestyle=":", linewidth=0.8)
        ax.set_xlabel("Annual look")
        ax.set_ylabel("Cumulative P(reject)")
        ax.set_title(f"Detection — {traj}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(save_dir / "detection_by_trajectory.png", dpi=150)
    plt.close(fig)


def _plot_type1(agg_by_traj: dict, save_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    looks = list(range(1, N_YEARS + 1))
    colors = {"naive": "tab:red", "obf": "tab:blue", "confidence_sequence": "tab:green"}
    labels = {"naive": "Naive", "obf": "OBF", "confidence_sequence": "Conf. Sequence"}
    agg = agg_by_traj["null"]
    for m, color in colors.items():
        rej = agg[m]["reject_rate_by_look"]
        ax.plot(looks, rej, marker="o", color=color, label=labels[m])
    ax.axhline(0.05, color="black", linestyle="--", linewidth=0.8, label="α=0.05")
    ax.set_xlabel("Annual look")
    ax.set_ylabel("False positive rate")
    ax.set_title("Type I error (null trajectory, true ATE=0)")
    ax.set_ylim(-0.01, 0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "type1_error.png", dpi=150)
    plt.close(fig)


def _plot_width(single_sim_results: dict[str, dict[str, SequentialResult]],
                save_dir: Path) -> None:
    """Width comparison across methods for one representative sim (stable)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    looks = list(range(1, N_YEARS + 1))
    colors = {"naive": "tab:red", "obf": "tab:blue", "confidence_sequence": "tab:green"}
    labels = {"naive": "Naive (±1.96 SE)", "obf": "OBF boundary",
              "confidence_sequence": "Conf. Sequence"}
    sim = single_sim_results["stable"]
    for m, color in colors.items():
        r = sim[m]
        widths = [hi - lo for lo, hi in zip(r.ci_lowers, r.ci_uppers)]
        ax.plot(looks, widths, marker="o", color=color, label=labels[m])
    ax.set_xlabel("Annual look")
    ax.set_ylabel("CI width")
    ax.set_title("CI width comparison — stable trajectory (one sim)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_dir / "width_comparison.png", dpi=150)
    plt.close(fig)


def run(n_sims: int = N_SIMS, seed: int = 42) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    trajectories = list(_TAU_SEQUENCES)
    all_results: dict[str, list] = {t: [] for t in trajectories}
    single_sim: dict[str, dict] = {}  # one representative sim per trajectory

    print(f"Exp 15: Sequential CED monitoring | n_sims={n_sims} | "
          f"n_per_year={N_PER_YEAR} | K={N_YEARS}")

    for traj in trajectories:
        print(f"  trajectory={traj}")
        for i in range(n_sims):
            sim_seed = int(rng.integers(0, 2**31))
            res = _run_one_sim(traj, sim_seed)
            all_results[traj].append(res)
            if i == 0 and res is not None:
                single_sim[traj] = res

    agg_by_traj = {t: _aggregate(all_results[t]) for t in trajectories}

    _plot_coverage(agg_by_traj, OUT_DIR)
    _plot_detection(agg_by_traj, OUT_DIR)
    _plot_type1(agg_by_traj, OUT_DIR)
    if single_sim:
        _plot_width(single_sim, OUT_DIR)

    print(f"\n── Type I error (null trajectory) ──────────────")
    for m in ("naive", "obf", "confidence_sequence"):
        fr = agg_by_traj["null"][m]["false_rejection_rate"]
        print(f"  {m:25s}: {fr:.3f}")

    print(f"\n── Coverage at final look (stable) ────────────")
    for m in ("naive", "obf", "confidence_sequence"):
        cov = agg_by_traj["stable"][m]["coverage_by_look"][-1]
        print(f"  {m:25s}: {cov:.3f}")

    print(f"\nSaved plots → {OUT_DIR}/")
    return agg_by_traj


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 15: Sequential CED monitoring")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, seed=args.seed)
