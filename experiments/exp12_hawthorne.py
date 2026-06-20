"""Exp 12: Hawthorne Decomposition — durable vs transient monitoring artifact.

Answers: "Of the outcome improvement at app-deployed sites, how much is durable
process improvement and how much is a transient Hawthorne monitoring effect?"

Why concrete can't do this:
  concrete estimates treatment effects in cross-sectional or survival settings.
  Hawthorne decomposition requires PANEL DATA with staggered deployment —
  sites adopt monitoring at different times, and the comparison is within-site
  before/after deployment vs never-deployed sites. This is difference-in-
  differences methodology, not TMLE.

DGP:
  20 sites × 12 monthly periods. 80% of sites deploy between periods 3-8
  (staggered). Outcomes follow:
    Y_st = baseline_s + secular_trend * t + durable * compliance_st
           + hawthorne * 2^(-t_since / halflife) + learning * min(t_since, plateau)

Estimators:
  naive_twfe          — TWFE; biased under heterogeneous staggered adoption
  event_study_twfe    — TWFE with relative-time dummies; shows the Hawthorne arc
  dchd_dynamic        — De Chaisemartin-D'Haultfoeuille group-time ATTs
  callaway_santhanna  — DR group-time ATTs (robustness check vs dchd)
  twfe_with_calendar  — TWFE + calendar time covariate (Senn fix for secular trend)

Sweeps:
  1. hawthorne_halflife: 1 to 6 periods (how fast Hawthorne fades)
  2. secular_trend: 0 to -0.05 (how much time-trend confounds TWFE)
  3. compliance_steady_state: 0.3 to 0.9 (dose-response)

Outputs:
  event_study.png        — ATT by event-time: Hawthorne arc + decay to durable
  decomposition.png      — stacked bar: durable + Hawthorne at each event-time
  neg_weight.png         — TWFE negative weight diagnostic
  secular_panel.png      — TWFE vs robust DiD bias as secular_trend increases
  dose_response.png      — compliance vs outcome (durable effect slope)
  concordance.png        — dchd_dynamic vs callaway_santhanna at each event-time
"""
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from causal_bench.dgp.hawthorne import (
    HawthorneConfig,
    generate_hawthorne_data,
    true_effect_decomposition,
)
from causal_bench.estimators.hawthorne import (
    run_all_hawthorne_estimators,
    naive_twfe,
    event_study_twfe,
    dchd_dynamic,
    callaway_santhanna,
    twfe_with_calendar,
    HawthorneEstimate,
)

OUT_DIR = Path("results/exp12_hawthorne")
N_SIMS = 50   # increase to 200 for publication


def _run_sim(config: HawthorneConfig, seed: int) -> dict[str, HawthorneEstimate]:
    cfg = config.model_copy(update={"seed": seed})
    df = generate_hawthorne_data(cfg)
    return run_all_hawthorne_estimators(df)


def _aggregate_event_ates(
    sim_results: list[dict[str, HawthorneEstimate]],
    method: str,
) -> dict[int, dict[str, float]]:
    """Aggregate event-time ATEs across sims: mean and se per event-time."""
    all_event: dict[int, list[float]] = {}
    for r in sim_results:
        est = r[method]
        for et, ate in est.event_time_ates.items():
            all_event.setdefault(et, []).append(ate)
    return {
        et: {"mean": float(np.mean(vals)), "se": float(np.std(vals) / np.sqrt(len(vals)))}
        for et, vals in all_event.items()
    }


# ─── Plots ────────────────────────────────────────────────────────────────────

def plot_event_study(
    config: HawthorneConfig,
    sim_results: list[dict[str, HawthorneEstimate]],
    save_path: str,
) -> None:
    """ATT by event-time: Hawthorne arc decaying toward durable effect."""
    true_decomp = true_effect_decomposition(config, max_event_time=config.n_periods - 2)
    true_total = {int(row.event_time): row.true_total for _, row in true_decomp.iterrows()}

    fig, ax = plt.subplots(figsize=(9, 5))

    colors = {
        "event_study_twfe":   ("tab:red",   "Event-study TWFE"),
        "dchd_dynamic":       ("tab:blue",  "DCHD dynamic"),
        "callaway_santhanna": ("tab:green", "Callaway-Sant'Anna"),
    }
    for method, (color, label) in colors.items():
        agg = _aggregate_event_ates(sim_results, method)
        ets = sorted(agg)
        means = [agg[et]["mean"] for et in ets]
        ses = [agg[et]["se"] for et in ets]
        ax.errorbar(ets, means, yerr=[1.96 * s for s in ses],
                    marker="o", color=color, label=label, capsize=3, linewidth=1.5)

    # True values
    ets_true = sorted(true_total)
    ax.plot(ets_true, [true_total[et] for et in ets_true],
            "k--", linewidth=1.5, label="True total effect")
    durable_asymptote = config.durable_effect * config.compliance_steady_state
    ax.axhline(durable_asymptote, color="gray", linestyle=":", linewidth=0.8,
               label=f"Durable asymptote ({durable_asymptote:.3f})")
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_xlabel("Event-time (periods since deployment)")
    ax.set_ylabel("Estimated ATT")
    ax.set_title(f"Exp 12: Event-study plot — Hawthorne halflife={config.hawthorne_halflife}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_decomposition(config: HawthorneConfig, save_path: str) -> None:
    """Stacked bar: durable + Hawthorne + learning at each event-time."""
    true_decomp = true_effect_decomposition(config, max_event_time=config.n_periods - 2)
    ets = true_decomp["event_time"].tolist()

    fig, ax = plt.subplots(figsize=(9, 5))

    durable = true_decomp["durable_comp"].tolist()
    hawthorne = true_decomp["hawthorne_comp"].tolist()
    learning = true_decomp["learning_comp"].tolist()

    ax.bar(ets, durable, label="Durable effect", color="tab:blue", alpha=0.8)
    ax.bar(ets, hawthorne, bottom=durable, label="Hawthorne effect", color="tab:orange", alpha=0.8)
    ax.bar(ets, learning, bottom=[d + h for d, h in zip(durable, hawthorne)],
           label="Learning curve", color="tab:green", alpha=0.8)
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xlabel("Event-time (periods since deployment)")
    ax.set_ylabel("Effect magnitude")
    ax.set_title(
        f"Exp 12: True effect decomposition — halflife={config.hawthorne_halflife} periods\n"
        f"Hawthorne fades; durable + learning persist"
    )
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_neg_weight(config: HawthorneConfig, n_sims: int, save_path: str) -> None:
    """TWFE negative weight fraction: naive TWFE vs robust estimators."""
    rng = np.random.default_rng(config.seed)
    naive_betas, dchd_betas, neg_fracs = [], [], []

    for i in range(n_sims):
        seed = int(rng.integers(0, 2**31))
        cfg = config.model_copy(update={"seed": seed})
        df = generate_hawthorne_data(cfg)
        est_naive = naive_twfe(df)
        est_dchd = dchd_dynamic(df)
        naive_betas.append(est_naive.beta)
        dchd_betas.append(est_naive.beta)   # same overall beta, different weights
        neg_fracs.append(est_naive.neg_weight_fraction)

    # True long-run effect (after Hawthorne fades)
    true_durable = config.durable_effect * config.compliance_steady_state

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.hist(neg_fracs, bins=20, color="tab:red", edgecolor="white", alpha=0.8)
    ax1.set_xlabel("Fraction of obs with negative TWFE weight")
    ax1.set_ylabel("Simulation count")
    ax1.set_title("TWFE negative weight distribution\n(higher = more contamination)")
    ax1.axvline(np.mean(neg_fracs), color="black", linestyle="--",
                label=f"Mean={np.mean(neg_fracs):.2f}")
    ax1.legend()

    ax2.scatter(neg_fracs, naive_betas, alpha=0.5, color="tab:red", s=15, label="Naive TWFE")
    ax2.axhline(true_durable, color="black", linestyle="--",
                label=f"True durable ({true_durable:.3f})")
    ax2.set_xlabel("Negative weight fraction")
    ax2.set_ylabel("TWFE β estimate")
    ax2.set_title("Negative weight fraction vs TWFE bias")
    ax2.legend(fontsize=8)

    fig.suptitle("Exp 12: TWFE negative weight diagnostic (staggered adoption)", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_secular_panel(
    secular_levels: tuple,
    sim_results_by_level: dict,
    config: HawthorneConfig,
    save_path: str,
) -> None:
    """TWFE vs robust DiD bias as secular_trend increases."""
    methods = {
        "naive_twfe":         ("tab:red",   "Naive TWFE"),
        "twfe_with_calendar": ("tab:purple", "TWFE + calendar (Senn)"),
        "dchd_dynamic":       ("tab:blue",  "DCHD dynamic"),
    }

    fig, (ax_bias, ax_est) = plt.subplots(1, 2, figsize=(12, 5))

    for method, (color, label) in methods.items():
        betas = [np.mean([r[method].beta for r in sim_results_by_level[sl]])
                 for sl in secular_levels]
        true_target = config.durable_effect * config.compliance_steady_state
        biases = [b - true_target for b in betas]
        ax_bias.plot(secular_levels, biases, marker="o", color=color, label=label)
        ax_est.plot(secular_levels, betas, marker="o", color=color, label=label)

    ax_bias.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bias.set_xlabel("Secular trend (outcome improvement per period)")
    ax_bias.set_ylabel("Bias (estimated − true durable)")
    ax_bias.set_title("Secular trend confounding bias\n(TWFE conflates trend with app effect)")
    ax_bias.legend(fontsize=8)
    ax_bias.grid(alpha=0.3)

    ax_est.axhline(config.durable_effect * config.compliance_steady_state,
                   color="black", linestyle="--", linewidth=0.8, label="True durable")
    ax_est.set_xlabel("Secular trend")
    ax_est.set_ylabel("β estimate")
    ax_est.set_title("Estimated overall ATT vs secular trend")
    ax_est.legend(fontsize=8)
    ax_est.grid(alpha=0.3)

    fig.suptitle("Exp 12: Secular trend confounding — TWFE vs robust DiD", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_dose_response(
    compliance_levels: tuple,
    config: HawthorneConfig,
    n_sims: int,
    save_path: str,
) -> None:
    """Compliance steady state vs long-run outcome: should recover durable_effect slope."""
    compliance_ates = {cl: [] for cl in compliance_levels}
    rng = np.random.default_rng(config.seed + 999)

    for cl in compliance_levels:
        for _ in range(n_sims):
            seed = int(rng.integers(0, 2**31))
            cfg = config.model_copy(update={
                "compliance_steady_state": cl,
                "seed": seed,
            })
            df = generate_hawthorne_data(cfg)
            est = dchd_dynamic(df)
            # Use the final event-time ATT (approximately durable + small hawthorne)
            max_et = max(est.event_time_ates, default=None)
            if max_et is not None:
                compliance_ates[cl].append(est.event_time_ates[max_et])

    cl_vals = list(compliance_levels)
    mean_ates = [np.mean(compliance_ates[cl]) if compliance_ates[cl] else np.nan
                 for cl in cl_vals]
    # True dose-response: durable_effect * compliance_steady_state (at large t)
    true_dr = [config.durable_effect * cl for cl in cl_vals]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(cl_vals, mean_ates, "o-", color="tab:blue", label="DCHD dynamic (final event-time)")
    ax.plot(cl_vals, true_dr, "k--", linewidth=1.5,
            label=f"True: {config.durable_effect:.2f} × compliance")
    ax.set_xlabel("Compliance steady state")
    ax.set_ylabel("Estimated ATT at final event-time")
    ax.set_title(
        "Exp 12: Dose-response — compliance vs long-run outcome\n"
        f"Slope should ≈ durable_effect = {config.durable_effect}"
    )
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_concordance(
    sim_results: list[dict[str, HawthorneEstimate]],
    config: HawthorneConfig,
    save_path: str,
) -> None:
    """DCHD vs Callaway-Sant'Anna at each event-time — agreement check."""
    agg_dchd = _aggregate_event_ates(sim_results, "dchd_dynamic")
    agg_csa = _aggregate_event_ates(sim_results, "callaway_santhanna")
    true_decomp = true_effect_decomposition(config, max_event_time=config.n_periods - 2)
    true_total = {int(row.event_time): row.true_total for _, row in true_decomp.iterrows()}

    ets = sorted(set(agg_dchd) & set(agg_csa) & set(true_total))
    if not ets:
        print("No overlapping event times for concordance plot — skipping")
        return

    dchd_means = [agg_dchd[et]["mean"] for et in ets]
    csa_means = [agg_csa[et]["mean"] for et in ets]
    true_vals = [true_total[et] for et in ets]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(ets, dchd_means, "o-", color="tab:blue", label="DCHD dynamic")
    ax1.plot(ets, csa_means, "s-", color="tab:green", label="Callaway-Sant'Anna")
    ax1.plot(ets, true_vals, "k--", linewidth=1.5, label="True total effect")
    ax1.set_xlabel("Event-time")
    ax1.set_ylabel("ATT")
    ax1.set_title("DCHD vs Callaway-Sant'Anna concordance\n(agreement = robust finding)")
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    # Scatter: DCHD vs CS (should be on y=x if they agree)
    ax2.scatter(dchd_means, csa_means, color="steelblue", alpha=0.8, s=50)
    for i, et in enumerate(ets):
        ax2.annotate(f"t={et}", (dchd_means[i], csa_means[i]),
                     fontsize=7, ha="left", va="bottom")
    lo = min(min(dchd_means), min(csa_means)) - 0.005
    hi = max(max(dchd_means), max(csa_means)) + 0.005
    ax2.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y=x (perfect concordance)")
    ax2.set_xlabel("DCHD dynamic ATT")
    ax2.set_ylabel("Callaway-Sant'Anna ATT")
    ax2.set_title("Concordance scatter by event-time")
    ax2.legend(fontsize=8)

    fig.suptitle("Exp 12: DCHD vs Callaway-Sant'Anna concordance check", fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ─── Sweeps ───────────────────────────────────────────────────────────────────

def sweep_halflife(
    halflife_values: tuple,
    base_config: HawthorneConfig,
    n_sims: int,
) -> dict[float, list]:
    print("\nSweep 1: hawthorne_halflife")
    rng = np.random.default_rng(base_config.seed)
    results = {}
    for hl in halflife_values:
        cfg = base_config.model_copy(update={"hawthorne_halflife": hl})
        sims = []
        for _ in range(n_sims):
            seed = int(rng.integers(0, 2**31))
            sims.append(_run_sim(cfg, seed))
        results[hl] = sims
        mean_beta = np.mean([r["naive_twfe"].beta for r in sims])
        true_dur = cfg.durable_effect * cfg.compliance_steady_state
        print(f"  halflife={hl:.1f}  naive_twfe_beta={mean_beta:.4f}  "
              f"true_durable={true_dur:.4f}")
    return results


def sweep_secular(
    secular_values: tuple,
    base_config: HawthorneConfig,
    n_sims: int,
) -> dict[float, list]:
    print("\nSweep 2: secular_trend")
    rng = np.random.default_rng(base_config.seed + 100)
    results = {}
    for sl in secular_values:
        cfg = base_config.model_copy(update={"secular_trend": sl})
        sims = []
        for _ in range(n_sims):
            seed = int(rng.integers(0, 2**31))
            sims.append(_run_sim(cfg, seed))
        results[sl] = sims
        mean_naive = np.mean([r["naive_twfe"].beta for r in sims])
        mean_senn = np.mean([r["twfe_with_calendar"].beta for r in sims])
        true_dur = cfg.durable_effect * cfg.compliance_steady_state
        print(f"  secular={sl:.3f}  naive={mean_naive:.4f}  "
              f"senn={mean_senn:.4f}  true_durable={true_dur:.4f}")
    return results


def sweep_compliance(
    compliance_values: tuple,
    base_config: HawthorneConfig,
    n_sims: int,
) -> dict[float, list]:
    print("\nSweep 3: compliance_steady_state")
    rng = np.random.default_rng(base_config.seed + 200)
    results = {}
    for cl in compliance_values:
        cfg = base_config.model_copy(update={"compliance_steady_state": cl})
        sims = []
        for _ in range(n_sims):
            seed = int(rng.integers(0, 2**31))
            sims.append(_run_sim(cfg, seed))
        results[cl] = sims
    return results


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(n_sims: int = N_SIMS, seed: int = 42) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    base_config = HawthorneConfig(seed=seed)

    print("Exp 12: Hawthorne Decomposition")
    print(f"  n_sites={base_config.n_sites}  n_periods={base_config.n_periods}  "
          f"n_sims={n_sims}")
    print(f"  durable={base_config.durable_effect}  "
          f"hawthorne={base_config.hawthorne_effect}  "
          f"halflife={base_config.hawthorne_halflife}")

    # ── Base-case event study and decomposition ──
    print("\nGenerating base-case event study")
    rng = np.random.default_rng(seed)
    base_sims = []
    for _ in range(n_sims):
        s = int(rng.integers(0, 2**31))
        base_sims.append(_run_sim(base_config, s))

    plot_event_study(base_config, base_sims, str(OUT_DIR / "event_study.png"))
    plot_decomposition(base_config, str(OUT_DIR / "decomposition.png"))
    plot_concordance(base_sims, base_config, str(OUT_DIR / "concordance.png"))
    plot_neg_weight(base_config, n_sims=min(n_sims, 30), save_path=str(OUT_DIR / "neg_weight.png"))

    # ── Sweep 1: halflife ──
    hl_values = (1.0, 2.0, 3.0, 4.0, 6.0)
    hl_results = sweep_halflife(hl_values, base_config, n_sims=n_sims)

    # Event-study per halflife (one panel per halflife value)
    fig_hl, axes_hl = plt.subplots(1, len(hl_values), figsize=(5 * len(hl_values), 4),
                                    sharey=True)
    for ax, hl in zip(axes_hl, hl_values):
        sims = hl_results[hl]
        cfg_hl = base_config.model_copy(update={"hawthorne_halflife": hl})
        agg = _aggregate_event_ates(sims, "dchd_dynamic")
        true_decomp = true_effect_decomposition(cfg_hl, max_event_time=cfg_hl.n_periods - 2)
        true_total = {int(r.event_time): r.true_total for _, r in true_decomp.iterrows()}
        ets = sorted(agg)
        ax.errorbar(ets, [agg[et]["mean"] for et in ets],
                    yerr=[1.96 * agg[et]["se"] for et in ets],
                    marker="o", color="tab:blue", capsize=2, linewidth=1.2)
        ax.plot(sorted(true_total), [true_total[et] for et in sorted(true_total)],
                "k--", linewidth=1.0)
        ax.axhline(cfg_hl.durable_effect * cfg_hl.compliance_steady_state,
                   color="gray", linestyle=":", linewidth=0.8)
        ax.set_title(f"halflife={hl}")
        ax.set_xlabel("Event-time")
        if ax is axes_hl[0]:
            ax.set_ylabel("ATT (DCHD dynamic)")
    fig_hl.suptitle("Exp 12: Hawthorne halflife sweep — event study by halflife", fontsize=10)
    fig_hl.tight_layout()
    hl_path = str(OUT_DIR / "halflife_sweep.png")
    fig_hl.savefig(hl_path, dpi=150)
    plt.close(fig_hl)
    print(f"Saved → {hl_path}")

    # ── Sweep 2: secular trend ──
    secular_values = (0.0, -0.01, -0.02, -0.03, -0.04, -0.05)
    sec_results = sweep_secular(secular_values, base_config, n_sims=n_sims)
    plot_secular_panel(secular_values, sec_results, base_config,
                       str(OUT_DIR / "secular_panel.png"))

    # ── Sweep 3: dose-response ──
    compliance_values = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    plot_dose_response(compliance_values, base_config, n_sims=min(n_sims, 30),
                       save_path=str(OUT_DIR / "dose_response.png"))

    # ── Print summary table ──
    print("\n── Base-case beta estimates (true durable = "
          f"{base_config.durable_effect * base_config.compliance_steady_state:.4f}) ──")
    for method in ("naive_twfe", "event_study_twfe", "dchd_dynamic",
                   "callaway_santhanna", "twfe_with_calendar"):
        betas = [r[method].beta for r in base_sims if not np.isnan(r[method].beta)]
        if betas:
            print(f"  {method:22s}: mean={np.mean(betas):+.4f}  "
                  f"SD={np.std(betas):.4f}")

    print(f"\nAll outputs → {OUT_DIR}/")
    return {
        "base_sims": base_sims,
        "hl_results": hl_results,
        "sec_results": sec_results,
    }


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 12: Hawthorne Decomposition")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, seed=args.seed)
