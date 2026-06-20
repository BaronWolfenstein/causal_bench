"""Exp 11: Transport Decomposition — trial-to-commercial generalizability.

Answers: "Does the treatment effect estimated in the trial apply to the
commercial population, or does the population shift change the answer?"

Why concrete can't do this:
  concrete estimates the treatment effect WITHIN a dataset. Transport asks
  whether that effect TRANSFERS to a different population with different
  covariate distributions. This requires two datasets (trial + target) and
  density ratio weighting / outcome-model extrapolation.

DGP:
  Trial patients are sicker (W1_mean=0.5) with a narrower covariate range.
  Commercial patients have the broader real-world distribution (W1_mean=0).
  Treatment effect varies with W1 (transport_heterogeneity) and can follow
  two patterns:
    symmetric:  aggregate ATEs agree, quantile-level ATEs diverge (GALILEO)
    asymmetric: aggregate ATEs differ systematically

Estimators:
  naive           — just reuse the trial ATE
  ipsw            — inverse probability of sampling weighting
  g_transport     — outcome regression, predict at commercial covariates
  dr_transport    — doubly-robust combination of the two

Outputs:
  transport_bias.png          — ATE_trial vs ATE_commercial across heterogeneity
  quantile_heatmap.png        — ATE divergence by W1 quantile and method
  overlap_diagnostic.png      — sampling weight distribution (trial → commercial)
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from causal_bench.dgp.transport import (
    TransportConfig,
    generate_transport_data,
    compute_true_ates,
)
from causal_bench.estimators.transport import (
    run_all_transport_estimators,
    transport_ipsw,
    transport_quantile,
)

OUT_DIR = Path("results/exp11_transport")
N_SIMS = 100


def _run_sim(config: TransportConfig, seed_offset: int) -> dict:
    """Run one simulation, return method → (trial_ate, estimated_commercial_ate)."""
    cfg = config.model_copy(update={"seed": config.seed + seed_offset})
    trial_df, commercial_df = generate_transport_data(cfg)
    ests = run_all_transport_estimators(trial_df, commercial_df)
    return {name: (e.trial_ate, e.commercial_ate) for name, e in ests.items()}


def sweep_heterogeneity(
    heterogeneity_levels: tuple = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
    divergence_pattern: str = "asymmetric",
    n_sims: int = N_SIMS,
    base_config: TransportConfig | None = None,
) -> dict:
    """Sweep transport_heterogeneity and collect mean estimated commercial ATE."""
    if base_config is None:
        base_config = TransportConfig()

    results: dict[float, dict[str, list]] = {h: {} for h in heterogeneity_levels}

    for h in heterogeneity_levels:
        cfg = base_config.model_copy(update={
            "transport_heterogeneity": h,
            "divergence_pattern": divergence_pattern,
        })
        true_ates = compute_true_ates(cfg)
        print(f"  hetero={h:.1f}  true_trial={true_ates['trial_ate']:+.3f}  "
              f"true_commercial={true_ates['commercial_ate']:+.3f}")

        for method in ("naive", "ipsw", "g_transport", "dr_transport"):
            results[h][method] = []
        results[h]["_true_trial"] = true_ates["trial_ate"]
        results[h]["_true_commercial"] = true_ates["commercial_ate"]

        for sim in range(n_sims):
            res = _run_sim(cfg, sim)
            for method, (_, comm_ate) in res.items():
                if method in results[h]:
                    results[h][method].append(comm_ate)

    return results


def sweep_divergence_pattern(
    patterns: tuple = ("none", "symmetric", "asymmetric"),
    transport_heterogeneity: float = 0.7,
    n_sims: int = N_SIMS,
    base_config: TransportConfig | None = None,
) -> dict:
    """Sweep divergence_pattern and compare estimated vs true commercial ATE."""
    if base_config is None:
        base_config = TransportConfig()

    results: dict[str, dict] = {}
    for pattern in patterns:
        cfg = base_config.model_copy(update={
            "transport_heterogeneity": transport_heterogeneity,
            "divergence_pattern": pattern,
        })
        true_ates = compute_true_ates(cfg)
        print(f"  pattern={pattern:12s}  true_trial={true_ates['trial_ate']:+.3f}  "
              f"true_commercial={true_ates['commercial_ate']:+.3f}")

        results[pattern] = {
            "_true": true_ates,
            "estimates": {m: [] for m in ("naive", "ipsw", "g_transport", "dr_transport")},
        }
        for sim in range(n_sims):
            res = _run_sim(cfg, sim)
            for method, (_, comm_ate) in res.items():
                if method in results[pattern]["estimates"]:
                    results[pattern]["estimates"][method].append(comm_ate)

    return results


# ─── Plots ────────────────────────────────────────────────────────────────────

_METHOD_COLORS = {
    "naive":       "tab:red",
    "ipsw":        "tab:blue",
    "g_transport": "tab:orange",
    "dr_transport": "tab:green",
}
_METHOD_LABELS = {
    "naive":       "Naive (trial ATE)",
    "ipsw":        "IPSW",
    "g_transport": "G-transport",
    "dr_transport": "DR-transport",
}


def plot_transport_bias(sweep_results: dict, save_path: str) -> None:
    """ATE_trial vs ATE_commercial for each method across heterogeneity levels."""
    h_levels = sorted(k for k in sweep_results if isinstance(k, float))
    true_trial = [sweep_results[h]["_true_trial"] for h in h_levels]
    true_comm = [sweep_results[h]["_true_commercial"] for h in h_levels]

    fig, (ax_bias, ax_ate) = plt.subplots(1, 2, figsize=(13, 5))

    # Left: estimated commercial ATE vs true commercial ATE
    ax_ate.plot(h_levels, true_comm, "k--", linewidth=1.5, label="True commercial ATE")
    ax_ate.plot(h_levels, true_trial, "k:", linewidth=1.0, label="True trial ATE")
    for method, color in _METHOD_COLORS.items():
        ates = [np.mean(sweep_results[h][method]) for h in h_levels]
        ax_ate.plot(h_levels, ates, marker="o", color=color, label=_METHOD_LABELS[method])
    ax_ate.set_xlabel("Transport heterogeneity")
    ax_ate.set_ylabel("ATE (commercial population)")
    ax_ate.set_title("Estimated vs true commercial ATE")
    ax_ate.legend(fontsize=8)
    ax_ate.grid(alpha=0.3)

    # Right: transport bias (estimated commercial - true commercial)
    for method, color in _METHOD_COLORS.items():
        bias = [np.mean(sweep_results[h][method]) - sweep_results[h]["_true_commercial"]
                for h in h_levels]
        ax_bias.plot(h_levels, bias, marker="o", color=color, label=_METHOD_LABELS[method])
    ax_bias.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_bias.set_xlabel("Transport heterogeneity")
    ax_bias.set_ylabel("Bias (estimated - true commercial ATE)")
    ax_bias.set_title("Transport bias by method")
    ax_bias.legend(fontsize=8)
    ax_bias.grid(alpha=0.3)

    fig.suptitle("Exp 11: Transport Decomposition — Heterogeneity Sweep (asymmetric pattern)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_quantile_heatmap(
    config: TransportConfig,
    n_sims: int = 20,
    save_path: str | None = None,
) -> None:
    """Quantile divergence heatmap: ATE at each W1 quantile, trial vs commercial."""
    quantiles = (0.10, 0.25, 0.50, 0.75, 0.90)
    methods = ("trial", "commercial")

    # Aggregate quantile ATEs across sims
    q_ates: dict[str, dict[float, list]] = {
        m: {q: [] for q in quantiles} for m in methods
    }

    for sim in range(n_sims):
        cfg = config.model_copy(update={"seed": config.seed + sim})
        trial_df, commercial_df = generate_transport_data(cfg)
        q_results = transport_quantile(trial_df, commercial_df, quantiles)
        for q, res in q_results.items():
            if not np.isnan(res["trial_ate"]):
                q_ates["trial"][q].append(res["trial_ate"])
            if not np.isnan(res["commercial_ate"]):
                q_ates["commercial"][q].append(res["commercial_ate"])

    true_ates = compute_true_ates(config)
    trial_mean_ate = true_ates["trial_ate"]
    commercial_mean_ate = true_ates["commercial_ate"]

    # Build divergence matrix: rows=quantile, cols=method
    n_q = len(quantiles)
    mat = np.zeros((n_q, 2))
    row_labels = [f"W1 p{int(q*100)}" for q in quantiles]
    for i, q in enumerate(quantiles):
        mat[i, 0] = np.nanmean(q_ates["trial"][q]) if q_ates["trial"][q] else np.nan
        mat[i, 1] = np.nanmean(q_ates["commercial"][q]) if q_ates["commercial"][q] else np.nan

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))

    # Heatmap of ATE by quantile
    ax = axes[0]
    vmax = max(abs(np.nanmin(mat)), abs(np.nanmax(mat)))
    im = ax.imshow(mat, cmap="RdYlGn", aspect="auto",
                   vmin=-vmax, vmax=vmax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Trial", "Commercial"])
    ax.set_yticks(range(n_q))
    ax.set_yticklabels(row_labels)
    ax.set_title("ATE by W1 quantile and population\n(green = treatment benefit)")
    plt.colorbar(im, ax=ax)
    for i in range(n_q):
        for j in range(2):
            v = mat[i, j]
            ax.text(j, i, f"{v:+.3f}" if not np.isnan(v) else "—",
                    ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > vmax * 0.5 else "black")

    # Divergence (commercial - trial) by quantile
    ax2 = axes[1]
    divs = mat[:, 1] - mat[:, 0]  # commercial - trial
    colors = ["tab:green" if d >= 0 else "tab:red" for d in divs]
    ax2.barh(row_labels, divs, color=colors)
    ax2.axvline(0, color="black", linewidth=0.8)
    ax2.axvline(commercial_mean_ate - trial_mean_ate, color="gray",
                linestyle="--", linewidth=0.8, label="Aggregate divergence")
    ax2.set_xlabel("ATE(commercial) − ATE(trial) by W1 quantile")
    ax2.set_title("Quantile-level divergence\n(GALILEO pattern: where do they disagree?)")
    ax2.legend(fontsize=8)

    pattern = config.divergence_pattern
    hetero = config.transport_heterogeneity
    fig.suptitle(f"Exp 11: Quantile Divergence — {pattern} pattern, "
                 f"hetero={hetero:.1f}", fontsize=10)
    fig.tight_layout()
    if save_path is None:
        save_path = str(OUT_DIR / "quantile_heatmap.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


def plot_overlap_diagnostic(
    config: TransportConfig,
    save_path: str | None = None,
) -> None:
    """Histogram of sampling weights — extreme weights = positivity violation."""
    trial_df, commercial_df = generate_transport_data(config)
    est = transport_ipsw(trial_df, commercial_df)

    # Re-compute weights for plotting
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _COVARIATES = ["W1", "W2", "W3", "W4"]

    trial_X = trial_df[_COVARIATES].values
    comm_X = commercial_df[_COVARIATES].values
    X_all = np.vstack([trial_X, comm_X])
    S_all = np.concatenate([np.ones(len(trial_df)), np.zeros(len(commercial_df))])
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_all)
    lr = LogisticRegression(max_iter=500)
    lr.fit(X_sc, S_all)
    p_trial = np.clip(lr.predict_proba(scaler.transform(trial_X))[:, 1], 0.02, 0.98)
    w = (1 - p_trial) / p_trial
    w /= w.mean()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.hist(w, bins=40, color="steelblue", edgecolor="white", alpha=0.8)
    ax1.axvline(1.0, color="black", linestyle="--", linewidth=0.8, label="Weight=1 (no reweighting)")
    ax1.axvline(5.0, color="red", linestyle=":", linewidth=0.8, label="5× threshold")
    ax1.set_xlabel("IPSW sampling weight")
    ax1.set_ylabel("Count (trial patients)")
    ax1.set_title("Sampling weight distribution (trial → commercial)")
    ax1.legend(fontsize=8)

    # W1 distributions: trial vs commercial
    ax2.hist(trial_df["W1"], bins=40, alpha=0.6, color="tab:blue", label="Trial", density=True)
    ax2.hist(commercial_df["W1"], bins=40, alpha=0.6, color="tab:orange", label="Commercial", density=True)
    ax2.set_xlabel("W1 (primary effect modifier)")
    ax2.set_ylabel("Density")
    ax2.set_title("Covariate overlap: W1 distribution shift")
    ax2.legend()

    ess = float((w.sum() ** 2) / (w ** 2).sum())
    fig.suptitle(
        f"Exp 11: Overlap Diagnostic — {config.divergence_pattern} pattern, "
        f"hetero={config.transport_heterogeneity:.1f}\n"
        f"ESS={ess:.0f}/{len(trial_df)} trial patients (higher = better overlap)",
        fontsize=9,
    )
    fig.tight_layout()
    if save_path is None:
        save_path = str(OUT_DIR / "overlap_diagnostic.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {save_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def run(n_sims: int = N_SIMS, seed: int = 42) -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Exp 11: Transport Decomposition")
    print(f"  n_sims={n_sims} per configuration")

    base_config = TransportConfig(seed=seed)

    # ── Sweep 1: heterogeneity levels (asymmetric pattern) ──
    print("\nSweep 1: transport_heterogeneity × asymmetric pattern")
    sweep1 = sweep_heterogeneity(
        heterogeneity_levels=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        divergence_pattern="asymmetric",
        n_sims=n_sims,
        base_config=base_config,
    )
    plot_transport_bias(sweep1, str(OUT_DIR / "transport_bias.png"))

    # Print summary table
    print("\n── Transport bias summary (asymmetric, bias = estimated - true commercial) ──")
    h_levels = sorted(k for k in sweep1 if isinstance(k, float))
    header = f"  {'hetero':>8} {'true_comm':>10} " + " ".join(
        f"{m:>14}" for m in ("naive", "ipsw", "g_transport", "dr_transport")
    )
    print(header)
    for h in h_levels:
        true_c = sweep1[h]["_true_commercial"]
        biases = {m: np.mean(sweep1[h][m]) - true_c
                  for m in ("naive", "ipsw", "g_transport", "dr_transport")}
        print(f"  {h:8.1f} {true_c:10.3f} " +
              " ".join(f"{b:+14.3f}" for b in biases.values()))

    # ── Sweep 2: divergence patterns (heterogeneity=0.7) ──
    print("\nSweep 2: divergence_pattern (hetero=0.7)")
    sweep2 = sweep_divergence_pattern(
        patterns=("none", "symmetric", "asymmetric"),
        transport_heterogeneity=0.7,
        n_sims=n_sims,
        base_config=base_config,
    )
    print("\n── Divergence pattern summary (hetero=0.7) ──")
    for pat, res in sweep2.items():
        true_t = res["_true"]["trial_ate"]
        true_c = res["_true"]["commercial_ate"]
        print(f"  {pat:12s}  true_trial={true_t:+.3f}  true_commercial={true_c:+.3f}")
        for method, ates in res["estimates"].items():
            bias = np.mean(ates) - true_c
            print(f"    {method:14s}: mean_est={np.mean(ates):+.3f}  bias={bias:+.3f}")

    # ── Quantile heatmap: symmetric pattern at high heterogeneity ──
    print("\nQuantile heatmap: symmetric pattern, hetero=0.8")
    cfg_sym = base_config.model_copy(update={
        "transport_heterogeneity": 0.8,
        "divergence_pattern": "symmetric",
    })
    plot_quantile_heatmap(cfg_sym, n_sims=min(n_sims, 30),
                          save_path=str(OUT_DIR / "quantile_heatmap_symmetric.png"))

    print("\nQuantile heatmap: asymmetric pattern, hetero=0.8")
    cfg_asym = base_config.model_copy(update={
        "transport_heterogeneity": 0.8,
        "divergence_pattern": "asymmetric",
    })
    plot_quantile_heatmap(cfg_asym, n_sims=min(n_sims, 30),
                          save_path=str(OUT_DIR / "quantile_heatmap_asymmetric.png"))

    # ── Overlap diagnostic: high vs low heterogeneity ──
    print("\nOverlap diagnostic")
    for h, patt in [(0.0, "asymmetric"), (0.8, "asymmetric"), (0.8, "symmetric")]:
        cfg_ov = base_config.model_copy(update={
            "transport_heterogeneity": h,
            "divergence_pattern": patt,
            "seed": seed + 99,
        })
        plot_overlap_diagnostic(
            cfg_ov,
            save_path=str(OUT_DIR / f"overlap_{patt}_h{int(h*10)}.png"),
        )

    print(f"\nAll outputs → {OUT_DIR}/")
    return {"sweep1": sweep1, "sweep2": sweep2}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 11: Transport Decomposition")
    p.add_argument("--n-sims", type=int, default=N_SIMS)
    p.add_argument("--seed",   type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, seed=args.seed)
