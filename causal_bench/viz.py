from __future__ import annotations
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers/CI
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from causal_bench.metrics import SimResult

COLORS = {
    "naive":            "#999999",
    "km":               "#7FCDBB",
    "cox":              "#FC8D59",
    "tmle_ipcw":        "#31A354",
    "tmle_ipcw_comply": "#006D2C",
}

LABELS = {
    "naive":            "Naive",
    "km":               "KM",
    "cox":              "Cox PH",
    "tmle_ipcw":        "TMLE+IPCW",
    "tmle_ipcw_comply": "TMLE+IPCW+Comply",
}

_FONT = dict(fontfamily="sans-serif", fontsize=11)


def _apply_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=9)


def plot_forest(
    results: dict[str, SimResult],
    title: str = "Estimator Comparison",
    save_path: str | None = None,
) -> plt.Figure:
    names = list(results.keys())
    fig, ax = plt.subplots(figsize=(9, max(3, len(names) * 0.8 + 1)))
    reversed_names = list(reversed(names))

    for i, name in enumerate(reversed_names):
        sr = results[name]
        mean_est = float(np.mean(sr.estimates))
        mean_lo  = float(np.mean(sr.ci_lowers))
        mean_hi  = float(np.mean(sr.ci_uppers))
        color = COLORS.get(name, "#555555")
        label = LABELS.get(name, name)
        ax.plot([mean_lo, mean_hi], [i, i], color=color, lw=2.5, solid_capstyle="round")
        ax.plot(mean_est, i, "o", color=color, ms=8, zorder=5)
        ax.text(mean_hi + 0.003, i, f"{mean_est:+.3f}", va="center",
                fontsize=8.5, color=color, fontweight="bold")

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([LABELS.get(n, n) for n in reversed_names], fontsize=10)

    true_val = next(iter(results.values())).true_value
    ax.axvline(true_val, ls="--", color="black", lw=1.5,
               label=f"True = {true_val:+.3f}")
    ax.axvline(0, ls=":", color="#bbbbbb", lw=1.0)
    _apply_style(ax)
    ax.set_xlabel("Risk difference at horizon", **_FONT)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.legend(fontsize=9, loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_panel(
    sweep_results: dict[str, list[SimResult]],
    param_values: list,
    param_name: str,
    title: str = "",
    save_path: str | None = None,
) -> plt.Figure:
    metrics_list = ["bias", "coverage", "rmse", "ci_width", "nc_bias"]
    ylabels = ["Bias", "Coverage (95%)", "RMSE", "CI Width", "NC Bias"]
    targets = [0.0, 0.95, None, None, 0.0]

    fig, axes = plt.subplots(5, 1, figsize=(8, 16), sharex=True)
    fig.subplots_adjust(hspace=0.3)

    for ax, metric, ylabel, target in zip(axes, metrics_list, ylabels, targets):
        for name, sr_list in sweep_results.items():
            vals = [getattr(sr, metric) for sr in sr_list if sr is not None]
            if not vals:
                continue
            color = COLORS.get(name, "#555555")
            label = LABELS.get(name, name)
            ax.plot(param_values[:len(vals)], vals, "o-", color=color,
                    label=label, lw=2.0, ms=6)
        if target is not None:
            ax.axhline(target, ls="--", color="#333333", lw=1.0, alpha=0.7)
        ax.set_ylabel(ylabel, **_FONT)
        _apply_style(ax)

    axes[-1].set_xlabel(param_name, **_FONT)
    fig.suptitle(title or f"Parameter sweep: {param_name}", fontsize=13,
                 fontweight="bold", y=1.01)

    handles = [mpatches.Patch(color=COLORS.get(n, "#555"), label=LABELS.get(n, n))
               for n in sweep_results]
    axes[0].legend(handles=handles, fontsize=8, loc="upper left")
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def generate_summary_table(results: dict[str, SimResult], fmt: str = "markdown") -> str:
    rows = [sr.summary() for sr in results.values()]
    cols = ["estimator", "estimand", "true", "bias", "rmse",
            "coverage", "ci_width", "se_ratio", "nc_bias"]
    if fmt == "markdown":
        header = "| " + " | ".join(cols) + " |"
        sep    = "| " + " | ".join(["---"] * len(cols)) + " |"
        lines  = [header, sep]
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
        return "\n".join(lines)
    raise NotImplementedError("Only markdown format supported in MVP")
