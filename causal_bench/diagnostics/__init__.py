"""Diagnostics for causal_bench simulations.

Three areas:
  1. Positivity/overlap  — propensity score distribution, extreme weight fraction
  2. Covariate balance   — standardized mean differences (SMD), Love plot
  3. SE calibration      — median(SE) vs empirical SE across estimators
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from typing import Optional


# ---------------------------------------------------------------------------
# 1. Positivity / overlap
# ---------------------------------------------------------------------------

def positivity_summary(df: pd.DataFrame, n_folds: int = 3) -> dict:
    """Fit a propensity model and return overlap diagnostics.

    Returns
    -------
    dict with keys:
      g_mean, g_min, g_max, g_std         — propensity score statistics
      pct_extreme                          — % with g < 0.05 or g > 0.95
      effective_sample_size               — ESS = (sum w)^2 / sum(w^2), IPW weights
      overlap_ratio                        — min(n_treated, n_control) / max(...), crude
    """
    from causal_bench.super_learner import SuperLearner
    W_cols = [c for c in ["W1", "W2", "W3", "W4"] if c in df.columns]
    A = df["A"].values
    g_sl = SuperLearner(task="classification", n_folds=n_folds, random_state=42)
    g_sl.fit(df[W_cols].values, A)
    g = g_sl.predict_proba(df[W_cols].values)

    extreme = ((g < 0.05) | (g > 0.95))
    w = np.where(A == 1, 1 / g, 1 / (1 - g))
    ess = float((w.sum() ** 2) / (w ** 2).sum())
    n_t, n_c = int(A.sum()), int((1 - A).sum())

    return {
        "g_mean":               float(g.mean()),
        "g_min":                float(g.min()),
        "g_max":                float(g.max()),
        "g_std":                float(g.std()),
        "pct_extreme":          float(extreme.mean() * 100),
        "effective_sample_size": ess,
        "overlap_ratio":        float(min(n_t, n_c) / max(n_t, n_c, 1)),
        "_g":                   g,   # kept for plot_overlap
        "_A":                   A,
    }


def plot_overlap(df: pd.DataFrame, n_folds: int = 3,
                 save_path: Optional[str] = None) -> plt.Figure:
    """Propensity score histogram by treatment arm + extreme region shading."""
    diag = positivity_summary(df, n_folds=n_folds)
    g, A = diag["_g"], diag["_A"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, 1, 31)
    ax.hist(g[A == 1], bins=bins, alpha=0.55, color="#E34A33", label="Treated", density=True)
    ax.hist(g[A == 0], bins=bins, alpha=0.55, color="#3182BD", label="Control", density=True)
    ax.axvspan(0, 0.05,  alpha=0.12, color="red", label="Extreme (<0.05)")
    ax.axvspan(0.95, 1,  alpha=0.12, color="red")
    ax.set_xlabel("P(A=1 | W)")
    ax.set_ylabel("Density")
    ax.set_title(
        f"Propensity overlap  |  extreme: {diag['pct_extreme']:.1f}%  "
        f"|  ESS: {diag['effective_sample_size']:.0f}"
    )
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 2. Covariate balance — SMD and Love plot
# ---------------------------------------------------------------------------

def smd_table(df: pd.DataFrame,
              cov_cols: Optional[list[str]] = None) -> pd.DataFrame:
    """Standardized mean differences before and after IPW weighting.

    SMD = (mean_treated - mean_control) / pooled_std

    Returns DataFrame with columns: covariate, smd_raw, smd_ipw
    where smd_ipw uses stabilized Horvitz-Thompson weights.
    """
    from causal_bench.super_learner import SuperLearner

    if cov_cols is None:
        cov_cols = [c for c in ["W1", "W2", "W3", "W4", "compliance",
                                 "enrollment_time"] if c in df.columns]
    A = df["A"].values
    W_cols = [c for c in ["W1", "W2", "W3", "W4"] if c in df.columns]

    # Fit propensity for IPW weights
    g_sl = SuperLearner(task="classification", n_folds=3, random_state=42)
    g_sl.fit(df[W_cols].values, A)
    g = g_sl.predict_proba(df[W_cols].values)
    p_A = A.mean()
    w1 = A * p_A / g
    w0 = (1 - A) * (1 - p_A) / (1 - g)

    rows = []
    for col in cov_cols:
        x = df[col].values.astype(float)
        x1_raw, x0_raw = x[A == 1], x[A == 0]
        pooled_std = np.sqrt((x1_raw.var() + x0_raw.var()) / 2)
        if pooled_std < 1e-10:
            smd_raw = 0.0
        else:
            smd_raw = float((x1_raw.mean() - x0_raw.mean()) / pooled_std)

        # IPW-weighted means
        mu1_ipw = np.sum(w1 * x) / np.sum(w1)
        mu0_ipw = np.sum(w0 * x) / np.sum(w0)
        smd_ipw = float((mu1_ipw - mu0_ipw) / pooled_std) if pooled_std > 1e-10 else 0.0

        rows.append({"covariate": col, "smd_raw": smd_raw, "smd_ipw": smd_ipw})

    return pd.DataFrame(rows).set_index("covariate")


def plot_love(df: pd.DataFrame, cov_cols: Optional[list[str]] = None,
              save_path: Optional[str] = None) -> plt.Figure:
    """Love plot: |SMD| before and after IPW weighting."""
    smd = smd_table(df, cov_cols=cov_cols).reset_index()
    smd = smd.sort_values("smd_raw", key=abs, ascending=True)

    fig, ax = plt.subplots(figsize=(7, max(3, len(smd) * 0.5 + 1)))
    y = np.arange(len(smd))
    ax.scatter(smd["smd_raw"].abs(), y, color="#E34A33", zorder=3, label="Unadjusted", s=50)
    ax.scatter(smd["smd_ipw"].abs(), y, color="#31A354", zorder=3, label="IPW adjusted", s=50, marker="D")
    ax.axvline(0.1, color="gray", linestyle="--", linewidth=0.8, label="|SMD|=0.1 threshold")
    ax.set_yticks(y)
    ax.set_yticklabels(smd["covariate"])
    ax.set_xlabel("|Standardized Mean Difference|")
    ax.set_title("Covariate balance (Love plot)")
    ax.legend(loc="lower right")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 3. SE calibration across estimators
# ---------------------------------------------------------------------------

def se_calibration_table(results: dict) -> pd.DataFrame:
    """SE calibration summary: median(SE) vs empirical SE, SE ratio.

    Parameters
    ----------
    results : dict[str, SimResult]
        Output from run_simulation or run_parameter_sweep slice.

    Returns
    -------
    DataFrame with columns: estimator, empirical_se, median_reported_se, se_ratio
    """
    rows = []
    for name, sr in results.items():
        if sr is None:
            continue
        emp_se    = float(np.std(sr.estimates, ddof=1))
        median_se = float(np.median(sr.se_estimates))
        rows.append({
            "estimator":          name,
            "empirical_se":       round(emp_se,    4),
            "median_reported_se": round(median_se, 4),
            "se_ratio":           round(sr.se_ratio, 3),
        })
    if not rows:
        return pd.DataFrame(columns=["empirical_se", "median_reported_se", "se_ratio"]).rename_axis("estimator")
    return pd.DataFrame(rows).set_index("estimator")


def plot_se_calibration(results: dict, save_path: Optional[str] = None) -> plt.Figure:
    """Scatter: empirical SE (x) vs median reported SE (y).

    Well-calibrated estimators fall on the y=x line.
    Over-conservative estimators are above; anti-conservative below.
    """
    tbl = se_calibration_table(results).reset_index()
    if tbl.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        return fig

    fig, ax = plt.subplots(figsize=(6, 5))
    lo = min(tbl["empirical_se"].min(), tbl["median_reported_se"].min()) * 0.9
    hi = max(tbl["empirical_se"].max(), tbl["median_reported_se"].max()) * 1.1
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.8, label="y = x (perfect calibration)")

    from causal_bench.viz import COLORS, LABELS
    for _, row in tbl.iterrows():
        color = COLORS.get(row["estimator"], "#888888")
        label = LABELS.get(row["estimator"], row["estimator"])
        ax.scatter(row["empirical_se"], row["median_reported_se"],
                   color=color, s=80, zorder=3, label=label)
        ax.annotate(label, (row["empirical_se"], row["median_reported_se"]),
                    fontsize=7, xytext=(4, 2), textcoords="offset points")

    ax.set_xlabel("Empirical SE (Monte Carlo std)")
    ax.set_ylabel("Median reported SE")
    ax.set_title("SE calibration")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 4. Tipping-point sensitivity
# ---------------------------------------------------------------------------

def tipping_point_table(results: dict, alpha: float = 0.05) -> pd.DataFrame:
    """Additive tipping-point analysis for each estimator.

    The tipping point is the minimum additive bias needed to move the
    mean estimate to the null (zero):
        tipping_bias = |mean(estimates)|

    Also expressed in SE units (how many median SEs away from null):
        tipping_z = tipping_bias / median(se_estimates)

    Returns DataFrame with columns:
        mean_estimate, tipping_bias, tipping_se_units, median_se, n_sim
    """
    rows = []
    for name, sr in results.items():
        if sr is None:
            continue
        mean_est  = float(np.mean(sr.estimates))
        med_se    = float(np.median(sr.se_estimates))
        tip_bias  = abs(mean_est)
        tip_z     = tip_bias / med_se if med_se > 1e-10 else np.nan
        rows.append({
            "estimator":        name,
            "mean_estimate":    mean_est,
            "tipping_bias":     tip_bias,
            "tipping_se_units": round(tip_z,    2),
            "median_se":        med_se,
            "n_sim":            int(sr.n_sim),
        })
    if not rows:
        return pd.DataFrame(
            columns=["mean_estimate", "tipping_bias", "tipping_se_units", "median_se", "n_sim"]
        ).rename_axis("estimator")
    return pd.DataFrame(rows).set_index("estimator")


def plot_tipping_point(results: dict, save_path: Optional[str] = None) -> plt.Figure:
    """Horizontal bar chart of tipping-point bias by estimator.

    Each bar = |mean estimate| = how much additive bias explains away the result.
    Bars are coloured by estimator (from viz.COLORS).
    Secondary x-axis label shows SE units for reference.
    """
    from causal_bench.viz import COLORS, LABELS
    tbl = tipping_point_table(results).reset_index()
    if tbl.empty:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        return fig

    tbl = tbl.sort_values("tipping_bias", ascending=True)
    fig, ax = plt.subplots(figsize=(7, max(3, len(tbl) * 0.5 + 1)))
    y = np.arange(len(tbl))
    colors = [COLORS.get(e, "#888888") for e in tbl["estimator"]]
    bars = ax.barh(y, tbl["tipping_bias"], color=colors, edgecolor="white", height=0.6)

    # Annotate with SE units
    for bar, (_, row) in zip(bars, tbl.iterrows()):
        ax.text(
            bar.get_width() + tbl["tipping_bias"].max() * 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{row['tipping_se_units']:.1f} SE",
            va="center", fontsize=8, color="#444"
        )

    ax.set_yticks(y)
    ax.set_yticklabels([LABELS.get(e, e) for e in tbl["estimator"]])
    ax.set_xlabel("Tipping-point bias (additive, |mean estimate|)")
    ax.set_title("Tipping-point sensitivity\n(bars = bias needed to explain away the effect)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 5. Effective sample size (ESS) across simulation draws
# ---------------------------------------------------------------------------

def ess_across_sims(
    dgp_config,
    n_draws: int = 50,
    n_folds: int = 3,
    seed: int = 0,
) -> dict:
    """Compute IPW ESS distribution across n_draws simulation draws.

    For each draw a fresh dataset is generated (seed + i), propensity fitted,
    and ESS = (sum w)^2 / sum(w^2) computed for stabilised HT weights.

    Returns dict with keys:
        ess_values   — list of n_draws ESS values
        mean_ess     — mean ESS
        median_ess   — median ESS
        min_ess      — min ESS
        max_ess      — max ESS
        ess_pct      — median ESS as % of n (nominal sample size)
    """
    from causal_bench.dgp.survival import generate_data
    from causal_bench.dgp.config import DGPConfig
    from causal_bench.super_learner import SuperLearner

    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 10_000, size=n_draws).tolist()

    # Build a new config with each seed, keeping all other params identical
    base_kwargs = {k: v for k, v in vars(dgp_config).items() if not k.startswith("_")}

    ess_vals = []
    W_cols = ["W1", "W2", "W3", "W4"]
    for s in seeds:
        cfg_i = DGPConfig(**{**base_kwargs, "seed": int(s)})
        df_i  = generate_data(cfg_i)
        A     = df_i["A"].values
        g_sl  = SuperLearner(task="classification", n_folds=n_folds, random_state=42)
        g_sl.fit(df_i[W_cols].values, A)
        g     = g_sl.predict_proba(df_i[W_cols].values)
        p_A   = A.mean()
        w     = np.where(A == 1, p_A / g, (1 - p_A) / (1 - g))
        ess   = float((w.sum() ** 2) / (w ** 2).sum())
        ess_vals.append(ess)

    n = dgp_config.n
    return {
        "ess_values":  ess_vals,
        "mean_ess":    float(np.mean(ess_vals)),
        "median_ess":  float(np.median(ess_vals)),
        "min_ess":     float(np.min(ess_vals)),
        "max_ess":     float(np.max(ess_vals)),
        "ess_pct":     float(np.median(ess_vals) / n * 100),
    }


def plot_ess_distribution(
    dgp_config,
    n_draws: int = 50,
    n_folds: int = 3,
    seed: int = 0,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Histogram of ESS across simulation draws with summary statistics."""
    summary = ess_across_sims(dgp_config, n_draws=n_draws, n_folds=n_folds, seed=seed)
    vals = summary["ess_values"]
    n    = dgp_config.n

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(vals, bins=20, color="#3182BD", alpha=0.7, edgecolor="white")
    ax.axvline(summary["median_ess"], color="#E34A33", linewidth=1.5,
               label=f"Median ESS = {summary['median_ess']:.0f} ({summary['ess_pct']:.1f}% of n={n})")
    ax.axvline(n, color="#31A354", linewidth=1, linestyle="--",
               label=f"Nominal n = {n}")
    ax.set_xlabel("Effective sample size (IPW stabilised)")
    ax.set_ylabel("Count")
    ax.set_title(f"ESS distribution across {n_draws} simulation draws")
    ax.legend()
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
