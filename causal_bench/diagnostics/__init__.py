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


# ---------------------------------------------------------------------------
# 6. Missing-data tipping-point (MNAR sensitivity)
# ---------------------------------------------------------------------------

def _impute_censored(
    df: pd.DataFrame,
    p_treated: float,
    p_control: float,
    horizon: float,
    t_impute: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Return a copy of df with imputed outcomes for informatively censored rows.

    Informatively censored = Delta==0 AND T_obs < horizon (dropped out before end).
    Administrative censoring (T_obs == horizon, Delta==0) is left untouched.

    For each arm, exactly round(p * n_censored_arm) randomly chosen censored
    patients are assigned Delta=1, T_obs=t_impute.
    """
    df2 = df.copy()
    censored_mask = (df2["Delta"] == 0) & (df2["T_obs"] < horizon - 1e-9)

    for arm, p in [(1, p_treated), (0, p_control)]:
        idx = df2.index[censored_mask & (df2["A"] == arm)].tolist()
        if not idx:
            continue
        k = int(round(p * len(idx)))
        if k == 0:
            continue
        chosen = rng.choice(idx, size=k, replace=False)
        df2.loc[chosen, "Delta"] = 1.0
        df2.loc[chosen, "T_obs"] = t_impute
        if "event_type" in df2.columns:
            df2.loc[chosen, "event_type"] = 1

    return df2


def tipping_point_mnar(
    df: pd.DataFrame,
    estimator,
    horizon: float,
    alpha: float = 0.05,
    n_grid: int = 10,
    t_impute: Optional[float] = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Missing-data tipping-point analysis (MNAR sensitivity).

    Sweeps a (n_grid x n_grid) grid of assumed event probabilities for
    informatively censored patients in each arm, re-runs the estimator
    under each imputation, and returns a DataFrame of results.

    Parameters
    ----------
    df : pd.DataFrame
        Single simulated dataset (T_obs, Delta, A, W1-W4, ...).
    estimator : estimator instance or str
        Any causal_bench estimator. Recommend a fast one (e.g. "km") for
        large grids; complex estimators (ltmle, tmle_ipcw) will be slow.
    horizon : float
        Study horizon (same value used in DGP).
    alpha : float
        Significance level for the tipping-point boundary.
    n_grid : int
        Number of grid points per axis (total runs = n_grid**2).
    t_impute : float, optional
        T_obs assigned to imputed events. Default: horizon (worst case).
    seed : int
        RNG seed for imputation assignments.

    Returns
    -------
    pd.DataFrame with columns:
        p_treated, p_control, estimate, se, ci_lower, ci_upper,
        significant (bool: CI excludes 0 at alpha), n_censored_treated,
        n_censored_control
    Also carries metadata attributes:
        .mar_p_treated, .mar_p_control  -- event rates under MAR reference
    """
    from causal_bench.estimators import ESTIMATOR_REGISTRY

    if isinstance(estimator, str):
        estimator = ESTIMATOR_REGISTRY[estimator]
    if t_impute is None:
        t_impute = horizon

    censored_mask = (df["Delta"] == 0) & (df["T_obs"] < horizon - 1e-9)
    n_ct = int(((df["A"] == 1) & censored_mask).sum())
    n_cc = int(((df["A"] == 0) & censored_mask).sum())

    # MAR reference: observed event rate among non-censored patients in each arm
    observed_mask = df["Delta"] == 1
    mar_pt = float(df.loc[(df["A"] == 1) & observed_mask, "Delta"].mean()) if (df["A"] == 1).any() else 0.5
    mar_pc = float(df.loc[(df["A"] == 0) & observed_mask, "Delta"].mean()) if (df["A"] == 0).any() else 0.5
    # Clamp to [0,1]
    mar_pt = min(max(mar_pt, 0.0), 1.0)
    mar_pc = min(max(mar_pc, 0.0), 1.0)

    grid = np.linspace(0, 1, n_grid)
    rng  = np.random.default_rng(seed)
    rows = []

    for p_t in grid:
        for p_c in grid:
            df_imp = _impute_censored(df, p_t, p_c, horizon, t_impute, rng)
            try:
                results = estimator.estimate(df_imp)
            except Exception:
                results = []

            if results:
                r = results[0]
                est      = float(r.point_estimate)
                se       = float(r.standard_error)
                ci_lo    = float(r.ci_lower)
                ci_hi    = float(r.ci_upper)
                sig      = not (ci_lo <= 0 <= ci_hi)
            else:
                est = se = ci_lo = ci_hi = float("nan")
                sig = False

            rows.append({
                "p_treated":           round(float(p_t), 6),
                "p_control":           round(float(p_c), 6),
                "estimate":            est,
                "se":                  se,
                "ci_lower":            ci_lo,
                "ci_upper":            ci_hi,
                "significant":         sig,
                "n_censored_treated":  n_ct,
                "n_censored_control":  n_cc,
            })

    result_df = pd.DataFrame(rows)
    result_df.attrs["mar_p_treated"] = mar_pt
    result_df.attrs["mar_p_control"] = mar_pc
    return result_df


def tipping_point_concrete(
    df: pd.DataFrame,
    horizon: float,
    deltas: Optional[list[float]] = None,
    mechanism: str = "all",
    crossover_col: Optional[str] = None,
) -> pd.DataFrame:
    """MAR sensitivity analysis via concrete::senseCensoring() with CensoringTV.

    Uses concrete's doubly-robust TMLE as the estimator (rather than a
    simpler Python estimator). L1 (when present) is forwarded to the
    censoring model via CensoringTV, so the baseline IPCW at delta=0 is
    already conditioned on the time-varying confounder.

    Complements tipping_point_mnar() — which sweeps a 2-D grid over
    independent per-arm imputation probabilities — with a principled 1-D
    delta-shift: what fraction of censored patients would need to be
    counterfactual events to overturn the conclusion?

    Parameters
    ----------
    df            : causal_bench DataFrame (with L1 for CensoringTV activation).
    horizon       : study horizon passed to concrete.
    deltas        : fractions swept (default [0.0, 0.05, 0.10, 0.15, 0.20]).
    mechanism     : "all" (default) | "dropout" | "crossover" — which censoring
                    pool to tip (concrete PR #28). "crossover" requires
                    crossover_col and a Crossover-aware fit.
    crossover_col : column name of per-subject switch times (None = ITT).

    Returns
    -------
    pd.DataFrame with columns:
        mechanism, delta, estimate, se, ci_lower, ci_upper, significant
    Attributes:
        .attrs["tipping_delta"]  — first delta where CI crosses 0, using
                                   concrete's own tipping-point value when
                                   available; falls back to Python scan
        .attrs["mar_estimate"]   — point estimate at delta = 0

    Raises
    ------
    RuntimeError if concrete R package is not available.
    """
    from causal_bench.estimators.concrete_rmst import concrete_sensitivity

    if deltas is None:
        deltas = [0.0, 0.05, 0.10, 0.15, 0.20]

    result = concrete_sensitivity(df, horizon=horizon, deltas=deltas,
                                  mechanism=mechanism, crossover_col=crossover_col)

    # Ensure CI columns exist
    if "ci_lower" not in result.columns and "se" in result.columns:
        result["ci_lower"] = result["estimate"] - 1.96 * result["se"]
        result["ci_upper"] = result["estimate"] + 1.96 * result["se"]

    result["significant"] = ~(
        (result["ci_lower"] <= 0) & (0 <= result["ci_upper"])
    )

    # Prefer concrete's own tipping point; fall back to Python scan
    concrete_tp = result.attrs.get("tipping_point")
    if concrete_tp is not None and np.isfinite(float(concrete_tp)):
        tipping = float(concrete_tp)
    else:
        crossing = result.loc[~result["significant"], "delta"]
        tipping  = float(crossing.iloc[0]) if len(crossing) else float("nan")

    mar_row = result.loc[result["delta"] == min(deltas, key=lambda d: abs(d))]
    mar_est = float(mar_row["estimate"].iloc[0]) if len(mar_row) else float("nan")

    result.attrs["tipping_delta"] = tipping
    result.attrs["mar_estimate"]  = mar_est
    result.attrs["mechanism"]     = mechanism
    return result


def positivity_dx_concrete(
    df: pd.DataFrame,
    horizon: float,
    crossover_col: Optional[str] = None,
) -> dict:
    """Positivity / inverse-weight diagnostics via concrete::getPositivityDx().

    Reports per-arm effective sample size (ESS as fraction of n), max IPCW
    weight, minimum observation probability, and the share of weights at the
    truncation bound.  Uses the full IPCW weight 1/(g·S_C[·S_X]), so this is
    heavier than positivity_summary() which covers only g(A|W).

    A CAUTION flag is set on arms where ESS < 50 %, pct_at_bound > 5 %, or
    max_weight > 20 — these indicate near-positivity violations that inflate
    variance and can bias the TMLE targeting step.

    Parameters
    ----------
    df            : causal_bench DataFrame.
    horizon       : study horizon.
    crossover_col : per-subject switch-time column (None = ITT estimand).

    Returns
    -------
    dict with keys:
        "summary"  — pd.DataFrame, one row per arm:
                     intervention, n, ESS_overall, ESS_worst, max_weight,
                     min_obs_prob, pct_at_bound
        "by_time"  — pd.DataFrame with per-evaluation-time ESS, max_weight,
                     min_obs_prob columns (one row per time × arm), or None
        "caution"  — list[str] arm names that triggered the CAUTION threshold

    Raises
    ------
    RuntimeError if concrete R package is not available.
    """
    from causal_bench.estimators.concrete_rmst import concrete_positivity_dx

    result = concrete_positivity_dx(df, horizon=horizon, crossover_col=crossover_col)
    summary = result["summary"]

    caution_arms = []
    if summary is not None and len(summary):
        mask = (
            (summary.get("ESS_worst",     pd.Series(dtype=float)) < 0.5) |
            (summary.get("pct_at_bound",  pd.Series(dtype=float)) > 5.0) |
            (summary.get("max_weight",    pd.Series(dtype=float)) > 20.0)
        )
        caution_arms = list(summary.loc[mask, "Intervention"]) if "Intervention" in summary.columns else []

    result["caution"] = caution_arms
    return result


def plot_tipping_point_mnar(
    tipping_df: pd.DataFrame,
    alpha: float = 0.05,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Heatmap of the MNAR tipping-point grid.

    Background colour = estimate value (diverging, centred at 0).
    White contour = tipping-point boundary (CI crosses 0).
    Star = MAR reference point (stored in tipping_df.attrs).
    """
    import matplotlib.colors as mcolors

    pivot_est = tipping_df.pivot(index="p_treated", columns="p_control", values="estimate")
    pivot_sig = tipping_df.pivot(index="p_treated", columns="p_control", values="significant")

    p_t_vals = pivot_est.index.values
    p_c_vals = pivot_est.columns.values

    vmax = np.nanmax(np.abs(pivot_est.values))
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.pcolormesh(
        p_c_vals, p_t_vals, pivot_est.values,
        cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto"
    )
    plt.colorbar(im, ax=ax, label="Estimate")

    # Tipping-point contour (boundary of significance)
    sig_arr = pivot_sig.values.astype(float)
    if sig_arr.min() < sig_arr.max():  # contour only if both sides exist
        ax.contour(p_c_vals, p_t_vals, sig_arr, levels=[0.5],
                   colors="white", linewidths=2, linestyles="--")

    # MAR reference point
    mar_pt = tipping_df.attrs.get("mar_p_treated", None)
    mar_pc = tipping_df.attrs.get("mar_p_control", None)
    if mar_pt is not None and mar_pc is not None:
        ax.plot(mar_pc, mar_pt, "w*", markersize=14,
                label=f"MAR reference ({mar_pc:.2f}, {mar_pt:.2f})", zorder=5)
        ax.legend(loc="upper right", fontsize=8)

    ax.set_xlabel("Assumed event prob -- censored control (p_control)")
    ax.set_ylabel("Assumed event prob -- censored treated (p_treated)")
    ax.set_title(title or "MNAR tipping-point sensitivity\n(dashed = CI crosses zero)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def convergence_table(df: pd.DataFrame, estimator_names: list) -> pd.DataFrame:
    """Run TMLE estimators on a single dataset and report IC-based convergence stats.

    Convergence condition: |IC.mean()| / IC.std() << 1 (targeting residual small
    relative to variance). McCoy's TMLE convergence criterion.

    Parameters
    ----------
    df               : one simulated dataset (output of generate_data)
    estimator_names  : names of TMLE-family estimators to diagnose

    Returns
    -------
    DataFrame with columns: estimator, ic_mean (eps), ic_sd, ic_ratio, n, converged
    """
    from causal_bench.estimators import get_estimator
    rows = []
    for name in estimator_names:
        est = get_estimator(name)
        try:
            results = est.estimate(df, horizon=float(df["T_obs"].max()), estimand="ATE")
        except Exception as exc:
            rows.append({"estimator": name, "ic_mean": float("nan"), "ic_sd": float("nan"),
                         "ic_ratio": float("nan"), "n": len(df), "converged": False,
                         "note": str(exc)[:60]})
            continue
        match = next((r for r in results if r.estimand == "ATE"), None)
        if match is None or match.ic is None:
            rows.append({"estimator": name, "ic_mean": float("nan"), "ic_sd": float("nan"),
                         "ic_ratio": float("nan"), "n": len(df), "converged": False,
                         "note": "no IC"})
            continue
        ic = match.ic
        ic_mean = float(np.mean(ic))
        ic_sd   = float(np.std(ic))
        ratio   = abs(ic_mean) / max(ic_sd, 1e-12)
        rows.append({
            "estimator": name,
            "ic_mean":   round(ic_mean, 6),
            "ic_sd":     round(ic_sd, 6),
            "ic_ratio":  round(ratio, 4),
            "n":         len(df),
            "converged": ratio < 0.1,
            "note":      "",
        })
    return pd.DataFrame(rows).set_index("estimator")


def plot_overlap_map(
    df: pd.DataFrame,
    x_col: str = "W1",
    y_col: str = "W3",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """'Who are we borrowing for?' overlap map.

    2D scatter of treated patients in (x_col, y_col) space.
    Marker size ∝ 1/g (inverse propensity weight — larger = harder to match).
    Background: KDE density contour of control patients.

    Parameters
    ----------
    df       : one simulated dataset with columns A, W1-W4
    x_col    : x-axis covariate (default W1)
    y_col    : y-axis covariate (default W3)
    save_path: if given, saves figure at 150 dpi
    """
    from sklearn.linear_model import LogisticRegression
    from scipy.stats import gaussian_kde

    feat_cols = [c for c in ["W1", "W2", "W3", "W4"] if c in df.columns]
    if not feat_cols or x_col not in df.columns or y_col not in df.columns:
        raise ValueError(f"Required columns missing from df: {feat_cols}, {x_col}, {y_col}")

    X = df[feat_cols].values
    A = df["A"].values
    lr = LogisticRegression(max_iter=500, C=1.0)
    lr.fit(X, A)
    g = np.clip(lr.predict_proba(X)[:, 1], 0.05, 0.95)

    treated  = df[A == 1]
    controls = df[A == 0]

    g_treated = g[A == 1]
    ipw_size  = np.clip(1.0 / g_treated, 1, 20)   # cap at 20 for display
    ipw_size  = 20 * ipw_size / ipw_size.max()     # normalize to [0, 20] point area

    fig, ax = plt.subplots(figsize=(7, 6))

    # Background: control patient density
    if len(controls) >= 4:
        ctrl_x = controls[x_col].values
        ctrl_y = controls[y_col].values
        try:
            kde = gaussian_kde(np.vstack([ctrl_x, ctrl_y]))
            xi = np.linspace(df[x_col].min(), df[x_col].max(), 80)
            yi = np.linspace(df[y_col].min(), df[y_col].max(), 80)
            Xi, Yi = np.meshgrid(xi, yi)
            Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
            ax.contourf(Xi, Yi, Zi, levels=8, cmap="Blues", alpha=0.35)
        except Exception:
            ax.scatter(ctrl_x, ctrl_y, c="#AECDE8", s=6, alpha=0.3, label="Control")

    # Treated patients — size ∝ 1/g
    sc = ax.scatter(
        treated[x_col].values, treated[y_col].values,
        s=ipw_size * 15, c=g_treated,
        cmap="YlOrRd_r", alpha=0.75, edgecolors="k", linewidths=0.3,
        label="Treated (size ∝ 1/g)",
        vmin=0.1, vmax=0.9,
    )
    cbar = fig.colorbar(sc, ax=ax, shrink=0.8)
    cbar.set_label("Propensity g(W)", fontsize=9)

    ax.set_xlabel(x_col, fontsize=10)
    ax.set_ylabel(y_col, fontsize=10)
    ax.set_title(
        "Overlap map: who are we borrowing for?\n"
        "Large markers = high IPW (extrapolation risk) | Blue = control density",
        fontsize=10,
    )
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def export_for_r(
    df: pd.DataFrame,
    config,
    out_dir: str,
    prefix: str = "sim",
) -> dict:
    """Export a simulated dataset as CSV + metadata JSON for R/concrete benchmarking.

    Writes:
      <out_dir>/<prefix>_data.csv   — full dataset with concrete-compatible column names
      <out_dir>/<prefix>_meta.json  — column name map + horizon + scenario parameters

    Parameters
    ----------
    df      : simulated dataset (output of generate_data)
    config  : DGPConfig used to generate df
    out_dir : directory to write into (created if absent)
    prefix  : filename prefix (default "sim")

    Returns
    -------
    dict with keys "csv_path", "meta_path"
    """
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    csv_path  = out / f"{prefix}_data.csv"
    meta_path = out / f"{prefix}_meta.json"

    df.to_csv(csv_path, index=False)

    meta = {
        "event_time_col":   "T_obs",
        "event_type_col":   "event_type" if "event_type" in df.columns else "Delta",
        "treatment_col":    "A",
        "covariate_cols":   [c for c in ["W1", "W2", "W3", "W4"] if c in df.columns],
        "L1_col":           "L1_obs" if "L1_obs" in df.columns else None,
        "compliance_col":   "compliance" if "compliance" in df.columns else None,
        "horizon":          float(config.horizon),
        "n":                len(df),
        "dgp_config":       config.model_dump(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return {"csv_path": str(csv_path), "meta_path": str(meta_path)}


def plot_tipping_point_concrete(
    tipping_df: pd.DataFrame,
    title: Optional[str] = None,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Line + CI ribbon: concrete::senseCensoring() delta-shift sensitivity.

    x-axis: delta (fraction of censored patients counterfactually treated
            as events)
    y-axis: doubly-robust TMLE risk-difference estimate + 95% CI ribbon
    Vertical dashed line: tipping delta (first delta where CI crosses 0)
    Horizontal reference: y = 0

    Parameters
    ----------
    tipping_df : output of tipping_point_concrete()
    title      : optional plot title override
    save_path  : if given, saves figure to this path at 150 dpi
    """
    df = tipping_df.copy().sort_values("delta")

    fig, ax = plt.subplots(figsize=(7, 4))

    # CI ribbon
    ax.fill_between(
        df["delta"], df["ci_lower"], df["ci_upper"],
        alpha=0.20, color="#3182BD", label="95% CI"
    )
    # Point estimate line
    ax.plot(df["delta"], df["estimate"], color="#3182BD", linewidth=2,
            label="concrete TMLE (RD)")
    # y = 0 reference
    ax.axhline(0, color="#444", linewidth=0.8, linestyle="--")

    # Tipping delta
    tipping = tipping_df.attrs.get("tipping_delta", float("nan"))
    if np.isfinite(tipping):
        ax.axvline(tipping, color="#E34A33", linewidth=1.5, linestyle=":",
                   label=f"Tipping δ = {tipping:.2f}")
    else:
        ax.text(
            0.97, 0.05,
            "Effect holds across all δ",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=8, color="#31A354",
        )

    # MAR estimate annotation
    mar_est = tipping_df.attrs.get("mar_estimate", float("nan"))
    if np.isfinite(mar_est):
        ax.annotate(
            f"MAR baseline\n(δ=0): {mar_est:.3f}",
            xy=(0, mar_est),
            xytext=(df["delta"].max() * 0.35, mar_est),
            fontsize=8, color="#555",
            arrowprops=dict(arrowstyle="->", color="#888", lw=0.8),
        )

    mechanism = tipping_df.attrs.get("mechanism", "all")
    mech_label = {"all": "all censored", "dropout": "dropout only",
                  "crossover": "crossover only"}.get(mechanism, mechanism)
    ax.set_xlabel("δ (fraction of censored patients assumed to be events)")
    ax.set_ylabel("Risk difference (TMLE)")
    ax.set_title(
        title or f"Censoring MAR sensitivity  —  concrete::senseCensoring() [{mech_label}]\n"
                 "L1 in CensoringTV: baseline IPCW already conditioned on time-varying covariates"
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
