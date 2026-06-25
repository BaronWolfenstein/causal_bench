"""Exp 21: HTE subgroup benchmarking — EffectXShift CV-TMLE vs BCF/BART posterior tree.

Operationalises the McCoy–Hahn head-to-head (LinkedIn, 2026-06-25) in a
known-ground-truth setting.  The DGP places a binary subgroup boundary at
W1 > median:

    true CATE (high-benefit stratum)  = −0.25
    true CATE (low-benefit stratum)   = −0.05
    true V − V^c contrast             = −0.20

Four estimators are compared across n ∈ {200, 400, 700, 1500}:
  1. EffectXShift (CV-TMLE selected rule) — post-selection valid CI
  2. BCF/BART posterior tree (Hahn et al. 2020) — Bayesian posterior CI
  3. X-learner with naive threshold — no post-selection correction (baseline)
  4. Oracle (true subgroup membership known) — upper bound

Four metrics (McCoy's proposed head-to-head):
  subgroup_recovery   — does the method identify W1 as the split variable?
  contrast_bias       — estimated V − V^c minus true contrast
  coverage            — does the 95 % CI contain the true contrast?
  null_fpr            — P(conclude V ≠ V^c) under H₀ (cate_high = cate_low)

Outputs (results/exp21_hte_subgroup/):
  results.parquet          — replicate-level raw records
  metrics_table.csv        — aggregated 4 metrics × estimator × n
  power_curve.png          — subgroup recovery rate vs n by method
  coverage_plot.png        — nominal (0.95) vs actual coverage by method × n
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data
from causal_bench.estimators.bcf_bart import BCFBARTEstimator
from causal_bench.estimators.effectxshift import EffectXShiftEstimator

OUT_DIR = Path("results/exp21_hte_subgroup")
N_REPS  = 200

SAMPLE_SIZES  = [200, 400, 700, 1500]
CATE_HIGH     = -0.25
CATE_LOW      = -0.05
TRUE_CONTRAST = CATE_HIGH - CATE_LOW   # −0.20
ALPHA         = 0.05

BASE_CFG = DGPConfig(
    subgroup_col="W1",
    cate_high=CATE_HIGH,
    cate_low=CATE_LOW,
)

NULL_CFG = DGPConfig(
    subgroup_col="W1",
    cate_high=-0.15,
    cate_low=-0.15,   # no contrast; FPR test
)

_EFX  = EffectXShiftEstimator(outcome_col="Y")
_BCF  = BCFBARTEstimator(outcome_col="Y")


# ---------------------------------------------------------------------------
# X-learner naive
# ---------------------------------------------------------------------------

def _xlearner_cate(df: pd.DataFrame, covariate_cols: list[str]) -> np.ndarray:
    W = df[covariate_cols]
    Y = df["Y"].values
    A = df["A"].values.astype(int)

    treated_mask = A == 1
    control_mask = A == 0

    mu0 = RandomForestRegressor(n_estimators=100, random_state=0)
    mu0.fit(W[control_mask], Y[control_mask])

    mu1 = RandomForestRegressor(n_estimators=100, random_state=0)
    mu1.fit(W[treated_mask], Y[treated_mask])

    d1 = Y[treated_mask] - mu0.predict(W[treated_mask])
    d0 = mu1.predict(W[control_mask]) - Y[control_mask]

    tau1 = RandomForestRegressor(n_estimators=100, random_state=0)
    tau1.fit(W[treated_mask], d1)

    tau0 = RandomForestRegressor(n_estimators=100, random_state=0)
    tau0.fit(W[control_mask], d0)

    return 0.5 * (tau1.predict(W) + tau0.predict(W))


def _xlearner_naive(df: pd.DataFrame) -> dict:
    cov = ["W1", "W2", "W3", "W4"]
    cate_hat = _xlearner_cate(df, cov)
    high_mask = cate_hat > np.median(cate_hat)

    true_high = df["subgroup_label"].values == 1
    # Jaccard overlap between predicted and true high-benefit group
    intersection = (high_mask & true_high).sum()
    union = (high_mask | true_high).sum()
    recovery = intersection / union if union > 0 else 0.0

    # Naive contrast: simple mean-difference within each predicted stratum
    Y = df["Y"].values
    A = df["A"].values

    def _naive_ate(mask: np.ndarray) -> tuple[float, float]:
        sub = df[mask]
        t1 = sub.loc[sub["A"] == 1, "Y"]
        t0 = sub.loc[sub["A"] == 0, "Y"]
        if len(t1) < 2 or len(t0) < 2:
            return np.nan, np.nan
        diff = t1.mean() - t0.mean()
        se   = np.sqrt(t1.var(ddof=1) / len(t1) + t0.var(ddof=1) / len(t0))
        return diff, se

    ate_high, se_high = _naive_ate(high_mask)
    ate_low,  se_low  = _naive_ate(~high_mask)

    if any(np.isnan(x) for x in [ate_high, se_high, ate_low, se_low]):
        return {"recovery": recovery, "contrast": np.nan, "ci_lower": np.nan,
                "ci_upper": np.nan, "covered": False}

    contrast = ate_high - ate_low
    se_cont  = np.sqrt(se_high**2 + se_low**2)
    ci_lo    = contrast - 1.96 * se_cont
    ci_hi    = contrast + 1.96 * se_cont

    return {
        "recovery": recovery,
        "contrast": contrast,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "covered": ci_lo <= TRUE_CONTRAST <= ci_hi,
    }


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------

def _oracle(df: pd.DataFrame) -> dict:
    high_mask = df["subgroup_label"].values == 1
    low_mask  = ~high_mask

    def _ate_ci(mask: np.ndarray) -> tuple[float, float, float, float]:
        sub = df[mask]
        t1 = sub.loc[sub["A"] == 1, "Y"]
        t0 = sub.loc[sub["A"] == 0, "Y"]
        diff = t1.mean() - t0.mean()
        se   = np.sqrt(t1.var(ddof=1) / len(t1) + t0.var(ddof=1) / len(t0))
        return diff, se, diff - 1.96 * se, diff + 1.96 * se

    ate_hi, se_hi, lo_hi, hi_hi = _ate_ci(high_mask)
    ate_lo, se_lo, lo_lo, hi_lo = _ate_ci(low_mask)
    contrast = ate_hi - ate_lo
    se_cont  = np.sqrt(se_hi**2 + se_lo**2)
    ci_lo    = contrast - 1.96 * se_cont
    ci_hi    = contrast + 1.96 * se_cont

    return {
        "recovery": 1.0,   # oracle always knows the true subgroup
        "contrast": contrast,
        "ci_lower": ci_lo,
        "ci_upper": ci_hi,
        "covered": ci_lo <= TRUE_CONTRAST <= ci_hi,
    }


# ---------------------------------------------------------------------------
# EffectXShift / BCF–BART wrappers
# ---------------------------------------------------------------------------

def _efx_result(df: pd.DataFrame) -> dict:
    results = _EFX.estimate(df)
    cont = next((r for r in results if r.name == "effectxshift_contrast"), None)
    if cont is None:
        return {"recovery": np.nan, "contrast": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "covered": False}
    rule = (cont.convergence_info or {}).get("rule", "")
    # Recovery: W1 appears in the rule and no other covariate appears alone
    w1_in   = "W1" in rule
    others  = any(f"W{i}" in rule for i in [2, 3, 4])
    recovery = 1.0 if (w1_in and not others) else (0.5 if w1_in else 0.0)
    covered  = cont.ci_lower <= TRUE_CONTRAST <= cont.ci_upper
    return {
        "recovery": recovery,
        "contrast": cont.point_estimate,
        "ci_lower": cont.ci_lower,
        "ci_upper": cont.ci_upper,
        "covered": covered,
    }


def _bcf_result(df: pd.DataFrame) -> dict:
    results = _BCF.estimate(df)
    cont = next((r for r in results if r.name == "bcf_bart_contrast"), None)
    if cont is None:
        return {"recovery": np.nan, "contrast": np.nan,
                "ci_lower": np.nan, "ci_upper": np.nan, "covered": False}
    top_var  = (cont.convergence_info or {}).get("top_split_var", "")
    recovery = 1.0 if top_var == "W1" else 0.0
    covered  = cont.ci_lower <= TRUE_CONTRAST <= cont.ci_upper
    return {
        "recovery": recovery,
        "contrast": cont.point_estimate,
        "ci_lower": cont.ci_lower,
        "ci_upper": cont.ci_upper,
        "covered": covered,
    }


# ---------------------------------------------------------------------------
# Null FPR helpers (same estimators, different cfg)
# ---------------------------------------------------------------------------

def _null_fpr_efx(df: pd.DataFrame) -> float:
    results = _EFX.estimate(df)
    cont = next((r for r in results if r.name == "effectxshift_contrast"), None)
    if cont is None or not np.isfinite(cont.standard_error):
        return np.nan
    z = abs(cont.point_estimate / cont.standard_error)
    return float(z > 1.96)


def _null_fpr_bcf(df: pd.DataFrame) -> float:
    results = _BCF.estimate(df)
    cont = next((r for r in results if r.name == "bcf_bart_contrast"), None)
    if cont is None or not np.isfinite(cont.standard_error):
        return np.nan
    z = abs(cont.point_estimate / cont.standard_error)
    return float(z > 1.96)


def _null_fpr_xlearner(df: pd.DataFrame) -> float:
    res = _xlearner_naive(df)
    if not np.isfinite(res["contrast"]):
        return np.nan
    se = (res["ci_upper"] - res["ci_lower"]) / (2 * 1.96)
    z  = abs(res["contrast"] / se) if se > 0 else np.inf
    return float(z > 1.96)


def _null_fpr_oracle(df: pd.DataFrame) -> float:
    res = _oracle(df)
    se = (res["ci_upper"] - res["ci_lower"]) / (2 * 1.96)
    z  = abs(res["contrast"] / se) if se > 0 else np.inf
    return float(z > 1.96)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

ESTIMATORS = {
    "effectxshift": _efx_result,
    "bcf_bart":     _bcf_result,
    "xlearner_naive": lambda df: _xlearner_naive(df),
    "oracle":         lambda df: _oracle(df),
}

NULL_FPR_FNS = {
    "effectxshift":   _null_fpr_efx,
    "bcf_bart":       _null_fpr_bcf,
    "xlearner_naive": _null_fpr_xlearner,
    "oracle":         _null_fpr_oracle,
}


def _run_sweep() -> pd.DataFrame:
    records = []
    for n in SAMPLE_SIZES:
        print(f"  n={n}")
        for rep in range(N_REPS):
            seed = 1000 * n + rep
            cfg      = BASE_CFG.with_overrides(n=n, seed=seed)
            null_cfg = NULL_CFG.with_overrides(n=n, seed=seed)
            df      = generate_data(cfg)
            df_null = generate_data(null_cfg)

            for est_name, fn in ESTIMATORS.items():
                res  = fn(df)
                null = NULL_FPR_FNS[est_name](df_null)
                records.append({
                    "n":            n,
                    "rep":          rep,
                    "estimator":    est_name,
                    "recovery":     res["recovery"],
                    "contrast":     res["contrast"],
                    "ci_lower":     res["ci_lower"],
                    "ci_upper":     res["ci_upper"],
                    "covered":      res["covered"],
                    "contrast_bias": res["contrast"] - TRUE_CONTRAST
                        if np.isfinite(res["contrast"]) else np.nan,
                    "null_fpr":     null,
                })
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_COLORS = {
    "effectxshift":   "#1f77b4",
    "bcf_bart":       "#ff7f0e",
    "xlearner_naive": "#2ca02c",
    "oracle":         "#9467bd",
}
_LABELS = {
    "effectxshift":   "EffectXShift (CV-TMLE)",
    "bcf_bart":       "BCF/BART tree",
    "xlearner_naive": "X-learner naive",
    "oracle":         "Oracle",
}


def _plot_power_curve(agg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for est in ESTIMATORS:
        sub = agg[agg["estimator"] == est].sort_values("n")
        ax.plot(sub["n"], sub["recovery_mean"], marker="o",
                color=_COLORS[est], label=_LABELS[est])
        ax.fill_between(sub["n"],
                        sub["recovery_mean"] - sub["recovery_se"],
                        sub["recovery_mean"] + sub["recovery_se"],
                        alpha=0.15, color=_COLORS[est])
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Subgroup recovery rate")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle="--")
    ax.legend(fontsize=8)
    ax.set_title("Subgroup recovery rate vs n")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _plot_coverage(agg: pd.DataFrame, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(1 - ALPHA, color="black", linewidth=1.0, linestyle="--",
               label="Nominal 95 %")
    for est in ESTIMATORS:
        sub = agg[agg["estimator"] == est].sort_values("n")
        ax.plot(sub["n"], sub["coverage_mean"], marker="s",
                color=_COLORS[est], label=_LABELS[est])
        ax.fill_between(sub["n"],
                        sub["coverage_mean"] - sub["coverage_se"],
                        sub["coverage_mean"] + sub["coverage_se"],
                        alpha=0.15, color=_COLORS[est])
    ax.set_xlabel("Sample size (n)")
    ax.set_ylabel("Empirical coverage (95 % CI)")
    ax.set_ylim(0.0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Coverage after selection vs n")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Running exp21 HTE subgroup sweep …")
    raw = _run_sweep()
    raw.to_parquet(OUT_DIR / "results.parquet", index=False)

    # Aggregate
    grp = raw.groupby(["estimator", "n"])
    agg = grp.agg(
        recovery_mean=("recovery",     "mean"),
        recovery_se=  ("recovery",     lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        bias_mean=    ("contrast_bias","mean"),
        bias_se=      ("contrast_bias",lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        coverage_mean=("covered",      "mean"),
        coverage_se=  ("covered",      lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        null_fpr_mean=("null_fpr",     "mean"),
        null_fpr_se=  ("null_fpr",     lambda x: x.std(ddof=1) / np.sqrt(len(x))),
        n_reps=       ("rep",          "count"),
    ).reset_index()

    agg.to_csv(OUT_DIR / "metrics_table.csv", index=False)

    # Console table
    print("\n── Metrics table (aggregated over reps) ──")
    print(agg.to_string(index=False, float_format="{:.3f}".format))

    _plot_power_curve(agg, OUT_DIR / "power_curve.png")
    _plot_coverage(agg,    OUT_DIR / "coverage_plot.png")

    print(f"\nDone. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
