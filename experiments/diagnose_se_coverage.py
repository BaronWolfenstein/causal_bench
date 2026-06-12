"""SE coverage diagnostics for TMLE+IPCW across exp4/exp6/exp9.

Three analyses:
  1. se_ratio gradient plots from existing summary tables (no re-run needed)
  2. ESS check — effective sample size of IPCW censoring weights at worst params
  3. IC bootstrap coverage test — IC-based SE vs bootstrap SE for a single dataset

Run:
    .venv/bin/python experiments/diagnose_se_coverage.py
"""
from __future__ import annotations

import io
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = Path("results/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TMLE_ESTIMATORS = ["tmle_ipcw", "tmle_ipcw_comply"]
PALETTE = {"tmle_ipcw": "#d62728", "tmle_ipcw_comply": "#ff7f0e",
           "naive": "#7f7f7f", "km": "#bcbd22", "cox": "#17becf"}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _parse_summary_dir(result_dir: Path, param_name: str) -> pd.DataFrame:
    """Parse all summary_*.md files in result_dir into a tidy DataFrame."""
    rows = []
    for f in sorted(result_dir.glob("summary_*.md")):
        # Extract param value from filename: e.g. summary_drift0.3.md → 0.3
        stem = f.stem.replace("summary_", "")
        val_str = "".join(c for c in stem if c in "0123456789.-")
        try:
            val = float(val_str)
        except ValueError:
            continue
        tbl = pd.read_csv(io.StringIO(f.read_text()), sep="|", skipinitialspace=True)
        tbl = tbl.dropna(how="all", axis=1).dropna(how="all")
        tbl.columns = [c.strip() for c in tbl.columns]
        tbl = tbl[~tbl["estimator"].str.contains("---")]
        tbl["estimator"] = tbl["estimator"].str.strip()
        for col in ["bias", "rmse", "coverage", "ci_width", "se_ratio"]:
            tbl[col] = pd.to_numeric(tbl[col].astype(str).str.strip(), errors="coerce")
        tbl[param_name] = val
        rows.append(tbl)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _plot_se_ratio_gradient(df: pd.DataFrame, param_name: str, title: str,
                            save_path: Path, target_estimators=None) -> None:
    if target_estimators is None:
        target_estimators = df["estimator"].unique().tolist()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for ax, metric in zip(axes, ["se_ratio", "coverage"]):
        for est in target_estimators:
            sub = df[df["estimator"] == est].sort_values(param_name)
            color = PALETTE.get(est, None)
            lw = 2.5 if est in TMLE_ESTIMATORS else 1.2
            ls = "--" if "comply" in est else "-"
            ax.plot(sub[param_name], sub[metric], label=est,
                    color=color, linewidth=lw, linestyle=ls, marker="o", markersize=5)

        if metric == "se_ratio":
            ax.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
            ax.set_ylabel("SE ratio  (median IC-SE / empirical SD)")
            ax.set_title(f"SE ratio vs {param_name}")
        else:
            ax.axhline(0.95, color="black", linewidth=0.8, linestyle=":")
            ax.set_ylabel("Coverage (nominal 95%)")
            ax.set_title(f"Coverage vs {param_name}")

        ax.set_xlabel(param_name)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {save_path}")


# ─── Part 1: se_ratio gradient plots ──────────────────────────────────────────

def plot_gradient_from_summaries() -> None:
    print("\n=== Part 1: SE ratio gradient plots ===")

    configs = [
        ("results/exp6_drift",       "enrollment_drift",  "Exp 6: Enrollment drift vs SE calibration"),
        ("results/exp9_sample_size", "n",                 "Exp 9: Sample size vs SE calibration"),
    ]

    for result_dir, param, title in configs:
        rdir = Path(result_dir)
        if not rdir.exists():
            print(f"  SKIP {result_dir} (directory not found)")
            continue

        df = _parse_summary_dir(rdir, param)
        if df.empty:
            print(f"  SKIP {result_dir} (no summary files)")
            continue

        all_ests = df["estimator"].unique().tolist()
        save_path = OUT_DIR / f"se_ratio_{rdir.name}.png"
        _plot_se_ratio_gradient(df, param, title, save_path, target_estimators=all_ests)

        # Print the tmle_ipcw se_ratio table
        sub = df[df["estimator"].isin(TMLE_ESTIMATORS)].sort_values(param)
        print(f"\n  {rdir.name} — tmle_ipcw se_ratio and coverage:")
        print(sub[[param, "estimator", "se_ratio", "coverage"]].to_string(index=False))


def _with_n(cfg, n: int):
    import dataclasses
    return dataclasses.replace(cfg, n=n)


# ─── Part 2: ESS check ────────────────────────────────────────────────────────

def _fit_ipcw_and_extract(df: pd.DataFrame, horizon: float = 1.0,
                          use_compliance: bool = False):
    """Fit the TMLE censoring model on df and return (G, ESS, ipcw_weights)."""
    from lifelines import CoxPHFitter

    W_cols = ["W1", "W2", "W3", "W4"]
    censor_feature_cols = W_cols + ["A"]
    if use_compliance and "compliance" in df.columns:
        censor_feature_cols = censor_feature_cols + ["compliance"]

    censor_df = df[censor_feature_cols + ["T_obs", "Delta"]].copy()
    censor_df = censor_df.rename(columns={"Delta": "event_obs"})
    censor_df["C_indicator"] = (
        (censor_df["event_obs"] == 0) & (censor_df["T_obs"] < horizon - 1e-9)
    ).astype(float)

    try:
        cph = CoxPHFitter(penalizer=0.1)
        cph.fit(censor_df[censor_feature_cols + ["T_obs", "C_indicator"]],
                duration_col="T_obs", event_col="C_indicator",
                fit_options={"max_steps": 50})
        T_obs = df["T_obs"].values
        sf = cph.predict_survival_function(censor_df[censor_feature_cols],
                                           times=np.sort(np.unique(T_obs)))
        n = len(T_obs)
        G = np.ones(n)
        for i, t in enumerate(T_obs):
            col = sf.iloc[:, i]
            idx_before = sf.index <= t
            if idx_before.any():
                G[i] = float(col[idx_before].iloc[-1])
            else:
                G[i] = 1.0
    except Exception:
        G = np.ones(len(df))

    G = np.clip(G, 0.05, 1.0)
    Delta = df["Delta"].values.astype(float)
    T_obs_arr = df["T_obs"].values
    admin_censored = (Delta == 0) & (T_obs_arr >= horizon - 1e-9)
    ipcw = np.where(Delta == 1, 1.0 / G,
                    np.where(admin_censored, 1.0, 0.0))

    w = ipcw[Delta == 1]
    ess = (w.sum() ** 2) / (w ** 2).sum() if len(w) > 0 else 0.0
    return G, ess, ipcw


def ess_check() -> None:
    """Run TMLE censoring model on worst-case params; report ESS distribution."""
    print("\n=== Part 2: ESS check ===")
    from causal_bench.dgp.config import DGPConfig
    from causal_bench.dgp.scenarios import get_scenario
    from causal_bench.dgp.survival import generate_data

    checks = [
        ("exp6 drift=0.5",  DGPConfig(n=500, censoring_informativeness=0.0,
                                      enrollment_drift=0.5, true_tau=-0.5)),
        ("exp9 n=2000",     _with_n(get_scenario("edwards_realistic"), 2000)),
    ]

    n_rep = 50
    rng = np.random.default_rng(42)

    for label, cfg in checks:
        esss, fracs = [], []
        for _ in range(n_rep):
            df = generate_data(cfg, rng)
            _, ess, ipcw = _fit_ipcw_and_extract(df)
            n_events = int((df["Delta"] == 1).sum())
            fracs.append(ess / n_events if n_events > 0 else np.nan)
            esss.append(ess)
        print(f"\n  {label}:")
        print(f"    median ESS         = {np.nanmedian(esss):.1f}")
        print(f"    median ESS/n_events = {np.nanmedian(fracs):.3f}")
        print(f"    min ESS            = {np.nanmin(esss):.1f}")


# ─── Part 3: IC bootstrap coverage test ───────────────────────────────────────

def ic_bootstrap_test() -> None:
    """Compare IC analytical SE vs bootstrap SE from IC resampling."""
    print("\n=== Part 3: IC bootstrap coverage test ===")
    from causal_bench.dgp.config import DGPConfig
    from causal_bench.dgp.scenarios import get_scenario
    from causal_bench.dgp.survival import generate_data
    from causal_bench.estimators import ESTIMATOR_REGISTRY

    checks = [
        ("exp6 drift=0.5",  DGPConfig(n=500, censoring_informativeness=0.0,
                                      enrollment_drift=0.5, true_tau=-0.5)),
        ("exp9 n=2000",     _with_n(get_scenario("edwards_realistic"), 2000)),
    ]

    estimator = ESTIMATOR_REGISTRY["tmle_ipcw"]
    n_rep = 30
    n_bootstrap = 500
    rng = np.random.default_rng(42)

    for label, cfg in checks:
        ic_ses, boot_ses, points = [], [], []
        for _ in range(n_rep):
            df = generate_data(cfg, rng)
            try:
                results = estimator.estimate(df)
            except Exception:
                continue
            for r in results:
                if r.estimand == "ATE" and r.ic is not None:
                    ic = r.ic
                    n = len(ic)
                    # IC analytical SE
                    ic_se = float(np.sqrt(np.var(ic, ddof=1) / n))
                    # Bootstrap SE via resampling the IC values (not full refits)
                    boot_means = [
                        float(np.mean(ic[rng.integers(0, n, size=n)]))
                        for _ in range(n_bootstrap)
                    ]
                    boot_se = float(np.std(boot_means, ddof=1))
                    ic_ses.append(ic_se)
                    boot_ses.append(boot_se)
                    points.append(r.point_estimate)
                    break

        if not ic_ses:
            print(f"  {label}: no IC vectors returned")
            continue

        ratio = np.median(ic_ses) / np.median(boot_ses)
        print(f"\n  {label}  (n={n_rep} datasets):")
        print(f"    median IC-analytical SE = {np.median(ic_ses):.5f}")
        print(f"    median IC-bootstrap SE  = {np.median(boot_ses):.5f}")
        print(f"    ratio (analytical/boot) = {ratio:.4f}  "
              f"({'underestimates' if ratio < 0.99 else 'calibrated'})")


# ─── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    plot_gradient_from_summaries()
    ess_check()
    ic_bootstrap_test()
    print("\nDone. Outputs in results/diagnostics/")
