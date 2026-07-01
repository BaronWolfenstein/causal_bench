"""Exp 25: Win ratio vs Bayesian CLMM on ordinal PROs — efficiency & coverage under PO violation.

SKELETON — gated on `blind-contours/concrete#36`.

The headline experiment of the ordinal-PRO epic (#25). A Monte-Carlo sweep pitting the
distribution-free **PRO win ratio** (concrete, GPC target) against the parametric
**Bayesian CLMM** (`clmm_ordinal`, cumulative-log-OR target) on the ordinal-PRO DGP
(`dgp/ordinal_pro.py`), demonstrating the efficiency-vs-robustness trade-off.

Hypothesis (state, then test):
  - Under PO-respecting truth the CLMM is MORE efficient (smaller SE/RMSE) and correctly
    covers; the win ratio is robust but less efficient.
  - Under PO violation the CLMM is BIASED with coverage collapse (one coefficient cannot
    represent effects that change sign across thresholds); the win ratio degrades gracefully.

Design (issue #28):
  - Primary sweep: the PO-violation knob, PO-respecting → strongly PO-violating.
  - Secondary sweep: ICC / site count (ties to exp24_site_clustering / exp19).
  - Each estimator scored on ITS OWN estimand against known truth (cumulative log-OR for
    the CLMM, ordinal win ratio for the GPC).
  - Outputs to results/exp25_ordinal_pro/*.parquet; figures wired into the index.qmd appendix.

╔══════════════════════════════════════════════════════════════════════════════════════╗
║ GATE — this experiment CANNOT fully run until concrete#36 merges.                      ║
║   • The PRO win-ratio comparator (ConcretePROWinRatioEstimator) self-gates via         ║
║     _concrete_available() and returns [] until concrete/#36 is installed.              ║
║   • The CLMM arms (bayes extra) run today, so the skeleton is runnable-but-degraded.   ║
╚══════════════════════════════════════════════════════════════════════════════════════╝

OPEN DESIGN QUESTIONS to resolve when #36 lands (do NOT silently paper over):
  1. SIGN RECONCILIATION (#28): the DGP reports cumulative log-OR on the P(Y<=j)
     convention (positive tau -> NEGATIVE log-OR); the CLMM reports on the latent /
     P(Y>=k) convention (positive tau -> POSITIVE). `_CLMM_SIGN` below negates the DGP
     truth to match the estimator. VERIFY this against a PO-respecting run before trusting
     any coverage number.
  2. TRUE CLMM TARGET UNDER VIOLATION: under PO violation there is no single cumulative
     log-OR. We use the mean of the per-threshold true log-ORs as the marginal target;
     confirm this is the estimand the CLMM's single coefficient actually converges to
     (it is a modeling choice, not a fact).
  3. PRO-ONLY vs COMPOSITE: the ordinal-PRO DGP has no survival composite. To score the
     concrete PRO win ratio we scaffold an administratively-censored survival part
     (T_obs = horizon, Delta = 0) so the pairwise ranking is driven purely by the ordinal
     tier. Confirm this matches the clinicalWinRatio(pro=...) contract post-#36.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from causal_bench.dgp.ordinal_pro import (
    OrdinalPROConfig,
    generate_data,
    compute_true_cumulative_logOR,
    compute_true_ordinal_win_ratio,
)
from causal_bench.estimators import ESTIMATOR_REGISTRY
from causal_bench.estimators.concrete_pro_win_ratio import ConcretePROWinRatioEstimator
from causal_bench.metrics import SimResult

OUT_DIR = Path("results/exp25_ordinal_pro")
N_REPS = 200

# See OPEN DESIGN QUESTION #1: negate DGP (P(Y<=j)) truth to the CLMM (P(Y>=k)) convention.
_CLMM_SIGN = -1.0

# Primary sweep — the PO-violation knob (per-threshold offsets, symmetric fan-out).
# strength 0.0 == PO-respecting; larger == stronger violation.
PO_VIOLATION_GRID = [0.0, 0.3, 0.6, 0.9, 1.2]

# Secondary sweep — site clustering (ties to exp24). (n_sites, ICC).
SITE_GRID = [(1, 0.0), (8, 0.05), (8, 0.20)]

BASE = dict(n=700, K=4, tau=0.7, treatment_prevalence=0.5, seed=42)

# CLMM arms — the pooling spectrum + random slope (all concrete-independent, run today).
CLMM_ARMS = ["clmm_ordinal", "clmm_ordinal_slope", "clmm_ordinal_nopool", "clmm_ordinal_cpool"]


def _make_config(po_strength: float, n_sites: int, icc: float, seed: int) -> OrdinalPROConfig:
    """Build an OrdinalPROConfig for one grid cell.

    The PO-violation knob is realized as symmetric per-threshold offsets
    (+s, 0, -s) that fan the true log-OR out across thresholds.
    """
    offsets = tuple(np.linspace(po_strength, -po_strength, BASE["K"] - 1)) if po_strength else ()
    return OrdinalPROConfig(
        **{**BASE, "seed": seed},
        tau_category_offsets=offsets,
        n_sites=n_sites,
        site_icc=icc,
    )


def _true_targets(cfg: OrdinalPROConfig) -> dict[str, float]:
    """True estimand values for this cfg, each on the estimator's own convention."""
    lor = compute_true_cumulative_logOR(cfg, n_ref=100_000)["log_OR"]
    wr = compute_true_ordinal_win_ratio(cfg, n_ref=100_000)["ATE"]
    return {
        # OPEN QUESTION #2: marginal cumulative log-OR = mean of per-threshold log-ORs.
        "cumulative_log_OR": _CLMM_SIGN * float(np.mean(lor)),
        "WR": float(wr),
    }


def _add_survival_scaffold(df: pd.DataFrame, horizon: float) -> pd.DataFrame:
    """OPEN QUESTION #3: PRO-only DGP → administratively-censored survival part so the
    concrete win ratio ranks purely on the ordinal tier. Revisit against the #36 contract."""
    df = df.copy()
    df["T_obs"] = horizon
    df["Delta"] = 0
    df["event_type"] = 0          # prepare_for_r requires this; 0 == censored (no event)
    return df


def _win_ratio_arm(horizon: float = 1.0) -> ConcretePROWinRatioEstimator:
    """The concrete PRO win-ratio comparator on the ordinal marker (inactive until #36)."""
    return ConcretePROWinRatioEstimator(
        pro_specs=[{"marker": "ordinal_pro", "type": "ordinal",
                    "direction": "higher.better", "landmark": horizon}],
        horizon=horizon,
    )


def _run_cell(cfg: OrdinalPROConfig, n_sim: int, horizon: float, seed: int) -> dict[str, SimResult]:
    """Run all arms for one grid cell; return {arm_name: SimResult}."""
    truths = _true_targets(cfg)
    wr_est = _win_ratio_arm(horizon)

    # arm -> lists of per-replicate point/SE/CI
    acc: dict[str, dict[str, list]] = {}

    def _stash(name: str, estimand: str, r) -> None:
        a = acc.setdefault(name, {"estimand": estimand, "pt": [], "se": [], "lo": [], "hi": []})
        a["pt"].append(r.point_estimate); a["se"].append(r.standard_error)
        a["lo"].append(r.ci_lower); a["hi"].append(r.ci_upper)

    rng = np.random.default_rng(seed)
    for i in range(n_sim):
        cfg_i = cfg.model_copy(update={"seed": int(rng.integers(1, 2**31))})
        # Column-contract adapter. The registered CLMM arms expect outcome_col="support"
        # and site_col="site"; the ordinal DGP emits "ordinal_pro" and "site_id". We alias
        # "support" (keeping "ordinal_pro" for the win-ratio marker) and rename the site.
        # TODO(#28/#26): reconcile at source so no adapter is needed.
        df = generate_data(cfg_i).rename(columns={"site_id": "site"})
        df["support"] = df["ordinal_pro"]

        for arm in CLMM_ARMS:
            for r in ESTIMATOR_REGISTRY[arm].estimate(df, estimand="cumulative_log_OR"):
                _stash(arm, "cumulative_log_OR", r)

        # Gated: skipped (with the whole experiment continuing) until concrete/#36 is
        # active. Any failure — concrete absent, or present-but-pre-#36 (missing R
        # functions / column contract) — is treated as "inactive", not a crash.
        try:
            for r in wr_est.estimate(_add_survival_scaffold(df, horizon), estimand="WR"):
                _stash(wr_est.name, "WR", r)
        except Exception:
            pass

    out: dict[str, SimResult] = {}
    for name, a in acc.items():
        if not a["pt"]:
            continue
        nrep = len(a["pt"])
        out[name] = SimResult(
            estimator_name=name,
            estimand=a["estimand"],
            true_value=truths[a["estimand"]],
            n_sim=nrep,
            estimates=np.asarray(a["pt"]),
            se_estimates=np.asarray(a["se"]),
            ci_lowers=np.asarray(a["lo"]),
            ci_uppers=np.asarray(a["hi"]),
            nc_estimates=np.full(nrep, np.nan),
        )
    return out


def run(n_sims: int = N_REPS, horizon: float = 1.0, seed: int = 42) -> pd.DataFrame:
    """Primary + secondary sweeps. Returns a tidy DataFrame of per-arm metrics per cell."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        _probe = _win_ratio_arm(horizon).estimate(
            _add_survival_scaffold(generate_data(_make_config(0.0, 1, 0.0, seed)), horizon)
        )
    except Exception:
        _probe = []
    if not _probe:
        print("NOTE: PRO win-ratio comparator inactive (concrete#36 not merged) — "
              "running CLMM arms only. Win-ratio rows will populate once #36 lands.")

    rows = []
    for po in PO_VIOLATION_GRID:
        for (n_sites, icc) in SITE_GRID:
            cfg = _make_config(po, n_sites, icc, seed)
            cell = _run_cell(cfg, n_sims, horizon, seed)
            for name, sr in cell.items():
                rows.append({
                    "arm": name, "estimand": sr.estimand,
                    "po_strength": po, "n_sites": n_sites, "icc": icc,
                    "bias": sr.bias, "rmse": sr.rmse,
                    "coverage": sr.coverage, "se_ratio": sr.se_ratio,
                    "true_value": sr.true_value, "n_sim": sr.n_sim,
                })
                sr.to_parquet(OUT_DIR / f"po{po}_sites{n_sites}_icc{icc}_{name}.parquet")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "metrics_table.csv", index=False)
    if not df.empty:
        _plot(df)
    print(f"\nSaved metrics + parquet → {OUT_DIR}/")
    return df


def _plot(df: pd.DataFrame) -> None:
    """Two-panel: RMSE (efficiency) and coverage vs PO-violation strength, per arm.
    Uses the no-clustering slice (n_sites==1) as the primary view."""
    prim = df[df["n_sites"] == 1] if (df["n_sites"] == 1).any() else df
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for arm, g in prim.groupby("arm"):
        g = g.sort_values("po_strength")
        axes[0].plot(g["po_strength"], g["rmse"], marker="o", label=arm)
        axes[1].plot(g["po_strength"], g["coverage"], marker="o", label=arm)
    axes[0].set(title="Efficiency", xlabel="PO-violation strength", ylabel="RMSE")
    axes[1].axhline(0.95, ls="--", color="gray", lw=1)
    axes[1].set(title="Coverage", xlabel="PO-violation strength", ylabel="coverage", ylim=(0, 1))
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "efficiency_coverage.png", dpi=120)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="Exp 25: win ratio vs CLMM on ordinal PROs")
    p.add_argument("--n-sims", type=int, default=N_REPS)
    p.add_argument("--horizon", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    run(n_sims=args.n_sims, horizon=args.horizon, seed=args.seed)
