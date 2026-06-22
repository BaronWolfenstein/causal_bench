"""Design validity diagnostics — RDD placebo calibration.

Pre-screen for observational design validity before the embedding-space
localisation procedure (localization.py). This module is independent of
SubgroupModel and issue #13 — it operates entirely in running-variable
space (R¹), not embedding space.

Role in the decision procedure
-------------------------------
run_diagnostic() in localization.py assumes the observational design is
already validated. This module provides that pre-screen:

    rdd_placebo_test() → passed → run_diagnostic() → terminals
                       → failed → escalate before touching embeddings

RDD as placebo calibration vs. primary identification
------------------------------------------------------
RDD identifies causal effects LOCAL to a threshold. It can serve as:

  1. Design validity check (this module's purpose): run RDD on a null
     period or placebo outcome — a significant discontinuity signals
     confounding at the threshold.

  2. Primary identification for the rare cohort: ONLY if the rare cohort
     clusters near the threshold with sufficient density.
     plot_running_var_density() provides the density diagnostic to test
     this prerequisite. In practice, if the rare cohort is defined by a
     severe positivity violation (e.g. failed-TEER history), it is
     typically far from any administrative threshold — RDD cannot serve as
     primary ID in that regime.  The embedding-space machinery
     (localization.py, embedding_eda.py) handles primary identification
     for patients far from any threshold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class DesignValidityResult:
    """Result of the RDD placebo calibration test."""
    passed: bool    # True = no spurious discontinuity in the null window
    metrics: dict   # gap, se_gap, z_stat, p_value, bandwidth_used,
                    # n_left, n_right, [rare_n_near_cutoff, rare_pct_near_cutoff]
    notes: str      # one-paragraph human-readable interpretation


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _auto_bandwidth(running_var: np.ndarray) -> float:
    """Rule-of-thumb bandwidth: 0.2 × range, clamped to ≥ 1.5 × std × n^{-1/5}."""
    rv = running_var
    range_bw   = 0.2 * (float(rv.max()) - float(rv.min()))
    silverman  = 1.5 * float(rv.std()) * len(rv) ** (-0.2)
    return max(range_bw, silverman, 1e-8)


def _wls_intercept_at_cutoff(
    y: np.ndarray,
    x_centered: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float]:
    """Weighted local linear regression; return (intercept, se_intercept).

    Design matrix is [1, x_centered], so the intercept estimates the local
    conditional mean at the cutoff (x_centered = 0).  SE uses the sandwich
    estimator assuming homoscedastic errors.
    """
    if len(y) < 3:
        return float("nan"), float("nan")

    W_sqrt = np.sqrt(weights)
    X = np.column_stack([np.ones(len(y)), x_centered])
    Xw = X * W_sqrt[:, None]
    yw = y * W_sqrt

    try:
        beta, _, _, _ = np.linalg.lstsq(Xw, yw, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), float("nan")

    residuals = y - X @ beta
    rss = float(np.sum(weights * residuals ** 2))
    sigma2 = rss / max(len(y) - 2, 1)

    XtWX = X.T @ (X * weights[:, None])
    try:
        cov = sigma2 * np.linalg.inv(XtWX)
        se_intercept = float(np.sqrt(max(cov[0, 0], 0.0)))
    except np.linalg.LinAlgError:
        se_intercept = float("nan")

    return float(beta[0]), se_intercept


# ─── Core test ────────────────────────────────────────────────────────────────

def rdd_placebo_test(
    placebo_outcomes: np.ndarray,
    running_var: np.ndarray,
    cutoff: float,
    bandwidth: Optional[float] = None,
    rare_mask: Optional[np.ndarray] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
) -> DesignValidityResult:
    """RDD placebo calibration: test for spurious discontinuity in a null window.

    Fits local linear regression on each side of the cutoff within `bandwidth`,
    using kernel weights.  The gap (right intercept − left intercept at the
    cutoff) should be near zero in a null/pre-period — a statistically
    significant gap signals design confounding at the threshold.

    Also reports rare-cohort density near the cutoff: if `rare_mask` is
    provided, `rare_n_near_cutoff` counts how many rare patients fall within
    the bandwidth.  Low counts mean RDD cannot serve as primary identification
    for the rare subgroup even if the placebo check passes.

    Parameters
    ----------
    placebo_outcomes : (n,) outcomes in a null period / on a placebo outcome.
        A significant discontinuity here → design is biased.
    running_var  : (n,) continuous score defining the threshold.
    cutoff       : threshold value (treatment assignment changes here).
    bandwidth    : half-width of the local window.  If None, auto-selected via
        rule-of-thumb (0.2 × range, ≥ 1.5 × std × n^{-1/5}).
    rare_mask    : (n,) bool, True for rare-cohort patients.  Used only to
        report density near the cutoff; does not affect the test statistic.
    kernel       : "triangular" (default, down-weights boundary patients) or
        "uniform" (equal weight within bandwidth).
    alpha        : significance level for the placebo test.

    Returns
    -------
    DesignValidityResult with:
        passed  True  = p_value ≥ alpha (no spurious discontinuity)
                False = p_value < alpha (design bias detected)
        metrics gap, se_gap, z_stat, p_value, bandwidth_used,
                n_left, n_right, [rare_n_near_cutoff, rare_pct_near_cutoff]
        notes   One-paragraph human-readable interpretation.
    """
    from scipy.stats import norm as _norm

    y = np.asarray(placebo_outcomes, dtype=float)
    x = np.asarray(running_var, dtype=float)

    bw = bandwidth if bandwidth is not None else _auto_bandwidth(x)

    left_mask  = (x <= cutoff) & (x >= cutoff - bw)
    right_mask = (x >  cutoff) & (x <= cutoff + bw)

    def _kernel_weights(x_side: np.ndarray) -> np.ndarray:
        u = (x_side - cutoff) / bw
        if kernel == "triangular":
            return np.maximum(1.0 - np.abs(u), 0.0)
        return np.ones(len(x_side))

    int_left, se_left = _wls_intercept_at_cutoff(
        y[left_mask], x[left_mask] - cutoff, _kernel_weights(x[left_mask])
    )
    int_right, se_right = _wls_intercept_at_cutoff(
        y[right_mask], x[right_mask] - cutoff, _kernel_weights(x[right_mask])
    )

    gap    = int_right - int_left
    se_gap = (
        float(np.sqrt(se_left ** 2 + se_right ** 2))
        if (np.isfinite(se_left) and np.isfinite(se_right))
        else float("nan")
    )

    if np.isfinite(gap) and np.isfinite(se_gap) and se_gap > 1e-12:
        z_stat  = gap / se_gap
        p_value = float(2 * (1 - _norm.cdf(abs(z_stat))))
    else:
        z_stat  = float("nan")
        p_value = float("nan")

    passed = bool(not np.isfinite(p_value) or p_value >= alpha)

    n_left  = int(left_mask.sum())
    n_right = int(right_mask.sum())

    metrics: dict = {
        "gap":            float(gap)     if np.isfinite(gap)     else float("nan"),
        "se_gap":         float(se_gap)  if np.isfinite(se_gap)  else float("nan"),
        "z_stat":         float(z_stat)  if np.isfinite(z_stat)  else float("nan"),
        "p_value":        float(p_value) if np.isfinite(p_value) else float("nan"),
        "bandwidth_used": float(bw),
        "n_left":         n_left,
        "n_right":        n_right,
    }

    if rare_mask is not None:
        rm        = np.asarray(rare_mask, dtype=bool)
        rare_near = int(((left_mask | right_mask) & rm).sum())
        rare_pct  = float(rare_near / max(int(rm.sum()), 1) * 100)
        metrics["rare_n_near_cutoff"]   = rare_near
        metrics["rare_pct_near_cutoff"] = rare_pct

    # ── Notes ──────────────────────────────────────────────────────────────────
    if not np.isfinite(p_value):
        notes = (
            f"RDD placebo test inconclusive: insufficient data within bandwidth "
            f"{bw:.3g} (n_left={n_left}, n_right={n_right}).  "
            "Widen the bandwidth or collect more data near the threshold."
        )
    elif passed:
        notes = (
            f"Placebo test PASSED (gap={gap:.3g}, SE={se_gap:.3g}, "
            f"z={z_stat:.2f}, p={p_value:.3f} ≥ {alpha}).  "
            f"No spurious discontinuity in the null window "
            f"(n_left={n_left}, n_right={n_right}, bandwidth={bw:.3g}).  "
            "Observational design appears valid at this threshold."
        )
    else:
        notes = (
            f"Placebo test FAILED (gap={gap:.3g}, SE={se_gap:.3g}, "
            f"z={z_stat:.2f}, p={p_value:.3f} < {alpha}).  "
            f"Significant discontinuity detected in the null window "
            f"(n_left={n_left}, n_right={n_right}, bandwidth={bw:.3g}).  "
            "Design is confounded at this threshold — do not use RDD here."
        )

    if rare_mask is not None and "rare_n_near_cutoff" in metrics:
        rn = metrics["rare_n_near_cutoff"]
        rp = metrics["rare_pct_near_cutoff"]
        if rn < 10:
            notes += (
                f"  NOTE: only {rn} rare-cohort patients ({rp:.1f}%) fall within "
                f"the bandwidth — RDD cannot serve as primary identification for "
                "the rare subgroup here regardless of placebo result.  "
                "Use the embedding-space localisation procedure (localization.py) instead."
            )
        else:
            notes += (
                f"  {rn} rare-cohort patients ({rp:.1f}%) fall within the bandwidth."
            )

    return DesignValidityResult(passed=passed, metrics=metrics, notes=notes)


# ─── Density plot ─────────────────────────────────────────────────────────────

def plot_running_var_density(
    running_var: np.ndarray,
    cutoff: float,
    bandwidth: float,
    rare_mask: Optional[np.ndarray] = None,
    bins: int = 40,
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Density histogram of patients along the running variable.

    Primary diagnostic for whether rare patients cluster near the threshold —
    a prerequisite for trusting RDD as primary identification of the rare
    subgroup.  If rare patients are sparse near the cutoff, RDD cannot serve as
    primary identification there regardless of whether the placebo check passes.

    Parameters
    ----------
    running_var : (n,) continuous score.
    cutoff      : threshold value.
    bandwidth   : half-width used in rdd_placebo_test(); shown as shaded region.
    rare_mask   : (n,) bool, rare-cohort patients (plotted in red).
    bins        : histogram bins (default 40).
    save_path   : if given, saves figure at 150 dpi.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x   = np.asarray(running_var, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))

    bin_edges = np.linspace(x.min(), x.max(), bins + 1)
    ax.hist(x, bins=bin_edges, color="#3182BD", alpha=0.55, density=True,
            label="All patients")

    if rare_mask is not None:
        rm     = np.asarray(rare_mask, dtype=bool)
        x_rare = x[rm]
        if len(x_rare) > 0:
            ax.hist(x_rare, bins=bin_edges, color="#E34A33", alpha=0.65,
                    density=True, label=f"Rare cohort (n={int(rm.sum())})")

    ax.axvspan(cutoff - bandwidth, cutoff + bandwidth,
               alpha=0.10, color="#31A354",
               label=f"Bandwidth ±{bandwidth:.3g}")
    ax.axvline(cutoff, color="#333", linewidth=1.8, linestyle="--",
               label=f"Cutoff = {cutoff:.3g}")

    title = "Running variable density — can RDD identify the rare subgroup?"
    if rare_mask is not None:
        rm = np.asarray(rare_mask, dtype=bool)
        n_near = int(((np.abs(x - cutoff) <= bandwidth) & rm).sum())
        total_rare = int(rm.sum())
        pct = 100 * n_near / max(total_rare, 1)
        title += f"\nRare patients within bandwidth: {n_near}/{total_rare} ({pct:.1f}%)"

    ax.set_xlabel("Running variable")
    ax.set_ylabel("Density")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ─── RDD scatter plot ─────────────────────────────────────────────────────────

def plot_rdd_scatter(
    outcomes: np.ndarray,
    running_var: np.ndarray,
    cutoff: float,
    bandwidth: float,
    rare_mask: Optional[np.ndarray] = None,
    kernel: str = "triangular",
    save_path: Optional[str] = None,
) -> "plt.Figure":
    """Scatter of (running_var, outcome) with local linear fits on each side.

    Visualises whether a discontinuity exists at the cutoff.  The visual gap
    between the two fitted lines at x = cutoff is the RDD estimate.  Absence of
    a gap in a placebo window confirms design validity.

    Patients within the bandwidth are drawn full-opacity; outside are greyed
    (they do not contribute to the local estimate).

    Parameters
    ----------
    outcomes    : (n,) outcome values.
    running_var : (n,) continuous score.
    cutoff      : threshold value.
    bandwidth   : half-width of the local estimation window.
    rare_mask   : (n,) bool, highlighted in red.
    kernel      : "triangular" or "uniform" (should match rdd_placebo_test).
    save_path   : if given, saves figure at 150 dpi.

    Returns
    -------
    matplotlib Figure.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    y    = np.asarray(outcomes, dtype=float)
    x    = np.asarray(running_var, dtype=float)
    in_bw = np.abs(x - cutoff) <= bandwidth
    left  = (x <= cutoff) & in_bw
    right = (x >  cutoff) & in_bw

    fig, ax = plt.subplots(figsize=(8, 5))

    # Outside-bandwidth points (grey)
    ax.scatter(x[~in_bw], y[~in_bw], c="#CCCCCC", s=8, alpha=0.4,
               label="Outside bandwidth", zorder=1)

    if rare_mask is not None:
        rm = np.asarray(rare_mask, dtype=bool)
        ax.scatter(x[in_bw & ~rm], y[in_bw & ~rm],
                   c="#3182BD", s=12, alpha=0.55, label="Common (in BW)", zorder=2)
        ax.scatter(x[in_bw & rm], y[in_bw & rm],
                   c="#E34A33", s=25, alpha=0.8, marker="^",
                   label="Rare (in BW)", zorder=3)
    else:
        ax.scatter(x[in_bw], y[in_bw], c="#3182BD", s=12, alpha=0.55,
                   label="In bandwidth", zorder=2)

    def _fit_and_plot(side_mask: np.ndarray, color: str,
                      x_grid: np.ndarray) -> None:
        x_s = x[side_mask] - cutoff
        y_s = y[side_mask]
        u   = x_s / bandwidth
        w_s = (np.maximum(1.0 - np.abs(u), 0.0)
               if kernel == "triangular" else np.ones(len(y_s)))
        if len(y_s) < 3:
            return
        W_sqrt = np.sqrt(w_s)
        X = np.column_stack([np.ones(len(y_s)), x_s])
        try:
            beta, _, _, _ = np.linalg.lstsq(
                X * W_sqrt[:, None], y_s * W_sqrt, rcond=None
            )
        except np.linalg.LinAlgError:
            return
        x_plot = x_grid - cutoff
        ax.plot(x_grid, beta[0] + beta[1] * x_plot, color=color, linewidth=2, zorder=4)
        ax.plot(cutoff, beta[0], "o", color=color, markersize=8, zorder=5)

    _fit_and_plot(left,  "#2171B5",
                  np.linspace(max(x.min(), cutoff - bandwidth), cutoff, 80))
    _fit_and_plot(right, "#D94801",
                  np.linspace(cutoff, min(x.max(), cutoff + bandwidth), 80))

    ax.axvline(cutoff, color="#333", linewidth=1.5, linestyle="--",
               label=f"Cutoff = {cutoff:.3g}", zorder=3)
    ax.axvspan(cutoff - bandwidth, cutoff + bandwidth,
               alpha=0.06, color="#31A354")

    ax.set_xlabel("Running variable")
    ax.set_ylabel("Outcome")
    ax.set_title(
        "RDD scatter: local linear fit on each side of the cutoff\n"
        "Gap at cutoff = RDD estimate  |  shaded region = bandwidth",
        fontsize=10,
    )
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
