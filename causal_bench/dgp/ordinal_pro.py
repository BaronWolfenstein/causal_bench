"""Ordinal PRO DGP — thresholded-latent model with site random effects and PO-violation knob.

Implements a cumulative logistic model (CLM) generative process for an ordinal
patient-reported outcome (e.g. NYHA I–IV or KCCQ tertiles).  Both the
proportional-odds (PO) respecting and PO-violating modes are supported so that
the benchmark can score both a CLMM (which targets the cumulative log-OR) and a
GPC win-ratio estimator (which targets the ordinal win ratio) against known truth.

Design — issue #26, epic #25
------------------------------
Thresholded-latent model::

    P(Y <= j | W, A, site) = logistic(c_j + delta_site_j - f(W) - b_site - tau_eff_j * A)

where:
    f(W) = 0.4*W1 - 0.3*W2 + 0.2*W3 - 0.2*W4        (prognostic signal, same as survival.py)
    b_site ~ N(0, sd_site²)                             (site random intercept)
    delta_site_j ~ N(0, site_threshold_sd²)             (site-varying threshold, optional)
    c_1 < c_2 < … < c_{K-1}                            (shared cutpoints)
    tau_eff_j = tau + tau_category_offsets[j]           (PO-respecting when all offsets = 0)

PO-violation knobs (all default OFF):
    tau_category_offsets  — K-1 per-threshold deviations from tau; zero → PO holds.
    site_threshold_sd     — per-site cutpoint spread; zero → no site threshold heterogeneity.
    floor_effect          — shrinks tau_eff at the lowest threshold (patients stuck at floor).
    ceiling_effect        — shrinks tau_eff at the highest threshold (patients capped at ceiling).

Estimands
---------
compute_true_cumulative_logOR  — marginal cumulative log-OR at each threshold (CLM target).
compute_true_ordinal_win_ratio — marginal ordinal win ratio P(Y1>Y0)/P(Y1<Y0) (GPC target).

Column contract
---------------
The generated DataFrame contains an ordinal marker column (default name ``ordinal_pro``)
with integer values 1..K.  This is directly consumable by::

    ConcretePROWinRatioEstimator(pro=[{"marker": "ordinal_pro", "type": "ordinal", ...}])

and by a CLMM estimator via the same column name.

Reproducibility
---------------
All random draws use ``np.random.default_rng(config.seed)``.  Same seed → identical
DataFrame, regardless of call context (keyed determinism at the config level).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def _default_cutpoints(K: int) -> np.ndarray:
    """Equally-spaced quantiles of the standard logistic CDF → balanced categories.

    Produces K-1 cutpoints c_1 < … < c_{K-1} such that each category has prior
    probability 1/K under the null (A=0, W=0, no site effect).
    """
    from scipy.stats import logistic as scipy_logistic
    probs = np.linspace(1.0 / K, (K - 1.0) / K, K - 1)
    return scipy_logistic.ppf(probs)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

class OrdinalPROConfig(BaseModel):
    """Configuration for the ordinal PRO DGP.

    Design mirrors ``DGPConfig`` in ``causal_bench/dgp/config.py``:
    frozen + extra="forbid" so mis-typed overrides raise at construction time.
    """

    model_config = {"frozen": True, "extra": "forbid"}

    # ── Sample ──────────────────────────────────────────────────────────────
    n: int = Field(500, ge=1, le=100_000)

    # ── Ordinal scale ───────────────────────────────────────────────────────
    # Number of ordinal categories (K=4 → NYHA I–IV, K=3 → KCCQ tertiles).
    K: int = Field(4, ge=2, le=10)

    # Explicit K-1 cutpoints c_1..c_{K-1}; None → equally-spaced logistic quantiles.
    # Must be strictly increasing when provided.
    cutpoints: Optional[tuple[float, ...]] = None

    # ── Treatment ───────────────────────────────────────────────────────────
    # Latent-scale treatment effect.  Positive tau shifts the latent upward → more
    # patients in higher (better) categories under treatment.  Under PO, this equals
    # the conditional cumulative log-OR at every threshold.
    tau: float = 0.5
    treatment_prevalence: float = Field(0.5, ge=0.0, le=1.0)

    # ── PO-violation knobs (all default OFF) ─────────────────────────────────
    # tau_category_offsets[j]: extra latent shift for threshold j beyond tau.
    # Empty tuple ≡ K-1 zeros → proportional odds holds.
    # len must equal K-1 when non-empty (validated below).
    tau_category_offsets: tuple[float, ...] = Field(default=())

    # SD of per-site cutpoint perturbations (independent of site random intercept).
    # When > 0, each site gets different effective thresholds → marginal PO violated.
    site_threshold_sd: float = Field(0.0, ge=0.0)

    # Floor/ceiling effects: shrink treatment benefit at extreme thresholds.
    # floor_effect > 0 → tau_eff at threshold 1 reduced by floor_effect.
    # ceiling_effect > 0 → tau_eff at threshold K-1 reduced by ceiling_effect.
    # Clinically: patients "stuck" at worst/best categories don't respond to treatment
    # the same way as patients in the middle → PO violated.
    floor_effect: float = Field(0.0, ge=0.0)
    ceiling_effect: float = Field(0.0, ge=0.0)

    # ── Site random intercept ───────────────────────────────────────────────
    # site_icc: intraclass correlation on the logistic scale (same formula as registry.py).
    # ICC = sd_site² / (sd_site² + π²/3)  ⟹  sd_site = sqrt(ICC · π²/3 / (1 − ICC))
    n_sites: int = Field(1, ge=1, le=500)
    site_icc: float = Field(0.0, ge=0.0, le=1.0)

    # ── Output ──────────────────────────────────────────────────────────────
    # Column name for the ordinal marker in the generated DataFrame.
    # ConcretePROWinRatioEstimator uses this name via pro=[{"marker": marker_col, ...}].
    marker_col: str = "ordinal_pro"

    seed: int = 42

    # ── Validators ──────────────────────────────────────────────────────────

    @model_validator(mode="after")
    def _check_couplings(self) -> "OrdinalPROConfig":
        if self.tau_category_offsets and len(self.tau_category_offsets) != self.K - 1:
            raise ValueError(
                f"tau_category_offsets must have K-1={self.K - 1} elements "
                f"(got {len(self.tau_category_offsets)}) — set one offset per "
                "threshold c_1..c_{K-1} or leave empty for PO-respecting defaults"
            )
        if self.cutpoints is not None:
            if len(self.cutpoints) != self.K - 1:
                raise ValueError(
                    f"cutpoints must have K-1={self.K - 1} elements "
                    f"(got {len(self.cutpoints)})"
                )
            for i in range(len(self.cutpoints) - 1):
                if self.cutpoints[i] >= self.cutpoints[i + 1]:
                    raise ValueError(
                        f"cutpoints must be strictly increasing "
                        f"(cutpoints[{i}]={self.cutpoints[i]} >= cutpoints[{i+1}]={self.cutpoints[i+1]})"
                    )
        if self.site_icc > 0.0 and self.n_sites < 2:
            raise ValueError(
                f"site_icc={self.site_icc} > 0 requires n_sites >= 2 "
                f"(got n_sites={self.n_sites}) — a single site cannot exhibit clustering"
            )
        if self.site_threshold_sd > 0.0 and self.n_sites < 2:
            raise ValueError(
                f"site_threshold_sd={self.site_threshold_sd} > 0 requires n_sites >= 2 "
                f"(got n_sites={self.n_sites})"
            )
        return self

    def with_overrides(self, **overrides) -> "OrdinalPROConfig":
        """Return a new, fully validated config with the given fields overridden."""
        return OrdinalPROConfig(**{**self.model_dump(), **overrides})


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def _effective_tau(config: OrdinalPROConfig) -> np.ndarray:
    """Return the K-1 effective treatment effects at each threshold.

    tau_eff[j] = tau + tau_category_offsets[j] - floor_adj[j] - ceiling_adj[j]

    When all offsets and adjustments are zero: tau_eff = tau everywhere → PO holds.
    """
    K = config.K
    offsets = np.array(config.tau_category_offsets if config.tau_category_offsets
                       else [0.0] * (K - 1))
    tau_eff = config.tau + offsets
    # Floor/ceiling shrink the treatment benefit at extreme thresholds.
    tau_eff[0] -= config.floor_effect
    tau_eff[-1] -= config.ceiling_effect
    return tau_eff


def generate_data(
    config: OrdinalPROConfig,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Generate one simulated ordinal PRO dataset.

    Parameters
    ----------
    config:
        Ordinal PRO DGP configuration.
    rng:
        Optional numpy Generator.  If None a fresh generator is seeded from
        ``config.seed`` (deterministic for the same seed).

    Returns
    -------
    pd.DataFrame with columns:
        A             — binary treatment (0/1)
        W1, W2, W3, W4 — covariates (same distribution as survival.py)
        site_id       — integer site assignment (0-indexed)
        <marker_col>  — ordinal outcome 1..K (``config.marker_col``, default "ordinal_pro")
    """
    if rng is None:
        rng = np.random.default_rng(config.seed)

    n = config.n
    K = config.K

    # ── Covariates (same distribution as survival.py for composability) ────
    W1 = rng.standard_normal(n)
    W2 = rng.binomial(1, 0.5, n).astype(float)
    W3 = rng.standard_normal(n)
    W4 = rng.binomial(1, 0.3, n).astype(float)

    # ── Treatment (same propensity model as survival.py) ───────────────────
    p = np.clip(config.treatment_prevalence, 1e-6, 1 - 1e-6)
    logit_A = (
        np.log(p / (1 - p))
        + 0.3 * W1
        + 0.2 * W2
        - 0.2 * W3
        + 0.1 * W4
    )
    A = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    # ── Cutpoints ──────────────────────────────────────────────────────────
    if config.cutpoints is not None:
        cuts = np.array(config.cutpoints, dtype=float)
    else:
        cuts = _default_cutpoints(K)

    # ── Site random intercept ──────────────────────────────────────────────
    if config.n_sites >= 2 and config.site_icc > 0.0:
        # Logistic-scale ICC: icc = sd_site² / (sd_site² + π²/3)
        # ⟹ sd_site = sqrt(icc · π²/3 / (1 − icc))  [same as registry.py]
        sd_site = np.sqrt(config.site_icc * (np.pi ** 2 / 3.0) / (1.0 - config.site_icc))
        site_effects = rng.normal(0.0, sd_site, config.n_sites)
        site_id = rng.integers(0, config.n_sites, n).astype(int)
        b_site = site_effects[site_id]
    else:
        site_id = np.zeros(n, dtype=int)
        b_site = np.zeros(n)

    # ── Site-varying threshold perturbations ───────────────────────────────
    # Shape (n_sites, K-1): each site gets an independent perturbation per threshold.
    if config.site_threshold_sd > 0.0 and config.n_sites >= 2:
        site_thresh_perturb = rng.normal(
            0.0, config.site_threshold_sd, (config.n_sites, K - 1)
        )
    else:
        site_thresh_perturb = None

    # ── Prognostic linear predictor ────────────────────────────────────────
    f_W = 0.4 * W1 - 0.3 * W2 + 0.2 * W3 - 0.2 * W4

    # ── Category-specific effective treatment effects ──────────────────────
    tau_eff = _effective_tau(config)   # shape (K-1,)

    # ── Compute cumulative probabilities P(Y <= j) ─────────────────────────
    # P(Y <= j | W, A, site) = logistic(c_j + delta_site_j − f(W) − b_site − tau_eff[j]·A)
    # Shape: (n, K-1)
    cum_prob = np.empty((n, K - 1))
    for j in range(K - 1):
        c_j = cuts[j]
        if site_thresh_perturb is not None:
            c_j = c_j + site_thresh_perturb[site_id, j]
        linear_pred = c_j - f_W - b_site - tau_eff[j] * A
        cum_prob[:, j] = _sigmoid(linear_pred)

    # Enforce monotonicity (PO-violating params can create small inversions).
    cum_prob = np.maximum.accumulate(cum_prob, axis=1)
    cum_prob = np.clip(cum_prob, 0.0, 1.0)

    # ── PMF: P(Y = k) for k = 1..K ────────────────────────────────────────
    pmf = np.empty((n, K))
    pmf[:, 0] = cum_prob[:, 0]
    for k in range(1, K - 1):
        pmf[:, k] = cum_prob[:, k] - cum_prob[:, k - 1]
    pmf[:, K - 1] = 1.0 - cum_prob[:, K - 2]
    pmf = np.maximum(pmf, 0.0)
    row_sums = pmf.sum(axis=1, keepdims=True)
    pmf = pmf / np.where(row_sums > 0, row_sums, 1.0)

    # ── Sample Y from PMF via inverse CDF ──────────────────────────────────
    u = rng.uniform(0.0, 1.0, n)
    cum_pmf = np.cumsum(pmf, axis=1)
    # Y = (number of thresholds j where u > P(Y<=j)) + 1, 1-indexed
    Y = (u[:, None] > cum_pmf).sum(axis=1).astype(int) + 1
    Y = np.clip(Y, 1, K)  # guard against floating-point edge cases

    return pd.DataFrame({
        "A": A,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "W4": W4,
        "site_id": site_id,
        config.marker_col: Y,
    })


# ---------------------------------------------------------------------------
# True estimands
# ---------------------------------------------------------------------------

def compute_true_cumulative_logOR(
    config: OrdinalPROConfig,
    n_ref: int = 100_000,
) -> dict:
    """Estimate the true marginal cumulative log-OR at each threshold.

    Marginalises over the covariate and site distributions via a large reference
    population.  Under proportional odds, all K-1 log-ORs equal tau (up to
    non-collapsibility attenuation).  Under PO violation they diverge.

    Parameters
    ----------
    config : OrdinalPROConfig
    n_ref  : Reference population size (default 100 000 for <1% MC error).

    Returns
    -------
    dict with keys:
        "log_OR"        — list of K-1 marginal cumulative log-ORs (one per threshold)
        "is_PO"         — True if max |log_OR[j] − log_OR[0]| < 0.05 (heuristic)
        "structural_tau_eff" — list of K-1 structural tau_eff values (DGP parameters)
    """
    rng = np.random.default_rng(config.seed ^ 0xCAFEBABE)

    # Use balanced treatment (50/50) to estimate population-level log-ORs,
    # avoiding confounding from the propensity model.
    cfg_balanced = config.with_overrides(n=n_ref, treatment_prevalence=0.5, seed=config.seed ^ 0xCAFEBABE)
    df = generate_data(cfg_balanced, rng=rng)

    K = config.K
    col = config.marker_col

    log_ors = []
    for j in range(1, K):  # threshold j: P(Y <= j)
        p1 = (df.loc[df["A"] == 1, col] <= j).mean()
        p0 = (df.loc[df["A"] == 0, col] <= j).mean()
        # Guard against boundary probs
        p1 = np.clip(p1, 1e-6, 1 - 1e-6)
        p0 = np.clip(p0, 1e-6, 1 - 1e-6)
        log_or = np.log(p1 / (1 - p1)) - np.log(p0 / (1 - p0))
        log_ors.append(float(log_or))

    tau_eff = _effective_tau(config).tolist()
    max_spread = max(abs(log_ors[j] - log_ors[0]) for j in range(len(log_ors)))

    return {
        "log_OR": log_ors,
        "is_PO": bool(max_spread < 0.05),
        "structural_tau_eff": tau_eff,
    }


def compute_true_ordinal_win_ratio(
    config: OrdinalPROConfig,
    n_ref: int = 100_000,
) -> dict:
    """Estimate the true marginal ordinal win ratio via U-statistic on potential outcomes.

    Win ratio = P(Y(1)_i > Y(0)_j) / P(Y(1)_i < Y(0)_j) for independent draws
    i, j from the treated and control potential-outcome distributions.

    Mirrors ``survival.py::compute_true_win_ratio`` for consistent API.

    Parameters
    ----------
    config : OrdinalPROConfig
    n_ref  : Reference population size (default 100 000).

    Returns
    -------
    dict with keys:
        "ATE"         — marginal win ratio (> 1 means treated category is higher)
        "p_win"       — P(Y(1) > Y(0)), marginalised
        "p_loss"      — P(Y(1) < Y(0)), marginalised
        "net_benefit" — p_win − p_loss
        "p_tie"       — P(Y(1) = Y(0)), marginalised
    """
    rng = np.random.default_rng(config.seed ^ 0xDEADBEEF)

    # Generate two large samples: one under A=1, one under A=0.
    # Use equal treatment prevalence to focus on the marginal PO distributions.
    # The win ratio is E_{X~P(X)}[P(Y(1)>Y(0)|X)] for independent X draws —
    # equivalently, draw two independent reference populations and compare.
    cfg_ref = config.with_overrides(n=n_ref, seed=config.seed ^ 0xDEADBEEF)

    # Arm A=1: force all patients treated
    rng1 = np.random.default_rng(config.seed ^ 0xDEADBEEF ^ 1)
    cfg1 = cfg_ref.with_overrides(treatment_prevalence=0.9999)
    df1 = generate_data(cfg1, rng=rng1)
    Y1 = df1[config.marker_col].values[df1["A"].values == 1]

    # Arm A=0: force all patients control
    rng0 = np.random.default_rng(config.seed ^ 0xDEADBEEF ^ 2)
    cfg0 = cfg_ref.with_overrides(treatment_prevalence=0.0001)
    df0 = generate_data(cfg0, rng=rng0)
    Y0 = df0[config.marker_col].values[df0["A"].values == 0]

    # U-statistic via searchsorted: O(n log n), exact for discrete distributions
    # For ordinal Y: Y1 > Y0 means patient i (treated) "wins" over patient j (control).
    Y0_sorted = np.sort(Y0)
    n1, n0 = len(Y1), len(Y0)

    # P(Y1 > Y0): fraction of (i,j) pairs where Y1_i > Y0_j
    # = mean over i of P(Y0 < Y1_i) = mean of searchsorted(Y0_sorted, Y1_i, side="left") / n0
    p_win  = float(np.searchsorted(Y0_sorted, Y1, side="left").mean()) / n0

    # P(Y1 < Y0): fraction of (i,j) pairs where Y1_i < Y0_j
    # = mean over i of P(Y0 > Y1_i) = mean of (n0 - searchsorted(Y0_sorted, Y1_i, side="right")) / n0
    p_loss = float((n0 - np.searchsorted(Y0_sorted, Y1, side="right")).mean()) / n0

    p_tie  = 1.0 - p_win - p_loss
    win_ratio = p_win / p_loss if p_loss > 1e-12 else float("inf")

    return {
        "ATE":         win_ratio,
        "p_win":       p_win,
        "p_loss":      p_loss,
        "net_benefit": p_win - p_loss,
        "p_tie":       p_tie,
    }
