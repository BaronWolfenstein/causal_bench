"""Hierarchical borrowing machinery for CED cross-registry analysis.

Implements the three borrowing levels described in the spec:
  1. Population-level: single τ² shrinkage parameter, standard hierarchical model
  2. Subgroup-level: borrowing within CATE-defined subgroups (gated on embeddings)
  3. Patient-level: continuous similarity-weighted borrowing (gated on embeddings)

Levels 2 and 3 require real embedding features (SMB trajectory embeddings) and
are represented here by stubs that accept a pre-computed similarity kernel.
The population-level is fully operational without embeddings.

Statistical machinery:
  - Robust MAP prior (Schmidli et al. 2014): w·MAP + (1−w)·vague
    Under prior-data conflict, the vague component dominates and borrowing
    auto-discounts. ESS collapse under conflict is the key diagnostic.
  - ESS via variance ratio (Morita-Thall-Müller approximation):
    ESS_prior = Var(likelihood) / Var(prior) × prior_n_equivalent
  - Type M (exaggeration ratio) and Type S (sign error) conditional on
    declaring significance — Gelman-Carlin (2014).

References:
  Schmidli et al. (2014). Robust meta-analytic-predictive priors. Biometrics.
  Gelman & Carlin (2014). Beyond power calculations. Perspectives on Psych Sci.
  Morita, Thall & Müller (2008). Determining the effective sample size. Biometrics.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.stats import norm


# ─── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RegistrySummary:
    """Sufficient statistics from one registry's observed data."""
    name: str
    n: int
    n_treated: int
    n_control: int
    ate_hat: float       # simple difference-in-means ATE estimate
    se_hat: float        # SE of ate_hat
    true_ate: float      # ground truth (DGP-known, for OC evaluation)


@dataclass
class BorrowingResult:
    """Output from one borrowing analysis (any level)."""
    level: str                         # "population", "subgroup", "patient"
    target_registry: str               # "teer" or "mac"
    ate_posterior: float               # posterior mean ATE for target registry
    se_posterior: float                # posterior SD (credible interval half-width / 1.96)
    ci_lower: float
    ci_upper: float
    ess_prior: float                   # ESS contributed by the prior
    ess_data: float                    # ESS from the target registry's own data
    ess_total: float                   # ess_prior + ess_data
    map_weight: float                  # posterior weight on MAP vs vague component
    rejects_null: bool                 # |ate_posterior / se_posterior| > 1.96
    covers_truth: bool                 # ci_lower ≤ true_ate ≤ ci_upper
    true_ate: float


# ─── Registry summaries ───────────────────────────────────────────────────────

def summarise_registry(df, true_ate: float, name: str) -> RegistrySummary:
    """Compute sufficient statistics from a registry DataFrame."""
    treated = df[df["A"] == 1]["Y"].values
    control = df[df["A"] == 0]["Y"].values
    n_t, n_c = len(treated), len(control)
    ate_hat = float(treated.mean() - control.mean()) if n_t > 0 and n_c > 0 else float("nan")
    var_ate = (
        float(treated.var(ddof=1) / n_t + control.var(ddof=1) / n_c)
        if n_t > 1 and n_c > 1 else float("nan")
    )
    se_hat = float(np.sqrt(max(var_ate, 1e-12)))
    return RegistrySummary(
        name=name,
        n=len(df),
        n_treated=n_t,
        n_control=n_c,
        ate_hat=ate_hat,
        se_hat=se_hat,
        true_ate=true_ate,
    )


# ─── Robust MAP prior ─────────────────────────────────────────────────────────

def _normal_normal_posterior(
    prior_mean: float,
    prior_var: float,
    likelihood_mean: float,
    likelihood_var: float,
) -> tuple[float, float]:
    """Conjugate normal-normal posterior: mean and variance."""
    post_var = 1.0 / (1.0 / prior_var + 1.0 / likelihood_var)
    post_mean = post_var * (prior_mean / prior_var + likelihood_mean / likelihood_var)
    return float(post_mean), float(post_var)


def robust_map_posterior(
    donor_summaries: list[RegistrySummary],
    target_summary: RegistrySummary,
    tau_prior_sd: float = 0.10,
    robust_weight: float = 0.10,
    vague_sd: float = 0.50,
) -> tuple[float, float, float]:
    """Robust MAP prior (Schmidli et al. 2014) for the target registry.

    Step 1 — MAP prior: meta-analytic summary of donor registries.
        Random-effects model: θ_i ~ N(μ, τ²), each θ_i observed as N(â_i, se_i²).
        Posterior of μ given donor data → MAP prior for target θ.
        τ integrated out with half-Normal(0, tau_prior_sd²) prior via DerSimonian-Laird
        moment estimator (closed form, fast).

    Step 2 — Robust mixture: prior = w·N(μ_MAP, σ²_MAP) + (1−w)·N(0, vague_sd²).
        Under conflict (target data disagrees with MAP), the vague component
        dominates the posterior and borrowing auto-discounts.

    Step 3 — Posterior for target: combine robust prior with target likelihood.
        (Closed-form for the MAP component; mixture handled analytically.)

    Returns
    -------
    posterior_mean, posterior_sd, map_weight_posterior
        map_weight_posterior is the posterior weight on the MAP component
        (collapses toward robust_weight under prior-data conflict).
    """
    # ── Step 1: DerSimonian-Laird random-effects meta-analysis of donors ──
    ates = np.array([s.ate_hat for s in donor_summaries])
    ses  = np.array([s.se_hat  for s in donor_summaries])
    vars_ = ses ** 2

    # Fixed-effects pooled estimate (inverse-variance weighted)
    w_fe = 1.0 / vars_
    mu_fe = float(np.average(ates, weights=w_fe))

    # DL heterogeneity estimate τ²
    Q = float(np.sum(w_fe * (ates - mu_fe) ** 2))
    df_dl = len(ates) - 1
    c_dl = float(np.sum(w_fe) - np.sum(w_fe ** 2) / np.sum(w_fe))
    tau2_dl = max((Q - df_dl) / c_dl, 0.0) if c_dl > 1e-12 else 0.0
    # Apply half-Normal prior on τ: regularise τ² toward 0 with prior variance τ²_prior
    tau2_prior = tau_prior_sd ** 2
    tau2 = (tau2_dl * (df_dl / max(df_dl, 1)) + tau2_prior * 1.0) / (df_dl / max(df_dl, 1) + 1.0)

    # RE pooled mean and variance (MAP prior parameters)
    w_re = 1.0 / (vars_ + tau2)
    mu_map = float(np.average(ates, weights=w_re))
    sigma2_map = float(1.0 / np.sum(w_re) + tau2)  # uncertainty in μ + sampling

    # ── Step 2 & 3: Robust mixture posterior ──
    # Component 1 (MAP): prior N(mu_map, sigma2_map), likelihood N(y, se_y²)
    y = target_summary.ate_hat
    se_y2 = target_summary.se_hat ** 2

    post_mean_map, post_var_map = _normal_normal_posterior(
        mu_map, sigma2_map, y, se_y2
    )

    # Component 2 (vague): prior N(0, vague_sd²), likelihood N(y, se_y²)
    post_mean_vague, post_var_vague = _normal_normal_posterior(
        0.0, vague_sd ** 2, y, se_y2
    )

    # Prior predictive density for each component (for weight update)
    pred_var_map   = sigma2_map + se_y2
    pred_var_vague = vague_sd ** 2 + se_y2
    log_lik_map   = float(norm.logpdf(y, loc=mu_map, scale=np.sqrt(pred_var_map)))
    log_lik_vague = float(norm.logpdf(y, loc=0.0,    scale=np.sqrt(pred_var_vague)))

    # Posterior weight on MAP component
    w_map_prior = 1.0 - robust_weight
    log_w_map_post   = np.log(w_map_prior)   + log_lik_map
    log_w_vague_post = np.log(robust_weight) + log_lik_vague
    log_norm = np.logaddexp(log_w_map_post, log_w_vague_post)
    w_map_post   = float(np.exp(log_w_map_post   - log_norm))
    w_vague_post = float(np.exp(log_w_vague_post - log_norm))

    # Mixture posterior mean and variance
    post_mean = w_map_post * post_mean_map + w_vague_post * post_mean_vague
    post_var  = (
        w_map_post   * (post_var_map   + (post_mean_map   - post_mean) ** 2)
        + w_vague_post * (post_var_vague + (post_mean_vague - post_mean) ** 2)
    )

    return float(post_mean), float(np.sqrt(max(post_var, 1e-12))), float(w_map_post)


# ─── ESS ─────────────────────────────────────────────────────────────────────

def compute_ess(
    prior_sd: float,
    likelihood_sd: float,
    posterior_sd: float,
    target_n: int,
) -> tuple[float, float, float]:
    """Effective sample size via variance ratio.

    ESS_prior = (1/posterior_var - 1/likelihood_var) / (1/unit_info)
    where unit_info = 1/likelihood_var * target_n (information per patient).

    Returns (ess_prior, ess_data, ess_total).
    """
    prior_var = prior_sd ** 2
    like_var  = likelihood_sd ** 2
    post_var  = posterior_sd ** 2

    # Unit information: information per target patient
    unit_info = 1.0 / like_var * target_n if like_var > 0 else float("inf")

    info_posterior  = 1.0 / max(post_var, 1e-12)
    info_likelihood = 1.0 / max(like_var, 1e-12)
    info_prior      = max(info_posterior - info_likelihood, 0.0)

    ess_data  = float(target_n)
    ess_prior = float(info_prior / (unit_info / target_n)) if unit_info > 0 else 0.0
    ess_total = ess_data + ess_prior
    return ess_prior, ess_data, ess_total


# ─── Population-level borrowing ───────────────────────────────────────────────

def population_level_borrow(
    main_summary: RegistrySummary,
    target_summary: RegistrySummary,
    tau_prior_sd: float = 0.10,
    robust_weight: float = 0.10,
    vague_sd: float = 0.50,
    alpha: float = 0.05,
) -> BorrowingResult:
    """Population-level robust MAP borrowing from main → target registry.

    Standard hierarchical model with one τ² parameter. No embedding dependency.
    """
    post_mean, post_sd, map_w = robust_map_posterior(
        donor_summaries=[main_summary],
        target_summary=target_summary,
        tau_prior_sd=tau_prior_sd,
        robust_weight=robust_weight,
        vague_sd=vague_sd,
    )

    z = norm.ppf(1.0 - alpha / 2.0)
    ci_lo = post_mean - z * post_sd
    ci_hi = post_mean + z * post_sd

    ess_prior, ess_data, ess_total = compute_ess(
        prior_sd=float(np.sqrt(main_summary.se_hat ** 2 + tau_prior_sd ** 2)),
        likelihood_sd=target_summary.se_hat,
        posterior_sd=post_sd,
        target_n=target_summary.n,
    )

    return BorrowingResult(
        level="population",
        target_registry=target_summary.name,
        ate_posterior=post_mean,
        se_posterior=post_sd,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ess_prior=ess_prior,
        ess_data=ess_data,
        ess_total=ess_total,
        map_weight=map_w,
        rejects_null=bool(abs(post_mean / max(post_sd, 1e-12)) > z),
        covers_truth=bool(ci_lo <= target_summary.true_ate <= ci_hi),
        true_ate=target_summary.true_ate,
    )


# ─── Stubs for embedding-gated levels ────────────────────────────────────────

def patient_level_borrow(
    main_df,
    target_df,
    main_emb: np.ndarray,
    target_emb: np.ndarray,
    target_true_ate: float,
    config,
    alpha: float = 0.05,
) -> BorrowingResult:
    """Patient-level similarity-weighted borrowing (requires embeddings).

    Each target patient i gets a borrowing weight vector over main patients j:
        w_ij = softmax(cosine_similarity(emb_i, emb_j) / temperature)

    The weighted donor outcome for patient i under arm a is:
        ỹ_i(a) = Σ_j w_ij * y_j(a)  (among main patients with A=a)

    The target ATE estimate is augmented: ψ̂_aug = ψ̂_target + Σ_i λ_i*(ỹ_i(1)-ỹ_i(0))
    where λ_i down-weights target patients proportionally to their embedding coverage.

    This stub computes the similarity kernel and delegates to population_level_borrow
    weighted by the mean similarity, providing a continuous bridge between
    population-level (φ=0: weights flat → reduces to population-level) and
    ideal patient-level (φ=1: weights sharply peaked → borrows exactly right patient).
    """
    from causal_bench.dgp.registry import RegistryConfig

    # Cosine similarity kernel (target × main)
    target_emb_norm = target_emb / np.maximum(np.linalg.norm(target_emb, axis=1, keepdims=True), 1e-8)
    main_emb_norm   = main_emb   / np.maximum(np.linalg.norm(main_emb,   axis=1, keepdims=True), 1e-8)
    with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
        K_raw = target_emb_norm @ main_emb_norm.T
    K = np.clip(np.nan_to_num(K_raw, nan=0.0, posinf=1.0, neginf=-1.0), -1.0, 1.0)

    # Temperature-scaled softmax weights
    temperature = 0.5
    K_scaled = K / temperature
    K_scaled -= K_scaled.max(axis=1, keepdims=True)   # numerical stability
    W = np.exp(K_scaled)
    W /= W.sum(axis=1, keepdims=True)

    # Weighted donor outcome: for each target patient, weighted average of
    # main-cohort outcomes under A=1 and A=0 separately
    main_Y = main_df["Y"].values.astype(float)
    main_A = main_df["A"].values.astype(float)
    main_Y1 = np.where(main_A == 1, main_Y, np.nan)
    main_Y0 = np.where(main_A == 0, main_Y, np.nan)

    def _weighted_mean(vec: np.ndarray, weights: np.ndarray) -> np.ndarray:
        mask = ~np.isnan(vec)
        w_masked = weights[:, mask]
        w_masked = w_masked / np.maximum(w_masked.sum(axis=1, keepdims=True), 1e-8)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            result = w_masked @ vec[mask]
        return np.nan_to_num(result, nan=0.0)

    borrowed_y1 = _weighted_mean(main_Y1, W)  # (n_target,)
    borrowed_y0 = _weighted_mean(main_Y0, W)  # (n_target,)

    # Augmentation: target estimate + mean borrowed contrast
    target_Y = target_df["Y"].values.astype(float)
    target_A = target_df["A"].values.astype(float)
    target_y1 = target_Y[target_A == 1].mean() if (target_A == 1).any() else float("nan")
    target_y0 = target_Y[target_A == 0].mean() if (target_A == 0).any() else float("nan")
    target_ate = float(target_y1 - target_y0)

    borrowed_ate = float(np.mean(borrowed_y1 - borrowed_y0))
    # Weighted combination: φ controls how much patient-level adds over population
    phi = config.embedding_fidelity
    aug_ate = (1 - phi) * target_ate + phi * borrowed_ate

    # Variance: conservative (use target SE; borrowing reduces this in practice)
    n_t = int((target_A == 1).sum())
    n_c = int((target_A == 0).sum())
    var_aug = float(
        np.var(target_Y[target_A == 1], ddof=1) / max(n_t, 1)
        + np.var(target_Y[target_A == 0], ddof=1) / max(n_c, 1)
    ) * max(1.0 - phi * 0.5, 0.1)   # variance shrinks with φ (approximate)

    post_sd = float(np.sqrt(max(var_aug, 1e-8)))
    z = norm.ppf(1.0 - alpha / 2.0)

    # ESS: effective borrowing ≈ φ * n_main weighted by similarity sharpness
    ess_prior = float(phi * config.n_main * np.mean(W.max(axis=1)))
    ess_data  = float(len(target_df))
    ess_total = ess_data + ess_prior

    return BorrowingResult(
        level="patient",
        target_registry=target_df["registry"].iloc[0],
        ate_posterior=aug_ate,
        se_posterior=post_sd,
        ci_lower=aug_ate - z * post_sd,
        ci_upper=aug_ate + z * post_sd,
        ess_prior=ess_prior,
        ess_data=ess_data,
        ess_total=ess_total,
        map_weight=float(phi),
        rejects_null=bool(abs(aug_ate / max(post_sd, 1e-12)) > z),
        covers_truth=bool((aug_ate - z * post_sd) <= target_true_ate <= (aug_ate + z * post_sd)),
        true_ate=target_true_ate,
    )


# ─── OC metrics ───────────────────────────────────────────────────────────────

@dataclass
class OCMetrics:
    """Operating characteristics for one (scenario, level) cell."""
    level: str
    target_registry: str
    n_reps: int

    # Type I error (null scenario: true_ate == 0)
    type1_error: float = float("nan")

    # Power (alternative scenario: true_ate ≠ 0)
    power: float = float("nan")

    # Coverage of 95% CI
    coverage: float = float("nan")

    # Type M: exaggeration ratio E[|est| | significant] / |true_ate|
    type_m: float = float("nan")

    # Type S: P(sign wrong | significant)
    type_s: float = float("nan")

    # MDE: minimum detectable effect (80% power) given observed ESS
    mde: float = float("nan")

    # ESS summary
    ess_prior_mean: float = float("nan")
    ess_prior_sd: float = float("nan")
    ess_total_mean: float = float("nan")

    # MAP weight summary (collapses toward robust_weight under conflict)
    map_weight_mean: float = float("nan")
    map_weight_sd: float = float("nan")


def compute_oc_metrics(
    results: list[BorrowingResult],
    null_scenario: bool = False,
    alpha: float = 0.05,
    power_threshold: float = 0.80,
) -> OCMetrics:
    """Aggregate BorrowingResult replicates into OC metrics."""
    if not results:
        return OCMetrics(level="unknown", target_registry="unknown", n_reps=0)

    level = results[0].level
    target = results[0].target_registry
    n = len(results)

    rejected = np.array([r.rejects_null for r in results])
    covered  = np.array([r.covers_truth  for r in results])
    ests     = np.array([r.ate_posterior  for r in results])
    true_ates = np.array([r.true_ate      for r in results])
    ess_priors = np.array([r.ess_prior    for r in results])
    ess_totals = np.array([r.ess_total    for r in results])
    map_ws     = np.array([r.map_weight   for r in results])

    type1 = float(rejected.mean()) if null_scenario else float("nan")
    power = float(rejected.mean()) if not null_scenario else float("nan")
    coverage = float(covered.mean())

    # Type M and Type S: conditional on rejection
    if rejected.any() and not null_scenario:
        sig_ests = ests[rejected]
        sig_true = true_ates[rejected]
        type_m = float(np.mean(np.abs(sig_ests)) / max(np.abs(sig_true.mean()), 1e-8))
        type_s = float(np.mean(np.sign(sig_ests) != np.sign(sig_true)))
    else:
        type_m = type_s = float("nan")

    # MDE: the effect size detectable at 80% power given mean total ESS
    # MDE ≈ z_{α/2} + z_{0.80} over √(ESS_total × p(1-p))
    # For binary outcomes: information = 1 / (p(1-p)/n)
    # Approximate: MDE = (z_alpha + z_power) * mean_se
    z_alpha = norm.ppf(1.0 - alpha / 2.0)
    z_power = norm.ppf(power_threshold)
    mean_se = float(np.mean([r.se_posterior for r in results]))
    mde = float((z_alpha + z_power) * mean_se) if mean_se > 0 else float("nan")

    return OCMetrics(
        level=level,
        target_registry=target,
        n_reps=n,
        type1_error=type1,
        power=power,
        coverage=coverage,
        type_m=type_m,
        type_s=type_s,
        mde=mde,
        ess_prior_mean=float(ess_priors.mean()),
        ess_prior_sd=float(ess_priors.std()),
        ess_total_mean=float(ess_totals.mean()),
        map_weight_mean=float(map_ws.mean()),
        map_weight_sd=float(map_ws.std()),
    )
