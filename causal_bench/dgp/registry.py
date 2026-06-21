"""Three-registry DGP for Exp 19: hierarchical borrowing operating characteristics.

Generates synthetic data for three CED registries:
  main     — large cohort (n≈700), the information donor
  teer     — Failed-TEER rare cohort (n≈80), primary borrowing target
  mac      — MAC rare cohort (n≈60), secondary borrowing target

The decisive lever is φ (embedding_fidelity) ∈ [0, 1]:
  φ = 1  borrowing weights are perfectly correlated with true effect similarity
       (patient-level borrowing ideal — proximity ≡ effect similarity)
  φ = 0  borrowing weights are uncorrelated with true effect similarity
       (patient-level borrowing is pure noise — only population-level is safe)

φ is synthetic here (no real embeddings). It controls the Spearman correlation
between the all-pairs embedding-similarity kernel and the all-pairs true-CATE
difference, so the OC study can sweep φ and quantify the precision/bias trade
before committing to a real embedding space.

Prior-data conflict is controlled by conflict_strength ∈ [0, 1]:
  0  rare cohort has the same true ATE as the main cohort (no conflict)
  1  rare cohort has a true ATE opposite in sign to the main cohort (full conflict)

The robust MAP prior (Schmidli et al. 2014) should auto-discount under conflict;
the OC study records ESS collapse as evidence the prior is behaving.
"""
from __future__ import annotations

from typing import Literal, Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, model_validator


class RegistryConfig(BaseModel):
    model_config = {"frozen": True, "extra": "forbid"}

    # Registry sizes
    n_main: int = Field(700, ge=10, le=50_000)
    n_teer: int = Field(80,  ge=5,  le=10_000)
    n_mac:  int = Field(60,  ge=5,  le=10_000)

    # True treatment effects (risk differences within horizon)
    # main cohort: treated arm has lower event rate (negative ATE = benefit)
    true_ate_main: float = -0.12
    # rare cohorts: controlled by conflict_strength (see below)
    true_ate_teer: Optional[float] = None   # None → derived from conflict
    true_ate_mac:  Optional[float] = None   # None → derived from conflict

    # Prior-data conflict: how different are rare-cohort ATEs from main
    # 0 = same effect; 1 = equal-and-opposite effect
    conflict_strength: float = Field(0.0, ge=0.0, le=1.0)

    # Baseline event rates (control arm, within horizon)
    baseline_rate_main: float = Field(0.35, gt=0.0, lt=1.0)
    baseline_rate_teer: float = Field(0.50, gt=0.0, lt=1.0)   # sicker
    baseline_rate_mac:  float = Field(0.45, gt=0.0, lt=1.0)

    # Treatment prevalence (proportion treated in each registry)
    treat_prev_main: float = Field(0.50, ge=0.0, le=1.0)
    treat_prev_teer: float = Field(0.45, ge=0.0, le=1.0)
    treat_prev_mac:  float = Field(0.40, ge=0.0, le=1.0)

    # Effect heterogeneity within registry (SD of per-patient CATE around registry ATE)
    cate_sd_main: float = Field(0.06, ge=0.0)
    cate_sd_teer: float = Field(0.08, ge=0.0)
    cate_sd_mac:  float = Field(0.08, ge=0.0)

    # Embedding fidelity φ ∈ [0, 1]
    # Controls Spearman ρ between synthetic embedding similarity and true CATE similarity.
    # At φ=1: pairwise embedding distance perfectly ranks pairwise |CATE_i − CATE_j|.
    # At φ=0: embedding similarity is independent of CATE similarity (pure noise).
    embedding_fidelity: float = Field(1.0, ge=0.0, le=1.0)

    # Number of synthetic embedding dimensions (for the all-pairs kernel)
    n_embedding_dims: int = Field(16, ge=2, le=256)

    # Borrowing controls for hierarchical model
    # tau_prior_sd: prior SD on the between-registry heterogeneity parameter τ
    tau_prior_sd: float = Field(0.10, gt=0.0)
    # robust_weight: mixture weight on the vague component (1−w on MAP)
    robust_weight: float = Field(0.10, ge=0.0, le=1.0)
    # vague_sd: SD of the vague normal component in the robust MAP prior
    vague_sd: float = Field(0.50, gt=0.0)

    seed: int = 42

    @model_validator(mode="after")
    def _derive_rare_ates(self) -> "RegistryConfig":
        # If true_ate_teer / true_ate_mac not set, derive from conflict_strength
        # conflict_strength=0 → same as main; 1 → opposite sign, same magnitude
        if self.true_ate_teer is None:
            object.__setattr__(
                self, "true_ate_teer",
                self.true_ate_main * (1.0 - 2.0 * self.conflict_strength),
            )
        if self.true_ate_mac is None:
            object.__setattr__(
                self, "true_ate_mac",
                self.true_ate_main * (1.0 - 2.0 * self.conflict_strength),
            )
        return self


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def _generate_registry_arm(
    n: int,
    true_ate: float,
    baseline_rate: float,
    treat_prev: float,
    cate_sd: float,
    registry: str,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Generate one registry's patient-level data.

    Binary outcome Y ~ Bernoulli(p(A, W)):
        p(A=0, W) = expit(logit(baseline_rate) + 0.3*W1 − 0.2*W2)
        p(A=1, W) = expit(logit(baseline_rate) + 0.3*W1 − 0.2*W2 + cate_i)
    where cate_i = true_ate + N(0, cate_sd²) is the per-patient effect.
    """
    W1 = rng.standard_normal(n)
    W2 = rng.binomial(1, 0.5, n).astype(float)
    W3 = rng.standard_normal(n)

    # Treatment assignment (mild confounding via W1, W2)
    p = np.clip(treat_prev, 1e-6, 1 - 1e-6)
    logit_A = np.log(p / (1 - p)) + 0.3 * W1 - 0.2 * W2
    A = rng.binomial(1, _sigmoid(logit_A)).astype(float)

    # Per-patient CATE (heterogeneous treatment effect)
    cate = true_ate + rng.normal(0, cate_sd, n)

    # Binary outcome
    log_odds_base = np.log(baseline_rate / (1 - baseline_rate)) + 0.3 * W1 - 0.2 * W2
    p0 = _sigmoid(log_odds_base)             # P(Y=1 | A=0, W)
    p1 = _sigmoid(log_odds_base + cate)      # P(Y=1 | A=1, W)
    p_obs = np.where(A == 1, p1, p0)
    Y = rng.binomial(1, p_obs).astype(float)

    return pd.DataFrame({
        "Y": Y,
        "A": A,
        "W1": W1,
        "W2": W2,
        "W3": W3,
        "cate": cate,       # true per-patient CATE (latent — for OC evaluation only)
        "registry": registry,
        "p0": p0,
        "p1": p1,
    })


def _make_embeddings(
    df: pd.DataFrame,
    phi: float,
    n_dims: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Synthetic embedding matrix (n × n_dims).

    Rows are patient embeddings. Pairwise cosine similarity has Spearman ρ ≈ φ
    with pairwise |CATE_i − CATE_j| similarity (inverse of absolute difference).

    Construction:
      signal = cate (the true per-patient effect, unit-normalised)
      noise  = standard normal (uninformative)
      embedding = φ * signal_broadcast + √(1−φ²) * noise
    """
    n = len(df)
    cate_norm = (df["cate"].values - df["cate"].mean()) / (df["cate"].std() + 1e-8)
    # Broadcast cate signal into n_dims (each dim is cate + independent noise)
    signal = np.outer(cate_norm, np.ones(n_dims))
    noise = rng.standard_normal((n, n_dims))
    emb = phi * signal + np.sqrt(max(1 - phi ** 2, 0.0)) * noise
    # L2-normalise rows
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    return emb / np.maximum(norms, 1e-8)


def generate_registry_data(
    config: RegistryConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    """Generate three registry datasets and their synthetic embeddings.

    Returns
    -------
    main_df, teer_df, mac_df : patient-level DataFrames
    embeddings : dict with keys "main", "teer", "mac" → (n × n_dims) arrays
    """
    rng = np.random.default_rng(config.seed)

    main_df = _generate_registry_arm(
        config.n_main, config.true_ate_main, config.baseline_rate_main,
        config.treat_prev_main, config.cate_sd_main, "main", rng,
    )
    teer_df = _generate_registry_arm(
        config.n_teer, config.true_ate_teer, config.baseline_rate_teer,
        config.treat_prev_teer, config.cate_sd_teer, "teer", rng,
    )
    mac_df = _generate_registry_arm(
        config.n_mac, config.true_ate_mac, config.baseline_rate_mac,
        config.treat_prev_mac, config.cate_sd_mac, "mac", rng,
    )

    phi = config.embedding_fidelity
    emb_main = _make_embeddings(main_df, phi, config.n_embedding_dims, rng)
    emb_teer = _make_embeddings(teer_df, phi, config.n_embedding_dims, rng)
    emb_mac  = _make_embeddings(mac_df,  phi, config.n_embedding_dims, rng)

    embeddings = {"main": emb_main, "teer": emb_teer, "mac": emb_mac}
    return main_df, teer_df, mac_df, embeddings
