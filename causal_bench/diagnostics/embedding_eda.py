"""Embedding EDA diagnostics — encoder-agnostic, CPU-only.

All functions take np.ndarray embeddings directly. No encoder loading, no MEDS,
no GPU. These are the CPU-side building blocks for the rare-detail localisation
diagnostic described in the GPU build spec, and for calibrating the DGP's φ
parameter against real cohort embeddings.

Functions
---------
phi_proxy          Empirical embedding fidelity: Spearman ρ(sim, -|ΔCATE|).
cluster_condition_numbers  Per-subgroup cond(Σ_k) from GMM covariances.
zca_whiten         ZCA whitening: cheap, invertible, identity covariance.
zca_unwhiten       Invert ZCA whitening.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from causal_bench.estimators.subgroup import SubgroupModel


def phi_proxy(
    main_emb: np.ndarray,
    cate_hat: np.ndarray,
    max_pairs: int = 10_000,
    random_state: int = 0,
) -> float:
    """Empirical estimate of embedding fidelity φ.

    Computes the Spearman ρ between pairwise cosine similarity and pairwise
    negative |CATE_i − CATE_j|. When the encoder preserves CATE-relevant
    structure, similar embeddings predict similar effect sizes → ρ is positive
    and approaches 1.

    This is the empirical counterpart of the DGP's `embedding_fidelity`
    parameter. Use it to anchor the exp19 φ-sweep to the range achievable by
    a real encoder on a specific cohort.

    Parameters
    ----------
    main_emb : (n, d) L2-normalised embedding matrix.
    cate_hat : (n,) per-patient CATE estimates.
    max_pairs : maximum sampled pairs (full pairwise is O(n²); subsampled for n>~200).
    random_state : for pair subsampling reproducibility.

    Returns
    -------
    phi_hat : float in [-1, 1]. Positive → informative embeddings. Near zero →
        encoder does not carry CATE-relevant signal.
    """
    from scipy.stats import spearmanr

    n = len(main_emb)
    rng = np.random.default_rng(random_state)

    n_pairs_total = n * (n - 1) // 2
    if n_pairs_total <= max_pairs:
        # All pairs
        i_idx, j_idx = np.triu_indices(n, k=1)
    else:
        # Random subsample without replacement
        all_i, all_j = np.triu_indices(n, k=1)
        chosen = rng.choice(len(all_i), size=max_pairs, replace=False)
        i_idx, j_idx = all_i[chosen], all_j[chosen]

    # Cosine similarity: embeddings are L2-normalised, so dot product = cosine
    sim = np.einsum("nd,nd->n", main_emb[i_idx], main_emb[j_idx])
    sim = np.clip(sim, -1.0, 1.0)

    delta_cate = np.abs(cate_hat[i_idx] - cate_hat[j_idx])

    # ρ(sim, -|ΔCATE|): high similarity → small |ΔCATE| → negative raw ρ;
    # negate so phi_hat is positive when encoder is informative.
    rho, _ = spearmanr(sim, -delta_cate)
    return float(rho)


def cluster_condition_numbers(
    model: SubgroupModel,
) -> Optional[np.ndarray]:
    """Per-subgroup condition number cond(Σ_k) from GMM component covariances.

    High condition number (cond > ~10) indicates an elongated cluster whose
    principal axes differ greatly — K-means (which assumes Σ = σ²I) would cut
    it arbitrarily. Use this as an EDA flag to decide between K-means and GMM
    clustering in discover_subgroups().

    Parameters
    ----------
    model : SubgroupModel from discover_subgroups(). Must have been fitted with
        clustering="gmm"; K-means models have component_covariances=None.

    Returns
    -------
    cond_numbers : (k,) array of per-subgroup condition numbers, or None if the
        model was fitted with K-means clustering.
    """
    if model.component_covariances is None:
        return None
    return np.array([
        float(np.linalg.cond(model.component_covariances[k]))
        for k in range(model.n_subgroups)
    ])


def zca_whiten(
    Z: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """ZCA whitening: transform Z to identity covariance.

    ZCA is cheap (one eigendecomposition), invertible, and minimal-rotation —
    the whitened coordinates stay as close as possible to the original space.
    Use it before training a score-based diffusion model on embeddings (as
    in Test B/B' of the rare-detail localisation diagnostic).

    Parameters
    ----------
    Z   : (n, d) embedding matrix.
    eps : regularisation added to eigenvalues to prevent divide-by-zero on
          near-rank-deficient covariance (e.g. when d > n).

    Returns
    -------
    Z_white : (n, d) whitened embeddings with identity covariance.
    W       : (d, d) whitening matrix (Z_white = (Z - mu) @ W).
    mu      : (d,) mean vector.
    """
    mu = Z.mean(axis=0)
    Zc = Z - mu
    cov = (Zc.T @ Zc) / max(len(Z) - 1, 1)

    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 0.0)  # numerical non-negativity
    scale = 1.0 / np.sqrt(eigvals + eps)
    W = eigvecs @ np.diag(scale) @ eigvecs.T

    Z_white = Zc @ W
    return Z_white, W, mu


def zca_unwhiten(
    Z_white: np.ndarray,
    W: np.ndarray,
    mu: np.ndarray,
) -> np.ndarray:
    """Invert ZCA whitening: recover original embedding space.

    Parameters
    ----------
    Z_white : (n, d) ZCA-whitened embeddings.
    W       : (d, d) whitening matrix returned by zca_whiten().
    mu      : (d,) mean vector returned by zca_whiten().

    Returns
    -------
    Z : (n, d) embeddings in the original pre-whitening space.
    """
    W_inv = np.linalg.inv(W)
    return Z_white @ W_inv + mu
