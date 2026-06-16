"""Provenance-linked synthetic augmentation.

Models a real-world failure mode: a rare-region cohort is augmented with
synthetic samples (e.g. twisted-diffusion) generated conditioned on / near
specific real units. Cross-fitting assumes folds are drawn independently; if
a synthetic unit shares structure with its real parent and the two land in
different folds, that independence is violated, the IC-based variance is too
small, and CI coverage drops below nominal. causal_bench's existing DGP is
fully synthetic and iid with no such provenance dependence — this module adds
it as a controllable knob (leakage_strength) so the violation can be measured
rather than assumed.

Each synthetic unit's latents are a convex (Gaussian-copula-style) mix of its
parent's actual draw and a fresh independent draw:

    X_synth = leakage_strength * X_parent + sqrt(1 - leakage_strength**2) * X_fresh

For X ~ N(0, 1) this preserves the marginal N(0, 1) distribution at every
leakage level (so the synthetic population doesn't drift from the DGP's
marginal covariate distribution) while Corr(X_synth, X_parent) = leakage_strength
exactly, and Var(X_synth - X_parent) = 2*(1 - leakage_strength) shrinks to 0 as
leakage_strength -> 1 (the "tight ball around the parent" the spec describes).
Binary covariates use a copy-with-probability-leakage_strength scheme instead,
since the continuous mixing formula doesn't apply.

leakage_strength=0 reduces exactly to fresh, independent draws from the same
DGP (no shared structure); leakage_strength=1 reduces exactly to the
synthetic unit's latents being identical to its parent's.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data

_BINARY_COVS = ("W2", "W4")
_CONTINUOUS_COVS = ("W1", "W3")


def generate_augmented_data(
    cfg: DGPConfig,
    n_real: int,
    n_synth_per_real: int,
    leakage_strength: float,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw a real cohort plus provenance-linked synthetic augmentation.

    Parameters
    ----------
    cfg:
        Base DGP config. Its `n` field is ignored — `n_real` and
        `n_synth_per_real` control sample sizes instead. compute_true_effects
        should still be called on this same `cfg` (unmodified `n` is fine,
        since compute_true_effects uses its own n_ref reference population) —
        augmentation does not change the estimand.
    n_real:
        Number of real (parent) units to draw from generate_data(cfg).
    n_synth_per_real:
        Number of synthetic children generated per real parent. 0 returns
        just the real cohort with provenance/is_synthetic columns added.
    leakage_strength:
        In [0, 1]. 0 = synthetic units are fresh independent DGP draws
        (no shared structure). 1 = synthetic units share the parent's exact
        latent U and covariates W1-W4 (strongest shared structure).
    rng:
        Optional shared generator; defaults to one seeded from cfg.seed.

    Returns
    -------
    pd.DataFrame with all of generate_data's columns plus:
      - provenance_group (int): real units get a unique id (0..n_real-1);
        every synthetic child shares its parent's id. This is the grouping
        key for fold assignment (see causal_bench.crossfit.make_folds).
      - is_synthetic (0/1).
    """
    if not 0.0 <= leakage_strength <= 1.0:
        raise ValueError(f"leakage_strength must be in [0, 1], got {leakage_strength}")
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    real_cfg = cfg.with_overrides(n=n_real)
    df_real, U_real = generate_data(real_cfg, rng=rng, return_latents=True)
    df_real = df_real.copy()
    df_real["provenance_group"] = np.arange(n_real)
    df_real["is_synthetic"] = 0

    if n_synth_per_real == 0:
        return df_real

    n_synth = n_real * n_synth_per_real
    # Repeat each parent's row n_synth_per_real times, contiguous per parent,
    # so rep[j] is the parent index of synthetic unit j.
    rep = np.repeat(np.arange(n_real), n_synth_per_real)

    a = leakage_strength
    b = float(np.sqrt(max(0.0, 1.0 - a ** 2)))

    latent_overrides = {}
    U_fresh = rng.standard_normal(n_synth)
    latent_overrides["U"] = a * U_real[rep] + b * U_fresh
    for col in _CONTINUOUS_COVS:
        parent = df_real[col].values[rep]
        fresh = rng.standard_normal(n_synth)
        latent_overrides[col] = a * parent + b * fresh
    for col in _BINARY_COVS:
        parent = df_real[col].values[rep]
        p_binom = 0.5 if col == "W2" else 0.3
        fresh = rng.binomial(1, p_binom, n_synth).astype(float)
        copy_parent = rng.uniform(0, 1, n_synth) < a
        latent_overrides[col] = np.where(copy_parent, parent, fresh)

    synth_cfg = cfg.with_overrides(n=n_synth)
    df_synth = generate_data(
        synth_cfg, rng=rng,
        U=latent_overrides["U"],
        W1=latent_overrides["W1"], W2=latent_overrides["W2"],
        W3=latent_overrides["W3"], W4=latent_overrides["W4"],
    )
    df_synth = df_synth.copy()
    df_synth["provenance_group"] = rep
    df_synth["is_synthetic"] = 1

    df = pd.concat([df_real, df_synth], ignore_index=True)
    if cfg.strata_cols:
        df.attrs["strata_cols"] = list(cfg.strata_cols)
    return df
