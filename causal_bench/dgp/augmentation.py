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
from typing import Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pydantic import BaseModel, Field, model_validator

from causal_bench.dgp.config import DGPConfig
from causal_bench.dgp.survival import generate_data

_BINARY_COVS = ("W2", "W4")
_CONTINUOUS_COVS = ("W1", "W3")


class AugmentationConfig(BaseModel):
    """Parameters governing provenance-linked synthetic augmentation and the
    cross-fitting strategy that should accompany it.

    n_real and n_synth_per_real together determine total augmented cohort size
    (n_real * (1 + n_synth_per_real)); leakage_strength controls how tightly
    synthetic children are tied to their real parent; fold_mode determines
    whether cross-fitting respects provenance groups (use 'group' when
    leakage_strength > 0 to avoid the independence violation that motivates
    this whole module).
    """
    model_config = {"frozen": True, "extra": "forbid"}

    n_real: int = Field(..., ge=1)
    n_synth_per_real: int = Field(..., ge=0)
    leakage_strength: float = Field(..., ge=0.0, le=1.0)
    fold_mode: Literal["iid", "group"] = "group"

    @model_validator(mode="after")
    def _warn_iid_with_leakage(self) -> "AugmentationConfig":
        if self.leakage_strength > 0 and self.fold_mode == "iid" and self.n_synth_per_real > 0:
            import warnings
            warnings.warn(
                f"AugmentationConfig: fold_mode='iid' with leakage_strength="
                f"{self.leakage_strength} > 0 — synthetic children share latent "
                "structure with their real parent, so iid cross-fitting violates "
                "the fold-independence assumption and will understate the EIC-based "
                "SE. Consider fold_mode='group' to keep each provenance_group intact "
                "within a single fold.",
                stacklevel=2,
            )
        return self


# Cross-row provenance integrity check: every synthetic unit's provenance_group
# must refer to a real unit's group, not to a synthetic one or a phantom id.
# Validates as a whole-dataframe check after augmented output is assembled.
_PROVENANCE_SCHEMA = pa.DataFrameSchema(
    checks=pa.Check(
        lambda df: (
            set(df.loc[df["is_synthetic"] == 1, "provenance_group"])
            <= set(df.loc[df["is_synthetic"] == 0, "provenance_group"])
        ),
        error="Synthetic provenance_group values must be a strict subset of real rows' groups",
    ),
    strict=False,
)


def generate_augmented_data(
    cfg: DGPConfig,
    aug_config: AugmentationConfig,
    rng: np.random.Generator | None = None,
) -> pd.DataFrame:
    """Draw a real cohort plus provenance-linked synthetic augmentation.

    Parameters
    ----------
    cfg:
        Base DGP config. Its `n` field is ignored — aug_config.n_real and
        aug_config.n_synth_per_real control sample sizes. compute_true_effects
        should still be called on this same `cfg` (unmodified `n` is fine) —
        augmentation does not change the estimand.
    aug_config:
        AugmentationConfig controlling n_real, n_synth_per_real,
        leakage_strength, and fold_mode.  fold_mode is not used inside this
        function (it governs cross-fitting downstream) but lives here so
        callers that pass aug_config to make_folds / TMLEIPCWEstimator share
        a single source of truth for both the data-generation and CV settings.
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
    n_real = aug_config.n_real
    n_synth_per_real = aug_config.n_synth_per_real
    leakage_strength = aug_config.leakage_strength
    if rng is None:
        rng = np.random.default_rng(cfg.seed)

    real_cfg = cfg.with_overrides(n=n_real)
    df_real, U_real = generate_data(real_cfg, rng=rng, return_latents=True)
    df_real = df_real.copy()
    df_real["provenance_group"] = np.arange(n_real)
    df_real["is_synthetic"] = 0

    if n_synth_per_real == 0:
        _PROVENANCE_SCHEMA.validate(df_real)
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

    _PROVENANCE_SCHEMA.validate(df)
    return df
