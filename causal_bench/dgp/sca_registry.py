"""SCA registry loader — Target Group + external Baseline Cohort under one schema.

The synthetic-control-arm pipeline compares a single-arm Target Group against an
external Baseline Cohort (registry). This module is the *seam* between two data
sources that must be interchangeable downstream:

- ``synthetic_registry`` — the benchmark path: a fully synthetic Target/Baseline
  DGP with an engineered sparse region on the severity covariate. No dependency
  on any real data.
- ``load_registry`` — validates an externally-supplied frame (a real registry
  export) against the same schema and returns it unchanged.

Both yield a long frame with columns ``[group, provenance, <covariates>]`` where
``group ∈ {target, baseline}``. The loader itself introduces **no** synthetic
records — provenance is ``real`` for everything it returns; generative
augmentation (adding synthetic Baseline records in the sparse region) is a
separate, later step that tags its rows ``synthetic``. Keeping those steps apart
is the anti-circularity discipline the borrowing spec §9 requires.

Covariates mirror the anatomical-covariate structure the weighting models
condition on: a smooth continuous block, a skewed covariate, a binary covariate,
and one continuous ``sev`` covariate whose upper tail is where the Baseline
Cohort is thin (the engineered positivity problem).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Covariate contract shared by the synthetic path and any real registry loader.
REGISTRY_COVS = ("x_a", "x_b", "x_skew", "x_bin", "sev")
_GROUPS = {"target", "baseline"}
_PROVENANCE = {"real", "synthetic"}


class RegistrySchemaError(ValueError):
    """Raised when a supplied registry frame violates the loader contract."""


def _draw(rng, n, loc, sd, bin_p, sev_loc, sev_tail):
    tail = rng.random(n) < sev_tail
    return pd.DataFrame({
        "x_a": rng.normal(loc[0], sd[0], n),
        "x_b": rng.normal(loc[1], sd[1], n),
        "x_skew": rng.lognormal(loc[2], sd[2], n),
        "x_bin": rng.binomial(1, bin_p, n).astype(float),
        "sev": np.where(tail, rng.normal(sev_loc + 2.2, 0.45, n),
                        rng.normal(sev_loc, 0.8, n)),
    })


def synthetic_registry(seed: int, n_target: int = 299,
                       n_baseline: int = 2000) -> pd.DataFrame:
    """Benchmark-path Target Group + Baseline Cohort with an engineered sparse
    region on ``sev`` (Target carries a fatter upper tail than Baseline)."""
    rng = np.random.default_rng(seed)
    target = _draw(rng, n_target, (0.35, -0.25, 0.15), (1.0, 1.1, 0.55),
                   bin_p=0.55, sev_loc=0.4, sev_tail=0.18)
    baseline = _draw(rng, n_baseline, (0.0, 0.0, 0.0), (1.0, 1.0, 0.5),
                     bin_p=0.42, sev_loc=0.0, sev_tail=0.02)
    target["group"], baseline["group"] = "target", "baseline"
    df = pd.concat([target, baseline], ignore_index=True)
    df["provenance"] = "real"
    return df[["group", "provenance", *REGISTRY_COVS]]


def load_registry(source, covs=REGISTRY_COVS) -> pd.DataFrame:
    """Validate and return an external registry frame (or a DataFrame in hand).

    ``source`` is a DataFrame or a path to a parquet/csv file. Enforces the
    loader contract: required covariate columns present and numeric, a valid
    ``group`` label, and (if present) a valid ``provenance`` label. Raises
    ``RegistrySchemaError`` on any violation — a real registry export must be
    made to conform before it reaches the weighting pipeline.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif str(source).endswith(".parquet"):
        df = pd.read_parquet(source)
    elif str(source).endswith(".csv"):
        df = pd.read_csv(source)
    else:
        raise RegistrySchemaError(f"unsupported registry source: {source!r}")

    if "group" not in df.columns:
        raise RegistrySchemaError("registry frame missing required column 'group'")
    bad_groups = set(df["group"].unique()) - _GROUPS
    if bad_groups:
        raise RegistrySchemaError(f"invalid group labels {bad_groups}; expected {_GROUPS}")

    missing = [c for c in covs if c not in df.columns]
    if missing:
        raise RegistrySchemaError(f"registry frame missing covariates: {missing}")
    for c in covs:
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise RegistrySchemaError(f"covariate {c!r} is not numeric")
        if df[c].isna().any():
            raise RegistrySchemaError(f"covariate {c!r} contains NaN")

    if "provenance" not in df.columns:
        df["provenance"] = "real"
    bad_prov = set(df["provenance"].unique()) - _PROVENANCE
    if bad_prov:
        raise RegistrySchemaError(f"invalid provenance labels {bad_prov}; expected {_PROVENANCE}")

    return df.reset_index(drop=True)
