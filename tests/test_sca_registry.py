"""Tests for the SCA registry loader (Target Group + external Baseline Cohort).

The loader is the seam between the synthetic benchmark path and a real external
registry: both must satisfy the same schema so the downstream weighting/balance
pipeline is identical. Real registry data is not available here; the synthetic
generator is the benchmark path and the schema is the contract a real loader
must meet.
"""
import numpy as np
import pandas as pd
import pytest

from causal_bench.dgp.sca_registry import (
    REGISTRY_COVS, RegistrySchemaError, load_registry, synthetic_registry)


def test_synthetic_registry_conforms_to_schema():
    df = synthetic_registry(seed=1, n_target=299, n_baseline=2000)
    # one row per record; required columns present and typed
    assert {"group", "provenance", *REGISTRY_COVS} <= set(df.columns)
    assert set(df["group"].unique()) <= {"target", "baseline"}
    assert (df["group"] == "target").sum() == 299
    assert (df["group"] == "baseline").sum() == 2000
    # real synthetic-benchmark records are all real-provenance (augmentation is
    # a separate, later step — the loader itself introduces no synthetic records)
    assert set(df["provenance"].unique()) == {"real"}


def test_load_registry_roundtrips_a_conforming_frame():
    df = synthetic_registry(seed=2)
    loaded = load_registry(df)
    assert len(loaded) == len(df)
    assert {"group", "provenance", *REGISTRY_COVS} <= set(loaded.columns)


def test_load_registry_rejects_missing_covariate():
    df = synthetic_registry(seed=3).drop(columns=[REGISTRY_COVS[0]])
    with pytest.raises(RegistrySchemaError):
        load_registry(df)


def test_load_registry_rejects_bad_group_label():
    df = synthetic_registry(seed=4)
    df.loc[df.index[0], "group"] = "treated"   # not in {target, baseline}
    with pytest.raises(RegistrySchemaError):
        load_registry(df)


def test_target_group_is_more_severe_than_baseline():
    # engineered sparsity: the Target Group carries a fatter upper tail on the
    # severity covariate than the external Baseline Cohort
    df = synthetic_registry(seed=5)
    t = df[df.group == "target"]["sev"]
    b = df[df.group == "baseline"]["sev"]
    assert t.mean() > b.mean()
    assert (t >= 2.0).mean() > 5 * (b >= 2.0).mean()
