"""Tests for site clustering in the registry DGP (issue #24 part 1)."""
from __future__ import annotations

import numpy as np
import pytest

from causal_bench.dgp.registry import RegistryConfig, generate_registry_data, _generate_registry_arm


# ── RegistryConfig field tests ───────────────────────────────────────────────

def test_registry_config_default_n_sites():
    cfg = RegistryConfig()
    assert cfg.n_sites == 20


def test_registry_config_default_icc():
    cfg = RegistryConfig()
    assert cfg.icc == 0.0


def test_registry_config_custom_clustering():
    cfg = RegistryConfig(n_sites=10, icc=0.2)
    assert cfg.n_sites == 10
    assert cfg.icc == 0.2


def test_registry_config_n_sites_bounds():
    with pytest.raises(Exception):
        RegistryConfig(n_sites=1)   # ge=2
    with pytest.raises(Exception):
        RegistryConfig(n_sites=501)  # le=500


def test_registry_config_icc_bounds():
    with pytest.raises(Exception):
        RegistryConfig(icc=-0.01)
    with pytest.raises(Exception):
        RegistryConfig(icc=1.01)


# ── _generate_registry_arm clustering tests ──────────────────────────────────

def test_site_id_column_always_present():
    """site_id column must be present even when icc=0."""
    rng = np.random.default_rng(0)
    df = _generate_registry_arm(200, -0.12, 0.35, 0.5, 0.06, "main", rng, n_sites=5, icc=0.0)
    assert "site_id" in df.columns


def test_site_id_zeros_when_no_clustering():
    """When icc=0, site_id should be all zeros (no-op path)."""
    rng = np.random.default_rng(0)
    df = _generate_registry_arm(200, -0.12, 0.35, 0.5, 0.06, "main", rng, n_sites=5, icc=0.0)
    assert (df["site_id"] == 0).all()


def test_site_id_range_with_clustering():
    """With icc > 0, site_id values should span [0, n_sites)."""
    rng = np.random.default_rng(1)
    n_sites = 8
    df = _generate_registry_arm(500, -0.12, 0.35, 0.5, 0.06, "main", rng, n_sites=n_sites, icc=0.2)
    assert df["site_id"].min() >= 0
    assert df["site_id"].max() < n_sites
    # With n=500 and 8 sites, all sites should be represented
    assert df["site_id"].nunique() == n_sites


def test_icc_zero_behaviour_unchanged():
    """icc=0 should produce the same outcomes as no clustering (same rng state)."""
    # Both arms use the same sequence of rng draws; with icc=0 no site effects drawn
    rng1 = np.random.default_rng(42)
    df_cluster = _generate_registry_arm(300, -0.12, 0.35, 0.5, 0.06, "main", rng1, n_sites=10, icc=0.0)

    rng2 = np.random.default_rng(42)
    df_plain = _generate_registry_arm(300, -0.12, 0.35, 0.5, 0.06, "main", rng2, n_sites=1, icc=0.0)

    # Y, A, W columns should be identical (site_id path adds no rng draws when icc=0)
    for col in ["Y", "A", "W1", "W2", "W3"]:
        np.testing.assert_array_equal(df_cluster[col].values, df_plain[col].values)


# ── ICC empirical verification ────────────────────────────────────────────────

def test_icc_zero_outcomes_uncorrelated():
    """At icc=0 with large n, within-site ICC ≈ 0 empirically."""
    rng = np.random.default_rng(7)
    n_sites = 20
    df = _generate_registry_arm(5000, -0.12, 0.35, 0.5, 0.0, "main", rng, n_sites=n_sites, icc=0.0)
    # Estimate ICC via ANOVA: MSB / (MSB + (n_bar - 1) * MSW)
    site_means = df.groupby("site_id")["Y"].mean()
    grand_mean = df["Y"].mean()
    n_per_site = df.groupby("site_id")["Y"].count()
    n_bar = n_per_site.mean()
    k = n_sites
    ssb = float((n_per_site * (site_means - grand_mean) ** 2).sum())
    msb = ssb / (k - 1)
    ssw = float(df.groupby("site_id")["Y"].apply(lambda g: ((g - g.mean()) ** 2).sum()).sum())
    msw = ssw / (len(df) - k)
    icc_est = (msb - msw) / (msb + (n_bar - 1) * msw) if msb > msw else 0.0
    assert abs(icc_est) < 0.10, f"Expected ICC ≈ 0 at icc=0, got {icc_est:.4f}"


def test_icc_positive_within_vs_between():
    """At icc=0.2, within-site Y-correlation > between-site Y-correlation."""
    rng = np.random.default_rng(9)
    n_sites = 20
    target_icc = 0.2
    df = _generate_registry_arm(4000, -0.12, 0.35, 0.5, 0.0, "main", rng,
                                 n_sites=n_sites, icc=target_icc)

    # Estimate ICC via ANOVA on Y
    site_means = df.groupby("site_id")["Y"].mean()
    grand_mean = df["Y"].mean()
    n_per_site = df.groupby("site_id")["Y"].count()
    n_bar = float(n_per_site.mean())
    k = n_sites
    ssb = float((n_per_site * (site_means - grand_mean) ** 2).sum())
    msb = ssb / (k - 1)
    ssw = float(df.groupby("site_id")["Y"].apply(lambda g: ((g - g.mean()) ** 2).sum()).sum())
    msw = ssw / (len(df) - k)
    icc_est = (msb - msw) / (msb + (n_bar - 1) * msw) if msb > msw else 0.0

    # With target ICC=0.2 and n=4000, empirical ICC should clearly exceed zero
    assert icc_est > 0.05, f"Expected ICC > 0.05 at target icc=0.2, got {icc_est:.4f}"


# ── generate_registry_data integration test ──────────────────────────────────

def test_generate_registry_data_site_id_present():
    """All three registry DataFrames should have a site_id column."""
    cfg = RegistryConfig(n_sites=10, icc=0.15, seed=0)
    main_df, teer_df, mac_df, _ = generate_registry_data(cfg)
    for df, name in [(main_df, "main"), (teer_df, "teer"), (mac_df, "mac")]:
        assert "site_id" in df.columns, f"site_id missing from {name} registry"


def test_generate_registry_data_backward_compatible():
    """Default RegistryConfig (icc=0) must produce the same results as before — no site_id in A/Y/W sense."""
    cfg = RegistryConfig(seed=99)
    assert cfg.icc == 0.0
    main_df, teer_df, mac_df, embeddings = generate_registry_data(cfg)
    assert len(main_df) == cfg.n_main
    assert len(teer_df) == cfg.n_teer
    assert len(mac_df)  == cfg.n_mac
    # Columns that existed before clustering
    for col in ["Y", "A", "W1", "W2", "W3", "cate", "registry", "p0", "p1"]:
        assert col in main_df.columns
