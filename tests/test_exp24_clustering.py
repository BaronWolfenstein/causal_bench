"""Tests for exp24_site_clustering.py (issue #24 part 3).

At ICC=0, coverage of both independence SE and cluster-robust SE should both be
≈ 0.95 (within 0.10 tolerance for N_REPS=200).
"""
from __future__ import annotations

import numpy as np
import pytest


def _compute_estimand(n_sites: int = 10, icc: float = 0.0, n_large: int = 50_000) -> float:
    """Compute the large-sample limit of the difference-in-means ATE estimator.

    The registry DGP has mild confounding (W1, W2 affect both A and Y), so the
    difference-in-means E[Y|A=1] − E[Y|A=0] does not equal cfg.true_ate_main.
    We estimate the true *estimand* of the naive estimator via a large-n draw.
    """
    from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
    cfg = RegistryConfig(n_main=n_large, n_sites=n_sites, icc=icc, seed=999999)
    main_df, _, _, _ = generate_registry_data(cfg)
    t = main_df[main_df["A"] == 1]["Y"].mean()
    c = main_df[main_df["A"] == 0]["Y"].mean()
    return float(t - c)


def test_exp24_icc0_coverage_both_estimators():
    """At ICC=0, both independence and cluster-robust SEs should yield ~95% coverage.

    The difference-in-means estimator is biased for cfg.true_ate_main (the DGP has
    mild confounding). We therefore test coverage of the *estimand* that the estimator
    actually targets — its large-sample limit — computed via a large-n reference draw.

    N_REPS=200 → MC SE ≈ sqrt(0.95*0.05/200) ≈ 0.015, so allow ±0.10 tolerance.
    """
    from scipy.stats import norm
    from causal_bench.dgp.registry import RegistryConfig, generate_registry_data
    from causal_bench.estimators.hierarchical import summarise_registry

    icc      = 0.0
    n_sites  = 10
    n_main   = 700
    n_reps   = 200
    alpha    = 0.05
    B        = 200
    z        = norm.ppf(1.0 - alpha / 2.0)
    tolerance = 0.10   # ±10 pp from 95%

    # Estimand: large-n limit of the difference-in-means at ICC=0
    target_ate = _compute_estimand(n_sites=n_sites, icc=icc)

    cover_indep  = 0
    cover_robust = 0

    for rep in range(n_reps):
        seed = rep * 1000 + n_sites
        cfg = RegistryConfig(n_main=n_main, n_sites=n_sites, icc=icc, seed=seed)
        main_df, _, _, _ = generate_registry_data(cfg)

        s_i = summarise_registry(main_df, target_ate, "main", cluster_robust=False)
        ate = s_i.ate_hat
        if ate - z * s_i.se_hat <= target_ate <= ate + z * s_i.se_hat:
            cover_indep += 1

        s_r = summarise_registry(
            main_df, target_ate, "main",
            cluster_robust=True, bootstrap_B=B,
            bootstrap_rng=np.random.default_rng(seed + 1),
        )
        if ate - z * s_r.se_hat <= target_ate <= ate + z * s_r.se_hat:
            cover_robust += 1

    cov_i = cover_indep  / n_reps
    cov_r = cover_robust / n_reps

    assert abs(cov_i - 0.95) <= tolerance, (
        f"Independence SE coverage at ICC=0 is {cov_i:.3f}, "
        f"expected 0.95 ± {tolerance} (estimand={target_ate:.4f})"
    )
    assert abs(cov_r - 0.95) <= tolerance, (
        f"Cluster-robust SE coverage at ICC=0 is {cov_r:.3f}, "
        f"expected 0.95 ± {tolerance} (estimand={target_ate:.4f})"
    )


def test_exp24_experiment_module_importable():
    """exp24_site_clustering.py must be importable without errors."""
    import importlib
    mod = importlib.import_module("experiments.exp24_site_clustering")
    assert hasattr(mod, "main")
    assert hasattr(mod, "run_coverage_experiment")
    assert hasattr(mod, "plot_coverage")


def test_exp24_run_coverage_experiment_small():
    """run_coverage_experiment with tiny settings should produce a valid DataFrame."""
    import importlib
    import unittest.mock as mock

    mod = importlib.import_module("experiments.exp24_site_clustering")

    # Patch the grid constants to run very fast
    with (
        mock.patch.object(mod, "ICC_GRID",    [0.0, 0.10]),
        mock.patch.object(mod, "NSITES_GRID", [5]),
        mock.patch.object(mod, "N_REPS",      10),
        mock.patch.object(mod, "BOOTSTRAP_B", 20),
        mock.patch.object(mod, "N_MAIN",      200),
    ):
        df = mod.run_coverage_experiment()

    assert len(df) == 2  # 2 ICC × 1 n_sites
    for col in ["icc", "n_sites", "coverage_indep", "coverage_robust", "ate_bias"]:
        assert col in df.columns, f"Missing column: {col}"
    assert (df["coverage_indep"].between(0, 1)).all()
    assert (df["coverage_robust"].between(0, 1)).all()
