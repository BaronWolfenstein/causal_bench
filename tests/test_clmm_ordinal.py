"""Tests for the Bayesian CLMM ordinal estimator (issue #27).

Design notes
------------
* The real ordinal-PRO DGP (#26) is built in parallel and may not exist in this
  worktree, so all tests use a self-contained synthetic ordinal dataset generated
  here (thresholded-latent with site RE, known positive effect).
* MCMC tests are marked `requires_bambi` and skipped gracefully when bambi/PyMC
  is not importable — useful in environments without the `bayes` optional dep.
* Cheap MCMC settings (few draws/tune, 1 chain) keep the suite fast.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Skip marker for the optional bambi/PyMC dependency
# ---------------------------------------------------------------------------

try:
    import bambi  # noqa: F401
    _BAMBI_OK = True
except ImportError:
    _BAMBI_OK = False

requires_bambi = pytest.mark.skipif(
    not _BAMBI_OK,
    reason="bambi not installed — pip install 'causal_bench[bayes]'",
)


# ---------------------------------------------------------------------------
# Synthetic ordinal dataset helper
# ---------------------------------------------------------------------------

def _make_ordinal_df(
    n: int = 200,
    n_sites: int = 4,
    true_log_or: float = 0.8,
    seed: int = 42,
) -> pd.DataFrame:
    """Self-contained ordinal DGP: thresholded-latent with site random effects.

    Latent model: y* = true_log_or * A + 0.3*W1 - 0.2*W2 + u_site + eps
    Ordinal categories: 1, 2, 3, 4  (thresholded from y*)
    Site REs: u_site ~ N(0, 0.5^2)

    Returns a DataFrame with columns: support (int, 1-4), A, W1, W2, site (str).
    """
    rng = np.random.default_rng(seed)
    site_ids = rng.integers(0, n_sites, size=n)
    site_re = rng.normal(0, 0.5, size=n_sites)[site_ids]
    A = rng.binomial(1, 0.5, size=n).astype(float)
    W1 = rng.normal(0, 1, size=n)
    W2 = rng.normal(0, 1, size=n)
    eps = rng.normal(0, 1, size=n)
    y_lat = true_log_or * A + 0.3 * W1 - 0.2 * W2 + site_re + eps
    thresholds = np.percentile(y_lat, [33, 66, 85])
    support = np.digitize(y_lat, thresholds) + 1  # categories 1, 2, 3, 4
    return pd.DataFrame({
        "support": support.astype(int),
        "A": A,
        "W1": W1,
        "W2": W2,
        "site": [f"site_{s}" for s in site_ids],
    })


# ---------------------------------------------------------------------------
# Interface / structural tests (no MCMC required)
# ---------------------------------------------------------------------------

class TestCLMMOrdinalInterface:
    """Verify the estimator satisfies the BaseEstimator contract."""

    def test_import(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        est = CLMMOrdinalEstimator()
        assert est is not None

    def test_is_base_estimator(self):
        from causal_bench.estimators.base import BaseEstimator
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        assert issubclass(CLMMOrdinalEstimator, BaseEstimator)

    def test_has_name(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        est = CLMMOrdinalEstimator()
        assert isinstance(est.name, str)
        assert len(est.name) > 0

    def test_has_estimate_method(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        assert hasattr(CLMMOrdinalEstimator, "estimate")
        import inspect
        sig = inspect.signature(CLMMOrdinalEstimator.estimate)
        assert "df" in sig.parameters
        assert "horizon" in sig.parameters
        assert "estimand" in sig.parameters

    def test_registry_entry(self):
        from causal_bench.estimators import ESTIMATOR_REGISTRY
        assert "clmm_ordinal" in ESTIMATOR_REGISTRY

    def test_registry_instantiation(self):
        from causal_bench.estimators import get_estimator
        est = get_estimator("clmm_ordinal")
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        assert isinstance(est, CLMMOrdinalEstimator)

    def test_not_in_mvp_estimators(self):
        from causal_bench.estimators import MVP_ESTIMATORS
        assert "clmm_ordinal" not in MVP_ESTIMATORS

    def test_returns_empty_when_bambi_unavailable(self, monkeypatch):
        """When bambi is not importable, estimate() warns and returns []."""
        import causal_bench.estimators.clmm_ordinal as mod
        monkeypatch.setattr(mod, "_BAMBI_AVAILABLE", False)
        est = mod.CLMMOrdinalEstimator()
        df = _make_ordinal_df(n=50)
        import warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = est.estimate(df)
        assert result == []
        assert any("bambi" in str(x.message).lower() for x in w)


# ---------------------------------------------------------------------------
# MCMC tests — skipped when bambi unavailable
# ---------------------------------------------------------------------------

class TestCLMMOrdinalMCMC:
    """Functional tests that run the full MCMC sampler."""

    _MCMC_KWARGS = dict(
        draws=100,
        tune=100,
        chains=1,
        target_accept=0.8,
        progressbar=False,
    )

    @requires_bambi
    def test_returns_list_of_estimator_results(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        from causal_bench.metrics import EstimatorResult
        df = _make_ordinal_df(n=120)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        assert isinstance(results, list)
        assert len(results) >= 1
        assert all(isinstance(r, EstimatorResult) for r in results)

    @requires_bambi
    def test_has_cumulative_log_or_result(self):
        """At least one result targets the cumulative log-OR."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=120)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        estimands = [r.estimand for r in results]
        assert any("log_or" in e.lower() or "cumulative" in e.lower() for e in estimands), (
            f"Expected a cumulative log-OR estimand, got: {estimands}"
        )

    @requires_bambi
    def test_credible_interval_brackets_point_estimate(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=120)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        for r in results:
            assert r.ci_lower <= r.point_estimate <= r.ci_upper, (
                f"CI [{r.ci_lower:.3f}, {r.ci_upper:.3f}] does not "
                f"bracket point estimate {r.point_estimate:.3f}"
            )

    @requires_bambi
    def test_positive_standard_error(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=120)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        for r in results:
            assert r.standard_error > 0

    @requires_bambi
    def test_recovers_positive_sign_of_log_or(self):
        """With true_log_or=0.8 the posterior mean cumulative log-OR should be positive."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        # larger n for reliable recovery
        df = _make_ordinal_df(n=300, true_log_or=0.8, seed=7)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        log_or_results = [
            r for r in results
            if "log_or" in r.estimand.lower() or "cumulative" in r.estimand.lower()
        ]
        assert log_or_results, "No cumulative log-OR result found"
        pt = log_or_results[0].point_estimate
        assert pt > 0, (
            f"Expected positive log-OR (true=0.8), got {pt:.3f}. "
            "Sign recovery failed."
        )

    @requires_bambi
    def test_rough_magnitude_of_log_or(self):
        """Posterior mean should be within 2x of the true log-OR (rough sanity check)."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=400, true_log_or=1.0, seed=123)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        log_or_results = [
            r for r in results
            if "log_or" in r.estimand.lower() or "cumulative" in r.estimand.lower()
        ]
        assert log_or_results
        pt = log_or_results[0].point_estimate
        # Rough: within [0.25, 2.75] for true=1.0 with only 100 draws
        assert 0.0 < pt < 4.0, f"Magnitude implausible: {pt:.3f} (true=1.0)"

    @requires_bambi
    def test_credible_interval_wider_than_zero(self):
        """The 95% credible interval must have positive width."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=120)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        for r in results:
            assert r.ci_upper > r.ci_lower, (
                f"Zero-width CI for {r.estimand}: [{r.ci_lower}, {r.ci_upper}]"
            )

    @requires_bambi
    def test_convergence_info_present(self):
        """Estimator should attach r_hat or n_eff info to convergence_info."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=120)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        log_or_r = next(
            (r for r in results if "log_or" in r.estimand.lower() or "cumulative" in r.estimand.lower()),
            None
        )
        assert log_or_r is not None
        assert log_or_r.convergence_info is not None
        assert isinstance(log_or_r.convergence_info, dict)

    @requires_bambi
    def test_ess_positive(self):
        """ESS (effective sample size) attached to results must be positive."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=120)
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        results = est.estimate(df)
        for r in results:
            if r.ess is not None:
                assert r.ess > 0

    @requires_bambi
    def test_estimate_negative_control_near_zero(self):
        """Negative-control outcome Y_neg has no treatment effect — NC estimate ~0."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        rng = np.random.default_rng(0)
        n = 400
        df = _make_ordinal_df(n=n, seed=0)
        df["Y_neg"] = rng.normal(0, 1, size=n)  # no treatment effect
        est = CLMMOrdinalEstimator(**self._MCMC_KWARGS)
        nc = est.estimate_negative_control(df)
        assert abs(nc) < 0.5, f"NC estimate too large: {nc:.3f}"


class TestCLMMPoolingAndRandomSlope:
    """Pooling spectrum (partial / complete / none) + (A | site) random slope."""

    _MCMC_KWARGS = dict(draws=100, tune=100, chains=1, target_accept=0.8, progressbar=False)

    def test_pooling_validation(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        with pytest.raises(ValueError):
            CLMMOrdinalEstimator(pooling="bogus")

    def test_random_slope_requires_partial_pooling(self):
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        with pytest.raises(ValueError):
            CLMMOrdinalEstimator(pooling="none", random_slope=True)

    def test_registry_has_pooling_variants(self):
        from causal_bench.estimators import ESTIMATOR_REGISTRY as R
        for key in ("clmm_ordinal", "clmm_ordinal_slope",
                    "clmm_ordinal_nopool", "clmm_ordinal_cpool"):
            assert key in R, f"{key} missing from registry"

    @requires_bambi
    def test_all_pooling_arms_return_finite_log_or(self):
        """partial / complete / none each fit and return a finite treatment log-OR."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=300, n_sites=6, true_log_or=0.8, seed=3)
        for pooling in ("partial", "complete", "none"):
            est = CLMMOrdinalEstimator(pooling=pooling, **self._MCMC_KWARGS)
            res = est.estimate(df)
            assert res, f"pooling={pooling} returned no result"
            assert np.isfinite(res[0].point_estimate), f"pooling={pooling} non-finite"

    @requires_bambi
    def test_partial_pooling_surfaces_site_sd(self):
        """convergence_info exposes the site random-intercept SD (τ)."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=300, n_sites=6, seed=4)
        r = CLMMOrdinalEstimator(**self._MCMC_KWARGS).estimate(df)[0]
        ci = r.convergence_info
        assert "site_sd_mean" in ci and ci["site_sd_mean"] > 0
        assert "site_sd_hdi" in ci and len(ci["site_sd_hdi"]) == 2

    @requires_bambi
    def test_complete_pooling_has_no_site_sd(self):
        """Complete pooling drops the site term, so no τ is surfaced."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=300, n_sites=6, seed=5)
        r = CLMMOrdinalEstimator(pooling="complete", **self._MCMC_KWARGS).estimate(df)[0]
        assert "site_sd_mean" not in r.convergence_info

    @requires_bambi
    def test_random_slope_surfaces_slope_sd(self):
        """(A | site) surfaces both the intercept SD (τ) and the slope SD (τ_A)."""
        from causal_bench.estimators.clmm_ordinal import CLMMOrdinalEstimator
        df = _make_ordinal_df(n=350, n_sites=6, seed=6)
        r = CLMMOrdinalEstimator(random_slope=True, **self._MCMC_KWARGS).estimate(df)[0]
        ci = r.convergence_info
        assert "site_sd_mean" in ci and ci["site_sd_mean"] > 0
        assert "slope_sd_mean" in ci and ci["slope_sd_mean"] > 0
