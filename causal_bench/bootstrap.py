"""IC bootstrap confidence intervals for EIF-based estimators.

All three methods bootstrap the influence-curve (IC) values that each
estimator already computes rather than re-fitting the model. This makes
them O(B·n) instead of O(B × full-fit-cost).

The estimator's first-order expansion is  θ̂ ≈ θ + n⁻¹ Σ IC_i, so the
bootstrap distribution of  θ̂* - θ̂  is well-approximated by the bootstrap
distribution of  n⁻¹ Σ (IC*_i - IC̄).

Methods
-------
percentile : Empirical quantiles of the bootstrap distribution.
             Fast, but no skewness correction.

t           : Bootstrap-t (Studentized bootstrap). Uses the SE estimated
             within each resample: t*_b = mean(IC*) / SE(IC*). Then
             CI = [θ̂ - q_{1-α/2}·SE, θ̂ - q_{α/2}·SE] where q comes from
             the empirical t* distribution. Corrects for skewness in the
             sampling distribution; Hesterberg (2015) recommends this as
             the most accurate single-level bootstrap CI.

bca         : Bias-corrected and accelerated (Efron & Tibshirani 1993).
             Adds bias correction z₀ and jackknife-based acceleration a
             to the percentile endpoints. Best coverage for skewed
             estimators.

Usage
-----
>>> result = TMLEIPCWEstimator().estimate(df)[0]
>>> lo, hi = ic_bootstrap_ci(result, B=2000, method="bca")
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from causal_bench.metrics import EstimatorResult


def ic_bootstrap_ci(
    result: EstimatorResult,
    B: int = 2000,
    method: str = "bca",
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, float]:
    """Return a bootstrap CI (lo, hi) from a result that carries IC values.

    Parameters
    ----------
    result : EstimatorResult with a non-None ``ic`` array of length n.
    B      : Number of bootstrap resamples (2000 recommended for BCa/t).
    method : "percentile", "t", or "bca".
    alpha  : Two-sided error level (default 0.05 → 95 % CI).
    rng    : Optional numpy Generator for reproducibility.

    Returns
    -------
    (ci_lower, ci_upper) as floats.

    Raises
    ------
    ValueError if result.ic is None.
    """
    if result.ic is None:
        raise ValueError(
            f"EstimatorResult for '{result.name}' has no ic array. "
            "Re-run the estimator — TMLE+IPCW, LTMLE, and AIPW all store IC."
        )

    ic = np.asarray(result.ic, dtype=float)
    n = len(ic)
    theta = result.point_estimate
    rng = rng or np.random.default_rng()

    # --- Bootstrap distribution of θ* ---
    # θ*_b = θ̂ + mean(IC*_b)   (first-order semiparametric expansion)
    boot_idx = rng.integers(0, n, size=(B, n))
    ic_boot_means = ic[boot_idx].mean(axis=1)       # shape (B,)
    theta_boot = theta + ic_boot_means              # shape (B,)

    if method == "percentile":
        lo, hi = np.percentile(theta_boot, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        return float(lo), float(hi)

    if method == "t":
        # SE within each resample: std(IC*_b) / sqrt(n)
        se_hat = float(np.std(ic, ddof=1)) / np.sqrt(n)
        if se_hat < 1e-10:
            # Degenerate IC — fall back to percentile
            return ic_bootstrap_ci(result, B=B, method="percentile",
                                   alpha=alpha, rng=rng)
        se_boot = ic[boot_idx].std(axis=1, ddof=1) / np.sqrt(n)  # shape (B,)
        # Replace near-zero boot SEs to avoid division blow-up
        se_boot = np.where(se_boot < 1e-10, se_hat, se_boot)
        t_star = ic_boot_means / se_boot                          # shape (B,)
        q_lo, q_hi = np.percentile(t_star, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        # Studentized CI: θ̂ ∓ q·SE  (note: lower quantile gives upper bound)
        lo = theta - q_hi * se_hat
        hi = theta - q_lo * se_hat
        return float(lo), float(hi)

    if method == "bca":
        # Bias-correction z₀
        prop_below = np.mean(theta_boot < theta)
        # Guard edge cases where prop_below is exactly 0 or 1
        prop_below = np.clip(prop_below, 1e-6, 1 - 1e-6)
        z0 = float(stats.norm.ppf(prop_below))

        # Acceleration a via jackknife leave-one-out IC means
        jk_means = (ic.sum() - ic) / (n - 1)          # shape (n,)
        jk_bar = jk_means.mean()
        diff = jk_bar - jk_means
        num = float(np.sum(diff ** 3))
        den = float(6.0 * np.sum(diff ** 2) ** 1.5)
        a = num / den if abs(den) > 1e-10 else 0.0

        z_lo = stats.norm.ppf(alpha / 2)
        z_hi = stats.norm.ppf(1 - alpha / 2)

        def _adj_quantile(z_a: float) -> float:
            denom = 1.0 - a * (z0 + z_a)
            if abs(denom) < 1e-10:
                return alpha / 2 if z_a < 0 else 1 - alpha / 2
            return float(stats.norm.cdf(z0 + (z0 + z_a) / denom))

        p_lo = _adj_quantile(z_lo)
        p_hi = _adj_quantile(z_hi)
        lo, hi = np.percentile(theta_boot, [100 * p_lo, 100 * p_hi])
        return float(lo), float(hi)

    raise ValueError(f"Unknown method {method!r}. Choose 'percentile', 't', or 'bca'.")


def row_bootstrap_ci(
    estimator,
    data,
    B: int = 1000,
    method: str = "percentile",
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Nonparametric row-resampling bootstrap CI for an arbitrary scalar estimator.

    Unlike ``ic_bootstrap_ci`` (which resamples a precomputed influence curve and
    is O(B·n)), this resamples ROWS with replacement and RE-RUNS the full
    ``estimator`` on each resample — so any internal calibration / nuisance step
    is re-estimated per replicate and its variance is captured. Use it for
    estimators with no closed-form influence curve, e.g. a regression-calibration
    + OLS pipeline whose plug-in SE understates the calibration uncertainty.

    Parameters
    ----------
    estimator : callable(data_like) -> float
        Maps a (row-resampled) copy of ``data`` to a scalar estimate. Must
        include every step whose variance should be captured (the calibration
        step, not just the final fit).
    data : pandas DataFrame or numpy ndarray
        Row-indexable sample; rows are resampled with replacement.
    method : {"percentile", "basic"}
        Percentile interval, or the basic (reflected) interval
        ``[2θ̂ − q_hi, 2θ̂ − q_lo]``.

    Returns ``(lo, hi)``. Replicates that raise or return non-finite are dropped.
    """
    n = len(data)
    use_iloc = hasattr(data, "iloc")

    def _take(idx):
        return data.iloc[idx] if use_iloc else data[idx]

    rng = np.random.default_rng(seed)
    theta = np.empty(B)
    theta[:] = np.nan
    for b in range(B):
        idx = rng.integers(0, n, n)
        try:
            val = float(estimator(_take(idx)))
        except Exception:
            continue
        theta[b] = val
    theta = theta[np.isfinite(theta)]
    if theta.size < max(20, B // 10):
        raise ValueError("too many bootstrap replicates failed to produce a finite estimate")

    q_lo, q_hi = np.percentile(theta, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    if method == "percentile":
        return float(q_lo), float(q_hi)
    if method == "basic":
        point = float(estimator(data))
        return float(2 * point - q_hi), float(2 * point - q_lo)
    raise ValueError(f"Unknown method {method!r}. Choose 'percentile' or 'basic'.")
