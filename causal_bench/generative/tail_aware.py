"""Tail-aware training weight ∝ 1/p(z): fit a GMM density on the embeddings and
importance-weight the (denoising) loss toward low-density rare regions. This is
the Test B′ fix — recover rare-mode fidelity without adding a learned latent."""
from __future__ import annotations

from typing import Optional

import numpy as np


def inverse_density_weights(X, n_components: int = 5,
                            clip: Optional[float] = None,
                            cap_percentile: float = 99.5,
                            cap_multiplier: Optional[float] = 5.0) -> np.ndarray:
    """Compute per-sample 1/p(z) importance weights from a GMM density fit to X.

    Rare / low-density points receive a higher weight than common / high-density
    points; the resulting vector is normalized to mean 1. By default, an outlier
    cap is applied to the raw 1/p(z) weights before normalization to control
    importance-weight variance (standard IPW practice): this trades a small
    amount of extreme-tail per-point fidelity (see cap_percentile/cap_multiplier
    below) for training stability in downstream consumers. Separately, and
    unconditionally regardless of the cap setting, an inf/nan safety guard is
    always applied so a single degenerate (near-zero-density) point can never
    corrupt the vector.

    Args:
        X: (n_samples, n_features) array of embeddings to fit the density on
            and to score.
        n_components: number of GMM mixture components used to estimate p(z).
        clip: optional opt-in additional truncation applied to the normalized
            weights (bias-for-variance truncation, last resort). Independent
            of the default outlier cap below; None (default) disables it.
        cap_percentile: percentile of the raw 1/p(z) weight distribution used
            as the base of the default outlier cap. Only used when
            cap_multiplier is not None.
        cap_multiplier: multiplier applied to the cap_percentile value to form
            the default outlier cap threshold (cap = percentile(w, cap_percentile)
            * cap_multiplier). Raw weights above this threshold are truncated
            to it before mean-normalization. Set to None to disable the default
            cap entirely (weights are then only bounded by the unconditional
            inf/nan guard and, if requested, the opt-in `clip`). Defaults to
            5.0, reproducing the historical always-on behavior.

    Returns:
        1-D array of weights, same length as X, normalized to mean 1.
    """
    from sklearn.mixture import GaussianMixture
    # n_init=5 keeps the best-of-5 EM fit (by lower-bound), guarding against
    # a single unlucky initialization producing a degenerate density estimate.
    # covariance_type='diag' retained: reasonable default for embeddings and,
    # combined with n_init, empirically robust across seeds (see
    # tests/test_gen_tail_aware.py's multi-seed sweep). reg_covar bumped from
    # sklearn's default (1e-6) to 1e-4 for extra numerical stability on small
    # per-component sample counts.
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                          random_state=0, n_init=5, reg_covar=1e-4).fit(X)
    log_p = gmm.score_samples(X)
    w = np.exp(-log_p)                     # 1/p(z)

    # Guard against non-finite weights: exp(-log_p) can overflow to inf for
    # pathological (near-zero-density) points. Replace any inf/nan entries
    # with a large-but-finite fallback derived from the finite tail of the
    # distribution, so a single degenerate point can never corrupt the
    # vector (e.g. propagate inf/nan into downstream consumers like Task 8's
    # torch net via the mean-normalization below).
    finite = np.isfinite(w)
    if not finite.all():
        fallback = np.percentile(w[finite], 99.9) * 10.0 if finite.any() else 1.0
        w = np.where(finite, w, fallback)

    # Default sane cap: even without an explicit `clip`, a single extreme-tail
    # point's weight can be orders of magnitude larger than the rest of the
    # vector and dominate mean-based aggregates downstream. Cap at a robust
    # multiple of the cap_percentile so the *aggregate* rare-vs-common ordering
    # (mean rare weight > mean common weight) is preserved. Note this does NOT
    # guarantee every individual rare-tail point is left untouched: in a
    # minority of cases (~10% of seeds in the multi-seed sweep below) the
    # single most-extreme rare point sits above the cap and gets truncated to
    # it; the aggregate ordering invariant still holds. This is independent of
    # the opt-in `clip` below. Set cap_multiplier=None to disable this cap.
    if cap_multiplier is not None:
        default_cap = np.percentile(w, cap_percentile) * cap_multiplier
        w = np.minimum(w, default_cap)

    w = w / w.mean()                        # normalize to mean 1 (stabilized)
    if clip is not None:
        w = np.minimum(w, clip)            # bias-for-variance truncation (last resort)
        w = w / w.mean()                    # re-normalize after user-requested clip
    return w
