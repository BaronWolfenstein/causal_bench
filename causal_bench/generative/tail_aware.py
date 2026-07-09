"""Tail-aware training weight ∝ 1/p(z): fit a GMM density on the embeddings and
importance-weight the (denoising) loss toward low-density rare regions. This is
the Test B′ fix — recover rare-mode fidelity without adding a learned latent."""
from __future__ import annotations

from typing import Optional

import numpy as np


def inverse_density_weights(X, n_components: int = 5,
                            clip: Optional[float] = None) -> np.ndarray:
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
    # multiple of the 99.5th percentile so genuine rare-vs-common separation
    # (which lives well below this threshold) is preserved while pathological
    # outliers are truncated. This is independent of the opt-in `clip` below.
    default_cap = np.percentile(w, 99.5) * 5.0
    w = np.minimum(w, default_cap)

    w = w / w.mean()                        # normalize to mean 1 (stabilized)
    if clip is not None:
        w = np.minimum(w, clip)            # bias-for-variance truncation (last resort)
        w = w / w.mean()                    # re-normalize after user-requested clip
    return w
