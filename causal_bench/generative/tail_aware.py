"""Tail-aware training weight ∝ 1/p(z): fit a GMM density on the embeddings and
importance-weight the (denoising) loss toward low-density rare regions. This is
the Test B′ fix — recover rare-mode fidelity without adding a learned latent."""
from __future__ import annotations

from typing import Optional

import numpy as np


def inverse_density_weights(X, n_components: int = 5,
                            clip: Optional[float] = None) -> np.ndarray:
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=n_components, covariance_type='diag',
                          random_state=0).fit(X)
    log_p = gmm.score_samples(X)
    w = np.exp(-log_p)                     # 1/p(z)
    w = w / w.mean()                        # normalize to mean 1 (stabilized)
    if clip is not None:
        w = np.minimum(w, clip)            # bias-for-variance truncation (last resort)
    return w
