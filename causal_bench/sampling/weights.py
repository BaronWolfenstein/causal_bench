"""Log-space weight normalization and Kish effective sample size."""
from __future__ import annotations

import numpy as np


def normalize_log_weights(log_w: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (normalized weights, log normalizer). Subtract the max before
    exponentiating so weights never under/overflow."""
    m = np.max(log_w)
    if not np.isfinite(m):
        # all -inf (total weight collapse / positivity failure) or a +inf leaked
        # in — surface it loudly rather than returning silent nan weights.
        raise ValueError(
            "normalize_log_weights: non-finite max log-weight "
            f"({m}) — total weight collapse (every particle out of support). "
            "This is a positivity failure; fix upstream (twist earlier), do not "
            "reweight."
        )
    shifted = np.exp(log_w - m)
    total = shifted.sum()
    log_norm = m + np.log(total)
    return shifted / total, float(log_norm)


def kish_ess(log_w: np.ndarray) -> float:
    """Kish ESS = (sum w)^2 / sum(w^2) = 1 / sum(w_norm^2)."""
    w, _ = normalize_log_weights(log_w)
    return float(1.0 / np.sum(w ** 2))
