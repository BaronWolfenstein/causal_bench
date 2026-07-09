"""Log-space weight normalization and Kish effective sample size."""
from __future__ import annotations

import numpy as np

from .backend import get_namespace


def normalize_log_weights(log_w) -> tuple:
    """Return (normalized weights, log normalizer). Subtract the max before
    exponentiating so weights never under/overflow. numpy or cupy in."""
    xp = get_namespace(log_w)
    m = xp.max(log_w)
    if not bool(xp.isfinite(m)):
        # all -inf (total weight collapse / positivity failure) or a +inf leaked
        # in — surface it loudly rather than returning silent nan weights.
        raise ValueError(
            "normalize_log_weights: non-finite max log-weight "
            f"({float(m)}) — total weight collapse (every particle out of "
            "support). This is a positivity failure; fix upstream (twist "
            "earlier), do not reweight."
        )
    shifted = xp.exp(log_w - m)
    total = shifted.sum()
    log_norm = m + xp.log(total)
    return shifted / total, float(log_norm)


def kish_ess(log_w) -> float:
    """Kish ESS = (sum w)^2 / sum(w^2) = 1 / sum(w_norm^2)."""
    xp = get_namespace(log_w)
    w, _ = normalize_log_weights(log_w)
    return float(1.0 / xp.sum(w ** 2))
