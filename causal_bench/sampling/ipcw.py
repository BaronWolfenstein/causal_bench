"""IPCW for out-of-band particle kills. Any kill that does NOT go through the
SMC weight bookkeeping (validity filters, heuristic pruning) is informative
censoring: model its survival probability G and weight survivors by 1/G. For
multi-step filters, G is the product of per-step survival probabilities
(discrete-time IPCW); the stabilized form multiplies by a marginal survival in
the numerator. Positivity: where G -> 0, 1/G explodes and no reweighting
recovers lost support — clip and FLAG rather than silently trust the weight."""
from __future__ import annotations

from typing import Optional

import numpy as np


def ipcw_weights(survival_probs: np.ndarray,
                 *, stabilize_by: Optional[np.ndarray] = None) -> np.ndarray:
    """Inverse-probability-of-selection weights 1/G (optionally stabilized by a
    marginal survival numerator). `survival_probs` may be a product of per-step
    survivals already."""
    G = np.asarray(survival_probs, float)
    w = 1.0 / G
    if stabilize_by is not None:
        w = np.asarray(stabilize_by, float) * w
    return w


def positivity_floor(survival_probs: np.ndarray,
                     floor: float) -> tuple[np.ndarray, np.ndarray]:
    """Clip G to `floor` and return (clipped_G, violation_mask). A violation is
    a structural positivity failure — the honest fix is upstream (twist earlier),
    not in the weights."""
    G = np.asarray(survival_probs, float)
    violations = G < floor
    return np.where(violations, floor, G), violations
