"""Stand-in frozen encoder — a fixed random linear map R^in -> R^out, standing
in for MOTOR/CLMBR so the whole pipeline (incl. the #88 decoupled-encoder guard)
runs with no GPU, no model weights, no license. The real encoder implements the
same FrozenEncoder call signature and drops in unchanged."""
from __future__ import annotations

from typing import Callable

import numpy as np

FrozenEncoder = Callable[[np.ndarray], np.ndarray]


class RandomProjectionEncoder:
    def __init__(self, in_dim: int, out_dim: int, seed: int):
        W = np.random.default_rng(seed).standard_normal((in_dim, out_dim))
        self.W = W / np.linalg.norm(W, axis=0, keepdims=True)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X) @ self.W


def make_encoder_pair(in_dim: int, out_dim: int) -> tuple[FrozenEncoder, FrozenEncoder]:
    """E_gen (generation encoder) and a DECOUPLED E_eval for the metric-hacking
    guard — different seeds => genuinely different geometries."""
    return (RandomProjectionEncoder(in_dim, out_dim, seed=11),
            RandomProjectionEncoder(in_dim, out_dim, seed=29))
