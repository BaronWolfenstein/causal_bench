"""Encode -> forward-noise to t_start -> reverse -> reconstruct. Emits the
per-mode reconstruction arrays the localization diagnostic consumes as recon_b /
recon_b_prime / recon_c. t_start < n_steps: a partial-noise round-trip is the
denoising-near-existing-points test (Test B), distinct from generation-from-noise
(Test B'')."""
from __future__ import annotations

import numpy as np

from .vpsde import Schedule, forward_sample


def reconstruct(x0, score_fn, sch: Schedule, t_start: int, rng):
    x = forward_sample(x0, t_start, sch, rng)
    betas = sch.betas
    for t in range(t_start, -1, -1):
        alpha_t = 1.0 - betas[t]
        mean = (x + betas[t] * score_fn(x, t)) / np.sqrt(alpha_t)
        x = mean + (np.sqrt(betas[t]) * rng.standard_normal(x.shape) if t > 0 else 0.0)
    return x


def per_mode_roundtrip(rare, common, score_fn, sch, t_start, rng):
    return (reconstruct(rare, score_fn, sch, t_start, rng),
            reconstruct(common, score_fn, sch, t_start, rng))
