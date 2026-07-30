"""Classifier-free guidance in embedding space: interpolate/extrapolate between
the conditional and unconditional scores. Clean in embedding space (ELF) — no
separate classifier. Output is `rare_guided`, the held-out generation the Test B″
CFG-landing check consumes. When CFG's structural bias can't land in R, the
Step-3 twisted-SMC reranker is the fix (terminal smc_required)."""
from __future__ import annotations

import numpy as np

from .vpsde import Schedule, ddpm_reverse


def cfg_score(x_t, t, cond_score, uncond_score, guidance_scale):
    # guidance_scale may be a scalar OR a schedule callable(step)->float, so the
    # CFG strength can be annealed with the noise level (e.g. sampling.linear_anneal):
    # weak while noisy (t large, x0 estimate unreliable), sharp as denoising finishes.
    gs = guidance_scale(t) if callable(guidance_scale) else guidance_scale
    return uncond_score + gs * (cond_score - uncond_score)


def generate_guided(n, cond_score_fn, uncond_score_fn, sch: Schedule, rng,
                    guidance_scale=3.0, dim: int = 1) -> np.ndarray:
    x_T = rng.standard_normal((n, dim))

    def guided_score_fn(x, t):
        return cfg_score(x, t, cond_score_fn(x, t), uncond_score_fn(x, t), guidance_scale)

    return ddpm_reverse(x_T, guided_score_fn, sch, rng)
