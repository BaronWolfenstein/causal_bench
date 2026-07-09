"""Twisted-SMC twist construction — the inference-time bridge for the
`smc_required` terminal.

`run_smc(x0, propagate, log_weight_fn, ...)` consumes the twist as a generic
`log_weight_fn(particles, step)`. This module builds that function from ANY score
source: the **analytic-vs-learned gate is simply which `score_fn` you pass** — the
analytic Gaussian score (`generative.vpsde.gaussian_score`) and the learned torch
net (`generative.score_net`) share the identical `score_fn(x_t, t) -> ndarray`
contract, so both plug in with no branching.

`sampling` stays independent of `generative`: the Tweedie step is reimplemented
here and the schedule enters as a plain `alphas_bar` array (pass `sch.alphas_bar`).
"""
from __future__ import annotations

import numpy as np


def tweedie_x0(particles, step, score_fn, alphas_bar):
    """Tweedie-denoised estimate ``x0_hat = (x + (1-a)·s) / sqrt(a)`` of the clean
    sample implied by a noisy particle at `step`, using the score ``s`` from
    `score_fn`. ``a = alphas_bar[step]``."""
    a = float(alphas_bar[step])
    s = np.asarray(score_fn(particles, step), dtype=float)
    return (np.asarray(particles, dtype=float) + (1.0 - a) * s) / np.sqrt(a)


def make_twist(score_fn, reward_fn, alphas_bar, *, lam: float = 1.0):
    """Build the SMC twist ``log_weight_fn(particles, step)``.

    Upweights particles whose Tweedie-denoised estimate ``x0_hat`` scores high
    under ``reward_fn`` (i.e. heads into the rare region R): the per-step twist
    potential is ``lam * reward_fn(x0_hat)``. This is reconstruction-guidance
    twisting — the reward is evaluated at the denoised estimate, not the noisy
    particle, so guidance is well-defined at every noise level.

    Note on accumulation: `run_smc` adds this increment to the particle weights
    each step and resets to uniform on resample, so between resamples the weights
    accumulate the per-step potentials. A strict telescoping-potential twist
    (Δφ across steps) would need the previous state; that is a documented
    refinement, not required for steering toward R.
    """
    def log_weight_fn(particles, step):
        x0_hat = tweedie_x0(particles, step, score_fn, alphas_bar)
        return lam * np.asarray(reward_fn(x0_hat), dtype=float)

    return log_weight_fn
