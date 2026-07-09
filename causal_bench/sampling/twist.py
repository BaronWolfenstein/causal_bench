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

    Note on weight accumulation (matters for tuning ``lam``): `run_smc` ADDS this
    increment to the particle weights every step and resets to uniform only on
    resample. Because this returns an ABSOLUTE per-step potential (not a
    telescoping Δφ between steps), the *effective* twist strength between two
    resamples is ``lam × (number of un-resampled steps)`` — so ``lam``'s meaning
    is coupled to the (adaptive, data-dependent) resample cadence, and a large
    ``lam`` with infrequent resampling can accelerate weight degeneracy / ESS
    collapse. The steering DIRECTION is still correct (particles whose ``x0_hat``
    heads into R accumulate more weight); only the magnitude couples to cadence.
    Tune ``lam`` with the resample cadence in mind (or resample frequently). A
    strict telescoping-potential twist (TDS-style Δφ) is a documented follow-up:
    it needs the previous step's potential tracked across ancestry, i.e. a
    resample hook in `run_smc`.
    """
    def log_weight_fn(particles, step):
        x0_hat = tweedie_x0(particles, step, score_fn, alphas_bar)
        return lam * np.asarray(reward_fn(x0_hat), dtype=float)

    return log_weight_fn


def run_twisted_smc(x0, propagate, potential_fn, n_steps, rng, ess_frac: float = 0.5):
    """Twisted-SMC with a TDS-style **telescoping** twist potential.

    Unlike `make_twist` (which emits the ABSOLUTE per-step potential and lets
    `run_smc` accumulate it — coupling the effective strength to the resample
    cadence), this drives the loop itself so the incremental log-weight is the
    twist RATIO ``Δ = φ_t(x_t) − φ_{t−1}(x_{t−1})``. Over a resample-free run the
    weights telescope to ``φ_final − φ_initial`` (bounded, cadence-independent);
    across a resample the previous potential is reindexed by ancestry so it
    follows the survivors (the "resample hook" the plain `log_weight_fn`
    interface can't provide).

    ``potential_fn(particles, step) -> (N,)`` returns the twist potential φ_t
    (e.g. ``lam * reward(tweedie_x0(...))`` — build it however you like). Returns
    the final `SMCState`.
    """
    from .smc import SMCState, smc_step

    x = np.asarray(x0, dtype=float)
    n = len(x)
    state = SMCState(x, np.zeros(n), np.arange(n))
    prev_phi = np.asarray(potential_fn(x, 0), dtype=float)   # φ_0
    for step in range(1, n_steps):
        x = propagate(state.particles, step)
        phi = np.asarray(potential_fn(x, step), dtype=float)  # φ_t(x_t)
        log_incr = phi - prev_phi                             # the twist ratio Δ
        state = SMCState(x, state.log_weights, state.ancestry)
        state, did = smc_step(state, log_incr, rng, ess_frac)
        # carry φ_t forward; after a resample it must follow the survivors
        prev_phi = phi[state.ancestry] if did else phi
    return state
