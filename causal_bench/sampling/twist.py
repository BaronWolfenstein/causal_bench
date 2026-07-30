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


def make_twist(score_fn, reward_fn, alphas_bar, *, lam: float = 1.0, bound=None):
    """Build the SMC twist potential ``φ_t(x_t) = twist(reward_fn(x0_hat))``.

    Upweights particles whose Tweedie-denoised estimate ``x0_hat`` scores high
    under ``reward_fn`` (i.e. heads into the rare region R). The reward is
    evaluated at the denoised estimate, not the noisy particle, so guidance is
    well-defined at every noise level (reconstruction-guidance twisting).

    ``lam`` is the **temperature** on the twist (β): larger sharpens toward the
    rare mode, smaller softens. It may be a scalar OR a **schedule** ``lam(step)
    -> float`` — annealing ``β_t`` with the noise level (weak/soft early, when the
    Tweedie estimate ``x0_hat`` is unreliable and a sharp twist would spike weight
    variance → ESS floor; strong/sharp late, when ``x0_hat`` is trustworthy). This
    defers conditioning fidelity to when it can be trusted rather than losing it.
    See `linear_anneal`.

    ``bound`` is a **variance-control** knob (also scalar OR ``bound(step)``). When
    ``None`` (default) the potential is the raw ``lam * reward`` (unbounded). When
    set to ``b > 0`` the reward is squashed through a saturating transform
    ``lam·b·tanh(reward/b)``, capping the per-step potential to ``[-lam·b, lam·b]``
    so no single particle's weight can blow up — a cheap guard against ESS
    collapse. It is ``≈ lam·reward`` in the small-reward (linear) regime and
    preserves reward ordering.

    Two consumers:
    - `run_smc(..., log_weight_fn=make_twist(...))` — accumulate mode. Note this
      returns an ABSOLUTE per-step potential, so between resamples the effective
      strength is ``lam × (un-resampled steps)`` (couples to the adaptive resample
      cadence). ``bound`` and frequent resampling both mitigate the resulting
      weight degeneracy.
    - `run_twisted_smc(..., potential_fn=make_twist(...))` — the TDS-style
      telescoping loop, which emits the ratio ``φ_t − φ_{t−1}`` and is
      cadence-independent (the principled fix). ``make_twist``'s output is a
      valid ``potential_fn`` there since it already returns ``φ_t(x_t)``.
    """
    def twist_potential(particles, step):
        lam_t = lam(step) if callable(lam) else lam          # anneal β_t if scheduled
        bound_t = bound(step) if callable(bound) else bound
        x0_hat = tweedie_x0(particles, step, score_fn, alphas_bar)
        r = np.asarray(reward_fn(x0_hat), dtype=float)
        if bound_t is None:
            return lam_t * r
        return lam_t * bound_t * np.tanh(r / bound_t)        # saturating cap ±lam·bound

    return twist_potential


def linear_anneal(lo: float, hi: float, n_steps: int):
    """A schedule callable ``step -> value`` interpolating ``lo → hi`` linearly
    over ``[0, n_steps-1]`` (clamped outside). Use as
    ``make_twist(..., lam=linear_anneal(lo, hi, n_steps))`` to ramp the twist
    temperature β_t with the noise level. Set ``lo``/``hi`` to match your loop's
    noise convention (e.g. weak → strong as denoising progresses)."""
    span = max(n_steps - 1, 1)

    def sched(step):
        f = min(max(step / span, 0.0), 1.0)
        return lo + f * (hi - lo)

    return sched


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
