"""Twisted-diffusion SMC loop. Propagation and twist evaluation are per-particle
(embarrassingly parallel); the only synchronization is the ESS reduction inside
the adaptive resample. Ancestry is tracked so lineage collapse is observable."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .resample import should_resample, systematic_resample
from .weights import normalize_log_weights


@dataclass
class SMCState:
    particles: np.ndarray        # (N, d)
    log_weights: np.ndarray      # (N,)
    ancestry: np.ndarray         # (N,) ancestor index at the last resample


@dataclass
class SMCResult:
    state: SMCState
    ess_trajectory: np.ndarray   # ESS after each step
    resample_steps: list         # step indices where the barrier fired
    lineage: list                # ancestor-index vectors (int32) per resample

    @property
    def n_resamples(self) -> int:
        return len(self.resample_steps)


def smc_step(state: SMCState, log_incr: np.ndarray, rng, ess_frac: float = 0.5):
    """One reweight → (adaptive) resample. Returns (new_state, resampled?)."""
    log_w = state.log_weights + log_incr
    if should_resample(log_w, ess_frac):
        w, _ = normalize_log_weights(log_w)
        idx = systematic_resample(w, rng)
        new = SMCState(
            particles=state.particles[idx],
            log_weights=np.zeros(len(idx)),          # reset to uniform post-resample
            ancestry=idx,
        )
        return new, True
    return SMCState(state.particles, log_w, state.ancestry), False


def run_smc(x0, propagate, log_weight_fn, n_steps, rng, ess_frac: float = 0.5,
            device: str = "cpu"):
    from .weights import kish_ess
    from .backend import array_namespace, to_numpy
    xp = array_namespace(device)
    x0 = xp.asarray(x0, dtype=float)          # float coercion, on-device
    state = SMCState(x0, xp.zeros(len(x0)), xp.arange(len(x0)))
    ess, resample_steps, lineage = [], [], []
    for step in range(1, n_steps):
        state = SMCState(propagate(state.particles, step),
                         state.log_weights, state.ancestry)
        state, did = smc_step(state, log_weight_fn(state.particles, step),
                              rng, ess_frac)
        ess.append(kish_ess(state.log_weights))
        if did:
            resample_steps.append(step)
            lineage.append(to_numpy(state.ancestry).astype(np.int32))
    # hand host numpy back to callers regardless of device (CPU path unchanged)
    final = SMCState(to_numpy(state.particles), to_numpy(state.log_weights),
                     to_numpy(state.ancestry))
    return SMCResult(final, np.asarray(ess), resample_steps, lineage)
