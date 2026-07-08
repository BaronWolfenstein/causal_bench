"""CPU-observable SMC diagnostics — the hardware-independent risks. Resample
trigger rate (dominant algorithmic risk; a property of how far R sits from base
mass, NOT the GPU), per-particle scaling (verifies batching is real), and the
lineage-multiplicity histogram (survivors' fan-out — the raw material for the
localization diagnostic's lineage-collapse component)."""
from __future__ import annotations

import time

import numpy as np


def resample_trigger_rate(result) -> float:
    steps = len(result.ess_trajectory)
    return result.n_resamples / steps if steps else 0.0


def per_particle_scaling(run_fn, ns) -> dict:
    out = {}
    for n in ns:
        t0 = time.perf_counter()
        run_fn(n)
        out[n] = time.perf_counter() - t0
    return out


def lineage_multiplicity(result) -> np.ndarray:
    """Histogram of how many descendants each particle index has at the last
    resample. Highly skewed under rare-event degeneracy (few survivors, huge
    multiplicity) — that skew IS the diagnostic signal. Returns a zero histogram
    (one bin per particle) when no resample ever fired."""
    if not result.lineage:
        return np.zeros(len(result.state.particles), dtype=int)
    last = result.lineage[-1]
    n = len(last)
    return np.bincount(last, minlength=n)
