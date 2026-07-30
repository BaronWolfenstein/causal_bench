"""Distributed (multi-GPU / NCCL) SMC loop — wires distributed.py's collectives
into the run_smc barrier (spec §1b/§1d/§1e).

Per-rank propagate + reweight (embarrassingly parallel), a single global ESS
``all_reduce`` to decide the adaptive resample, then either:
  * mode="global"  -> all_gather(w) + shared-rng systematic + all_to_all: exactly
                      reproduces the single-device serial run_smc (the §1e test).
  * mode="island"  -> each rank resamples its own slice, no particle movement:
                      the low-communication default (§1d), a small bias/variance
                      approximation to the global resample.

The rng is a SHARED numpy Generator (identical seed on every rank); the resample
uniform therefore advances in lockstep with — and matches — the serial stream.
"""
from __future__ import annotations

import numpy as np


def run_smc_distributed(x0_local, propagate, log_weight_fn, n_steps, seed,
                        ess_frac: float = 0.5, rank: int = 0, world: int = 1,
                        group=None, mode: str = "global", plan_on: str = "gpu",
                        dedup: bool = False):
    """x0_local: this rank's (n_local, d) torch tensor.  propagate(x, step) and
    log_weight_fn(x, step) are per-particle.  Returns (final_local_particles,
    resample_steps)."""
    import torch
    import torch.distributed as dist
    from .distributed import distributed_ess, all_to_all_resample
    from .resample import systematic_resample

    dev = x0_local.device
    rng = np.random.default_rng(seed)                 # shared across ranks
    x = x0_local
    n_local = int(x.shape[0]); N = n_local * world
    log_w = torch.zeros(n_local, device=dev, dtype=x.dtype)
    active = world > 1 and dist.is_initialized()
    resample_steps = []

    for step in range(1, n_steps):
        x = propagate(x, step)
        log_w = log_w + log_weight_fn(x, step)
        ess = distributed_ess(log_w, group)           # global ESS (all_reduce; local if world=1)
        if ess < ess_frac * N:
            m = log_w.max()
            if active:
                dist.all_reduce(m, op=dist.ReduceOp.MAX, group=group)   # global max scaling
            w = torch.exp(log_w - m)                  # consistently scaled across ranks
            if mode == "global" and active:
                # dedup = ancestor-index indirection (§1d): ship each surviving
                # ancestor once, replicate locally — big win under degeneracy.
                x = all_to_all_resample(x, w, rng, rank, world, group, plan_on, dedup)
            else:
                # island (any world), OR global on a single rank: this rank's
                # slice is resampled locally.  For world==1 the slice IS the whole
                # population, so this equals the serial global resample.
                wl = (w / w.sum()).detach().cpu().numpy()
                idx_l = systematic_resample(wl, rng)
                x = x[torch.as_tensor(idx_l, device=dev, dtype=torch.long)]
            log_w = torch.zeros(n_local, device=dev, dtype=x.dtype)
            resample_steps.append(step)
    return x, resample_steps
