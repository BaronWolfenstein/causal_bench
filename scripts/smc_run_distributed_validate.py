"""On-box §1e decisive test: the FULL distributed SMC loop == single-device
serial run_smc, on the same seed, to numerical tolerance.

    CUDA_VISIBLE_DEVICES=<ids> torchrun --nproc_per_node=<N> \
        scripts/smc_run_distributed_validate.py --seed 7

Runs run_smc_distributed(mode='global') across ranks, gathers the final
particles, and compares to a single-process serial run_smc reference (same
propagate/reweight/seed). Also runs mode='island' to characterize the low-comm
approximation's bias vs the exact global resample.
"""
from __future__ import annotations
import argparse
import numpy as np
import torch
import torch.distributed as dist

from causal_bench.sampling.smc import run_smc
from causal_bench.sampling.smc_distributed import run_smc_distributed


def propagate(x, step):
    return x * 0.999 + 0.01                       # per-particle affine drift (np/torch identical)


def log_weight_fn(x, step):
    if isinstance(x, torch.Tensor):
        return -0.02 * (x * x).sum(dim=1) * (1 + 0.1 * step)
    return -0.02 * (x * x).sum(axis=1) * (1 + 0.1 * step)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--steps", type=int, default=30)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    N = args.n or world * 512
    assert N % world == 0
    Nl, d = N // world, 4
    full_x0 = np.random.default_rng(123).standard_normal((N, d))          # shared on all ranks

    def run_dist(mode):
        xl = torch.as_tensor(full_x0[rank*Nl:(rank+1)*Nl], device=dev, dtype=torch.float64)
        xf, rs = run_smc_distributed(xl, propagate, log_weight_fn, args.steps, args.seed,
                                     ess_frac=0.5, rank=rank, world=world, mode=mode)
        bufs = [torch.empty_like(xf) for _ in range(world)]
        dist.all_gather(bufs, xf.contiguous())
        return torch.cat(bufs).cpu().numpy(), rs

    dist_global, rs_g = run_dist("global")
    dist_island, rs_i = run_dist("island")

    if rank == 0:
        serial = run_smc(full_x0, propagate, log_weight_fn, args.steps,
                         np.random.default_rng(args.seed), ess_frac=0.5, device="cpu")
        ref = serial.state.particles
        gmax = np.abs(dist_global - ref).max()
        imax = np.abs(dist_island - ref).max()
        steps_ok = rs_g == serial.resample_steps
        print(f"[world={world}] resamples: serial={serial.resample_steps}")
        print(f"                distributed(global)={rs_g}  match={steps_ok}")
        print(f"[GLOBAL] max|dist - serial| = {gmax:.2e}  -> "
              f"{'distributed==serial OK' if gmax < 1e-6 and steps_ok else 'MISMATCH'}")
        print(f"[ISLAND] max|dist - serial| = {imax:.2e}  (low-comm approx bias; "
              f"expected nonzero, no all_to_all)")
        assert gmax < 1e-6 and steps_ok
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
