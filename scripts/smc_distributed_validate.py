"""On-box multi-GPU SMC validation. Launch with:
    CUDA_VISIBLE_DEVICES=<free,NVLink-adjacent ids> \
    torchrun --nproc_per_node=<N> scripts/smc_distributed_validate.py --seed 7
Asserts distributed indices == the single-rank numpy oracle byte-for-byte
(shared-seed systematic-resample invariant, gathered via all_gather), and
that the fused ESS all_reduce (ess_num, ess_den) stays finite. This does NOT
compare a full distributed SMC run against a single-GPU reference — that
requires all_to_all particle redistribution and is deferred; island
resampling (this script's low-comm default) never needs it."""
from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.distributed as dist

from causal_bench.sampling.resample import systematic_resample
from causal_bench.sampling.sharded import sharded_systematic_resample


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n", type=int, default=1 << 16)
    args = ap.parse_args()

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(rank)

    # Every rank holds the same all-gathered weights (simulated here from a
    # shared seed; in the full loop this is an all_gather of local weights).
    w = np.random.default_rng(args.seed).random(args.n)
    w = w / w.sum()

    # Shared-seed systematic indices — must be byte-identical across ranks and
    # equal to the single-rank oracle. This is the decisive invariant.
    idx = systematic_resample(w, np.random.default_rng(args.seed))
    oracle = sharded_systematic_resample(w, k=world, seed=args.seed)

    assert args.n % world == 0, (
        f"--n ({args.n}) must be divisible by world size ({world}) so all_gather "
        f"buffers are equal-sized; pick n divisible by nproc_per_node."
    )

    # Each rank owns a contiguous slice; concatenation across ranks == oracle.
    my_slice = np.array_split(idx, world)[rank]

    # NCCL collectives require every rank to participate — no rank-conditional
    # collectives below, or non-zero ranks skip the call and rank 0 hangs forever.
    gathered = [torch.empty(len(my_slice), dtype=torch.int64, device="cuda")
                for _ in range(world)]
    dist.all_gather(gathered, torch.as_tensor(my_slice, device="cuda"))

    # weight-finiteness fused into the ESS reduce (spec §1b)
    ess_num = torch.tensor(float(w.sum() ** 2), device="cuda")
    ess_den = torch.tensor(float((w ** 2).sum()), device="cuda")
    dist.all_reduce(ess_num)
    dist.all_reduce(ess_den)

    if rank == 0:
        full = torch.cat(gathered).cpu().numpy()
        assert np.array_equal(full, oracle), "distributed indices != numpy oracle"
        assert torch.isfinite(ess_num) and torch.isfinite(ess_den)
        print(f"[rank0] world={world} n={args.n}: distributed==oracle OK")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
