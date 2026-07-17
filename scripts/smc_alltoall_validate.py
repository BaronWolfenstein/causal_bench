"""On-box global (all-to-all) distributed resample: correctness + NVLink profile.

    CUDA_VISIBLE_DEVICES=<ids> torchrun --nproc_per_node=<N> \
        scripts/smc_alltoall_validate.py --seed 7

Unlike scripts/smc_distributed_validate.py (which checks the shared-seed INDEX
invariant only), this exercises the real particle movement: builds a shared
population, shards it, runs distributed.all_to_all_resample, and asserts each
rank's output slice equals the SERIAL systematic resample of the full set — the
distributed==serial equivalence island resampling deliberately gives up (spec
§1e).  Then profiles the all_to_all NVLink throughput across N x dim (§1d/§3).
"""
from __future__ import annotations
import argparse, time
import numpy as np
import torch
import torch.distributed as dist

from causal_bench.sampling.resample import systematic_resample
from causal_bench.sampling.distributed import all_to_all_resample, distributed_ess


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)

    # ---- correctness: distributed all_to_all == serial resample --------------
    for N in (world * 4, 1 << 12, 1 << 16):
        assert N % world == 0
        Nl, d = N // world, 3
        g = np.random.default_rng(args.seed)                  # shared across ranks
        full_x = g.standard_normal((N, d)).astype(np.float32)
        full_w = g.random(N); full_w /= full_w.sum()
        serial = full_x[systematic_resample(full_w / full_w.sum(),
                                            np.random.default_rng(args.seed))]
        local_x = torch.as_tensor(full_x[rank*Nl:(rank+1)*Nl], device=dev)
        local_w = torch.as_tensor(full_w[rank*Nl:(rank+1)*Nl], device=dev, dtype=torch.float32)
        ref = serial[rank*Nl:(rank+1)*Nl]
        oks = {}
        for plan in ("cpu", "gpu"):
            out = all_to_all_resample(local_x, local_w, args.seed, rank, world,
                                      plan_on=plan).cpu().numpy()
            ok = np.array_equal(out, ref)
            flag = torch.tensor([1 if ok else 0], device=dev)
            dist.all_reduce(flag, op=dist.ReduceOp.MIN)
            oks[plan] = int(flag) == 1
        ess = distributed_ess(torch.log(local_w))
        if rank == 0:
            print(f"[N={N:7d}] all_to_all==serial  cpu-plan:{'OK' if oks['cpu'] else 'X'} "
                  f"gpu-plan:{'OK' if oks['gpu'] else 'X'}   ESS={ess:.1f}")
        assert oks["cpu"] and oks["gpu"]

    # ---- NVLink throughput profile (§1d): cpu-plan vs gpu-plan ---------------
    def bench(local_x, local_w, plan):
        for _ in range(2):
            all_to_all_resample(local_x, local_w, args.seed, rank, world, plan_on=plan)
        torch.cuda.synchronize(); dist.barrier()
        reps = 10
        t0 = time.perf_counter()
        for _ in range(reps):
            all_to_all_resample(local_x, local_w, args.seed, rank, world, plan_on=plan)
        torch.cuda.synchronize(); dist.barrier()
        return (time.perf_counter() - t0) / reps * 1e3

    if rank == 0:
        print(f"\n[profile] all_to_all resample, world={world}  (cpu-plan vs GPU-native plan)")
        print(f"  {'N':>9s} {'dim':>4s} {'payload_MB':>11s} {'cpu ms':>8s} {'gpu ms':>8s} "
              f"{'speedup':>8s} {'gpu GB/s':>9s}")
    for N in (1 << 16, 1 << 18, 1 << 20):
        if N % world:
            continue
        Nl = N // world
        for d in (1, 16, 64):
            g = np.random.default_rng(args.seed)
            full_w = g.random(N); full_w /= full_w.sum()
            local_x = torch.randn(Nl, d, device=dev, dtype=torch.float32)
            local_w = torch.as_tensor(full_w[rank*Nl:(rank+1)*Nl], device=dev, dtype=torch.float32)
            ms_cpu = bench(local_x, local_w, "cpu")
            ms_gpu = bench(local_x, local_w, "gpu")
            payload_mb = N * d * 4 / 1e6
            gbps = payload_mb / 1e3 / (ms_gpu / 1e3)
            if rank == 0:
                print(f"  {N:9d} {d:4d} {payload_mb:11.1f} {ms_cpu:8.2f} {ms_gpu:8.2f} "
                      f"{ms_cpu/ms_gpu:7.1f}x {gbps:9.1f}")

    if rank == 0:
        print("\n[note] GPU-native plan keeps the systematic resample + send/recv "
              "planning on-device (cupy CUB); only world-length split lists cross to "
              "host. The O(N*dim) NVLink all_to_all then dominates (spec §1d roofline).")
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
