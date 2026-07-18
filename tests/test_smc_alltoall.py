"""Off-box unit test for the distributed all-to-all resample PLAN (pure numpy).

The NCCL wiring is exercised on-box by scripts/smc_alltoall_validate.py; here we
prove the index math — global shared-seed indices + per-rank send/recv splits +
the place-order permutation — reconstructs the SERIAL systematic resample for
every rank count, with no GPU/process-group needed.
"""
import numpy as np
import pytest

from causal_bench.sampling.resample import systematic_resample
from causal_bench.sampling.distributed import (
    simulate_all_to_all, plan_all_to_all, global_indices,
)


@pytest.mark.parametrize("world", [1, 2, 4, 8])
@pytest.mark.parametrize("N", [8, 1024, 65536])
def test_all_to_all_equals_serial(world, N):
    if N % world:
        pytest.skip("N must divide evenly across ranks")
    rng = np.random.default_rng(0)
    X = rng.standard_normal((N, 5)).astype(np.float32)
    w = rng.random(N); w /= w.sum()
    serial = X[systematic_resample(w / w.sum(), np.random.default_rng(7))]
    dist = simulate_all_to_all(X, w, world, seed=7)
    assert np.array_equal(serial, dist)


def test_plan_splits_are_consistent():
    """Every row sent by some rank is received by exactly one rank: the global
    send matrix is the transpose of the receive matrix."""
    world, N = 4, 4096
    rng = np.random.default_rng(1)
    w = rng.random(N); w /= w.sum()
    idx = global_indices(w, seed=3)
    Nl = N // world
    send = np.zeros((world, world), int)   # send[s, d]
    recv = np.zeros((world, world), int)   # recv[r, s]
    for r in range(world):
        in_split, out_split, _, _ = plan_all_to_all(idx, r, world, Nl)
        send[r, :] = in_split              # r sends in_split[d] to d
        recv[r, :] = out_split             # r receives out_split[s] from s
    assert np.array_equal(send, recv.T)    # sends match receives
    assert send.sum() == N                 # every output particle fetched once
