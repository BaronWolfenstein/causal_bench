"""Off-NCCL test of the distributed SMC loop: with world=1 the single rank owns
the whole population, so run_smc_distributed(mode='global') must equal the serial
run_smc byte-for-byte on the same seed.  The true multi-rank distributed==serial
equivalence (all_reduce barrier + all_to_all) is exercised on-box by
scripts/smc_run_distributed_validate.py.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from causal_bench.sampling.smc import run_smc
from causal_bench.sampling.smc_distributed import run_smc_distributed


def _propagate(x, step):
    return x * 0.999 + 0.01


def _log_weight_fn(x, step):
    if isinstance(x, torch.Tensor):
        return -0.02 * (x * x).sum(dim=1) * (1 + 0.1 * step)
    return -0.02 * (x * x).sum(axis=1) * (1 + 0.1 * step)


def test_world1_global_equals_serial():
    N, d, steps, seed = 256, 4, 30, 7
    x0 = np.random.default_rng(123).standard_normal((N, d))
    serial = run_smc(x0, _propagate, _log_weight_fn, steps,
                     np.random.default_rng(seed), ess_frac=0.5, device="cpu")
    xl = torch.as_tensor(x0, dtype=torch.float64)          # cpu, world=1
    xf, rs = run_smc_distributed(xl, _propagate, _log_weight_fn, steps, seed,
                                 ess_frac=0.5, rank=0, world=1, mode="global")
    assert rs == serial.resample_steps
    assert np.abs(xf.numpy() - serial.state.particles).max() < 1e-9
