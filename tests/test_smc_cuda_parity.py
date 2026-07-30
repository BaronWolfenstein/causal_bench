import numpy as np
import pytest


def _run(xp, device, rng):
    from causal_bench.sampling.smc import run_smc
    betas = np.linspace(0, 1, 12)
    mu = xp.asarray([2.0])
    x0 = xp.asarray(np.random.default_rng(0).standard_normal((64, 1)))

    def propagate(x, s):
        step_xp = type(x).__module__.split(".")[0]
        noise = np.random.default_rng(s).standard_normal(x.shape)
        return x + 0.3 * (xp.asarray(noise) if step_xp == "cupy" else noise)

    def log_weight_fn(x, s):
        return (betas[s] - betas[s - 1]) * (
            -0.5 * ((x - mu) ** 2).sum(1) + 0.5 * (x ** 2).sum(1))

    return run_smc(x0, propagate, log_weight_fn, len(betas), rng, device=device)


def test_cuda_matches_cpu_on_same_seed():
    cp = pytest.importorskip("cupy")            # skips off-box; runs on the A100 box
    cpu = _run(np, "cpu", np.random.default_rng(7))
    cuda = _run(cp, "cuda", np.random.default_rng(7))
    assert np.allclose(cpu.state.particles, cuda.state.particles, atol=1e-6)
    assert np.allclose(cpu.ess_trajectory, cuda.ess_trajectory, atol=1e-6)
    assert cpu.resample_steps == cuda.resample_steps
