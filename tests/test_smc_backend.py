import numpy as np
import pytest

from causal_bench.sampling.backend import array_namespace, asarray, to_numpy


def test_cpu_namespace_is_numpy_and_roundtrips():
    assert array_namespace("cpu") is np
    x = asarray([1.0, 2.0, 3.0], "cpu")
    assert isinstance(x, np.ndarray)
    assert np.array_equal(to_numpy(x), np.array([1.0, 2.0, 3.0]))


def test_unknown_device_raises():
    with pytest.raises(ValueError):
        array_namespace("tpu")


def test_run_smc_cpu_device_returns_numpy():
    from causal_bench.sampling.smc import run_smc
    rng = np.random.default_rng(0)
    betas = np.linspace(0, 1, 10)
    mu = np.array([2.0])
    x0 = rng.standard_normal((50, 1))
    prop = lambda x, s: x + 0.3 * np.random.default_rng(s).standard_normal(x.shape)
    lw = lambda x, s: (betas[s] - betas[s - 1]) * (
        -0.5 * ((x - mu) ** 2).sum(1) + 0.5 * (x ** 2).sum(1))
    res = run_smc(x0, prop, lw, len(betas), rng, device="cpu")
    assert isinstance(res.state.particles, np.ndarray)
    assert isinstance(res.state.log_weights, np.ndarray)


def test_get_namespace_returns_numpy_for_numpy_arrays():
    from causal_bench.sampling.backend import get_namespace
    assert get_namespace(np.zeros(3)) is np
    assert get_namespace(np.zeros(3), np.ones(2)) is np
