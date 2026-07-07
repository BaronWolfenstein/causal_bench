import numpy as np
from causal_bench.sampling.smc import run_smc
from causal_bench.sampling.diagnostics import (
    resample_trigger_rate, per_particle_scaling, lineage_multiplicity)

def _toy_run(n, seed=0):
    rng = np.random.default_rng(seed)
    mu = np.array([4.0]); betas = np.linspace(0, 1, 15)
    x0 = rng.standard_normal((n, 1))
    prop = lambda x, s: x + 0.3 * np.random.default_rng(s).standard_normal(x.shape)
    lw = lambda x, s: (betas[s]-betas[s-1]) * (-0.5*((x-mu)**2).sum(1) + 0.5*(x**2).sum(1))
    return run_smc(x0, prop, lw, len(betas), rng)

def test_trigger_rate_in_unit_interval():
    r = resample_trigger_rate(_toy_run(200))
    assert 0.0 <= r <= 1.0

def test_per_particle_cost_is_not_superlinear():
    # The real risk this guards is an accidental O(N^2) Python loop. A wall-clock
    # cost/particle "flatness" test is unreliable at sub-ms runtimes (fixed
    # overhead dominates, making cost/particle fall with N). Instead assert total
    # cost does not grow super-linearly at N large enough for O(N) work to matter:
    # size ratio 16x -> O(N) ~16x, O(N^2) ~256x; allow 64x for overhead/noise.
    ns = [200, 800, 3200]
    _toy_run(ns[0])                       # warm up numpy/JIT before timing
    scaling = per_particle_scaling(lambda n: _toy_run(n), ns=ns)
    ratio = scaling[ns[-1]] / scaling[ns[0]]
    size_ratio = ns[-1] / ns[0]
    assert ratio < size_ratio * 4, (
        f"total cost grew {ratio:.1f}x for a {size_ratio:.0f}x particle increase "
        "— super-linear (accidental O(N^2)?)")

def test_lineage_multiplicity_sums_to_n():
    r = _toy_run(100)
    if r.n_resamples:
        mult = lineage_multiplicity(r)
        assert int(mult.sum()) == 100
