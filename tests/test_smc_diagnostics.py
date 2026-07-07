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

def test_per_particle_cost_scales_roughly_linearly():
    scaling = per_particle_scaling(lambda n: _toy_run(n), ns=[50, 100, 200])
    # cost/particle should be roughly flat (linear total), not growing with N
    per = [scaling[n] / n for n in (50, 100, 200)]
    assert max(per) / min(per) < 3.0

def test_lineage_multiplicity_sums_to_n():
    r = _toy_run(100)
    if r.n_resamples:
        mult = lineage_multiplicity(r)
        assert int(mult.sum()) == 100
