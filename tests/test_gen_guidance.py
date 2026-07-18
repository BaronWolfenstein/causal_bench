import numpy as np
from causal_bench.generative.vpsde import Schedule, gaussian_score
from causal_bench.generative.guidance import generate_guided

def test_cfg_pulls_samples_toward_the_conditioned_region():
    sch = Schedule(n_steps=300); rng = np.random.default_rng(0)
    cond = lambda x, t: gaussian_score(x, t, np.array([4.0]), np.eye(1), sch)   # rare
    uncond = lambda x, t: gaussian_score(x, t, np.array([0.0]), np.eye(1), sch) # bulk
    guided = generate_guided(2000, cond, uncond, sch, rng, guidance_scale=3.0)
    unguided = generate_guided(2000, cond, uncond, sch, rng, guidance_scale=0.0)
    assert guided.mean() > unguided.mean()          # CFG shifts toward rare
    assert guided.mean() > 2.0                       # meaningfully into R
