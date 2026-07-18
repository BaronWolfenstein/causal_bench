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


def test_cfg_score_accepts_scale_schedule_callable():
    from causal_bench.generative.guidance import cfg_score
    cond = np.array([[1.0]]); uncond = np.array([[0.0]])
    # guidance_scale(t) resolved per step: uncond + gs(t)*(cond-uncond)
    assert np.isclose(cfg_score(np.zeros((1, 1)), 5, cond, uncond, lambda t: float(t))[0, 0], 5.0)
    assert np.isclose(cfg_score(np.zeros((1, 1)), 2, cond, uncond, lambda t: float(t))[0, 0], 2.0)


def test_generate_guided_runs_with_annealed_scale():
    from causal_bench.generative.guidance import generate_guided
    from causal_bench.generative.vpsde import Schedule, gaussian_score
    from causal_bench.sampling.twist import linear_anneal
    sch = Schedule(n_steps=40)

    def cond(x, t):
        return gaussian_score(x, t, np.array([4.0]), np.eye(1), sch)

    def bulk(x, t):
        return gaussian_score(x, t, np.array([0.0]), np.eye(1), sch)

    g = generate_guided(20, cond, bulk, sch, np.random.default_rng(0),
                        guidance_scale=linear_anneal(3.0, 0.5, sch.n_steps))
    assert g.shape == (20, 1) and np.isfinite(g).all()
