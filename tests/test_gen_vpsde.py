import numpy as np
from causal_bench.generative.vpsde import (
    Schedule, forward_sample, gaussian_score, tweedie_denoise, ddpm_reverse, alpha_bar)

def test_forward_marginal_variance_grows_to_one():
    sch = Schedule(n_steps=200)
    x0 = np.zeros((2000, 1))
    xT = forward_sample(x0, sch.n_steps - 1, sch, np.random.default_rng(0))
    assert 0.7 < xT.var() < 1.3                       # ~ N(0,1) at the end

def test_tweedie_denoise_recovers_mean():
    sch = Schedule(n_steps=200); mu = np.array([5.0]); cov = np.eye(1)
    rng = np.random.default_rng(0)
    x0 = mu + rng.standard_normal((4000, 1))
    t = 100
    xt = forward_sample(x0, t, sch, rng)
    score = gaussian_score(xt, t, mu, cov, sch)
    x0_hat = tweedie_denoise(xt, t, score, sch)
    assert abs(x0_hat.mean() - 5.0) < 0.15

def test_ddpm_reverse_recovers_far_target():
    sch = Schedule(n_steps=300); mu = np.array([4.0]); cov = np.eye(1)
    rng = np.random.default_rng(0)
    xT = rng.standard_normal((3000, 1))
    score_fn = lambda x, t: gaussian_score(x, t, mu, cov, sch)
    x0 = ddpm_reverse(xT, score_fn, sch, rng)
    assert abs(x0.mean() - 4.0) < 0.5                 # generation lands on the target
