import numpy as np
from causal_bench.sampling.smc import run_smc

def test_smc_recovers_a_far_target_mean():
    rng = np.random.default_rng(0)
    d, mu = 2, np.array([4.0, 0.0])        # rare region: 4 sigma out
    betas = np.linspace(0.0, 1.0, 20)      # annealing schedule
    x0 = rng.standard_normal((300, d))     # base samples

    def propagate(x, step):                # random-walk move (keeps support alive)
        return x + 0.3 * np.random.default_rng(step).standard_normal(x.shape)

    def log_weight_fn(x, step):            # incremental tilt toward N(mu, I)
        db = betas[step] - betas[step - 1]
        return db * (-0.5 * np.sum((x - mu) ** 2, axis=1) + 0.5 * np.sum(x ** 2, axis=1))

    res = run_smc(x0, propagate, log_weight_fn, n_steps=len(betas), rng=rng)
    est = np.average(res.state.particles, axis=0,
                     weights=np.exp(res.state.log_weights - res.state.log_weights.max()))
    assert np.linalg.norm(est - mu) < 0.6           # recovers the far mean
    assert res.n_resamples >= 1                      # degeneracy forced a resample
    assert res.ess_trajectory[0] >= res.ess_trajectory.min()
