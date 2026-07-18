"""run_twisted_smc: TDS-style telescoping twist. The incremental log-weight is
Δ = φ_t(x_t) − φ_{t−1}(x_{t−1}), so over a resample-free run the accumulated
weight telescopes to φ_final − φ_initial (bounded, cadence-independent), unlike
make_twist's absolute per-step potential."""
import numpy as np

from causal_bench.sampling.twist import run_twisted_smc


def test_telescoping_no_resample_accumulates_phi_final_minus_initial():
    N, n_steps = 5, 8
    x0 = np.random.default_rng(0).standard_normal((N, 1))

    def propagate(x, step):
        return x                                  # frozen particles

    def potential_fn(x, step):
        return step * x[:, 0]                     # φ depends on step AND particle

    # ess_frac=0 => resampling never triggers => weights telescope exactly
    state = run_twisted_smc(x0, propagate, potential_fn, n_steps,
                            np.random.default_rng(1), ess_frac=0.0)
    expected = (n_steps - 1) * x0[:, 0] - 0 * x0[:, 0]     # φ_{T} − φ_0
    assert np.allclose(state.log_weights, expected)


def test_twisted_smc_with_resampling_stays_finite_and_conserves_N():
    N, n_steps = 64, 20
    rng = np.random.default_rng(2)
    x0 = rng.standard_normal((N, 1))

    def propagate(x, step):
        return x + 0.1 * np.random.default_rng(step).standard_normal(x.shape)

    R = 3.0

    def potential_fn(x, step):
        return -((x[:, 0] - R) ** 2)              # reward heads toward R

    state = run_twisted_smc(x0, propagate, potential_fn, n_steps, rng, ess_frac=0.5)
    assert state.particles.shape == (N, 1)
    assert np.isfinite(state.log_weights).all()


def test_telescoping_with_moving_particles_no_resample():
    # stronger than the frozen case: with a MOVING deterministic trajectory and no
    # resample, the weights must telescope to φ(x_T) − φ(x_0) along that trajectory.
    N, n_steps = 5, 6
    x0 = np.random.default_rng(3).standard_normal((N, 1))

    def propagate(x, step):
        return x + 1.0                            # deterministic: +1 each step

    def potential_fn(x, step):
        return x[:, 0] ** 2                        # φ depends on the CURRENT position

    state = run_twisted_smc(x0, propagate, potential_fn, n_steps,
                            np.random.default_rng(4), ess_frac=0.0)
    xT = x0[:, 0] + (n_steps - 1)                  # trajectory end after steps 1..T
    expected = xT ** 2 - x0[:, 0] ** 2            # φ(x_T) − φ(x_0)
    assert np.allclose(state.log_weights, expected)
