"""Twisted-SMC twist bridge: turn any score source (analytic OR learned — same
score_fn(x_t,t) contract) into the log-weight function run_smc consumes to steer
generation toward the rare region R (the smc_required terminal)."""
import numpy as np

from causal_bench.sampling.twist import make_twist, tweedie_x0


def _abar(n):
    return np.linspace(0.9, 0.05, n)             # decreasing ᾱ over the schedule


def test_tweedie_x0_recovers_target_from_noised_mean():
    ab = _abar(50); step = 5; a = ab[step]; R = 3.0

    def score_fn(x, t):                          # analytic score of N(R,1) VP marginal
        aa = ab[t]
        return -(x - np.sqrt(aa) * R) / (aa * 1.0 + (1 - aa))

    x = np.array([[np.sqrt(a) * R]])             # sits at the noised mean
    assert abs(tweedie_x0(x, step, score_fn, ab)[0, 0] - R) < 1e-6


def test_tweedie_x0_uses_the_score_term():
    # x is OFF the noised mean with a NONZERO score, so the (1-a)*score term is
    # actually exercised (a sign/factor error there would be caught here).
    ab = _abar(50); step = 8; a = ab[step]

    def score_fn(x, t):
        return np.full_like(x, 0.5)              # constant nonzero score

    x = np.array([[2.0]])
    expected = (2.0 + (1 - a) * 0.5) / np.sqrt(a)
    assert np.isclose(tweedie_x0(x, step, score_fn, ab)[0, 0], expected)


def test_twist_upweights_particles_heading_into_R():
    ab = _abar(50); step = 10; a = ab[step]; R = 4.0

    def score_fn(x, t):
        aa = ab[t]
        return -(x - np.sqrt(aa) * R) / (aa * 1.0 + (1 - aa))

    def reward_fn(x0):
        return -((x0 - R) ** 2).sum(axis=1)      # high near R, low far away

    tw = make_twist(score_fn, reward_fn, ab, lam=1.0)
    near = np.array([[np.sqrt(a) * R]])          # denoises to ~R
    far = np.array([[np.sqrt(a) * (-8.0)]])
    assert tw(near, step)[0] > tw(far, step)[0]


def test_twist_gate_accepts_any_score_fn_contract():
    # the analytic-vs-learned "gate" is just which score_fn you pass; a stand-in
    # learned net with the same (x_t,t)->ndarray contract must plug in unchanged.
    ab = _abar(50)

    def learned_score(x, t):
        return np.zeros_like(x)

    def reward_fn(x0):
        return -(x0 ** 2).sum(axis=1)

    tw = make_twist(learned_score, reward_fn, ab)
    out = tw(np.random.default_rng(0).standard_normal((7, 3)), 5)
    assert out.shape == (7,) and np.isfinite(out).all()


# ---- tempered / bounded twist knob (variance control) ----

def test_unbounded_twist_is_linear_in_reward():
    ab = _abar(50)

    def score_fn(x, t):
        return np.zeros_like(x)

    def reward_fn(x0):
        return np.array([0.3, -0.7])

    tw = make_twist(score_fn, reward_fn, ab, lam=2.0)          # bound=None default
    assert np.allclose(tw(np.zeros((2, 1)), 5), 2.0 * np.array([0.3, -0.7]))


def test_bounded_twist_caps_magnitude_at_lam_times_bound():
    ab = _abar(50)

    def score_fn(x, t):
        return np.zeros_like(x)

    def reward_fn(x0):
        return np.array([1000.0, -1000.0, 0.0])               # extreme rewards

    tw = make_twist(score_fn, reward_fn, ab, lam=2.0, bound=1.0)
    out = tw(np.zeros((3, 1)), 5)
    assert np.all(np.abs(out) <= 2.0 * 1.0 + 1e-9)            # capped at lam*bound


def test_bounded_twist_preserves_reward_ordering():
    ab = _abar(50)

    def score_fn(x, t):
        return np.zeros_like(x)

    def reward_fn(x0):
        return np.array([2.0, 0.5, -3.0])

    tw = make_twist(score_fn, reward_fn, ab, lam=1.0, bound=1.0)
    out = tw(np.zeros((3, 1)), 5)
    assert out[0] > out[1] > out[2]                           # monotone in reward
