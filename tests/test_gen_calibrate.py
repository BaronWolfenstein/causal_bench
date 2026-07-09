"""CFG guidance calibration: sweep guidance_scale so guided samples LAND in the
rare region R (separation AUC ~0.5) instead of overshooting past it."""
import numpy as np

from causal_bench.generative.vpsde import Schedule, gaussian_score
from causal_bench.generative.calibrate import landing_auc, calibrate_guidance


def test_landing_auc_near_half_when_indistinguishable():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((80, 1)) + 4.0
    assert abs(landing_auc(X, X.copy()) - 0.5) < 0.05      # same points -> ~0.5


def test_landing_auc_high_when_separated():
    rng = np.random.default_rng(1)
    guided = rng.standard_normal((80, 1)) + 20.0
    real_rare = rng.standard_normal((80, 1)) + 4.0
    assert landing_auc(guided, real_rare) > 0.9            # far apart -> separable


def test_calibrate_prefers_landing_scale_over_overshoot():
    sch = Schedule(n_steps=60)
    rare_mean = 4.0
    real_rare = np.random.default_rng(0).standard_normal((60, 1)) + rare_mean

    def cond(x, t):
        return gaussian_score(x, t, np.array([rare_mean]), np.eye(1), sch)

    def uncond(x, t):
        return gaussian_score(x, t, np.array([0.0]), np.eye(1), sch)

    res = calibrate_guidance(60, cond, uncond, sch, real_rare,
                             scales=(0.0, 0.5, 1.0, 2.0, 4.0), seed=1, dim=1)
    table = res["table"]
    # best_scale minimizes |auc - 0.5| over the swept table (definitional)
    assert abs(table[res["best_scale"]] - 0.5) == min(abs(a - 0.5) for a in table.values())
    # the largest scale overshoots -> strictly more separable than the landing scale
    assert table[4.0] > res["best_auc"]
