"""exp45: multi-period app co-intervention estimators (g-estimation / ICE / naive)."""
import numpy as np

from causal_bench.validation.longitudinal_cointervention import (
    g_estimation, g_estimation_effect_mod, ice_contrast, naive_effects, sim_longitudinal,
    true_values,
)


def test_g_estimation_recovers_blips():
    """g-estimation recovers the structural blips ψ (A0 blip = total effect not through A1,
    incl. the L1-mediated path)."""
    tv = true_values(psi0=0.3, psi1=0.5)
    d = sim_longitudinal(10000, psi0=0.3, psi1=0.5, seed=1)
    p0, p1 = g_estimation(d)
    assert abs(p1 - tv["blip_A1"]) < 0.06
    assert abs(p0 - tv["blip_A0"]) < 0.06


def test_ice_recovers_regime_contrast():
    """Sequential regression (ICE) recovers E[Y_{1,1}]−E[Y_{0,0}]."""
    tv = true_values(psi0=0.3, psi1=0.5)
    d = sim_longitudinal(10000, psi0=0.3, psi1=0.5, seed=2)
    assert abs(ice_contrast(d) - tv["contrast"]) < 0.06


def test_naive_is_biased_low_for_A0():
    """Naive adjusts the mediator/feedback confounder L1 → blocks A0's mediated path → its A0
    coefficient is well below the true A0 blip."""
    tv = true_values(psi0=0.3, psi1=0.5)
    d = sim_longitudinal(10000, psi0=0.3, psi1=0.5, seed=3)
    naive_a0 = naive_effects(d)["A0"]
    assert naive_a0 < tv["blip_A0"] - 0.3           # substantially biased low (blocks the path)


def test_g_estimation_recovers_effect_modification():
    """The effect-mod arm: g-estimation recovers the A1 blip's modification by L1 — the object a
    marginal regime-mean does not surface."""
    d = sim_longitudinal(10000, psi0=0.3, psi1=0.5, effect_mod=0.4, seed=4)
    em = g_estimation_effect_mod(d)
    assert abs(em["effect_mod"] - 0.4) < 0.06
    assert abs(em["psi1"] - 0.5) < 0.06
