"""TMLE seam for a SUPPLIED clever covariate (#182, step 2).

A projected clever covariate ``E[H(A,W_true)|W_obs,A]`` cannot be delivered the way
exp32 delivers regression calibration — as a swapped data column — because the estimator
would rebuild ``H`` from that column and produce ``H(E[W])``, not ``E[H(W)]``. So the
estimator needs a seam that ACCEPTS the covariate. These pin the contract:

1. the default path is untouched (regression guard);
2. supplying the identity projection reproduces the default exactly;
3. a genuinely different projection actually moves the estimate (the seam is live).
"""
import numpy as np
import pytest

from causal_bench.dgp.survival import DGPConfig, generate_data
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator


def _data(n=600, seed=0):
    return generate_data(DGPConfig(n=n, seed=seed))


def _ate(res):
    return next(r.point_estimate for r in res if r.estimand == "ATE")


def test_default_path_unchanged_when_no_projection_supplied():
    df = _data()
    base = _ate(TMLEIPCWEstimator().estimate(df))
    again = _ate(TMLEIPCWEstimator(clever_projection=None).estimate(df))
    assert base == pytest.approx(again, rel=1e-12)
    assert np.isfinite(base)


def test_identity_projection_reproduces_the_default_exactly():
    # Supplying exactly 1/g and 1/(1-g) must be a no-op — proves the seam substitutes
    # the intended quantities and nothing else.
    df = _data()
    base = _ate(TMLEIPCWEstimator().estimate(df))

    def identity_projection(predict_g, W, A):
        g = np.clip(predict_g(W), 1e-6, 1 - 1e-6)
        return 1.0 / g, 1.0 / (1.0 - g)

    got = _ate(TMLEIPCWEstimator(clever_projection=identity_projection).estimate(df))
    assert got == pytest.approx(base, rel=1e-10)


def test_targeting_is_invariant_to_a_CONSTANT_rescaling_of_the_clever_covariate():
    # A genuine TMLE property, worth pinning: the Newton step is
    # eps = mean(H*resid)/mean(H^2), so H -> c*H gives eps -> eps/c and the update
    # eps*H1 is unchanged. Consequence for #182: a projection that differed from the
    # plug-in only by an overall factor would do NOTHING — the projection earns its
    # keep through the UNIT-VARYING Jensen gap, not through overall magnitude.
    df = _data()
    base = _ate(TMLEIPCWEstimator().estimate(df))

    def halved(predict_g, W, A):
        g = np.clip(predict_g(W), 1e-6, 1 - 1e-6)
        return 0.5 / g, 0.5 / (1.0 - g)

    got = _ate(TMLEIPCWEstimator(clever_projection=halved).estimate(df))
    assert got == pytest.approx(base, rel=1e-8)


def test_seam_is_live_and_receives_a_working_predict_g():
    # A UNIT-VARYING covariate must move the estimate, and the callback must be handed a
    # usable predict_g it can re-evaluate at modified covariates (what the real
    # projection needs, to integrate g over the calibration posterior).
    df = _data()
    base = _ate(TMLEIPCWEstimator().estimate(df))
    seen = {}

    def reshaped(predict_g, W, A):
        g = np.clip(predict_g(W), 1e-6, 1 - 1e-6)
        W2 = W.copy()
        W2[:, 0] = W2[:, 0] + 1.0                     # re-evaluate g at shifted W1
        seen["reeval_ok"] = bool(np.any(predict_g(W2) != g))
        seen["n"] = len(g)
        bump = 1.0 + 0.5 * np.abs(W[:, 0])            # unit-varying, not a constant
        return bump / g, bump / (1.0 - g)

    got = _ate(TMLEIPCWEstimator(clever_projection=reshaped).estimate(df))
    assert seen["reeval_ok"] is True                  # predict_g really is re-usable
    assert seen["n"] == len(df)
    assert np.isfinite(got)
    assert got != pytest.approx(base, rel=1e-6)       # the seam actually bites
