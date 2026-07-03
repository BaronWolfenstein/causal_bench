"""Phase-2 g_learner/q_learner override for TMLEIPCWEstimator.

The default path must be numerically identical to the pre-phase-2 estimator
(the whole point of the "defaults preserve behavior" guarantee); custom
Donsker learners must produce valid finite estimates.
"""
import numpy as np

from causal_bench.dgp.config import CovariateDependentCensoringConfig, DGPConfig
from causal_bench.dgp.survival import generate_data
from causal_bench.estimators.tmle_ipcw import TMLEIPCWEstimator


def _df(n=600, seed=123):
    cfg = DGPConfig(n=n,
                    censoring=CovariateDependentCensoringConfig(informativeness=0.5),
                    censoring_rate=0.25, true_tau=-0.5)
    return generate_data(cfg, rng=np.random.default_rng(seed))


def test_default_path_numerically_unchanged():
    # Regression guard: g_learner/q_learner=None must reproduce the pre-phase-2
    # estimate exactly. Baseline captured from main (1ca1e2c) before the change.
    r = TMLEIPCWEstimator(random_state=42).estimate(_df())[0]
    assert abs(r.point_estimate - 0.1654252158) < 1e-8
    assert abs(r.standard_error - 0.0369054683) < 1e-8


def test_explicit_default_logistic_matches_default():
    # Passing the default logistic proto explicitly must not change anything,
    # confirming _q_predict (predict_proba) == the old expit(decision_function).
    from sklearn.linear_model import LogisticRegression
    df = _df()
    r0 = TMLEIPCWEstimator(random_state=42).estimate(df)[0]
    r1 = TMLEIPCWEstimator(
        random_state=42,
        q_learner=LogisticRegression(max_iter=1000, C=1.0)).estimate(df)[0]
    assert abs(r0.point_estimate - r1.point_estimate) < 1e-9


def test_custom_ltb_g_q_runs_finite():
    from causal_bench.ltb import LTBClassifier, LTBRegressor
    r = TMLEIPCWEstimator(
        random_state=42,
        g_learner=LTBClassifier(random_state=0),
        q_learner=LTBRegressor(random_state=0)).estimate(_df(n=300, seed=7))[0]
    assert np.isfinite(r.point_estimate) and np.isfinite(r.standard_error)
    assert r.standard_error > 0
