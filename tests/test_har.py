import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge

from causal_bench.har import HARClassifier, HARRegressor, har_kernel


def test_kernel_hand_computed():
    # p=1, training points {0, 1}. K(a,b) = sum_i 2^{#coords where X_i <= min(a,b)}
    X = np.array([[0.0], [1.0]])
    K = har_kernel(np.array([[2.0]]), np.array([[3.0]]), X)
    # both training points dominated: 2^1 + 2^1 = 4
    assert K.shape == (1, 1) and K[0, 0] == 4.0
    K2 = har_kernel(np.array([[0.5]]), np.array([[3.0]]), X)
    # min(0.5,3)=0.5: only X=0 dominated -> 2^1 + 2^0 = 3
    assert K2[0, 0] == 3.0


def test_regressor_clone_and_beats_ridge_on_smooth_nonlinear():
    rng = np.random.default_rng(0)
    Xtr = rng.uniform(-2, 2, size=(400, 3))
    Xte = rng.uniform(-2, 2, size=(400, 3))
    f = lambda X: np.tanh(2 * X[:, 0]) + 0.5 * X[:, 1] ** 2
    ytr = f(Xtr) + rng.normal(0, 0.3, 400)
    yte = f(Xte)

    har = clone(HARRegressor(random_state=0)).fit(Xtr, ytr)
    ridge = Ridge(alpha=1.0).fit(Xtr, ytr)
    mse_har = np.mean((har.predict(Xte) - yte) ** 2)
    mse_ridge = np.mean((ridge.predict(Xte) - yte) ** 2)
    assert mse_har < mse_ridge


def test_classifier_proba_clipped_and_shaped():
    rng = np.random.default_rng(1)
    X = rng.uniform(-2, 2, size=(300, 3))
    y = rng.binomial(1, 0.2 + 0.6 * (np.tanh(X[:, 0]) > 0))
    m = HARClassifier(random_state=0).fit(X, y)
    proba = m.predict_proba(X)
    assert proba.shape == (300, 2)
    assert proba[:, 1].min() >= 1e-6 and proba[:, 1].max() <= 1 - 1e-6
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(m.classes_) == [0, 1]
    assert set(np.unique(m.predict(X))) <= {0, 1}


def test_zero_jitter_zero_lambda_rank_deficient_is_finite():
    # duplicate rows -> rank-deficient kernel; jitter=0 + lambda grid touching 0
    X = np.tile(np.array([[0.5, -0.5, 1.0]]), (30, 1))
    y = np.zeros(30)
    m = HARRegressor(lambdas=[0.0, 1.0], jitter=0.0, random_state=0).fit(X, y)
    assert np.all(np.isfinite(m.alpha_))
    assert np.all(np.isfinite(m.predict(X[:3])))
