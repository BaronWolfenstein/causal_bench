import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, LogisticRegression

from causal_bench.ltb import LTBClassifier, LTBRegressor


def _step_data(n, seed, noise=0.3):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-2, 2, size=(n, 3))
    y = (X[:, 0] > 0).astype(float) + 0.3 * (X[:, 1] > 0.5) + rng.normal(0, noise, n)
    return X, y


def test_regressor_clone_and_shapes():
    m = LTBRegressor(max_blocks=5, random_state=0)
    m2 = clone(m)  # raises if __init__ mutates params
    X, y = _step_data(300, 0)
    m2.fit(X, y)
    assert m2.predict(X[:10]).shape == (10,)


def test_regressor_beats_linear_on_step_function():
    Xtr, ytr = _step_data(800, 1)
    Xte, yte = _step_data(800, 2)
    ltb = LTBRegressor(random_state=0).fit(Xtr, ytr)
    lin = LinearRegression().fit(Xtr, ytr)
    mse_ltb = np.mean((ltb.predict(Xte) - yte) ** 2)
    mse_lin = np.mean((lin.predict(Xte) - yte) ** 2)
    assert mse_ltb < mse_lin


def test_classifier_proba_valid_and_beats_logistic_on_step():
    rng = np.random.default_rng(3)
    X = rng.uniform(-2, 2, size=(1000, 3))
    p = 0.15 + 0.7 * (X[:, 0] > 0)
    y = rng.binomial(1, p)
    Xte = rng.uniform(-2, 2, size=(1000, 3))
    pte = 0.15 + 0.7 * (Xte[:, 0] > 0)
    yte = rng.binomial(1, pte)

    ltb = LTBClassifier(random_state=0).fit(X, y)
    proba = ltb.predict_proba(Xte)
    assert proba.shape == (1000, 2)
    assert np.all(proba >= 0) and np.all(proba <= 1)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert list(ltb.classes_) == [0, 1]

    lr = LogisticRegression(max_iter=1000).fit(X, y)
    rmse_ltb = np.sqrt(np.mean((proba[:, 1] - pte) ** 2))
    rmse_lr = np.sqrt(np.mean((lr.predict_proba(Xte)[:, 1] - pte) ** 2))
    assert rmse_ltb < rmse_lr
