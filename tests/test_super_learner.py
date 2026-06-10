import numpy as np
from sklearn.datasets import make_classification, make_regression
from causal_bench.super_learner import SuperLearner

def test_fit_predict_proba_shape():
    X, y = make_classification(n_samples=300, n_features=5, random_state=0)
    sl = SuperLearner(task="classification", n_folds=3, random_state=0)
    sl.fit(X, y)
    probs = sl.predict_proba(X)
    assert probs.shape == (300,)
    assert np.all((probs >= 0) & (probs <= 1))

def test_weights_sum_to_one():
    X, y = make_classification(n_samples=300, n_features=5, random_state=1)
    sl = SuperLearner(task="classification", n_folds=3, random_state=1)
    sl.fit(X, y)
    assert abs(sum(sl.weights_) - 1.0) < 1e-6
    assert all(w >= 0 for w in sl.weights_)

def test_regression_predict_shape():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 4))
    y = X[:, 0] + rng.standard_normal(300) * 0.1
    sl = SuperLearner(task="regression", n_folds=3, random_state=0)
    sl.fit(X, y)
    preds = sl.predict(X)
    assert preds.shape == (300,)

def test_proba_clipped():
    """predict_proba output should be in (0, 1), never exactly 0 or 1."""
    X, y = make_classification(n_samples=200, n_features=4, random_state=2)
    sl = SuperLearner(task="classification", n_folds=3, random_state=2)
    sl.fit(X, y)
    probs = sl.predict_proba(X)
    assert np.all(probs > 0)
    assert np.all(probs < 1)

def test_custom_candidates():
    from sklearn.linear_model import LogisticRegression
    X, y = make_classification(n_samples=200, n_features=4, random_state=3)
    sl = SuperLearner(
        candidates=[LogisticRegression(), LogisticRegression(C=0.1)],
        task="classification", n_folds=3, random_state=3
    )
    sl.fit(X, y)
    assert len(sl.weights_) == 2
    assert abs(sum(sl.weights_) - 1.0) < 1e-6

def test_raises_before_fit():
    sl = SuperLearner(task="classification")
    try:
        sl.predict_proba(np.ones((5, 3)))
        assert False, "Should have raised"
    except (TypeError, AttributeError):
        pass  # expected — _fitted_candidates is None
