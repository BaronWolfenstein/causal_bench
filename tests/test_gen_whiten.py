import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from causal_bench.generative.whiten import zca_fit


def test_zca_roundtrip_and_isotropy():
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 6)) @ rng.standard_normal((6, 6))  # correlated
    z = zca_fit(X)
    Z = z.transform(X)
    assert np.allclose(z.inverse(Z), X, atol=1e-8)            # invertible
    C = np.cov(Z, rowvar=False)
    assert np.allclose(C, np.eye(6), atol=0.15)              # ~ identity covariance


def test_zca_preserves_separation_auc():
    rng = np.random.default_rng(1)
    common = rng.standard_normal((200, 6))
    rare = rng.standard_normal((40, 6)) + 3.0
    X = np.vstack([rare, common]); y = np.r_[np.ones(40), np.zeros(200)]
    Z = zca_fit(X).transform(X)
    auc_raw = roc_auc_score(y, LogisticRegression(max_iter=500).fit(X, y).predict_proba(X)[:,1])
    auc_zca = roc_auc_score(y, LogisticRegression(max_iter=500).fit(Z, y).predict_proba(Z)[:,1])
    assert abs(auc_raw - auc_zca) < 0.03                     # invertible => AUC preserved
