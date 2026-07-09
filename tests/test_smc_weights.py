import numpy as np
from causal_bench.sampling.weights import normalize_log_weights, kish_ess


def test_normalize_sums_to_one_and_is_stable():
    log_w = np.array([-1000.0, -1000.0, -1000.0])   # underflow-prone
    w, log_norm = normalize_log_weights(log_w)
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 1/3)

def test_kish_ess_uniform_is_n_and_degenerate_is_one():
    n = 8
    assert np.isclose(kish_ess(np.zeros(n)), n)              # uniform -> ESS = N
    spike = np.full(n, -1e9); spike[0] = 0.0
    assert np.isclose(kish_ess(spike), 1.0, atol=1e-6)      # one survivor -> ESS = 1


def test_normalize_uses_inferred_namespace_not_bare_np(monkeypatch):
    import causal_bench.sampling.weights as W
    calls = {"n": 0}
    real = W.get_namespace
    def spy(*a):
        calls["n"] += 1
        return real(*a)
    monkeypatch.setattr(W, "get_namespace", spy)
    w, log_norm = normalize_log_weights(np.array([-1.0, -2.0, -3.0]))
    assert calls["n"] >= 1                     # went through get_namespace
    assert np.isclose(w.sum(), 1.0)
    assert np.isfinite(log_norm)

