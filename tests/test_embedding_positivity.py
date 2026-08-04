"""Self-validating checks for exp50 (embedding as causal adjustment set).

These pin the three claims that make the experiment interpretable:
  1. no confounding (conf=0) => every method unbiased;
  2. a LINEAR outcome model shows NO attenuation even at severe positivity;
  3. a FLEXIBLE outcome model attenuates toward null, monotonically in positivity.
"""
from causal_bench.validation.embedding_positivity import report_rows


def _rows(confs, flex, n=1500, n_reps=5):
    return report_rows(n=n, n_reps=n_reps, confs=confs, flex=flex, seed=0)


def test_no_confounding_unbiased():
    r = _rows((0.0,), flex=True)[0]
    assert abs(r["oracle_Ustar"]) < 0.08
    assert abs(r["naive_fullW"]) < 0.08


def test_linear_outcome_has_no_pathology():
    # linear ridge outcome recovers the effect even at severe positivity (conf=5)
    r = _rows((5.0,), flex=False)[0]
    assert abs(r["oracle_Ustar"]) < 0.08
    assert abs(r["naive_fullW"]) < 0.08


def test_flexible_outcome_attenuates_with_positivity():
    low, high = _rows((1.0, 5.0), flex=True)
    b_low, b_high = low["naive_fullW"], high["naive_fullW"]
    assert b_high < -0.10           # clearly attenuated toward null at severe positivity
    assert b_high < b_low - 0.05     # attenuation grows with positivity severity
