"""Regression tests for the DDM drift detector in validation/calibration_monitor.py.

Guards the fixed degenerate-baseline false alarm: a clean (zero-error) stream has p = sd = 0 throughout,
so a non-strict `>=` drift test fired spuriously at `0 >= 0`. The fix is the strict inequality (drift must
*exceed* the historical-minimum band, not equal it)."""
import numpy as np
import pytest

from causal_bench.validation.calibration_monitor import ddm, simulate_error_stream


def test_zero_error_stream_no_alarm():
    # the reported bug: all-zeros must NOT alarm (previously drifted at obs 30 from 0 >= 0)
    assert ddm([0] * 100) == (None, None)
    assert ddm([0] * 1000) == (None, None)


def test_constant_nonzero_stream_no_alarm():
    # a constant error rate (all-ones) is also drift-free: no increase over the baseline band
    assert ddm([1] * 100) == (None, None)


def test_short_stream_below_min_n_no_alarm():
    # fewer than min_n observations => never evaluated
    assert ddm([0, 1, 0, 1] * 5) == (None, None)  # 20 < 30
    assert ddm([1] * 29) == (None, None)


def test_real_increase_is_detected():
    # a genuine jump 5% -> 40% must still be caught shortly after the change point
    _, drift_at = ddm(simulate_error_stream(300, 0.05, p1=0.40, change_at=100, seed=0))
    assert drift_at is not None
    assert 100 <= drift_at <= 160  # detected near the change, with a short CUSUM-like delay


def test_warning_precedes_drift():
    warn_at, drift_at = ddm(simulate_error_stream(300, 0.05, p1=0.40, change_at=100, seed=0))
    assert warn_at is not None and drift_at is not None
    assert warn_at <= drift_at


def test_strict_inequality_boundary():
    # construct a stream whose band exactly equals (not exceeds) the threshold: must NOT alarm.
    # a perfectly clean run keeps p=sd=0 => band is 0 and current is 0 => equality, not exceedance.
    for n in (30, 31, 50, 200):
        assert ddm([0] * n) == (None, None)


def test_rate_only_shifts_down_no_alarm():
    # error rate that only DECREASES (improvement) must never alarm as drift-up
    stream = [1] * 40 + [0] * 160  # 100% then 0% — rate falls
    _, drift_at = ddm(stream)
    assert drift_at is None


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
