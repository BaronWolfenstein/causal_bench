import json

import numpy as np
import pytest

from causal_bench.validation.calibration_monitor import StreamingPageHinkley, page_hinkley
from experiments import exp58_calibration_monitor as exp


def historical(stream, delta, threshold):
    mean = cumulative = minimum = 0.0
    for n, value in enumerate(stream, 1):
        value = float(value)
        mean += (value - mean) / n
        cumulative += value - mean - delta
        minimum = min(minimum, cumulative)
        if cumulative - minimum > threshold:
            return n


@pytest.mark.parametrize('seed', range(8))
def test_batch_preserves_historical_first_alarm(seed):
    stream = np.random.default_rng(seed).normal(size=600)
    assert page_hinkley(stream, delta=.05, lam=6) == historical(stream, .05, 6)


def test_strict_boundary_warmup_and_latched_first_alarm():
    monitor = StreamingPageHinkley(delta=0, lam=.5)
    assert not monitor.update(0)
    assert not monitor.update(1)  # statistic equals .5
    assert monitor.update(1)
    for _ in range(10):
        assert monitor.update(0)
    assert monitor.detected_at == 3
    assert page_hinkley([0, 1, 1, 1], delta=0, lam=.01, min_n=4) == 4
    assert page_hinkley([0, 1, 1], delta=0, lam=.01, min_n=4) is None


@pytest.mark.parametrize('value', [0., .1, 1.])
def test_constant_stream_has_no_alarm(value):
    assert page_hinkley([value] * 1000) is None


@pytest.mark.parametrize('split', [0, 100, 503, 700])
def test_json_restart_exactly_matches_uninterrupted(split):
    stream = np.r_[np.random.default_rng(41).beta(1, 9, 500), np.full(200, .8)]
    full, restarted = StreamingPageHinkley(min_n=100), StreamingPageHinkley(min_n=100)
    full_flags = [full.update(x) for x in stream]
    flags = [restarted.update(x) for x in stream[:split]]
    restarted = StreamingPageHinkley.from_state_dict(json.loads(json.dumps(restarted.state_dict())))
    flags += [restarted.update(x) for x in stream[split:]]
    assert flags == full_flags
    assert restarted.state_dict() == full.state_dict()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), True, np.bool_(False), [0.1],
                               np.array([.1]), np.array([[.1]])])
def test_non_scalar_or_nonfinite_loss_rejected_without_mutation(bad):
    monitor = StreamingPageHinkley()
    saved = monitor.state_dict()
    with pytest.raises(ValueError):
        monitor.update(bad)
    assert monitor.state_dict() == saved


def test_overflow_rejected_without_mutation():
    monitor = StreamingPageHinkley()
    monitor.update(1e308)
    saved = monitor.state_dict()
    with pytest.raises(ValueError, match='overflowed'):
        monitor.update(-1e308)
    assert monitor.state_dict() == saved


@pytest.mark.parametrize('changes', [dict(lam=0), dict(delta=-1), dict(min_n=True), dict(min_n=0),
    dict(n=-1), dict(alarm=True), dict(detected_at=1), dict(n=0, mean=.1), dict(minimum=1)])
def test_malformed_configuration_or_restored_state_rejected(changes):
    saved = StreamingPageHinkley().state_dict()
    with pytest.raises(ValueError):
        StreamingPageHinkley.from_state_dict(dict(saved, **changes))
    with pytest.raises(ValueError):
        StreamingPageHinkley.from_state_dict(dict(saved, unknown=1))


def test_first_changed_observation_is_not_last_baseline():
    assert exp.alarm_outcome(500, 500) == ('early', None)
    assert exp.alarm_outcome(501, 500) == ('detected', 1)
    assert exp.alarm_outcome(None, 500) == ('missed', None)


@pytest.mark.parametrize('state', [None, [], 'not a state'])
def test_non_mapping_state_rejected(state):
    with pytest.raises(ValueError):
        StreamingPageHinkley.from_state_dict(state)


def test_numpy_scalar_configuration_is_json_serializable():
    monitor = StreamingPageHinkley(delta=np.float32(.005), lam=np.float32(6))
    monitor.update(np.float32(.2))
    saved = json.loads(json.dumps(monitor.state_dict()))
    assert StreamingPageHinkley.from_state_dict(saved).state_dict() == monitor.state_dict()


def test_experiment_never_credits_early_alarms_as_detection(monkeypatch):
    monitor_calls = iter([None] * 3 + [100, 101, None])
    ph_calls = iter([None] * 3 + [100, 101, None])
    ddm_calls = iter([100, 101, None] + [None] * 3)
    monkeypatch.setattr(exp, 'run_monitor', lambda *a, **k: ([], next(monitor_calls)))
    monkeypatch.setattr(exp, 'page_hinkley', lambda *a, **k: next(ph_calls))
    monkeypatch.setattr(exp, 'ddm', lambda *a, **k: (None, next(ddm_calls)))
    monkeypatch.setattr(exp, 'naive_repeated_test', lambda *a, **k: None)
    result = exp.run(n=200, change_at=100, n_fa=3, n_det=3)
    for prefix in ('', 'ph_', 'ddm_'):
        assert result[prefix + 'detected'] == 1
        assert result[prefix + 'early_alarms'] == 1
        assert result[prefix + 'missed'] == 1
        assert result[prefix + 'median_delay'] == 1
    assert 'early 1, missed 1' in exp.report(result)


def test_noisy_early_alarm_is_a_failure_not_a_short_delay():
    result = exp.run(n=300, change_at=200, n_fa=6, n_det=6, ph_delta=0, ph_lam=.01)
    assert result['ph_early_alarms'] == 6
    assert result['ph_detected'] == 0
    assert result['ph_missed'] == 0
    assert np.isnan(result['ph_median_delay'])
