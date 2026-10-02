"""
The ``since``-windowed time averages in ``MetricsTracker`` against the same
quantities rebuilt independently from each vehicle's own timestamps.

Between events the station state is constant, so over any window
``[since, T]``:

* connector busy time = sum over served EVs of
  ``[service_start, departure or T]`` clipped to the window,
* time-average L      = sum over EVs of ``[arrival, departure or T]``
  clipped, divided by the window length,
* time-average Q      = the same with ``[arrival, service_start or T]``.

These are exact for any ``since`` (aligned with an event or not), so the
windowed metrics must match them to rounding. Regression for the bug where
``event_times`` (each interval's START) was read as the interval's END,
shifting every interval one step earlier: on an overloaded run whose
connectors were busy the whole measured window, ``rho_sim`` read 0.942.

    python -m pytest tests/test_metrics_windowing.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.objective_sweep import TrialConfig, run_episode


def _episode(mean_interarrival: float):
    cfg = TrialConfig(
        n_piles=1, n_connectors=2, n_modules=6, p_module=25.0, battery_cap_kwh=(75.0,),
        warmup_period=360.0, max_time=240.0, mean_interarrival=mean_interarrival, seed=41,
    )
    env, _, _ = run_episode(cfg)
    return env


def _overlap(a: float, b: float, lo: float, hi: float) -> float:
    return max(0.0, min(b, hi) - max(a, lo))


def _reference(env, since: float):
    """(busy minutes per connector [P x C], L-area, Q-area) over [since, T]."""
    m = env.engine.metrics
    T = float(env.engine.current_time)
    assert not m.dropped_evs and not m.flushed_evs  # every arrival stays until served or T
    busy = np.zeros((m.n_piles, m.n_connectors))
    L_area = Q_area = 0.0
    for ev in m.arrived_evs:
        end = ev.departure_time if math.isfinite(ev.departure_time) else T
        start = ev.service_start_time if ev.service_start_time is not None else T
        L_area += _overlap(ev.arrival_time, end, since, T)
        Q_area += _overlap(ev.arrival_time, start, since, T)
        if ev.service_start_time is not None:
            # *_tracker: the pile/slot it used, kept after departure.
            busy[ev.pile_tracker.id, ev.connector_id_tracker] += _overlap(start, end, since, T)
    return busy, L_area, Q_area


@pytest.mark.parametrize("mean_interarrival", [7.5, 20.0])  # overloaded / stable
def test_windowed_metrics_match_per_vehicle_reconstruction(mean_interarrival):
    env = _episode(mean_interarrival)
    m = env.engine.metrics
    T = float(env.engine.current_time)
    # Interval i is [event_times[i], event_times[i] + dt]: consecutive, from 0 to T.
    et, dt = np.array(m.event_times), np.array(m.event_durations)
    assert et[0] == 0.0 and np.allclose(et[:-1] + dt[:-1], et[1:]) and math.isclose(et[-1] + dt[-1], T)

    # The warm-up boundary (an event), an arbitrary mid-interval cut, and 0.
    for since in (m.warmup_period, 123.456, 0.0):
        busy, L_area, Q_area = _reference(env, since)
        window = T - since
        np.testing.assert_allclose(m.connector_utilization(T, since), busy / window, atol=1e-9)
        assert m.mean_connector_utilization(T, since) == pytest.approx(busy.mean() / window, abs=1e-9)
        assert m.average_L(since) == pytest.approx(L_area / window, abs=1e-9)
        assert m.average_Q(since) == pytest.approx(Q_area / window, abs=1e-9)


def test_overloaded_measured_window_is_fully_busy():
    """The case that exposed the bug: a backlog at the warm-up boundary that
    outlasts the window keeps both connectors busy throughout."""
    env = _episode(7.5)
    m = env.engine.metrics
    T = float(env.engine.current_time)
    assert len(m.queued_at_warmup_end) > 0
    assert m.mean_connector_utilization(T, m.warmup_period) == pytest.approx(1.0, abs=1e-9)


def test_windowed_energy_is_additive_and_matches_whole_run():
    """Energy accrues at a non-constant rate inside an interval (taper), so
    there is no exact per-vehicle reference for a mid-interval cut -- but at
    the event-aligned warm-up boundary, [0, s] + [s, T] must equal the
    whole-run total, and the window [s, T] must equal the intervals that
    start at or after s."""
    env = _episode(20.0)
    m = env.engine.metrics
    s = m.warmup_period
    after = sum(e for t, e in zip(m.event_times, m.history_pile_energy) if t >= s)
    before = sum(e for t, e in zip(m.event_times, m.history_pile_energy) if t < s)
    np.testing.assert_allclose(m.pile_energy_sold_since(s), after, atol=1e-9)
    np.testing.assert_allclose(before + m.pile_energy_sold_since(s), m.pile_energy_sold, atol=1e-6)
