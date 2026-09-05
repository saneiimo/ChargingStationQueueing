"""
Arrival-time grid (`delta_arr`): snap math, generate_arrivals, and the DES engine.

Run from the repo root:

    python -m pytest tests/test_arrivals.py -s -v
"""

from __future__ import annotations

import numpy as np
import pytest

from models.ev import EV
from models.station import ChargingStation
from metrics.metrics_tracker import MetricsTracker
from policy.power.proportional import ProportionalPower
from simulation.arrivals import (
    generate_arrivals,
    snap_arrival_time,
    snap_arrivals,
    validate_delta_arr,
)
from simulation.engine import SimulationEngine
from simulation.event import EventQueue


def test_snap_arrival_time_examples():
    """User-specified rounding of 3.69 onto 1, 2, 3, and 5 minute grids."""
    t = 3.69
    assert snap_arrival_time(t, 1) == 4.0
    assert snap_arrival_time(t, 2) == 4.0
    assert snap_arrival_time(t, 3) == 3.0
    assert snap_arrival_time(t, 5) == 5.0


def test_validate_delta_arr_rejects_nonpositive():
    assert validate_delta_arr(None) is None
    with pytest.raises(ValueError, match="positive"):
        validate_delta_arr(0)
    with pytest.raises(ValueError, match="positive"):
        validate_delta_arr(-1)


def test_delta_arr_none_leaves_continuous_times():
    """Omitting delta_arr must match the historical continuous sampler."""
    kwargs = dict(mean_interarrival=5.0, max_time=80.0)
    a = generate_arrivals(**kwargs, rng=np.random.default_rng(0), delta_arr=None)
    b = generate_arrivals(**kwargs, rng=np.random.default_rng(0))
    assert [ev.arrival_time for ev in a] == [ev.arrival_time for ev in b]
    assert any(ev.arrival_time != round(ev.arrival_time) for ev in a), (
        "sanity: a continuous draw should not land on integers only"
    )


def test_generate_arrivals_snaps_to_grid():
    evs = generate_arrivals(
        mean_interarrival=4.0,
        max_time=60.0,
        rng=np.random.default_rng(1),
        delta_arr=2,
    )
    assert evs, "expected at least one arrival on this seed/horizon"
    for ev in evs:
        assert ev.arrival_time >= 0.0
        assert ev.arrival_time <= 60.0
        n = round(ev.arrival_time / 2.0)
        assert abs(ev.arrival_time - n * 2.0) < 1e-12, (
            f"{ev.arrival_time} is not a multiple of 2"
        )
    times = [ev.arrival_time for ev in evs]
    assert times == sorted(times)


def test_snap_arrivals_drops_times_past_horizon():
    evs = [
        EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=3.69),
        EV(id=1, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=9.6),
    ]
    snapped = snap_arrivals(evs, delta_arr=1, max_time=5.0)
    assert [ev.id for ev in snapped] == [0]
    assert snapped[0].arrival_time == 4.0


def test_engine_snaps_external_arrivals():
    """delta_arr on the engine snaps an installed continuous list."""
    station = ChargingStation(
        n_piles=1,
        n_connectors=1,
        n_modules=4,
        p_module=25.0,
        queue_capacity=5,
        power_policy=ProportionalPower(),
        mean_interarrival=None,
    )
    engine = SimulationEngine(
        station,
        MetricsTracker(n_piles=1, n_connectors=1),
        EventQueue(),
        max_time=30.0,
        delta_arr=1,
    )
    engine.set_arrivals(
        [EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=3.69)]
    )
    engine.reset(seed=0)
    # reset() processes the first event, so the snapped arrival is already in metrics.
    assert engine.metrics.arrived_evs[0].arrival_time == 4.0
    assert engine.current_time == 4.0


def test_engine_delta_arr_none_keeps_external_times():
    station = ChargingStation(
        n_piles=1,
        n_connectors=1,
        n_modules=4,
        p_module=25.0,
        queue_capacity=5,
        power_policy=ProportionalPower(),
        mean_interarrival=None,
    )
    engine = SimulationEngine(
        station,
        MetricsTracker(n_piles=1, n_connectors=1),
        EventQueue(),
        max_time=30.0,
        delta_arr=None,
    )
    engine.set_arrivals(
        [EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=3.69)]
    )
    engine.reset(seed=0)
    assert engine.metrics.arrived_evs[0].arrival_time == 3.69
    assert engine.current_time == 3.69
