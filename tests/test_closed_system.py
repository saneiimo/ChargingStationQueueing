"""
Closed-system trials: the engine's drain mode and the objective sweep built
on it.

A closed-system trial runs every source on one vehicle set -- the queue and
the plugged-in vehicles at t0 plus every arrival in [t0, T] -- drains the
station instead of stopping at T, and gives the offline models a horizon read
off the discretized FIFO run. These tests pin down the pieces that make that
comparison sound: who is admitted, that nothing is censored on the
simulation side, that the models see exactly the simulated set, and that the
reported bracket on the closed-system optimum is valid.

    python -m pytest tests/test_closed_system.py -v
"""

from __future__ import annotations

import math

import pandas as pd
import pytest

from metrics.metrics_tracker import MetricsTracker
from models.ev import EV
from models.station import ChargingStation
from policy.power.proportional import ProportionalPower
from simulation.engine import SimulationEngine
from simulation.event import Event, EventQueue, EventType


# --------------------------------------------------------------------------- #
# Engine: drain mode
# --------------------------------------------------------------------------- #


def _engine(arrivals, *, drain, max_time=30.0, delta_arr=None, n_connectors=1):
    station = ChargingStation(
        n_piles=1, n_connectors=n_connectors, n_modules=4, p_module=25.0, queue_capacity=50,
        power_policy=ProportionalPower(), mean_interarrival=None,
    )
    engine = SimulationEngine(
        station, MetricsTracker(n_piles=1, n_connectors=n_connectors), EventQueue(),
        max_time=max_time, delta_arr=delta_arr, warmup_period=0.0, drain=drain,
    )
    engine.set_arrivals(arrivals)
    return engine


def _run_fifo(engine) -> None:
    """Head-of-line onto any free connector until the episode ends."""
    engine.reset(seed=0)
    while not engine.terminated:
        if engine.needs_assignment_decision():
            pile = next(i for i, p in enumerate(engine.station.piles) if not p.is_full)
            engine.assign_ev(pile)
        else:
            engine.advance_time()


def _ev(i, t):
    return EV(id=i, c_b=50.0 * 60, s_i=0.2, s_f=0.8, arrival_time=t)


def test_without_drain_the_run_stops_at_max_time():
    engine = _engine([_ev(0, 1.0), _ev(1, 2.0), _ev(2, 3.0)], drain=False, max_time=10.0)
    _run_fifo(engine)
    assert engine.end_time == engine.current_time == 10.0
    assert len(engine.metrics.finished_evs) < 3  # cut off with vehicles still in the station


def test_drain_serves_everyone_admitted_and_nobody_after_T():
    # Continuous times (delta_arr=None), which snap_arrivals does not filter:
    # the 40-minute arrival is after T and must never be admitted.
    arrivals = [_ev(0, 1.0), _ev(1, 2.0), _ev(2, 3.0), _ev(3, 40.0)]
    engine = _engine(arrivals, drain=True, max_time=10.0)
    _run_fifo(engine)
    m = engine.metrics
    assert {ev.id for ev in m.arrived_evs} == {0, 1, 2}
    assert {ev.id for ev in m.finished_evs} == {0, 1, 2}
    assert engine.station_is_empty()
    assert engine.end_time == engine.current_time > 10.0
    assert engine.end_time == max(ev.departure_time for ev in m.finished_evs)


@pytest.mark.parametrize("sim_over_first", [False, True])
def test_arrival_exactly_at_T_is_admitted_whatever_the_tie_order(monkeypatch, sim_over_first):
    """Equal-time events have no fixed order. With the station empty at T,
    popping SIM_OVER before the T arrival must not end the run early."""
    if sim_over_first:
        def lt(a, b):
            return (a.time, a.event_type is not EventType.SIM_OVER) < (
                b.time, b.event_type is not EventType.SIM_OVER
            )
        monkeypatch.setattr(Event, "__lt__", lt)
    engine = _engine([_ev(0, 30.0)], drain=True, max_time=30.0)
    _run_fifo(engine)
    assert [ev.id for ev in engine.metrics.finished_evs] == [0]
    assert engine.end_time > 30.0


def test_drain_ends_at_T_when_the_station_is_already_empty():
    engine = _engine([_ev(0, 1.0)], drain=True, max_time=200.0)
    _run_fifo(engine)
    assert len(engine.metrics.finished_evs) == 1
    assert engine.end_time == 200.0


def test_drain_raises_if_waiting_vehicles_are_never_assigned():
    engine = _engine([_ev(0, 5.0)], drain=True, max_time=30.0)
    engine.reset(seed=0)  # processes the arrival; nobody assigns it
    with pytest.raises(RuntimeError, match="queue policy must keep assigning"):
        while not engine.terminated:
            engine.advance_time()


# --------------------------------------------------------------------------- #
# The objective sweep in closed-system mode
# --------------------------------------------------------------------------- #

gp = pytest.importorskip("gurobipy")


def _cfg(**kw):
    from experiments.objective_sweep import TrialConfig

    base = dict(
        n_piles=1, n_connectors=2, n_modules=6, max_time=60, warmup_period=120, delta=5.0,
        seed=41, battery_cap_kwh=(75.0,), time_limit=120.0, dw_time_limit=120.0,
        bp_time_limit=120.0, gap_tolerance_target_min=None, mean_interarrival=15.0,
    )
    base.update(kw)
    return TrialConfig(**base)


def test_closed_system_config_rejects_mismatched_vehicle_sets():
    from offline_cl_opt import COHORTS_MEASUREMENT

    for bad in (
        dict(include_queued=False),
        dict(boundary_mode=None),
        dict(flush_queue_at_warmup=True),
        dict(objective_cohorts=COHORTS_MEASUREMENT),
    ):
        with pytest.raises(ValueError, match="closed_system"):
            _cfg(**bad)
        _cfg(closed_system=False, **bad)  # fine as a measured-window trial
    with pytest.raises(ValueError, match="horizon_slack"):
        _cfg(horizon_slack=-0.1)


def test_offline_horizon_is_not_a_config_value_in_a_closed_trial():
    with pytest.raises(ValueError, match="derived per trial"):
        _cfg().offline_horizon
    assert _cfg(closed_system=False).offline_horizon == 60.0


def test_closed_trial_runs_every_source_on_the_same_vehicle_set():
    from experiments.objective_sweep import run_trial

    r = run_trial(_cfg(), run_exact_model=True, run_bp_model=False, run_dw_model=True, verbose=False)
    inst, counts = r.instance, r.counts
    n = inst["n_vehicles"]

    # The simulation drained the station: nothing censored, both groups equal.
    assert counts["drain_minutes"] >= 0.0
    for src in (r.sim, r.grid):
        arrived, completed = src["arrived_by_cohort"]["all"], src["completed_by_cohort"]["all"]
        assert arrived["n"] == completed["n"] == n
        assert arrived["n_censored"] == 0
        assert arrived["mean_sojourn"] == pytest.approx(completed["mean_sojourn"])
    # ...and the models were given exactly that set, all optimized.
    assert inst["n_dropped_late_arrivals"] == 0
    assert r.exact["n_vehicles"] == r.dw["n_vehicles"] == n == inst["n_optimized"]
    assert r.exact["arrived_by_cohort"]["all"]["n"] == n

    # Horizon: read off FIFO, at least both makespans, in whole slots.
    assert inst["K"] >= max(inst["K_grid_fifo"], inst["K_feasible_fifo"]) == inst["K_fifo"]
    assert inst["horizon_minutes"] == pytest.approx(inst["K"] * 5.0)
    assert r.exact["K"] == r.dw["K"] == inst["K"]

    # The bracket is valid and the FIFO schedule is one of its upper bounds.
    c = r.closed
    assert c["lower_bound"] <= c["upper_bound"] + 1e-6
    assert c["upper_bound"] <= inst["feasible_fifo_total_sojourn"] + 1e-6
    assert r.dw["LB"] <= c["upper_bound"] + 1e-6
    assert c["mean_sojourn_LB"] == pytest.approx(c["lower_bound"] / n)


def test_closed_trial_bracket_closes_and_both_exact_solvers_agree():
    """Small enough to solve exactly: the closed-system optimum is proven, and
    the compact MILP and branch-and-price report the same value, from
    schedules that complete every vehicle (repaired at K if tied there)."""
    from experiments.objective_sweep import run_trial

    for ia in (15.0, 25.0):
        r = run_trial(_cfg(mean_interarrival=ia), run_exact_model=True, run_bp_model=True,
                      run_dw_model=False, verbose=False)
        assert r.exact["status"] == "OPTIMAL" and r.bp["status"] == "OPTIMAL"
        assert r.exact["objective"] == pytest.approx(r.bp["objective"])
        assert r.exact["all_completed"] and r.bp["all_completed"]
        assert r.closed["proven_optimal"]
        assert r.closed["upper_bound"] == pytest.approx(r.exact["objective"])
        # The optimum cannot be worse than the discretized FIFO it was bounded by.
        assert r.closed["upper_bound"] <= r.instance["feasible_fifo_total_sojourn"] + 1e-6


def test_horizon_slack_lengthens_the_horizon():
    from experiments.objective_sweep import TrialConfig, build_instance, run_episode

    for slack in (0.0, 0.5):
        cfg = _cfg(horizon_slack=slack)
        env, _, _ = run_episode(cfg)
        _, _, summary = build_instance(cfg, env)
        assert summary["K"] == math.ceil(summary["K_fifo"] * (1 + slack))
    assert isinstance(cfg, TrialConfig)


def test_full_queue_is_rejected_rather_than_compared_on_different_sets():
    from experiments.objective_sweep import run_trial

    with pytest.raises(ValueError, match="queue_capacity"):
        run_trial(_cfg(queue_capacity=1, mean_interarrival=5.0), run_exact_model=False,
                  run_dw_model=False, verbose=False)


def test_trial_json_without_closed_system_replays_as_a_window():
    """Trials written before closed_system existed have no such key; they
    must replay as the hard-stop episode they were."""
    from experiments.objective_sweep import replay_episode, run_episode

    cfg = _cfg(closed_system=False)
    env, _, specs = run_episode(cfg)
    row = cfg.to_row()
    del row["closed_system"]
    replayed = replay_episode(row, specs)
    assert not replayed.engine.drain
    assert replayed.engine.end_time == env.engine.end_time == 180.0


def test_fixed_boundary_vehicle_that_departed_counts_as_completed():
    """A FIXED vehicle's pinned power stops at its last whole slot, so its
    final partial slot's energy is missing -- yet it did finish if it left
    before the horizon. One still charging at the horizon has not."""
    from experiments.objective_sweep.trial import _model_groups

    pv = pd.DataFrame([
        dict(vehicle_id=1, cohort="boundary", boundary_mode="fixed", departure_slot=3.0,
             energy_kwh=9.0, energy_delivered_before_kwh=5.0, energy_required_kwh=15.0,
             sojourn_min=15.0),
        dict(vehicle_id=2, cohort="boundary", boundary_mode="fixed", departure_slot=10.0,
             energy_kwh=9.0, energy_delivered_before_kwh=5.0, energy_required_kwh=15.0,
             sojourn_min=50.0),
    ])
    _, _, diag = _model_groups(pv, K=10)
    assert diag["n_completed"] == 1 and diag["n_censored"] == 1
    assert not diag["all_completed"]
