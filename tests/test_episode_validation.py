"""
Post-simulation metric validation and queueing-law reporting.

Run after a finished episode (verbose prints PASS lines; failures always print)::

    from metrics.validate import validate_episode, report_queueing_laws
    from visualization.pile_power import run_fifo_episode

    env = run_fifo_episode(seed=42)
    validate_episode(env, verbose=True)
    report_queueing_laws(env, verbose=True)

Or with pytest::

    python -m pytest tests/test_episode_validation.py -s -v
"""

from __future__ import annotations

import numpy as np

from metrics.validate import (
    report_queueing_laws,
    validate_episode,
)
from policy.queue.fifo import FIFOQueuePolicy
from env.charging_env import ChargingStationEnv


def _run_short_episode(seed: int = 0) -> ChargingStationEnv:
    env = ChargingStationEnv(
        n_piles=2,
        n_connectors=1,
        n_modules=5,
        p_module=25.0,
        queue_capacity=10,
        mean_interarrival=5.0,
    )
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(1)
    policy = FIFOQueuePolicy()
    done = bool(env.engine.terminated)
    steps = 0
    while not done and steps < 50000:
        if not env.engine.needs_assignment_decision():
            break
        mask = env.action_masks()
        ev, pile_id = policy.decide(obs, mask, rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)
        steps += 1
    return env


def test_energy_needed_is_kwh():
    """energy_needed must use the same kWh units as energy_received."""
    from config import HR2MIN
    from models.ev import EV

    ev = EV(id=0, c_b=50.0 * HR2MIN, s_i=0.2, s_f=0.8, arrival_time=0.0)
    expected = (0.8 - 0.2) * (50.0 * HR2MIN) / HR2MIN
    assert abs(ev.energy_needed - expected) < 1e-12
    assert abs(ev.energy_needed - 30.0) < 1e-12  # 0.6 * 50 kWh
    print("  PASS energy_needed units (kWh)")


def test_validate_episode_passes_on_fifo_day():
    """Full-day FIFO run should satisfy all post-sim metric checks."""
    print("\n=== test_validate_episode_passes_on_fifo_day ===")
    env = _run_short_episode(seed=2)
    report = validate_episode(env, verbose=True, raise_on_fail=True)
    assert report.ok
    laws = report_queueing_laws(env, verbose=True)
    assert laws["n_finished"] >= 1
    print("  PASS validate_episode + queueing laws")


def test_finished_missing_timestamps_raise():
    """Finished EVs without service/departure times must error in the tracker."""
    from metrics.metrics_tracker import MetricsError, MetricsTracker
    from models.ev import EV

    m = MetricsTracker(n_piles=1, n_connectors=1)
    ev = EV(id=7, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=1.0)
    # Pretend finished without plug-in / departure stamps.
    m.finished_evs.append(ev)
    try:
        m.finished_time_arrays()
        raise AssertionError("expected MetricsError")
    except MetricsError as exc:
        assert "EV 7" in str(exc)
        print(f"  PASS MetricsError on unfinished stamps: {exc}")


def test_validate_silent_when_ok_and_not_verbose(capsys):
    """With verbose=False, a clean episode prints nothing."""
    env = _run_short_episode(seed=3)
    validate_episode(env, verbose=False, raise_on_fail=True)
    captured = capsys.readouterr()
    assert captured.out == ""


if __name__ == "__main__":
    test_energy_needed_is_kwh()
    test_validate_episode_passes_on_fifo_day()
    test_finished_missing_timestamps_raise()
    print("All validation self-checks done.")


def test_boundary_cohorts_are_disjoint_when_an_arrival_lands_on_the_boundary():
    """The three measured-window cohorts must stay disjoint even when an
    arrival falls exactly ON the warm-up boundary.

    Regression test. With a gridded arrival stream (``delta_arr``) whose
    grid divides ``warmup_period`` this is routine, not exotic: such an EV
    passes ``arrived_post_warmup``'s ``arrival_time >= warmup_period`` test
    *and* can already be queued or plugged in when the boundary snapshot is
    taken, putting it in two cohorts at once. Downstream,
    ``build_measurement_instance`` then emits it twice and the offline model
    dies on a bare ``KeyError: 'Duplicate keys in Model.addVars()'``.
    """
    from collections import Counter

    from offline_cl_opt import BoundaryMode, StationSpec, build_measurement_instance
    from policy.power.proportional import ProportionalPower
    from simulation.arrivals import generate_arrivals

    delta, warmup, measured = 5.0, 360.0, 60.0
    # delta_arr=delta snaps arrivals onto the model's own grid, and
    # warmup % delta == 0, so an arrival can land exactly on the boundary.
    evs = generate_arrivals(
        mean_interarrival=20.0,
        max_time=900.0,  # fixed: the draw horizon is itself part of the stream
        rng=np.random.default_rng(39),
        battery_cap_options=[75.0 * 60],
        delta_arr=delta,
    )
    env = ChargingStationEnv(
        n_piles=1,
        n_connectors=2,
        n_modules=6,
        p_module=25.0,
        queue_capacity=1000,
        power_policy=ProportionalPower(),
        mean_interarrival=None,
        arrivals=evs,
        max_time=measured,
        battery_cap_options=[75.0 * 60],
        delta_arr=delta,
        warmup_period=warmup,
    )
    obs, _ = env.reset(seed=39)
    rng = np.random.default_rng(12)
    policy = FIFOQueuePolicy()
    done = False
    while not done:
        mask = env.action_masks()
        ev, pile_id = policy.decide(obs, mask, rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)

    m = env.engine.metrics
    # This seed/grid genuinely produces the collision -- if it stops doing
    # so the test is no longer exercising anything, so assert the setup too.
    boundary_ids = {snap.ev_id for snap in m.in_service_at_warmup_end}
    assert any(
        ev.arrival_time == m.warmup_period for ev in m.arrived_evs
    ), "setup no longer puts an arrival exactly on the boundary"
    assert boundary_ids, "setup no longer leaves anyone in service at the boundary"

    post = {ev.id for ev in m.arrived_post_warmup}
    queued_ids = {ev.id for ev in m.queued_at_warmup_end}
    assert not (post & boundary_ids)
    assert not (post & queued_ids)

    # ... and the instance built from them has no duplicate vehicle.
    inst = build_measurement_instance(
        arrived_post_warmup=m.arrived_post_warmup,
        queued_at_warmup_end=m.queued_at_warmup_end,
        in_service_at_warmup_end=m.in_service_at_warmup_end,
        warmup_period=env.engine.warmup_period,
        delta=delta,
        horizon_minutes=measured,
        include_queued=True,
        boundary_mode=BoundaryMode.OPTIMIZE,
        station=StationSpec(n_piles=1, n_connectors=2, n_modules=6, p_module=25.0),
    )
    dupes = [i for i, c in Counter(v.id for v in inst.vehicles).items() if c > 1]
    assert not dupes, f"duplicate vehicles in the instance: {dupes}"
    print("  PASS boundary cohorts disjoint (arrival exactly on the boundary)")
