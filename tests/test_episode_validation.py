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
        n_nozzles=1,
        n_bricks=5,
        p_brick=25.0,
        queue_capacity=10,
        lam=5.0,
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

    m = MetricsTracker(n_piles=1, n_nozzles=1)
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
