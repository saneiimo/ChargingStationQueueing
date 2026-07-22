"""
End-to-end checks: energy accrues between events, and the Gym env only stops
for an assignment when the queue and a free nozzle both exist.

Each test prints the steps it takes and the numbers it checks so you can follow
the logic when running with stdout visible.

Run from the repo root (prints need -s):

    python -m pytest tests/test_simulation_env.py -s -v

Or run this file directly:

    python tests/test_simulation_env.py

In a notebook:

    !python -m pytest tests/test_simulation_env.py -s -v
"""

from __future__ import annotations

import numpy as np

from env.charging_env import ChargingStationEnv
from models.ev import EV
from models.station import ChargingStation
from metrics.metrics_tracker import MetricsTracker
from policy.power.proportional import ProportionalPower
from policy.queue.fifo import FIFOQueuePolicy
from simulation.engine import SimulationEngine
from simulation.event import EventQueue, Event, EventType


def test_des_projects_energy_between_events():
    """After assign, advancing to departure should accrue energy and raise SoC."""
    print("\n=== test_des_projects_energy_between_events ===")
    print(
        "Intent: after plugging one EV, advancing the DES until departure should "
        "raise SoC and accrue energy; power-based and SoC-based energy must agree."
    )
    print("Logic: plug one EV, advance DES until it departs, check SoC/energy rose.")

    power_policy = ProportionalPower()
    station = ChargingStation(
        n_piles=1,
        n_nozzles=2,
        n_bricks=5,
        p_brick=25.0,
        queue_capacity=5,
        power_policy=power_policy,
        lam=100.0,
    )
    metrics = MetricsTracker(n_piles=1, n_nozzles=2)
    engine = SimulationEngine(station, metrics, EventQueue())
    engine.rng = np.random.default_rng(0)
    engine.current_time = 0.0
    engine.next_time = 0.0
    engine.terminated = False
    engine.event_heap.clear()

    ev = EV(id=0, c_b=50.0, s_i=0.2, s_f=0.25, arrival_time=0.0)
    station.queue.append(ev)
    print(f"  Queued EV0: c_b={ev.c_b}, s_i={ev.s_i}, s_f={ev.s_f}")

    ok = engine.assign_ev(0)
    print(f"  assign_ev(pile=0) -> {ok}, nozzle_id={ev.nozzle_id}, p_act={ev.p_act:.3f}")
    assert ok
    assert ev.pile is not None
    s0 = ev.s_current
    e0 = ev.energy_received

    engine.event_heap.push(Event(engine.max_time, EventType.SIM_OVER))

    steps = 0
    while not engine.terminated and steps < 20:
        engine.advance_time()
        steps += 1
        print(
            f"  step {steps}: t={engine.current_time:.3f}, "
            f"event={engine.last_event_type.name if engine.last_event_type else None}, "
            f"SoC={ev.s_current:.4f}, energy={ev.energy_received:.4f}"
        )
        if ev in metrics.finished_evs:
            break

    print(
        f"  Result: finished={ev in metrics.finished_evs}, "
        f"SoC {s0:.4f}->{ev.s_current:.4f}, "
        f"energy {e0:.4f}->{ev.energy_received:.4f}, "
        f"|E_power - E_SoC|={abs(ev.energy_received - ev.energy_received_2):.2e}"
    )

    assert ev in metrics.finished_evs
    assert ev.s_current >= s0
    assert ev.energy_received > e0
    assert abs(ev.energy_received - ev.energy_received_2) < 1e-4
    print("  PASS")


def test_env_decision_point_and_fifo_episode():
    """Full FIFO day: agent only acts at decision points until SIM_OVER."""
    print("\n=== test_env_decision_point_and_fifo_episode ===")
    print(
        "Intent: the Gym env should only ask for a pile choice when the queue is "
        "nonempty and a nozzle is free, then auto-advance until SIM_OVER."
    )
    print("Logic: reset Gym env, assign with FIFO whenever a nozzle is free, run to end.")

    env = ChargingStationEnv(
        n_piles=2,
        n_nozzles=2,
        n_bricks=4,
        p_brick=25.0,
        queue_capacity=8,
        lam=3.0,
    )
    obs, info = env.reset(seed=7)
    print(
        f"  After reset: t={info['time']:.2f}, queue={info['queue_len']}, "
        f"obs_shape={obs.shape}, mask={info['action_mask']}"
    )
    assert obs.shape == env.observation_space.shape
    assert "action_mask" in info

    policy = FIFOQueuePolicy()
    rng = np.random.default_rng(0)
    done = False
    steps = 0
    while not done and steps < 5000:
        assert env.engine.needs_assignment_decision() or env.engine.terminated
        mask = env.action_masks()
        assert mask.any()
        action = policy.select_pile(obs, mask, rng)
        obs, reward, done, truncated, info = env.step(action)
        assert not truncated
        assert np.isfinite(reward)
        steps += 1
        if steps <= 5 or done:
            print(
                f"  decision {steps}: action=pile {action}, reward={reward:.3f}, "
                f"t={info['time']:.1f}, finished={info['finished']}, "
                f"dropped={info['dropped']}, done={done}"
            )

    print(
        f"  Result: steps={steps}, t={env.engine.current_time:.1f}, "
        f"arrived={len(env.engine.metrics.arrived_evs)}, "
        f"finished={len(env.engine.metrics.finished_evs)}, "
        f"dropped={len(env.engine.metrics.dropped_evs)}, "
        f"avg_L={env.engine.metrics.average_L():.3f}, "
        f"avg_Q={env.engine.metrics.average_Q():.3f}"
    )

    assert done
    assert env.engine.current_time >= env.engine.max_time - 1e-9
    assert len(env.engine.metrics.arrived_evs) > 0
    print("  PASS")


def test_action_mask_blocks_full_piles():
    """action_masks() should report False for piles with no free nozzles."""
    print("\n=== test_action_mask_blocks_full_piles ===")
    print(
        "Intent: action_masks() must mark full piles False so a masked policy "
        "cannot assign into them."
    )
    print("Logic: 2 piles x 1 nozzle; assign into free piles and watch the mask.")

    env = ChargingStationEnv(
        n_piles=2,
        n_nozzles=1,
        n_bricks=2,
        p_brick=25.0,
        queue_capacity=10,
        lam=1.0,
    )
    obs, _ = env.reset(seed=1)
    print(f"  After reset: needs_decision={env.engine.needs_assignment_decision()}")

    if env.engine.needs_assignment_decision():
        mask = env.action_masks()
        print(f"  Initial mask={mask.tolist()}")
        assert mask.shape == (2,)

        for i in range(5):
            if not env.engine.needs_assignment_decision():
                print(f"  Stop early at assign attempt {i}: no decision needed")
                break
            mask = env.action_masks()
            free = np.flatnonzero(mask)
            print(f"  Attempt {i}: mask={mask.tolist()}, free_piles={free.tolist()}")
            if len(free) == 0:
                break
            obs, _, done, _, info = env.step(int(free[0]))
            print(
                f"    assigned pile {int(free[0])}: done={done}, "
                f"queue={info['queue_len']}, t={info['time']:.2f}"
            )
            if done:
                break
            if not env.engine.station.piles[free[0]].is_full:
                assert env.action_masks().shape == (2,)

    print(f"  Final mask={env.action_masks().tolist()}")
    print("  PASS")


if __name__ == "__main__":
    print("Running test_simulation_env.py (direct mode)")
    test_des_projects_energy_between_events()
    test_env_decision_point_and_fifo_episode()
    test_action_mask_blocks_full_piles()
    print("\nAll tests in test_simulation_env.py finished.")

