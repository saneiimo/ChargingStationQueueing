"""
End-to-end checks: energy accrues between events, and the Gym env only stops
for an assignment when the queue and a free nozzle both exist.
"""

from __future__ import annotations

import numpy as np

from env.charging_env import ChargingStationEnv
from models.ev import EV
from models.pile import ChargingPile
from models.station import ChargingStation
from metrics.metrics_tracker import MetricsTracker
from policy.power.proportional import ProportionalPower
from policy.queue.fifo import FIFOQueuePolicy
from simulation.engine import SimulationEngine
from simulation.event import EventQueue, Event, EventType


def test_des_projects_energy_between_events():
    """After assign, advancing to departure should accrue energy and raise SoC."""
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
    assert engine.assign_ev(0)
    assert ev.pile is not None
    s0 = ev.s_current
    e0 = ev.energy_received

    # Inject a SIM_OVER far enough that the EV should depart first.
    engine.event_heap.push(Event(engine.max_time, EventType.SIM_OVER))

    # Run until departure or end.
    steps = 0
    while not engine.terminated and steps < 20:
        engine.advance_time()
        steps += 1
        if ev in metrics.finished_evs:
            break

    assert ev in metrics.finished_evs
    assert ev.s_current >= s0
    assert ev.energy_received > e0
    assert abs(ev.energy_received - ev.energy_received_2) < 1e-4


def test_env_decision_point_and_fifo_episode():
    env = ChargingStationEnv(
        n_piles=2,
        n_nozzles=2,
        n_bricks=4,
        p_brick=25.0,
        queue_capacity=8,
        lam=3.0,
    )
    obs, info = env.reset(seed=7)
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

    assert done
    assert env.engine.current_time >= env.engine.max_time - 1e-9
    assert len(env.engine.metrics.arrived_evs) > 0


def test_action_mask_blocks_full_piles():
    env = ChargingStationEnv(
        n_piles=2,
        n_nozzles=1,
        n_bricks=2,
        p_brick=25.0,
        queue_capacity=10,
        lam=1.0,
    )
    obs, _ = env.reset(seed=1)
    # Fill pile 0 if a decision is available.
    if env.engine.needs_assignment_decision():
        mask = env.action_masks()
        assert mask.shape == (2,)
        # Assign to first free pile repeatedly until one is full or episode needs advance.
        for _ in range(5):
            if not env.engine.needs_assignment_decision():
                break
            mask = env.action_masks()
            free = np.flatnonzero(mask)
            if len(free) == 0:
                break
            obs, _, done, _, _ = env.step(int(free[0]))
            if done:
                break
            if not env.engine.station.piles[free[0]].is_full:
                # May have auto-advanced and freed capacity; still check mask length.
                assert env.action_masks().shape == (2,)
