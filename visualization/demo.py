"""Small helpers to roll out demo episodes for plotting."""

from __future__ import annotations

import numpy as np

from env.charging_env import ChargingStationEnv
from policy.queue.fifo import FIFOQueuePolicy


def run_fifo_episode(
    n_piles: int = 4,
    n_connectors: int = 2,
    n_modules: int = 5,
    p_module: float = 25.0,
    queue_capacity: int = 10,
    mean_interarrival: float = 5.0,
    seed: int = 42,
    policy_seed: int = 1,
) -> ChargingStationEnv:
    """Roll out one full day under FIFO pile choice (records charge traces)."""
    env = ChargingStationEnv(
        n_piles=n_piles,
        n_connectors=n_connectors,
        n_modules=n_modules,
        p_module=p_module,
        queue_capacity=queue_capacity,
        mean_interarrival=mean_interarrival,
    )
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(policy_seed)
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
