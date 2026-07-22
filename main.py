"""
Roll out one simulated day with the FIFO pile-choice baseline.

Useful as a smoke test after changing the station model or the Gym wrapper.
For training, replace FIFOQueuePolicy with a masked RL policy that reads
env.action_masks().
"""

import numpy as np
from env.charging_env import ChargingStationEnv
from policy.queue.fifo import FIFOQueuePolicy

env = ChargingStationEnv(
    n_piles=4,
    n_nozzles=2,
    n_bricks=5,
    p_brick=25,
    queue_capacity=10,
    lam=5.0,
)

obs, info = env.reset(seed=42)
done = False
rng = np.random.default_rng(1)
policy = FIFOQueuePolicy()
total_reward = 0.0

while not done:
    mask = env.action_masks()
    action = policy.select_pile(obs, mask, rng)
    obs, reward, done, _, info = env.step(action)
    total_reward += reward

print(f"Finished EVs: {len(env.engine.metrics.finished_evs)}")
print(f"Dropped EVs: {len(env.engine.metrics.dropped_evs)}")
print(f"Avg L: {env.engine.metrics.average_L():.3f}")
print(f"Avg Q: {env.engine.metrics.average_Q():.3f}")
print(f"Total reward: {total_reward:.3f}")
print(f"Sim time: {env.engine.current_time:.1f}")
