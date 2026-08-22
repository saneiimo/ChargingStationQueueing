"""
Roll out one simulated day with a queue-assignment baseline.

Useful as a smoke test after changing the station model or the Gym wrapper.
For training, replace the heuristic with a masked RL policy that reads
env.action_masks() and calls env.step(pile_id) (head-of-line EV).
"""

import numpy as np
from env.charging_env import ChargingStationEnv
from policy.queue.fifo import FIFOQueuePolicy

env = ChargingStationEnv(
    n_piles=4,
    n_dispensers=2,
    n_modules=5,
    p_module=25,
    queue_capacity=10,
    mean_interarrival=5.0,
)

obs, info = env.reset(seed=42)
done = False
rng = np.random.default_rng(1)
policy = FIFOQueuePolicy()
total_reward = 0.0

while not done:
    mask = env.action_masks()
    ev, pile_id = policy.decide(obs, mask, rng, env.engine.station)
    obs, reward, done, _, info = env.step(pile_id, ev=ev)
    total_reward += reward

print(f"Finished EVs: {len(env.engine.metrics.finished_evs)}")
print(f"Dropped EVs: {len(env.engine.metrics.dropped_evs)}")
print(f"Avg L: {env.engine.metrics.average_L():.3f}")
print(f"Avg Q: {env.engine.metrics.average_Q():.3f}")
print(f"Total reward: {total_reward:.3f}")
print(f"Sim time: {env.engine.current_time:.1f}")
