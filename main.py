# main.py

import numpy as np
from env.charging_env import ChargingStationEnv
from policy.queue.fifo import FIFOQueuePolicy

env = ChargingStationEnv(
    n_piles=4, n_nozzle=2, n_brick=5, p_brick=25, c_q=10, lam=5.0, max_time=1440
)

obs, _ = env.reset(seed=42)
done = False
rng = np.random.default_rng(1)
policy = FIFOQueuePolicy()

while not done:
    # obs is stored as [piles_loads + pile_nozzles + [queue_len]]
    # Get queue_len
    q_len = obs[-1]
    # Get pile states
    piles_state = obs[:-1]
    action = policy.select_pile(q_len, piles_state, rng)
    obs, reward, done, _, _ = env.step(action)
