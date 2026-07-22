"""
Join the pile with the most free nozzles (random tie-break).

This is a lightweight stand-in for "FIFO / join-the-shortest-queue" style
routing. It reads free_frac from the ChargingStationEnv observation layout:

  per pile: [occ_frac, free_frac, overload, p_req_frac, brick_frac, mean_soc]
  then HOL features, then queue_frac.
"""

from __future__ import annotations
from .base import QueuePolicy
import numpy as np


class FIFOQueuePolicy(QueuePolicy):

    FEATURES_PER_PILE = 6

    def select_pile(
        self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
    ) -> int:
        n_piles = len(action_mask)
        free_fracs = [
            float(obs[i * self.FEATURES_PER_PILE + 1]) for i in range(n_piles)
        ]

        candidates = [i for i, ok in enumerate(action_mask) if ok]
        if not candidates:
            raise RuntimeError("FIFOQueuePolicy called with no valid piles")

        best_free = max(free_fracs[i] for i in candidates)
        best = [i for i in candidates if free_fracs[i] == best_free]
        return int(rng.choice(best))
