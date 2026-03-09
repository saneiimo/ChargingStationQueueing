from __future__ import annotations
from .base import QueuePolicy
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np


class FIFOQueuePolicy(QueuePolicy):

    def select_pile(
        self, q_len: int, pile_state: List[int], rng: np.random.Generator
    ) -> int:

        num_piles = len(pile_state) // 2
        pile_loads = pile_state[:num_piles]
        pile_nozzles = pile_state[num_piles:]

        # If queue is empty, we must advance time.
        if q_len == 0:
            return num_piles

        nozzle_capacity = [
            nozzle - num_evs for nozzle, num_evs in zip(pile_nozzles, pile_loads)
        ]

        # If queue has cars but all piles are full, we must advance time.
        if sum(nozzle_capacity) == 0:
            return num_piles

        # Standard FIFO logic: pick the least occupied pile
        max_cap = max(nozzle_capacity)
        best_piles = [
            idx for idx, nozzle in enumerate(nozzle_capacity) if nozzle == max_cap
        ]

        return rng.choice(best_piles)
