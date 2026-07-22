"""
Baseline (non-RL) rules for choosing which pile gets the head-of-line EV.

The Gym env exposes the same decision: given an observation and an action mask
of free piles, return a pile index. RL agents replace this class; FIFO is the
simple benchmark in main.py.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


class QueuePolicy(ABC):

    @abstractmethod
    def select_pile(
        self, obs: np.ndarray, action_mask: np.ndarray, rng: np.random.Generator
    ) -> int:
        """Return a pile index in 0..n_piles-1 that is allowed by action_mask."""
        pass
