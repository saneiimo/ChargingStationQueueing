from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    import numpy as np


class QueuePolicy(ABC):

    @abstractmethod
    def select_pile(
        self, q_len: int, pile_state: List[int], rng: np.random.Generator
    ) -> int:
        pass
