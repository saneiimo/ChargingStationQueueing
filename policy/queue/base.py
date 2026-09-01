"""
Baseline (non-RL) rules for queue assignment.

Assignment has two parts:

1. **Which waiting EV to serve** — ``select_ev(station)``. Optional ``max_wait``
   on the base class serves overdue customers first (longest wait wins).
   Otherwise subclasses implement ``_select_ev``.
2. **Which pile to use** — ``select_pile(obs, mask, rng)``. Default: join the
   pile with the most free connectors (random tie-break). Subclasses may override.

Heuristics should call ``decide(...)`` then pass both results into the env/engine:

    ev, pile_id = policy.decide(obs, mask, rng, station)
    env.step(pile_id, ev=ev)

RL agents typically only choose the pile and leave ``ev=None`` (head-of-line).
"""

from __future__ import annotations
from abc import ABC
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from models.ev import EV
    from models.station import ChargingStation


class QueuePolicy(ABC):
    """Shared max-wait override + default free-connector pile routing."""

    FEATURES_PER_PILE = 6

    def __init__(self, max_wait: float | None = None):
        """
        Parameters
        ----------
        max_wait :
            Queue-wait threshold (minutes). If set, any EV with
            ``current_time - arrival_time > max_wait`` is served before the
            subclass rule; among overdue EVs, longest wait wins.
            ``None`` disables this override.
        """
        self.max_wait = max_wait

    def select_ev(self, station: ChargingStation) -> EV:
        """Choose which waiting EV to assign next (does not remove it)."""
        queue = station.queue
        if not queue:
            raise RuntimeError(
                f"{type(self).__name__}.select_ev called with empty queue"
            )

        now = station.current_time
        if self.max_wait is not None:
            overdue = [ev for ev in queue if (now - ev.arrival_time) > self.max_wait]
            if overdue:
                return max(overdue, key=lambda ev: now - ev.arrival_time)

        return self._select_ev(station)

    def _select_ev(self, station: ChargingStation) -> EV:
        """Subclass customer discipline when no one is overdue. Default: FIFO HOL."""
        return station.queue[0]

    def select_pile(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        rng: np.random.Generator,
        station: ChargingStation | None = None,
    ) -> int:
        """
        Choose a pile index allowed by ``action_mask``.

        Default: most free connectors (obs free_frac), random among ties.
        ``station`` is accepted for API symmetry / future pile rules.
        """
        del station  # unused by the default rule
        n_piles = len(action_mask)
        free_fracs = [
            float(obs[i * self.FEATURES_PER_PILE + 1]) for i in range(n_piles)
        ]
        candidates = [i for i, ok in enumerate(action_mask) if ok]
        if not candidates:
            raise RuntimeError(f"{type(self).__name__} called with no valid piles")

        best_free = max(free_fracs[i] for i in candidates)
        best = [i for i in candidates if free_fracs[i] == best_free]
        return int(rng.choice(best))

    def decide(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        rng: np.random.Generator,
        station: ChargingStation,
    ) -> tuple[EV, int]:
        """Pick EV and pile together for a heuristic assignment step."""
        ev = self.select_ev(station)
        pile_id = self.select_pile(obs, action_mask, rng, station=station)
        return ev, pile_id
