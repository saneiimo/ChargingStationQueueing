"""
Assign waiting EVs to piles by closest remaining-power match.

At each assignment decision:

1. For every pile with a free nozzle, compute leftover supply::

       remaining = pile.power_supp - sum(p_act of plugged EVs)

   (``power_supp`` is the pile's brick pool in kW; ``p_act`` is what each
   plugged EV is drawing right now.)

2. For each candidate waiting EV, read its BMS request ``p_req`` at the
   current SoC.

3. Choose the (EV, pile) pair that minimizes ``|remaining - p_req|``.
   Ties break by earlier arrival, then randomly.

Customer set:

* If ``max_wait`` is set and some EVs are overdue, only those are candidates
  (same overdue idea as ``QueuePolicy.select_ev``).
* Otherwise every EV in the queue is a candidate.

Usage::

    from policy.queue.closest_power_match import ClosestPowerMatchQueuePolicy

    policy = ClosestPowerMatchQueuePolicy()
    ev, pile_id = policy.decide(obs, mask, rng, station=env.engine.station)
    env.step(pile_id, ev=ev)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import QueuePolicy

if TYPE_CHECKING:
    from models.ev import EV
    from models.pile import ChargingPile
    from models.station import ChargingStation


def pile_remaining_power(pile: ChargingPile) -> float:
    """
    Leftover pile power (kW) after what plugged EVs are currently drawing.

    remaining = max_power - sum(p_act). Clamped at 0 so tiny float noise
    cannot produce a negative leftover.
    """
    drawn = sum(ev.p_act for ev in pile.evs)
    return max(pile.power_supp - drawn, 0.0)


class ClosestPowerMatchQueuePolicy(QueuePolicy):
    """
    Match a waiting EV to a free pile whose leftover power is closest to
    that EV's ``p_req``.
    """

    def _candidate_evs(self, station: ChargingStation) -> list[EV]:
        """
        Waiting EVs eligible for this assignment step.

        Honours ``max_wait``: if anyone is overdue, only overdue EVs may be
        matched; otherwise the whole queue is eligible.
        """
        queue = list(station.queue)
        if not queue:
            return []

        if self.max_wait is None:
            return queue

        now = station.current_time
        overdue = [ev for ev in queue if (now - ev.arrival_time) > self.max_wait]
        return overdue if overdue else queue

    def _best_pair(
        self,
        station: ChargingStation,
        action_mask: np.ndarray,
        rng: np.random.Generator,
        *,
        candidate_evs: list[EV] | None = None,
    ) -> tuple[EV, int]:
        """
        Pick (EV, pile_id) minimizing |remaining_power - p_req|.

        Only piles with ``action_mask[i]`` True are considered (free nozzle).
        """
        evs = self._candidate_evs(station) if candidate_evs is None else candidate_evs
        if not evs:
            raise RuntimeError(
                f"{type(self).__name__}: no waiting EVs to match"
            )

        piles = station.piles
        candidates: list[tuple[float, float, int, EV]] = []
        # Tuple key: (distance, arrival_time, pile_id, ev) for stable sort;
        # we collect ties then rng.choice among the best distance/arrival set.

        for pile_id, ok in enumerate(action_mask):
            if not ok:
                continue
            remaining = pile_remaining_power(piles[pile_id])
            for ev in evs:
                dist = abs(remaining - float(ev.p_req))
                candidates.append((dist, float(ev.arrival_time), pile_id, ev))

        if not candidates:
            raise RuntimeError(
                f"{type(self).__name__} called with no valid piles"
            )

        # Closest power distance first; earlier arrival breaks residual ties.
        best_dist, best_arr, _, _ = min(candidates, key=lambda x: (x[0], x[1]))
        tied = [
            (pile_id, ev)
            for dist, arr, pile_id, ev in candidates
            if dist == best_dist and arr == best_arr
        ]
        pile_id, ev = tied[int(rng.integers(0, len(tied)))]
        return ev, int(pile_id)

    def select_pile(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        rng: np.random.Generator,
        station: ChargingStation | None = None,
    ) -> int:
        """
        Match head-of-line to the pile whose leftover power is closest to
        ``p_req``.

        Prefer ``decide(...)``, which searches over all eligible waiting EVs.
        This method needs a live ``station`` (leftover power is not in ``obs``).
        """
        del obs
        if station is None:
            raise ValueError(
                f"{type(self).__name__}.select_pile needs station to compute "
                "remaining pile power; pass station=... or use decide(...)."
            )
        if not station.queue:
            raise RuntimeError(
                f"{type(self).__name__}.select_pile called with empty queue"
            )

        # Single-EV path: HOL only. Full queue matching lives in decide().
        hol = station.queue[0]
        _, pile_id = self._best_pair(
            station, action_mask, rng, candidate_evs=[hol]
        )
        return pile_id

    def decide(
        self,
        obs: np.ndarray,
        action_mask: np.ndarray,
        rng: np.random.Generator,
        station: ChargingStation,
    ) -> tuple[EV, int]:
        """
        Jointly choose waiting EV and pile by closest remaining-power match.

        ``obs`` is unused; leftover power and ``p_req`` come from ``station``.
        """
        del obs
        return self._best_pair(station, action_mask, rng)
