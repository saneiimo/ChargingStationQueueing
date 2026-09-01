"""
Serve the waiting EV with the smallest remaining SoC gap first.

SoC difference is ``s_f - s_current``. Pile routing (most free connectors) and
optional ``max_wait`` overdue override come from ``QueuePolicy``.

Usage::

    ev, pile_id = policy.decide(obs, mask, rng, station=env.engine.station)
    env.step(pile_id, ev=ev)
"""

from __future__ import annotations
from .base import QueuePolicy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.ev import EV
    from models.station import ChargingStation


class LowestSoCDiffQueuePolicy(QueuePolicy):
    """Lowest remaining SoC gap among waiting EVs; default free-connector piles."""

    def _select_ev(self, station: ChargingStation) -> EV:
        queue = station.queue
        return min(
            queue,
            key=lambda ev: (ev.s_f - ev.s_current, ev.arrival_time),
        )
