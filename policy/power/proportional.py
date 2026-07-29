"""
Proportional brick sharing on a single pile.

Not overloaded: each EV gets ceil(p_req / p_brick) bricks (fits by definition).
Overloaded: give bricks in proportion to p_req (at least one each), repair if
we overshot the pile's brick count, then hand leftover bricks to whoever still
has the largest unmet request (_micro_distribute).

When the engine passes the triggering EV (CHARGE_CHANGE), we free one brick
from that EV first, then run the same leftover fill.
"""

from __future__ import annotations
from .base import PowerPolicy
from math import ceil
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.pile import ChargingPile
    from models.ev import EV


class ProportionalPower(PowerPolicy):

    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
    ) -> list[tuple[EV, float]]:
        if not pile.evs:
            pile.ev_bricks = [0] * pile.n_nozzles
            return []

        if ev is not None and pile.is_overloaded:
            self._micro_distribute(pile, ev)
        else:
            self._distribute(pile)

        pile.check_invariants()

        assignments = []
        for connected in pile.evs:
            power = pile.ev_bricks[connected.nozzle_id] * pile.p_brick
            assignments.append((connected, power))
        return assignments

    def _clear_bricks(self, pile: ChargingPile) -> None:
        pile.ev_bricks = [0] * pile.n_nozzles

    def _distribute(self, pile: ChargingPile) -> None:
        """Full rebuild of brick counts from current requests."""
        self._clear_bricks(pile)

        if pile.is_overloaded:
            # Overloaded: demand exceeds the brick pool.
            # Guarantee one brick per plugged EV (safe: num_bricks >= n_nozzles >= n_evs),
            # then greedily hand remaining bricks to the largest unmet p_req.
            total_req = sum(ev.p_req for ev in pile.evs)
            if total_req <= 0:
                raise ValueError(
                    f"Overloaded pile {pile.id} has total_req={total_req} <= 0"
                )
            for connected in pile.evs:
                pile.ev_bricks[connected.nozzle_id] = 1
            self._micro_distribute(pile)  # fills up to num_bricks
        else:
            # Not overloaded: each EV can take its isolated ceil(p_req / p_brick).
            for connected in pile.evs:
                pile.ev_bricks[connected.nozzle_id] = ceil(
                    connected.p_req / pile.p_brick
                )

    def _micro_distribute(self, pile: ChargingPile, ev: EV | None = None) -> None:
        """
        Give any free bricks to the nozzle with the largest remaining request.
        Optional `ev`: drop one of that EV's bricks first (underutilization path).
        """
        if not pile.evs:
            self._clear_bricks(pile)
            return

        if ev is not None:
            if ev.nozzle_id is None or pile.nozzles[ev.nozzle_id] is not ev:
                raise ValueError("CHARGE_CHANGE EV is not on this pile")
            if pile.ev_bricks[ev.nozzle_id] > 0:
                pile.ev_bricks[ev.nozzle_id] -= 1

        while sum(pile.ev_bricks) < pile.num_bricks:
            reqs = pile.remaining_reqs
            idx = int(np.argmax(reqs))
            if not np.isfinite(reqs[idx]):
                break
            pile.ev_bricks[idx] += 1
