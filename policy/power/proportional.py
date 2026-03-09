from __future__ import annotations
from .base import PowerPolicy
from math import ceil
import numpy as np
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from models.pile import ChargingPile
    from models.ev import EV


class ProportionalPower(PowerPolicy):

    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
        # kind: Literal["micro"] | None = None,
    ):
        assignments = []
        if ev:
            self._micro_distribute(pile, ev)
        else:
            self._distribute(pile)

        for ev in pile.evs:
            power = pile.ev_bricks[ev.nozzle_id] * pile.p_brick
            assignments.append((ev, power))
        return assignments

    def _distribute(self, pile: ChargingPile):
        """
        Takes a power requests, assigns power bricks based on the weighted ratio of the
        power requests to EVs, there is one edge case: the total power req might be less than
        the available supply but still the pile is_overloaded(), since power is assigned in the
        form of whole bricks. For example there are 4 * 50kW power bricks, and the requests
        are 105 kW and 90 kW, the first needs 3 and the second needs 2 power bricks but there
        are 4 power bricks are available.
        """
        if pile.is_overloaded:

            total_req = sum(pile.power_reqs)

            allocatable_power = min(total_req, pile.power_supp)

            pile.ev_bricks = []
            for ev in pile.evs:
                share = ev.p_req / total_req
                ev_power = share * allocatable_power
                bricks = int(ev_power / pile.p_brick)
                pile.ev_bricks.append(max(1, bricks))
                # At least assign one brick to each request

            self._micro_distribute(pile)

        else:

            pile.ev_bricks = [ceil(ev.p_req / pile.p_brick) for ev in pile.evs]

    def _micro_distribute(self, pile: ChargingPile, ev: EV | None = None):
        """
        Do a microDistribution, which assigns any remaining power brick to
        the vehicle most needing it.
        """
        if ev:  # Unassign brick from the EV underultilizing it
            pile.ev_bricks[ev.nozzle_id] -= 1
        while sum(pile.ev_bricks) < pile.num_bricks:
            idx = np.argmax(pile.remaining_reqs)
            pile.ev_bricks[idx] += 1
