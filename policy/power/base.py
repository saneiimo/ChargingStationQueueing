"""
Interface for how a pile splits its power bricks among plugged EVs.

The SimulationEngine calls update_power after plug-in, departure, or a
CHARGE_CHANGE. Implementations write pile.ev_bricks[nozzle_id] and return
(EV, allotted_kW) pairs so the engine can update each EV's p_act.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.pile import ChargingPile
    from models.ev import EV


class PowerPolicy(ABC):

    @abstractmethod
    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
    ) -> list[tuple[EV, float]]:
        """
        Recompute brick allotment on `pile`.

        If `ev` is set, this is a micro-update triggered by that EV underusing
        a brick. Otherwise rebuild the allotment from scratch (arrival/departure).
        """
        pass
