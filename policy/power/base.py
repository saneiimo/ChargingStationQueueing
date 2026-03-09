from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from models.pile import ChargingPile
    from models.ev import EV


class PowerPolicy(ABC):

    @abstractmethod
    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
        # kind: str | None = None,
    ):
        pass

    # @abstractmethod
    # def _distribute(self, pile: ChargingPile):
    #     pass

    # @abstractmethod
    # def _micro_distribute(self, pile: ChargingPile, ev: EV | None = None):
    #     pass
