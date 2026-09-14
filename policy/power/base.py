"""
Interface for how a pile splits its power modules among plugged EVs.

The SimulationEngine calls ``update_power`` after plug-in, departure, or a
``CHARGE_CHANGE``. Implementations write ``pile.ev_modules[connector_id]`` and
return ``(EV, allotted_kW)`` pairs so the engine can update each EV's ``p_act``.

``CHARGE_CHANGE`` (underuse reallocation) is **opt-in**. Set
``supports_underuse_reallocation = True`` only if ``update_power(pile, ev=...)``
actually frees / reassigns an underused module. Policies that keep a fixed split
until the plugged set changes (e.g. Static) must leave the flag ``False`` so
``EV.next_state`` schedules ``DEPARTURE`` instead — otherwise equal rebuilds
hand the module back and the DES loops forever on ``CHARGE_CHANGE``.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.pile import ChargingPile
    from models.ev import EV


class PowerPolicy(ABC):
    """
    Module-sharing rule used inside DES events.

    Display names for experiments are *not* stored on the instance. Pass a
    parallel ``power_names`` list (same idea as ``queue_names``) into
    ``experiments.policy_sweep.labeled_policy_grid``.

    Class attributes
    ----------------
    supports_underuse_reallocation :
        If True, overloaded EVs with more than one module may schedule
        ``CHARGE_CHANGE`` so the policy can free an underused module.
        Default False; Proportional opts in.
    """

    # Opt-in: EV.next_state only schedules CHARGE_CHANGE when this is True.
    supports_underuse_reallocation: bool = False

    @abstractmethod
    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
    ) -> list[tuple[EV, float]]:
        """
        Recompute module allotment on `pile`.

        If `ev` is set, this is a micro-update triggered by that EV underusing
        a module (only meaningful when ``supports_underuse_reallocation``).
        Otherwise rebuild the allotment from scratch (arrival/departure).
        """
        pass
