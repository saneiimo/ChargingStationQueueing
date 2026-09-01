"""
Static (equal) module sharing on a single pile.

Rule
----
1. Split the module pool as evenly as possible among plugged EVs:
   each gets ``num_modules // n_evs`` modules.
2. Hand leftover modules (``num_modules % n_evs``) **once**, in descending
   order of unmet request after the equal base
   ``remaining = p_req - base * p_module``.
   Each leftover EV gets at most one extra module (so leftovers never
   re-stack on the same EV in one update).

Example (5 modules, ``p_module = 25`` kW, three EVs requesting 75 / 250 / 200 kW)::

    base = 5 // 3 = 1 module each → unmet 50, 225, 175
    leftovers = 2 → ranked 225 then 175 (not re-greedy; 250 would win twice)
    final modules: [1, 2, 2] for the 75 / 250 / 200 kW EVs

With a single plugged EV the equal share is the whole pool
(``num_modules // 1``). Empty piles clear all module slots.

Underuse / CHARGE_CHANGE
------------------------
Static keeps the equal split until the plugged set changes (plug-in or
departure). It does **not** free underused modules, so
``supports_underuse_reallocation`` stays ``False`` and ``EV.next_state``
never schedules ``CHARGE_CHANGE`` for this policy. Rebuilding the equal
split on underuse would hand the module back and loop forever in the DES.

``ev`` on ``update_power`` is ignored if ever passed; allotment is always a
full equal rebuild from the current plugged EVs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import PowerPolicy

if TYPE_CHECKING:
    from models.pile import ChargingPile
    from models.ev import EV


class StaticPower(PowerPolicy):
    """Equal base modules per plugged EV; leftovers by ranked unmet ``p_req``."""

    # Fixed split until composition changes — do not schedule CHARGE_CHANGE.
    supports_underuse_reallocation = False

    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
    ) -> list[tuple[EV, float]]:
        # ``ev`` unused: Static only rebuilds on plug/unplug (ev is None).
        del ev

        if not pile.evs:
            pile.ev_modules = [0] * pile.n_connectors
            return []

        self._distribute(pile)
        pile.check_invariants()

        assignments: list[tuple[EV, float]] = []
        for connected in pile.evs:
            power = pile.ev_modules[connected.connector_id] * pile.p_module
            assignments.append((connected, power))
        return assignments

    def _clear_modules(self, pile: ChargingPile) -> None:
        pile.ev_modules = [0] * pile.n_connectors

    def _distribute(self, pile: ChargingPile) -> None:
        """Assign equal base modules, then one leftover each by unmet rank."""
        self._clear_modules(pile)
        n = len(pile.evs)
        if n == 0:
            return

        # Safe: station requires num_modules >= n_connectors >= n plugged EVs.
        base = pile.num_modules // n
        leftovers = pile.num_modules - base * n
        for connected in pile.evs:
            pile.ev_modules[connected.connector_id] = base

        if leftovers == 0:
            return

        # Rank once by unmet request after the equal base (descending).
        # Ties broken by connector_id so the assignment is deterministic.
        ranked = sorted(
            pile.evs,
            key=lambda e: (
                -(e.p_req - base * pile.p_module),
                e.connector_id,
            ),
        )
        for connected in ranked[:leftovers]:
            pile.ev_modules[connected.connector_id] += 1
