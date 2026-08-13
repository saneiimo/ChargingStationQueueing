"""
Static (equal) brick sharing on a single pile.

Rule
----
1. Split the brick pool as evenly as possible among plugged EVs:
   each gets ``num_bricks // n_evs`` bricks.
2. Hand leftover bricks (``num_bricks % n_evs``) **once**, in descending
   order of unmet request after the equal base
   ``remaining = p_req - base * p_brick``.
   Each leftover EV gets at most one extra brick (so leftovers never
   re-stack on the same EV in one update).

Example (5 bricks, ``p_brick = 25`` kW, three EVs requesting 75 / 250 / 200 kW)::

    base = 5 // 3 = 1 brick each → unmet 50, 225, 175
    leftovers = 2 → ranked 225 then 175 (not re-greedy; 250 would win twice)
    final bricks: [1, 2, 2] for the 75 / 250 / 200 kW EVs

With a single plugged EV the equal share is the whole pool
(``num_bricks // 1``). Empty piles clear all brick slots.

Underuse / CHARGE_CHANGE
------------------------
Static keeps the equal split until the plugged set changes (plug-in or
departure). It does **not** free underused bricks, so
``supports_underuse_reallocation`` stays ``False`` and ``EV.next_state``
never schedules ``CHARGE_CHANGE`` for this policy. Rebuilding the equal
split on underuse would hand the brick back and loop forever in the DES.

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
    """Equal base bricks per plugged EV; leftovers by ranked unmet ``p_req``."""

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
            pile.ev_bricks = [0] * pile.n_nozzles
            return []

        self._distribute(pile)
        pile.check_invariants()

        assignments: list[tuple[EV, float]] = []
        for connected in pile.evs:
            power = pile.ev_bricks[connected.nozzle_id] * pile.p_brick
            assignments.append((connected, power))
        return assignments

    def _clear_bricks(self, pile: ChargingPile) -> None:
        pile.ev_bricks = [0] * pile.n_nozzles

    def _distribute(self, pile: ChargingPile) -> None:
        """Assign equal base bricks, then one leftover each by unmet rank."""
        self._clear_bricks(pile)
        n = len(pile.evs)
        if n == 0:
            return

        # Safe: station requires num_bricks >= n_nozzles >= n plugged EVs.
        base = pile.num_bricks // n
        leftovers = pile.num_bricks - base * n
        for connected in pile.evs:
            pile.ev_bricks[connected.nozzle_id] = base

        if leftovers == 0:
            return

        # Rank once by unmet request after the equal base (descending).
        # Ties broken by nozzle_id so the assignment is deterministic.
        ranked = sorted(
            pile.evs,
            key=lambda e: (
                -(e.p_req - base * pile.p_brick),
                e.nozzle_id,
            ),
        )
        for connected in ranked[:leftovers]:
            pile.ev_bricks[connected.nozzle_id] += 1
