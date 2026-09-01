"""
Proportional module sharing on a single pile.

Not overloaded: each EV gets ceil(p_req / p_module) modules (fits by definition).
Overloaded: give modules in proportion to p_req (at least one each), repair if
we overshot the pile's module count, then hand leftover modules to whoever still
has the largest unmet request (_micro_distribute).

When the engine passes the triggering EV (CHARGE_CHANGE), we free one module
from that EV first, then run the same leftover fill.

``supports_underuse_reallocation = True`` so ``EV.next_state`` may schedule
``CHARGE_CHANGE`` while the pile is overloaded.
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
    # Free underused modules and reassign them (see _micro_distribute).
    supports_underuse_reallocation = True

    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
    ) -> list[tuple[EV, float]]:
        if not pile.evs:
            pile.ev_modules = [0] * pile.n_connectors
            return []

        if ev is not None and pile.is_overloaded:
            self._micro_distribute(pile, ev)
        else:
            self._distribute(pile)

        pile.check_invariants()

        assignments = []
        for connected in pile.evs:
            power = pile.ev_modules[connected.connector_id] * pile.p_module
            assignments.append((connected, power))
        return assignments

    def _clear_modules(self, pile: ChargingPile) -> None:
        pile.ev_modules = [0] * pile.n_connectors

    def _distribute(self, pile: ChargingPile) -> None:
        """Full rebuild of module counts from current requests."""
        self._clear_modules(pile)

        if pile.is_overloaded:
            # Overloaded: demand exceeds the module pool.
            # Guarantee one module per plugged EV (safe: num_modules >= n_connectors >= n_evs),
            # then greedily hand remaining modules to the largest unmet p_req.
            total_req = sum(ev.p_req for ev in pile.evs)
            if total_req <= 0:
                raise ValueError(
                    f"Overloaded pile {pile.id} has total_req={total_req} <= 0"
                )
            for connected in pile.evs:
                pile.ev_modules[connected.connector_id] = 1
            self._micro_distribute(pile)  # fills up to num_modules
        else:
            # Not overloaded: each EV can take its isolated ceil(p_req / p_module).
            # Epsilon guards against floating-point noise pushing a p_req that's
            # essentially a module multiple into the next ceil bucket.
            for connected in pile.evs:
                pile.ev_modules[connected.connector_id] = ceil(
                    connected.p_req / pile.p_module - 1e-9
                )

    def _micro_distribute(self, pile: ChargingPile, ev: EV | None = None) -> None:
        """
        Give any free modules to the connector with the largest remaining request.
        Optional `ev`: drop one of that EV's modules first (underutilization path).
        """
        if not pile.evs:
            self._clear_modules(pile)
            return

        if ev is not None:
            if ev.connector_id is None or pile.connectors[ev.connector_id] is not ev:
                raise ValueError("CHARGE_CHANGE EV is not on this pile")
            if pile.ev_modules[ev.connector_id] > 0:
                pile.ev_modules[ev.connector_id] -= 1

        while sum(pile.ev_modules) < pile.num_modules:
            reqs = pile.remaining_reqs
            idx = int(np.argmax(reqs))
            if not np.isfinite(reqs[idx]):
                break
            pile.ev_modules[idx] += 1
