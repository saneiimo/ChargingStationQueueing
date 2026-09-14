"""
Constant (fixed, per-connector) module sharing on a single pile.

Rule
----
Every connector owns a FIXED share of the module pool, partitioned once
from the pile's own geometry (``num_modules // n_connectors``) and never
touched again -- a plugged EV can never draw more than its own connector's
fixed share, however many neighbouring connectors sit idle.

Unlike Static (which repartitions the whole pool among however many EVs are
CURRENTLY plugged in) or Proportional (which shares based on request),
Constant never redistributes: an idle connector's modules stay reserved to
that connector and simply go unused, never lent to a busy neighbour.

Example: a pile with 2 connectors and 6 modules gives each connector a
fixed share of ``6 // 2 = 3`` modules, always -- whether 1 EV is plugged in
or 2. With an uneven split (e.g. 7 modules, 2 connectors: base 3 each, 1
leftover), the leftover module goes to the lowest-numbered connector(s),
fixed once from pile geometry -- there is nothing to rank by occupancy,
since the split does not depend on occupancy at all.

A connector's allotment is not capped to what its EV actually needs
(``p_req``) -- same convention as Static. ``EV.p_act = min(p_req, power)``
downstream already clamps delivered power to the request, so an
over-sized allotment is unused headroom, never overcharging.

Underuse / CHARGE_CHANGE
------------------------
The split never changes after a pile is constructed, let alone in response
to underuse, so ``supports_underuse_reallocation`` stays ``False`` and
``EV.next_state`` never schedules ``CHARGE_CHANGE`` for this policy --
there is nothing for a rebuild to change.

``ev`` on ``update_power`` is ignored if ever passed; allotment is always
read straight off pile geometry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .base import PowerPolicy

if TYPE_CHECKING:
    from models.ev import EV
    from models.pile import ChargingPile


class ConstantPower(PowerPolicy):
    """Each connector's module share is fixed by pile geometry, never by occupancy."""

    # The split is pinned to pile geometry, never revisited on underuse --
    # do not schedule CHARGE_CHANGE.
    supports_underuse_reallocation = False

    def update_power(
        self,
        pile: ChargingPile,
        ev: EV | None = None,
    ) -> list[tuple[EV, float]]:
        # ``ev`` unused: the split never depends on who is plugged in.
        del ev

        caps = self._connector_caps(pile)
        pile.ev_modules = [
            caps[i] if pile.connectors[i] is not None else 0
            for i in range(pile.n_connectors)
        ]
        pile.check_invariants()

        return [
            (connected, pile.ev_modules[connected.connector_id] * pile.p_module)
            for connected in pile.evs
        ]

    @staticmethod
    def _connector_caps(pile: ChargingPile) -> list[int]:
        """
        Each connector's fixed module share, from pile geometry alone.

        Base share for every connector, plus one leftover module each for
        the lowest-numbered connectors when ``num_modules`` does not
        divide evenly. Deterministic and independent of who (if anyone) is
        plugged in -- matches this policy's whole point.
        """
        n = pile.n_connectors
        base, leftover = divmod(pile.num_modules, n)
        return [base + (1 if i < leftover else 0) for i in range(n)]
