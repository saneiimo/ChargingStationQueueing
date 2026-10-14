"""
Columns ("plans") for the Dantzig-Wolfe decomposition -- Section 4 of
``dantzig_wolfe_decomposition.html``.

A plan is a complete, self-contained description of one vehicle's stay,
ignoring every other vehicle: which pile, when it plugs in, when it departs,
and its power profile in between. It is exactly one feasible point of the
compact model's per-vehicle constraints (Groups A, D, E, (13) and the
completion row) -- see ``pricer.py``, which builds and solves exactly that
restricted model to generate plans with negative reduced cost.

The compact model is a closed system (``offline_cl_opt.model``): every
vehicle is served and receives its full energy within the horizon. So every
plan has a pile and a start, and delivers ``W_j`` by its departure
``D <= K``; there is no "never served" plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Plan:
    """
    One column, eq. (21): ``omega = (pile, S, D, (p_k)_{k=S}^{D-1})``.

    ``vehicle_id`` is not part of the mathematical column (a plan is
    per-vehicle by construction, so the document doesn't index it), but is
    carried here since ``Plan`` objects for different vehicles are pooled
    together in ``preprocess.py``/``colgen.py``.
    """

    vehicle_id: int
    pile: int
    start: int
    departure: int  # D_omega, at most K
    power: dict[int, float] = field(default_factory=dict)  # {slot: kW}, occupied slots only

    def occupied_slots(self) -> range:
        """``[S, D)``."""
        return range(self.start, self.departure)

    def alpha(self, pile: int, k: int) -> int:
        """(22): 1 iff this plan occupies a connector on ``pile`` at slot ``k``."""
        if pile != self.pile:
            return 0
        return 1 if self.start <= k < self.departure else 0

    def beta(self, pile: int, k: int) -> float:
        """(23): the power this plan draws on ``pile`` at slot ``k`` (0 if idle/elsewhere)."""
        if pile != self.pile:
            return 0.0
        return self.power.get(k, 0.0)

    def energy_kwh(self, h: float) -> float:
        """Total delivered energy, kWh -- equals the vehicle's own ``W_j``
        (less any energy a boundary vehicle had before t=0): completion is
        part of every plan's feasibility (Section 4)."""
        return h * sum(self.power.values())
