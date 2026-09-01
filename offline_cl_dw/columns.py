"""
Columns ("plans") for the Dantzig-Wolfe decomposition -- Section 4 of
``dantzig_wolfe_decomposition.html``.

A plan is a complete, self-contained description of one vehicle's stay,
ignoring every other vehicle: which pile, when it plugs in, when it departs,
and its power profile in between. It is exactly one feasible point of the
compact model's per-vehicle constraints (Groups A, D, E, plus (13)) -- see
``pricer.py``, which builds and solves exactly that restricted model to
generate plans with negative reduced cost.
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

    The null plan (never served, eq. after (21)) is represented by
    ``pile=None``, ``start=None``, ``departure=K``, ``power={}`` -- see
    ``null_plan``. Every vehicle's column pool always contains exactly one
    of these, seeded once at initialisation (Section 7.2), and it is what
    makes the restricted master problem feasible from the very first
    iteration with no artificial variables needed.
    """

    vehicle_id: int
    pile: int | None
    start: int | None
    departure: int  # D_omega; K for the null plan
    power: dict[int, float] = field(default_factory=dict)  # {slot: kW}, occupied slots only

    @property
    def is_null(self) -> bool:
        return self.pile is None

    def occupied_slots(self) -> range:
        """``[S, D)`` -- empty (``range(0, 0)``) for the null plan."""
        if self.is_null:
            return range(0, 0)
        return range(self.start, self.departure)  # type: ignore[arg-type]

    def alpha(self, pile: int, k: int) -> int:
        """(22): 1 iff this plan occupies a connector on ``pile`` at slot ``k``."""
        if self.is_null or pile != self.pile:
            return 0
        return 1 if self.start <= k < self.departure else 0  # type: ignore[operator]

    def beta(self, pile: int, k: int) -> float:
        """(23): the power this plan draws on ``pile`` at slot ``k`` (0 if idle/elsewhere)."""
        if self.is_null or pile != self.pile:
            return 0.0
        return self.power.get(k, 0.0)

    def energy_kwh(self, h: float) -> float:
        """Total delivered energy, kWh -- should equal the vehicle's own
        ``W_j`` for any non-null plan (completion at departure is part of
        every plan's feasibility, Section 4)."""
        return h * sum(self.power.values())


def null_plan(vehicle_id: int, K: int) -> Plan:
    """The distinguished never-served column, eq. after (21): no pile, no
    occupied slots, no power, ``D = K`` -- the censoring convention of the
    compact model (Section 6.3 there) expressed as a column."""
    return Plan(vehicle_id=vehicle_id, pile=None, start=None, departure=K, power={})
