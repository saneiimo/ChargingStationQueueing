"""
Branching restrictions: the structural attributes a node fixes about each
vehicle's plan.

Every restriction is on an attribute of the *plan itself* (departure,
start, pile, a slot's module count), never on a master variable, so the
same restriction can be (a) tested against any stored column
(``NodeRestrictions.column_satisfies``) and (b) imposed inside that
vehicle's pricing MILP (``pricer.PBPricer.apply``). Together (a) and (b)
make the pricer's feasible set, plus the closed-form null plan, exactly the
set of columns the node allows.

Null-plan semantics (the censoring convention): ``S = D = K``, no pile, and
``q_k = 0`` in every slot. So the null plan satisfies a restriction iff
``K`` is inside both the departure and start ranges, no pile is required,
and no slot has a positive module lower bound.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .columns import PBPlan


@dataclass(frozen=True)
class VehicleRestriction:
    """One vehicle's accumulated branching restrictions (``None`` = no bound)."""

    d_min: int | None = None
    d_max: int | None = None
    s_min: int | None = None
    s_max: int | None = None
    pile_only: int | None = None
    piles_forbidden: frozenset[int] = frozenset()
    q_min: dict[int, int] = field(default_factory=dict)  # slot -> lower bound on q_jk
    q_max: dict[int, int] = field(default_factory=dict)  # slot -> upper bound on q_jk

    def allows_pile(self, pile: int) -> bool:
        return (self.pile_only is None or pile == self.pile_only) and pile not in self.piles_forbidden

    def _in_range(self, value: int, lo: int | None, hi: int | None) -> bool:
        return (lo is None or value >= lo) and (hi is None or value <= hi)

    def satisfied_by(self, plan: PBPlan, K: int) -> bool:
        """Whether ``plan`` meets every restriction (null plan: S = D = K, q = 0)."""
        start = plan.start_slot(K)
        if not self._in_range(plan.departure, self.d_min, self.d_max):
            return False
        if not self._in_range(start, self.s_min, self.s_max):
            return False
        if plan.is_null:
            if self.pile_only is not None:
                return False
        elif not self.allows_pile(plan.pile):  # type: ignore[arg-type]
            return False
        for k, lo in self.q_min.items():
            if plan.q(k) < lo:
                return False
        for k, hi in self.q_max.items():
            if plan.q(k) > hi:
                return False
        return True

    def allows_null(self, K: int, vehicle_id: int = -1) -> bool:
        from .columns import null_pb_plan

        return self.satisfied_by(null_pb_plan(vehicle_id, K), K)

    # --- tightening (each returns a new, stricter restriction) --------------
    def with_d_max(self, t: int) -> "VehicleRestriction":
        return replace(self, d_max=t if self.d_max is None else min(self.d_max, t))

    def with_d_min(self, t: int) -> "VehicleRestriction":
        return replace(self, d_min=t if self.d_min is None else max(self.d_min, t))

    def with_s_max(self, t: int) -> "VehicleRestriction":
        return replace(self, s_max=t if self.s_max is None else min(self.s_max, t))

    def with_s_min(self, t: int) -> "VehicleRestriction":
        return replace(self, s_min=t if self.s_min is None else max(self.s_min, t))

    def with_pile_only(self, pile: int) -> "VehicleRestriction":
        if self.pile_only is not None and self.pile_only != pile:
            raise ValueError(f"pile_only already {self.pile_only}, cannot also require {pile}")
        return replace(self, pile_only=pile)

    def with_pile_forbidden(self, pile: int) -> "VehicleRestriction":
        return replace(self, piles_forbidden=self.piles_forbidden | {pile})

    def with_q_max(self, k: int, t: int) -> "VehicleRestriction":
        q_max = dict(self.q_max)
        q_max[k] = t if k not in q_max else min(q_max[k], t)
        return replace(self, q_max=q_max)

    def with_q_min(self, k: int, t: int) -> "VehicleRestriction":
        q_min = dict(self.q_min)
        q_min[k] = t if k not in q_min else max(q_min[k], t)
        return replace(self, q_min=q_min)

    def describe(self) -> str:
        parts = []
        if self.d_min is not None or self.d_max is not None:
            parts.append(f"D in [{self.d_min}, {self.d_max}]")
        if self.s_min is not None or self.s_max is not None:
            parts.append(f"S in [{self.s_min}, {self.s_max}]")
        if self.pile_only is not None:
            parts.append(f"pile == {self.pile_only}")
        if self.piles_forbidden:
            parts.append(f"pile not in {sorted(self.piles_forbidden)}")
        for k in sorted(set(self.q_min) | set(self.q_max)):
            parts.append(f"q[{k}] in [{self.q_min.get(k)}, {self.q_max.get(k)}]")
        return ", ".join(parts) if parts else "none"


EMPTY_RESTRICTION = VehicleRestriction()


@dataclass(frozen=True)
class NodeRestrictions:
    """All vehicles' restrictions at one branch-and-price node."""

    by_vehicle: dict[int, VehicleRestriction] = field(default_factory=dict)

    def get(self, vehicle_id: int) -> VehicleRestriction:
        return self.by_vehicle.get(vehicle_id, EMPTY_RESTRICTION)

    def with_vehicle(self, vehicle_id: int, restriction: VehicleRestriction) -> "NodeRestrictions":
        by_vehicle = dict(self.by_vehicle)
        by_vehicle[vehicle_id] = restriction
        return NodeRestrictions(by_vehicle)

    def column_satisfies(self, plan: PBPlan, K: int) -> bool:
        return self.get(plan.vehicle_id).satisfied_by(plan, K)

    def describe(self) -> str:
        items = [f"v{j}: {r.describe()}" for j, r in sorted(self.by_vehicle.items())]
        return "; ".join(items) if items else "root"
