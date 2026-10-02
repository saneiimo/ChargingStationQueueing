"""
Whole-module single-vehicle plans (columns).

``PBPlan`` extends ``offline_cl_dw.columns.Plan`` -- same ``(pile, start,
departure, power)`` fields and helpers -- with the one thing an exact
decomposition needs and the continuous-module DW plan does not have: the
number of whole power modules ``q_k`` the vehicle's connector holds in each
occupied slot. The master's module row counts ``q``, not kW, so the integer
master is exactly the compact whole-module model (see ``README.md``).

A plan's master coefficients and cost depend only on its *discrete
signature* ``(pile, start, departure, q profile)``; the power profile only
has to exist. Two plans with the same signature are therefore the same
master column, which is what ``column_key`` encodes.

Null plan: ``pile=None``, ``start=None``, ``departure=K``, no power, no
modules -- the "never served" column of the censoring convention. Where a
start slot is needed (restrictions, branching) the null plan's start is
``K``, matching the compact model's ``S_j = K`` for an unserved vehicle.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from offline_cl_dw.columns import Plan
from offline_cl_opt.boundary import BoundaryMode, BoundaryVehicle
from offline_cl_opt.instance import StationSpec

from .tolerances import MODULE_EPS


@dataclass(frozen=True)
class PBPlan(Plan):
    """
    A ``Plan`` plus its whole-module profile: ``modules[k]`` is the integer
    module count on this vehicle's connector in occupied slot ``k`` (0 is
    allowed -- an occupied connector need not hold modules). Slots missing
    from ``modules`` read as 0.
    """

    modules: dict[int, int] = field(default_factory=dict)

    def start_slot(self, K: int) -> int:
        """``S`` with the compact model's convention: ``K`` for the null plan."""
        return K if self.is_null else int(self.start)  # type: ignore[arg-type]

    def q(self, k: int) -> int:
        """Modules held in slot ``k`` (0 outside ``[S, D)`` and for the null plan)."""
        if self.is_null or not (self.start <= k < self.departure):  # type: ignore[operator]
            return 0
        return int(self.modules.get(k, 0))

    def q_profile(self) -> tuple[int, ...]:
        """``(q_S, ..., q_{D-1})``; empty for the null plan."""
        return tuple(self.q(k) for k in self.occupied_slots())


def null_pb_plan(vehicle_id: int, K: int) -> PBPlan:
    """The never-served column: no pile, no occupancy, no power, ``D = K``."""
    return PBPlan(vehicle_id=vehicle_id, pile=None, start=None, departure=K, power={}, modules={})


def column_key(plan: PBPlan) -> tuple:
    """
    Deduplication key: vehicle plus discrete signature. Master coefficients
    and cost are functions of exactly this, so two plans with equal keys are
    the same master column however their power profiles differ.
    """
    if plan.is_null:
        return (plan.vehicle_id, None)
    return (plan.vehicle_id, plan.pile, plan.start, plan.departure, plan.q_profile())


def modules_needed(p_kw: float, p_module: float) -> int:
    """Fewest whole modules that can deliver ``p_kw``: ``ceil(p/Delta)``, float-guarded."""
    if p_kw <= 0.0:
        return 0
    return max(0, math.ceil(p_kw / p_module - MODULE_EPS))


def fixed_boundary_plan(bv: BoundaryVehicle, station: StationSpec) -> PBPlan:
    """
    The single, mandatory column of a FIXED-mode boundary vehicle: exactly
    the trajectory ``offline_cl_opt.model.build_cl_model`` pins (occupied
    ``[0, departure_slot)``, power ``bv.power``) with the fewest whole
    modules that trajectory needs. ``build_measurement_instance`` already
    rejects instances where those module counts do not fit a pile
    (``assert_whole_module_feasible``), and the fewest modules is the choice
    that leaves every other vehicle the most room -- any compact solution's
    routing holds at least these many on the vehicle's lane.
    """
    assert bv.mode is BoundaryMode.FIXED
    if bv.departure_slot <= 0:
        raise ValueError(
            f"FIXED boundary vehicle {bv.vehicle_id} has departure_slot={bv.departure_slot}; "
            "build_measurement_instance discards such vehicles -- pass the instance it builds."
        )
    occupied = range(0, bv.departure_slot)
    power = {k: float(bv.power.get(k, 0.0)) for k in occupied if bv.power.get(k, 0.0) > 0.0}
    modules = {k: modules_needed(bv.power.get(k, 0.0), station.p_module) for k in occupied}
    return PBPlan(
        vehicle_id=bv.vehicle_id,
        pile=bv.pile,
        start=0,
        departure=bv.departure_slot,
        power=power,
        modules=modules,
    )
