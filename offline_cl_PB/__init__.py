"""
Exact branch-and-price for the connector-lane charging model (whole modules).

Solves ``offline_cl_opt``'s compact MILP -- the same instance objects, the
same boundary vehicles and objective cohorts -- to proven optimality by a
Dantzig-Wolfe decomposition per vehicle whose columns carry whole-module
counts, embedded in a branch-and-bound tree over structural plan
attributes. See ``README.md`` for the formulation and the exactness
argument.

Typical usage::

    from offline_cl_PB import solve_branch_and_price

    sol = solve_branch_and_price(
        inst.vehicles, station, delta=2.0, horizon_minutes=120.0,
        boundary_vehicles=inst.boundary_vehicles, cohorts=inst.cohorts,
        progress=True,
    )
    print(sol.status, sol.objective, sol.mean_sojourn)
    sol.per_vehicle
"""

from __future__ import annotations

from .bp import BranchAndPrice, BranchAndPriceError, solve_branch_and_price
from .columns import PBPlan, column_key, fixed_boundary_plan, null_pb_plan
from .heuristics import greedy_list_schedule, greedy_plan, schedule_from_compact, schedule_from_simulation
from .restrictions import NodeRestrictions, VehicleRestriction
from .solution import PBSolution
from .validation import (
    ScheduleValidationError,
    compact_model_check,
    schedule_to_cl_model,
    validate_plan,
    validate_schedule,
)

__all__ = [
    "BranchAndPrice",
    "BranchAndPriceError",
    "solve_branch_and_price",
    "PBSolution",
    "PBPlan",
    "column_key",
    "fixed_boundary_plan",
    "null_pb_plan",
    "greedy_list_schedule",
    "greedy_plan",
    "schedule_from_compact",
    "schedule_from_simulation",
    "schedule_to_cl_model",
    "NodeRestrictions",
    "VehicleRestriction",
    "ScheduleValidationError",
    "compact_model_check",
    "validate_plan",
    "validate_schedule",
]
