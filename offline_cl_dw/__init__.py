"""
Dantzig-Wolfe decomposition by vehicle for the connector-lane offline MILP
-- ``dantzig_wolfe_decomposition.html``.

Where ``offline_cl_opt`` solves the compact model (equations (1)-(19) of
``connector_lane_model.html``) directly, this package decomposes it by
vehicle: a master problem picks one whole-schedule "plan" per vehicle from
a pool generated on demand by per-(vehicle, pile) pricing subproblems
(column generation), then a small integer program over the generated
columns ("price-and-branch") gives a genuine feasible schedule. See
``README.md`` for why this is worth the extra machinery, when the compact
model's own relaxations (``offline_cl_opt.solve_cl_model_adaptive`` etc.)
aren't tight enough at scale.

Deliverable implemented here: column generation for a certified lower
bound, paired with price-and-branch for an upper bound (a "bracket") --
the source document's own recommended first step (Section 0), not the
full branch-and-price tree (Section 9.2), which is a documented but
unbuilt follow-on -- see README.md, "Scope".

Typical usage::

    from offline_cl_opt import StationSpec, vehicles_from_evs
    from offline_cl_dw import solve_by_decomposition

    solution, colgen = solve_by_decomposition(
        vehicles, station, delta=1.0, horizon_minutes=1440.0, progress=True,
    )
    print(solution.lower_bound, solution.upper_bound, solution.gap)
    print(solution.per_vehicle)
"""

from __future__ import annotations

from .colgen import ColGenResult, run_column_generation
from .columns import Plan, null_plan
from .master import (
    IntegerResult,
    LPResult,
    RestrictedMaster,
    add_column,
    build_master,
    purge_columns,
    solve_integer,
    solve_lp,
)
from .postprocess import assign_connectors, rounded_module_routing, validate_schedule, whole_module_failures
from .preprocess import (
    boundary_seed_plan,
    columns_from_evs,
    earliest_departures,
    fixed_plan,
    seed_columns,
)
from .pricer import VehiclePricer, build_pricer, price
from .solution import DWSolution, extract_solution, solve_by_decomposition
from .warm_start import apply_mip_start, refine_plan, refine_solution, refinement_ratio

__all__ = [
    "Plan",
    "null_plan",
    "VehiclePricer",
    "build_pricer",
    "price",
    "RestrictedMaster",
    "LPResult",
    "IntegerResult",
    "build_master",
    "add_column",
    "purge_columns",
    "solve_lp",
    "solve_integer",
    "earliest_departures",
    "seed_columns",
    "columns_from_evs",
    "boundary_seed_plan",
    "fixed_plan",
    "ColGenResult",
    "run_column_generation",
    "assign_connectors",
    "whole_module_failures",
    "rounded_module_routing",
    "validate_schedule",
    "DWSolution",
    "extract_solution",
    "solve_by_decomposition",
    "refinement_ratio",
    "refine_plan",
    "refine_solution",
    "apply_mip_start",
]
