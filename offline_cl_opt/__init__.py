"""
Connector-lane offline MILP: Section 4 ("The optimisation model") of the
connector-lane formulation for offline optimal scheduling of an EV
charging station.

Explicitly assigns each vehicle to a physical lane (pile, connector) for
its whole stay and sequences vehicles that share a lane via a disjunctive
precedence variable, rather than the slot-indexed pile-occupancy scheme in
``offline_opt``. See ``offline_cl_opt/README.md`` for the full formulation,
scope, and unit conventions.

Typical usage::

    from offline_cl_opt import StationSpec, VehicleData, build_cl_model, solve_cl_model
    from offline_cl_opt.solution import extract_solution

    cl_model = build_cl_model(vehicles, station, delta=1.0, horizon_minutes=120.0)
    solve_cl_model(cl_model)
    solution = extract_solution(cl_model)
    print(solution.mean_sojourn, solution.per_vehicle)
"""

from __future__ import annotations

from .boundary import (
    COHORTS_ALL,
    COHORTS_MEASUREMENT,
    COHORTS_MEASUREMENT_QUEUED,
    BoundaryMode,
    BoundaryVehicle,
    Cohort,
    MeasurementInstance,
    assert_whole_module_feasible,
    boundary_vehicles_from_in_service,
    build_measurement_instance,
    cohort_totals,
    realized_slot_power,
    vehicles_from_boundary,
)
from .instance import StationSpec, VehicleData, vehicles_from_evs
from .model import ConnectorLaneModel, build_cl_model, solve_cl_model
from .solution import ConnectorLaneSolution, extract_solution
from .adaptive import (
    AdaptiveSolveResult,
    conservative_feasible_solution,
    rounded_module_routing,
    rounding_test_failures,
    solve_cl_model_adaptive,
)
from .preprocess import (
    earliest_departures,
    incumbent_departure_total,
)
from .visualization import (
    plot_pile_power_and_modules,
    plot_pile_power_and_modules_v2,
    plot_vehicle_power_and_modules,
)

__all__ = [
    "StationSpec",
    "VehicleData",
    "vehicles_from_evs",
    "BoundaryMode",
    "BoundaryVehicle",
    "Cohort",
    "COHORTS_MEASUREMENT",
    "COHORTS_MEASUREMENT_QUEUED",
    "COHORTS_ALL",
    "MeasurementInstance",
    "build_measurement_instance",
    "vehicles_from_boundary",
    "boundary_vehicles_from_in_service",
    "realized_slot_power",
    "assert_whole_module_feasible",
    "cohort_totals",
    "ConnectorLaneModel",
    "build_cl_model",
    "solve_cl_model",
    "ConnectorLaneSolution",
    "extract_solution",
    "AdaptiveSolveResult",
    "solve_cl_model_adaptive",
    "conservative_feasible_solution",
    "rounding_test_failures",
    "rounded_module_routing",
    "earliest_departures",
    "incumbent_departure_total",
    "plot_vehicle_power_and_modules",
    "plot_pile_power_and_modules",
    "plot_pile_power_and_modules_v2",
]
