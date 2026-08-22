"""
Offline (clairvoyant) lower-bound MILP for the charging-station queueing problem.

Given every vehicle's arrival time and charging requirement up front, finds
the pile assignment / module schedule that minimizes total sojourn time.
Because it optimizes with information no causal (deployable) policy could
have, its optimum lower-bounds the expected cost of every causal queue/power
policy (FIFO, heuristics, RL). See ``offline_opt/README.md`` for the full
formulation and ``offline_opt/example_toy.py`` for a worked example.

Typical usage::

    from offline_opt import compute_offline_bound, StationSpec

    solution = compute_offline_bound(evs, station, delta=1.0)
    print(solution.total_sojourn, solution.per_vehicle)
"""

from __future__ import annotations

from .instance import (
    StationSpec,
    VehicleData,
    full_power_time,
    taper_time_constant,
    vehicles_from_evs,
)
from .model import OfflineModel, build_offline_model, solve_offline_model
from .relaxed_model import OfflineRelaxedModel, build_relaxed_model
from .solution import OfflineSolution, extract_solution
from .bound import compute_ip_bounds, compute_offline_bound, default_horizon_minutes
from .visualization import plot_pile_power_and_modules, plot_vehicle_power_and_modules

__all__ = [
    "StationSpec",
    "VehicleData",
    "full_power_time",
    "taper_time_constant",
    "vehicles_from_evs",
    "OfflineModel",
    "build_offline_model",
    "solve_offline_model",
    "OfflineRelaxedModel",
    "build_relaxed_model",
    "OfflineSolution",
    "extract_solution",
    "compute_offline_bound",
    "compute_ip_bounds",
    "default_horizon_minutes",
    "plot_vehicle_power_and_modules",
    "plot_pile_power_and_modules",
]
