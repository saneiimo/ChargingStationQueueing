"""
The compact model's instance, flattened into the arrays the search uses.

Takes exactly the inputs of ``offline_cl_opt.model.build_cl_model`` and
``offline_cl_PB.BranchAndPrice`` (vehicles, station, delta, horizon,
boundary vehicles, cohorts, objective cohorts), so a sweep can hand the
same instance to every solver.

Who is searched over
--------------------
* Ordinary vehicles (``MEASUREMENT`` and ``QUEUED`` cohorts): release slot
  ``k_j = ceil(a_j/delta)``, energy from 0.
* ``OPTIMIZE`` boundary vehicles: already connected at boundary 0 on their own
  pile (the search starts with them plugged in), energy from
  ``initial_energy_kwh``. Their connector inside the pile is restored when
  connectors are assigned (``offline_cl_PB.validation.assign_connectors``).
* ``FIXED`` boundary vehicles are not searched over: their whole trajectory
  is pinned (``offline_cl_PB.fixed_boundary_plan``), so they only *reserve*
  a connector and ``ceil(p/Delta)`` modules on their pile in each slot they
  occupy (``free_connectors`` / ``free_modules``), and add the constant
  ``w_j * D_j`` to the objective (``fixed_departure_total``).

Units
-----
The search works in **slots**: its objective is ``z = sum_j w_j D_j`` (an
integer; ``w_j`` in {0, 1}). The compact model's objective, total sojourn in
minutes, is ``delta * z - arrival_total`` with
``arrival_total = delta * sum_j w_j k_j`` (``to_minutes``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from offline_cl_opt.boundary import COHORTS_ALL, BoundaryMode, BoundaryVehicle, Cohort
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import _release_slot
from offline_cl_PB import PBPlan, fixed_boundary_plan

from .dynamics import ceil_to_grid, module_energy, vehicle_params

# Energies must stay below this for the grid arithmetic to be exact (see dynamics.py).
MAX_ENERGY_KWH = 2.0**10


@dataclass
class SearchInstance:
    # --- the compact model's inputs, as given -------------------------------
    vehicles: list[VehicleData]
    station: StationSpec
    delta: float
    horizon_minutes: float
    boundary_vehicles: dict[int, BoundaryVehicle]
    cohorts: dict[int, Cohort]
    objective_cohorts: frozenset[Cohort]
    weights: dict[int, float]  # vehicle id -> w_j in {0, 1}
    K: int  # slots; every vehicle departs at a boundary <= K
    # --- searched vehicles (all but FIXED), index i = 0..J-1 ----------------
    ids: np.ndarray  # int64[J], vehicle id of search index i
    release: np.ndarray  # int64[J], k_j (0 for boundary vehicles)
    weight: np.ndarray  # int64[J], w_j
    params: np.ndarray  # float64[J, 6], dynamics.vehicle_params rows
    start_energy: np.ndarray  # float64[J], energy at release, rounded UP to the grid
    start_pile: np.ndarray  # int64[J], pile an OPTIMIZE vehicle is plugged into, else -1
    kappa: float  # grid energy of one module over one slot
    # --- FIXED boundary vehicles --------------------------------------------
    fixed_plans: dict[int, PBPlan]
    free_connectors: np.ndarray  # int64[M, K], C minus FIXED vehicles present
    free_modules: np.ndarray  # int64[M, K], N minus FIXED vehicles' modules
    symmetric_from: int  # first slot from which all piles are identical (no FIXED vehicle left)
    fixed_departure_total: int  # sum over FIXED vehicles of w_j * D_j
    # --- objective conversion -------------------------------------------------
    arrival_total: float  # delta * sum_j w_j k_j over every vehicle (minutes)

    @property
    def J(self) -> int:
        return len(self.ids)

    @property
    def n_optimized(self) -> int:
        return sum(1 for w in self.weights.values() if w > 0)

    def to_minutes(self, z: float) -> float:
        """Search objective (slots, ``sum_j w_j D_j``) -> total sojourn in minutes."""
        return self.delta * z - self.arrival_total

    def vehicle(self, i: int) -> VehicleData:
        return self._by_id[int(self.ids[i])]

    def __post_init__(self) -> None:
        self._by_id = {v.id: v for v in self.vehicles}


def build_search_instance(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
    cohorts: dict[int, Cohort] | None = None,
    objective_cohorts: frozenset[Cohort] | None = None,
) -> SearchInstance:
    """Validate the inputs (as ``BranchAndPrice`` does) and build the arrays."""
    if not vehicles:
        raise ValueError("need at least one vehicle")
    if delta <= 0 or horizon_minutes <= 0:
        raise ValueError("delta and horizon_minutes must be positive")
    ids = [v.id for v in vehicles]
    if len(set(ids)) != len(ids):
        raise ValueError("vehicle ids must be unique")
    boundary = dict(boundary_vehicles or {})
    by_id = {v.id: v for v in vehicles}
    unknown = set(boundary) - set(by_id)
    if unknown:
        raise ValueError(f"boundary_vehicles lists unknown vehicle ids {sorted(unknown)}")
    cohorts = dict(cohorts or {})
    objective_cohorts = objective_cohorts if objective_cohorts is not None else COHORTS_ALL
    weights = {
        v.id: 1.0 if cohorts.get(v.id, Cohort.MEASUREMENT) in objective_cohorts else 0.0 for v in vehicles
    }
    if not any(weights.values()):
        raise ValueError("objective_cohorts selects no vehicle; the objective would be empty")

    K = math.ceil(round(horizon_minutes / delta, 9))
    M, C, N, Delta = station.n_piles, station.n_connectors, station.n_modules, station.p_module
    h = delta / 60.0
    kappa = module_energy(h, Delta)
    releases = {v.id: _release_slot(v.a, delta) for v in vehicles}
    for j, k0 in releases.items():
        if k0 > K - 1:
            raise ValueError(f"Vehicle {j}: release slot {k0} is outside the horizon (K={K}).")
        if by_id[j].R >= MAX_ENERGY_KWH:
            raise ValueError(f"Vehicle {j}: R={by_id[j].R} kWh is too large for the exact energy grid")
    for j, bv in boundary.items():
        if releases[j] != 0:
            raise ValueError(f"boundary vehicle {j} must have a=0 (release slot 0)")
        if not (0 <= bv.pile < M and 0 <= bv.connector < C):
            raise ValueError(f"boundary vehicle {j} sits on a lane outside the station")

    # FIXED vehicles reserve their connector and modules.
    fixed_plans: dict[int, PBPlan] = {}
    free_conn = np.full((M, K), C, dtype=np.int64)
    free_mod = np.full((M, K), N, dtype=np.int64)
    fixed_departure_total = 0
    symmetric_from = 0
    for j, bv in boundary.items():
        if bv.mode is not BoundaryMode.FIXED:
            continue
        plan = fixed_boundary_plan(bv, station)
        if plan.departure > K:
            raise ValueError(f"FIXED vehicle {j} departs at slot {plan.departure}, after the horizon K={K}")
        fixed_plans[j] = plan
        for k in plan.occupied_slots():
            free_conn[plan.pile, k] -= 1
            free_mod[plan.pile, k] -= plan.q(k)
        fixed_departure_total += int(weights[j]) * plan.departure
        symmetric_from = max(symmetric_from, plan.departure)
    if (free_conn < 0).any() or (free_mod < 0).any():
        raise ValueError("FIXED boundary vehicles alone exceed a pile's connectors or modules")

    searched = [v for v in vehicles if v.id not in fixed_plans]
    J = len(searched)
    params = np.zeros((J, 6))
    start_energy = np.zeros(J)
    start_pile = np.full(J, -1, dtype=np.int64)
    for i, v in enumerate(searched):
        p_bar = min(v.p_max, N * Delta)
        phi = math.exp(-h / v.tau_hours)
        params[i] = vehicle_params(v.W, v.R, phi, p_bar, Delta, kappa)
        bv = boundary.get(v.id)
        if bv is not None:  # OPTIMIZE
            start_energy[i] = ceil_to_grid(bv.initial_energy_kwh)
            start_pile[i] = bv.pile

    return SearchInstance(
        vehicles=list(vehicles),
        station=station,
        delta=float(delta),
        horizon_minutes=float(horizon_minutes),
        boundary_vehicles=boundary,
        cohorts=cohorts,
        objective_cohorts=objective_cohorts,
        weights=weights,
        K=K,
        ids=np.array([v.id for v in searched], dtype=np.int64),
        release=np.array([releases[v.id] for v in searched], dtype=np.int64),
        weight=np.array([int(weights[v.id]) for v in searched], dtype=np.int64),
        params=params,
        start_energy=start_energy,
        start_pile=start_pile,
        kappa=kappa,
        fixed_plans=fixed_plans,
        free_connectors=free_conn,
        free_modules=free_mod,
        symmetric_from=symmetric_from,
        fixed_departure_total=fixed_departure_total,
        arrival_total=float(sum(weights[v.id] * delta * releases[v.id] for v in vehicles)),
    )
