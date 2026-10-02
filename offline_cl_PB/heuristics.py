"""
Feasible schedules and seed columns that need no optimisation.

* ``greedy_list_schedule`` -- always feasible, so it is the solver's
  default initial incumbent. Boundary vehicles go first (FIXED as pinned,
  OPTIMIZE from slot 0 on their own pile), then every other vehicle in
  arrival order takes the (pile, start) that lets it finish earliest using
  only the connectors and whole modules still free; a vehicle that cannot
  finish inside the horizon is left unserved (the null plan scores ``K``,
  the same as finishing at the horizon, and uses no capacity).
* ``schedule_from_simulation`` -- the same construction, but following a
  simulation's plug-in order and pile choices: a discretized simulation
  incumbent that is feasible by construction (see its docstring).
* ``greedy_plan`` -- one vehicle charging as fast as its BMS and a per-slot
  module cap allow; used for the above and for seed columns.
* ``schedule_from_compact`` -- read a solved ``offline_cl_opt`` compact
  model back as one column per vehicle (an incumbent, or a test fixture).

Every schedule built here still goes through ``validation`` before the
solver uses it; nothing here is trusted on its own.
"""

from __future__ import annotations

from typing import Callable

from offline_cl_opt.boundary import BoundaryMode, BoundaryVehicle
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import ConnectorLaneModel, _release_slot

from .columns import PBPlan, fixed_boundary_plan, modules_needed, null_pb_plan


def greedy_plan(
    v: VehicleData,
    pile: int,
    start: int,
    station: StationSpec,
    delta: float,
    K: int,
    module_cap: Callable[[int], int],
    *,
    initial_energy: float = 0.0,
    occupancy_ok: Callable[[int], bool] | None = None,
) -> PBPlan | None:
    """
    Plug ``v`` in at ``start`` on ``pile`` and charge at
    ``min(Delta*cap_k, P_bar, taper cap, W - x)`` each slot, holding
    ``ceil(p/Delta)`` modules, until ``W`` is delivered (departure) or the
    horizon ends (censored, ``D = K``). Returns ``None`` if some slot before
    that fails ``occupancy_ok`` (no free connector).
    """
    h = delta / 60.0
    Delta = station.p_module
    p_bar = min(v.p_max, station.n_modules * Delta)
    tau_d = v.tau_delta_hours(delta)
    x = initial_energy
    power: dict[int, float] = {}
    modules: dict[int, int] = {}
    k = start
    while True:
        if occupancy_ok is not None and not occupancy_ok(k):
            return None
        cap = max(0, int(module_cap(k)))
        p = max(0.0, min(Delta * cap, p_bar, (v.R - x) / tau_d, (v.W - x) / h))
        q = modules_needed(p, Delta)
        p = min(p, Delta * q)
        if p > 0.0:
            power[k] = p
        modules[k] = q
        x += h * p
        k += 1
        if k >= K or x >= v.W - 1e-9:
            break
    return PBPlan(vehicle_id=v.id, pile=pile, start=start, departure=k, power=power, modules=modules)


class _Ledger:
    """Connectors and whole modules already committed, per (pile, slot)."""

    def __init__(self, station: StationSpec, delta: float, K: int) -> None:
        self.station, self.delta, self.K = station, delta, K
        self.occ = {(m, k): 0 for m in range(station.n_piles) for k in range(K)}
        self.used = {(m, k): 0 for m in range(station.n_piles) for k in range(K)}

    def commit(self, plan: PBPlan) -> None:
        if plan.is_null:
            return
        for k in plan.occupied_slots():
            self.occ[plan.pile, k] += 1  # type: ignore[index]
            self.used[plan.pile, k] += plan.q(k)  # type: ignore[index]

    def place_boundary(self, v: VehicleData, bv: BoundaryVehicle) -> PBPlan:
        """OPTIMIZE boundary vehicle: slot 0 on its own pile, leftover modules."""
        N = self.station.n_modules
        plan = greedy_plan(
            v, bv.pile, _release_slot(v.a, self.delta), self.station, self.delta, self.K,
            lambda k, m=bv.pile: N - self.used[m, k],
            initial_energy=bv.initial_energy_kwh,
        )
        assert plan is not None  # no occupancy check: it physically holds its connector
        return plan

    def earliest_on_pile(self, v: VehicleData, m: int, first_start: int) -> PBPlan | None:
        """Earliest start >= ``first_start`` on pile ``m`` that finishes inside the horizon."""
        C, N = self.station.n_connectors, self.station.n_modules
        for start in range(max(first_start, _release_slot(v.a, self.delta)), self.K):
            if self.occ[m, start] >= C:
                continue
            plan = greedy_plan(
                v, m, start, self.station, self.delta, self.K,
                lambda k, m=m: N - self.used[m, k],
                occupancy_ok=lambda k, m=m: self.occ[m, k] < C,
            )
            if plan is None:
                continue  # a connector fills up before it finishes: start later
            # Greedy from an earlier start reaches every energy level no later,
            # so the first start that fits is the best on this pile.
            return plan if plan.departure < self.K else None
        return None

    def best_anywhere(self, v: VehicleData) -> PBPlan | None:
        best = None
        for m in range(self.station.n_piles):
            plan = self.earliest_on_pile(v, m, 0)
            if plan is not None and (best is None or plan.departure < best.departure):
                best = plan
        return best


def _boundary_first(
    vehicles: list[VehicleData], ledger: _Ledger, boundary_vehicles: dict[int, BoundaryVehicle]
) -> dict[int, PBPlan]:
    schedule: dict[int, PBPlan] = {}
    by_id = {v.id: v for v in vehicles}
    for j, bv in boundary_vehicles.items():
        if bv.mode is BoundaryMode.FIXED:
            schedule[j] = fixed_boundary_plan(bv, ledger.station)
            ledger.commit(schedule[j])
    for j, bv in sorted(boundary_vehicles.items()):
        if bv.mode is BoundaryMode.OPTIMIZE:
            schedule[j] = ledger.place_boundary(by_id[j], bv)
            ledger.commit(schedule[j])
    return schedule


def greedy_list_schedule(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
    boundary_vehicles: dict[int, BoundaryVehicle],
) -> dict[int, PBPlan]:
    """A feasible whole-module schedule by list scheduling (see module docstring)."""
    ledger = _Ledger(station, delta, K)
    schedule = _boundary_first(vehicles, ledger, boundary_vehicles)
    ordinary = sorted((v for v in vehicles if v.id not in boundary_vehicles), key=lambda v: (v.a, v.id))
    for v in ordinary:
        best = ledger.best_anywhere(v)
        schedule[v.id] = best if best is not None else null_pb_plan(v.id, K)
        ledger.commit(schedule[v.id])
    return schedule


def schedule_from_simulation(
    evs: list,
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
    *,
    warmup_period: float = 0.0,
) -> tuple[dict[int, PBPlan], dict[str, int]]:
    """
    A feasible whole-module schedule that follows a simulation's decisions
    (typically ``env.engine.metrics.arrived_evs``), for use as
    ``BranchAndPrice(initial_schedule=...)``.

    What is taken from the simulation: the **order** in which vehicles were
    plugged in and the **pile** each one used. Everything else is rebuilt on
    the slot grid so the result is feasible for the exact model by
    construction -- it is not a slot-by-slot copy of the simulated power:

    * vehicles are placed in simulated plug-in order (vehicles the
      simulation never plugged in go last, by arrival);
    * each one starts on its simulated pile at the first slot, from
      ``floor(service_start / delta)`` on, where a connector is free and it
      can still finish, charging as fast as its BMS and the *remaining* whole
      modules allow;
    * if that fails (the discretized pile is too busy to finish it inside
      the horizon), it gets the best start on any pile, and if even that
      fails it is left unserved -- which scores ``K``, exactly as a vehicle
      still charging at the horizon does, so it can only make the incumbent
      worse, never invalid.

    Boundary vehicles are handled exactly as in ``greedy_list_schedule``.
    The returned objective is therefore that of a genuine schedule (the
    solver re-validates it anyway), typically a little above the
    simulation's own grid-replayed objective because of slot rounding.

    Returns ``(schedule, counts)``; ``counts`` says how many vehicles kept
    their simulated pile, were moved, or were left unserved.
    """
    boundary_vehicles = boundary_vehicles or {}
    ledger = _Ledger(station, delta, K)
    schedule = _boundary_first(vehicles, ledger, boundary_vehicles)
    sim = {ev.id: ev for ev in evs}
    counts = {"followed_simulation": 0, "moved": 0, "unserved": 0, "not_started_in_simulation": 0}

    def sim_start(v: VehicleData):
        ev = sim.get(v.id)
        if ev is None or ev.service_start_time is None or ev.pile_tracker is None:
            return None
        return float(ev.service_start_time) - warmup_period, int(ev.pile_tracker.id)

    ordinary = [v for v in vehicles if v.id not in boundary_vehicles]
    started = sorted((v for v in ordinary if sim_start(v) is not None), key=lambda v: (sim_start(v)[0], v.id))
    rest = sorted((v for v in ordinary if sim_start(v) is None), key=lambda v: (v.a, v.id))
    for v in started + rest:
        info = sim_start(v)
        plan = None
        if info is not None:
            t_start, pile = info
            if 0 <= pile < station.n_piles:
                first = max(0, int(t_start // delta))
                plan = ledger.earliest_on_pile(v, pile, first)
            if plan is not None:
                counts["followed_simulation"] += 1
        else:
            counts["not_started_in_simulation"] += 1
        if plan is None:
            plan = ledger.best_anywhere(v)
            if plan is not None and info is not None:
                counts["moved"] += 1
        if plan is None:
            plan = null_pb_plan(v.id, K)
            counts["unserved"] += 1
        schedule[v.id] = plan
        ledger.commit(plan)
    return schedule, counts


def schedule_from_compact(cl_model: ConnectorLaneModel) -> tuple[dict[int, PBPlan], dict[int, int]]:
    """
    One column per vehicle from a *solved* compact model: occupancy from
    ``u``, lane from ``y``, power from ``p``, and ``q`` from the integer
    ``r`` of the vehicle's own lane. Returns ``(schedule, connectors)``.
    FIXED boundary vehicles get their fixed column. Values are read as the
    solver left them (tolerance ~1e-6); ``validation`` accepts that.
    """
    K = cl_model.K
    schedule: dict[int, PBPlan] = {}
    connectors: dict[int, int] = {}
    for j in cl_model.vehicles:
        bv = cl_model.boundary_vehicles.get(j)
        if bv is not None and bv.mode is BoundaryMode.FIXED:
            schedule[j] = fixed_boundary_plan(bv, cl_model.station)
            connectors[j] = bv.connector
            continue
        k0 = cl_model.releases[j]
        occupied = [k for k in range(k0, K) if cl_model.u[j, k].X > 0.5]
        if not occupied:
            schedule[j] = null_pb_plan(j, K)
            continue
        lane = next((mc for mc in cl_model.lanes if cl_model.y[j, mc[0], mc[1]].X > 0.5), None)
        if lane is None:
            raise ValueError(f"compact solution gives served vehicle {j} no lane")
        mm, cc = lane
        power = {k: max(0.0, cl_model.p[j, k].X) for k in occupied if cl_model.p[j, k].X > 0.0}
        modules = {k: int(round(cl_model.r[mm, cc, k].X)) for k in occupied}
        schedule[j] = PBPlan(
            vehicle_id=j,
            pile=mm,
            start=occupied[0],
            departure=occupied[-1] + 1,
            power=power,
            modules=modules,
        )
        connectors[j] = cc
    return schedule, connectors
