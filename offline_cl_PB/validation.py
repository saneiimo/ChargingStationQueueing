"""
Independent checks of plans and whole schedules.

Nothing here trusts the master or the pricer. ``validate_plan`` re-derives a
plan's energy trajectory from its power profile and checks every
single-vehicle constraint; ``validate_schedule`` adds the station-level
rows (connectors, whole modules), assigns physical connectors and returns
the objective; ``compact_model_check`` goes one step further and evaluates
every row of ``offline_cl_opt.model.build_cl_model``'s own compact model at
the reconstructed ``(u, y, b, r, p, eta, x)`` -- the exact integer program
this package claims to solve.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass

from gurobipy import GRB

from offline_cl_opt.boundary import BoundaryMode, BoundaryVehicle, Cohort
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import _release_slot, build_cl_model, sojourn_minutes

from .columns import PBPlan, fixed_boundary_plan
from .tolerances import VALIDATION_TOL


class ScheduleValidationError(ValueError):
    pass


def validate_plan(
    plan: PBPlan,
    v: VehicleData,
    bv: BoundaryVehicle | None,
    station: StationSpec,
    delta: float,
    K: int,
    *,
    tol: float = VALIDATION_TOL,
) -> list[str]:
    """Every single-vehicle constraint of the compact model; returns the violations found."""
    errs: list[str] = []
    j = v.id
    if plan.vehicle_id != j:
        return [f"plan belongs to vehicle {plan.vehicle_id}, not {j}"]

    if bv is not None and bv.mode is BoundaryMode.FIXED:
        expected = fixed_boundary_plan(bv, station)
        if (plan.pile, plan.start, plan.departure) != (expected.pile, expected.start, expected.departure):
            errs.append(f"v{j}: FIXED plan must be pile/start/departure {expected.pile}/0/{expected.departure}")
        for k in set(plan.power) | set(expected.power):
            if abs(plan.power.get(k, 0.0) - expected.power.get(k, 0.0)) > tol:
                errs.append(f"v{j}: FIXED power differs at slot {k}")
        if plan.q_profile() != expected.q_profile():
            errs.append(f"v{j}: FIXED module profile differs from ceil(p/Delta)")
        return errs

    if plan.is_null:
        if bv is not None:
            errs.append(f"v{j}: an in-service boundary vehicle cannot be unserved")
        if plan.departure != K or plan.power or any(plan.modules.values()):
            errs.append(f"v{j}: null plan must have departure K={K} and no power/modules")
        return errs

    N, Delta = station.n_modules, station.p_module
    k0 = _release_slot(v.a, delta)
    S, D = plan.start, plan.departure
    if plan.pile is None or not (0 <= plan.pile < station.n_piles):
        errs.append(f"v{j}: pile {plan.pile} out of range")
    if S is None or not (k0 <= S < D <= K):
        return errs + [f"v{j}: need k0={k0} <= S={S} < D={D} <= K={K}"]
    if bv is not None:  # OPTIMIZE
        if S != k0:
            errs.append(f"v{j}: OPTIMIZE boundary vehicle must occupy slot {k0}")
        if plan.pile != bv.pile:
            errs.append(f"v{j}: OPTIMIZE boundary vehicle must stay on pile {bv.pile}")
    extra = [k for k in set(plan.power) | set(plan.modules) if not (S <= k < D)]
    if extra:
        errs.append(f"v{j}: power/modules outside [S, D): slots {sorted(extra)}")

    h = delta / 60.0
    p_bar = min(v.p_max, N * Delta)
    tau_d = v.tau_delta_hours(delta)
    x = bv.initial_energy_kwh if bv is not None else 0.0
    for k in range(S, D):
        p = plan.power.get(k, 0.0)
        q = plan.modules.get(k, 0)
        if int(q) != q or not (0 <= q <= N):
            errs.append(f"v{j} slot {k}: modules {q} not an integer in [0, {N}]")
        if p < -tol:
            errs.append(f"v{j} slot {k}: negative power {p}")
        if p > p_bar + tol:
            errs.append(f"v{j} slot {k}: power {p:.6f} > P_bar {p_bar:.6f}")
        if p > Delta * q + tol:
            errs.append(f"v{j} slot {k}: power {p:.6f} > {q} modules x {Delta}")
        if tau_d * p + x > v.R + tol:
            errs.append(f"v{j} slot {k}: taper (18) violated ({tau_d * p + x:.6f} > R={v.R:.6f})")
        x += h * p
        if x > v.W + tol:
            errs.append(f"v{j} slot {k}: delivered {x:.6f} kWh > W={v.W:.6f}")
    if D < K and x < v.W - tol:
        errs.append(f"v{j}: departs at {D} < K with {x:.6f} kWh < W={v.W:.6f} (19)")
    return errs


def assign_connectors(
    schedule: dict[int, PBPlan],
    station: StationSpec,
    boundary_vehicles: dict[int, BoundaryVehicle],
) -> dict[int, int]:
    """
    Left-edge interval colouring per pile, with boundary vehicles pinned to
    their physical connector. Pinned vehicles all start at slot 0 and are
    placed before anyone else starting at 0, so the greedy argument (a free
    connector exists whenever fewer than ``C`` intervals are active) is
    unchanged. Raises if the schedule needs more than ``C`` connectors.
    """
    C = station.n_connectors
    by_pile: dict[int, list[tuple[int, int, int, int]]] = {}
    for j, plan in schedule.items():
        if plan.is_null:
            continue
        pinned = 0 if j in boundary_vehicles else 1
        by_pile.setdefault(plan.pile, []).append((plan.start, pinned, j, plan.departure))  # type: ignore[arg-type]

    connectors: dict[int, int] = {}
    for pile, entries in by_pile.items():
        entries.sort()
        free = set(range(C))
        active: list[tuple[int, int]] = []  # (departure, connector)
        for start, pinned, j, departure in entries:
            while active and active[0][0] <= start:
                _, conn = heapq.heappop(active)
                free.add(conn)
            if pinned == 0:
                bv = boundary_vehicles[j]
                conn = bv.connector
                if bv.pile != pile or conn not in free:
                    raise ScheduleValidationError(
                        f"boundary vehicle {j} cannot keep its connector ({bv.pile}, {bv.connector})"
                    )
                free.remove(conn)
            else:
                if not free:
                    raise ScheduleValidationError(f"pile {pile}: more than {C} vehicles at slot {start}")
                conn = min(free)
                free.remove(conn)
            connectors[j] = conn
            heapq.heappush(active, (departure, conn))
    return connectors


@dataclass
class ScheduleCheck:
    objective: float  # total sojourn sum_j w_j (delta*D_j - a_j), minutes
    connectors: dict[int, int]  # vehicle -> connector (served vehicles only)


def validate_schedule(
    schedule: dict[int, PBPlan],
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
    boundary_vehicles: dict[int, BoundaryVehicle],
    weights: dict[int, float],
    *,
    tol: float = VALIDATION_TOL,
) -> ScheduleCheck:
    """Raise ``ScheduleValidationError`` unless ``schedule`` is feasible for the exact model."""
    errs: list[str] = []
    ids = {v.id for v in vehicles}
    if set(schedule) != ids:
        raise ScheduleValidationError(
            f"schedule covers {sorted(schedule)} but the instance has vehicles {sorted(ids)}"
        )
    for v in vehicles:
        errs += validate_plan(schedule[v.id], v, boundary_vehicles.get(v.id), station, delta, K, tol=tol)

    count: dict[tuple[int, int], int] = {}
    modules: dict[tuple[int, int], int] = {}
    for plan in schedule.values():
        if plan.is_null:
            continue
        for k in plan.occupied_slots():
            count[plan.pile, k] = count.get((plan.pile, k), 0) + 1  # type: ignore[index]
            modules[plan.pile, k] = modules.get((plan.pile, k), 0) + plan.q(k)  # type: ignore[index]
    for key, c in count.items():
        if c > station.n_connectors:
            errs.append(f"pile-slot {key}: {c} vehicles > C={station.n_connectors}")
    for key, q in modules.items():
        if q > station.n_modules:
            errs.append(f"pile-slot {key}: {q} modules > N={station.n_modules}")
    if errs:
        raise ScheduleValidationError("; ".join(errs[:20]) + (" ..." if len(errs) > 20 else ""))

    connectors = assign_connectors(schedule, station, boundary_vehicles)
    objective = float(
        sum(weights[v.id] * sojourn_minutes(schedule[v.id].departure, v.a, delta) for v in vehicles)
    )
    return ScheduleCheck(objective=objective, connectors=connectors)


# --------------------------------------------------------------------------- #
# Row-by-row check against the compact model
# --------------------------------------------------------------------------- #


@dataclass
class CompactCheck:
    max_violation: float
    worst: str  # name of the worst row / bound, "" if none violated
    objective: float  # the compact model's own objective at this point


def _compact_values(cl, schedule, connectors, vehicles, delta, boundary_vehicles) -> dict[int, float]:
    """``var.index -> value`` for every variable of the compact model ``cl`` at ``schedule``."""
    K = cl.K
    h = delta / 60.0
    values: dict[int, float] = {}  # var.index -> value

    lane: dict[int, tuple[int, int]] = {}
    S_of: dict[int, int] = {}
    D_of: dict[int, int] = {}
    for v in vehicles:
        j = v.id
        plan = schedule[j]
        k0 = cl.releases[j]
        bv = boundary_vehicles.get(j)
        S_of[j], D_of[j] = plan.start_slot(K), plan.departure
        if plan.is_null:
            lane[j] = (bv.pile, bv.connector) if bv is not None else (0, 0)
        else:
            lane[j] = (plan.pile, connectors[j])  # type: ignore[assignment]
        x = bv.initial_energy_kwh if bv is not None else 0.0
        for k in range(k0, K):
            occ = (not plan.is_null) and plan.start <= k < plan.departure  # type: ignore[operator]
            p = plan.power.get(k, 0.0) if occ else 0.0
            values[cl.u[j, k].index] = 1.0 if occ else 0.0
            values[cl.p[j, k].index] = p
            values[cl.eta[j, k].index] = 1.0 if (occ and k == plan.start) else 0.0
            x += h * p
            # FIXED vehicles have no energy rows; their x only has to sit in [0, W].
            values[cl.x[j, k + 1].index] = min(max(x, 0.0), v.W) if (bv and bv.mode is BoundaryMode.FIXED) else x
        for (mm, cc) in cl.lanes:
            values[cl.y[j, mm, cc].index] = 1.0 if (mm, cc) == lane[j] else 0.0

    occupant: dict[tuple[int, int, int], int] = {}
    for j, plan in schedule.items():
        if plan.is_null:
            continue
        for k in plan.occupied_slots():
            occupant[plan.pile, connectors[j], k] = plan.q(k)  # type: ignore[index]
    for (mm, cc, k), var in cl.r.items():
        values[var.index] = float(occupant.get((mm, cc, k), 0))

    for (i, j), var in cl.b.items():
        same = lane[i] == lane[j]
        values[var.index] = 1.0 if (same and D_of[i] <= S_of[j]) else 0.0
    missing = [var.VarName for var in cl.model.getVars() if var.index not in values]
    if missing:
        raise ScheduleValidationError(f"compact model: no value for {missing[:5]}")
    return values


def compact_model_check(
    schedule: dict[int, PBPlan],
    connectors: dict[int, int],
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
    cohorts: dict[int, Cohort] | None = None,
    objective_cohorts=None,
) -> CompactCheck:
    """
    Build the compact model (``bound_departures=True``, ``break_symmetry``
    and ``tie_break`` off), set every variable from ``schedule``, and
    evaluate every constraint, variable bound and integrality requirement.
    ``max_violation`` is the largest violation found (0.0 if none).
    """
    boundary_vehicles = boundary_vehicles or {}
    cl = build_cl_model(
        vehicles,
        station,
        delta,
        horizon_minutes,
        boundary_vehicles=boundary_vehicles,
        cohorts=cohorts,
        objective_cohorts=objective_cohorts,
        break_symmetry=False,
        bound_departures=True,
        tie_break=False,
        model_name="pb_compact_check",
    )
    try:
        m = cl.model
        values = _compact_values(cl, schedule, connectors, vehicles, delta, boundary_vehicles)
        all_vars = m.getVars()
        worst, worst_name = 0.0, ""
        for var in all_vars:
            val = values[var.index]
            viol = max(var.LB - val, val - var.UB, 0.0)
            if var.VType in (GRB.BINARY, GRB.INTEGER):
                viol = max(viol, abs(val - round(val)))
            if viol > worst:
                worst, worst_name = viol, var.VarName
        for con in m.getConstrs():
            row = m.getRow(con)
            lhs = sum(row.getCoeff(t) * values[row.getVar(t).index] for t in range(row.size()))
            rhs = con.RHS
            if con.Sense == GRB.LESS_EQUAL:
                viol = lhs - rhs
            elif con.Sense == GRB.GREATER_EQUAL:
                viol = rhs - lhs
            else:
                viol = abs(lhs - rhs)
            if viol > worst:
                worst, worst_name = viol, con.ConstrName

        obj_expr = m.getObjective()
        objective = obj_expr.getConstant() + sum(
            obj_expr.getCoeff(t) * values[obj_expr.getVar(t).index] for t in range(obj_expr.size())
        )
        return CompactCheck(max_violation=worst, worst=worst_name if worst > 0 else "", objective=float(objective))
    finally:
        cl.model.dispose()


def compact_check_passes(check: CompactCheck, expected_objective: float, *, tol: float = VALIDATION_TOL) -> bool:
    return check.max_violation <= tol and math.isclose(check.objective, expected_objective, abs_tol=1e-6)


def schedule_to_cl_model(
    schedule: dict[int, PBPlan],
    connectors: dict[int, int],
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
    cohorts: dict[int, Cohort] | None = None,
    objective_cohorts=None,
):
    """
    A *solved* ``offline_cl_opt`` ``ConnectorLaneModel`` whose solution is
    ``schedule`` -- a drop-in replacement for a compact model you solved
    yourself, so ``offline_cl_opt.solution.extract_solution`` and the
    ``offline_cl_opt`` plotting functions work on it unchanged.

    The compact model is built (``tie_break``/``break_symmetry`` off), the
    schedule's discrete decisions (``u``, ``y``, ``b``, ``r``) and powers
    ``p`` are fixed to their values, and Gurobi solves what is left (``eta``
    and ``x`` are then determined by the constraints). Raises if Gurobi
    does not find the fixed point feasible, i.e. the schedule is not a
    solution of the exact model.
    """
    boundary_vehicles = boundary_vehicles or {}
    cl = build_cl_model(
        vehicles,
        station,
        delta,
        horizon_minutes,
        boundary_vehicles=boundary_vehicles,
        cohorts=cohorts,
        objective_cohorts=objective_cohorts,
        break_symmetry=False,
        bound_departures=True,
        tie_break=False,
        model_name="pb_schedule",
    )
    values = _compact_values(cl, schedule, connectors, vehicles, delta, boundary_vehicles)
    for group in (cl.u, cl.y, cl.b, cl.r):
        for var in group.values():
            var.LB = var.UB = float(round(values[var.index]))
    for var in cl.p.values():
        val = values[var.index]
        var.LB = var.UB = val
    m = cl.model
    m.Params.OutputFlag = 0
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise ScheduleValidationError(
            f"the compact model rejects this schedule (Gurobi status {m.Status}); it is not a feasible solution"
        )
    return cl
