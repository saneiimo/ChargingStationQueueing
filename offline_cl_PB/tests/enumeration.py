"""
Brute-force reference for tiny instances: every whole-module column, and the
full (not restricted) master built from them.

A column's master coefficients and cost depend only on its discrete
signature ``(pile, S, D, q_S..q_{D-1})``; its power profile only has to
exist. So the full column set is finite: enumerate every signature and keep
those for which a small LP finds a feasible power profile. This is written
independently of ``offline_cl_PB.pricer`` (no shared code) so the two can
check each other.
"""

from __future__ import annotations

import itertools

import gurobipy as gp
from gurobipy import GRB

from offline_cl_opt.boundary import BoundaryMode, BoundaryVehicle
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import _release_slot

from offline_cl_PB.columns import PBPlan, fixed_boundary_plan, null_pb_plan


def _power_profile(v, x0, S, D, q, station, delta, K):
    """A feasible power profile for the signature, or None (Gurobi LP)."""
    h = delta / 60.0
    Delta = station.p_module
    p_bar = min(v.p_max, station.n_modules * Delta)
    tau_d = v.tau_delta_hours(delta)
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.FeasibilityTol = 1e-9
    slots = list(range(S, D))
    p = {k: m.addVar(lb=0.0, ub=min(p_bar, Delta * q[i])) for i, k in enumerate(slots)}
    x = x0
    for k in slots:
        m.addConstr(tau_d * p[k] + x <= v.R)
        x = x + h * p[k]
        m.addConstr(x <= v.W)
    if D < K:
        m.addConstr(x >= v.W)
    # Prefer front-loaded power; any feasible profile would do.
    m.setObjective(gp.quicksum(-(K - k) * p[k] for k in slots), GRB.MINIMIZE)
    m.optimize()
    try:
        if m.Status != GRB.OPTIMAL:
            return None
        return {k: p[k].X for k in slots if p[k].X > 1e-12}
    finally:
        m.dispose()


def enumerate_columns(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
) -> dict[int, list[PBPlan]]:
    boundary_vehicles = boundary_vehicles or {}
    N = station.n_modules
    out: dict[int, list[PBPlan]] = {}
    for v in vehicles:
        bv = boundary_vehicles.get(v.id)
        if bv is not None and bv.mode is BoundaryMode.FIXED:
            out[v.id] = [fixed_boundary_plan(bv, station)]
            continue
        plans: list[PBPlan] = []
        if bv is None:
            plans.append(null_pb_plan(v.id, K))
        k0 = _release_slot(v.a, delta)
        piles = [bv.pile] if bv is not None else range(station.n_piles)
        starts = [k0] if bv is not None else range(k0, K)
        x0 = bv.initial_energy_kwh if bv is not None else 0.0
        for m in piles:
            for S in starts:
                for D in range(S + 1, K + 1):
                    for q in itertools.product(range(N + 1), repeat=D - S):
                        power = _power_profile(v, x0, S, D, q, station, delta, K)
                        if power is None:
                            continue
                        plans.append(
                            PBPlan(
                                vehicle_id=v.id, pile=m, start=S, departure=D, power=power,
                                modules={S + i: q[i] for i in range(D - S)},
                            )
                        )
        out[v.id] = plans
    return out


def solve_full_master(
    columns: dict[int, list[PBPlan]],
    station: StationSpec,
    K: int,
    weights: dict[int, float],
    *,
    delta: float,
    arrivals: dict[int, float],
    integer: bool,
    allowed=None,
) -> float | None:
    """Optimum of the full master (LP or binary) over the given columns; None if infeasible.
    ``allowed(plan) -> bool`` filters columns (a node's restrictions). A column
    costs its vehicle's sojourn ``w_j * (delta*D - a_j)`` (minutes)."""
    m = gp.Model()
    m.Params.OutputFlag = 0
    m.Params.OptimalityTol = 1e-9
    m.Params.FeasibilityTol = 1e-9
    if integer:
        m.Params.MIPGap = 0.0
    conn: dict[tuple[int, int], gp.LinExpr] = {}
    mod: dict[tuple[int, int], gp.LinExpr] = {}
    for j, plans in columns.items():
        lams = []
        for plan in plans:
            if allowed is not None and not allowed(plan):
                continue
            lam = m.addVar(lb=0.0, ub=1.0, vtype=GRB.BINARY if integer else GRB.CONTINUOUS,
                           obj=weights[j] * (delta * plan.departure - arrivals[j]))
            lams.append(lam)
            if not plan.is_null:
                for k in plan.occupied_slots():
                    conn.setdefault((plan.pile, k), gp.LinExpr()).add(lam, 1.0)
                    if plan.q(k):
                        mod.setdefault((plan.pile, k), gp.LinExpr()).add(lam, float(plan.q(k)))
        m.addConstr(gp.quicksum(lams) == 1)
    for expr in conn.values():
        m.addConstr(expr <= station.n_connectors)
    for expr in mod.values():
        m.addConstr(expr <= station.n_modules)
    m.optimize()
    try:
        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            return None
        assert m.Status == GRB.OPTIMAL, m.Status
        return float(m.ObjVal)
    finally:
        m.dispose()
