"""
The pricing subproblem -- Section 6.2 of ``dantzig_wolfe_decomposition.html``.

For one vehicle and one (fixed) pile, this is the compact model's Groups A,
D, E, (13) and the completion row -- the vehicle's own physics -- restricted
to that vehicle alone, with no lane/module/sequencing variables at all (Proposition 1/2
move that coupling entirely into the master's capacity rows (25)-(26); the
pricer only ever sees the *price* of using them, via the objective (30)).

Exact-MILP pricer only (this project's own choice, per the source
document's Section 6.3 offering faster DP/enumeration alternatives as
optional speed-ups for later): every pricing round solves each
``(vehicle, pile)`` pair to proven optimality. This is simpler and always
correct; if profiling later shows pricing itself dominates colgen's wall
time on a large instance, Section 6.3's dynamic-programming pricer is the
documented next step (not built here).

One ``VehiclePricer`` (a live gurobipy ``Model``) is built once per
``(vehicle, pile)`` pair and reused for the life of the whole column
generation run -- only its *objective coefficients* change between rounds
(``price()`` calls ``setObjective`` fresh each time), never its constraint
matrix. This is the single largest speed factor the document's own
implementation checklist calls out (Section 11, item 4): besides skipping a
full rebuild, it lets Gurobi reuse the previous LP basis across rounds
(the same basis-warm-start principle ``offline_cl_opt.adaptive`` already
relies on for its own iterative loop).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from offline_cl_opt.instance import StationSpec, VehicleData

from .columns import Plan


def _release_slot(a: float, delta: float) -> int:
    """k_j = ceil(a_j / delta) -- same convention as offline_cl_opt.model."""
    return math.ceil(round(a / delta, 9))


@dataclass
class VehiclePricer:
    """A live, reusable pricing model for one (vehicle, pile) pair."""

    vehicle_id: int
    pile: int
    model: gp.Model
    k0: int
    K: int
    p_bar: float
    u: gp.tupledict
    eta: gp.tupledict
    p: gp.tupledict
    x: gp.tupledict
    S: gp.LinExpr
    D: gp.LinExpr
    delta: float
    arrival: float  # release time delta*k_j, minutes (see offline_cl_opt.model.sojourn_minutes)
    # False when this vehicle is outside the objective (see master.build_master's
    # objective_ids): its sojourn drops out of the pricing objective (30) too,
    # so the pricer minimises pure rent and the reduced costs stay consistent
    # with the master the duals came from.
    in_objective: bool = True


def build_pricer(
    v: VehicleData,
    pile: int,
    station: StationSpec,
    delta: float,
    K: int,
    *,
    initial_energy: float = 0.0,
    force_occupied_at_start: bool = False,
    in_objective: bool = True,
    model_name: str | None = None,
) -> VehiclePricer:
    """
    Build (but do not solve) the pricing subproblem (30) for vehicle ``v``
    on ``pile``, over ``k in [k_j, K)`` -- no upper truncation, matching
    Section 8.1's own conclusion that a right-edge window "adds notation
    and a boundary case without removing a single variable" in the regime
    this model targets.

    Closed system: the vehicle plugs in exactly once (``sum_k eta_k = 1``)
    and holds its full request by the end of the horizon (``x_K >= W_j``),
    as in ``offline_cl_opt.model``. The subproblem is therefore infeasible
    when the vehicle cannot finish on this pile within the horizon even
    alone; ``price`` then reports ``zeta = inf`` and no plan.

    ``initial_energy``/``force_occupied_at_start`` : for an OPTIMIZE-mode
    boundary vehicle (``offline_cl_opt.boundary.BoundaryVehicle`` -- see
    ``offline_cl_opt.model``'s own docstring, "Boundary conditions", which
    this mirrors exactly): ``initial_energy`` seeds the ``x`` recursion's
    base case (energy already delivered before t=0, relative to ``v.s_i``)
    instead of the usual 0, and ``force_occupied_at_start=True`` pins
    ``u[k0]=1`` (it is a fact, not a decision, that this vehicle is already
    plugged in). Both default to the ordinary, non-boundary behaviour.

    Mirrors ``offline_cl_opt.model.build_cl_model``'s per-vehicle block
    (Groups A, D, E, (13), completion) exactly, restricted to this one vehicle with no
    lane/module/sequencing constructs -- see this module's docstring for
    why those aren't needed here at all.
    """
    h = delta / 60.0
    k0 = _release_slot(v.a, delta)
    if k0 > K - 1:
        raise ValueError(
            f"Vehicle {v.id} arrives at t={v.a} min; its release slot k0={k0} "
            f"falls outside the horizon (K={K})."
        )
    p_bar = min(v.p_max, station.n_modules * station.p_module)
    tau_d = v.tau_delta_hours(delta)

    m = gp.Model(model_name or f"pricer_v{v.id}_m{pile}")
    m.Params.OutputFlag = 0

    ks = range(k0, K)
    u = m.addVars(ks, vtype=GRB.BINARY, name="u")
    eta = m.addVars(ks, lb=0.0, ub=1.0, name="eta")
    p = m.addVars(ks, lb=0.0, ub=p_bar, name="p")
    # x_k (17): real variable for k in [k0+1, K], same O(1)-sparse recursion
    # pattern as offline_cl_opt.model -- see that package's README, "Why x
    # is a real variable", for why this matters at scale.
    x = m.addVars(range(k0 + 1, K + 1), lb=0.0, ub=v.W, name="x")

    # Boundary condition: this vehicle is already plugged in at k0 -- not a
    # decision (see the docstring above; mirrors offline_cl_opt.model's own
    # u[j,k0] pin for the same reason).
    if force_occupied_at_start:
        u[k0].lb = u[k0].ub = 1.0

    # Group A: (3)-(6) eta pinning / exactly one start, (8)-(9) S_j/D_j.
    for k in ks:
        u_prev = u[k - 1] if k > k0 else 0.0
        m.addConstr(eta[k] >= u[k] - u_prev, name=f"eta_lb[{k}]")
        m.addConstr(eta[k] <= u[k], name=f"eta_le_u[{k}]")
        m.addConstr(eta[k] <= 1 - u_prev, name=f"eta_le_1mprev[{k}]")
    m.addConstr(gp.quicksum(eta[k] for k in ks) == 1, name="one_start")
    S_lin = gp.quicksum(k * eta[k] for k in ks)
    occ_len = gp.quicksum(u[k] for k in ks)
    D_lin = S_lin + occ_len

    # (13): power only while connected, capped at P_bar.
    for k in ks:
        m.addConstr(p[k] <= p_bar * u[k], name=f"power_cap[{k}]")

    # Groups D+E: (17) recursion, (18) taper, (19) departure rule -- same
    # per-slot pass as offline_cl_opt.model, real x variable throughout.
    # x_prev's base case (k=k0) is 0 for an ordinary vehicle, or
    # initial_energy for a boundary one -- see the docstring above; x keeps
    # meaning "energy delivered since arrival" either way.
    for k in ks:
        x_prev = x[k] if k > k0 else initial_energy
        m.addConstr(x[k + 1] == x_prev + h * p[k], name=f"energy_recursion[{k}]")
        m.addConstr(tau_d * p[k] + x_prev <= v.R, name=f"taper_cap[{k}]")
        if k > k0:
            u_prev = u[k - 1]
            m.addConstr(x[k] >= v.W * (u_prev - u[k]), name=f"departure_rule[{k}]")

    # Completion: the full request is delivered by the end of the horizon.
    m.addConstr(x[K] >= v.W, name="completion")

    m.update()

    return VehiclePricer(
        vehicle_id=v.id,
        pile=pile,
        model=m,
        k0=k0,
        K=K,
        p_bar=p_bar,
        u=u,
        eta=eta,
        p=p,
        x=x,
        S=S_lin,
        D=D_lin,
        delta=delta,
        arrival=delta * k0,
        in_objective=in_objective,
    )


def price(
    pricer: VehiclePricer,
    pi: dict[tuple[int, int], float],
    mu: dict[tuple[int, int], float],
    *,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
) -> tuple[float, Plan | None]:
    """
    (30): re-solve ``pricer`` with the objective built from this round's
    duals -- ``delta*(D_j - k_j) - sum_k pi[pile,k]*u[k] - sum_k mu[pile,k]*p[k]``
    (the vehicle's sojourn in minutes, the same cost ``master.add_column``
    gives its columns), using
    only this pricer's own fixed pile's entries of ``pi``/``mu`` (both
    keyed ``(pile, k)``, matching ``master.LPResult``) -- and return
    ``(zeta, plan)``, ``zeta`` being the pricer's own optimal value (used
    for the reduced cost (29) and the Lagrangian bound (32) by the caller)
    and ``plan`` the ``Plan`` read off the solution.

    Only the objective changes between calls; the constraint matrix is
    untouched, so Gurobi reuses the previous basis automatically (same
    principle as ``offline_cl_opt.adaptive``'s own iterative solves).
    Returns ``(inf, None)`` when the subproblem is infeasible -- the vehicle
    cannot finish on this pile within the horizon, which does not depend on
    the duals (the feasible set never changes). Raises if Gurobi stops
    without a solution for any other reason (e.g. ``time_limit``).

    ``threads``: caps this one solve's own internal Gurobi thread count.
    Left at ``None`` (Gurobi's own default) when pricers are solved one at
    a time; ``colgen.run_column_generation`` sets this to a small number
    (default ``1``) when it runs many pricers *concurrently* across
    Python threads (``max_workers > 1``), to avoid every pricer trying to
    claim every core at once and thrashing.
    """
    m = pricer.model
    k0, K, pile = pricer.k0, pricer.K, pricer.pile
    # pi/mu are keyed by (pile, k) -- see master.LPResult -- so every
    # lookup here must include this pricer's own fixed pile, not the bare
    # slot index (a previous version of this line looked up `pi.get(k,
    # 0.0)`, which always missed and silently priced everything at zero).
    # (30): sojourn - rent, with the sojourn dropped entirely for a vehicle
    # outside the objective so this matches the coefficient add_column gave
    # its columns.
    sojourn_term = (
        pricer.delta * pricer.D - pricer.arrival if pricer.in_objective else gp.LinExpr(0.0)
    )
    obj = sojourn_term - gp.quicksum(
        pi.get((pile, k), 0.0) * pricer.u[k] + mu.get((pile, k), 0.0) * pricer.p[k]
        for k in range(k0, K)
    )
    m.setObjective(obj, GRB.MINIMIZE)
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if threads is not None:
        m.Params.Threads = threads
    m.optimize()

    # Every variable is bounded, so INF_OR_UNBD can only mean infeasible.
    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        return math.inf, None
    if m.SolCount == 0:
        raise RuntimeError(
            f"Pricer for vehicle {pricer.vehicle_id}, pile {pricer.pile} found no solution "
            f"(status={m.Status}) -- a numerical or time_limit issue, not infeasibility."
        )

    zeta = float(m.ObjVal)
    occupied = [k for k in range(k0, K) if pricer.u[k].X > 0.5]
    start, departure = min(occupied), max(occupied) + 1
    power = {k: pricer.p[k].X for k in occupied if pricer.p[k].X > 1e-9}
    plan = Plan(
        vehicle_id=pricer.vehicle_id,
        pile=pricer.pile,
        start=start,
        departure=departure,
        power=power,
    )
    return zeta, plan
