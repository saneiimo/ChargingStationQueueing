"""
Offline lower-bound MILP, solved with gurobipy.

Slot convention: the horizon T is split into K = ceil(T/delta) half-open
slots, slot k = [k*delta, (k+1)*delta), k = 0, ..., K-1. Every decision is
made at a slot boundary.

Release slot k_j = ceil(a_j / delta) -- see ``_release_slot`` -- the earliest
slot a vehicle can occupy for its whole duration without predating its
arrival.

Only variables for k >= k_j (a vehicle's release slot) are created; every
earlier slot is fixed at alpha = sigma = 0 by construction rather than by an
explicit constraint, which keeps instances with staggered arrivals far
smaller than a dense k = 0..K-1 grid would be.

The primary objective minimizes total sojourn ``sum_j (c_j - a_j)`` directly
(eq. 1). Vehicles are *not* forced to finish by the horizon (no constraint
(8)); unfinished vehicles contribute a sojourn through to slot ``K``, and
delivered energy is capped by ``W_j`` rather than forced to equal it (17).

Optional tie-break (``tie_break=True``): the primary objective only cares
*when* each vehicle finishes (and who finishes), never how its power is
distributed within its own charging window, so many power profiles can tie
the true optimum exactly. A lower-priority secondary objective (see the
comment above ``setObjectiveN`` in ``build_offline_model``) prefers
front-loaded profiles among those ties, using Gurobi's native hierarchical
multi-objective mode -- which is what actually guarantees the primary
optimum (and hence total_sojourn) cannot change, not a hand-tuned weighting.
See README.md, "Tie breaking".
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from .instance import StationSpec, VehicleData


def _release_slot(a: float, delta: float) -> int:
    """
    k_j: first slot vehicle j may be plugged into.

    k_j = ceil(a_j / delta). Since slot k starts at k*delta, this is the
    smallest k whose slot start k*delta >= a_j -- i.e. the earliest slot the
    vehicle could occupy for its entire duration without being treated as
    present before it actually arrives. A vehicle arriving exactly on a slot
    boundary (a_j = k*delta) gets k_j = k with zero slack; one arriving
    mid-slot waits out the rest of that slot (< delta of unavoidable slack,
    the price of only allowing whole-slot occupancy decisions).
    """
    return math.ceil(round(a / delta, 9))


@dataclass
class OfflineModel:
    """Bundles the gurobipy model with the index metadata solution extraction
    needs (delta, per-vehicle release slots, variable dicts)."""

    model: gp.Model
    vehicles: dict[int, VehicleData]
    station: StationSpec
    delta: float
    K: int
    releases: dict[int, int]
    tie_break: bool
    y: gp.tupledict
    alpha: gp.tupledict
    sigma: gp.tupledict
    z: gp.tupledict
    n: gp.tupledict
    p: gp.tupledict


def build_offline_model(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    tau: float,
    *,
    tie_break: bool = False,
    model_name: str = "offline_lower_bound",
) -> OfflineModel:
    """
    Build (but do not solve) the offline MILP: minimize total sojourn (3)
    subject to constraints (4)-(7), (9)-(20) (module efficiency (20) is not
    in the original write-up -- see README.md, "Module efficiency").
    Constraint (8) (finish-by-horizon) is intentionally omitted.

    Parameters
    ----------
    vehicles :
        One ``VehicleData`` per vehicle in the instance.
    station :
        Pile / dispenser / module layout (homogeneous across piles): M, N, B, Delta.
    delta :
        Slot length in minutes.
    horizon_minutes :
        T, the horizon length; K = ceil(T / delta) slots. Vehicles that cannot
        finish by T remain unfinished (sigma stays 0) and still contribute to
        sojourn through the end of the horizon. See
        ``bound.default_horizon_minutes`` for a generous starting guess when
        you want everyone to be finishable.
    tau :
        Shared taper time constant (minutes); see ``instance.taper_time_constant``.
    tie_break :
        If True, add a secondary, lower-priority objective that prefers
        front-loaded power delivery, purely to break ties among the (often
        many) power profiles that tie the true optimum -- see the comment
        above the objective and ``offline_opt/README.md``, "Tie breaking",
        for why this cannot change the reported ``total_sojourn``. Roughly
        doubles solve effort (two hierarchical optimization phases); off by
        default.
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive, got {horizon_minutes}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not vehicles:
        raise ValueError("Need at least one vehicle")

    K = math.ceil(round(horizon_minutes / delta, 9))
    by_id = {v.id: v for v in vehicles}
    releases: dict[int, int] = {}
    active_by_slot: dict[int, list[int]] = defaultdict(list)
    for v in vehicles:
        k_j = _release_slot(v.a, delta)
        if k_j > K - 1:
            raise ValueError(
                f"Vehicle {v.id} arrives at t={v.a} min; its release slot "
                f"k_j={k_j} falls outside the horizon (K={K} slots of "
                f"{delta} min, valid slot indices 0..{K - 1}). "
                "Increase horizon_minutes."
            )
        releases[v.id] = k_j
        for k in range(k_j, K):
            active_by_slot[k].append(v.id)

    jk_pairs = [(v.id, k) for v in vehicles for k in range(releases[v.id], K)]
    jmk_pairs = [
        (v.id, mm, k)
        for v in vehicles
        for mm in range(station.n_piles)
        for k in range(releases[v.id], K)
    ]

    m = gp.Model(model_name)
    m.Params.OutputFlag = 0  # solve_offline_model turns this on if requested

    M_range = range(station.n_piles)

    # --- Decision variables (bounds/types set explicitly) ----------------------
    # y_jm: 1 if vehicle j is assigned to pile m.
    y = m.addVars([v.id for v in vehicles], M_range, vtype=GRB.BINARY, name="y")
    # alpha_jk: 1 if vehicle j has been plugged in by the start of slot k.
    alpha = m.addVars(jk_pairs, vtype=GRB.BINARY, name="alpha")
    # sigma_jk: 1 if vehicle j has completed by the start of slot k.
    sigma = m.addVars(jk_pairs, vtype=GRB.BINARY, name="sigma")
    # z_jmk: 1 if vehicle j occupies a dispenser on pile m during slot k.
    # Continuous on [0,1], not binary: fixing j and k, (9) picks exactly one
    # pile m* with y[j,m*]=1, (11) then forces z[j,m,k]=0 for every m != m*,
    # and (10) forces z[j,m*,k] = alpha[j,k]-sigma[j,k] in {0,1}. So z is an
    # exact linearization of y*(alpha-sigma), pinned to {0,1} by (9)-(11)
    # regardless of its declared type -- relaxing it away from BINARY drops
    # the single largest block of branching variables (J*M*K of them) without
    # changing the feasible region or the optimum. See README.md.
    z = m.addVars(jmk_pairs, lb=0.0, ub=1.0, name="z")
    # n_jmk: power modules (integer, 0..B) allotted to vehicle j on pile m in slot k.
    n = m.addVars(jmk_pairs, lb=0, ub=station.n_modules, vtype=GRB.INTEGER, name="n")
    # p_jk: power delivered to vehicle j during slot k (kW), bounded by its own peak.
    p = m.addVars(
        jk_pairs,
        lb=0.0,
        ub={(v.id, k): v.p_max for v in vehicles for k in range(releases[v.id], K)},
        name="p",
    )

    # --- (9): vehicle j is charged at exactly one pile -------------------------
    m.addConstrs((y.sum(j, "*") == 1 for j in by_id), name="C9_one_pile")

    # --- (4)-(7): timeline logic (no (8): finish-by-horizon is not enforced) ----
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        for k in range(k0, K - 1):
            # (4): once plugged in, stays plugged in.
            m.addConstr(alpha[j, k] <= alpha[j, k + 1], name=f"C4_alpha_mono[{j},{k}]")
            # (5): once finished, stays finished.
            m.addConstr(sigma[j, k] <= sigma[j, k + 1], name=f"C5_sigma_mono[{j},{k}]")
        for k in range(k0, K):
            # (6): a vehicle cannot finish before it has started.
            m.addConstr(sigma[j, k] <= alpha[j, k], name=f"C6_sigma_le_alpha[{j},{k}]")
        # (7) [alpha_jk = 0 for k < k_j] is enforced implicitly: those
        # variables are simply never created. (8) is omitted on purpose.

    # --- (10)-(12): dispenser occupancy and capacity --------------------------------
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        for k in range(k0, K):
            # (10): occupies exactly one dispenser while plugged in and unfinished,
            # zero dispensers otherwise.
            m.addConstr(
                gp.quicksum(z[j, mm, k] for mm in M_range) == alpha[j, k] - sigma[j, k],
                name=f"C10_occupancy[{j},{k}]",
            )
            for mm in M_range:
                # (11): that occupied dispenser must be on the vehicle's assigned pile.
                m.addConstr(z[j, mm, k] <= y[j, mm], name=f"C11_pile_link[{j},{mm},{k}]")
    for mm in M_range:
        for k in range(K):
            # (12): a pile has only N physical dispensers.
            m.addConstr(
                gp.quicksum(z[j, mm, k] for j in active_by_slot[k]) <= station.n_dispensers,
                name=f"C12_dispenser_cap[{mm},{k}]",
            )

    # --- (13)-(15), (20): modules -----------------------------------------------------
    Delta = station.p_module
    B = station.n_modules
    for mm in M_range:
        for k in range(K):
            # (13): a pile has only B modules to give out.
            m.addConstr(
                gp.quicksum(n[j, mm, k] for j in active_by_slot[k]) <= B,
                name=f"C13_module_pool[{mm},{k}]",
            )
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        for k in range(k0, K):
            for mm in M_range:
                # (14): modules can only be held on a pile the vehicle is plugged into.
                m.addConstr(n[j, mm, k] <= B * z[j, mm, k], name=f"C14_module_link[{j},{mm},{k}]")
            # (15): delivered power cannot exceed the power of the modules held.
            m.addConstr(
                p[j, k] <= Delta * gp.quicksum(n[j, mm, k] for mm in M_range),
                name=f"C15_power_from_modules[{j},{k}]",
            )
            # (20) [added, not in the original write-up]: at most one held
            # module may go unused. Prevents the solver from parking idle
            # modules on a vehicle that isn't drawing their power, while still
            # allowing exactly the slack module granularity requires: with 1
            # module held, any p_jk in [0, Delta] is fine (can't hold a
            # fractional module); holding a 2nd module requires actually using
            # more than the first one's worth. Does not change the optimal
            # total_sojourn -- see README.md, "Module efficiency".
            m.addConstr(
                p[j, k] >= Delta * (gp.quicksum(n[j, mm, k] for mm in M_range) - 1),
                name=f"C20_no_idle_modules[{j},{k}]",
            )

    # --- (16)-(19): energy requirement and BMS acceptance curve -------------------
    # x_jk (auxiliary, eq. 2) = energy delivered to vehicle j before slot k.
    # Built incrementally (O(K) per vehicle) rather than as a fresh sum each
    # slot (which would be O(K^2)): x[j,k] is the running total *before*
    # slot k's own delta*p[j,k] is added.
    x: dict[tuple[int, int], gp.LinExpr] = {}
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        running = gp.LinExpr(0.0)
        for k in range(k0, K):
            x[j, k] = running
            # (16): a vehicle may only be marked finished once it has received
            # its full energy requirement W_j.
            m.addConstr(x[j, k] >= v.W * sigma[j, k], name=f"C16_completion[{j},{k}]")
            # (18): flat acceptance cap; also forces p_jk = 0 whenever the
            # vehicle is not plugged-in-and-unfinished.
            m.addConstr(
                p[j, k] <= v.p_max * (alpha[j, k] - sigma[j, k]), name=f"C18_flat_cap[{j},{k}]"
            )
            # (19): taper cap, written without division: p*tau <= R0 - x.
            m.addConstr(p[j, k] * tau <= v.R0 - x[j, k], name=f"C19_taper_cap[{j},{k}]")
            running = running + delta * p[j, k]
        x[j, K] = running  # total energy delivered over the whole horizon
        # (17): delivered energy cannot exceed the requirement W_j (vehicles
        # may leave unfinished with a partial charge; C16 still requires a
        # full W_j before sigma may flip to 1).
        m.addConstr(x[j, K] <= v.W, name=f"C17_energy_cap[{j}]")

    # --- (3): objective -- minimize total sojourn sum_j (c_j - a_j) -------------
    # From eq. (1): c_j = delta * (k_j + sum_{k>=k_j} (1 - sigma[j,k])), with
    # the k < k_j unfinished slots contributing delta * k_j.
    primary_obj = gp.LinExpr(0.0)
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        unfinished = gp.quicksum(1.0 - sigma[j, k] for k in range(k0, K))
        primary_obj += delta * (k0 + unfinished) - v.a

    if tie_break:
        # Optional secondary objective, purely to break ties among solutions
        # that already achieve the true sojourn optimum. The primary only cares
        # *when* each vehicle finishes, never how its power is distributed
        # within its own charging window, so many power profiles can tie
        # exactly (see README.md, "Tie breaking"). Maximizing sum(x[j,k]) --
        # cumulative energy delivered before each slot, already built above
        # for C16/C19 -- rewards front-loading for a fixed completion schedule.
        # Negated because ModelSense is MINIMIZE (primary); -sum(x) under
        # minimize is equivalent to maximizing sum(x).
        #
        # Priority 1 > 0 makes this strictly hierarchical (lexicographic):
        # Gurobi first solves the priority-1 objective to its true optimum,
        # then re-optimizes the priority-0 objective *holding that value
        # fixed* (within abstol/reltol below). Sojourn values from (1) differ
        # by multiples of delta across distinct finish-slot patterns, far
        # above the 1e-6 tolerance for typical delta, so the tie-break cannot
        # change which schedules count as optimal.
        tie_break_obj = gp.quicksum(x[j, k] for j, k in jk_pairs)
        m.ModelSense = GRB.MINIMIZE
        m.setObjectiveN(
            primary_obj,
            index=0,
            priority=1,
            weight=1.0,
            abstol=1e-6,
            reltol=0.0,
            name="total_sojourn",
        )
        m.setObjectiveN(
            -tie_break_obj,
            index=1,
            priority=0,
            weight=1.0,
            name="front_load_tiebreak",
        )
    else:
        m.setObjective(primary_obj, GRB.MINIMIZE)

    m.update()

    return OfflineModel(
        model=m,
        vehicles=by_id,
        station=station,
        delta=delta,
        K=K,
        releases=releases,
        tie_break=tie_break,
        y=y,
        alpha=alpha,
        sigma=sigma,
        z=z,
        n=n,
        p=p,
    )


def solve_offline_model(
    offline_model: OfflineModel,
    *,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
    verbose: bool = False,
) -> None:
    """
    Solve ``offline_model`` in place (sets gurobi params, calls ``optimize``).

    Raises ``RuntimeError`` if the model is provably infeasible/unbounded or
    the solver produced no feasible incumbent at all. A time-limited run that
    still found a feasible (possibly suboptimal) solution returns normally;
    check ``OfflineSolution.mip_gap`` / ``status`` for that case.
    """
    m = offline_model.model
    m.Params.OutputFlag = 1 if verbose else 0
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if threads is not None:
        m.Params.Threads = threads

    m.optimize()

    if m.Status == GRB.INFEASIBLE:
        raise RuntimeError(
            "Offline MILP is infeasible. A common cause is a vehicle's release "
            "slot falling outside the horizon (arrival after T). Try a larger "
            "horizon_minutes. Note: a short horizon alone no longer forces "
            "infeasibility -- unfinished vehicles are allowed."
        )
    if m.Status in (GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        raise RuntimeError(
            f"Offline MILP status={m.Status} (infeasible-or-unbounded / unbounded)."
        )
    if m.SolCount == 0:
        raise RuntimeError(
            f"Offline MILP produced no feasible solution (status={m.Status}). "
            "If this was a time limit, raise time_limit or relax mip_gap."
        )
