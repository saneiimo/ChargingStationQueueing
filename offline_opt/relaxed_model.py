"""
Continuous relaxation RP(pile_capacity_kw) of the offline MILP in
``model.py``: replaces the discrete power-module variable ``n[j,m,k]``
with a continuous ``q[j,m,k]`` -- power delivered
to vehicle j *by pile m* during slot k -- and the module-count constraints
(13)-(15) with an aggregate pile-capacity constraint (26)-(29). Everything
else (timeline, pile assignment, dispenser capacity, BMS curve) is identical
to ``build_offline_model``; constraint (20) (module efficiency) does not
apply here -- there is no discrete module count left to waste.

``p[j,k]`` is kept as an explicit variable, tied to ``q`` by an equality
constraint (28), rather than inlined as ``sum_m q[j,m,k]`` everywhere it's
used -- matching the write-up's own stated choice ("kept to make it more
readable"). This costs only J*K extra variables/equality constraints, cheap
next to the combinatorial branching (13)-(15) used to cost before this
relaxation removed it.

Used to sandwich the true integer-program optimum IP(N*Delta) between two
cheap, tractable bounds (see ``bound.compute_ip_bounds`` / README.md,
"Continuous relaxation bounds"), all in *total sojourn* (minimize):

    RP(N*Delta) <= IP(N*Delta) <= RP((N-C+1)*Delta)

RP(N*Delta) -- the relaxation at full pile capacity -- is a pure relaxation
of IP (drops integrality of the module count and of z), so its optimum can
only be at least as good: a valid *lower* bound on IP's total_sojourn.
RP((N-C+1)*Delta) -- the relaxation at reduced capacity -- is proven
roundable into a feasible IP solution that finishes no vehicle later, so its
total_sojourn is a valid *upper* bound on IP's.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from .instance import StationSpec, VehicleData
from .model import _release_slot


@dataclass
class OfflineRelaxedModel:
    """
    Mirrors ``OfflineModel``'s field names (model, vehicles, station, delta,
    K, releases, tie_break, y, alpha, sigma, z, p) so ``solution.extract_solution``
    works on it unchanged -- it only ever reads those fields. ``q`` replaces
    the discrete module variable ``n``; there is no ``n`` here.
    """

    model: gp.Model
    vehicles: dict[int, VehicleData]
    station: StationSpec
    delta: float
    K: int
    releases: dict[int, int]
    tie_break: bool
    pile_capacity_kw: float
    y: gp.tupledict
    alpha: gp.tupledict
    sigma: gp.tupledict
    z: gp.tupledict
    p: gp.tupledict
    q: gp.tupledict


def build_relaxed_model(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    tau: float,
    *,
    pile_capacity_kw: float | None = None,
    tie_break: bool = False,
    model_name: str = "offline_relaxed",
) -> OfflineRelaxedModel:
    """
    Build (but do not solve) the continuous relaxation RP(pile_capacity_kw):
    minimize total sojourn (3), constraints (4)-(7), (9)-(12), (16)-(19),
    (26)-(29). Constraint (8) is omitted, matching ``build_offline_model``.

    Parameters
    ----------
    vehicles, station, delta, horizon_minutes, tau, tie_break, model_name :
        See ``build_offline_model``.
    pile_capacity_kw :
        ``N*Delta`` in the write-up's notation: the aggregate power each
        pile can hand out per slot. Defaults to ``station.n_modules *
        station.p_module`` (the station's real capacity -- solving with this
        gives the RP(N*Delta) *lower* bound on total_sojourn). Pass
        ``(station.n_modules - station.n_dispensers + 1) * station.p_module`` for
        the RP([N-C+1]*Delta) *upper* bound -- ``bound.compute_ip_bounds``
        does both calls for you.
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive, got {horizon_minutes}")
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    if not vehicles:
        raise ValueError("Need at least one vehicle")

    cap = (
        pile_capacity_kw
        if pile_capacity_kw is not None
        else station.n_modules * station.p_module
    )
    if cap <= 0:
        raise ValueError(f"pile_capacity_kw must be positive, got {cap}")

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
    # z_jmk: continuous [0,1] -- exact linearization of y*(alpha-sigma),
    # pinned to {0,1} by (9)-(11) regardless of type. See README.md.
    z = m.addVars(jmk_pairs, lb=0.0, ub=1.0, name="z")
    # p_jk: power delivered to vehicle j during slot k (kW). Tied to q by
    # (28) below rather than inlined -- see module docstring.
    p = m.addVars(
        jk_pairs,
        lb=0.0,
        ub={(v.id, k): v.p_max for v in vehicles for k in range(releases[v.id], K)},
        name="p",
    )
    # q_jmk: power delivered to vehicle j *by pile m* during slot k (kW).
    # Replaces the discrete module-count variable n[j,m,k] of the integer model.
    q = m.addVars(jmk_pairs, lb=0.0, name="q")

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
            # (10): occupies exactly one dispenser while plugged in and
            # unfinished, zero dispensers otherwise.
            m.addConstr(
                gp.quicksum(z[j, mm, k] for mm in M_range) == alpha[j, k] - sigma[j, k],
                name=f"C10_occupancy[{j},{k}]",
            )
            for mm in M_range:
                # (11): that occupied dispenser must be on the vehicle's assigned pile.
                m.addConstr(z[j, mm, k] <= y[j, mm], name=f"C11_pile_link[{j},{mm},{k}]")
    for mm in M_range:
        for k in range(K):
            # (12): a pile has only C physical dispensers.
            m.addConstr(
                gp.quicksum(z[j, mm, k] for j in active_by_slot[k]) <= station.n_dispensers,
                name=f"C12_dispenser_cap[{mm},{k}]",
            )

    # --- (26)-(29): pile capacity, continuous -----------------------------------
    for mm in M_range:
        for k in range(K):
            # (26): a pile hands out at most `cap` kW combined this slot.
            m.addConstr(
                gp.quicksum(q[j, mm, k] for j in active_by_slot[k]) <= cap,
                name=f"C26_pile_capacity[{mm},{k}]",
            )
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        for k in range(k0, K):
            for mm in M_range:
                # (27): a vehicle draws power from a pile only while plugged into it.
                m.addConstr(q[j, mm, k] <= cap * z[j, mm, k], name=f"C27_pile_link[{j},{mm},{k}]")
            # (28): the power a vehicle receives is what's allocated to it,
            # summed over piles (at most one term is nonzero, per (9)-(11)).
            m.addConstr(
                p[j, k] == gp.quicksum(q[j, mm, k] for mm in M_range),
                name=f"C28_power_from_piles[{j},{k}]",
            )

    # --- (16)-(19): energy requirement and BMS acceptance curve -------------------
    # x_jk (auxiliary, eq. 2) = energy delivered to vehicle j before slot k.
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

    # --- (3): objective -- minimize total sojourn, same as build_offline_model --
    primary_obj = gp.LinExpr(0.0)
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        unfinished = gp.quicksum(1.0 - sigma[j, k] for k in range(k0, K))
        primary_obj += delta * (k0 + unfinished) - v.a

    if tie_break:
        # Same tie-break as build_offline_model -- see its comment there and
        # README.md, "Tie breaking". Negated under ModelSense=MINIMIZE so the
        # secondary still *maximizes* cumulative energy among sojourn-optimal
        # schedules. Even more freely tied here than in the integer model,
        # since q offers continuum-many ways to split a pile's capacity.
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

    return OfflineRelaxedModel(
        model=m,
        vehicles=by_id,
        station=station,
        delta=delta,
        K=K,
        releases=releases,
        tie_break=tie_break,
        pile_capacity_kw=cap,
        y=y,
        alpha=alpha,
        sigma=sigma,
        z=z,
        p=p,
        q=q,
    )
