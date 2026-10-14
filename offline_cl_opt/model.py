"""
Connector-lane offline MILP for a **closed system**, solved with gurobipy.

Every vehicle in the instance must be served and must receive its full
energy requirement within the horizon: the model has no "unserved" or
"unfinished" option. Constraints, in this module's own numbering (Groups
A-E):

  - (3)-(6): one uninterrupted stay per vehicle, starting exactly once
    (``sum_k eta_jk = 1`` -- every vehicle is connected).
  - (8)-(9): start slot ``S_j = sum_k k*eta_jk`` and departure boundary
    ``D_j = S_j + sum_k u_jk``.
  - (10)-(12): one lane per vehicle, no overlap on a lane (big-M
    ``K - k_j``, the smallest that keeps an inactive row slack).
  - (13)-(15): power only while connected, from whole modules on the
    vehicle's own lane, at most ``N`` modules per pile.
  - (17)-(19): energy recursion, battery taper with the discrete constant
    ``tau^delta_j``, departure rule (full energy at a departure inside the
    horizon).
  - **completion**: ``x_{j,K} >= W_j`` -- every vehicle holds its full
    request by the end of the horizon. With ``x <= W_j`` this is an
    equality, and since power is zero after departure it is the same as
    "full request at departure". The departure rule (19) is implied by it
    for integer solutions but kept because it tightens the LP relaxation.

The horizon must therefore be long enough for every vehicle to finish
(otherwise the model is infeasible), and long enough not to restrict the
optimum; choosing it is the caller's job (the objective sweep reads it off
the discretized FIFO run).

Optional pile/connector symmetry breaking (constraints (25)-(26)) is
implemented via ``break_symmetry``, off by default -- see its parameter
docstring below and README.md, "Symmetry breaking".

"Section N" references below are to the original design note
(``connector_lane_model.html``); constraint numbers are this module's own.

No per-vehicle right-edge windowing
-------------------------------------
An earlier revision of both the source document and this module narrowed
each vehicle's slot range on *both* ends -- left (arrival) and right (an
incumbent-derived ``kappa_j``) -- and needed an extra restored-obligation
constraint to stay correct once the right edge was in play. The current
document's Section 9.2 explicitly drops the right edge: the slack any
known incumbent leaves above the sum of individual best cases is shared
across *all* vehicles, so it only narrows anything when that slack is
smaller than the horizon itself -- which the document argues does not
happen in the congested regime this model targets ("Carrying a right edge
that is always the horizon adds notation and a boundary case without
removing a single variable"). Every vehicle's variables/rows here are
therefore generated over the *whole* ``[k_j, K)``, exactly the document's
``Kset_j`` -- see ``preprocess.py`` for what an incumbent is used for
instead (an objective cutoff, not windowing).

No ``theta`` variable
----------------------
Earlier revisions of both this document and this module introduced a
falling-edge indicator ``theta_jk`` (mirroring the rising-edge ``eta_jk``)
to state the departure rule. The current document's Section 5.6 proves this
was never necessary: because the delivered-energy variable ``x_jk`` is
already non-negative and upper-bounded by ``W_j``, the raw difference
``x_jk >= W_j*(u_{j,k-1} - u_jk)`` (19) cuts off exactly the same points, in
the linear relaxation as well as at integer points, as the version written
with an explicit indicator. Dropping ``theta`` removes ``O(JK)`` variables
and three constraint families for no loss of correctness -- see the
per-vehicle loop below and Section 5.6's own proof.

``x`` is a real variable, not a running expression
----------------------------------------------------
The delivered-energy state ``x_jk`` (energy delivered to vehicle ``j``
strictly before slot ``k``) is declared as an explicit Gurobi variable,
governed by the two-term recursion (17), ``x_{j,k+1} = x_jk + h*p_jk`` --
*not* accumulated as a growing Python-side linear expression substituted
directly into (18)/(19). Section 7.1 explains why this matters: writing the
cumulative sum out inline would give slot ``k``'s row ``O(k)`` nonzeros,
``O(K^2)`` per vehicle in total -- "enough to make presolve alone run for
minutes" on a realistic instance. Carrying ``x`` as a variable keeps every
row of (17)/(18)/(19) at a constant few nonzeros regardless of ``k``, since
the recursion is a chain of equalities rather than a substitution. The
upper bound ``x <= W_j`` is a per-variable bound (cheaper than a row), and
the completion row ``x_{j,K} >= W_j`` is one row per vehicle.

Adaptive module integrality (``adaptive.py``) and preprocessing
(``preprocess.py`` -- ``incumbent_objective``, used for an objective
cutoff, not windowing) are implemented on top of this model -- see their
module docstrings.

Objective: total sojourn on the slot grid, not sum_j D_j
---------------------------------------------------------
The source document's objective (1) is ``sum_j D_j`` (departure slot
boundaries). This module minimizes total sojourn time in minutes instead,
measured on the slot grid like every other decision of the model:

    sum_{j in objective} delta * (D_j - k_j),      k_j = ceil(a_j / delta),

i.e. from the vehicle's release slot (the first slot it may use) to its
departure slot. That is (1) scaled by ``delta > 0`` and shifted by the
constant ``delta * sum_j k_j`` (the objective's vehicle set is fixed: every
vehicle is served), so it has exactly the same optimal schedules and the
same ranking of every feasible schedule. What changes is the *value*:
``ObjVal``/``ObjBound`` are total sojourn in minutes, and Gurobi's relative
``MIPGap`` is measured against total sojourn. With arrivals already on the
grid (``a_j = delta * k_j``) this is the raw ``delta * D_j - a_j``; with
continuous arrivals it does not charge a vehicle for the part of a slot
before its release, which no schedule can use. Every objective value is a
whole multiple of ``delta`` (with 0/1 objective weights), so any proven
lower bound rounds up to the next multiple -- see
``sojourn_objective_round_up``.

Slot convention: horizon T split into K = ceil(T/delta) half-open slots,
slot k = [k*delta, (k+1)*delta), k = 0, ..., K-1. Release slot
k_j = ceil(a_j/delta) (Section 3.4); variables for k < k_j are never
created (Section 5.2's note), which also makes the convention "u_{j,-1}=0"
(Section 5.2) exact without a real k=-1 variable, and likewise
``x_{j,k_j} = 0`` (17) is never materialised as a variable -- see the
per-vehicle loop below.

Boundary conditions (vehicles already in the station at t=0)
--------------------------------------------------------------
``boundary_vehicles`` (see ``boundary.py``) lets ``vehicles`` include EVs
that were already queued or already plugged in when the modeled horizon
began (e.g. from a simulation's own warm-up boundary snapshot -- see
``boundary.py``'s own module docstring). Queued-but-not-yet-plugged
vehicles need no special handling at all (they're ordinary vehicles with
``a=0``, added via ``boundary.vehicles_from_boundary``). In-service
(already plugged in, mid-charge) vehicles are the ones this model
actually treats specially, and only in two small, surgical ways -- almost
everything about them (sequencing, module capacity, the objective) falls
out of the *existing* per-vehicle machinery unchanged, once their id
appears in ``vehicles``/``boundary_vehicles`` with ``a=0``:

  1. Right after the decision variables are created, every boundary
     vehicle's ``u``/``y``/``p`` bounds are pinned (``BoundaryMode.FIXED``:
     its whole known trajectory; ``BoundaryMode.OPTIMIZE``: only
     ``u_{j,0}=1`` and its lane ``y_{j,pile,connector}=1`` -- everything
     else about it stays a free decision).
  2. In the Groups D+E loop, a ``FIXED`` vehicle's (17)-(19) rows and its
     completion row are skipped entirely (nothing to decide or verify --
     its trajectory is exogenous ground truth that did finish in the
     simulation, and re-checking it against ``W_j`` would trip on the
     departure-slot floor-truncation, which drops its last partial slot of
     energy). An ``OPTIMIZE`` vehicle's energy recursion (17) starts from
     its own ``initial_energy_kwh`` instead of 0 -- ``x_jk`` keeps meaning
     "energy delivered since arrival" exactly as for any other vehicle,
     just seeded at a nonzero value reflecting the head start it already
     has; R_j/W_j (defined off the vehicle's real original ``s_i``) need
     no change at all, and the completion row asks only for the
     remainder.

Caution: ``break_symmetry`` forces piles/connectors to be used in
ascending-vehicle-id order (Section 10) -- a boundary vehicle's pinned
lane has no reason to already respect that ordering, so combining
``break_symmetry=True`` with ``boundary_vehicles`` risks infeasibility;
leave ``break_symmetry=False`` whenever boundary vehicles are present
(same caution ``offline_cl_dw.apply_mip_start`` already documents for the
same underlying reason).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB

from .boundary import COHORTS_ALL, BoundaryMode, BoundaryVehicle, Cohort
from .instance import StationSpec, VehicleData


def _release_slot(a: float, delta: float) -> int:
    """k_j = ceil(a_j / delta) -- Section 3.4."""
    return math.ceil(round(a / delta, 9))


def release_time(arrival: float, delta: float) -> float:
    """``delta * k_j``: the arrival rounded up to the slot grid, where the
    vehicle's first usable slot starts."""
    return delta * _release_slot(arrival, delta)


def sojourn_minutes(departure_slot: float, arrival: float, delta: float) -> float:
    """
    One vehicle's contribution to the objective: ``delta * (D_j - k_j)``
    with ``k_j = ceil(a_j / delta)``, minutes from its release slot to its
    departure slot boundary (see this module's docstring, "Objective").
    Takes the raw arrival ``a_j`` and discretizes it. Every package
    (compact model, DW, branch-and-price) prices a vehicle with this, so
    their objective values are directly comparable.
    """
    return delta * departure_slot - release_time(arrival, delta)


def sojourn_objective_round_up(value: float, delta: float, arrival_total: float, slack: float = 0.0) -> float:
    """
    The smallest attainable objective value ``>= value - slack``.

    With objective weights in {0, 1}, ``sum_j delta * (D_j - k_j)`` over a
    fixed vehicle set equals ``delta * n - arrival_total`` for an integer
    ``n = sum_j D_j`` and ``arrival_total = delta * sum_j k_j`` -- a whole
    multiple of ``delta`` -- so any proven lower bound rounds up to that
    lattice: ``delta * ceil((value - slack + arrival_total) / delta) -
    arrival_total``. Pass ``arrival_total=0`` for the plain multiples of
    ``delta`` (the same lattice). ``slack`` absorbs floating-point noise in
    ``value`` so a bound sitting exactly on a lattice point is not pushed one
    step too high. ``-inf``/``inf`` pass through unchanged.
    """
    if not math.isfinite(value):
        return value
    n = math.ceil(round((value - slack + arrival_total) / delta, 9))
    return delta * n - arrival_total


def lattice_round_up(value: float, delta: float) -> float:
    """
    The free tightening: a proven lower bound on total sojourn, rounded up
    to the next whole multiple of ``delta`` -- every attainable objective
    value is one (see this module's docstring, "Objective"), so the optimum
    is at least that. A ``1e-6`` relative slack keeps solver noise just
    above a multiple from pushing the bound a whole step too high.
    ``-inf``/``inf``/``nan`` pass through unchanged.
    """
    if not math.isfinite(value):
        return value
    return sojourn_objective_round_up(value, delta, 0.0, slack=1e-6 * max(1.0, abs(value)))


def lattice_abs_gap(tolerance: float | None, delta: float) -> float:
    """
    Gurobi ``MIPGapAbs`` that stops exactly when the incumbent is proven
    within ``tolerance`` minutes (total sojourn) of the optimum, using the
    free tightening.

    Objective values are multiples of ``delta``, so the tolerance only
    matters in whole steps: ``g = delta * floor(tolerance / delta)``. Once
    ``incumbent - bound < g + delta``, rounding the bound up to a multiple
    of ``delta`` leaves a gap of at most ``g``. So the returned value sits a
    hair under ``g + delta``; ``tolerance=None``/0 gives just under
    ``delta``, i.e. "stop as soon as the incumbent is proven optimal".
    """
    steps = math.floor(tolerance / delta + 1e-9) if tolerance else 0
    return delta * (steps + 1) * (1.0 - 1e-6)


def _earliest_departure_slot(
    v: VehicleData, p_bar: float, delta: float, k0: int, K: int
) -> int:
    """
    The earliest departure boundary vehicle ``v`` could possibly achieve,
    alone on its own connector with the whole module pool to itself,
    charging at its own acceptance limit throughout.

    Uses the model's own discrete update (flat cap ``p_bar``, then the
    taper cap via ``tau_delta``), not a continuous closed form -- only the
    discrete recursion is guaranteed to give a bound valid for the
    discretized model itself. Capped at ``K``: ``K`` means "cannot finish
    inside the horizon even alone" (the closed model is then infeasible).
    """
    h = delta / 60.0
    tau_d = v.tau_delta_hours(delta)
    x = 0.0  # cumulative energy delivered so far, kWh
    k = k0
    while k < K and x < v.W - 1e-9:
        step_power = min(p_bar, (v.R - x) / tau_d)
        x += h * step_power
        k += 1
    return min(K, k)


def earliest_departures(
    vehicles: list[VehicleData], station: StationSpec, delta: float, horizon_minutes: float
) -> dict[int, int]:
    """
    Per-vehicle earliest possible departure boundary ``E_j``, for every
    vehicle in ``vehicles``. A pure calculation -- no MILP solve needed.
    Not a constraint of the model; used for diagnostics (e.g. a cheap
    lower bound on total sojourn, ``sum_j (delta*E_j - a_j)``).
    """
    K = math.ceil(round(horizon_minutes / delta, 9))
    N, Delta = station.n_modules, station.p_module
    E: dict[int, int] = {}
    for v in vehicles:
        p_bar = min(v.p_max, N * Delta)
        k0 = _release_slot(v.a, delta)
        E[v.id] = _earliest_departure_slot(v, p_bar, delta, k0, K)
    return E


@dataclass
class ConnectorLaneModel:
    """Bundles the gurobipy model with the index metadata solution
    extraction needs."""

    model: gp.Model
    vehicles: dict[int, VehicleData]
    station: StationSpec
    delta: float
    K: int
    releases: dict[int, int]
    break_symmetry: bool
    tie_break: bool
    # vehicle_id -> BoundaryVehicle for every already-in-service vehicle at
    # t=0 (see boundary.py); {} if none. Kept for post-hoc inspection --
    # build_cl_model itself only reads this once, while pinning bounds.
    boundary_vehicles: dict[int, BoundaryVehicle]
    # vehicle_id -> Cohort (missing ids are MEASUREMENT), and the cohorts
    # the objective was actually summed over -- both echoed back so
    # extract_solution can report per-cohort sojourn without being told again.
    cohorts: dict[int, Cohort]
    objective_cohorts: frozenset[Cohort]
    module_pool_cap: int  # RHS actually used in (15); N unless overridden
    lanes: list[tuple[int, int]]  # (pile, connector) pairs
    u: gp.tupledict
    y: gp.tupledict
    r: gp.tupledict
    p: gp.tupledict
    b: gp.tupledict
    eta: gp.tupledict
    # x_jk: delivered energy strictly before slot k, kWh (17). A real Gurobi
    # variable (see the module docstring, "x is a real variable") -- keyed
    # (j, k) for k in [releases[j]+1, K], i.e. every vehicle's own release
    # shifted one slot late plus one terminal value at the horizon;
    # x_{j,k_j} = 0 (17) is never materialised, matching the u_{j,-1}=0
    # convention.
    x: gp.tupledict
    # S_j, D_j are dependent linear expressions in eta/u, not decision
    # variables -- introducing them as Gurobi Vars would only add branching
    # surface for no benefit, since they're already exactly pinned by
    # (8)-(9).
    S: dict[int, gp.LinExpr]
    D: dict[int, gp.LinExpr]


def build_cl_model(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    relax_modules: bool = False,
    module_pool_cap: int | None = None,
    break_symmetry: bool = False,
    tie_break: bool = False,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
    cohorts: dict[int, Cohort] | None = None,
    objective_cohorts: frozenset[Cohort] | None = None,
    model_name: str = "connector_lane_offline",
) -> ConnectorLaneModel:
    """
    Build (but do not solve) the closed-system connector-lane MILP:
    minimize total sojourn ``sum_j delta*(D_j - k_j)`` over
    ``objective_cohorts`` (minutes) subject to (3)-(15), (17)-(19) and the
    completion constraint -- every vehicle served and fully charged by the
    end of the horizon (see the module docstring).

    Parameters
    ----------
    vehicles :
        One ``VehicleData`` per vehicle in the instance.
    station :
        Pile / connector / module layout (homogeneous across piles and
        connectors): M, C, N, Delta.
    delta :
        Slot length in minutes.
    horizon_minutes :
        T, the horizon length; K = ceil(T / delta) slots. Every vehicle
        must finish by T, so a horizon too short for that makes the model
        infeasible, and a horizon shorter than the makespan of every optimal
        schedule restricts the optimum (the value can only decrease, never
        increase, as T grows). ``per_vehicle['departure_slot'] == K`` in a
        solution flags a vehicle finishing right at the horizon edge.
    relax_modules :
        If True, declare ``r`` as continuous on ``[0, module_pool_cap]``
        instead of integer -- gives ``CL_R``, the relaxation Section 8's
        adaptive-integrality procedure starts from (``adaptive.py``).
        Everything else in the model is unchanged (``u``, ``y``, ``b`` stay
        binary): ``CL_R`` is still a MIP, just without the module block's
        combinatorics. Off by default, matching the exact model.
    module_pool_cap :
        Overrides the right-hand side of (15) (a pile's module budget),
        which otherwise defaults to ``station.n_modules`` (``N``). Pass
        ``station.n_modules - station.n_connectors + 1`` for Section 8.4's
        conservative shortcut, which guarantees a solution roundable into a
        feasible schedule for the exact model without any further repair --
        see ``adaptive.conservative_feasible_solution``. Fewer modules mean
        slower charging, so this can make a tight horizon infeasible.
    break_symmetry :
        If True, add constraints (25)-(26) (current Section 10): all piles
        are identical and, within a pile, all connectors are identical, so
        any solution has up to ``M! * (C!)^M`` relabelled twins that the
        solver may waste effort re-proving are no better than each other.
        (25)/(26) force piles, and connectors within a pile, to be used in
        order of the lowest-id vehicle that occupies them -- see README.md,
        "Symmetry breaking" for the full argument and proof that this never
        excludes the true optimum. Off by default: the source
        document explicitly warns aggressive symmetry breaking "can
        interfere with warm starts" (Section 8.2's warm-start note), which
        matters more here since ``adaptive.py``'s whole strategy leans on
        warm-starting -- measure the effect on your own instance before
        turning this on for adaptive solves.
    tie_break :
        If True, add a secondary, lower-priority objective that prefers
        front-loaded power delivery, purely to break ties among the (often
        many) power profiles that tie the true optimum -- the primary
        objective only cares *when* each vehicle departs, never how its
        power is distributed within its own occupied window. Uses
        Gurobi's hierarchical multi-objective mode: total sojourn is
        optimized first (priority 1), then the tie-break (priority 0) with
        total sojourn held within ``abstol=1e-6`` of its optimum. Distinct
        departure patterns differ by multiples of ``delta``, far above that
        tolerance, so the tie-break cannot change the reported total
        sojourn (see the comment at the objective below).
        Roughly doubles solve effort (two hierarchical optimization
        phases); off by default. When True, read the primary objective via
        ``extract_solution`` (not ``cl_model.model.ObjVal`` directly -- see
        that function's own docstring for why).
    cohorts :
        ``{vehicle_id: Cohort}`` tagging each vehicle as having arrived in
        the measured window, been queued at the boundary, or been already
        plugged in at it (``boundary.Cohort``). Any id absent from the map
        is treated as ``MEASUREMENT``, so omitting this entirely keeps the
        old behaviour (every vehicle in the objective).
    objective_cohorts :
        Which cohorts the objective is summed over; ``None`` means all
        of them. Vehicles outside the selection still appear in the model
        in full -- they occupy connectors, draw modules and constrain
        everyone else -- their sojourn simply carries no objective weight.
        ``boundary.COHORTS_MEASUREMENT`` / ``COHORTS_MEASUREMENT_QUEUED`` /
        ``COHORTS_ALL`` are the three nested selections worth using.
        Excluding ``BOUNDARY`` is the usual choice when boundary vehicles
        are ``FIXED``: their ``D_j`` is pinned, so including it only adds a
        constant to the objective and dilutes the reported mean sojourn
        with a number the optimizer never chose. Note ``solution.
        extract_solution`` reports sojourn for all three cohort levels
        regardless of what was optimised here.
    boundary_vehicles :
        ``{vehicle_id: BoundaryVehicle}`` for any vehicle in ``vehicles``
        that was already plugged in (mid-charge) when the modeled horizon
        began -- see ``boundary.py`` and this module's own docstring,
        "Boundary conditions", for the full mechanics and the
        ``break_symmetry`` caution. ``None`` (default) or ``{}``: no
        boundary vehicles, i.e. the modeled horizon starts with an empty
        system (matching a simulation with no warm-up, or one where the
        warm-up-era queue/in-service vehicles are deliberately ignored).
    """
    if delta <= 0:
        raise ValueError(f"delta must be positive, got {delta}")
    if horizon_minutes <= 0:
        raise ValueError(f"horizon_minutes must be positive, got {horizon_minutes}")
    if not vehicles:
        raise ValueError("Need at least one vehicle")
    cap = module_pool_cap if module_pool_cap is not None else station.n_modules
    if not (0 < cap <= station.n_modules):
        raise ValueError(
            f"module_pool_cap must be in (0, station.n_modules={station.n_modules}], got {cap}"
        )

    h = delta / 60.0
    K = math.ceil(round(horizon_minutes / delta, 9))
    by_id = {v.id: v for v in vehicles}

    releases: dict[int, int] = {}
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

    N, Delta = station.n_modules, station.p_module
    lanes = [
        (mm, cc)
        for mm in range(station.n_piles)
        for cc in range(station.n_connectors)
    ]
    # P_bar_j = min(P_max_j, N*Delta): a vehicle can never be served by
    # more modules than one pile owns. Used as the tightened big-M in
    # (13)/(14) rather than P_max_j alone.
    p_bar = {v.id: min(v.p_max, N * Delta) for v in vehicles}

    m = gp.Model(model_name)
    m.Params.OutputFlag = 0  # solve_cl_model turns this on if requested

    # Every per-vehicle slot range below is [k_j, K) -- Section 9.2: only
    # the left (arrival) edge is used, never a per-vehicle right edge.
    jk_pairs = [(v.id, k) for v in vehicles for k in range(releases[v.id], K)]
    # x_jk (17) is generated for k in [k_j+1, K] -- one slot later than
    # u/p/eta (x_{j,k_j}=0 is the implicit, never-materialised start of the
    # recursion) plus one terminal value at the horizon, closing it.
    jk_x_pairs = [(v.id, k) for v in vehicles for k in range(releases[v.id] + 1, K + 1)]
    jmc_list = [(v.id, mm, cc) for v in vehicles for (mm, cc) in lanes]
    mck_list = [(mm, cc, k) for (mm, cc) in lanes for k in range(K)]
    sorted_ids = sorted(by_id)
    pair_list = list(combinations(sorted_ids, 2))  # (i, j), i < j by id

    # --- Decision variables (bounds/types set explicitly) ----------------------
    # u_jk: 1 if vehicle j occupies a connector during slot k (Sec. 3, decision variables).
    u = m.addVars(jk_pairs, vtype=GRB.BINARY, name="u")
    # y_jmc: 1 if vehicle j is assigned to lane (m,c) for its whole stay.
    y = m.addVars(jmc_list, vtype=GRB.BINARY, name="y")
    # r_mck: whole power modules routed to lane (m,c) during slot k.
    # Continuous when relax_modules=True -- see adaptive.py, Section 8.
    r = m.addVars(
        mck_list,
        lb=0,
        ub=N,
        vtype=GRB.CONTINUOUS if relax_modules else GRB.INTEGER,
        name="r",
    )
    # p_jk: power delivered to vehicle j during slot k, kW.
    p = m.addVars(
        jk_pairs,
        lb=0.0,
        ub={(v.id, k): p_bar[v.id] for v in vehicles for k in range(releases[v.id], K)},
        name="p",
    )
    # b_ij: 1 if i precedes j, for i<j by id -- meaningful only if they end
    # up sharing a lane. Correctness of (11)-(12) does not depend on
    # which vehicle is labelled i vs j for a given pair (either ordering is
    # representable via b in {0,1}); ascending id is used purely to generate
    # exactly one variable per unordered pair. break_symmetry's (25)-(26)
    # reuse this same ascending-id order for the same reason -- the source
    # document's proof only needs *some* fixed total order over vehicles,
    # not specifically the arrival order Section 3 otherwise uses it for.
    b = m.addVars(pair_list, vtype=GRB.BINARY, name="b")
    # eta_jk: continuous [0,1], pinned to the exact 0/1 plug-in (rising)
    # edge indicator by (3)-(5) regardless of declared type -- see
    # Proposition 2 in the source document. Leaving it continuous drops
    # J*K variables from branching for free: once u is integral, (3)-(5)
    # leave eta no fractional option.
    eta = m.addVars(jk_pairs, lb=0.0, ub=1.0, name="eta")
    # x_jk (17): delivered energy strictly before slot k, kWh. A real Gurobi
    # variable -- see the module docstring, "x is a real variable, not a
    # running expression" -- with its own per-vehicle upper bound W_j (a
    # vehicle never receives more than it asked for); the completion row
    # below makes it receive exactly that.
    x = m.addVars(
        jk_x_pairs,
        lb=0.0,
        ub={(v.id, k): v.W for v in vehicles for k in range(releases[v.id] + 1, K + 1)},
        name="x",
    )

    # ============================================================
    # Boundary conditions -- pin u/y/p for vehicles already plugged in at
    # t=0 (see the module docstring, "Boundary conditions"). Everything
    # else about them (sequencing (11)-(12), module capacity (14)-(15),
    # the objective) needs no special handling: it already falls out of
    # the ordinary constraints below once these bounds are fixed.
    # ============================================================
    boundary_vehicles = boundary_vehicles or {}
    for bid, bv in boundary_vehicles.items():
        j, k0 = bid, releases[bid]
        # Lane is never a decision for an in-service vehicle: pin its own
        # (pile, connector) to 1 and every other lane to 0 (redundant with
        # (10) once one lane is pinned, but cheap and removes any ambiguity).
        for (mm, cc) in lanes:
            pinned = 1.0 if (mm, cc) == (bv.pile, bv.connector) else 0.0
            y[j, mm, cc].lb = y[j, mm, cc].ub = pinned

        if bv.mode is BoundaryMode.FIXED:
            # No control: pin the entire already-known trajectory.
            for k in range(k0, K):
                occupied = k < bv.departure_slot
                u[j, k].lb = u[j, k].ub = 1.0 if occupied else 0.0
                p[j, k].lb = p[j, k].ub = bv.power.get(k, 0.0) if occupied else 0.0
        else:
            # Optimizer keeps control of power/departure; only "already
            # occupying its connector right now" is a fact, not a choice.
            u[j, k0].lb = u[j, k0].ub = 1.0

    # ============================================================
    # Group A -- occupancy timeline: (2) implicit, (3)-(9).
    # ============================================================

    # --- (3)-(6): one uninterrupted stay per vehicle ----------------------------
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        for k in range(k0, K):
            u_prev = u[j, k - 1] if k > k0 else 0.0  # convention u_{j,-1}=0 (Sec. 5.2)
            m.addConstr(eta[j, k] >= u[j, k] - u_prev, name=f"C3_eta_lb[{j},{k}]")
            m.addConstr(eta[j, k] <= u[j, k], name=f"C4_eta_le_u[{j},{k}]")
            m.addConstr(eta[j, k] <= 1 - u_prev, name=f"C5_eta_le_1mprev[{j},{k}]")
        # (6): exactly one plug-in edge -- every vehicle is served (closed
        # system: there is no "never connected" option).
        m.addConstr(
            gp.quicksum(eta[j, k] for k in range(k0, K)) == 1, name=f"C6_one_start[{j}]"
        )

    # --- (8)-(9): start slot and departure boundary -----------------------------
    S: dict[int, gp.LinExpr] = {}
    D: dict[int, gp.LinExpr] = {}
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        S_lin = gp.quicksum(k * eta[j, k] for k in range(k0, K))  # (8)
        occ_len = gp.quicksum(u[j, k] for k in range(k0, K))
        S[j] = S_lin
        D[j] = S_lin + occ_len  # (9): D_j = S_j + sum_k u_jk

    # ============================================================
    # Group B -- lane assignment and exclusive use: (10)-(12).
    # ============================================================

    # --- (10): every vehicle gets exactly one lane --------------------------------
    for v in vehicles:
        m.addConstr(y.sum(v.id, "*", "*") == 1, name=f"C10_one_lane[{v.id}]")

    # --- (25)-(26) [added, current Section 10]: pile/connector symmetry --------
    # All piles are identical, and within a pile all connectors are
    # identical, so any solution has many relabelled twins. Force piles,
    # and connectors within a pile, to be used in order of the lowest-id
    # vehicle occupying them -- see README.md, "Symmetry breaking" for the
    # relabelling argument that this never excludes the true optimum.
    # Placed here (right after (10), which it constrains) rather than after
    # Section 10's position in the document's own narrative, which comes
    # much later -- the constraint only involves y, so there's no reason to
    # defer it.
    if break_symmetry:
        # (25): vehicle j may use pile mm only if some vehicle with a
        # strictly smaller id already uses pile mm-1 (any connector on it).
        for mm in range(1, station.n_piles):
            earlier_on_prev_pile = gp.LinExpr(0.0)
            for j in sorted_ids:
                m.addConstr(
                    y.sum(j, mm, "*") <= earlier_on_prev_pile,
                    name=f"C25_pile_symmetry[{j},{mm}]",
                )
                earlier_on_prev_pile = earlier_on_prev_pile + y.sum(j, mm - 1, "*")
        # (26): within each pile, vehicle j may use connector cc only if
        # some vehicle with a strictly smaller id already uses connector
        # cc-1 on that same pile.
        for mm in range(station.n_piles):
            for cc in range(1, station.n_connectors):
                earlier_on_prev_connector = gp.LinExpr(0.0)
                for j in sorted_ids:
                    m.addConstr(
                        y[j, mm, cc] <= earlier_on_prev_connector,
                        name=f"C26_connector_symmetry[{j},{mm},{cc}]",
                    )
                    earlier_on_prev_connector = earlier_on_prev_connector + y[j, mm, cc - 1]

    # --- (11)-(12): no two vehicles overlap on one lane ---------------------------
    # Big-M K - k_j (resp. K - k_i): when the bracket is >= 1 the row reads
    # S_j >= D_i - (K - k_j) * (>=1), and D_i <= K, S_j >= k_j make that
    # slack -- so this is the smallest coefficient that still switches the
    # row off when the two vehicles do not share the lane in that order.
    for (i, j) in pair_list:
        Si, Di = S[i], D[i]
        Sj, Dj = S[j], D[j]
        bij = b[i, j]
        Mj, Mi = K - releases[j], K - releases[i]
        for (mm, cc) in lanes:
            yi, yj = y[i, mm, cc], y[j, mm, cc]
            m.addConstr(
                Sj >= Di - Mj * (3 - yi - yj - bij),
                name=f"C11_seq[{i},{j},{mm},{cc}]",
            )
            m.addConstr(
                Si >= Dj - Mi * (2 - yi - yj + bij),
                name=f"C12_seq[{i},{j},{mm},{cc}]",
            )

    # ============================================================
    # Group C -- power delivery and module routing: (13)-(15).
    # ============================================================

    # --- (13): power only while connected, capped at P_bar_j --------------------
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        for k in range(k0, K):
            m.addConstr(p[j, k] <= p_bar[j] * u[j, k], name=f"C13_power_cap[{j},{k}]")

    # --- (14): power capped by the modules on the vehicle's own lane ------------
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        pbj = p_bar[j]
        for k in range(k0, K):
            for (mm, cc) in lanes:
                m.addConstr(
                    p[j, k] <= Delta * r[mm, cc, k] + pbj * (1 - y[j, mm, cc]),
                    name=f"C14_power_from_modules[{j},{mm},{cc},{k}]",
                )

    # --- (15): a pile cannot route more modules than it owns --------------------
    # RHS is N unless module_pool_cap tightens it (Section 8.4's conservative
    # shortcut passes N - C + 1 here).
    for mm in range(station.n_piles):
        for k in range(K):
            m.addConstr(
                gp.quicksum(r[mm, cc, k] for cc in range(station.n_connectors)) <= cap,
                name=f"C15_module_pool[{mm},{k}]",
            )

    # ============================================================
    # Groups D+E -- energy accounting, battery acceptance, the departure
    # rule (17)-(19) and the completion row. One per-vehicle pass, matching the
    # document's own D-then-E order exactly, since every row here is
    # already O(1)-sparse (real x variables, not an inline running sum --
    # see the module docstring) so there's no efficiency reason to
    # interleave differently.
    # ============================================================
    for v in vehicles:
        j, k0 = v.id, releases[v.id]
        bv = boundary_vehicles.get(j)

        # FIXED-mode boundary vehicles have nothing to decide or verify:
        # their whole trajectory is already pinned above (one that did
        # finish in the simulation), so (17)-(19) and the completion row
        # would only risk a spurious infeasibility (the departure-slot
        # floor-truncation in boundary.py drops the energy of its last
        # partial slot -- see this module's own docstring, "Boundary
        # conditions"). Skip Groups D+E entirely for them.
        if bv is not None and bv.mode is BoundaryMode.FIXED:
            continue

        tau_d = v.tau_delta_hours(delta)
        # x_{j,k0} convention: 0 for an ordinary freshly-arriving vehicle,
        # but an OPTIMIZE-mode boundary vehicle already has some energy
        # (relative to its own s_i) before t=0 -- see boundary.py's own
        # initial_energy_kwh docstring. x keeps meaning "energy delivered
        # since arrival" either way; only this seed value changes.
        x0 = bv.initial_energy_kwh if bv is not None else 0.0
        for k in range(k0, K):
            x_prev = x[j, k] if k > k0 else x0  # convention x_{j,k_j}=0 (17), or x0 for a boundary vehicle
            # (17): energy recursion -- an equality chain, two variable
            # terms per row (three once k>k0, since x_prev is then itself a
            # variable), never a growing sum.
            m.addConstr(x[j, k + 1] == x_prev + h * p[j, k], name=f"C17_energy_recursion[{j},{k}]")
            # (18): taper cap -- the sloped branch of the acceptance curve,
            # in terms of the *effective discrete* time constant
            # tau^delta_j (see instance.py, tau_delta_hours).
            m.addConstr(tau_d * p[j, k] + x_prev <= v.R, name=f"C18_taper_cap[{j},{k}]")
            # (19): departure rule -- the raw occupancy difference directly,
            # no falling-edge indicator needed (see the module docstring,
            # "No theta variable"). Rows for k in (k0, K-1]: at k=k0,
            # u_{j,k0-1}=0 makes the row read x >= W_j*(0-u) <= 0, always
            # true. A departure exactly at the horizon (k=K) is covered by
            # the completion row below. For integer solutions (19) is
            # implied by completion (power is zero after departure, so
            # x_{j,D_j} = x_{j,K} = W_j); it is kept because it removes
            # fractional LP points whose occupancy drops before the request
            # is delivered.
            if k > k0:
                u_prev = u[j, k - 1]
                m.addConstr(
                    x[j, k] >= v.W * (u_prev - u[j, k]), name=f"C19_departure_rule[{j},{k}]"
                )

        # Completion: every vehicle holds its full request by the end of the
        # horizon (closed system -- nobody is left unserved or unfinished).
        # With x <= W_j this pins x_{j,K} = W_j.
        m.addConstr(x[j, K] >= v.W, name=f"C_completion[{j}]")

    # --- objective -- minimize total sojourn over the selected cohorts ---------
    # (the document's (1), sum_j D_j, times delta minus sum_j a_j -- see the
    # module docstring). Vehicles outside objective_cohorts stay fully
    # modelled (they still hold connectors and draw modules); only their
    # sojourn is dropped from the sum --
    # see the objective_cohorts parameter docstring.
    cohorts = cohorts or {}
    objective_cohorts = objective_cohorts if objective_cohorts is not None else COHORTS_ALL
    objective_ids = [
        v.id
        for v in vehicles
        if cohorts.get(v.id, Cohort.MEASUREMENT) in objective_cohorts
    ]
    if not objective_ids:
        raise ValueError(
            "objective_cohorts selects no vehicle at all -- the objective would be "
            f"empty. Selected {sorted(c.value for c in objective_cohorts)}, but the "
            f"{len(vehicles)} vehicles present cover "
            f"{sorted({cohorts.get(v.id, Cohort.MEASUREMENT).value for v in vehicles})}."
        )
    # Total sojourn on the slot grid, minutes: sum_j delta * (D_j - k_j).
    # The release sum is a constant (Gurobi keeps it in ObjCon), so
    # ObjVal/ObjBound/MIPGap all read in total-sojourn terms.
    primary_obj = gp.quicksum(delta * D[j] for j in objective_ids) - sum(
        release_time(by_id[j].a, delta) for j in objective_ids
    )

    if tie_break:
        # Optional secondary objective, purely to break ties among solutions
        # that already achieve the true total-sojourn optimum -- it only cares
        # *when* each vehicle departs, never how its power is distributed
        # within its own occupied window, so many power profiles can tie
        # exactly. Maximizing sum(x[j,k]) --
        # cumulative energy delivered strictly before each slot, already a
        # real Gurobi variable (see the module docstring, "x is a real
        # variable") -- rewards front-loading for a fixed departure
        # schedule. Negated because ModelSense is MINIMIZE (primary);
        # -sum(x) under minimize is equivalent to maximizing sum(x).
        #
        # Priority 1 > 0 makes this strictly hierarchical (lexicographic):
        # Gurobi first solves the priority-1 objective to its true optimum,
        # then re-optimizes the priority-0 objective *holding that value
        # fixed* (within abstol/reltol below). Total-sojourn values differ
        # by multiples of delta across distinct departure-slot patterns, far
        # above the 1e-6 tolerance, so the tie-break cannot change which
        # schedules count as optimal.
        tie_break_obj = gp.quicksum(x[j, k] for j, k in jk_x_pairs)
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

    return ConnectorLaneModel(
        model=m,
        vehicles=by_id,
        station=station,
        delta=delta,
        K=K,
        releases=releases,
        break_symmetry=break_symmetry,
        tie_break=tie_break,
        boundary_vehicles=boundary_vehicles,
        cohorts=cohorts,
        objective_cohorts=objective_cohorts,
        module_pool_cap=cap,
        lanes=lanes,
        u=u,
        y=y,
        r=r,
        p=p,
        b=b,
        eta=eta,
        x=x,
        S=S,
        D=D,
    )


def solve_cl_model(
    cl_model: ConnectorLaneModel,
    *,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
    presolve: int | None = None,
    pre_passes: int | None = None,
    cutoff: float | None = None,
    tolerance: float | None = None,
    verbose: bool = False,
) -> None:
    """
    Solve ``cl_model`` in place (sets gurobi params, calls ``optimize``).

    Raises ``RuntimeError`` if the model is provably infeasible/unbounded,
    the given ``cutoff`` turned out to be inconsistent (see below), or the
    solver produced no feasible incumbent at all. A time-limited run that
    still found a feasible (possibly suboptimal) solution returns normally;
    check ``ConnectorLaneSolution.mip_gap`` / ``status`` for that case.

    Parameters
    ----------
    presolve, pre_passes :
        Direct pass-throughs to Gurobi's own ``Presolve``
        (``-1``=automatic (default), ``0``=off, ``1``=conservative,
        ``2``=aggressive) and ``PrePasses`` (``-1``=unlimited (default), or
        a small integer to cap the number of presolve rounds) parameters.
        Worth trying on a large instance where presolve itself -- not the
        branch-and-bound search after it -- dominates the solve time: the
        big-M sequencing rows (11)-(12) make probing (a presolve
        subroutine) expensive on instances with many vehicles/binaries, and
        Gurobi's own log makes this visible as many consecutive
        ``Presolve removed ... (presolve time = Ns)`` lines reporting the
        *same* reduction for a long stretch before finding the next one --
        that stretch is presolve effort spent for no payoff. There is no
        guaranteed win here: a shorter presolve starts the real search
        sooner, but on a less-tightened model that may need more nodes to
        close the same gap -- measure total wall-clock time (not just the
        presolve line) before deciding this helped on your instance. Left
        at Gurobi's own defaults (``None``, meaning "don't touch the
        parameter") unless given explicitly.
    cutoff :
        Section 9.2/11's recommended technique: pass a known feasible
        schedule's objective value (``UB``, total sojourn in minutes --
        e.g. from ``preprocess.incumbent_objective``) as Gurobi's own
        ``Cutoff`` parameter, telling the solver not to bother proving
        anything worse than a bound it's already known can be matched.
        This prunes nodes the moment their own dual bound reaches ``UB``,
        without needing to actually find a matching incumbent first. Safe
        by construction whenever ``cutoff`` really is an achievable
        objective value (whether for this exact model or, since ``M(I)``
        is always a relaxation of it for any ``I``, for one of
        ``adaptive.py``'s intermediate models too -- the true optimum of a
        relaxation is never worse than the exact model's, hence never
        worse than a valid ``UB`` on it): Gurobi's own semantics accept any
        solution *at least as good as* the cutoff, so the model's true
        optimum is still found and reported normally whenever it equals or
        beats ``cutoff``. A ``cutoff`` that was *not* actually achievable
        (built from an inconsistent incumbent) instead makes Gurobi return
        status ``GRB.CUTOFF`` with no solution at all -- handled below as
        a distinct ``RuntimeError``, not confused with genuine
        infeasibility.
    tolerance :
        Stop as soon as the incumbent is proven within this many minutes of
        total sojourn of the optimum (for a mean-sojourn tolerance of ``x``
        minutes pass ``x * n_optimized``). ``None`` (default): solve to
        proven optimality. Either way Gurobi's ``MIPGapAbs`` is set from
        ``lattice_abs_gap``, which uses the free tightening (objective
        values are multiples of ``delta``) to stop as early as that
        guarantee allows -- without a tolerance, as soon as the incumbent is
        proven optimal. ``mip_gap`` (relative) still applies too; Gurobi
        stops at whichever gap is met first. Not applied with
        ``tie_break`` (two objectives; Gurobi's gaps are then per
        objective).
    """
    m = cl_model.model
    m.Params.OutputFlag = 1 if verbose else 0
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    if threads is not None:
        m.Params.Threads = threads
    if presolve is not None:
        m.Params.Presolve = presolve
    if pre_passes is not None:
        m.Params.PrePasses = pre_passes
    if cutoff is not None:
        m.Params.Cutoff = cutoff
    if not cl_model.tie_break:
        m.Params.MIPGapAbs = lattice_abs_gap(tolerance, cl_model.delta)
    elif tolerance is not None:
        raise ValueError("tolerance needs tie_break=False (one objective)")

    m.optimize()

    if m.Status == GRB.CUTOFF:
        raise RuntimeError(
            f"Connector-lane MILP found no solution at or better than the given "
            f"cutoff={cutoff}. A cutoff built from a genuinely achievable schedule "
            "can never do this -- the model can always at least match it -- so this "
            "means the incumbent used to compute it wasn't actually feasible for this "
            "instance. Recompute it, or drop cutoff and let the solver search normally."
        )
    if m.Status == GRB.INFEASIBLE:
        raise RuntimeError(
            "Connector-lane MILP is infeasible. A common cause is a vehicle's "
            "release slot falling outside the horizon -- try a larger "
            "horizon_minutes."
        )
    if m.Status in (GRB.INF_OR_UNBD, GRB.UNBOUNDED):
        raise RuntimeError(
            f"Connector-lane MILP status={m.Status} (infeasible-or-unbounded / unbounded)."
        )
    if m.SolCount == 0:
        raise RuntimeError(
            f"Connector-lane MILP produced no feasible solution (status={m.Status}). "
            "If this was a time limit, raise time_limit or relax mip_gap."
        )
