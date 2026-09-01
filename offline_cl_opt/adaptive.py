"""
Adaptive module integrality -- Section 8 of ``connector_lane_model.html``.

The module-routing block ``r`` is the only intrinsically combinatorial part
of the model that (13)-(16)/(17)-(19) don't already force into place, and
in most pile-slots it resolves itself: either one vehicle occupies the
lane, in which case any whole-module routing is trivially realisable, or
the powers leave enough headroom that rounding each connector's
requirement up to whole modules still fits the pile's pool. Declaring every
``r_mck`` integer up front (``build_cl_model``'s default) makes the solver
branch on a decision that has, in most places, already resolved itself.

This module implements Section 8's alternative: solve with ``r`` relaxed to
continuous (``CL_R``), test whether the continuous solution is already
"secretly integral" pile-slot by pile-slot (8.1's Lemma / test (20)), and
promote only the pile-slots that fail to actually being integer, re-solving
until none do (8.2's 5-step procedure). Section 8.3 proves this converges,
in at most ``M*K`` iterations, to a solution that is exactly optimal for
the fully-integer exact model -- not merely a good heuristic answer.

Two solving aids from the source document are used as directly as it
describes them:

  - **MIP start** (8.2's note, 8.4): a genuinely feasible schedule is
    needed to seed the search, not a previous iteration's relaxed optimum
    (which is typically infeasible at the pile-slots just promoted).
    ``conservative_feasible_solution`` builds one cheaply, by solving
    ``CL_R`` with the pile budget tightened to ``N - C + 1`` (21) -- Section
    8.4 shows rounding up at that budget can never breach ``N``, so its
    power values are *always* roundable into a feasible schedule for the
    exact model, no repair needed. As the source document also notes
    (8.2's warm-start box), a MIP start's main value here is removing the
    feasibility phase and leaving a schedule in hand if the solve is cut
    short -- not pruning, since the incumbent it provides is generally not
    close enough to the true optimum to prune much.
  - **Basis warm start**: promoting a pile-slot only changes a few
    variables' declared type, never the constraint matrix, bounds, or
    objective -- so the LP relaxation at the root is the literal same LP
    before and after. Gurobi reuses the existing basis across successive
    ``optimize()`` calls on the same live ``Model`` object automatically
    (as long as it's never ``reset()``), so "implementing" this is mostly
    about *not breaking it*: ``solve_cl_model_adaptive`` builds one
    ``ConnectorLaneModel`` and mutates ``r[...].VType`` on it in place
    across iterations, rather than rebuilding from scratch each time. See
    ``tests/test_connector_lane_optimization.py`` for a direct timing
    comparison against a deliberately-cold control that verifies this
    actually helps rather than just asserting it does.

A third MIP-start source, alongside the conservative shortcut: a real
simulation's own output (``warm_start_evs``). ``_seed_values_from_evs``
reconstructs a discretized schedule directly from each vehicle's recorded
pile/connector, timing, and power trace -- no MILP solve needed to build
it, unlike the conservative shortcut. Because a simulation has no reason to
respect the canonical pile/connector labeling ``break_symmetry`` enforces
((25)-(26)), that reconstruction relabels the raw assignment into canonical
form first whenever ``break_symmetry=True`` -- see
``_relabel_lanes_for_symmetry``.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from gurobipy import GRB

if TYPE_CHECKING:
    from models.ev import EV

from .instance import StationSpec, VehicleData
from .model import ConnectorLaneModel, build_cl_model, solve_cl_model


def _pile_slot_power(cl_model: ConnectorLaneModel) -> dict[tuple[int, int, int], float]:
    """
    (pile, connector, slot) -> power drawn by whoever occupies that lane at
    that slot; idle lane-slots are simply absent from the returned dict.

    Reads the solved ``y`` (lane assignment) and ``u``/``p`` (occupancy and
    power) off ``cl_model`` and reshapes them from "per vehicle, per slot"
    into "per lane, per slot" -- the layout ``rounding_test_failures`` and
    ``rounded_module_routing`` both need, since the rounding test (20) is
    stated per pile-slot, not per vehicle.
    """
    power: dict[tuple[int, int, int], float] = {}
    for j in cl_model.vehicles:
        # Each vehicle has exactly one lane for its whole stay (10), so a
        # single pass to find it (rather than checking per-slot) is enough.
        k0 = cl_model.releases[j]
        lane = None
        for (mm, cc) in cl_model.lanes:
            if cl_model.y[j, mm, cc].X > 0.5:
                lane = (mm, cc)
                break
        if lane is None:
            continue  # vehicle never served (10)'s nominal assignment is meaningless
        mm, cc = lane
        # Only record slots where the vehicle is actually plugged in (u=1);
        # p is 0 (by (13)) everywhere else and would just clutter the dict.
        # Loop bound is the global K -- its p/u variables were never
        # created before its own release slot (Section 9.2).
        for k in range(k0, cl_model.K):
            if cl_model.u[j, k].X > 0.5:
                power[mm, cc, k] = cl_model.p[j, k].X
    return power


def rounding_test_failures(cl_model: ConnectorLaneModel) -> list[tuple[int, int]]:
    """
    Section 8.1, test (20): pile-slots ``(m,k)`` where the *ceiling* module
    requirement of the occupants -- ``sum_c ceil(p_{j(m,c),k}/Delta)`` --
    exceeds ``N``, the pile's *real* module count (always ``N`` here, never
    a tightened ``module_pool_cap``: this test asks whether the current
    power values are realisable by the exact model, not by whatever
    relaxation was solved to get them).

    Empty return means every pile-slot's continuous ``r`` is "secretly
    integral" -- Proposition 3's stopping condition.
    """
    N = cl_model.station.n_modules
    Delta = cl_model.station.p_module
    # Reshape once (per-lane) rather than re-scanning all vehicles for
    # every (pile, slot) pair below -- O(J) instead of O(J*M*K).
    power = _pile_slot_power(cl_model)

    failures: list[tuple[int, int]] = []
    for mm in range(cl_model.station.n_piles):
        for k in range(cl_model.K):
            # Sum the whole-module requirement of every connector on this
            # pile at this slot; idle connectors (absent from `power`)
            # contribute 0, matching the Lemma's "term is zero for an idle
            # connector."
            total = 0
            for cc in range(cl_model.station.n_connectors):
                p_val = power.get((mm, cc, k))
                if p_val is not None and p_val > 1e-9:
                    # Epsilon guards against floating-point noise pushing a
                    # power that's essentially a module multiple into the
                    # next ceil bucket (same class of bug as
                    # models/pile.py's is_overloaded -- see its comment).
                    total += math.ceil(p_val / Delta - 1e-9)
            if total > N:
                failures.append((mm, k))
    return failures


def rounded_module_routing(cl_model: ConnectorLaneModel) -> dict[tuple[int, int, int], int]:
    """
    Section 8.1's Lemma, made concrete: ``hat_r_{mck} = ceil(p/Delta)`` for
    the lane's occupant, ``0`` for an idle lane-slot. This is a genuine
    integer routing satisfying (14)-(15) exactly whenever
    ``rounding_test_failures`` is empty for the pile-slot in question --
    ``r`` itself need not have been declared integer in the model for this
    to be a correct answer; ``extract_solution`` never reads ``r`` at all
    (only ``p``, ``u``, ``y``), so this helper exists purely for inspecting
    or visualizing the actual module routing, not because anything else
    depends on it.
    """
    Delta = cl_model.station.p_module
    power = _pile_slot_power(cl_model)
    routing: dict[tuple[int, int, int], int] = {}
    for (mm, cc) in cl_model.lanes:
        for k in range(cl_model.K):
            p_val = power.get((mm, cc, k), 0.0)
            routing[mm, cc, k] = math.ceil(p_val / Delta - 1e-9) if p_val > 1e-9 else 0
    return routing


def conservative_feasible_solution(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    break_symmetry: bool = False,
    bound_departures: bool = True,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
    presolve: int | None = None,
    pre_passes: int | None = None,
    verbose: bool = False,
) -> ConnectorLaneModel:
    """
    Section 8.4: solve ``CL_R`` (module count relaxed to continuous) with
    the pile budget tightened to ``N - C + 1`` (21) in place of ``N``.

    Rounding up at most ``C`` positive numbers inflates their sum by less
    than ``C``, so ``sum_c ceil(r_mck) <= ceil(sum_c r_mck) + C - 1 <= N``:
    this solution's power values are *always* roundable into a feasible
    schedule for the exact model, with no repair loop needed -- one solve
    away from a guaranteed-feasible incumbent. The price is conservatism
    (``C-1`` modules reserved for rounding at every pile-slot regardless of
    actual occupancy), so its objective is a valid but generally loose
    *upper* bound, not the optimum.

    ``break_symmetry``/``bound_departures`` are passed straight through to
    ``build_cl_model`` -- when this is used to seed
    ``solve_cl_model_adaptive``'s MIP start, both must match the target
    model's own settings, or the seed's symmetry constraints (if broken
    only on one side) won't line up with the model being seeded;
    ``solve_cl_model_adaptive`` handles this automatically.
    ``presolve``/``pre_passes`` are passed straight through to
    ``solve_cl_model`` -- see its own docstring.

    Raises ``RuntimeError`` if ``rounding_test_failures`` is ever nonempty
    on the result -- a defensive check of the proof above, not something
    the proof allows to actually happen.
    """
    cap = station.n_modules - station.n_connectors + 1
    cl_model = build_cl_model(
        vehicles,
        station,
        delta,
        horizon_minutes,
        relax_modules=True,
        module_pool_cap=cap,
        break_symmetry=break_symmetry,
        bound_departures=bound_departures,
        model_name="connector_lane_conservative",
    )
    solve_cl_model(
        cl_model,
        mip_gap=mip_gap,
        time_limit=time_limit,
        threads=threads,
        presolve=presolve,
        pre_passes=pre_passes,
        verbose=verbose,
    )
    failures = rounding_test_failures(cl_model)
    if failures:
        raise RuntimeError(
            "Internal error: Section 8.4's conservative shortcut should always "
            f"be roundable, but {len(failures)} pile-slot(s) failed the "
            f"rounding test: {failures[:5]}"
        )
    return cl_model


def _apply_mip_start(target: ConnectorLaneModel, seed: ConnectorLaneModel) -> None:
    """
    Seed ``target``'s MIP start (``.Start``) from a solved, structurally
    identical ``seed`` model (same vehicles/station/delta/horizon/
    break_symmetry, so every variable's index set matches exactly --
    ``solve_cl_model_adaptive`` guarantees this by building both from the
    same arguments).

    Per Section 8.2's own warning, this must come from a genuinely feasible
    schedule -- here, ``conservative_feasible_solution`` -- never from a
    previous adaptive iteration's relaxed optimum, which is typically
    infeasible for the pile-slots just promoted to integer.

    Every decision variable is seeded explicitly (rather than leaving
    ``eta``/``x``/``b`` for Gurobi's own partial-start completion
    heuristic to infer): ``eta``/``x`` are read straight off the seed's
    already-pinned values, and ``b`` off whatever precedence the seed's own
    solve chose for any pair sharing a lane -- both trivially consistent
    with the seed's own ``u``/``y`` by construction, so there's no
    reconstruction logic needed here beyond copying.
    """
    for key, var in seed.u.items():
        target.u[key].Start = round(var.X)
    for key, var in seed.y.items():
        target.y[key].Start = round(var.X)
    for key, var in seed.eta.items():
        target.eta[key].Start = var.X
    for key, var in seed.x.items():
        if key in target.x:
            target.x[key].Start = var.X
    for key, var in seed.b.items():
        target.b[key].Start = round(var.X)
    # r may be continuous or (partially) integer on target at this point;
    # .Start on a continuous var is a harmless hint either way.
    for key, r_val in rounded_module_routing(seed).items():
        if key in target.r:
            target.r[key].Start = r_val
    target.model.update()


def _relabel_lanes_for_symmetry(
    raw_lane_by_vehicle: dict[int, tuple[int, int]],
    sorted_ids: list[int],
) -> dict[int, tuple[int, int]]:
    """
    Relabel a raw ``{vehicle_id: (pile, connector)}`` assignment into the
    canonical order (25)-(26) enforce -- piles, and connectors within a
    pile, labelled in order of the lowest-id vehicle that occupies them --
    exactly the constructive relabelling the symmetry-breaking correctness
    proof uses (README.md, "Symmetry breaking"), applied here so an
    externally-sourced assignment (e.g. a real simulation's, which has no
    reason to already respect that order) can be used as a MIP start
    without Gurobi discarding it as infeasible for (25)-(26).

    Piles are relabelled in the order their first (smallest-id) occupant is
    encountered while scanning vehicles by ascending id; connectors are
    relabelled the same way, independently within each (already-relabelled)
    pile. Vehicles absent from ``raw_lane_by_vehicle`` (unserved) are
    simply absent from the result too.
    """
    pile_canon: dict[int, int] = {}
    connector_canon: dict[int, dict[int, int]] = {}  # canonical pile -> {raw connector: canonical connector}
    relabeled: dict[int, tuple[int, int]] = {}
    for j in sorted_ids:
        if j not in raw_lane_by_vehicle:
            continue
        raw_pile, raw_connector = raw_lane_by_vehicle[j]
        if raw_pile not in pile_canon:
            pile_canon[raw_pile] = len(pile_canon)
        cpile = pile_canon[raw_pile]
        conn_map = connector_canon.setdefault(cpile, {})
        if raw_connector not in conn_map:
            conn_map[raw_connector] = len(conn_map)
        relabeled[j] = (cpile, conn_map[raw_connector])
    return relabeled


def _seed_values_from_evs(cl_model: ConnectorLaneModel, evs: list["EV"], *, break_symmetry: bool) -> None:
    """
    Set ``.Start`` values on ``cl_model``'s variables directly from a list
    of simulator ``EV`` objects (e.g. ``env.engine.metrics.finished_evs``)
    -- an alternative MIP-start source to ``conservative_feasible_solution``
    that needs no MILP solve to build, reconstructing a discretized
    schedule from each vehicle's own recorded pile/connector, timing, and
    power trace instead.

    Only EVs with a real pile/connector assignment and a finite
    ``departure_time`` contribute a seed; anything else (never served,
    dropped, still queued when the simulation ended) is simply left unset
    for Gurobi's own partial-start completion.

    If ``break_symmetry``, the raw (pile, connector) assignment read off
    the EVs is relabelled into canonical (25)-(26) order first (see
    ``_relabel_lanes_for_symmetry``) -- a real simulation has no reason to
    already respect that order, and without this step the seed would
    likely violate (25)-(26) and simply be discarded by Gurobi rather than
    actually helping.
    """
    delta = cl_model.delta

    # Step 1: each served EV's raw (pile, connector), straight off the
    # simulator's own tracking fields.
    raw_lane_by_vehicle: dict[int, tuple[int, int]] = {}
    evs_by_id: dict[int, "EV"] = {}
    for ev in evs:
        if ev.id not in cl_model.vehicles:
            continue
        if ev.pile_tracker is None or ev.connector_id_tracker is None:
            continue
        if ev.service_start_time is None or not math.isfinite(ev.departure_time):
            continue
        raw_lane_by_vehicle[ev.id] = (ev.pile_tracker.id, ev.connector_id_tracker)
        evs_by_id[ev.id] = ev

    sorted_ids = sorted(cl_model.vehicles)
    lane_by_vehicle = (
        _relabel_lanes_for_symmetry(raw_lane_by_vehicle, sorted_ids)
        if break_symmetry
        else raw_lane_by_vehicle
    )

    # Step 2: per served vehicle, discretize [service_start, departure) onto
    # this model's own slot grid and its own window, resampling the EV's
    # charge_trace ((t, s, p_req, p_act, p_allot), sorted by t) as a
    # piecewise-constant hold of the most recently recorded p_act.
    schedules: dict[int, tuple[int, int, dict[int, float]]] = {}
    for j, (pile, connector) in lane_by_vehicle.items():
        if pile >= cl_model.station.n_piles or connector >= cl_model.station.n_connectors:
            raise ValueError(
                f"Vehicle {j}'s simulated lane (pile={pile}, connector={connector}) doesn't "
                f"fit this model's station (n_piles={cl_model.station.n_piles}, "
                f"n_connectors={cl_model.station.n_connectors}) -- was warm_start_evs run on "
                "a different station layout than this model?"
            )
        ev = evs_by_id[j]
        assert ev.service_start_time is not None  # guaranteed by Step 1's filter
        k0 = cl_model.releases[j]
        k_start = max(k0, math.floor(round(ev.service_start_time / delta, 9)))
        k_departure = min(cl_model.K, math.ceil(round(ev.departure_time / delta, 9)))
        if k_departure <= k_start:
            continue  # degenerate/instant service -- nothing meaningful to seed

        trace = sorted(ev.charge_trace, key=lambda row: row[0])
        p_at: dict[int, float] = {}
        ti = 0
        current_p = 0.0
        for k in range(k_start, k_departure):
            t_slot = k * delta
            while ti < len(trace) and trace[ti][0] <= t_slot + 1e-9:
                current_p = trace[ti][3]  # p_act
                ti += 1
            p_at[k] = max(0.0, current_p)
        schedules[j] = (k_start, k_departure, p_at)

    # Step 3: apply as .Start -- y, u, p, eta, x for every seeded vehicle,
    # and b for any pair that (after relabelling) shares a lane. x is
    # reconstructed from this same p_at via the model's own recursion (17)
    # -- a plain float accumulator here, not a Gurobi expression, since this
    # is only computing .Start *values*, not building constraint rows.
    h = delta / 60.0
    lanes_used: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for j, (pile, connector) in lane_by_vehicle.items():
        if j not in schedules:
            continue
        k_start, k_departure, p_at = schedules[j]
        k0 = cl_model.releases[j]

        for (mm, cc) in cl_model.lanes:
            cl_model.y[j, mm, cc].Start = 1 if (mm, cc) == (pile, connector) else 0
        running = 0.0
        for k in range(k0, cl_model.K):
            occupied = k_start <= k < k_departure
            p_val = p_at.get(k, 0.0) if occupied else 0.0
            cl_model.u[j, k].Start = 1 if occupied else 0
            cl_model.p[j, k].Start = p_val
            cl_model.eta[j, k].Start = 1 if k == k_start else 0
            running += h * p_val
            if (j, k + 1) in cl_model.x:
                cl_model.x[j, k + 1].Start = running

        lanes_used[pile, connector].append((j, k_start, k_departure))

    # b_ij: whichever of a lane-sharing pair starts first precedes the other.
    for occupants in lanes_used.values():
        occupants.sort(key=lambda item: item[1])
        for a in range(len(occupants) - 1):
            i_id, _, _ = occupants[a]
            j_id, _, _ = occupants[a + 1]
            lo, hi = (i_id, j_id) if i_id < j_id else (j_id, i_id)
            if (lo, hi) in cl_model.b:
                cl_model.b[lo, hi].Start = 1 if lo == i_id else 0

    cl_model.model.update()


@dataclass
class AdaptiveSolveResult:
    """Outcome of ``solve_cl_model_adaptive``."""

    cl_model: ConnectorLaneModel
    iterations: int
    # z_1 <= z_2 <= ... -- each entry is one iteration's solved objective;
    # non-decreasing by Proposition 3's first inequality (every z_t is a
    # valid lower bound on the true optimum, since relaxing integrality only
    # enlarges the feasible set), and the last entry is z* once converged.
    objective_history: list[float]
    # Every pile-slot ever promoted to integer, across all iterations.
    integer_pile_slots: set[tuple[int, int]] = field(default_factory=set)
    # Pile-slots newly promoted to integer at each iteration (index t = the
    # set that triggered re-solve t+1); lets a caller replay the exact same
    # promotion schedule elsewhere, e.g. to benchmark the basis warm start
    # against a from-scratch rebuild at each step.
    promotions_by_iteration: list[list[tuple[int, int]]] = field(default_factory=list)
    # True iff the loop stopped because rounding_test_failures came back
    # empty (Proposition 3's exact-optimality guarantee applies). False
    # only if max_iterations was hit first, which the proof says shouldn't
    # happen -- see solve_cl_model_adaptive's docstring.
    converged: bool = False


def _log(progress: bool, message: str) -> None:
    """Print one progress line if ``progress=True``; a no-op otherwise.
    Kept as a tiny helper so every call site below reads as plain English
    rather than repeating an ``if progress:`` guard everywhere."""
    if progress:
        print(message)


def solve_cl_model_adaptive(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    break_symmetry: bool = False,
    bound_departures: bool = True,
    warm_start_from_conservative: bool = True,
    warm_start_evs: list["EV"] | None = None,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
    presolve: int | None = None,
    pre_passes: int | None = None,
    cutoff: float | None = None,
    verbose: bool = False,
    progress: bool = False,
    max_iterations: int | None = None,
) -> AdaptiveSolveResult:
    """
    Section 8.2's 5-step adaptive-integrality procedure.

        1. Build and solve ``CL_R`` (``r`` continuous).
        2. Test every pile-slot with ``rounding_test_failures``.
        3. If none fail, stop -- the solution is exactly optimal for the
           fully-integer exact model (Proposition 3).
        4. Otherwise, promote every failing pile-slot's ``r`` to integer.
        5. Re-solve (same live model -- see this module's docstring on the
           basis warm start), and go to 2.

    Guaranteed by Proposition 3 to converge within ``M*K`` iterations;
    ``max_iterations`` (default ``M*K``) is a purely defensive cap that
    should never actually bind -- ``result.converged`` is ``False`` if it
    ever does, which would indicate a bug, not expected behavior.

    Parameters
    ----------
    break_symmetry, bound_departures :
        See ``build_cl_model``. Applied identically to the main model and
        (when ``warm_start_from_conservative=True``) to the conservative
        seed model, so their symmetry constraints always match -- required
        for ``_apply_mip_start`` to be valid.
    warm_start_from_conservative :
        See this module's docstring, "MIP start". On by default; costs one
        extra (cheap-ish) solve up front. Ignored when ``warm_start_evs``
        is given (see below).
    warm_start_evs :
        Optional list of simulator ``EV`` objects (e.g.
        ``env.engine.metrics.finished_evs`` from a real, causal simulation
        on the same vehicles/station) to seed the MIP start from instead of
        the conservative shortcut -- see ``_seed_values_from_evs``. Needs
        no MILP solve to build, unlike ``warm_start_from_conservative``,
        but is only as good a starting point as the simulation itself was.
        When given, this takes priority and ``warm_start_from_conservative``
        is ignored. If ``break_symmetry=True``, the EVs' raw pile/connector
        assignment is relabelled into the canonical (25)-(26) order first
        (a real simulation has no reason to already respect it) -- see
        ``_relabel_lanes_for_symmetry``.
    mip_gap, time_limit, threads, presolve, pre_passes, verbose :
        Apply to *every* Gurobi solve in the loop, including the
        conservative warm-start solve -- see ``solve_cl_model``'s own
        docstring for ``presolve``/``pre_passes``. ``verbose`` toggles each
        individual solve's own Gurobi console log (can be very noisy across
        several iterations); use ``progress`` instead for a compact
        per-iteration summary of the adaptive loop itself.
    cutoff :
        See ``solve_cl_model``'s own docstring. Applied to every iteration
        of the loop (not the conservative warm-start solve, whose whole
        purpose is to *discover* a UB, not to be accelerated by one) --
        always safe here even though most iterations solve a relaxation
        ``M(I_t)`` rather than the exact model: every ``M(I_t)`` has
        ``z(I_t) <= z*`` (Section 8.2's own monotonicity argument), so any
        valid upper bound on the exact optimum ``z*`` is automatically also
        a valid, non-binding-until-it-should-be cutoff for every
        intermediate relaxation solved along the way.
    progress :
        If True, print one line per iteration (iteration number, objective,
        Gurobi's own MIP gap, wall-clock time for that solve, and how many
        pile-slots failed/were promoted) plus a final summary -- independent
        of ``verbose``, which is about Gurobi's own log, not this loop's.
    max_iterations :
        Safety cap on the loop, default ``M*K`` (Proposition 3's own
        finiteness bound).
    """
    K = math.ceil(round(horizon_minutes / delta, 9))
    total_pile_slots = station.n_piles * K  # M*K -- also Proposition 3's iteration cap
    if max_iterations is None:
        max_iterations = total_pile_slots

    # Step 1: build CL_R -- everything binary/integer as usual (u, y, b)
    # except r, which is relaxed to continuous so the solver never branches
    # on module routing at all in this first pass.
    cl_model = build_cl_model(
        vehicles,
        station,
        delta,
        horizon_minutes,
        relax_modules=True,
        break_symmetry=break_symmetry,
        bound_departures=bound_departures,
    )

    if warm_start_evs is not None:
        _log(progress, "[adaptive] building MIP start from the supplied simulation output...")
        t0 = time.time()
        _seed_values_from_evs(cl_model, warm_start_evs, break_symmetry=break_symmetry)
        _log(progress, f"[adaptive] MIP start ready from simulation ({time.time() - t0:.2f}s)")
    elif warm_start_from_conservative:
        _log(progress, "[adaptive] building MIP start from the conservative shortcut (8.4)...")
        t0 = time.time()
        # Built with the *same* break_symmetry/bound_departures as the
        # target above, so its variables line up index-for-index with
        # cl_model's -- see _apply_mip_start.
        seed = conservative_feasible_solution(
            vehicles,
            station,
            delta,
            horizon_minutes,
            break_symmetry=break_symmetry,
            bound_departures=bound_departures,
            mip_gap=mip_gap,
            time_limit=time_limit,
            threads=threads,
            presolve=presolve,
            pre_passes=pre_passes,
            verbose=False,
        )
        _apply_mip_start(cl_model, seed)
        _log(
            progress,
            f"[adaptive] MIP start ready: conservative objective={seed.model.ObjVal:.3f} "
            f"({time.time() - t0:.2f}s)",
        )

    integer_pile_slots: set[tuple[int, int]] = set()
    promotions_by_iteration: list[list[tuple[int, int]]] = []
    history: list[float] = []
    converged = False
    it = 0
    while it < max_iterations:
        it += 1
        t0 = time.time()
        # Step 2/5: (re-)solve the current model -- same live gurobipy
        # Model object every time, so Gurobi can warm-start the root LP
        # from whatever basis the previous solve (if any) left behind; see
        # this module's docstring, "Basis warm start".
        solve_cl_model(
            cl_model,
            mip_gap=mip_gap,
            time_limit=time_limit,
            threads=threads,
            presolve=presolve,
            pre_passes=pre_passes,
            cutoff=cutoff,
            verbose=verbose,
        )
        solve_time = time.time() - t0
        history.append(float(cl_model.model.ObjVal))

        # Step 2: does the current (partially relaxed) solution's power
        # profile already imply a valid whole-module routing everywhere?
        failures = rounding_test_failures(cl_model)
        _log(
            progress,
            f"[adaptive] iter {it}: objective={cl_model.model.ObjVal:.4f}, "
            f"gap={cl_model.model.MIPGap:.2%}, solve_time={solve_time:.2f}s, "
            f"integer pile-slots so far={len(integer_pile_slots)}/{total_pile_slots}, "
            f"failing this iteration={len(failures)}",
        )

        # Step 3: stop -- Proposition 3 says this solution is exactly
        # optimal for the fully-integer exact model, not just a good guess.
        if not failures:
            converged = True
            _log(progress, f"[adaptive] converged after {it} iteration(s).")
            break

        # Step 4: promote every failing pile-slot's r (all C connectors of
        # that pile, at that slot) from continuous to integer in place --
        # mutating the live model, not rebuilding it (see the module
        # docstring on why this preserves the warm-startable basis).
        for (mm, k) in failures:
            for cc in range(station.n_connectors):
                cl_model.r[mm, cc, k].VType = GRB.INTEGER
            integer_pile_slots.add((mm, k))
        promotions_by_iteration.append(failures)
        cl_model.model.update()

    if not converged:
        _log(
            progress,
            f"[adaptive] hit max_iterations={max_iterations} without converging -- "
            "this should not happen per Proposition 3; treat as a bug report.",
        )

    return AdaptiveSolveResult(
        cl_model=cl_model,
        iterations=it,
        objective_history=history,
        integer_pile_slots=integer_pile_slots,
        promotions_by_iteration=promotions_by_iteration,
        converged=converged,
    )
