"""
Inputs the algorithm needs -- Section 8 of ``dantzig_wolfe_decomposition.html``.

  - ``earliest_departures`` (8.1, eq. 34): re-exported directly from
    ``offline_cl_opt.model`` -- the identical greedy single-vehicle
    recursion (eq. 22 there), needed here for the same two reasons Section
    8.1 gives: as the pricer's own valid inequality ``D_j >= E_j``
    (``pricer.build_pricer``), and as a cheap sanity-check lower bound
    ``sum_j E_j`` on the optimum from the outset.
  - ``seed_columns``: builds the initial column pool column generation
    needs (Section 7.2) -- the null plan for every vehicle (mandatory, is
    what makes the restricted master feasible from iteration one with no
    artificial variables) plus, for every vehicle and pile, the
    "charge at the acceptance limit from k_j" plan -- individually optimal
    for that (vehicle, pile) pair alone, and cheap to build (no MILP
    solve, the same greedy recursion ``earliest_departures`` already uses).
  - ``columns_from_evs``: an optional second seed source, converting a real
    simulation's own output into plans -- the document's other suggested
    source ("simulate first-come-first-served... convert each vehicle's
    realised schedule into a plan"), generalised here to accept any
    already-run simulation (FCFS or otherwise), matching the
    ``warm_start_evs`` pattern ``offline_cl_opt.adaptive`` already uses for
    the same purpose in the compact model.
  - ``boundary_seed_plan``/``fixed_plan``: seeding for boundary vehicles
    (``offline_cl_opt.boundary.BoundaryVehicle``, see that module's own
    docstring and ``offline_cl_dw.colgen``'s "Boundary conditions") --
    these never get the ordinary null-plan-based seeding above, since a
    vehicle already queued/plugged in cannot be "never served".
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from offline_cl_opt.boundary import BoundaryMode, BoundaryVehicle
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import earliest_departures  # noqa: F401  (re-exported)

from .columns import Plan, null_plan
from .pricer import _release_slot

if TYPE_CHECKING:
    from models.ev import EV


def _greedy_solo_plan(
    v: VehicleData,
    pile: int,
    station: StationSpec,
    delta: float,
    K: int,
    *,
    initial_energy: float = 0.0,
) -> Plan:
    """
    The "charge at the acceptance limit from k_j" plan for vehicle ``v`` on
    ``pile`` -- almost the same discrete trajectory ``earliest_departures``
    simulates (eq. 34), just also keeping the power profile instead of only
    its stopping slot. Individually optimal for this (vehicle, pile) pair
    alone (no contention assumed), and needs no MILP solve.

    One real difference from ``earliest_departures``'s own recursion: that
    function's ``step_power = min(p_bar, (R-x)/tau_d)`` is capped only by
    distance to a *full* battery (``R``), which is fine there since it only
    ever reads off the slot index at which ``x`` first reaches ``W_j`` --
    the exact energy value in that final step is never used. Here the
    power profile itself becomes a real column, so an uncapped final step
    can deliver strictly more than ``W_j`` (whenever ``R > W_j``, i.e. the
    driver isn't charging to 100%) -- a genuine, if small, violation of
    ``x_jk <= W_j``. A third cap, ``(W_j-x)/h`` -- "don't deliver more this
    slot than what's left to reach exactly W_j" -- fixes it.

    ``initial_energy``: 0.0 for an ordinary vehicle; for an OPTIMIZE-mode
    boundary vehicle (see ``boundary_seed_plan`` below) this seeds ``x``
    from how much it already has instead of 0 -- same "x keeps meaning
    energy delivered since arrival" convention as
    ``offline_cl_opt.model``/``pricer.build_pricer``.
    """
    h = delta / 60.0
    N, Delta = station.n_modules, station.p_module
    p_bar = min(v.p_max, N * Delta)
    tau_d = v.tau_delta_hours(delta)
    k0 = _release_slot(v.a, delta)

    power: dict[int, float] = {}
    x = initial_energy
    k = k0
    while k < K and x < v.W - 1e-9:
        step_power = min(p_bar, (v.R - x) / tau_d, (v.W - x) / h)
        power[k] = step_power
        x += h * step_power
        k += 1
    departure = min(K, k)
    if not power:
        return null_plan(v.id, K)
    return Plan(vehicle_id=v.id, pile=pile, start=k0, departure=departure, power=power)


def boundary_seed_plan(
    v: VehicleData, bv: BoundaryVehicle, station: StationSpec, delta: float, K: int
) -> Plan:
    """
    A guaranteed-feasible starting column for an OPTIMIZE-mode boundary
    vehicle (``offline_cl_opt.boundary.BoundaryVehicle``).

    These vehicles get no null plan (a vehicle already mid-charge cannot be
    "never served" -- see ``run_column_generation``'s own docstring,
    "Boundary conditions"), so the master needs *some* feasible column for
    them from iteration one, the role the null plan plays for everyone
    else. Unlike an ordinary vehicle's seed, though, that column is not
    optional: convexity forces it to lambda=1, so it lands in the capacity
    rows (25)/(26) whether it fits or not.

    That is why the simulation's own realized trajectory
    (``BoundaryVehicle.seed_power``) is preferred over the greedy fallback.
    A greedy solo plan is built as if the vehicle owned the whole pile
    (``_greedy_solo_plan``'s ``p_bar = min(p_max, N*Delta)``), so two
    boundary vehicles sharing a pile seed two columns that each demand the
    full pool -- and with no null plan to fall back on, the restricted
    master is infeasible before column generation starts. The realized
    trajectories cannot do that: they actually coexisted, so they satisfy
    the capacity rows jointly by construction.

    The realized profile is also feasible for this vehicle's *own*
    subproblem, which (unlike a FIXED vehicle's) really does enforce
    (17)-(19): its energy-equivalent power satisfies the discrete taper
    (18) automatically (see ``boundary.realized_slot_power``), and its
    departure slot rounds *up*, so cumulative energy reaches exactly
    ``W_j`` and the departure rule (19) holds -- flooring, as FIXED does,
    would truncate energy and make the column infeasible here.

    Falls back to the greedy construction only when no realized profile was
    recorded (e.g. a hand-built ``BoundaryVehicle``), which is safe for a
    single boundary vehicle per pile but not in general.
    """
    assert bv.mode is BoundaryMode.OPTIMIZE
    if bv.seed_power and bv.seed_departure_slot > 0:
        return Plan(
            vehicle_id=bv.vehicle_id,
            pile=bv.pile,
            start=0,
            departure=min(K, bv.seed_departure_slot),
            power=dict(bv.seed_power),
        )
    return _greedy_solo_plan(
        v, bv.pile, station, delta, K, initial_energy=bv.initial_energy_kwh
    )


def fixed_plan(bv: BoundaryVehicle) -> Plan:
    """
    The single, mandatory column for a FIXED-mode boundary vehicle -- its
    whole trajectory is already known (see
    ``offline_cl_opt.boundary.boundary_vehicles_from_in_service``), so this
    is its only valid plan; nothing else is ever seeded or priced for it
    (see ``run_column_generation``'s own docstring, "Boundary conditions").
    """
    assert bv.mode is BoundaryMode.FIXED
    return Plan(
        vehicle_id=bv.vehicle_id,
        pile=bv.pile,
        start=0,
        departure=bv.departure_slot,
        power=dict(bv.power),
    )


def seed_columns(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
) -> dict[int, list[Plan]]:
    """
    Section 7.2's initial column pool: the null plan (mandatory) plus, for
    every vehicle, one "charge at the limit" plan per pile.
    """
    columns: dict[int, list[Plan]] = {}
    for v in vehicles:
        plans = [null_plan(v.id, K)]
        for pile in range(station.n_piles):
            plan = _greedy_solo_plan(v, pile, station, delta, K)
            if not plan.is_null:
                plans.append(plan)
        columns[v.id] = plans
    return columns


def columns_from_evs(
    evs: list["EV"],
    vehicles_by_id: dict[int, VehicleData],
    delta: float,
    K: int,
) -> dict[int, list[Plan]]:
    """
    Build one plan per served, finished EV from a real simulation's own
    output (e.g. ``env.engine.metrics.finished_evs``) -- Section 7.2's
    other suggested seed source ("simulate first-come-first-served...").
    EVs with no real pile assignment, no finite departure, or not present
    in ``vehicles_by_id`` are simply skipped (nothing meaningful to seed).

    Unlike ``offline_cl_opt.adaptive._seed_values_from_evs``, this needs no
    symmetry-aware relabelling: the master's rows (25)/(26) don't
    distinguish piles by label the way ``break_symmetry`` does in the
    compact model, so a real simulation's raw pile indices are used as-is.
    Connector identity is dropped entirely (a plan only names a pile, not a
    connector -- see ``postprocess.assign_connectors`` for where it's
    reconstructed after the fact).
    """
    columns: dict[int, list[Plan]] = {}
    for ev in evs:
        j = ev.id
        v = vehicles_by_id.get(j)
        if v is None:
            continue
        if ev.pile_tracker is None or ev.service_start_time is None or not math.isfinite(ev.departure_time):
            continue
        pile = ev.pile_tracker.id
        k0 = _release_slot(v.a, delta)
        k_start = max(k0, math.floor(round(ev.service_start_time / delta, 9)))
        k_departure = min(K, math.ceil(round(ev.departure_time / delta, 9)))
        if k_departure <= k_start:
            continue

        trace = sorted(ev.charge_trace, key=lambda row: row[0])
        p_at: dict[int, float] = {}
        ti = 0
        current_p = 0.0
        for k in range(k_start, k_departure):
            t_slot = k * delta
            while ti < len(trace) and trace[ti][0] <= t_slot + 1e-9:
                current_p = trace[ti][3]  # p_act
                ti += 1
            if current_p > 1e-9:
                p_at[k] = current_p

        plan = Plan(vehicle_id=j, pile=pile, start=k_start, departure=k_departure, power=p_at)
        columns.setdefault(j, []).append(plan)
    return columns
