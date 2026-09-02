"""
End-to-end orchestration and a plain, DataFrame-friendly result object --
mirrors ``offline_cl_opt.solution.ConnectorLaneSolution`` in shape/naming
so the two packages feel consistent to use side by side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from offline_cl_opt.instance import StationSpec, VehicleData

from .colgen import ColGenResult, run_column_generation
from .columns import Plan
from .master import IntegerResult, solve_integer
from .postprocess import assign_connectors, validate_schedule, whole_module_failures


@dataclass
class DWSolution:
    """
    Bracket + schedule from ``solve_by_decomposition``.

    ``LB`` is ``best_lower_bound`` from column generation -- a certified
    lower bound on the true optimum (32), valid even if ``converged`` is
    False (an anytime bound, Section 6.4). ``UB`` is the price-and-branch
    integer master's objective (Section 9.1) -- a genuine feasible
    schedule's cost, valid regardless of convergence. ``gap`` is
    ``UB - LB``, the certified bracket width. All three are in the compact
    model's own raw objective units (eq. 1: ``sum_j D_j``, each vehicle's
    *absolute* departure slot counted from ``t=0``) -- **not** sojourn
    minutes, and not directly comparable to a simulation's total-sojourn
    metric (``sum_j (departure - arrival)``), which nets out arrival time
    and this objective does not. Comparing ``LB``/``UB`` straight against a
    sojourn number is an apples-to-oranges mistake that looks exactly like
    an invalid bound (e.g. ``LB`` appearing to exceed a feasible sojourn
    total) without actually being one.

    ``rmp_gap`` is a *different* quantity from ``gap`` -- it is
    ``z_RMP - LB`` at the end of column generation (the final entry of
    ``ColGenResult.z_rmp_history``, minus ``best_lower_bound``), Section
    7.4's own stopping-criterion gap. Unlike ``gap``, it is **not** a
    certified bound on the distance to the true optimum: the *restricted*
    master's own LP value ``z_RMP`` has no guaranteed ordering relative to
    the true optimum before convergence -- only ``LB`` does (see
    ``colgen.py``'s own module docstring). Read ``rmp_gap`` as "how close is
    column generation to proving ``z_RMP == z_MP``", never as a substitute
    for ``gap`` (``UB - LB``), which remains the only certified bracket.
    They typically end up close once ``converged`` is True, but are not the
    same thing even then, since price-and-branch's ``UB`` need not equal
    the LP optimum exactly.

    ``total_sojourn_UB``/``mean_sojourn_UB`` give the *schedule*'s (i.e.
    ``UB``'s) cost already converted to sojourn minutes -- mirrors
    ``offline_cl_opt.solution.ConnectorLaneSolution`` exactly (``delta*
    objective - sum_j a_j``) and equals ``per_vehicle["sojourn_min"].sum()``
    /``.mean()``. ``total_sojourn_LB`` applies the same conversion to
    ``LB`` instead, so it -- not ``LB`` itself -- is what should be
    compared against a simulation's total sojourn.

    ``columns_purged`` is copied straight from ``ColGenResult.columns_purged``
    (Section 8.3) -- the cumulative count of non-basic columns swept out of
    the master over the run, ``0`` if ``purge_every=None`` disabled it. Purely
    informational: purging never changes ``LB``/``UB``, only how much work
    reaching them cost.
    """

    LB: float
    UB: float
    gap: float
    rmp_gap: float
    total_sojourn_UB: float
    mean_sojourn_UB: float
    total_sojourn_LB: float
    converged: bool
    iterations: int
    columns_purged: int
    n_vehicles: int
    delta: float
    K: int
    whole_module_feasible: bool
    per_vehicle: pd.DataFrame  # vehicle_id, arrival, served, pile, connector,
    # start_slot, departure_slot, sojourn_min, energy_kwh, energy_required_kwh


def extract_solution(
    chosen: dict[int, Plan],
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
    *,
    LB: float,
    UB: float,
    rmp_gap: float,
    converged: bool,
    iterations: int,
    columns_purged: int = 0,
) -> DWSolution:
    """Build a ``DWSolution`` from an integer master result -- assigns
    connectors (10.1) and checks (not repairs) whole-module feasibility
    (10.2); see ``postprocess.rounded_module_routing`` if you need an
    actual whole-module routing, not just the pass/fail check. Prints a
    warning (does not raise, does not repair) if the check fails -- see the
    warning's own text for why ``UB`` stops being a valid upper bound on
    the true (whole-module) optimum in that case, even though ``LB``
    remains valid regardless."""
    h = delta / 60.0
    connectors = assign_connectors(chosen, station)
    failures = whole_module_failures(chosen, station)
    if failures:
        # (26) only enforces the *continuous*-module relaxation
        # (Proposition 2), so the schedule UB was computed from may not be
        # realisable with real, indivisible power modules at these
        # pile-slots -- meaning it is not actually a feasible solution to
        # the true (whole-module) problem, and UB is therefore not a valid
        # upper bound on that problem's optimum until this is repaired
        # (LB is unaffected: it lower-bounds the continuous relaxation,
        # which is itself <= the true optimum, regardless of this check).
        print(
            f"WARNING: whole-module infeasible at {len(failures)} pile-slot(s) "
            f"{failures[:5]}{', ...' if len(failures) > 5 else ''} -- UB's schedule "
            "cannot actually be delivered with real, indivisible power modules there "
            "(Proposition 2's continuous-module relaxation, not the true problem). "
            "UB is NOT a valid upper bound on the true optimum until this is repaired "
            "-- rerun with conservative_modules=True, or see "
            "postprocess.rounded_module_routing / Section 10.2 for other repair "
            "options. LB remains a valid lower bound regardless."
        )

    rows: list[dict[str, object]] = []
    for v in vehicles:
        plan = chosen[v.id]
        served = not plan.is_null
        energy_kwh = h * sum(plan.power.values())
        start_slot = (
            float(plan.start) if served and plan.start is not None else float(K)
        )
        rows.append(
            {
                "vehicle_id": v.id,
                "arrival": v.a,
                "served": served,
                "pile": plan.pile,
                "connector": connectors.get(v.id) if served else None,
                "start_slot": start_slot,
                "departure_slot": float(plan.departure),
                "sojourn_min": delta * plan.departure - v.a,
                "energy_kwh": energy_kwh,
                "energy_required_kwh": v.W,
            }
        )
    per_vehicle = pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)

    n = len(vehicles)
    total_arrival = sum(v.a for v in vehicles)
    total_sojourn_UB = delta * UB - total_arrival
    mean_sojourn_UB = total_sojourn_UB / n if n else 0.0
    total_sojourn_LB = delta * LB - total_arrival

    return DWSolution(
        LB=LB,
        UB=UB,
        gap=UB - LB,
        rmp_gap=rmp_gap,
        total_sojourn_UB=total_sojourn_UB,
        mean_sojourn_UB=mean_sojourn_UB,
        total_sojourn_LB=total_sojourn_LB,
        converged=converged,
        iterations=iterations,
        columns_purged=columns_purged,
        n_vehicles=len(vehicles),
        delta=delta,
        K=K,
        whole_module_feasible=not failures,
        per_vehicle=per_vehicle,
    )


def solve_by_decomposition(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    extra_seed_columns: dict[int, list[Plan]] | None = None,
    conservative_modules: bool = False,
    gamma: float = 0.5,
    eps_rc: float = 1e-6,
    gap_tolerance: float = 1e-6,
    max_iterations: int = 500,
    max_columns_per_round: int | None = None,
    time_limit: float | None = None,
    pricer_mip_gap: float | None = 1e-2,
    max_workers: int | None = None,
    pricer_threads: int | None = 1,
    exact_mip_gap: float | None = 0.0,
    purge_every: int | None = 25,
    purge_threshold: float = 10.0,
    integer_mip_gap: float | None = 1e-3,
    integer_time_limit: float | None = None,
    validate: bool = True,
    progress: bool = False,
) -> tuple[DWSolution, ColGenResult]:
    """
    The whole pipeline, end to end: column generation for the bracket
    (Section 7), then price-and-branch (Section 9.1) for a feasible
    schedule. Returns ``(solution, colgen_result)`` -- keep
    ``colgen_result`` if you want the iteration-by-iteration history
    (``z_rmp_history``/``lower_bound_history``) or want to add more
    columns and re-run price-and-branch yourself (e.g. after inspecting a
    wide gap).

    See ``colgen.run_column_generation`` for the meaning of ``gamma``,
    ``eps_rc``, ``gap_tolerance``, ``max_iterations``,
    ``max_columns_per_round``, ``time_limit``, ``pricer_mip_gap``,
    ``exact_mip_gap``, ``max_workers``, ``pricer_threads``, ``purge_every``,
    and ``purge_threshold`` -- ``max_workers``/``pricer_threads`` are the
    main lever for wall-clock time on a large instance (many vehicles/piles,
    or a long horizon making each pricer itself slow): pricing every
    ``(vehicle, pile)`` pair concurrently is where most of the time goes
    at that scale, far more than the master LP.

    ``pricer_mip_gap`` only loosens the *candidate-hunting* pricer solves
    (smoothed-dual rounds); rounds that determine ``best_lower_bound`` or
    convergence always solve to ``exact_mip_gap`` regardless, since Section
    6.3 of the source document requires the final pricing pass to be
    genuinely optimal -- a nonzero gap there would make each ``zeta_jm``
    only an upper bound on the true pricer optimum, which can silently
    make ``best_lower_bound`` exceed the true optimum (``postprocess.
    validate_schedule``'s ``best_lower_bound`` check exists to catch
    exactly that).

    ``purge_every``/``purge_threshold`` (Section 8.3): every ``purge_every``
    iterations, sweep non-basic master columns whose reduced cost exceeds
    ``purge_threshold`` -- keeps ``solve_lp``'s own cost from growing
    unbounded as the master accumulates columns over a long run. Set
    ``purge_every=None`` to disable. Never changes ``LB``/``UB``; a purged
    column can always be regenerated later if it
    becomes attractive again (the pricer searches the full plan space, not
    the historical pool -- see ``master.purge_columns``'s own docstring).
    The cumulative count removed is reported on the returned
    ``DWSolution.columns_purged`` and, when ``progress=True``, printed
    inline whenever a sweep actually removes something.

    ``validate``: run ``postprocess.validate_schedule``'s full assertion
    checklist (Section 10.3) on the result before returning. Leave this on
    unless you have a specific reason not to -- it is cheap relative to
    the solve itself and catches exactly the class of bug ("the bound
    looked fine but the schedule wasn't really feasible") that a
    performance-ceiling result can least afford to hide.
    """
    cg = run_column_generation(
        vehicles,
        station,
        delta,
        horizon_minutes,
        extra_seed_columns=extra_seed_columns,
        conservative_modules=conservative_modules,
        gamma=gamma,
        eps_rc=eps_rc,
        gap_tolerance=gap_tolerance,
        max_iterations=max_iterations,
        max_columns_per_round=max_columns_per_round,
        time_limit=time_limit,
        pricer_mip_gap=pricer_mip_gap,
        max_workers=max_workers,
        pricer_threads=pricer_threads,
        exact_mip_gap=exact_mip_gap,
        purge_every=purge_every,
        purge_threshold=purge_threshold,
        progress=progress,
    )

    integer_result: IntegerResult = solve_integer(
        cg.master, mip_gap=integer_mip_gap, time_limit=integer_time_limit
    )

    K = math.ceil(round(horizon_minutes / delta, 9))
    z_rmp_final = cg.z_rmp_history[-1] if cg.z_rmp_history else float("nan")
    solution = extract_solution(
        integer_result.chosen,
        vehicles,
        station,
        delta,
        K,
        LB=cg.best_lower_bound,
        UB=integer_result.objective,
        rmp_gap=z_rmp_final - cg.best_lower_bound,
        converged=cg.converged,
        iterations=cg.iterations,
        columns_purged=cg.columns_purged,
    )

    if validate:
        validate_schedule(
            integer_result.chosen,
            {v.id: v for v in vehicles},
            station,
            delta,
            K,
            best_lower_bound=cg.best_lower_bound,
        )

    return solution, cg
