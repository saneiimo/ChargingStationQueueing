"""
End-to-end orchestration and a plain, DataFrame-friendly result object --
mirrors ``offline_cl_opt.solution.ConnectorLaneSolution`` in shape/naming
so the two packages feel consistent to use side by side.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from offline_cl_opt.boundary import (
    COHORTS_ALL,
    BoundaryVehicle,
    Cohort,
    cohort_totals,
)
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import sojourn_minutes

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
    ``UB - LB``, the certified bracket width. All three are in the
    objective's own units: total sojourn ``sum_j (delta*D_j - a_j)`` in
    minutes over ``objective_cohorts`` -- the same objective as
    ``offline_cl_opt``'s compact model, so directly comparable to it and to
    a simulation's total sojourn over the same vehicles.

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

    ``total_sojourn_UB`` equals ``UB`` and ``mean_sojourn_UB`` is it divided
    by ``n_optimized`` (equal to ``per_vehicle["sojourn_min"]`` summed /
    averaged over the objective's vehicles); ``total_sojourn_LB`` equals
    ``LB`` and ``mean_sojourn_LB`` is it per vehicle -- a valid lower bound
    on the optimal mean sojourn, since ``LB <= z*``.

    ``columns_purged`` is copied straight from ``ColGenResult.columns_purged``
    (Section 8.3) -- the cumulative count of non-basic columns swept out of
    the master over the run, ``0`` if ``purge_every=None`` disabled it. Purely
    informational: purging never changes ``LB``/``UB``, only how much work
    reaching them cost.

    With ``solve_by_decomposition(..., solve_integer_ub=False)`` no integer
    master is solved: ``UB``, ``gap`` and both ``*_UB`` sojourn figures are
    ``nan``, ``per_vehicle`` has no schedule columns, and
    ``whole_module_feasible`` is vacuously ``True`` (no schedule was
    checked). Everything on the ``LB`` side stays valid.

    ``LB``/``UB`` and every sojourn figure derived from them cover exactly
    the vehicles the objective was summed over
    (``ColGenResult.objective_cohorts``). ``by_cohort`` additionally
    reports total/mean sojourn of the *schedule* for all three nested
    cohort levels -- ``"measurement"``, ``"measurement_queued"``, ``"all"``
    -- regardless of what was optimised. There is no per-cohort ``LB``:
    the lower bound is a single scalar certifying the objective as a
    whole, so it cannot be split across cohorts after the fact.
    """

    LB: float
    UB: float
    gap: float
    rmp_gap: float
    total_sojourn_UB: float
    mean_sojourn_UB: float
    total_sojourn_LB: float
    mean_sojourn_LB: float
    converged: bool
    iterations: int
    columns_purged: int
    n_vehicles: int  # every vehicle in the model, optimized or not
    n_optimized: int  # those inside objective_cohorts
    delta: float
    K: int
    whole_module_feasible: bool
    by_cohort: dict[str, dict[str, float]]
    per_vehicle: pd.DataFrame  # vehicle_id, cohort, boundary_mode, in_objective,
    # arrival, served, pile, connector, start_slot, departure_slot, sojourn_min,
    # energy_kwh, energy_delivered_before_kwh, energy_required_kwh


def extract_solution(
    chosen: dict[int, Plan] | None,
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
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
    cohorts: dict[int, Cohort] | None = None,
    objective_cohorts: frozenset[Cohort] | None = None,
) -> DWSolution:
    """Build a ``DWSolution`` from an integer master result -- assigns
    connectors (10.1) and checks (not repairs) whole-module feasibility
    (10.2); see ``postprocess.rounded_module_routing`` if you need an
    actual whole-module routing, not just the pass/fail check. Prints a
    warning (does not raise, does not repair) if the check fails -- see the
    warning's own text for why ``UB`` stops being a valid upper bound on
    the true (whole-module) optimum in that case, even though ``LB``
    remains valid regardless.

    ``chosen=None`` builds an **LB-only** solution: no integer master was
    solved, so no schedule exists. Every schedule-derived figure is
    ``nan`` (pass ``UB=float("nan")`` to keep ``gap``/``*_UB`` consistent)
    and ``per_vehicle`` keeps only the identity/cohort/arrival columns.
    ``by_cohort`` still reports correct per-cohort ``n`` (it comes from the
    vehicle list, not the schedule) with ``nan`` sojourns."""
    h = delta / 60.0
    # No schedule to post-process in LB-only mode; the whole-module check is
    # vacuous rather than passing, so nothing is reported as a failure.
    connectors = assign_connectors(chosen, station) if chosen is not None else {}
    failures = whole_module_failures(chosen, station) if chosen is not None else []
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

    boundary_vehicles = boundary_vehicles or {}
    cohorts = cohorts or {}
    objective_cohorts = (
        objective_cohorts if objective_cohorts is not None else COHORTS_ALL
    )

    nan = float("nan")
    rows: list[dict[str, object]] = []
    for v in vehicles:
        # LB-only mode (chosen is None): the identity / cohort / arrival
        # fields are still meaningful, every schedule-derived one is nan.
        plan = chosen[v.id] if chosen is not None else None
        served = (not plan.is_null) if plan is not None else None
        energy_kwh = h * sum(plan.power.values()) if plan is not None else nan
        start_slot = departure_slot = sojourn_min = nan
        if plan is not None:
            start_slot = (
                float(plan.start) if served and plan.start is not None else float(K)
            )
            departure_slot = float(plan.departure)
            sojourn_min = sojourn_minutes(plan.departure, v.a, delta)
        # A boundary vehicle's plan only covers the modeled window, so its
        # in-window energy is reported next to what it already had at t=0 --
        # otherwise the row reads as a shortfall against energy_required.
        bv = boundary_vehicles.get(v.id)
        cohort = cohorts.get(v.id, Cohort.MEASUREMENT)
        rows.append(
            {
                "vehicle_id": v.id,
                "cohort": cohort.value,
                "boundary_mode": bv.mode.value if bv is not None else None,
                "in_objective": cohort in objective_cohorts,
                "arrival": v.a,
                "served": served,
                "pile": plan.pile if plan is not None else None,
                "connector": connectors.get(v.id) if served else None,
                "start_slot": start_slot,
                "departure_slot": departure_slot,
                "sojourn_min": sojourn_min,
                "energy_kwh": energy_kwh,
                "energy_delivered_before_kwh": bv.initial_energy_kwh if bv is not None else 0.0,
                "energy_required_kwh": v.W,
            }
        )
    per_vehicle = pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)
    by_cohort = cohort_totals(rows, cohorts)

    # LB/UB already are total sojourn over objective_cohorts (minutes).
    optimized = [r for r in rows if r["in_objective"]]
    n = len(optimized)
    total_sojourn_UB = UB
    # nan on an empty objective (see offline_cl_opt.solution) -- never 0.0.
    mean_sojourn_UB = total_sojourn_UB / n if n else float("nan")
    total_sojourn_LB = LB
    mean_sojourn_LB = total_sojourn_LB / n if n else float("nan")

    return DWSolution(
        LB=LB,
        UB=UB,
        gap=UB - LB,
        rmp_gap=rmp_gap,
        total_sojourn_UB=total_sojourn_UB,
        mean_sojourn_UB=mean_sojourn_UB,
        total_sojourn_LB=total_sojourn_LB,
        mean_sojourn_LB=mean_sojourn_LB,
        converged=converged,
        iterations=iterations,
        columns_purged=columns_purged,
        n_vehicles=len(vehicles),
        n_optimized=n,
        delta=delta,
        K=K,
        whole_module_feasible=not failures,
        by_cohort=by_cohort,
        per_vehicle=per_vehicle,
    )


def solve_by_decomposition(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    extra_seed_columns: dict[int, list[Plan]] | None = None,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
    cohorts: dict[int, Cohort] | None = None,
    objective_cohorts: frozenset[Cohort] | None = None,
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
    purge_threshold: float | None = None,
    integer_mip_gap: float | None = 1e-3,
    integer_time_limit: float | None = None,
    solve_integer_ub: bool = True,
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

    ``boundary_vehicles``: forwarded as-is to ``run_column_generation`` and
    ``postprocess.validate_schedule`` -- see ``colgen.run_column_generation``'s
    own docstring, "Boundary conditions", for what it does.

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

    ``solve_integer_ub``: set ``False`` to stop after column generation and
    skip price-and-branch (Section 9.1) entirely -- useful when only the
    certified lower bound is wanted and the integer master is the expensive
    part. No schedule is produced, so ``UB``/``gap`` and every ``*_UB``
    figure on the returned ``DWSolution`` are ``nan``, ``per_vehicle``
    carries no schedule columns, ``by_cohort`` reports ``n`` with ``nan``
    sojourns, and ``validate`` is skipped (there is nothing to validate).
    ``LB`` and its ``*_LB`` sojourn conversions are unaffected -- they come
    from column generation alone.

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
        boundary_vehicles=boundary_vehicles,
        cohorts=cohorts,
        objective_cohorts=objective_cohorts,
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

    integer_result: IntegerResult | None = None
    if solve_integer_ub:
        integer_result = solve_integer(
            cg.master, mip_gap=integer_mip_gap, time_limit=integer_time_limit
        )

    K = math.ceil(round(horizon_minutes / delta, 9))
    z_rmp_final = cg.z_rmp_history[-1] if cg.z_rmp_history else float("nan")
    solution = extract_solution(
        integer_result.chosen if integer_result is not None else None,
        vehicles,
        station,
        delta,
        K,
        LB=cg.best_lower_bound,
        UB=integer_result.objective if integer_result is not None else float("nan"),
        rmp_gap=z_rmp_final - cg.best_lower_bound,
        converged=cg.converged,
        iterations=cg.iterations,
        columns_purged=cg.columns_purged,
        boundary_vehicles=cg.boundary_vehicles,
        cohorts=cg.cohorts,
        objective_cohorts=cg.objective_cohorts,
    )

    if validate and integer_result is not None:
        validate_schedule(
            integer_result.chosen,
            {v.id: v for v in vehicles},
            station,
            delta,
            K,
            best_lower_bound=cg.best_lower_bound,
            boundary_vehicles=boundary_vehicles,
            objective_ids=cg.master.objective_ids,
        )

    return solution, cg
