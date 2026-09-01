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

    ``lower_bound`` is ``best_lower_bound`` from column generation -- a
    certified lower bound on the true optimum (32), valid even if
    ``converged`` is False (an anytime bound, Section 6.4). ``upper_bound``
    is the price-and-branch integer master's objective (Section 9.1) -- a
    genuine feasible schedule's cost, valid regardless of convergence.
    ``gap`` is ``upper_bound - lower_bound``. All three are in the compact
    model's own raw objective units (eq. 1: ``sum_j D_j``, each vehicle's
    *absolute* departure slot counted from ``t=0``) -- **not** sojourn
    minutes, and not directly comparable to a simulation's total-sojourn
    metric (``sum_j (departure - arrival)``), which nets out arrival time
    and this objective does not. Comparing ``lower_bound``/``upper_bound``
    straight against a sojourn number is an apples-to-oranges mistake that
    looks exactly like an invalid bound (e.g. ``lower_bound`` appearing to
    exceed a feasible sojourn total) without actually being one.

    ``total_sojourn``/``mean_sojourn`` give the *schedule*'s (i.e.
    ``upper_bound``'s) cost already converted to sojourn minutes -- mirrors
    ``offline_cl_opt.solution.ConnectorLaneSolution`` exactly (``delta*
    objective - sum_j a_j``) and equals ``per_vehicle["sojourn_min"].sum()``
    /``.mean()``. ``total_sojourn_lower_bound`` applies the same conversion
    to ``lower_bound`` instead, so it -- not ``lower_bound`` itself -- is
    what should be compared against a simulation's total sojourn.

    ``columns_purged`` is copied straight from ``ColGenResult.columns_purged``
    (Section 8.3) -- the cumulative count of non-basic columns swept out of
    the master over the run, ``0`` if ``purge_every=None`` disabled it. Purely
    informational: purging never changes ``lower_bound``/``upper_bound``, only
    how much work reaching them cost.
    """

    lower_bound: float
    upper_bound: float
    gap: float
    total_sojourn: float
    mean_sojourn: float
    total_sojourn_lower_bound: float
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
    lower_bound: float,
    upper_bound: float,
    converged: bool,
    iterations: int,
    columns_purged: int = 0,
) -> DWSolution:
    """Build a ``DWSolution`` from an integer master result -- assigns
    connectors (10.1) and checks (not repairs) whole-module feasibility
    (10.2); see ``postprocess.rounded_module_routing`` if you need an
    actual whole-module routing, not just the pass/fail check."""
    h = delta / 60.0
    connectors = assign_connectors(chosen, station)
    failures = whole_module_failures(chosen, station)

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
    total_sojourn = delta * upper_bound - total_arrival
    mean_sojourn = total_sojourn / n if n else 0.0
    total_sojourn_lower_bound = delta * lower_bound - total_arrival

    return DWSolution(
        lower_bound=lower_bound,
        upper_bound=upper_bound,
        gap=upper_bound - lower_bound,
        total_sojourn=total_sojourn,
        mean_sojourn=mean_sojourn,
        total_sojourn_lower_bound=total_sojourn_lower_bound,
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
    ``purge_every=None`` to disable. Never changes ``lower_bound``/
    ``upper_bound``; a purged column can always be regenerated later if it
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
    solution = extract_solution(
        integer_result.chosen,
        vehicles,
        station,
        delta,
        K,
        lower_bound=cg.best_lower_bound,
        upper_bound=integer_result.objective,
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
