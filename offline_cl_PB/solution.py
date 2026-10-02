"""
The result of an exact branch-and-price solve, in the same shape as
``offline_cl_opt.solution.ConnectorLaneSolution`` / ``offline_cl_dw``'s
``DWSolution`` (same ``per_vehicle`` columns, same ``by_cohort`` levels via
``offline_cl_opt.boundary.cohort_totals``), plus the bracket and the
search diagnostics.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import pandas as pd

from offline_cl_opt.boundary import Cohort, cohort_totals
from offline_cl_opt.model import sojourn_minutes

from .columns import PBPlan
from .validation import CompactCheck, compact_model_check, schedule_to_cl_model

if TYPE_CHECKING:
    from .bp import BPStats, BranchAndPrice, NodeRecord


@dataclass
class PBSolution:
    """
    ``status``: ``"OPTIMAL"`` (the tree was exhausted: ``objective`` is the
    proven optimum of the whole-module compact model), ``"TIME_LIMIT"`` or
    ``"NODE_LIMIT"`` (``[lower_bound, objective]`` is still a valid bracket).

    ``objective`` / ``lower_bound`` / ``gap`` are in the compact model's own
    units: total sojourn ``sum_j (delta*D_j - a_j)`` over
    ``objective_cohorts``, in minutes. ``total_sojourn`` / ``total_sojourn_LB``
    repeat ``objective`` / ``lower_bound``, and ``mean_sojourn`` /
    ``mean_sojourn_LB`` divide them by ``n_optimized``.

    ``compact_check`` evaluates every row of ``offline_cl_opt``'s compact
    model at the returned schedule (``None`` if skipped); its
    ``max_violation`` should be ~0 and its ``objective`` equal to
    ``objective``.
    """

    status: str
    objective: float
    lower_bound: float
    gap: float
    root_lower_bound: float
    root_lp_value: float
    total_sojourn: float
    mean_sojourn: float
    total_sojourn_LB: float
    mean_sojourn_LB: float
    n_vehicles: int
    n_optimized: int
    delta: float
    K: int
    nodes_processed: int
    columns: int
    runtime: float
    incumbent_source: str
    schedule: dict[int, PBPlan]
    connectors: dict[int, int]
    module_routing: dict[tuple[int, int, int], int]  # (pile, connector, slot) -> whole modules
    by_cohort: dict[str, dict[str, float]]
    per_vehicle: pd.DataFrame
    compact_check: CompactCheck | None
    stats: "BPStats"
    nodes: list["NodeRecord"] = field(default_factory=list)
    # The certified bracket over time: one dict per change, keys ``time_s``
    # (since the solver started), ``lower_bound`` (proven, non-decreasing,
    # minutes of total sojourn like ``objective``;
    # -inf until the root is first priced), ``upper_bound`` (incumbent,
    # non-increasing), ``nodes`` (tree nodes solved so far) and ``event``
    # ("incumbent: <source>", "node", "tree", "end"). The last row matches
    # ``lower_bound`` / ``objective``.
    bound_history: list[dict] = field(default_factory=list)
    # The instance this was solved for (vehicles, station, delta,
    # horizon_minutes, boundary_vehicles, cohorts, objective_cohorts).
    inputs: dict = field(default_factory=dict, repr=False)

    def to_cl_model(self):
        """
        The returned schedule as a *solved* ``offline_cl_opt``
        ``ConnectorLaneModel``: pass it anywhere you would pass a compact model
        you solved yourself (``extract_solution``, ``plot_vehicle_power_and_modules``,
        ``plot_pile_power_and_modules``, ...). See
        ``validation.schedule_to_cl_model``.
        """
        return schedule_to_cl_model(self.schedule, self.connectors, **self.inputs)


def build_solution(bp: "BranchAndPrice", status: str, *, run_compact_check: bool = True) -> PBSolution:
    if bp.incumbent is None:
        raise RuntimeError("branch-and-price finished without any feasible schedule (should be impossible)")
    delta, K, h = bp.delta, bp.K, bp.delta / 60.0
    schedule = bp.incumbent
    connectors = bp.incumbent_connectors
    UB = float(bp.upper_bound)
    if status == "OPTIMAL":
        LB = UB
    else:
        open_bounds = [b for b in bp.open_bounds_at_stop if math.isfinite(b)]
        LB = min(open_bounds) if open_bounds else -math.inf
        if len(open_bounds) < len(bp.open_bounds_at_stop):
            LB = -math.inf  # an open node was never bounded
        # Any global bound proven earlier is still valid; keep the best one.
        LB = min(max(LB, bp._global_lb), UB)
    bp._note_bounds("end", LB, force=True)

    rows: list[dict[str, object]] = []
    routing: dict[tuple[int, int, int], int] = {}
    for v in bp.vehicles:
        j = v.id
        plan = schedule[j]
        bv = bp.boundary.get(j)
        cohort = bp.cohorts.get(j, Cohort.MEASUREMENT)
        served = not plan.is_null
        if served:
            for k in plan.occupied_slots():
                routing[plan.pile, connectors[j], k] = plan.q(k)  # type: ignore[index]
        rows.append(
            {
                "vehicle_id": j,
                "cohort": cohort.value,
                "boundary_mode": bv.mode.value if bv is not None else None,
                "in_objective": bp.weights[j] > 0,
                "arrival": v.a,
                "served": served,
                "pile": plan.pile if served else None,
                "connector": connectors.get(j) if served else None,
                "start_slot": float(plan.start_slot(K)),
                "departure_slot": float(plan.departure),
                "sojourn_min": sojourn_minutes(plan.departure, v.a, delta),
                "energy_kwh": h * sum(plan.power.values()),
                "energy_delivered_before_kwh": bv.initial_energy_kwh if bv is not None else 0.0,
                "energy_required_kwh": v.W,
            }
        )
    per_vehicle = pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)
    by_cohort = cohort_totals(rows, bp.cohorts)
    optimized = [r for r in rows if r["in_objective"]]
    n = len(optimized)
    total_sojourn = UB  # the objective is total sojourn over these vehicles
    total_sojourn_LB = LB

    check = None
    if run_compact_check:
        t = time.perf_counter()
        check = compact_model_check(
            schedule,
            connectors,
            bp.vehicles,
            bp.station,
            delta,
            bp.horizon_minutes,
            boundary_vehicles=bp.boundary,
            cohorts=bp.cohorts,
            objective_cohorts=bp.objective_cohorts,
        )
        bp.stats.time_validation += time.perf_counter() - t

    return PBSolution(
        status=status,
        objective=UB,
        lower_bound=LB,
        gap=UB - LB,
        root_lower_bound=float(bp.root_lower_bound),
        root_lp_value=float(bp.root_lp_value),
        total_sojourn=total_sojourn,
        mean_sojourn=total_sojourn / n if n else math.nan,
        total_sojourn_LB=total_sojourn_LB,
        mean_sojourn_LB=total_sojourn_LB / n if n else math.nan,
        n_vehicles=len(rows),
        n_optimized=n,
        delta=delta,
        K=K,
        nodes_processed=bp.stats.nodes_processed,
        columns=len(bp.master.columns),
        runtime=time.perf_counter() - bp._t0,
        incumbent_source=bp.incumbent_source,
        schedule=dict(schedule),
        connectors=dict(connectors),
        module_routing=routing,
        by_cohort=by_cohort,
        per_vehicle=per_vehicle,
        compact_check=check,
        stats=bp.stats,
        nodes=list(bp.records),
        bound_history=list(bp.bound_history),
        inputs=dict(
            vehicles=bp.vehicles,
            station=bp.station,
            delta=delta,
            horizon_minutes=bp.horizon_minutes,
            boundary_vehicles=bp.boundary,
            cohorts=bp.cohorts,
            objective_cohorts=bp.objective_cohorts,
        ),
    )
