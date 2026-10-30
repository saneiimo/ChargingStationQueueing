"""
The result of a price-directed search, in the same shape as
``offline_cl_PB.PBSolution`` (same ``per_vehicle`` columns, same
``by_cohort`` levels, ``to_cl_model()`` for ``extract_solution`` and the
plots), plus the method's own diagnostics: the prices, the tables and every
search pass.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import pandas as pd

from offline_cl_opt.boundary import Cohort, cohort_totals
from offline_cl_opt.model import sojourn_minutes
from offline_cl_PB import PBPlan, schedule_to_cl_model
from offline_cl_PB.validation import CompactCheck

from .colgen import SlotPrices
from .instance import SearchInstance
from .search import PassResult


@dataclass
class PDSSolution:
    """
    ``status``: ``"OPTIMAL"`` (``objective`` is the proven optimum of the
    compact model), ``"WITHIN_TOLERANCE"`` (``objective - lower_bound <=
    gap_abs``), or why the search stopped early -- ``"TIME_LIMIT"``,
    ``"EXPANSION_LIMIT"``, ``"MEMORY_LIMIT"`` -- in which case
    ``[lower_bound, objective]`` is still a certified bracket.

    Units are the compact model's: total sojourn ``sum_j w_j delta*(D_j - k_j)``
    in minutes. ``lower_bound`` is a whole multiple of ``delta`` above
    ``-arrival_total`` (the search's slot objective is an integer).
    ``master_bound`` is the Lagrangian bound of the master problem (not
    rounded); ``root_lower_bound`` is the search's bound at the empty
    schedule, rounded up -- both before any search pass.
    """

    status: str
    objective: float
    lower_bound: float
    gap: float
    master_bound: float
    root_lower_bound: float
    total_sojourn: float
    mean_sojourn: float
    total_sojourn_LB: float
    mean_sojourn_LB: float
    n_vehicles: int
    n_optimized: int
    delta: float
    K: int
    runtime: float
    incumbent_source: str
    schedule: dict[int, PBPlan]
    connectors: dict[int, int]
    module_routing: dict[tuple[int, int, int], int]  # (pile, connector, slot) -> whole modules
    by_cohort: dict[str, dict[str, float]]
    per_vehicle: pd.DataFrame
    compact_check: CompactCheck | None
    prices: SlotPrices
    table_repairs: int  # cost-to-go steps corrected by the verification (expected 0)
    table_breakpoints: int
    timings: dict[str, float]  # seconds per stage: prices, tables, beam, search
    passes: list[dict] = field(default_factory=list)  # one row per search pass
    # The certified bracket over time: time_s, lower_bound, upper_bound (minutes), event.
    bound_history: list[dict] = field(default_factory=list)
    inputs: dict = field(default_factory=dict, repr=False)

    def to_cl_model(self):
        """The schedule as a *solved* ``offline_cl_opt`` ``ConnectorLaneModel`` (see ``offline_cl_PB.PBSolution``)."""
        return schedule_to_cl_model(self.schedule, self.connectors, **self.inputs)


def pass_row(res: PassResult, inst: SearchInstance) -> dict:
    """One row of ``PDSSolution.passes``, values in minutes as well as slots."""
    return {
        "kind": "beam" if res.beam_width else "exact",
        "beam_width": res.beam_width,
        "threshold": inst.to_minutes(res.threshold),
        "found": inst.to_minutes(res.value) if res.found else math.nan,
        "certified_lower_bound": inst.to_minutes(res.certified),
        "status": res.status_name,
        "expanded": res.expanded,
        "stored": res.stored,
        "peak_states": res.peak,
        "runtime_s": res.runtime,
    }


def per_vehicle_table(inst: SearchInstance, schedule: dict[int, PBPlan], connectors: dict[int, int]):
    """``(per_vehicle DataFrame, by_cohort, module_routing)`` with ``offline_cl_PB``'s columns."""
    h = inst.delta / 60.0
    rows: list[dict[str, object]] = []
    routing: dict[tuple[int, int, int], int] = {}
    for v in inst.vehicles:
        j = v.id
        plan = schedule[j]
        bv = inst.boundary_vehicles.get(j)
        for k in plan.occupied_slots():
            routing[plan.pile, connectors[j], k] = plan.q(k)
        rows.append(
            {
                "vehicle_id": j,
                "cohort": inst.cohorts.get(j, Cohort.MEASUREMENT).value,
                "boundary_mode": bv.mode.value if bv is not None else None,
                "in_objective": inst.weights[j] > 0,
                "arrival": v.a,
                "served": True,
                "pile": plan.pile,
                "connector": connectors.get(j),
                "start_slot": float(plan.start),
                "departure_slot": float(plan.departure),
                "sojourn_min": sojourn_minutes(plan.departure, v.a, inst.delta),
                "energy_kwh": h * sum(plan.power.values()),
                "energy_delivered_before_kwh": bv.initial_energy_kwh if bv is not None else 0.0,
                "energy_required_kwh": v.W,
            }
        )
    per_vehicle = pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)
    return per_vehicle, cohort_totals(rows, inst.cohorts), routing
