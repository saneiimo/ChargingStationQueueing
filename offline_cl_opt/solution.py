"""
Extracts a solved ``ConnectorLaneModel`` into a plain, notebook/DataFrame-
friendly ``ConnectorLaneSolution``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from gurobipy import GRB

from .boundary import Cohort, cohort_totals
from .model import ConnectorLaneModel

_STATUS_NAMES = {
    getattr(GRB, name): name
    for name in ("OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL", "INTERRUPTED", "USER_OBJ_LIMIT")
    if hasattr(GRB, name)
}


@dataclass
class ConnectorLaneSolution:
    """
    Solved-instance summary: aggregate cost plus a per-vehicle timeline.

    ``total_sojourn``/``mean_sojourn`` cover exactly the vehicles the
    objective was summed over (``ConnectorLaneModel.objective_cohorts``),
    so they always match the objective they came from.

    ``by_cohort`` reports the same two figures for all three nested cohort
    levels regardless of what was optimised -- keys ``"measurement"``
    (arrived in the measured window), ``"measurement_queued"`` (plus those
    queued at the boundary) and ``"all"`` (plus those already plugged in),
    each mapping to ``{"n", "total_sojourn", "mean_sojourn"}``. These come
    from summing per-vehicle sojourns rather than from a partial objective,
    since ``delta*Z - sum_j a_j`` is only valid when both sums range over
    the same vehicle set.
    """

    status: str
    objective: float  # sum_j D_j over objective_cohorts, in slot units (eq. 1)
    total_sojourn: float  # minutes, over objective_cohorts only
    mean_sojourn: float  # total_sojourn / n_optimized, minutes
    mip_gap: float
    runtime: float
    n_vehicles: int  # every vehicle in the model, optimized or not
    n_optimized: int  # those inside objective_cohorts
    delta: float
    K: int
    by_cohort: dict[str, dict[str, float]]
    per_vehicle: pd.DataFrame  # vehicle_id, cohort, boundary_mode, in_objective,
    # arrival, served, pile, connector, start_slot, departure_slot, sojourn_min,
    # energy_kwh, energy_delivered_before_kwh, energy_required_kwh


def extract_solution(cl_model: ConnectorLaneModel) -> ConnectorLaneSolution:
    """
    Read variable values off a solved ``ConnectorLaneModel`` (call
    ``solve_cl_model`` first).

    ``start_slot`` / ``departure_slot`` are read straight off ``S[j]`` /
    ``D[j]`` -- dependent linear expressions, evaluated via ``getValue()``
    since they were never materialised as Gurobi variables (see
    ``ConnectorLaneModel``'s docstring). A vehicle with ``served=False``
    (Section 6.3: never plugged in within the horizon) has
    ``start_slot = departure_slot = K``; its nominal lane assignment from
    (10) is not reported (``pile``/``connector`` are ``None``), since the
    source document notes that assignment has no physical meaning for an
    unserved vehicle.

    ``objective`` and ``mip_gap`` need to account for ``cl_model.tie_break``:
    with two objectives set, plain ``model.ObjVal`` reports the *last*
    (lowest-priority, tie-break) phase, not the real ``sum_j D_j`` -- so when
    ``tie_break`` is set this reads it via ``ObjNVal`` at index 0 instead
    (mirrors ``offline_opt.solution.extract_solution``'s own handling
    exactly -- see that function's docstring). ``model.MIPGap`` goes further
    and is not retrievable at all once more than one objective is set
    (Gurobi raises ``AttributeError``) -- in that case ``mip_gap`` is
    reported as ``nan``. By the time the tie-break phase runs, the primary
    ``sum_j D_j`` is already fixed within ``abstol=1e-6`` of its optimum
    (see the comment in ``build_cl_model``).
    """
    m = cl_model.model
    if m.SolCount == 0:
        raise RuntimeError("No feasible solution to extract; call solve_cl_model first.")

    delta = cl_model.delta
    h = delta / 60.0
    K = cl_model.K
    rows: list[dict[str, object]] = []
    for j, v in cl_model.vehicles.items():
        k0 = cl_model.releases[j]
        S_val = cl_model.S[j].getValue()
        D_val = cl_model.D[j].getValue()
        served = any(cl_model.u[j, k].X > 0.5 for k in range(k0, K))

        pile = connector = None
        if served:
            for (mm, cc) in cl_model.lanes:
                if cl_model.y[j, mm, cc].X > 0.5:
                    pile, connector = mm, cc
                    break

        energy_kwh = h * sum(cl_model.p[j, k].X for k in range(k0, K))

        # A boundary vehicle's plan only covers the modeled window, so its
        # in-window energy is reported alongside what it already had at t=0
        # -- otherwise the row reads as a shortfall against energy_required.
        bv = cl_model.boundary_vehicles.get(j)
        cohort = cl_model.cohorts.get(j, Cohort.MEASUREMENT)
        rows.append(
            {
                "vehicle_id": j,
                "cohort": cohort.value,
                "boundary_mode": bv.mode.value if bv is not None else None,
                "in_objective": cohort in cl_model.objective_cohorts,
                "arrival": v.a,
                "served": served,
                "pile": pile,
                "connector": connector,
                "start_slot": S_val,
                "departure_slot": D_val,
                "sojourn_min": delta * D_val - v.a,
                "energy_kwh": energy_kwh,
                "energy_delivered_before_kwh": bv.initial_energy_kwh if bv is not None else 0.0,
                "energy_required_kwh": v.W,
            }
        )

    per_vehicle = pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)
    by_cohort = cohort_totals(rows, cl_model.cohorts)
    optimized = [r for r in rows if r["in_objective"]]
    n = len(optimized)

    if cl_model.tie_break:
        m.Params.ObjNumber = 0
        objective = float(m.ObjNVal)
        mip_gap = float("nan")
    else:
        objective = float(m.ObjVal)
        mip_gap = float(m.MIPGap) if m.IsMIP else 0.0

    # delta*Z - sum_j a_j is only valid when both sums range over the SAME
    # vehicles, so the arrival sum is restricted to the objective's own
    # cohorts -- mixing the two sets here would silently corrupt both the
    # total and the mean.
    total_arrival = sum(float(r["arrival"]) for r in optimized)  # type: ignore[arg-type]
    total_sojourn = delta * objective - total_arrival
    # nan when the objective covers no vehicle at all -- 0.0 would read as
    # a perfect mean sojourn rather than as an empty objective.
    mean_sojourn = total_sojourn / n if n else float("nan")

    return ConnectorLaneSolution(
        status=_STATUS_NAMES.get(m.Status, str(m.Status)),
        objective=objective,
        total_sojourn=total_sojourn,
        mean_sojourn=mean_sojourn,
        mip_gap=mip_gap,
        runtime=float(m.Runtime),
        n_vehicles=len(rows),
        n_optimized=n,
        delta=delta,
        K=K,
        by_cohort=by_cohort,
        per_vehicle=per_vehicle,
    )
