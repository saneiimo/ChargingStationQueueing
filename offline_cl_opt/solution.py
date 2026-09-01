"""
Extracts a solved ``ConnectorLaneModel`` into a plain, notebook/DataFrame-
friendly ``ConnectorLaneSolution``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from gurobipy import GRB

from .model import ConnectorLaneModel

_STATUS_NAMES = {
    getattr(GRB, name): name
    for name in ("OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL", "INTERRUPTED", "USER_OBJ_LIMIT")
    if hasattr(GRB, name)
}


@dataclass
class ConnectorLaneSolution:
    """Solved-instance summary: aggregate cost plus a per-vehicle timeline."""

    status: str
    objective: float  # sum_j D_j, in slot units (eq. 1)
    total_sojourn: float  # minutes; delta*objective - sum_j a_j (matches per_vehicle["sojourn_min"].sum())
    mean_sojourn: float  # total_sojourn / n_vehicles, minutes
    mip_gap: float
    runtime: float
    n_vehicles: int
    delta: float
    K: int
    per_vehicle: pd.DataFrame  # vehicle_id, arrival, served, pile, connector,
    # start_slot, departure_slot, sojourn_min, energy_kwh, energy_required_kwh


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

        rows.append(
            {
                "vehicle_id": j,
                "arrival": v.a,
                "served": served,
                "pile": pile,
                "connector": connector,
                "start_slot": S_val,
                "departure_slot": D_val,
                "sojourn_min": delta * D_val - v.a,
                "energy_kwh": energy_kwh,
                "energy_required_kwh": v.W,
            }
        )

    per_vehicle = pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)
    n = len(per_vehicle)

    if cl_model.tie_break:
        m.Params.ObjNumber = 0
        objective = float(m.ObjNVal)
        mip_gap = float("nan")
    else:
        objective = float(m.ObjVal)
        mip_gap = float(m.MIPGap) if m.IsMIP else 0.0

    total_arrival = sum(v.a for v in cl_model.vehicles.values())
    total_sojourn = delta * objective - total_arrival
    mean_sojourn = total_sojourn / n if n else 0.0

    return ConnectorLaneSolution(
        status=_STATUS_NAMES.get(m.Status, str(m.Status)),
        objective=objective,
        total_sojourn=total_sojourn,
        mean_sojourn=mean_sojourn,
        mip_gap=mip_gap,
        runtime=float(m.Runtime),
        n_vehicles=n,
        delta=delta,
        K=K,
        per_vehicle=per_vehicle,
    )
