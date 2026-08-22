"""
Extracts a solved ``OfflineModel`` into a plain, notebook/DataFrame-friendly
``OfflineSolution``.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from gurobipy import GRB

from config import HR2MIN
from .model import OfflineModel

_STATUS_NAMES = {
    getattr(GRB, name): name
    for name in ("OPTIMAL", "TIME_LIMIT", "SUBOPTIMAL", "INTERRUPTED", "USER_OBJ_LIMIT")
    if hasattr(GRB, name)
}


@dataclass
class OfflineSolution:
    """Solved-instance summary: aggregate cost plus a per-vehicle timeline."""

    status: str
    objective: float  # sum_{j,k} sigma[j,k] (vehicle-slots already finished)
    total_sojourn: float  # sum_j (c_j - a_j), minutes
    mean_sojourn: float  # total_sojourn / n_vehicles, minutes
    mip_gap: float  # primary objective's gap; see extract_solution docstring re: tie_break
    runtime: float
    n_vehicles: int
    delta: float
    per_vehicle: pd.DataFrame  # vehicle_id, arrival, pile, service_start, departure, sojourn, energy_kwh


def extract_solution(offline_model: OfflineModel) -> OfflineSolution:
    """
    Read variable values off a solved ``OfflineModel`` (call
    ``solve_offline_model`` first).

    Under the 0-indexed slot convention (slot k starts at k*delta),
    ``service_start = delta * k_service`` directly (no -1 needed). Departure
    follows eq. (1), ``c_j = delta * sum_k (1 - sigma[j,k])`` over the full
    0..K-1 range: the k < k_j portion (sigma implicitly 0, never modeled as a
    variable) contributes ``delta * k_j`` on its own. ``total_sojourn`` is
    always derived this way, directly off ``alpha``/``sigma`` values -- never
    off ``model.ObjVal`` -- so it is unaffected by whether ``tie_break`` added
    a second objective.

    ``objective`` and ``mip_gap`` do need to account for ``tie_break``: with
    two objectives set, plain ``model.ObjVal`` reports the *last*
    (lowest-priority, tie-break) phase, not the real objective, so when
    ``offline_model.tie_break`` is set we read it via ``ObjNVal`` at index 0
    instead. ``model.MIPGap`` goes further and is not retrievable at all once
    more than one objective is set (Gurobi raises ``AttributeError``) -- in
    that case ``mip_gap`` is reported as ``nan``. This isn't a meaningful
    accuracy loss in practice: by the time the tie-break phase runs, the
    primary objective is already fixed within ``abstol=1e-6`` of its optimum
    (see the comment in ``build_offline_model``), and since it's a sum of
    binary variables (always integer), a gap that small already certifies
    the exact optimal value. Pass ``verbose=True`` to ``solve_offline_model``
    if you want to see Gurobi's own per-phase gap reporting in the log.
    """
    m = offline_model.model
    if m.SolCount == 0:
        raise RuntimeError(
            "No feasible solution to extract; call solve_offline_model first."
        )

    if offline_model.tie_break:
        m.Params.ObjNumber = 0
        objective_value = float(m.ObjNVal)
        mip_gap = float("nan")
    else:
        objective_value = float(m.ObjVal)
        mip_gap = float(m.MIPGap) if m.IsMIP else 0.0

    delta = offline_model.delta
    rows: list[dict[str, object]] = []
    for j, v in offline_model.vehicles.items():
        k_j = offline_model.releases[j]
        K = offline_model.K

        alpha_on = [k for k in range(k_j, K) if offline_model.alpha[j, k].X > 0.5]
        service_start = delta * min(alpha_on) if alpha_on else float("nan")

        tail_unfinished = sum(
            1.0 - offline_model.sigma[j, k].X for k in range(k_j, K)
        )
        departure = delta * (k_j + tail_unfinished)
        sojourn = departure - v.a

        energy_delivered = delta * sum(offline_model.p[j, k].X for k in range(k_j, K))

        pile = None
        for mm in range(offline_model.station.n_piles):
            if offline_model.y[j, mm].X > 0.5:
                pile = mm
                break

        rows.append(
            {
                "vehicle_id": j,
                "arrival": v.a,
                "pile": pile,
                "service_start": service_start,
                "departure": departure,
                "sojourn": sojourn,
                "energy_kwh": energy_delivered / HR2MIN,
                "energy_required_kwh": v.W_kwh,
            }
        )

    per_vehicle = pd.DataFrame(rows).sort_values("vehicle_id").reset_index(drop=True)
    total_sojourn = float(per_vehicle["sojourn"].sum())
    n = len(per_vehicle)

    return OfflineSolution(
        status=_STATUS_NAMES.get(m.Status, str(m.Status)),
        objective=objective_value,
        total_sojourn=total_sojourn,
        mean_sojourn=total_sojourn / n if n else 0.0,
        mip_gap=mip_gap,
        runtime=float(m.Runtime),
        n_vehicles=n,
        delta=delta,
        per_vehicle=per_vehicle,
    )
