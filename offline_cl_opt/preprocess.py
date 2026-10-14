"""
Preprocessing -- Section 9 of ``connector_lane_model.html``.

  - **E_j** (9.1, eq. 22): the earliest departure boundary vehicle j could
    possibly achieve, alone on its own connector with the whole module pool
    to itself, charging at its own acceptance limit throughout. A pure
    per-vehicle calculation via the model's own discrete greedy recursion
    -- no MILP solve needed. Lives in ``model.py`` (``earliest_departures``,
    re-exported here). It is not a constraint of the model, only a cheap
    per-vehicle floor for diagnostics. It must use the *discrete*
    recursion, not a continuous closed form, or it need not be a valid
    floor for the discrete model.
  - **UB** (9.2): the objective value (total sojourn
    ``sum_j delta*(D_j - k_j)``, minutes) of *any* known feasible schedule.
    The document suggests two sources, and both are supported here via
    ``incumbent_departures``:
      - a reference schedule you already have -- e.g. a finished causal
        (FCFS or otherwise) simulation -- pass its per-vehicle departure
        times (minutes) directly;
      - Section 8.4's conservative shortcut, used automatically if
        ``incumbent_departures`` is not supplied.

Section 9.2 is explicit that this ``UB`` is *not* used to narrow which
variables ``build_cl_model`` creates (an earlier revision of both the
source document and this module did exactly that, trimming each vehicle's
slot range on the right as well as the left): the slack any incumbent
leaves above the sum of individual best cases is shared across every
vehicle, so it only narrows anything when that slack is smaller than the
horizon itself, which the document argues does not happen in the congested
regime this model targets. ``UB`` is worth computing anyway, for two
different, cheaper uses (Section 9.2/11):

  - pass it as ``solve_cl_model(..., cutoff=UB)`` (or
    ``solve_cl_model_adaptive(..., cutoff=UB)``), an objective cutoff --
    see ``solve_cl_model``'s own docstring;
  - use the underlying schedule itself as a MIP start (already how
    ``adaptive.conservative_feasible_solution``'s own result, or a real
    simulation via ``adaptive.solve_cl_model_adaptive(...,
    warm_start_evs=...)``, are used).
"""

from __future__ import annotations

import math

from .adaptive import conservative_feasible_solution
from .instance import StationSpec, VehicleData
from .model import earliest_departures  # noqa: F401  (re-exported for callers of this module)
from .model import sojourn_minutes
from .solution import extract_solution


def _departures_from_conservative(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    mip_gap: float | None,
    time_limit: float | None,
    threads: int | None,
    verbose: bool,
) -> dict[int, float]:
    """
    Fallback UB source when no ``incumbent_departures`` is supplied:
    solve Section 8.4's conservative shortcut and read departure times
    (minutes) off its solution. In the closed model every vehicle is
    served and departs, so every vehicle is included.
    """
    cons = conservative_feasible_solution(
        vehicles,
        station,
        delta,
        horizon_minutes,
        mip_gap=mip_gap,
        time_limit=time_limit,
        threads=threads,
        verbose=verbose,
    )
    sol = extract_solution(cons)
    return {
        int(row.vehicle_id): float(row.departure_slot) * delta
        for row in sol.per_vehicle.itertuples()
    }


def incumbent_objective(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    incumbent_departures: dict[int, float] | None = None,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
    verbose: bool = False,
) -> float:
    """
    UB (Section 9.2): total sojourn ``sum_j delta*(D_j - k_j)`` in
    minutes, from a supplied reference schedule or, if
    ``incumbent_departures`` is ``None``, Section 8.4's conservative
    shortcut. Directly usable as ``solve_cl_model(..., cutoff=...)`` /
    ``solve_cl_model_adaptive(..., cutoff=...)``, since both minimize the
    same total-sojourn objective.

    Summed over every vehicle in ``vehicles``. When the model's
    ``objective_cohorts`` covers only some of them the result is still a
    valid (looser) cutoff: every extra term is a sojourn, hence ``>= 0``.

    Parameters
    ----------
    incumbent_departures :
        Optional ``{vehicle_id: departure_time_minutes}`` from any known
        feasible schedule -- e.g. build it from a finished simulation with
        ``{ev.id: ev.departure_time for ev in env.engine.metrics.finished_evs}``.
        It must cover every vehicle in ``vehicles`` (a closed-system
        schedule serves and completes everyone; a missing vehicle raises).
        Departure times are converted to slot boundaries by rounding *up*
        (``ceil``): a real schedule that finishes at continuous time ``t``
        can always be read as "done" by the next slot boundary at or after
        ``t``, so this is a safe (if occasionally one slot conservative)
        upper bound -- important since an under-count would make it an
        invalid cutoff (see ``solve_cl_model``'s own docstring on what
        happens if ``cutoff`` was never actually achievable).
        If ``None`` (default), ``_departures_from_conservative`` supplies
        this by solving the conservative shortcut itself.
    mip_gap, time_limit, threads, verbose :
        Only used when falling back to the conservative shortcut (i.e.
        ``incumbent_departures is None``); ignored otherwise.
    """
    K = math.ceil(round(horizon_minutes / delta, 9))
    if incumbent_departures is None:
        incumbent_departures = _departures_from_conservative(
            vehicles,
            station,
            delta,
            horizon_minutes,
            mip_gap=mip_gap,
            time_limit=time_limit,
            threads=threads,
            verbose=verbose,
        )

    missing = sorted(v.id for v in vehicles if v.id not in incumbent_departures)
    if missing:
        raise ValueError(
            f"incumbent_departures has no departure for vehicles {missing[:10]}: a cutoff must "
            "come from a schedule that serves and completes every vehicle."
        )
    total = 0.0
    for v in vehicles:
        D = math.ceil(round(incumbent_departures[v.id] / delta, 9))
        if D > K:
            raise ValueError(
                f"vehicle {v.id} departs at {incumbent_departures[v.id]:g} min, after the "
                f"horizon ({K} slots of {delta:g} min): not a schedule of this model."
            )
        total += sojourn_minutes(D, v.a, delta)
    return total
