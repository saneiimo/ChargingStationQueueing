"""
Preprocessing -- Section 9 of ``connector_lane_model.html``.

  - **E_j** (9.1, eq. 22): the earliest departure boundary vehicle j could
    possibly achieve, alone on its own connector with the whole module pool
    to itself, charging at its own acceptance limit throughout. A pure
    per-vehicle calculation via the model's own discrete greedy recursion
    -- no MILP solve needed. Lives in ``model.py`` (``earliest_departures``,
    re-exported here) rather than being duplicated in this module, since
    ``build_cl_model`` itself also needs ``E_j`` internally for the optional
    departure lower bound (24, Section 9.3) -- see ``model.py``'s module
    docstring. The document is explicit that this must use the *discrete*
    recursion, not a continuous closed form, or the resulting bound need
    not be valid for the discrete model being preprocessed.
  - **UB** (9.2): the objective value (``sum_j D_j``, slot units) of *any*
    known feasible schedule. The document suggests two sources, and both
    are supported here via ``incumbent_departures``:
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
    (minutes) off its solution. Only served vehicles are included --
    ``incumbent_departure_total`` treats anything absent the same way (9)
    does for a vehicle that never departs: contributes ``K``.
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
        if row.served
    }


def incumbent_departure_total(
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
    UB (Section 9.2): ``sum_j D_j`` in slot units, from a supplied
    reference schedule or, if ``incumbent_departures`` is ``None``, Section
    8.4's conservative shortcut. Directly usable as
    ``solve_cl_model(..., cutoff=...)`` / ``solve_cl_model_adaptive(...,
    cutoff=...)``, since both are already in the same slot-unit objective.

    Parameters
    ----------
    incumbent_departures :
        Optional ``{vehicle_id: departure_time_minutes}`` from any known
        feasible schedule -- e.g. build it from a finished simulation with
        ``{ev.id: ev.departure_time for ev in env.engine.metrics.finished_evs}``.
        A vehicle missing from the dict is treated as never served/never
        finished, contributing ``K`` -- the same convention (9) uses.
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

    total = 0.0
    for v in vehicles:
        dep = incumbent_departures.get(v.id)
        if dep is None:
            total += K
        else:
            total += min(K, math.ceil(round(dep / delta, 9)))
    return total
