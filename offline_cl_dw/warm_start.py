"""
Cross-resolution warm starting: solve at a coarser slot length (bigger
``delta``, fewer slots -- both the master's ``2MK`` rows and every pricer
MILP's ``K-k_j`` binaries shrink with it), then reuse that solution at the
real, finer ``delta`` -- either as extra seed columns for another
Dantzig-Wolfe run (``solve_by_decomposition``) or as a Gurobi MIP start for
the exact compact IP (``offline_cl_opt.model.build_cl_model``).

Why this is sound, not just a heuristic
----------------------------------------
``refine_plan`` maps one vehicle's coarse-grid plan onto the fine grid by
"staircase" replication: coarse slot ``k`` becomes the ``r`` fine slots
``[k*r, (k+1)*r)``, all carrying the *same* power (kW is a rate, so it needs
no rescaling). Provided ``delta_coarse`` is an exact integer multiple
(``r``) of ``delta_fine``, this is an *exact* refinement -- a plan that was
feasible on the coarse grid is automatically feasible on the fine grid too,
not merely "probably close":

  - Energy: ``h_coarse*p`` (one coarse step) exactly equals the sum of ``r``
    fine steps ``h_fine*p`` (since ``h_coarse = r*h_fine``), so the
    cumulative energy trajectory -- and completion at departure,
    ``x_D = W_j`` -- transfers exactly, not approximately.
  - Taper: ``tau^delta_j(h)`` (``instance.py``'s ``tau_delta_hours``) is
    increasing in ``h`` (``-> tau_j`` as ``h -> 0``, the document's own
    "always ``>= tau_j``, ``>= h``"), so a power profile that already
    satisfies the coarser, *more restrictive* cap ``tau^delta_coarse*p + x
    <= R_j`` automatically satisfies the finer, less restrictive one at the
    same power.
  - Capacity: every fine sub-slot within a coarse slot has exactly the same
    occupant set and power values as its parent coarse slot, so per-(pile,
    slot) connector/module capacity feasibility (25)/(26) transfers
    unchanged.
  - Release time: every coarse slot boundary is also a fine slot boundary
    (``delta_coarse = r*delta_fine``), so ``k0_coarse*delta_coarse >=
    k0_fine*delta_fine`` always -- refining a plan can only ever move its
    start slot to something at or after the vehicle's own fine-grid release
    slot, never before it.

So this is a genuine warm start, not an approximation that needs re-solving
slot-by-slot -- though ``postprocess.validate_schedule`` should still be run
on whatever the fine solve eventually returns, as always (a downstream bug,
not this refinement, would be what it catches).

One caveat, inherited directly from the coarse solve itself: the DW
master's module-capacity row (26) is the *continuous*-module relaxation
(Proposition 2), so a coarse DW solution is not automatically whole-module
feasible even before refinement -- check ``postprocess.whole_module_
failures`` on the coarse ``chosen`` first (same caveat the coarse solve
already has on its own, refinement doesn't add a new one). ``apply_mip_
start``'s own ``r`` values (whole modules) are a best-effort ``ceil(p/
Delta)`` per connector, same as ``postprocess.rounded_module_routing`` --
Gurobi's own MIP-start repair silently discards/fixes any variable its
given values don't jointly satisfy, so a rare local mismatch here costs a
bit of the start's quality, not correctness.
"""

from __future__ import annotations

import math

from offline_cl_opt.instance import StationSpec
from offline_cl_opt.model import ConnectorLaneModel

from .columns import Plan, null_plan
from .postprocess import assign_connectors


def refinement_ratio(delta_coarse: float, delta_fine: float) -> int:
    """
    ``r = delta_coarse / delta_fine``, validated to be a positive integer
    (to floating-point tolerance) -- required for ``refine_plan`` to be an
    *exact* refinement rather than an approximate one, since every coarse
    slot boundary must land exactly on a fine slot boundary.
    """
    if delta_coarse <= 0 or delta_fine <= 0:
        raise ValueError("delta_coarse and delta_fine must both be positive")
    if delta_fine > delta_coarse:
        raise ValueError(
            f"delta_fine ({delta_fine}) must be <= delta_coarse ({delta_coarse}) -- "
            "refinement only ever goes from a coarser grid to a finer one."
        )
    ratio = delta_coarse / delta_fine
    r = round(ratio)
    if r < 1 or abs(ratio - r) > 1e-6:
        raise ValueError(
            f"delta_coarse ({delta_coarse}) must be an integer multiple of delta_fine "
            f"({delta_fine}) for the refinement to be exact -- got ratio={ratio:g}."
        )
    return r


def refine_plan(plan: Plan, r: int, K_fine: int) -> Plan:
    """One vehicle's coarse-grid plan, replicated onto the fine grid at
    ratio ``r`` -- see this module's own docstring for why this is exact,
    not approximate. The null plan simply becomes the fine-grid null plan."""
    if plan.is_null:
        return null_plan(plan.vehicle_id, K_fine)
    start = plan.start * r  # type: ignore[operator]
    departure = plan.departure * r
    power = {
        fine_k: p
        for coarse_k, p in plan.power.items()
        for fine_k in range(coarse_k * r, (coarse_k + 1) * r)
    }
    return Plan(vehicle_id=plan.vehicle_id, pile=plan.pile, start=start, departure=departure, power=power)


def refine_solution(
    chosen: dict[int, Plan], delta_coarse: float, delta_fine: float, K_fine: int
) -> dict[int, Plan]:
    """
    Apply ``refine_plan`` to every vehicle in a coarse-grid solution --
    typically ``solve_integer(coarse_colgen.master).chosen`` from a
    coarse-delta ``run_column_generation``/``solve_by_decomposition`` call,
    but any ``{vehicle_id: Plan}`` built at ``delta_coarse`` works.

    The result is directly usable two ways:

      - as ``extra_seed_columns`` for ``run_column_generation``/
        ``solve_by_decomposition`` at ``delta_fine`` (seeding the
        decomposition with an already-good, mutually feasible schedule);
      - via ``apply_mip_start`` below, as a Gurobi MIP start for the exact
        compact IP (``offline_cl_opt.model.build_cl_model``) at
        ``delta_fine``.
    """
    r = refinement_ratio(delta_coarse, delta_fine)
    return {j: refine_plan(plan, r, K_fine) for j, plan in chosen.items()}


def apply_mip_start(
    cl_model: ConnectorLaneModel,
    chosen: dict[int, Plan],
    station: StationSpec,
) -> None:
    """
    Set Gurobi ``.Start`` values on ``cl_model`` -- an *unsolved*
    ``offline_cl_opt.model.ConnectorLaneModel``, built at ``chosen``'s own
    (fine) delta, e.g. via ``build_cl_model(vehicles, station, delta_fine,
    horizon_minutes)`` -- from ``chosen``, a ``{vehicle_id: Plan}`` already
    at that fine delta (run coarse-grid results through ``refine_solution``
    first).

    Connector labels are reconstructed per pile via
    ``postprocess.assign_connectors`` (Proposition 1's interval-graph
    left-edge colouring), since a ``Plan`` only names a pile, not a
    connector, and the compact IP's ``y``/``b`` variables need both.

    Sets ``u``, ``p``, ``eta``, ``x``, ``y``, ``r`` and ``b`` -- everything
    the compact model declares. This is deliberately not a hard fix: Gurobi
    treats ``.Start`` as a candidate solution to validate and, where it
    isn't already feasible (most likely ``r``, see this module's own
    docstring), repair -- so it is fine, and expected, for the eventual
    solve to adjust some of these values rather than take every one as
    given.
    """
    connectors = assign_connectors(chosen, station)
    h = cl_model.delta / 60.0
    Delta = station.p_module

    for j, plan in chosen.items():
        k0 = cl_model.releases[j]
        served = not plan.is_null
        cum = 0.0
        for k in range(k0, cl_model.K):
            occupied = served and plan.start <= k < plan.departure  # type: ignore[operator]
            p_val = plan.power.get(k, 0.0) if occupied else 0.0
            cl_model.u[j, k].Start = 1.0 if occupied else 0.0
            cl_model.p[j, k].Start = p_val
            cl_model.eta[j, k].Start = 1.0 if (served and k == plan.start) else 0.0
            cum += h * p_val
            if (j, k + 1) in cl_model.x:
                cl_model.x[j, k + 1].Start = cum

        conn = connectors.get(j)
        for (mm, cc) in cl_model.lanes:
            cl_model.y[j, mm, cc].Start = 1.0 if (served and plan.pile == mm and conn == cc) else 0.0

    # r[mm,cc,k]: whole modules routed -- best-effort ceil(p/Delta) per
    # occupied connector, same convention as postprocess.rounded_module_routing.
    occ_by_lane_slot: dict[tuple[int, int, int], float] = {}
    for j, plan in chosen.items():
        if plan.is_null:
            continue
        conn = connectors[j]
        for k, p_val in plan.power.items():
            if p_val > 1e-9:
                occ_by_lane_slot[plan.pile, conn, k] = p_val  # type: ignore[index]
    for mm in range(station.n_piles):
        for cc in range(station.n_connectors):
            for k in range(cl_model.K):
                p_val = occ_by_lane_slot.get((mm, cc, k), 0.0)
                cl_model.r[mm, cc, k].Start = math.ceil(p_val / Delta - 1e-9) if p_val > 1e-9 else 0.0

    # b[i,j]: only meaningful for a pair sharing one (pile, connector) --
    # their intervals never overlap there by construction of
    # assign_connectors, so "who started first" is a well-defined order.
    for (i, jv) in cl_model.b:
        pi_, pj = chosen.get(i), chosen.get(jv)
        if pi_ is None or pj is None or pi_.is_null or pj.is_null:
            continue
        if pi_.pile != pj.pile or connectors.get(i) != connectors.get(jv):
            continue
        cl_model.b[i, jv].Start = 1.0 if pi_.start <= pj.start else 0.0  # type: ignore[operator]
