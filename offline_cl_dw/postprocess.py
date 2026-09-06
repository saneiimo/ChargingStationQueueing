"""
Turning an integral master solution into a physical schedule -- Section 10
of ``dantzig_wolfe_decomposition.html``.

Two things are missing from a price-and-branch solution (``master.IntegerResult``):
connector labels within each pile (Proposition 1 guarantees they exist but
the master never names them, since row (25) only counts vehicles per pile)
and whole-module feasibility (row (26) only enforces the continuous-module
condition, Proposition 2 -- Section 8.4-style rounding may need repair).
``validate_schedule`` then re-checks everything the master and pricers were
supposed to guarantee, directly off the assigned schedule.
"""

from __future__ import annotations

import heapq
import math

from offline_cl_opt.boundary import BoundaryMode, BoundaryVehicle
from offline_cl_opt.instance import StationSpec, VehicleData

from .columns import Plan


def assign_connectors(chosen: dict[int, Plan], station: StationSpec) -> dict[int, int]:
    """
    Section 10.1: the greedy left-edge sweep, exact for interval graphs.
    Returns ``{vehicle_id: connector}`` for every *served* vehicle in
    ``chosen`` (null plans are simply absent from the result).

    Raises ``AssertionError`` if more than ``C`` vehicles are ever
    simultaneously active on one pile -- this cannot happen if row (25)
    genuinely held in the master solution, so firing here means the
    integer master solution was not actually valid, not that this
    function's own logic is at fault (see the source document's own note
    on this exact assertion).
    """
    by_pile: dict[int, list[tuple[int, int, int]]] = {}  # pile -> [(start, departure, vehicle_id)]
    for j, plan in chosen.items():
        if plan.is_null:
            continue
        by_pile.setdefault(plan.pile, []).append((plan.start, plan.departure, j))  # type: ignore[arg-type]

    connectors: dict[int, int] = {}
    for pile, entries in by_pile.items():
        entries.sort(key=lambda e: e[0])
        free: list[int] = list(range(station.n_connectors))
        heapq.heapify(free)
        active: list[tuple[int, int]] = []  # min-heap keyed by departure: (departure, connector)
        for start, departure, j in entries:
            while active and active[0][0] <= start:
                _dep, conn = heapq.heappop(active)
                heapq.heappush(free, conn)
            assert free, (
                f"Pile {pile}: more than {station.n_connectors} vehicles simultaneously active "
                f"at slot {start} -- the integer master solution violated row (25) for this "
                "pile-slot, which should be impossible if it solved correctly."
            )
            conn = heapq.heappop(free)
            connectors[j] = conn
            heapq.heappush(active, (departure, conn))
    return connectors


def whole_module_failures(
    chosen: dict[int, Plan], station: StationSpec
) -> list[tuple[int, int]]:
    """
    (37): pile-slots where ``sum_c ceil(p/Delta) > N`` -- the stricter,
    whole-module condition the continuous master (Proposition 2) doesn't
    enforce. Same test as ``offline_cl_opt.adaptive.rounding_test_failures``,
    applied here to the DW solution's own per-pile power sums.
    """
    Delta = station.p_module
    power_by_pile_slot: dict[tuple[int, int], list[float]] = {}
    for plan in chosen.values():
        if plan.is_null:
            continue
        for k, p_val in plan.power.items():
            if p_val > 1e-9:
                power_by_pile_slot.setdefault((plan.pile, k), []).append(p_val)  # type: ignore[arg-type]

    failures: list[tuple[int, int]] = []
    for (pile, k), powers in power_by_pile_slot.items():
        total = sum(math.ceil(p / Delta - 1e-9) for p in powers)
        if total > station.n_modules:
            failures.append((pile, k))
    return failures


def rounded_module_routing(
    chosen: dict[int, Plan], station: StationSpec
) -> dict[tuple[int, int, int], int]:
    """
    Section 10.2's repair: ``r_mck = ceil(p/Delta)`` for the connector's
    occupant, given the connector labels from ``assign_connectors``. Only
    a genuine whole-module-feasible routing when ``whole_module_failures``
    is empty -- exactly the Lemma from the compact-model document (Section
    8.1 there), reused here at the pile-slot level.
    """
    connectors = assign_connectors(chosen, station)
    Delta = station.p_module
    routing: dict[tuple[int, int, int], int] = {}
    for j, plan in chosen.items():
        if plan.is_null:
            continue
        cc = connectors[j]
        for k, p_val in plan.power.items():
            if p_val > 1e-9:
                routing[plan.pile, cc, k] = math.ceil(p_val / Delta - 1e-9)  # type: ignore[index]
    return routing


def validate_schedule(
    chosen: dict[int, Plan],
    vehicles: dict[int, VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
    *,
    best_lower_bound: float | None = None,
    boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
) -> None:
    """
    Section 10.3's checklist, run as assertions. Raises ``AssertionError``
    with a specific message on the first violation found; call this before
    trusting a price-and-branch result the way you would for any solver
    output, not only while debugging this package.

    ``boundary_vehicles``: the same mapping passed to
    ``colgen.run_column_generation`` (see ``offline_cl_opt.boundary``).
    Their energy-completion check (below) is relaxed to account for energy
    already delivered before t=0, which ``plan.power`` never reflects: a
    FIXED-mode vehicle's trajectory is exogenous and skipped entirely (it
    is never re-verified against an energy target -- see
    ``offline_cl_opt.boundary._discretize_trace_power``'s own docstring),
    an OPTIMIZE-mode one's check adds back ``initial_energy_kwh`` before
    comparing to ``v.W``.
    """
    boundary_vehicles = boundary_vehicles or {}
    h = delta / 60.0
    objective = 0.0
    occupancy: dict[tuple[int, int], list[int]] = {}  # (pile, k) -> [vehicle_id, ...]
    power_by_pile_slot: dict[tuple[int, int], float] = {}

    for j, plan in chosen.items():
        v = vehicles[j]
        objective += plan.departure
        if plan.is_null:
            continue

        assert plan.start is not None and plan.start >= _release_slot(v.a, delta), (
            f"Vehicle {j} starts at slot {plan.start}, before its own release slot -- "
            "constraint (2)."
        )
        assert plan.start < plan.departure <= K, f"Vehicle {j} has a degenerate interval."

        bv = boundary_vehicles.get(j)
        energy = h * sum(plan.power.values())
        if plan.departure < K and (bv is None or bv.mode is BoundaryMode.OPTIMIZE):
            # FIXED-mode boundary vehicles are skipped: their plan.power is
            # only the post-t=0 tail of a trajectory the model never chose,
            # so it need not sum to W_j on its own. OPTIMIZE-mode ones add
            # back the energy they already had at t=0 before comparing.
            already = bv.initial_energy_kwh if bv is not None else 0.0
            assert abs(energy + already - v.W) < 1e-4, (
                f"Vehicle {j} departs at slot {plan.departure} < K={K} with "
                f"{energy + already:.4f} kWh delivered (of which {already:.4f} pre-t=0), "
                f"not its full W_j={v.W:.4f} -- constraint (19)/completion at departure was "
                "violated by whichever plan was selected."
            )
        p_bar = min(v.p_max, station.n_modules * station.p_module)
        tau_d = v.tau_delta_hours(delta)
        # The taper cap the model enforced is tau_d*p + x_prev <= R with
        # x_prev seeded at the vehicle's head start, so the accumulator has
        # to start there too -- starting at 0 would check something strictly
        # weaker than the model guaranteed and quietly stop catching real
        # (18) violations.
        cum = bv.initial_energy_kwh if bv is not None else 0.0
        # FIXED-mode vehicles are exempt: build_cl_model skips Groups D+E
        # for them entirely, so (18) is not something their pinned,
        # exogenous trajectory was ever required to satisfy -- asserting it
        # here would fail on a trajectory the model never claimed to
        # constrain. Their power is instead checked against pile capacity,
        # which IS enforced, by boundary.assert_whole_module_feasible.
        check_physics = bv is None or bv.mode is BoundaryMode.OPTIMIZE
        for k in plan.occupied_slots():
            p_val = plan.power.get(k, 0.0)
            if check_physics:
                assert p_val <= p_bar + 1e-6, (
                    f"Vehicle {j}, slot {k}: p={p_val:.3f} > P_bar={p_bar:.3f}."
                )
                assert tau_d * p_val + cum <= v.R + 1e-4, (
                    f"Vehicle {j}, slot {k}: taper cap (18) violated "
                    f"({tau_d * p_val + cum:.4f} > R_j={v.R:.4f})."
                )
            cum += h * p_val
            occupancy.setdefault((plan.pile, k), []).append(j)  # type: ignore[index]
            power_by_pile_slot[plan.pile, k] = power_by_pile_slot.get((plan.pile, k), 0.0) + p_val  # type: ignore[index]

    for (pile, k), occs in occupancy.items():
        assert len(occs) <= station.n_connectors, (
            f"Pile {pile}, slot {k}: {len(occs)} vehicles simultaneously active > "
            f"C={station.n_connectors} -- row (25) was violated."
        )
    for (pile, k), total_p in power_by_pile_slot.items():
        cap = station.n_modules * station.p_module
        assert total_p <= cap + 1e-4, (
            f"Pile {pile}, slot {k}: total power {total_p:.3f} kW > N*Delta={cap:.3f} kW -- "
            "row (26) was violated."
        )

    if best_lower_bound is not None:
        assert best_lower_bound <= objective + 1e-4, (
            f"best_lower_bound={best_lower_bound:.3f} exceeds the recomputed schedule objective "
            f"{objective:.3f} -- the bound is invalid, most likely because a non-exact pricing "
            "round's Lagrangian value was used somewhere it shouldn't have been."
        )


def _release_slot(a: float, delta: float) -> int:
    return math.ceil(round(a / delta, 9))
