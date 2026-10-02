"""
Reading a node's master LP solution: positive support, integer recovery,
and structural branching.

Branching never acts on a master variable. It splits one vehicle's plan
space on an attribute the pricer can enforce, in this order:

1. departure  ``D_j <= t``  |  ``D_j >= t+1``
2. start      ``S_j <= t``  |  ``S_j >= t+1``   (null plan: ``S = K``)
3. pile       ``m_j == m``  |  ``m_j != m``     (null plan: the ``!=`` side)
4. modules    ``q_jk <= t`` |  ``q_jk >= t+1``

A vehicle needs branching when its *positive-weight columns disagree* on
one of these -- not when an averaged value is fractional
(``0.5*10 + 0.5*30 = 20`` is integral but mixes two departures). Each
split puts every plan in exactly one child, and both children keep some
of the current support, so the parent's LP solution is cut off.

Integer recovery. When every vehicle's support agrees on pile, start and
departure, the LP solution already describes one timetable. If the
supports also agree on the module profile, any one supporting column per
vehicle is an exact schedule (all of them have identical master
coefficients). If only module profiles differ, the power profiles are
averaged -- every single-vehicle constraint is linear for a fixed
occupancy, so the average is feasible -- and each slot is given
``ceil(p/Delta)`` modules; the result is exact whenever those counts fit
every pile (the compact document's Lemma 8.1). Otherwise the solver
branches on a module count.
"""

from __future__ import annotations

from dataclasses import dataclass

from offline_cl_opt.boundary import BoundaryVehicle
from offline_cl_opt.instance import StationSpec, VehicleData

from .columns import PBPlan, modules_needed
from .master import MasterLP, PBMaster
from .restrictions import NodeRestrictions
from .validation import ScheduleValidationError, validate_plan


@dataclass
class Branch:
    vehicle_id: int
    kind: str  # "departure" | "start" | "pile" | "modules"
    description: str
    left: NodeRestrictions
    right: NodeRestrictions
    left_weight: float = 0.0  # LP weight of the current support on each side
    right_weight: float = 0.0


def positive_support(lp: MasterLP, master: PBMaster) -> dict[int, list[tuple[PBPlan, float]]]:
    support: dict[int, list[tuple[PBPlan, float]]] = {}
    for index, lam in lp.lam.items():
        plan = master.columns[index].plan
        support.setdefault(plan.vehicle_id, []).append((plan, lam))
    return support


def _averaged_plan(
    items: list[tuple[PBPlan, float]],
    v: VehicleData,
    bv: BoundaryVehicle | None,
    station: StationSpec,
    delta: float,
    K: int,
) -> PBPlan | None:
    """Weighted average of plans sharing (pile, S, D); ``ceil(p/Delta)`` modules."""
    total = sum(lam for _, lam in items)
    first = items[0][0]
    power: dict[int, float] = {}
    modules: dict[int, int] = {}
    for k in first.occupied_slots():
        p = sum(lam * plan.power.get(k, 0.0) for plan, lam in items) / total
        if p > 0.0:
            power[k] = p
        modules[k] = modules_needed(p, station.p_module)
    plan = PBPlan(
        vehicle_id=first.vehicle_id,
        pile=first.pile,
        start=first.start,
        departure=first.departure,
        power=power,
        modules=modules,
    )
    if validate_plan(plan, v, bv, station, delta, K):
        return None
    return plan


@dataclass
class Recovery:
    schedule: dict[int, PBPlan] | None
    averaged: bool  # True if some vehicle's plan was built by averaging
    failing_pile_slots: set[tuple[int, int]]  # where averaged module counts overflowed


def try_recover(
    support: dict[int, list[tuple[PBPlan, float]]],
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    K: int,
    boundary_vehicles: dict[int, BoundaryVehicle],
) -> Recovery:
    schedule: dict[int, PBPlan] = {}
    averaged = False
    for v in vehicles:
        items = support.get(v.id, [])
        if not items:
            return Recovery(None, False, set())
        timetable = {(plan.pile, plan.start_slot(K), plan.departure) for plan, _ in items}
        if len(timetable) > 1:
            return Recovery(None, False, set())
        profiles = {plan.q_profile() for plan, _ in items}
        if len(profiles) == 1:
            schedule[v.id] = max(items, key=lambda it: it[1])[0]
        else:
            plan = _averaged_plan(items, v, boundary_vehicles.get(v.id), station, delta, K)
            if plan is None:
                return Recovery(None, True, set())
            schedule[v.id] = plan
            averaged = True

    count: dict[tuple[int, int], int] = {}
    used: dict[tuple[int, int], int] = {}
    for plan in schedule.values():
        if plan.is_null:
            continue
        for k in plan.occupied_slots():
            count[plan.pile, k] = count.get((plan.pile, k), 0) + 1  # type: ignore[index]
            used[plan.pile, k] = used.get((plan.pile, k), 0) + plan.q(k)  # type: ignore[index]
    over_connectors = [key for key, c in count.items() if c > station.n_connectors]
    if over_connectors:
        # Every vehicle's occupancy is unanimous, so the master's connector
        # rows already bound these counts -- this cannot happen unless the
        # master itself is inconsistent.
        raise ScheduleValidationError(f"recovered timetable overfills connectors at {over_connectors[:5]}")
    failing = {key for key, q in used.items() if q > station.n_modules}
    if failing:
        if not averaged:
            raise ScheduleValidationError(f"unanimous module profiles overfill pile-slots {sorted(failing)[:5]}")
        return Recovery(None, True, failing)
    return Recovery(schedule, averaged, set())


def _best_split(values: dict[int, float]) -> tuple[int, float, float]:
    """Threshold t between two adjacent supported values, as balanced as possible."""
    ordered = sorted(values.items())
    total = sum(w for _, w in ordered)
    best = None
    cum = 0.0
    for i in range(len(ordered) - 1):
        cum += ordered[i][1]
        score = abs(cum - total / 2.0)
        if best is None or score < best[0]:
            best = (score, ordered[i][0], cum, total - cum)
    assert best is not None
    return best[1], best[2], best[3]


def select_branch(
    support: dict[int, list[tuple[PBPlan, float]]],
    restrictions: NodeRestrictions,
    K: int,
    *,
    preferred_pile_slots: set[tuple[int, int]] | None = None,
) -> Branch | None:
    """The branch to take at a converged, unrecovered node (``None`` if supports agree everywhere)."""
    preferred_pile_slots = preferred_pile_slots or set()

    def attribute_candidates(attr):
        cands = []
        for j, items in support.items():
            weights: dict[int, float] = {}
            for plan, lam in items:
                key = attr(plan)
                weights[key] = weights.get(key, 0.0) + lam
            if len(weights) > 1:
                t, left_w, right_w = _best_split(weights)
                cands.append((min(left_w, right_w), -j, j, t, left_w, right_w))
        return cands

    for kind, attr in (
        ("departure", lambda plan: plan.departure),
        ("start", lambda plan: plan.start_slot(K)),
    ):
        cands = attribute_candidates(attr)
        if cands:
            _, _, j, t, left_w, right_w = max(cands)
            r = restrictions.get(j)
            if kind == "departure":
                left, right = r.with_d_max(t), r.with_d_min(t + 1)
                desc = f"v{j}: D <= {t} | D >= {t + 1}"
            else:
                left, right = r.with_s_max(t), r.with_s_min(t + 1)
                desc = f"v{j}: S <= {t} | S >= {t + 1}"
            return Branch(
                j, kind, desc, restrictions.with_vehicle(j, left), restrictions.with_vehicle(j, right), left_w, right_w
            )

    # Pile: start/departure now agree for every vehicle, so a vehicle whose
    # support contains the null plan has only null plans in it.
    cands = []
    for j, items in support.items():
        weights: dict[int, float] = {}
        for plan, lam in items:
            if plan.is_null:
                continue
            weights[plan.pile] = weights.get(plan.pile, 0.0) + lam  # type: ignore[index]
        if len(weights) > 1:
            total = sum(weights.values())
            m_best, w_best = max(weights.items(), key=lambda kv: (kv[1], -kv[0]))
            cands.append((min(w_best, total - w_best), -j, j, m_best, w_best, total - w_best))
    if cands:
        _, _, j, m_best, left_w, right_w = max(cands)
        r = restrictions.get(j)
        return Branch(
            j,
            "pile",
            f"v{j}: pile == {m_best} | pile != {m_best}",
            restrictions.with_vehicle(j, r.with_pile_only(m_best)),
            restrictions.with_vehicle(j, r.with_pile_forbidden(m_best)),
            left_w,
            right_w,
        )

    cands = []
    for j, items in support.items():
        first = items[0][0]
        if first.is_null:
            continue
        for k in first.occupied_slots():
            weights = {}
            for plan, lam in items:
                weights[plan.q(k)] = weights.get(plan.q(k), 0.0) + lam
            if len(weights) > 1:
                t, left_w, right_w = _best_split(weights)
                preferred = 1 if (first.pile, k) in preferred_pile_slots else 0
                cands.append((preferred, min(left_w, right_w), -j, -k, j, k, t, left_w, right_w))
    if cands:
        *_, j, k, t, left_w, right_w = max(cands)
        r = restrictions.get(j)
        return Branch(
            j,
            "modules",
            f"v{j}: q[{k}] <= {t} | q[{k}] >= {t + 1}",
            restrictions.with_vehicle(j, r.with_q_max(k, t)),
            restrictions.with_vehicle(j, r.with_q_min(k, t + 1)),
            left_w,
            right_w,
        )
    return None
