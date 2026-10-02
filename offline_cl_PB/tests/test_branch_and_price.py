"""
Tests for the exact branch-and-price solver (``offline_cl_PB``).

What each group establishes
---------------------------
1. Columns: every priced column passes the independent plan validator and
   its reduced cost, recomputed from its own coefficients, matches the
   pricing MILP's objective.
2. Column generation: on tiny instances the root restricted-master LP
   converges to the LP of the *full* master built by brute-force
   enumeration of every whole-module column (``enumeration.py``), and the
   root Lagrangian bound never exceeds it.
3. Nodes: under random branching restrictions the node's bound never
   exceeds the full master LP filtered by the same restrictions; a child
   whose columns were all filtered out is revived by Phase I rather than
   pruned; a truly empty node is proved infeasible; the pricer is never
   fooled by the null plan when the node forbids it.
4. Branching: every branch partitions the enumerated column set; support
   disagreement is detected even when the averaged value is integral;
   continuous power profiles are averaged instead of branched on.
5. Exactness: branch-and-price's optimum equals the compact MILP's proven
   optimum (``offline_cl_opt``), equals the full enumerated integer master,
   and its schedule satisfies every row of the compact model -- with and
   without boundary vehicles (FIXED and OPTIMIZE), with objective cohorts,
   with or without an external incumbent and the restricted-integer-master
   heuristic, sequential or parallel pricing.

Run: python -m pytest offline_cl_PB/tests -s -v
"""

from __future__ import annotations

import math
import random

import pytest

gp = pytest.importorskip("gurobipy")

from offline_cl_opt.boundary import COHORTS_MEASUREMENT, BoundaryMode, BoundaryVehicle, Cohort
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import build_cl_model, solve_cl_model

from offline_cl_PB import (
    BranchAndPrice,
    PBPlan,
    greedy_list_schedule,
    null_pb_plan,
    schedule_from_compact,
    solve_branch_and_price,
    validate_schedule,
)
from offline_cl_PB.branching import select_branch, try_recover
from offline_cl_PB.bp import Node
from offline_cl_PB.columns import column_key
from offline_cl_PB.master import MasterLP
from offline_cl_PB.pricer import reduced_cost
from offline_cl_PB.restrictions import NodeRestrictions, VehicleRestriction
from offline_cl_PB.validation import compact_model_check, validate_plan

from .enumeration import enumerate_columns, solve_full_master


def _veh(i, a, Q, s_i, s_f, p_max=60.0, s_th=0.4):
    return VehicleData(id=i, a=a, Q=Q, s_i=s_i, s_f=s_f, s_th=s_th, p_max=p_max)


# A tiny instance family: 10-minute slots, K = 5, piles owning 3 x 25 kW.
TINY_DELTA = 10.0
TINY_HORIZON = 50.0
TINY_K = 5


def _tiny_instance(seed: int, n_vehicles: int = 3, n_piles: int = 1, n_connectors: int = 2, n_modules: int = 3):
    rng = random.Random(seed)
    vehicles = []
    for i in range(n_vehicles):
        a = rng.choice([0.0, 0.0, 5.0, 10.0, 20.0])
        Q = rng.choice([20.0, 30.0, 40.0])
        s_i = rng.choice([0.1, 0.2, 0.3])
        s_f = rng.choice([0.7, 0.8, 0.9])
        vehicles.append(_veh(i, a, Q, s_i, s_f, p_max=rng.choice([40.0, 60.0, 90.0])))
    station = StationSpec(n_piles=n_piles, n_connectors=n_connectors, n_modules=n_modules, p_module=25.0)
    return vehicles, station


def _arrivals(vehicles):
    return {v.id: v.a for v in vehicles}


def _compact_optimum(vehicles, station, delta, horizon, **kw):
    cl = build_cl_model(vehicles, station, delta, horizon, tie_break=False, **kw)
    solve_cl_model(cl, mip_gap=0.0)
    return float(cl.model.ObjVal), cl


def _bp(vehicles, station, delta, horizon, **kw):
    kw.setdefault("rim_time_limit", 10.0)
    return BranchAndPrice(vehicles, station, delta, horizon, **kw)


def _no_incumbents(bp):
    """Node-level tests compare converged node LPs, so nothing may prune them:
    drop the incumbent and stop the solver from adopting new ones."""
    bp.upper_bound = math.inf
    bp._offer = lambda *args, **kwargs: False


# ---------------------------------------------------------------------------
# 1. Columns
# ---------------------------------------------------------------------------


def test_priced_columns_are_valid_and_reduced_costs_match():
    vehicles, station = _tiny_instance(1, n_vehicles=3)
    bp = _bp(vehicles, station, TINY_DELTA, TINY_HORIZON)
    rng = random.Random(7)
    checked = 0
    for trial in range(40):
        pi = {key: -rng.random() * 3 for key in bp.master.connector_rows}
        mu = {key: -rng.random() * 2 for key in bp.master.module_rows}
        for (j, m), pr in bp.pricers.items():
            pr.apply(VehicleRestriction())
            sigma = rng.uniform(0, 2 * TINY_DELTA * TINY_K)
            res = pr.price(pi, mu, sigma, include_departure=True, eps_rc=1e-6)
            if res.plan is None:
                continue
            assert not validate_plan(res.plan, bp.by_id[j], None, station, TINY_DELTA, TINY_K)
            # The objective cost, written out independently: sojourn in minutes.
            cost = bp.weights[j] * (TINY_DELTA * res.plan.departure - bp.by_id[j].a)
            recomputed = reduced_cost(res.plan, cost, pi, mu, sigma)
            assert recomputed == pytest.approx(res.reduced_cost, abs=1e-9)
            # clean-up can only lower module counts (mu <= 0), never raise the RC
            assert recomputed <= res.objective - sigma + 1e-6
            assert res.bound <= res.objective + 1e-6
            checked += 1
    bp._shutdown()
    assert checked > 20


def _labeling_vs_milp(vehicles, station, delta, horizon, boundary, rng, n_trials, restrict):
    from offline_cl_PB.labeling import LabelingPricer
    from offline_cl_PB.pricer import PBPricer
    from offline_cl_opt.model import earliest_departures

    K = math.ceil(round(horizon / delta, 9))
    E = earliest_departures(vehicles, station, delta, horizon)
    compared = 0
    for v in vehicles:
        bv = boundary.get(v.id)
        if bv is not None and bv.mode is BoundaryMode.FIXED:
            continue  # one fixed column, never priced
        piles = [bv.pile] if bv is not None else range(station.n_piles)
        for m in piles:
            kw = dict(initial_energy=bv.initial_energy_kwh if bv else 0.0, forced_start=bv is not None)
            ed = 0 if bv is not None else E[v.id]
            lab = LabelingPricer(v, m, station, delta, K, ed, **kw)
            mil = PBPricer(v, m, station, delta, K, ed, **kw)
            for _ in range(n_trials):
                pi = {(mm, k): -(rng.random() * 3 if rng.random() < 0.6 else 0.0) for mm in range(station.n_piles) for k in range(K)}
                mu = {(mm, k): -(rng.random() * 2 if rng.random() < 0.5 else 0.0) for mm in range(station.n_piles) for k in range(K)}
                r = VehicleRestriction()
                if restrict:
                    r = _random_restrictions(rng, [v], station, K).get(v.id)
                lab.apply(r)
                mil.apply(r)
                include = rng.random() < 0.8
                a = lab.price(pi, mu, 1e6, include_departure=include, eps_rc=1e-6)
                b = mil.price(pi, mu, 1e6, include_departure=include, eps_rc=1e-6)
                assert (a.status == "infeasible") == (b.status == "infeasible"), (v.id, m, r.describe())
                if a.status != "infeasible":
                    assert a.objective == pytest.approx(b.objective, abs=1e-6), (v.id, m, r.describe())
                    assert a.bound <= b.objective + 1e-6
                    assert not validate_plan(a.plan, v, bv, station, delta, K)
                    assert r.satisfied_by(a.plan, K)
                    compared += 1
            assert lab.fallbacks == 0
            mil.dispose()
            lab.dispose()
    return compared


@pytest.mark.parametrize("seed", range(3))
def test_labeling_pricer_matches_milp_pricer(seed):
    rng = random.Random(seed)
    vehicles, station = _tiny_instance(40 + seed, n_vehicles=3, n_piles=2)
    assert _labeling_vs_milp(vehicles, station, TINY_DELTA, TINY_HORIZON, {}, rng, 15, restrict=True) > 30
    vehicles, station = _contended_instance(seed, 4, 1)
    assert _labeling_vs_milp(vehicles, station, 5.0, 60.0, {}, rng, 10, restrict=True) > 20
    vehicles, station, delta, horizon, boundary, _ = _boundary_instance()
    assert _labeling_vs_milp(vehicles, station, delta, horizon, boundary, rng, 8, restrict=False) > 10


def test_greedy_list_schedule_is_feasible():
    for seed in range(6):
        vehicles, station = _tiny_instance(seed, n_vehicles=4, n_piles=2)
        sched = greedy_list_schedule(vehicles, station, TINY_DELTA, TINY_K, {})
        validate_schedule(sched, vehicles, station, TINY_DELTA, TINY_K, {}, {v.id: 1.0 for v in vehicles})


# ---------------------------------------------------------------------------
# 2. Root column generation vs the full enumerated master
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed,n_piles", [(0, 1), (1, 1), (2, 2), (3, 2)])
def test_root_colgen_matches_full_master_lp(seed, n_piles):
    vehicles, station = _tiny_instance(seed, n_vehicles=3, n_piles=n_piles)
    weights = {v.id: 1.0 for v in vehicles}
    columns = enumerate_columns(vehicles, station, TINY_DELTA, TINY_K)
    z_full = solve_full_master(columns, station, TINY_K, weights, delta=TINY_DELTA, arrivals=_arrivals(vehicles), integer=False)

    bp = _bp(vehicles, station, TINY_DELTA, TINY_HORIZON)
    _no_incumbents(bp)  # no pruning: run the root to convergence
    out = bp._solve_node(Node(0, None, 0, NodeRestrictions(), -math.inf))
    bp._shutdown()
    assert out.status == "converged"
    assert out.lp.objective == pytest.approx(z_full, abs=1e-5)
    assert out.lower_bound <= z_full + 1e-6
    assert out.lower_bound >= z_full - len(vehicles) * 2e-6
    # every generated column is one of the enumerated ones
    enumerated = {column_key(p) for plans in columns.values() for p in plans}
    assert all(c.key in enumerated for c in bp.master.columns)


# ---------------------------------------------------------------------------
# 3. Nodes: restricted LPs, Phase I, the null plan
# ---------------------------------------------------------------------------


def _random_restrictions(rng, vehicles, station, K):
    restr = NodeRestrictions()
    for v in vehicles:
        r = VehicleRestriction()
        kind = rng.choice(["none", "dmax", "dmin", "smax", "smin", "pile", "nopile", "qmin", "qmax"])
        if kind == "dmax":
            r = r.with_d_max(rng.randint(1, K))
        elif kind == "dmin":
            r = r.with_d_min(rng.randint(1, K))
        elif kind == "smax":
            r = r.with_s_max(rng.randint(0, K - 1))
        elif kind == "smin":
            r = r.with_s_min(rng.randint(0, K))
        elif kind == "pile":
            r = r.with_pile_only(rng.randrange(station.n_piles))
        elif kind == "nopile":
            r = r.with_pile_forbidden(rng.randrange(station.n_piles))
        elif kind == "qmin":
            r = r.with_q_min(rng.randrange(K), rng.randint(1, station.n_modules))
        elif kind == "qmax":
            r = r.with_q_max(rng.randrange(K), rng.randint(0, station.n_modules - 1))
        restr = restr.with_vehicle(v.id, r)
    return restr


@pytest.mark.parametrize("seed", range(8))
def test_node_bound_under_random_restrictions_matches_filtered_full_master(seed):
    vehicles, station = _tiny_instance(10 + seed, n_vehicles=3, n_piles=2)
    weights = {v.id: 1.0 for v in vehicles}
    columns = enumerate_columns(vehicles, station, TINY_DELTA, TINY_K)
    rng = random.Random(100 + seed)
    bp = _bp(vehicles, station, TINY_DELTA, TINY_HORIZON)
    _no_incumbents(bp)
    try:
        for _ in range(4):
            restr = _random_restrictions(rng, vehicles, station, TINY_K)
            z = solve_full_master(columns, station, TINY_K, weights, delta=TINY_DELTA, arrivals=_arrivals(vehicles), integer=False,
                                  allowed=lambda p, r=restr: r.column_satisfies(p, TINY_K))
            out = bp._solve_node(Node(1, 0, 1, restr, -math.inf))
            if z is None:
                assert out.status == "infeasible", restr.describe()
            else:
                assert out.status == "converged", restr.describe()
                assert out.lp.objective == pytest.approx(z, abs=1e-5), restr.describe()
                assert out.lower_bound <= z + 1e-6
    finally:
        bp._shutdown()


def test_phase1_revives_node_whose_columns_were_all_filtered():
    """Root pool has only early departures for vehicle 0; the child D >= t+1
    filters all of them out. The filtered RMP is infeasible, but the child is
    not: Phase I must find the missing later-departure column."""
    vehicles, station = _tiny_instance(3, n_vehicles=2, n_piles=1)
    bp = _bp(vehicles, station, TINY_DELTA, TINY_HORIZON)
    _no_incumbents(bp)
    v0 = vehicles[0]
    departures = {c.plan.departure for c in bp.master.columns if c.plan.vehicle_id == v0.id and not c.plan.is_null}
    t = max(departures)
    assert t < TINY_K  # a later, served departure exists and is not in the pool yet
    restr = NodeRestrictions().with_vehicle(v0.id, VehicleRestriction().with_d_min(t + 1).with_s_max(TINY_K - 1))
    bp.master.activate(restr)
    assert not any(bp.master.is_active(c.index) for c in bp.master.columns if c.plan.vehicle_id == v0.id)
    out = bp._solve_node(Node(1, 0, 1, restr, -math.inf))
    bp._shutdown()
    assert out.status == "converged"
    assert out.phase1_iterations > 0
    columns = enumerate_columns(vehicles, station, TINY_DELTA, TINY_K)
    z = solve_full_master(columns, station, TINY_K, {v.id: 1.0 for v in vehicles}, delta=TINY_DELTA, arrivals=_arrivals(vehicles), integer=False,
                          allowed=lambda p: restr.column_satisfies(p, TINY_K))
    assert out.lp.objective == pytest.approx(z, abs=1e-5)


def test_phase1_proves_true_infeasibility():
    vehicles, station = _tiny_instance(4, n_vehicles=2, n_piles=1)
    bp = _bp(vehicles, station, TINY_DELTA, TINY_HORIZON)
    _no_incumbents(bp)
    v0 = vehicles[0]
    # Served (S <= K-1) but departing before its own earliest possible departure.
    E = min(c.plan.departure for c in bp.master.columns if c.plan.vehicle_id == v0.id and not c.plan.is_null)
    restr = NodeRestrictions().with_vehicle(v0.id, VehicleRestriction().with_s_max(TINY_K - 1).with_d_max(E - 1))
    out = bp._solve_node(Node(1, 0, 1, restr, -math.inf))
    bp._shutdown()
    assert out.status == "infeasible"
    assert out.phase1_iterations > 0


def test_pricer_never_returns_null_and_node_forbidding_null_is_priced_correctly():
    """pile == m forbids the null plan. The pricer is served-only, so it must
    find the best served plan even when the convexity dual exceeds the null plan's cost."""
    vehicles, station = _tiny_instance(5, n_vehicles=3, n_piles=2)
    bp = _bp(vehicles, station, TINY_DELTA, TINY_HORIZON)
    pr = bp.pricers[(vehicles[0].id, 0)]
    pr.apply(VehicleRestriction().with_pile_only(0))
    res = pr.price({}, {}, sigma=10 * TINY_DELTA * TINY_K, include_departure=True, eps_rc=1e-6)
    assert res.plan is not None and not res.plan.is_null
    _no_incumbents(bp)
    columns = enumerate_columns(vehicles, station, TINY_DELTA, TINY_K)
    for j in (vehicles[0].id, vehicles[1].id):
        restr = NodeRestrictions().with_vehicle(j, VehicleRestriction().with_pile_only(1))
        out = bp._solve_node(Node(1, 0, 1, restr, -math.inf))
        z = solve_full_master(columns, station, TINY_K, {v.id: 1.0 for v in vehicles}, delta=TINY_DELTA, arrivals=_arrivals(vehicles), integer=False,
                              allowed=lambda p, r=restr: r.column_satisfies(p, TINY_K))
        assert out.lp.objective == pytest.approx(z, abs=1e-5)
    bp._shutdown()


# ---------------------------------------------------------------------------
# 4. Branching
# ---------------------------------------------------------------------------


def test_branches_partition_the_enumerated_columns():
    vehicles, station = _tiny_instance(6, n_vehicles=2, n_piles=2)
    columns = enumerate_columns(vehicles, station, TINY_DELTA, TINY_K)
    j = vehicles[0].id
    plans = columns[j]
    base = VehicleRestriction()
    splits = []
    for t in range(0, TINY_K + 1):
        splits.append((base.with_d_max(t), base.with_d_min(t + 1)))
        splits.append((base.with_s_max(t), base.with_s_min(t + 1)))
    for m in range(station.n_piles):
        splits.append((base.with_pile_only(m), base.with_pile_forbidden(m)))
    for k in range(TINY_K):
        for t in range(station.n_modules):
            splits.append((base.with_q_max(k, t), base.with_q_min(k, t + 1)))
    for left, right in splits:
        for plan in plans:
            assert left.satisfied_by(plan, TINY_K) != right.satisfied_by(plan, TINY_K), (left, right, plan)


def _plan(j, pile, S, D, q, p):
    return PBPlan(vehicle_id=j, pile=pile, start=S, departure=D,
                  power={S + i: p[i] for i in range(D - S) if p[i] > 0},
                  modules={S + i: q[i] for i in range(D - S)})


def test_support_disagreement_detected_even_when_average_is_integral():
    a = _plan(0, 0, 0, 1, [1], [10.0])
    b = _plan(0, 0, 0, 3, [1, 1, 1], [10.0, 10.0, 10.0])
    null = null_pb_plan(0, TINY_K)
    # departures 1 and 3 with weights 0.5/0.5 average to 2 -- integral, still mixed
    branch = select_branch({0: [(a, 0.5), (b, 0.5)]}, NodeRestrictions(), TINY_K)
    assert branch is not None and branch.kind == "departure"
    # null (S = K) vs a served plan with the same D = K: start disagrees
    c = _plan(0, 0, 2, TINY_K, [1, 1, 1], [5.0, 5.0, 5.0])
    branch = select_branch({0: [(null, 0.5), (c, 0.5)]}, NodeRestrictions(), TINY_K)
    assert branch is not None and branch.kind == "start"


def test_continuous_profiles_are_averaged_not_branched():
    v = _veh(0, 0.0, 30.0, 0.2, 0.8, p_max=60.0)
    station = StationSpec(n_piles=1, n_connectors=2, n_modules=3, p_module=25.0)
    h = TINY_DELTA / 60.0
    W = v.W  # 18 kWh
    # Same timetable (slots 0-3) and same module profile, different power splits.
    p1 = [30.0, 30.0, 30.0, (W - 90 * h) / h]
    p2 = [40.0, 20.0, 30.0, (W - 90 * h) / h]
    q = [2, 2, 2, 2]
    a, b = _plan(0, 0, 0, 4, q, p1), _plan(0, 0, 0, 4, q, p2)
    assert not validate_plan(a, v, None, station, TINY_DELTA, TINY_K)
    assert not validate_plan(b, v, None, station, TINY_DELTA, TINY_K)
    support = {0: [(a, 0.3), (b, 0.7)]}
    rec = try_recover(support, [v], station, TINY_DELTA, TINY_K, {})
    assert rec.schedule is not None and select_branch(support, NodeRestrictions(), TINY_K) is None
    # Different module profiles, same timetable: averaged power, ceil modules.
    c = _plan(0, 0, 0, 4, [2, 1, 2, 2], [30.0, 25.0, 35.0, (W - 90 * h) / h])
    rec = try_recover({0: [(a, 0.5), (c, 0.5)]}, [v], station, TINY_DELTA, TINY_K, {})
    assert rec.schedule is not None and rec.averaged
    assert not validate_plan(rec.schedule[0], v, None, station, TINY_DELTA, TINY_K)


# ---------------------------------------------------------------------------
# 5. Exactness against the compact model and the full integer master
# ---------------------------------------------------------------------------


def _assert_matches(sol, z_star, vehicles):
    assert sol.status == "OPTIMAL"
    assert sol.objective == pytest.approx(z_star, abs=1e-6)
    assert sol.lower_bound == pytest.approx(z_star, abs=1e-6)
    assert sol.compact_check is not None
    assert sol.compact_check.max_violation <= 1e-5, sol.compact_check
    assert sol.compact_check.objective == pytest.approx(z_star, abs=1e-6)


@pytest.mark.parametrize("seed,n_piles,n_vehicles", [(20, 1, 3), (21, 1, 4), (22, 2, 4), (23, 2, 5), (24, 1, 5)])
def test_bp_matches_compact_and_full_integer_master(seed, n_piles, n_vehicles):
    vehicles, station = _tiny_instance(seed, n_vehicles=n_vehicles, n_piles=n_piles)
    z_star, _ = _compact_optimum(vehicles, station, TINY_DELTA, TINY_HORIZON)
    sol = solve_branch_and_price(vehicles, station, TINY_DELTA, TINY_HORIZON, rim_time_limit=10.0)
    _assert_matches(sol, z_star, vehicles)
    if n_vehicles <= 4:
        columns = enumerate_columns(vehicles, station, TINY_DELTA, TINY_K)
        z_int = solve_full_master(columns, station, TINY_K, {v.id: 1.0 for v in vehicles}, delta=TINY_DELTA, arrivals=_arrivals(vehicles), integer=True)
        assert z_int == pytest.approx(z_star, abs=1e-6)


def _contended_instance(seed, n_vehicles, n_piles):
    """5-minute slots, K = 12, 4 x 25 kW per pile: vehicles fight over modules,
    so these need real branching (found by scanning seeds; node counts below)."""
    rng = random.Random(seed)
    vehicles = []
    for i in range(n_vehicles):
        a = float(rng.choice([0, 0, 5, 10, 15, 20, 25]))
        Q = float(rng.choice([20, 25, 30, 40]))
        s_i = rng.choice([0.1, 0.2, 0.3])
        s_f = rng.choice([0.7, 0.8, 0.9])
        vehicles.append(_veh(i, a, Q, s_i, s_f, p_max=float(rng.choice([50, 75, 100]))))
    station = StationSpec(n_piles=n_piles, n_connectors=2, n_modules=4, p_module=25.0)
    return vehicles, station


@pytest.mark.parametrize("pricer", ["labeling", "milp"])
@pytest.mark.parametrize(
    "seed,n_piles,n_vehicles,min_nodes",
    # min_nodes: how much of the tree must be exercised. (4, 1, 4) may close at
    # the root or branch depending on the pricer's path, so it only checks the answer.
    [(0, 2, 6, 10), (3, 1, 5, 3), (4, 1, 4, 1), (0, 1, 5, 3)],
)
def test_bp_matches_compact_on_instances_that_branch(seed, n_piles, n_vehicles, min_nodes, pricer):
    vehicles, station = _contended_instance(seed, n_vehicles, n_piles)
    z_star, _ = _compact_optimum(vehicles, station, 5.0, 60.0)
    # Defaults (heuristics on).
    sol = solve_branch_and_price(vehicles, station, 5.0, 60.0, rim_time_limit=5.0, pricer=pricer)
    _assert_matches(sol, z_star, vehicles)
    assert sol.root_lower_bound <= z_star + 1e-6
    # Heuristics off: incumbents can only come from the tree itself, so the
    # tree (not just the root) must do the work.
    sol = solve_branch_and_price(vehicles, station, 5.0, 60.0, rim_time_limit=0.0, dive_every=None, pricer=pricer)
    _assert_matches(sol, z_star, vehicles)
    assert sol.nodes_processed >= min_nodes


def test_known_instance_optimum_113():
    """tests/test_connector_lane_optimization.py's instance: compact optimum
    sum_j D_j = 118 slots, i.e. total sojourn 1.0*118 - (0 + 0 + 5) = 113 min."""
    vs = [
        _veh(0, 0.0, 50.0, 0.2, 0.8, p_max=100.0),
        _veh(1, 0.0, 100.0, 0.15, 0.85, p_max=200.0),
        _veh(2, 5.0, 50.0, 0.2, 0.8, p_max=100.0),
    ]
    station = StationSpec(n_piles=2, n_connectors=2, n_modules=4, p_module=25.0)
    z_star, _ = _compact_optimum(vs, station, 1.0, 100.0)
    assert z_star == pytest.approx(113.0)
    # K = 100 one-minute slots: the root LP (108) is far below 113, so proving
    # optimality takes a long tree. Within a time limit the bracket must still
    # contain the optimum and the schedule must satisfy the compact model.
    sol = solve_branch_and_price(vs, station, 1.0, 100.0, time_limit=90.0)
    assert sol.lower_bound <= z_star + 1e-6 <= sol.objective + 2e-6
    assert sol.compact_check.max_violation <= 1e-5
    assert sol.compact_check.objective == pytest.approx(sol.objective, abs=1e-6)
    if sol.status == "OPTIMAL":
        assert sol.objective == pytest.approx(z_star, abs=1e-6)


def _boundary_instance():
    """Two in-service vehicles on pile 0 at t=0 (one FIXED, one OPTIMIZE), plus
    arrivals, on a 1-pile station -- modules are scarce so they interact."""
    delta, horizon = 5.0, 60.0
    station = StationSpec(n_piles=1, n_connectors=2, n_modules=4, p_module=25.0)
    fixed_v = _veh(10, 0.0, 30.0, 0.3, 0.8, p_max=60.0)
    opt_v = _veh(11, 0.0, 40.0, 0.2, 0.85, p_max=90.0)
    arrivals = [_veh(0, 5.0, 30.0, 0.2, 0.8), _veh(1, 10.0, 25.0, 0.3, 0.9, p_max=90.0), _veh(2, 12.0, 20.0, 0.1, 0.7)]
    boundary = {
        10: BoundaryVehicle(vehicle_id=10, pile=0, connector=1, mode=BoundaryMode.FIXED,
                            departure_slot=3, power={0: 50.0, 1: 45.0, 2: 20.0}),
        11: BoundaryVehicle(vehicle_id=11, pile=0, connector=0, mode=BoundaryMode.OPTIMIZE,
                            initial_energy_kwh=6.0),
    }
    cohorts = {10: Cohort.BOUNDARY, 11: Cohort.BOUNDARY, 0: Cohort.MEASUREMENT, 1: Cohort.MEASUREMENT,
               2: Cohort.QUEUED}
    return [fixed_v, opt_v] + arrivals, station, delta, horizon, boundary, cohorts


@pytest.mark.parametrize("objective_cohorts", [None, COHORTS_MEASUREMENT])
def test_bp_matches_compact_with_boundary_vehicles_and_cohorts(objective_cohorts):
    vehicles, station, delta, horizon, boundary, cohorts = _boundary_instance()
    kw = dict(boundary_vehicles=boundary, cohorts=cohorts, objective_cohorts=objective_cohorts)
    z_star, _ = _compact_optimum(vehicles, station, delta, horizon, **kw)
    sol = solve_branch_and_price(vehicles, station, delta, horizon, rim_time_limit=10.0, **kw)
    _assert_matches(sol, z_star, vehicles)
    # boundary vehicles keep their pile and connector
    assert sol.connectors[10] == 1 and sol.connectors[11] == 0
    assert sol.schedule[10].departure == 3


def test_same_optimum_with_any_incumbent_and_with_or_without_rim():
    vehicles, station = _tiny_instance(30, n_vehicles=5, n_piles=2)
    z_star, cl = _compact_optimum(vehicles, station, TINY_DELTA, TINY_HORIZON)
    good, _ = schedule_from_compact(cl)
    poor = {v.id: null_pb_plan(v.id, TINY_K) for v in vehicles}
    for kw in (
        dict(),
        dict(initial_schedule=good),
        dict(initial_schedule=poor),
        dict(rim_time_limit=0.0),
        dict(rim_time_limit=0.0, initial_schedule=good),
        dict(max_workers=2),
    ):
        kw.setdefault("rim_time_limit", 10.0)
        sol = solve_branch_and_price(vehicles, station, TINY_DELTA, TINY_HORIZON, **kw)
        _assert_matches(sol, z_star, vehicles)


def test_compact_solution_round_trips_through_columns():
    vehicles, station, delta, horizon, boundary, cohorts = _boundary_instance()
    z_star, cl = _compact_optimum(vehicles, station, delta, horizon, boundary_vehicles=boundary, cohorts=cohorts)
    schedule, connectors = schedule_from_compact(cl)
    K = cl.K
    check = validate_schedule(schedule, vehicles, station, delta, K, boundary, {v.id: 1.0 for v in vehicles})
    assert check.objective == pytest.approx(z_star, abs=1e-6)
    cc = compact_model_check(schedule, check.connectors, vehicles, station, delta, horizon,
                             boundary_vehicles=boundary, cohorts=cohorts)
    assert cc.max_violation <= 1e-5 and cc.objective == pytest.approx(z_star, abs=1e-6)


def test_time_limit_returns_valid_bracket():
    vehicles, station = _tiny_instance(31, n_vehicles=5, n_piles=2)
    z_star, _ = _compact_optimum(vehicles, station, TINY_DELTA, TINY_HORIZON)
    sol = solve_branch_and_price(vehicles, station, TINY_DELTA, TINY_HORIZON, node_limit=1, rim_time_limit=0.0)
    assert sol.status in ("OPTIMAL", "NODE_LIMIT")
    assert sol.lower_bound <= z_star + 1e-6 <= sol.objective + 2e-6


@pytest.mark.parametrize("node_limit", [None, 2])
def test_bound_history_is_a_monotone_bracket_ending_at_the_result(node_limit):
    """Every recorded [lower, upper] contains z*, lower never drops, upper
    never rises, and the last entry is the returned (lower_bound, objective)."""
    vehicles, station = _contended_instance(0, 5, 1)
    z_star, _ = _compact_optimum(vehicles, station, 5.0, 60.0)
    sol = solve_branch_and_price(vehicles, station, 5.0, 60.0, rim_time_limit=0.0, dive_every=None,
                                 node_limit=node_limit)
    h = sol.bound_history
    assert len(h) >= 2 and h[-1]["event"] == "end"
    for prev, cur in zip(h, h[1:]):
        assert cur["time_s"] >= prev["time_s"]
        assert cur["lower_bound"] >= prev["lower_bound"]
        assert cur["upper_bound"] <= prev["upper_bound"]
        assert cur["nodes"] >= prev["nodes"]
    for p in h:
        assert p["lower_bound"] <= p["upper_bound"]
        assert p["lower_bound"] <= z_star + 1e-6 <= p["upper_bound"] + 2e-6
    assert h[-1]["lower_bound"] == sol.lower_bound and h[-1]["upper_bound"] == sol.objective
    if sol.status == "OPTIMAL":
        assert h[-1]["lower_bound"] == h[-1]["upper_bound"]


# ---------------------------------------------------------------------------
# 6. Simulation incumbent and model export
# ---------------------------------------------------------------------------


def _cl_model_notebook_instance():
    """cl_model.ipynb's scenario: FIFO simulation with warm-up -> offline instance."""
    import numpy as np
    from config import HR2MIN
    from env.charging_env import ChargingStationEnv
    from models.ev import EV
    from offline_cl_opt.boundary import build_measurement_instance
    from policy.queue.fifo import FIFOQueuePolicy

    warmup, measure, delta = 12.0, 60.0, 1.0
    station = StationSpec(n_piles=1, n_connectors=2, n_modules=6, p_module=25.0)
    specs = [(0, 0.0, 40.0, 0.20, 0.75), (1, 3.0, 40.0, 0.25, 0.75), (2, 11.0, 100.0, 0.20, 0.75),
             (3, 20.0, 40.0, 0.20, 0.80), (4, 30.0, 100.0, 0.25, 0.80), (5, 42.0, 35.0, 0.25, 0.70)]
    evs = [EV(id=i, c_b=kwh * HR2MIN, s_i=s_i, s_f=s_f, arrival_time=t) for (i, t, kwh, s_i, s_f) in specs]
    env = ChargingStationEnv(n_piles=1, n_connectors=2, n_modules=6, p_module=25.0, queue_capacity=10,
                             mean_interarrival=None, arrivals=evs, max_time=measure, warmup_period=warmup)
    obs, _ = env.reset(seed=0)
    rng, policy, done = np.random.default_rng(1), FIFOQueuePolicy(), False
    while not done:
        ev, pile_id = policy.decide(obs, env.action_masks(), rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)
    m = env.engine.metrics
    inst = build_measurement_instance(
        arrived_post_warmup=m.arrived_post_warmup, queued_at_warmup_end=m.queued_at_warmup_end,
        in_service_at_warmup_end=m.in_service_at_warmup_end, warmup_period=warmup, delta=delta,
        horizon_minutes=measure, include_queued=True, boundary_mode=BoundaryMode.OPTIMIZE, station=station)
    return m, inst, station, delta, measure, warmup


def test_simulation_incumbent_is_feasible_and_accepted():
    from offline_cl_PB import schedule_from_simulation

    m, inst, station, delta, horizon, warmup = _cl_model_notebook_instance()
    K = int(round(horizon / delta))
    sched, counts = schedule_from_simulation(m.arrived_evs, inst.vehicles, station, delta, K,
                                             inst.boundary_vehicles, warmup_period=warmup)
    weights = {v.id: 1.0 for v in inst.vehicles}
    check = validate_schedule(sched, inst.vehicles, station, delta, K, inst.boundary_vehicles, weights)
    cc = compact_model_check(sched, check.connectors, inst.vehicles, station, delta, horizon,
                             boundary_vehicles=inst.boundary_vehicles, cohorts=inst.cohorts)
    assert cc.max_violation <= 1e-5 and cc.objective == pytest.approx(check.objective)
    assert sum(counts.values()) == len(inst.vehicles) - len(inst.boundary_vehicles)
    bp = _bp(inst.vehicles, station, delta, horizon, boundary_vehicles=inst.boundary_vehicles,
             cohorts=inst.cohorts, initial_schedule=sched, dive_every=None, rim_time_limit=0.0)
    assert bp.upper_bound <= check.objective  # adopted (or beaten by the greedy)
    bp._shutdown()


def test_to_cl_model_is_a_drop_in_compact_model():
    from offline_cl_opt.solution import extract_solution

    vehicles, station, delta, horizon, boundary, cohorts = _boundary_instance()
    sol = solve_branch_and_price(vehicles, station, delta, horizon, boundary_vehicles=boundary,
                                 cohorts=cohorts, rim_time_limit=5.0)
    cl = sol.to_cl_model()
    ext = extract_solution(cl)
    assert ext.objective == pytest.approx(sol.objective, abs=1e-6)
    by_id = ext.per_vehicle.set_index("vehicle_id")
    for j, plan in sol.schedule.items():
        assert by_id.loc[j, "departure_slot"] == pytest.approx(plan.departure)
        for k, p in plan.power.items():
            assert cl.p[j, k].X == pytest.approx(p, abs=1e-6)
