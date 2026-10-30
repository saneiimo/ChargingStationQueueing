"""
Tests for the price-directed state-space search (``offline_cl_PDS``).

What each group establishes
---------------------------
1. Dynamics: the grid dynamics (46) never deliver less than the compact
   model's dynamics (5), are monotone, and stay on the grid.
2. Prices: the aggregate master's LP value equals the full pile-indexed
   master built by brute-force column enumeration (``offline_cl_PB``'s
   test helper, independent of this package); the configuration master's
   bound is at least the aggregate one and at most the optimum.
3. Tables: every cost-to-go value equals a brute-force minimum over all
   module sequences, and the pricing labelling returns the same minimum.
4. Exactness: the optimum equals the compact MILP's proven optimum
   (``offline_cl_opt``) -- one and two piles, two and three connectors,
   FIXED and OPTIMIZE boundary vehicles, objective cohorts, with and without
   beams, with and without deepening -- and the schedule passes every row of
   the compact model.
5. Early stops: memory cuts, an expansion budget and a tolerance still return
   a valid bracket; the bound history is a monotone bracket.
6. Closed system: a horizon too short for every vehicle raises, as the
   compact model reports it.

All instances are small (a few vehicles, at most ~30 slots); the first run
compiles the numba kernels (about a minute), later runs use the cache.

Run: python -m pytest offline_cl_PDS/tests -v
"""

from __future__ import annotations

import itertools
import math
import random

import numpy as np
import pytest

gp = pytest.importorskip("gurobipy")
pytest.importorskip("numba")

from offline_cl_opt.boundary import COHORTS_MEASUREMENT, BoundaryMode, BoundaryVehicle, Cohort
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import build_cl_model, solve_cl_model
from offline_cl_opt.solution import extract_solution
from offline_cl_PB.tests.enumeration import enumerate_columns, solve_full_master

from offline_cl_PDS import build_search_instance, pile_configurations, slot_prices, solve_price_directed_search
from offline_cl_PDS.cost_to_go import INF, build_tables, table_value
from offline_cl_PDS.dynamics import GRID, fastest_trajectory, slot_energy, useful_modules
from offline_cl_PDS.pricing import price_vehicle


def _veh(i, a, Q, s_i, s_f, p_max=60.0, s_th=0.4):
    return VehicleData(id=i, a=a, Q=Q, s_i=s_i, s_f=s_f, s_th=s_th, p_max=p_max)


def _tiny(seed, n, n_piles):
    """10-minute slots, K = 5, piles of 3 x 25 kW (the family offline_cl_PB's tests use)."""
    rng = random.Random(seed)
    vs = [_veh(i, rng.choice([0.0, 0.0, 5.0, 10.0, 20.0]), rng.choice([10.0, 15.0, 20.0]), rng.choice([0.1, 0.2, 0.3]),
               rng.choice([0.7, 0.8, 0.9]), p_max=rng.choice([40.0, 60.0, 90.0])) for i in range(n)]
    return vs, StationSpec(n_piles=n_piles, n_connectors=2, n_modules=3, p_module=25.0), 10.0, 50.0


def _contended(seed, n, n_piles, n_connectors=2):
    """5-minute slots, piles of 4 x 25 kW: vehicles compete for modules."""
    rng = random.Random(seed)
    vs = [_veh(i, float(rng.choice([0, 0, 5, 10, 15, 20, 25])), float(rng.choice([20, 25, 30, 40])),
               rng.choice([0.1, 0.2, 0.3]), rng.choice([0.7, 0.8, 0.9]), p_max=float(rng.choice([50, 75, 100])))
          for i in range(n)]
    return vs, StationSpec(n_piles=n_piles, n_connectors=n_connectors, n_modules=4, p_module=25.0)


def _boundary(n_piles):
    """In-service vehicles at t=0 (FIXED and OPTIMIZE) plus arrivals."""
    station = StationSpec(n_piles=n_piles, n_connectors=2, n_modules=4, p_module=25.0)
    vs = [_veh(10, 0.0, 30.0, 0.3, 0.8), _veh(11, 0.0, 40.0, 0.2, 0.85, p_max=90.0),
          _veh(0, 5.0, 30.0, 0.2, 0.8), _veh(1, 10.0, 25.0, 0.3, 0.9, p_max=90.0), _veh(2, 12.0, 20.0, 0.1, 0.7)]
    boundary = {
        10: BoundaryVehicle(vehicle_id=10, pile=n_piles - 1, connector=1, mode=BoundaryMode.FIXED, departure_slot=3,
                            power={0: 50.0, 1: 45.0, 2: 20.0}),
        11: BoundaryVehicle(vehicle_id=11, pile=0, connector=0, mode=BoundaryMode.OPTIMIZE, initial_energy_kwh=6.0),
    }
    cohorts = {10: Cohort.BOUNDARY, 11: Cohort.BOUNDARY, 0: Cohort.MEASUREMENT, 1: Cohort.MEASUREMENT,
               2: Cohort.QUEUED}
    return vs, station, 5.0, 60.0, boundary, cohorts


def _compact_optimum(vehicles, station, delta, horizon, **kw):
    cl = build_cl_model(vehicles, station, delta, horizon, **kw)
    solve_cl_model(cl, mip_gap=0.0)
    return float(cl.model.ObjVal)


def _assert_optimal(sol, z_star):
    assert sol.status == "OPTIMAL"
    assert sol.objective == pytest.approx(z_star, abs=1e-6)
    assert sol.lower_bound == pytest.approx(z_star, abs=1e-6)
    assert sol.master_bound <= z_star + 1e-6 and sol.root_lower_bound <= z_star + 1e-6
    assert sol.compact_check.max_violation <= 1e-5, sol.compact_check
    assert sol.compact_check.objective == pytest.approx(z_star, abs=1e-6)
    assert sol.table_repairs == 0


# ---------------------------------------------------------------------------
# 1. Dynamics
# ---------------------------------------------------------------------------


def test_grid_dynamics_relax_the_compact_dynamics():
    vs, station = _contended(1, 6, 1)
    inst = build_search_instance(vs, station, 5.0, 120.0)
    h, Delta, N = 5.0 / 60.0, station.p_module, station.n_modules
    rng = np.random.default_rng(0)
    for i, v in enumerate(vs):
        prm = inst.params[i]
        p_bar = min(v.p_max, N * Delta)
        tau_d = v.tau_delta_hours(5.0)
        for x in np.ceil(rng.uniform(0, v.W, 30) / GRID) * GRID:
            prev = x
            for q in range(N + 1):
                e = slot_energy(x, q, inst.kappa, prm)
                exact = min(x + h * min(Delta * q, p_bar, (v.R - x) / tau_d), v.W) if x < v.W else x
                assert e >= exact - 1e-12  # never below (5)
                assert e - exact < 1e-7  # and only by grid rounding
                assert e >= prev  # non-decreasing in q
                assert e == math.ceil(e / GRID) * GRID  # on the grid
                prev = e
            assert slot_energy(x + GRID, N, inst.kappa, prm) >= slot_energy(x, N, inst.kappa, prm)  # in x
            q_hat = useful_modules(x, N, inst.kappa, prm)
            assert slot_energy(x, q_hat, inst.kappa, prm) == slot_energy(x, N, inst.kappa, prm)


def test_pile_configurations():
    got = {tuple(r) for r in pile_configurations(2, 6)}
    expected = {tuple(sum(1 for q in c if q >= t) for t in range(7)) for c in [(0, 6), (1, 5), (2, 4), (3, 3)]}
    assert got == expected


# ---------------------------------------------------------------------------
# 2. Prices
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed,n_piles", [(20, 1), (21, 1), (22, 2)])
def test_aggregate_master_equals_full_enumerated_master(seed, n_piles):
    vs, station, delta, horizon = _tiny(seed, 4, n_piles)
    inst = build_search_instance(vs, station, delta, horizon)
    prices = slot_prices(inst, [], kind="aggregate")
    assert prices.converged
    z_full = solve_full_master(enumerate_columns(vs, station, delta, inst.K), station, inst.K,
                               {v.id: 1.0 for v in vs}, delta=delta, arrivals={v.id: v.a for v in vs}, integer=False)
    assert inst.to_minutes(prices.master_value) == pytest.approx(z_full, abs=1e-5)
    assert inst.to_minutes(prices.lagrangian_bound) == pytest.approx(z_full, abs=1e-5)


@pytest.mark.parametrize("seed,n_piles", [(0, 2), (12, 1)])
def test_configuration_bound_between_aggregate_bound_and_optimum(seed, n_piles):
    vs, station = _contended(seed, 5, n_piles)
    inst = build_search_instance(vs, station, 5.0, 60.0)
    agg = slot_prices(inst, [], kind="aggregate")
    cfg = slot_prices(inst, [], kind="configuration")
    z_star = _compact_optimum(vs, station, 5.0, 60.0)
    assert cfg.aggregate_bound == pytest.approx(agg.lagrangian_bound, abs=1e-6)
    assert agg.lagrangian_bound - 1e-6 <= cfg.lagrangian_bound
    assert inst.to_minutes(cfg.lagrangian_bound) <= z_star + 1e-6


# ---------------------------------------------------------------------------
# 3. Tables and pricing
# ---------------------------------------------------------------------------


def _brute_force_cost_to_go(i, k, x, inst, rent):
    """min over every module sequence from (k, x) of w*D + rent (eq. (31)), by enumeration."""
    K, N = inst.K, inst.station.n_modules
    prm, w = inst.params[i], int(inst.weight[i])
    best = INF
    for length in range(1, K - k + 1):
        for qs in itertools.product(range(N + 1), repeat=length):
            e, cost = x, 0.0
            for t, q in enumerate(qs):
                e = slot_energy(e, q, inst.kappa, prm)
                cost += rent[k + t, q]
                if e >= prm[0]:
                    if t == length - 1:
                        best = min(best, cost + w * (k + length))
                    break
    return best


def test_tables_and_pricing_equal_brute_force():
    station = StationSpec(n_piles=1, n_connectors=2, n_modules=3, p_module=25.0)
    vs = [_veh(0, 0.0, 20.0, 0.2, 0.8, p_max=60.0), _veh(1, 10.0, 15.0, 0.1, 0.9, p_max=90.0)]
    inst = build_search_instance(vs, station, 10.0, 60.0)  # K = 6
    rng = np.random.default_rng(3)
    mu = rng.uniform(0, 0.4, size=(inst.K, 4)) * (rng.random((inst.K, 4)) < 0.6)
    tables = build_tables(inst, mu, pile_configurations(2, 3))
    assert tables.repairs == 0
    for i in range(inst.J):
        for k in range(int(inst.release[i]), inst.K):
            for x in (0.0, 1.0, 5.5, inst.params[i, 0] - 0.5):
                x = math.ceil(x / GRID) * GRID
                bf = _brute_force_cost_to_go(i, k, x, inst, tables.rent)
                assert table_value(tables.off, tables.cnt, tables.bp, tables.val, i, k, x) == pytest.approx(bf, abs=1e-9)
        traj = fastest_trajectory(3, inst.kappa, inst.params[i], inst.K)
        best, *_rest, ok = price_vehicle(inst.release[i], inst.K, False, 0.0, 3, inst.kappa, inst.params[i],
                                         inst.weight[i], traj, tables.rent, INF, 1, 100_000)
        assert ok and best == pytest.approx(tables.U[i, 0], abs=1e-9)


# ---------------------------------------------------------------------------
# 4. Exactness against the compact model
# ---------------------------------------------------------------------------


SEARCH_OPTIONS = [dict(), dict(beam_widths=()), dict(beam_widths=(), deepen=False), dict(master="configuration")]


@pytest.mark.parametrize("seed,n,n_piles", [(20, 3, 1), (23, 5, 2), (28, 5, 1), (31, 5, 2)])
def test_matches_compact_on_tiny_instances(seed, n, n_piles):
    vs, station, delta, horizon = _tiny(seed, n, n_piles)
    z_star = _compact_optimum(vs, station, delta, horizon)
    for opts in SEARCH_OPTIONS:
        _assert_optimal(solve_price_directed_search(vs, station, delta, horizon, **opts), z_star)


@pytest.mark.parametrize("seed,n,n_piles,n_connectors,horizon", [(0, 6, 2, 2, 60.0), (3, 5, 1, 2, 90.0),
                                                                  (12, 5, 1, 2, 60.0), (2, 5, 1, 3, 90.0)])
def test_matches_compact_on_contended_instances(seed, n, n_piles, n_connectors, horizon):
    vs, station = _contended(seed, n, n_piles, n_connectors)
    z_star = _compact_optimum(vs, station, 5.0, horizon)
    for opts in SEARCH_OPTIONS:
        _assert_optimal(solve_price_directed_search(vs, station, 5.0, horizon, **opts), z_star)


@pytest.mark.parametrize("n_piles", [1, 2])
@pytest.mark.parametrize("objective_cohorts", [None, COHORTS_MEASUREMENT])
def test_matches_compact_with_boundary_vehicles_and_cohorts(n_piles, objective_cohorts):
    vs, station, delta, horizon, boundary, cohorts = _boundary(n_piles)
    kw = dict(boundary_vehicles=boundary, cohorts=cohorts, objective_cohorts=objective_cohorts)
    z_star = _compact_optimum(vs, station, delta, horizon, **kw)
    for opts in SEARCH_OPTIONS:
        sol = solve_price_directed_search(vs, station, delta, horizon, **kw, **opts)
        _assert_optimal(sol, z_star)
        assert sol.schedule[10].departure == 3  # FIXED keeps its trajectory
        assert (sol.schedule[11].pile, sol.connectors[11], sol.schedule[11].start) == (0, 0, 0)  # OPTIMIZE keeps its lane
        assert (sol.schedule[10].pile, sol.connectors[10]) == (n_piles - 1, 1)


def test_to_cl_model_is_a_drop_in_compact_model():
    vs, station, delta, horizon, boundary, cohorts = _boundary(1)
    sol = solve_price_directed_search(vs, station, delta, horizon, boundary_vehicles=boundary, cohorts=cohorts)
    ext = extract_solution(sol.to_cl_model())
    assert ext.objective == pytest.approx(sol.objective, abs=1e-6)
    by_id = ext.per_vehicle.set_index("vehicle_id")
    for j, plan in sol.schedule.items():
        assert by_id.loc[j, "departure_slot"] == pytest.approx(plan.departure)
    for _, row in sol.per_vehicle.iterrows():
        assert row["energy_kwh"] + row["energy_delivered_before_kwh"] == pytest.approx(row["energy_required_kwh"], abs=1e-5) \
            or row["boundary_mode"] == BoundaryMode.FIXED.value


# ---------------------------------------------------------------------------
# 5. Early stops
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("opts,allowed", [
    (dict(beam_widths=(), layer_cap=20), {"OPTIMAL", "MEMORY_LIMIT"}),
    (dict(beam_widths=(), max_expand=50), {"OPTIMAL", "EXPANSION_LIMIT"}),
    (dict(beam_widths=(50,), layer_cap=30, max_expand=200), {"OPTIMAL", "MEMORY_LIMIT", "EXPANSION_LIMIT"}),
])
def test_early_stops_return_a_valid_bracket(opts, allowed):
    from offline_cl_PB import greedy_list_schedule

    vs, station = _contended(0, 6, 2)
    # An early stop needs some schedule to report: here the greedy list schedule.
    assert greedy_list_schedule(vs, station, 5.0, 12, {}) is not None
    z_star = _compact_optimum(vs, station, 5.0, 60.0)
    sol = solve_price_directed_search(vs, station, 5.0, 60.0, **opts)
    assert sol.status in allowed
    assert sol.lower_bound <= z_star + 1e-6 <= sol.objective + 1e-6
    assert sol.compact_check.max_violation <= 1e-5
    if sol.status == "OPTIMAL":
        assert sol.objective == pytest.approx(z_star, abs=1e-6)


def test_tolerance_bounds_the_gap():
    vs, station = _contended(3, 5, 1)
    z_star = _compact_optimum(vs, station, 5.0, 90.0)
    tol = 20.0  # minutes of total sojourn = 4 slots of 5 minutes
    sol = solve_price_directed_search(vs, station, 5.0, 90.0, gap_abs=tol, beam_widths=())
    assert sol.status in ("OPTIMAL", "WITHIN_TOLERANCE")
    assert sol.objective - sol.lower_bound <= tol + 1e-6
    assert sol.lower_bound <= z_star + 1e-6 <= sol.objective + 1e-6


def test_bound_history_is_a_monotone_bracket_ending_at_the_result():
    vs, station = _contended(0, 6, 2)
    z_star = _compact_optimum(vs, station, 5.0, 60.0)
    sol = solve_price_directed_search(vs, station, 5.0, 60.0, beam_widths=())
    h = sol.bound_history
    assert h[-1]["event"] == "end"
    for prev, cur in zip(h, h[1:]):
        assert cur["lower_bound"] >= prev["lower_bound"] and cur["upper_bound"] <= prev["upper_bound"]
    for p in h:
        assert p["lower_bound"] <= z_star + 1e-6 <= p["upper_bound"] + 1e-6
    assert (h[-1]["lower_bound"], h[-1]["upper_bound"]) == (sol.lower_bound, sol.objective)


# ---------------------------------------------------------------------------
# 6. Closed system
# ---------------------------------------------------------------------------


def test_horizon_too_short_raises_like_the_compact_model():
    from offline_cl_opt.model import earliest_departures

    station = StationSpec(n_piles=1, n_connectors=1, n_modules=3, p_module=25.0)
    v0, v1 = _veh(0, 0.0, 30.0, 0.2, 0.8), _veh(1, 0.0, 30.0, 0.2, 0.8)
    E = earliest_departures([v0], station, 5.0, 500.0)[0]
    horizon = 5.0 * (E + 1)
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_cl_model(build_cl_model([v0, v1], station, 5.0, horizon), mip_gap=0.0)
    with pytest.raises(RuntimeError, match="(?i)no schedule serves"):
        solve_price_directed_search([v0, v1], station, 5.0, horizon)
