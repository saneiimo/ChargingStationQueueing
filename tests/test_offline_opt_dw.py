"""
Tests for the Dantzig-Wolfe decomposition (``offline_opt_dw`` package),
implementing ``decomposition_implementation_spec.md``.

Validates:
  1. Column mechanics (score/zeta/modules, the null plan, the "plugs in but
     never completes" censoring case) match their definitions in Section 4.2
     directly.
  2. The pricer alone reproduces ``earliest_departures``/``isolated_plan``
     (E_j) when duals are zero -- both are the same "charge alone at the
     limit" recursion (25).
  3. The master's dual-sign convention holds (rows (28)-(30) are ``<=`` in a
     *maximisation*, so duals must be non-negative -- Section 4.3).
  4. Single vehicle, empty station: Validation plan item 2 -- the optimum
     must equal ``K - E_j`` exactly, and energy delivered must equal
     ``W_j``, never more.
  5. End to end: on a small congested instance, the decomposition's own
     certified bracket ``[Z_lb, Z_ub]`` matches the monolithic model's exact
     optimum (``offline_opt.model``) -- validation plan item 3, "Monolith
     agreement", the primary correctness test.
  6. A congested-but-infeasible-to-finish instance (no vehicle can complete
     within the horizon): the bracket still matches the monolithic model's
     Z=0, and ``validate_schedule`` accepts the resulting censored plan
     (Section 4.2's "plugs in but never completes" case) without complaint.
  7. Step P5's congestion-free early exit fires exactly when
     ``preprocess.congestion_free_bound`` says it should, and returns the
     exact closed form ``sum_j (K - E_j)``.
  8. Deduplication actually prevents an identical column from being added
     twice.

Run: python -m pytest tests/test_offline_opt_dw.py -s -v
"""

from __future__ import annotations

import pytest

gp = pytest.importorskip("gurobipy")

from offline_opt import (
    StationSpec,
    VehicleData,
    build_offline_model,
    discrete_taper_time_constant,
    solve_offline_model,
    taper_time_constant,
)
from offline_opt.solution import extract_solution as extract_monolithic
from offline_opt_dw.columns import Plan, null_plan
from offline_opt_dw.master import add_column, build_master, solve_lp
from offline_opt_dw.pricer import build_pricer, price
from offline_opt_dw.preprocess import (
    active_counts,
    congestion_free_bound,
    earliest_departures,
    isolated_plan,
    saturation_rhs,
    seed_columns,
)
from offline_opt_dw.preprocess import fifo_reference_schedule
from offline_opt_dw.solution import solve_by_decomposition

TAU = taper_time_constant()


def _v(vid: int, a: float, Q_kwh: float, s_i: float, s_f: float, p_max: float = 100.0) -> VehicleData:
    return VehicleData(id=vid, a=a, Q=Q_kwh * 60.0, s_i=s_i, s_f=s_f, p_max=p_max)


def _z_from_mono(mono, vehicles, delta, K) -> float:
    J = len(vehicles)
    sum_a = sum(v.a for v in vehicles)
    return J * K - (mono.total_sojourn + sum_a) / delta


# ---------------------------------------------------------------------------
# 1. Column mechanics
# ---------------------------------------------------------------------------


def test_plan_score_zeta_modules_match_definitions():
    K = 20
    plan = Plan(vehicle_id=0, pile=1, start=5, departure=8, nu={5: 2, 6: 3, 7: 1}, power={5: 40.0, 6: 60.0, 7: 20.0})

    assert plan.score(K) == K - 8
    assert plan.zeta(5) == 1 and plan.zeta(7) == 1
    assert plan.zeta(8) == 0  # departure itself excluded (half-open)
    assert plan.zeta(4) == 0  # before start
    assert plan.modules(6) == 3
    assert plan.modules(4) == 0
    assert list(plan.occupied_slots()) == [5, 6, 7]

    null = null_plan(vehicle_id=0, K=K)
    assert null.is_null
    assert null.score(K) == 0
    assert null.zeta(0) == 0
    assert list(null.occupied_slots()) == []


def test_censored_plan_occupies_through_horizon_but_scores_zero():
    """A plan that plugs in but never completes (departure sentinel == K)
    still occupies its connector/modules through slot K-1 -- constraint (4)
    gives alpha no 'abandon without finishing' transition. This is the trap
    the module docstring calls out explicitly."""
    K = 10
    plan = Plan(vehicle_id=0, pile=0, start=3, departure=K, nu={k: 1 for k in range(3, K)}, power={})

    assert plan.score(K) == 0
    assert list(plan.occupied_slots()) == list(range(3, K))
    for k in range(3, K):
        assert plan.zeta(k) == 1
    assert not plan.completes_by(K - 1, K)


# ---------------------------------------------------------------------------
# 2. Pricer at zero duals reproduces the isolated/earliest-departure plan
# ---------------------------------------------------------------------------


def test_pricer_at_zero_duals_matches_isolated_plan():
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=8, p_module=25.0)
    v = _v(0, 0.0, 50.0, 0.2, 0.8)
    delta, K = 1.0, 90
    tau_delta = discrete_taper_time_constant(TAU, delta)

    E = earliest_departures([v], station, delta, K, tau_delta)
    pricer = build_pricer(v, 0, station, delta, K, tau_delta)
    phi, plan = price(pricer, {}, {}, {})  # zero duals -> pure score maximisation

    assert not plan.is_null
    assert plan.departure == E[v.id]
    assert phi == pytest.approx(K - E[v.id])

    iso = isolated_plan(v, 0, station, delta, K, tau_delta)
    assert iso.departure == plan.departure
    assert iso.energy(delta) == pytest.approx(plan.energy(delta), abs=1e-6)


# ---------------------------------------------------------------------------
# 3. Master dual-sign convention (maximisation: gamma, mu, eta >= 0)
# ---------------------------------------------------------------------------


def test_master_dual_signs_are_non_negative():
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=8, p_module=25.0)
    v0, v1 = _v(0, 0.0, 50.0, 0.2, 0.8), _v(1, 0.0, 50.0, 0.2, 0.8)
    delta, K = 1.0, 90
    tau_delta = discrete_taper_time_constant(TAU, delta)

    fifo = fifo_reference_schedule([v0, v1], station, delta, K, tau_delta)
    seeds = seed_columns([v0, v1], station, delta, K, tau_delta, fifo)
    active = active_counts([v0, v1], delta, K)
    B, k_sat = saturation_rhs([v0, v1], station, delta, K)
    rm = build_master([v0, v1], station, delta, K, active, B, k_sat, seeds)
    lp = solve_lp(rm)  # raises AssertionError internally if signs are wrong
    assert all(val >= -1e-9 for val in lp.gamma.values())
    assert all(val >= -1e-9 for val in lp.mu.values())
    assert all(val >= -1e-9 for val in lp.eta.values())


# ---------------------------------------------------------------------------
# 4. Single vehicle, empty station: optimum == K - E_j, energy == W_j exactly
# ---------------------------------------------------------------------------


def test_single_vehicle_matches_earliest_departure_and_delivers_exact_energy():
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=8, p_module=25.0)
    v = _v(0, 0.0, 50.0, 0.2, 0.8)  # s_f=0.8 < 1 -> R_j > W_j: taper cap alone doesn't stop overshoot

    sol, cg = solve_by_decomposition([v], station, delta=1.0, horizon_minutes=60.0, tau=TAU, progress=False)
    assert sol.status == "congestion_free"
    assert sol.Z_lb == sol.Z_ub
    row = sol.per_vehicle.iloc[0]
    assert row["served"]
    assert row["energy_kwh"] == pytest.approx(v.W / 60.0, abs=1e-4)
    assert row["energy_kwh"] <= v.W / 60.0 + 1e-6  # never *more* than requested


# ---------------------------------------------------------------------------
# 5. End to end: bracket matches the monolithic model's exact optimum
# ---------------------------------------------------------------------------


def _congested_instance():
    vehicles = [
        _v(0, 0.0, 50.0, 0.2, 0.8, p_max=100.0),
        _v(1, 0.0, 100.0, 0.15, 0.85, p_max=200.0),
        _v(2, 5.0, 50.0, 0.2, 0.8, p_max=100.0),
        _v(3, 8.0, 60.0, 0.3, 0.9, p_max=120.0),
    ]
    station = StationSpec(n_piles=2, n_dispensers=1, n_modules=4, p_module=25.0)
    return vehicles, station, 1.0, 100.0


def test_bracket_matches_monolithic_optimum_on_congested_instance():
    vehicles, station, delta, horizon = _congested_instance()

    om = build_offline_model(vehicles, station, delta, horizon, TAU, break_pile_symmetry=False)
    solve_offline_model(om, mip_gap=1e-6)
    mono = extract_monolithic(om)
    z_mono = _z_from_mono(mono, vehicles, delta, om.K)

    sol, cg = solve_by_decomposition(
        vehicles, station, delta=delta, horizon_minutes=horizon, tau=TAU,
        gap_tol=0.99, max_iterations=200, progress=False,
    )
    assert sol.Z_lb - 1e-6 <= z_mono <= sol.Z_ub + 1e-6
    assert sol.status == "optimal"
    assert sol.Z_lb == pytest.approx(z_mono, abs=1e-6)


# ---------------------------------------------------------------------------
# 6. Congested + nobody can finish: bracket matches monolithic Z=0, censored
#    plan validates cleanly.
# ---------------------------------------------------------------------------


def test_bracket_matches_monolithic_when_nobody_can_finish():
    vehicles = [_v(0, 0.0, 100.0, 0.1, 0.9, p_max=50.0), _v(1, 1.0, 100.0, 0.1, 0.9, p_max=50.0)]
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=2, p_module=25.0)
    delta, horizon = 1.0, 40.0

    om = build_offline_model(vehicles, station, delta, horizon, TAU, break_pile_symmetry=False)
    solve_offline_model(om, mip_gap=1e-6)
    mono = extract_monolithic(om)
    assert not mono.per_vehicle["finished"].any()
    z_mono = _z_from_mono(mono, vehicles, delta, om.K)
    assert z_mono == pytest.approx(0.0, abs=1e-6)

    sol, cg = solve_by_decomposition(
        vehicles, station, delta=delta, horizon_minutes=horizon, tau=TAU, progress=False,
    )
    assert sol.Z_lb == pytest.approx(0.0, abs=1e-6)
    assert sol.Z_ub == pytest.approx(0.0, abs=1e-6)
    # validate_schedule already ran inside solve_by_decomposition (validate=True
    # by default); reaching here without an AssertionError is the real check.


# ---------------------------------------------------------------------------
# 7. Congestion-free early exit (Step P5)
# ---------------------------------------------------------------------------


def test_congestion_free_bound_matches_closed_form_when_uncongested():
    station = StationSpec(n_piles=3, n_dispensers=1, n_modules=8, p_module=25.0)
    vehicles = [_v(0, 0.0, 50.0, 0.2, 0.8), _v(1, 50.0, 50.0, 0.2, 0.8), _v(2, 100.0, 50.0, 0.2, 0.8)]
    delta, K = 1.0, 200
    tau_delta = discrete_taper_time_constant(TAU, delta)

    E = earliest_departures(vehicles, station, delta, K, tau_delta)
    z_star = congestion_free_bound(vehicles, station, delta, K, tau_delta)
    assert z_star is not None
    assert z_star == pytest.approx(sum(K - E[v.id] for v in vehicles))

    sol, cg = solve_by_decomposition(vehicles, station, delta=delta, horizon_minutes=K * delta, tau=TAU)
    assert sol.status == "congestion_free"
    assert cg is None
    assert sol.Z_lb == pytest.approx(z_star)


def test_congestion_free_bound_is_none_when_congested():
    vehicles, station, delta, horizon = _congested_instance()
    import math

    K = math.ceil(round(horizon / delta, 9))
    tau_delta = discrete_taper_time_constant(TAU, delta)
    assert congestion_free_bound(vehicles, station, delta, K, tau_delta) is None


# ---------------------------------------------------------------------------
# 8. Deduplication
# ---------------------------------------------------------------------------


def test_add_column_deduplicates():
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=8, p_module=25.0)
    v = _v(0, 0.0, 50.0, 0.2, 0.8)
    delta, K = 1.0, 90
    tau_delta = discrete_taper_time_constant(TAU, delta)

    fifo = fifo_reference_schedule([v], station, delta, K, tau_delta)
    seeds = seed_columns([v], station, delta, K, tau_delta, fifo)
    active = active_counts([v], delta, K)
    B, k_sat = saturation_rhs([v], station, delta, K)
    rm = build_master([v], station, delta, K, active, B, k_sat, seeds)

    before = len(rm.columns[v.id])
    iso = isolated_plan(v, 0, station, delta, K, tau_delta)
    result = add_column(rm, iso)  # already present via seed_columns
    assert result is None
    assert len(rm.columns[v.id]) == before
