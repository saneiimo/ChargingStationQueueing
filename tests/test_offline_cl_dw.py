"""
Tests for the Dantzig-Wolfe decomposition (``offline_cl_dw`` package).

Validates:
  1. Column mechanics (alpha/beta, the null plan) match their definitions
     (22)-(23) directly.
  2. The pricer alone reproduces ``earliest_departures`` (E_j) when duals
     are zero -- both are the same "charge alone at the limit" recursion.
  3. The master's dual-sign convention holds (the source document's own
     "Sign convention" warning box).
  4. Column generation converges (``z_RMP - best_LB -> 0``) on a trivial
     single-vehicle instance, and the resulting schedule is genuinely
     feasible (in particular, delivers exactly W_j, not more -- the exact
     bug this project's own development hit and fixed in
     ``preprocess._greedy_solo_plan``).
  5. End to end: on the exact instance ``tests/test_connector_lane_optimization.py``
     already established has true optimum 118.0 (compact model, whole
     modules), the decomposition's own certified bracket
     ``[lower_bound, upper_bound]`` contains 118.0, and the returned
     schedule passes ``postprocess.validate_schedule`` in full.
  6. Deduplication (Section 8.3) actually prevents an identical column
     from being added twice.

Run: python -m pytest tests/test_offline_cl_dw.py -s -v
"""

from __future__ import annotations

import pytest

gp = pytest.importorskip("gurobipy")

from offline_cl_dw.columns import Plan, null_plan
from offline_cl_dw.master import add_column, build_master, solve_lp
from offline_cl_dw.postprocess import validate_schedule, whole_module_failures
from offline_cl_dw.pricer import build_pricer, price
from offline_cl_dw.preprocess import earliest_departures, seed_columns
from offline_cl_dw.solution import solve_by_decomposition
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import build_cl_model, solve_cl_model


def _toy_vehicle(vid: int, a: float, Q: float, s_i: float, s_f: float, p_max: float = 100.0) -> VehicleData:
    return VehicleData(id=vid, a=a, Q=Q, s_i=s_i, s_f=s_f, s_th=0.4, p_max=p_max)


# ---------------------------------------------------------------------------
# 1. Column mechanics
# ---------------------------------------------------------------------------


def test_plan_alpha_beta_match_definitions():
    plan = Plan(vehicle_id=0, pile=1, start=5, departure=8, power={5: 10.0, 6: 20.0, 7: 15.0})

    # (22): alpha is 1 only on this plan's own pile, within [S, D).
    assert plan.alpha(1, 5) == 1
    assert plan.alpha(1, 7) == 1
    assert plan.alpha(1, 8) == 0  # D itself is excluded (half-open)
    assert plan.alpha(1, 4) == 0  # before S
    assert plan.alpha(0, 5) == 0  # wrong pile

    # (23): beta is the power on the plan's own pile, 0 elsewhere/idle.
    assert plan.beta(1, 6) == 20.0
    assert plan.beta(0, 6) == 0.0
    assert plan.beta(1, 8) == 0.0

    null = null_plan(vehicle_id=0, K=100)
    assert null.is_null
    assert null.alpha(0, 0) == 0
    assert null.beta(0, 0) == 0.0
    assert null.departure == 100


# ---------------------------------------------------------------------------
# 2. Pricer alone reproduces E_j at zero duals
# ---------------------------------------------------------------------------


def test_pricer_at_zero_duals_matches_earliest_departure():
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)
    v = _toy_vehicle(0, 0.0, 50.0, 0.2, 0.8)
    delta, horizon = 1.0, 90.0
    K = 90

    E = earliest_departures([v], station, delta, horizon)
    pricer = build_pricer(v, 0, station, delta, K, E[v.id])
    zeta, plan = price(pricer, {}, {})  # no duals -> pure D_j minimisation

    assert not plan.is_null
    assert plan.departure == E[v.id]
    assert zeta == pytest.approx(E[v.id])


# ---------------------------------------------------------------------------
# 3. Master dual-sign convention
# ---------------------------------------------------------------------------


def test_master_dual_signs_are_non_positive():
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)
    v0 = _toy_vehicle(0, 0.0, 50.0, 0.2, 0.8)
    v1 = _toy_vehicle(1, 0.0, 50.0, 0.2, 0.8)
    delta, horizon, K = 1.0, 90.0, 90

    seeds = seed_columns([v0, v1], station, delta, K)
    rm = build_master([v0.id, v1.id], station, delta, K, 0, seeds)
    lp = solve_lp(rm)  # raises AssertionError internally if signs are wrong
    assert all(val <= 1e-9 for val in lp.pi.values())
    assert all(val <= 1e-9 for val in lp.mu.values())


# ---------------------------------------------------------------------------
# 4. Column generation converges and the schedule is genuinely feasible
# ---------------------------------------------------------------------------


def test_single_vehicle_converges_and_delivers_exact_energy():
    """Regression test for the exact bug this project's own development
    caught: preprocess._greedy_solo_plan's seed column could deliver
    *more* than W_j in its final slot (the taper cap alone doesn't stop
    that -- it's capped by distance to a *full* battery, R_j, not to the
    requested W_j)."""
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)
    v = _toy_vehicle(0, 0.0, 50.0, 0.2, 0.8)  # s_f=0.8 < 1 -> R_j > W_j, so this can trigger

    solution, cg = solve_by_decomposition(
        [v], station, delta=1.0, horizon_minutes=60.0, gap_tolerance=1e-4, progress=False
    )
    assert solution.gap == pytest.approx(0.0, abs=1e-3)
    assert solution.lower_bound == pytest.approx(solution.upper_bound, abs=1e-3)
    row = solution.per_vehicle.iloc[0]
    assert row["served"]
    assert row["energy_kwh"] == pytest.approx(v.W, abs=1e-4)
    assert row["energy_kwh"] <= v.W + 1e-6  # never *more* than requested


# ---------------------------------------------------------------------------
# 5. End to end: bracket contains the compact model's known true optimum
# ---------------------------------------------------------------------------


def _known_optimum_instance():
    """Identical to tests/test_connector_lane_optimization.py's own
    _multi_pile_test_instance -- verified there (mip_gap=1e-6) to have true
    optimum 118.0 for the compact, whole-module model."""
    v0 = _toy_vehicle(0, 0.0, 50.0, 0.2, 0.8, p_max=100.0)
    v1 = _toy_vehicle(1, 0.0, 100.0, 0.15, 0.85, p_max=200.0)
    v2 = _toy_vehicle(2, 5.0, 50.0, 0.2, 0.8, p_max=100.0)
    station = StationSpec(n_piles=2, n_connectors=2, n_modules=4, p_module=25.0)
    return [v0, v1, v2], station, 1.0, 100.0


def test_bracket_contains_known_compact_model_optimum():
    vehicles, station, delta, horizon = _known_optimum_instance()

    # Independently re-confirm the compact model's own optimum on this
    # instance (rather than hard-coding 118.0 twice across two test files).
    exact = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(exact, mip_gap=1e-6)
    true_optimum = float(exact.model.ObjVal)

    solution, cg = solve_by_decomposition(
        vehicles, station, delta, horizon, max_iterations=200, progress=False
    )

    print(
        f"\nDW bracket=[{solution.lower_bound:.3f}, {solution.upper_bound:.3f}], "
        f"compact model true optimum={true_optimum:.3f}"
    )
    # (32)/Section 5's own inequality chain: the DW lower bound is a
    # relaxation (continuous modules) of the compact model, so it can only
    # be <= the compact model's true (whole-module) optimum.
    assert solution.lower_bound <= true_optimum + 1e-4
    # price-and-branch's result is a genuine feasible schedule for the
    # compact model too (whole-module feasible, by construction or repair),
    # so it can only be >= the true optimum.
    assert solution.upper_bound >= true_optimum - 1e-4
    assert solution.whole_module_feasible


def test_validate_schedule_passes_on_price_and_branch_result():
    vehicles, station, delta, horizon = _known_optimum_instance()
    solution, cg = solve_by_decomposition(
        vehicles, station, delta, horizon, max_iterations=200, validate=True, progress=False
    )
    # solve_by_decomposition(validate=True) already ran
    # postprocess.validate_schedule internally and would have raised if
    # anything were wrong; re-run it here explicitly too so this test is
    # self-contained and would fail directly (not via an import-time side
    # effect) if that ever regresses.
    integer_result_chosen = {
        j: next(plan for plan, var in plans if var.X > 0.5) for j, plans in cg.master.columns.items()
    }
    validate_schedule(
        integer_result_chosen,
        {v.id: v for v in vehicles},
        station,
        delta,
        solution.K,
        best_lower_bound=solution.lower_bound,
    )
    assert not whole_module_failures(integer_result_chosen, station)


# ---------------------------------------------------------------------------
# 6. Deduplication
# ---------------------------------------------------------------------------


def test_add_column_deduplicates_identical_plans():
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)
    v = _toy_vehicle(0, 0.0, 50.0, 0.2, 0.8)
    K = 90
    seeds = seed_columns([v], station, 1.0, K)
    rm = build_master([v.id], station, 1.0, K, 0, seeds)

    n_before = len(rm.columns[v.id])
    solo_plan = next(p for p in seeds[v.id] if not p.is_null)
    result = add_column(rm, Plan(vehicle_id=v.id, pile=solo_plan.pile, start=solo_plan.start,
                                  departure=solo_plan.departure, power=dict(solo_plan.power)))
    assert result is None  # identical to an already-seeded column
    assert len(rm.columns[v.id]) == n_before  # nothing was actually added


if __name__ == "__main__":
    print("Running test_offline_cl_dw.py (direct mode)")
    test_plan_alpha_beta_match_definitions()
    test_pricer_at_zero_duals_matches_earliest_departure()
    test_master_dual_signs_are_non_positive()
    test_single_vehicle_converges_and_delivers_exact_energy()
    test_bracket_contains_known_compact_model_optimum()
    test_validate_schedule_passes_on_price_and_branch_result()
    test_add_column_deduplicates_identical_plans()
    print("\nAll tests in test_offline_cl_dw.py finished.")
