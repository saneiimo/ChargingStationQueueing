"""
The release-slot objective, the free tightening, and optimality tolerances
(small instances only).

  1. Sojourn is measured from the release slot k_j = ceil(a_j / delta), so
     with off-grid arrivals every objective value is still a multiple of
     delta, in the compact model and in branch-and-price alike.
  2. Free tightening: lattice_round_up / lattice_abs_gap, and a proven
     optimum's rounded bound equals its objective.
  3. Tolerances: the exact model and B&P stop with a schedule proven within
     the tolerance -- never further from the true optimum than allowed --
     and B&P reports WITHIN_TOLERANCE when it stopped short.
  4. The objective sweep wires both tolerances through to its columns.

    python -m pytest tests/test_tolerance.py -v
"""

from __future__ import annotations

import math
import random

import pytest

gp = pytest.importorskip("gurobipy")

from offline_cl_opt import (
    StationSpec,
    VehicleData,
    build_cl_model,
    lattice_abs_gap,
    lattice_round_up,
    sojourn_minutes,
    solve_cl_model,
)
from offline_cl_opt.solution import extract_solution
from offline_cl_PB import solve_branch_and_price


def _veh(i, a, Q, s_i, s_f, p_max=60.0):
    return VehicleData(id=i, a=a, Q=Q, s_i=s_i, s_f=s_f, s_th=0.4, p_max=p_max)


def _contended(seed, n_vehicles, n_piles=1, offgrid=False):
    """5-minute slots, 4 x 25 kW per pile, vehicles fighting over modules."""
    rng = random.Random(seed)
    vehicles = []
    for i in range(n_vehicles):
        a = float(rng.choice([0, 0, 5, 10, 15, 20, 25])) + (rng.choice([0.4, 1.7, 3.1]) if offgrid else 0.0)
        vehicles.append(_veh(i, a, float(rng.choice([20, 25, 30, 40])), rng.choice([0.1, 0.2, 0.3]),
                             rng.choice([0.7, 0.8, 0.9]), p_max=float(rng.choice([50, 75, 100]))))
    return vehicles, StationSpec(n_piles=n_piles, n_connectors=2, n_modules=4, p_module=25.0)


def _optimum(vehicles, station, delta, horizon):
    cl = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(cl, mip_gap=0.0)
    return extract_solution(cl)


# ---------------------------------------------------------------------------
# 1. Release-slot objective
# ---------------------------------------------------------------------------


def test_sojourn_is_measured_from_the_release_slot():
    assert sojourn_minutes(4, 7.3, 2.0) == pytest.approx(2.0 * (4 - 4))  # k = ceil(7.3/2) = 4
    assert sojourn_minutes(9, 6.0, 2.0) == pytest.approx(2.0 * (9 - 3))  # on the grid: raw formula


def test_offgrid_arrivals_keep_the_objective_on_the_delta_lattice():
    vehicles, station = _contended(12, 5, offgrid=True)
    sol = _optimum(vehicles, station, 5.0, 90.0)
    expected = sum(5.0 * (row.departure_slot - math.ceil(row.arrival / 5.0)) for row in sol.per_vehicle.itertuples())
    assert sol.objective == pytest.approx(expected)
    assert sol.objective / 5.0 == pytest.approx(round(sol.objective / 5.0), abs=1e-6)
    bp = solve_branch_and_price(vehicles, station, 5.0, 90.0, rim_time_limit=5.0)
    assert bp.status == "OPTIMAL" and bp.objective == pytest.approx(sol.objective)


# ---------------------------------------------------------------------------
# 2. Free tightening
# ---------------------------------------------------------------------------


def test_lattice_helpers():
    assert lattice_round_up(203.1, 2.0) == pytest.approx(204.0)
    assert lattice_round_up(204.0000001, 2.0) == pytest.approx(204.0)  # noise does not cost a step
    assert lattice_round_up(-math.inf, 2.0) == -math.inf
    # No tolerance: stop once the raw gap is under one step.
    assert lattice_abs_gap(None, 2.0) == pytest.approx(2.0, rel=1e-5) and lattice_abs_gap(None, 2.0) < 2.0
    # 5 minutes of tolerance = 2 whole steps of 2 minutes; stop under 3 steps.
    assert lattice_abs_gap(5.0, 2.0) == pytest.approx(6.0, rel=1e-5) and lattice_abs_gap(5.0, 2.0) < 6.0


def test_proven_optimum_has_its_rounded_bound_at_the_objective():
    vehicles, station = _contended(4, 4)
    sol = _optimum(vehicles, station, 5.0, 60.0)
    assert sol.status == "OPTIMAL"
    assert sol.best_bound == pytest.approx(sol.objective)


# ---------------------------------------------------------------------------
# 3. Tolerances
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tol_steps", [1, 3])
def test_exact_tolerance_is_never_exceeded(tol_steps):
    vehicles, station = _contended(0, 5)
    z_star = _optimum(vehicles, station, 5.0, 60.0).objective
    tol = 5.0 * tol_steps
    cl = build_cl_model(vehicles, station, 5.0, 60.0)
    solve_cl_model(cl, mip_gap=0.0, tolerance=tol)
    sol = extract_solution(cl)
    assert sol.objective - sol.best_bound <= tol + 1e-6
    assert sol.best_bound <= z_star + 1e-6
    assert sol.objective <= z_star + tol + 1e-6


@pytest.mark.parametrize("seed,n_vehicles,horizon", [(12, 5, 60.0), (0, 6, 60.0)])
def test_bp_tolerance_is_never_exceeded(seed, n_vehicles, horizon):
    n_piles = 2 if n_vehicles == 6 else 1
    vehicles, station = _contended(seed, n_vehicles, n_piles)
    z_star = _optimum(vehicles, station, 5.0, horizon).objective
    exact = solve_branch_and_price(vehicles, station, 5.0, horizon, rim_time_limit=0.0, dive_every=None)
    tol = 10.0  # two slots of total sojourn
    loose = solve_branch_and_price(vehicles, station, 5.0, horizon, rim_time_limit=0.0, dive_every=None,
                                   gap_abs=tol)
    assert exact.status == "OPTIMAL" and exact.objective == pytest.approx(z_star)
    assert loose.status in ("OPTIMAL", "WITHIN_TOLERANCE")
    assert loose.objective - loose.lower_bound <= tol + 1e-6
    assert loose.lower_bound <= z_star + 1e-6 <= loose.objective + 1e-6
    assert loose.objective <= z_star + tol + 1e-6
    assert loose.nodes_processed <= exact.nodes_processed
    assert loose.lower_bound / 5.0 == pytest.approx(round(loose.lower_bound / 5.0), abs=1e-6)  # rounded


def test_bp_reports_within_tolerance_when_it_stops_short():
    """A tolerance as wide as the seed's own gap closes at the root with a
    bracket wider than zero."""
    vehicles, station = _contended(0, 6, 2)
    z_star = _optimum(vehicles, station, 5.0, 60.0).objective
    sol = solve_branch_and_price(vehicles, station, 5.0, 60.0, rim_time_limit=0.0, dive_every=None,
                                 gap_abs=1000.0)
    assert sol.status in ("WITHIN_TOLERANCE", "OPTIMAL")
    assert sol.lower_bound <= z_star + 1e-6 <= sol.objective + 1e-6
    if sol.objective > sol.lower_bound + 1e-6:
        assert sol.status == "WITHIN_TOLERANCE"
    h = sol.bound_history
    assert h[-1]["lower_bound"] == sol.lower_bound and h[-1]["upper_bound"] == sol.objective


# ---------------------------------------------------------------------------
# 4. The objective sweep
# ---------------------------------------------------------------------------


def test_sweep_reports_tolerances_and_rounded_bounds():
    from experiments.objective_sweep import TrialConfig, flatten_trial, run_trial

    cfg = TrialConfig(
        n_piles=1, n_connectors=2, n_modules=6, max_time=60, warmup_period=120, delta=5.0, seed=41,
        battery_cap_kwh=(75.0,), mean_interarrival=15.0, time_limit=120.0, dw_time_limit=120.0,
        bp_time_limit=120.0, gap_tolerance_target_min=None, exact_tolerance_min=2.0, bp_tolerance_min=2.0,
    )
    result = run_trial(cfg, run_exact_model=True, run_bp_model=True, run_dw_model=True, verbose=False)
    row = flatten_trial(result, 0)
    n = row["exact_n_optimized"]
    for stage in ("exact", "bp"):
        assert row[f"{stage}_tolerance_used"] == pytest.approx(2.0 * n)
        assert row[f"{stage}_within_tolerance"]
        assert row[f"{stage}_mean_gap_min"] <= 2.0 + 1e-9
        assert row[f"{stage}_best_bound"] / 5.0 == pytest.approx(round(row[f"{stage}_best_bound"] / 5.0), abs=1e-6)
    assert row["exact_best_bound"] >= row["exact_best_bound_raw"] - 1e-6
    assert row["dw_LB"] >= row["dw_LB_raw"] - 1e-9
    assert row["dw_LB"] <= min(row["exact_objective"], row["bp_objective"]) + 1e-6
    # The two exact solvers bracket the same optimum.
    assert max(row["exact_best_bound"], row["bp_best_bound"]) <= min(row["exact_objective"], row["bp_objective"]) + 1e-6
    with pytest.raises(ValueError, match="tie_break"):
        TrialConfig(exact_tolerance_min=1.0, tie_break=True)
