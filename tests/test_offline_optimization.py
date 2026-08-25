"""
Tests for the offline lower-bound MILP (``offline_opt`` package).

Validates the model against:
  1. The shared taper time constant tau matches the BMS curve exactly
     (constraints 18/19 are wired correctly).
  2. Closed-form single-vehicle charge time (theory) -- no contention.
  3. The "no contention" case with two vehicles, one pile each.
  4. Energy delivered never exceeds W_j (17); finished vehicles meet W_j.
  5. The central sanity check: the offline optimum must never exceed what a
     causal (FIFO) simulation achieves on the same instance.
  6. A short horizon stays feasible (no finish-by-horizon constraint) with
     unfinished vehicles contributing sojourn through T.

Run: python -m pytest tests/test_offline_optimization.py -s -v
"""

from __future__ import annotations

import pytest

gp = pytest.importorskip("gurobipy")

from config import C_RATE, HR2MIN, S_THRESH
from models.ev import EV
from offline_opt import (
    StationSpec,
    VehicleData,
    build_offline_model,
    compute_ip_bounds,
    compute_offline_bound,
    full_power_time,
    solve_offline_model,
    taper_time_constant,
    vehicles_from_evs,
)
from offline_opt.solution import extract_solution
from toy_demo.runner import run_toy_episode
from toy_demo.scenario import ToyEVSpec, ToyStationSpec, build_evs


def _make_ev(id: int, arrival: float, battery_kwh: float, s_i: float, s_f: float) -> EV:
    return EV(id=id, c_b=battery_kwh * HR2MIN, s_i=s_i, s_f=s_f, arrival_time=arrival)


TAU = taper_time_constant()  # shared across every test, matches config defaults


# ---------------------------------------------------------------------------
# 1. Taper time constant
# ---------------------------------------------------------------------------


def test_tau_matches_taper_formula():
    """tau = (1 - s_th) / c_rate; should reproduce EV.p_req exactly at any
    SoC past the knee: p_req(s) == (R0 - x(s)) / tau."""
    assert TAU == pytest.approx((1.0 - S_THRESH) / C_RATE)

    ev = _make_ev(0, 0.0, 75.0, 0.2, 0.95)
    v = VehicleData.from_ev(ev)
    for s in (0.4, 0.6, 0.8, 0.9):
        ev.s_current = s
        r_remaining = v.Q * (1.0 - s)  # R(s), internal units
        assert ev.p_req == pytest.approx(r_remaining / TAU, rel=1e-9)


# ---------------------------------------------------------------------------
# 2. Single vehicle matches closed-form theory
# ---------------------------------------------------------------------------


def test_single_ev_matches_theory():
    ev = _make_ev(0, 0.0, 75.0, 0.2, 0.8)
    v = VehicleData.from_ev(ev)
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=8, p_module=25.0)  # 200 kW >> p_max
    delta = 0.5
    t_min = full_power_time(v, S_THRESH)
    horizon = t_min + 20.0

    om = build_offline_model([v], station, delta, horizon, TAU)
    solve_offline_model(om, mip_gap=1e-6)
    sol = extract_solution(om)

    assert sol.n_vehicles == 1
    row = sol.per_vehicle.iloc[0]
    print(f"\nsingle EV: T_min={t_min:.3f}, MILP sojourn={row['sojourn']:.3f}")

    # Up to ~1 slot of unavoidable release slack (mid-slot arrival) plus
    # completion rounding up to a slot boundary.
    assert row["sojourn"] == pytest.approx(t_min, abs=2 * delta + 1e-6)
    assert row["finished"]
    assert row["energy_kwh"] == pytest.approx(v.W_kwh, rel=1e-6)
    assert sol.objective == pytest.approx(sol.total_sojourn, abs=1e-4)


# ---------------------------------------------------------------------------
# 3. No contention: identical EVs, one pile each -> both match solo theory
# ---------------------------------------------------------------------------


def test_two_piles_no_contention_matches_independent_theory():
    ev0 = _make_ev(0, 0.0, 50.0, 0.2, 0.8)
    ev1 = _make_ev(1, 0.0, 50.0, 0.2, 0.8)
    vehicles = vehicles_from_evs([ev0, ev1])
    station = StationSpec(n_piles=2, n_dispensers=1, n_modules=8, p_module=25.0)
    delta = 0.5
    t_min = full_power_time(vehicles[0], S_THRESH)
    horizon = t_min + 20.0

    om = build_offline_model(vehicles, station, delta, horizon, TAU)
    solve_offline_model(om, mip_gap=1e-6)
    sol = extract_solution(om)

    assert sol.per_vehicle["pile"].nunique() == 2  # each took its own pile
    for _, row in sol.per_vehicle.iterrows():
        assert row["sojourn"] == pytest.approx(t_min, abs=2 * delta + 1e-6)


# ---------------------------------------------------------------------------
# 4. Offline optimum never exceeds a causal (FIFO) simulation
# ---------------------------------------------------------------------------


def test_offline_bound_never_exceeds_fifo_simulation():
    """Core validation: OPT(omega) <= Cost(pi, omega) for every causal pi,
    pointwise in omega. FIFO is causal, so its total sojourn on this instance
    must be >= the offline optimum."""
    specs = [
        ToyEVSpec(id=0, arrival_time=0.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
        ToyEVSpec(id=1, arrival_time=2.0, battery_kwh=100.0, s_i=0.15, s_f=0.85),
        ToyEVSpec(id=2, arrival_time=8.0, battery_kwh=50.0, s_i=0.25, s_f=0.75),
        ToyEVSpec(id=3, arrival_time=12.0, battery_kwh=150.0, s_i=0.10, s_f=0.80),
    ]
    station_spec = ToyStationSpec(n_piles=2, n_dispensers=2, n_modules=5, p_module=25.0)

    env = run_toy_episode(station_spec, specs, seed=0, policy_seed=1)
    fifo_total_sojourn = float(
        sum(ev.departure_time - ev.arrival_time for ev in env.engine.metrics.finished_evs)
    )

    offline_station = StationSpec(
        n_piles=station_spec.n_piles,
        n_dispensers=station_spec.n_dispensers,
        n_modules=station_spec.n_modules,
        p_module=station_spec.p_module,
    )
    sol = compute_offline_bound(
        build_evs(specs), offline_station, delta=1.0, mip_gap=1e-4, verbose=False
    )

    print(
        f"\nOPT total sojourn={sol.total_sojourn:.3f} min "
        f"(status={sol.status}, gap={sol.mip_gap:.2%}); "
        f"FIFO total sojourn={fifo_total_sojourn:.3f} min"
    )
    assert sol.n_vehicles == len(specs)
    assert sol.total_sojourn <= fifo_total_sojourn + 1e-6


# ---------------------------------------------------------------------------
# 5. Short horizon: feasible but unfinished
# ---------------------------------------------------------------------------


def test_short_horizon_allows_unfinished_vehicles():
    """Without (8), a horizon too short to finish is still feasible: the EV
    stays unfinished (sigma=0) and sojourn runs through the end of T."""
    ev = _make_ev(0, 0.0, 75.0, 0.2, 0.8)
    v = VehicleData.from_ev(ev)
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=8, p_module=25.0)
    delta = 1.0
    horizon = 5.0  # far too short to finish
    om = build_offline_model([v], station, delta, horizon, TAU)
    solve_offline_model(om, mip_gap=1e-6)
    sol = extract_solution(om)
    row = sol.per_vehicle.iloc[0]
    K = om.K
    print(
        f"\nshort horizon: finished={row['finished']}, "
        f"sojourn={row['sojourn']:.3f}, energy={row['energy_kwh']:.3f} kWh"
    )
    assert not row["finished"]
    assert row["sojourn"] == pytest.approx(delta * K - v.a, abs=1e-6)
    assert row["energy_kwh"] <= v.W_kwh + 1e-6


# ---------------------------------------------------------------------------
# 6. tie_break: front-loads power without changing the true optimum
# ---------------------------------------------------------------------------


def _tie_break_scenario():
    """Two vehicles, one pile: vehicle 0 departs early and frees the whole
    module pool to vehicle 1, whose own taper then sits below the pile's
    module cap for a long uncontended tail -- exactly the kind of slack
    tie_break is meant to remove."""
    ev0 = _make_ev(0, 0.0, 30.0, 0.2, 0.5)
    ev1 = _make_ev(1, 0.0, 100.0, 0.15, 0.85)
    vehicles = vehicles_from_evs([ev0, ev1])
    station = StationSpec(n_piles=1, n_dispensers=2, n_modules=5, p_module=25.0)
    horizon = max(full_power_time(v, S_THRESH) for v in vehicles) + 30.0
    return vehicles, station, horizon


def test_tie_break_preserves_optimal_objective():
    """tie_break=True must never change total_sojourn or the primary
    objective -- only which (of possibly many tied-optimal) power profiles
    is returned. See offline_opt/README.md, "Tie breaking"."""
    import math

    vehicles, station, horizon = _tie_break_scenario()

    om_plain = build_offline_model(vehicles, station, 1.0, horizon, TAU, tie_break=False)
    solve_offline_model(om_plain, mip_gap=1e-6)
    sol_plain = extract_solution(om_plain)

    om_tb = build_offline_model(vehicles, station, 1.0, horizon, TAU, tie_break=True)
    solve_offline_model(om_tb, mip_gap=1e-6)
    sol_tb = extract_solution(om_tb)

    print(
        f"\nplain:     total_sojourn={sol_plain.total_sojourn:.3f}, objective={sol_plain.objective:.3f}\n"
        f"tie_break: total_sojourn={sol_tb.total_sojourn:.3f}, objective={sol_tb.objective:.3f}"
    )
    assert sol_tb.total_sojourn == pytest.approx(sol_plain.total_sojourn, abs=1e-6)
    assert sol_tb.objective == pytest.approx(sol_plain.objective, abs=1e-6)
    # MIPGap is unavailable from Gurobi once a second objective is set.
    assert math.isnan(sol_tb.mip_gap)


def test_tie_break_front_loads_after_contention_ends():
    """Once vehicle 0 departs and vehicle 1 has the whole pile to itself,
    tie_break should hold p at exactly min(pile_cap, taper_cap) every slot --
    no artificial dip below what's physically achievable.

    Exception: the vehicle's very last (partial) slot is correctly capped by
    *remaining energy needed* (eq. 17 caps total delivered at w_1, and the
    objective has no reason to overshoot), which is generally less than a
    full slot at the physical rate caps -- that slot is skipped, it isn't a
    tie-break artifact to check for.
    """
    vehicles, station, horizon = _tie_break_scenario()
    pile_cap = station.n_modules * station.p_module

    om = build_offline_model(vehicles, station, 1.0, horizon, TAU, tie_break=True)
    solve_offline_model(om, mip_gap=1e-6)
    delta = om.delta

    k0_0 = om.releases[0]
    finish_0 = next(k for k in range(k0_0, om.K) if om.sigma[0, k].X > 0.5)

    v1 = om.vehicles[1]
    k0_1 = om.releases[1]
    x = 0.0
    checked_any = False
    for k in range(k0_1, om.K):
        p = om.p[1, k].X
        if k > finish_0:  # vehicle 0 is gone; pile fully available to vehicle 1
            taper_cap = (v1.R0 - x) / TAU
            expected = min(pile_cap, taper_cap)
            residual = v1.W - x
            # Only check slots with enough residual energy left that the
            # physical caps -- not "just enough to finish" -- are what binds.
            if expected > 1e-6 and delta * expected <= residual + 1e-6:
                assert p == pytest.approx(expected, rel=1e-4, abs=1e-4)
                checked_any = True
        x += delta * p
        if om.sigma[1, k].X > 0.5:
            break

    assert checked_any, "expected an uncontended tail for vehicle 1 to check"


# ---------------------------------------------------------------------------
# 7. Continuous relaxation bounds sandwich the true integer optimum
# ---------------------------------------------------------------------------


def test_ip_bounds_sandwich_the_true_optimum():
    """RP(N*Delta).total_sojourn <= IP(N*Delta).total_sojourn <=
    RP([N-C+1]*Delta).total_sojourn -- see README.md, "Continuous relaxation
    bounds". Solves all three (both relaxations plus the true integer
    program) on the same instance and checks the bracket directly, rather
    than trusting the proof alone."""
    specs = [
        ToyEVSpec(id=0, arrival_time=0.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
        ToyEVSpec(id=1, arrival_time=2.0, battery_kwh=100.0, s_i=0.15, s_f=0.85),
        ToyEVSpec(id=2, arrival_time=8.0, battery_kwh=50.0, s_i=0.25, s_f=0.75),
        ToyEVSpec(id=3, arrival_time=12.0, battery_kwh=150.0, s_i=0.10, s_f=0.80),
    ]
    station = StationSpec(n_piles=2, n_dispensers=2, n_modules=5, p_module=25.0)
    evs = build_evs(specs)

    ip_sol = compute_offline_bound(evs, station, delta=1.0, mip_gap=1e-4)
    lower, upper = compute_ip_bounds(evs, station, delta=1.0, mip_gap=1e-4)

    print(
        f"\nlower (RP@N*Delta)     total_sojourn={lower.total_sojourn:.3f} ({lower.status})\n"
        f"IP    (true integer)   total_sojourn={ip_sol.total_sojourn:.3f} ({ip_sol.status})\n"
        f"upper (RP@[N-C+1]*Delta) total_sojourn={upper.total_sojourn:.3f} ({upper.status})"
    )
    assert lower.total_sojourn <= ip_sol.total_sojourn + 1e-6
    assert ip_sol.total_sojourn <= upper.total_sojourn + 1e-6


def test_ip_bounds_release_outside_horizon_raises():
    """Arrival after T still fails at build time (release slot out of range)."""
    ev = _make_ev(0, 10.0, 75.0, 0.2, 0.8)
    station = StationSpec(n_piles=1, n_dispensers=1, n_modules=8, p_module=25.0)
    with pytest.raises(ValueError, match="outside the horizon"):
        compute_ip_bounds([ev], station, delta=1.0, horizon_minutes=5.0)


if __name__ == "__main__":
    print("Running test_offline_optimization.py (direct mode)")
    test_tau_matches_taper_formula()
    test_single_ev_matches_theory()
    test_two_piles_no_contention_matches_independent_theory()
    test_offline_bound_never_exceeds_fifo_simulation()
    test_short_horizon_allows_unfinished_vehicles()
    test_tie_break_preserves_optimal_objective()
    test_tie_break_front_loads_after_contention_ends()
    test_ip_bounds_sandwich_the_true_optimum()
    test_ip_bounds_release_outside_horizon_raises()
    print("\nAll tests in test_offline_optimization.py finished.")
