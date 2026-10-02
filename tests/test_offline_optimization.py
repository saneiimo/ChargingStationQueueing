"""
Tests for the offline lower-bound MILP (``offline_opt`` package).

Validates the model against:
  1. The shared taper time constant tau matches the BMS curve exactly
     (constraints 15/16 are wired correctly).
  2. Closed-form single-vehicle charge time (theory) -- no contention.
  3. The "no contention" case with two vehicles, one pile each.
  4. Energy delivered never exceeds W_j (18); finished vehicles meet W_j.
  5. The central sanity check: the offline optimum must never exceed what a
     causal (FIFO) simulation achieves on the same instance.
  6. A short horizon stays feasible (no finish-by-horizon constraint) with
     unfinished vehicles contributing sojourn through T.
  9. The exact discrete taper constant tau_delta (16) reproduces the true
     exponential decay of the BMS's acceptance curve at slot boundaries.

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
    discrete_taper_time_constant,
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
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)  # 200 kW >> p_max
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
    station = StationSpec(n_piles=2, n_connectors=1, n_modules=8, p_module=25.0)
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
    station_spec = ToyStationSpec(n_piles=2, n_connectors=2, n_modules=5, p_module=25.0)

    env = run_toy_episode(station_spec, specs, seed=0, policy_seed=1)
    fifo_total_sojourn = float(
        sum(ev.departure_time - ev.arrival_time for ev in env.engine.metrics.finished_evs)
    )

    offline_station = StationSpec(
        n_piles=station_spec.n_piles,
        n_connectors=station_spec.n_connectors,
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
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)
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
    station = StationSpec(n_piles=1, n_connectors=2, n_modules=5, p_module=25.0)
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
    *remaining energy needed* (eq. 18 caps total delivered at w_1, and the
    objective has no reason to overshoot), which is generally less than a
    full slot at the physical rate caps -- that slot is skipped, it isn't a
    tie-break artifact to check for.
    """
    vehicles, station, horizon = _tie_break_scenario()
    pile_cap = station.n_modules * station.p_module

    om = build_offline_model(vehicles, station, 1.0, horizon, TAU, tie_break=True)
    solve_offline_model(om, mip_gap=1e-6)
    delta = om.delta
    tau_delta = discrete_taper_time_constant(TAU, delta)

    k0_0 = om.releases[0]
    finish_0 = next(k for k in range(k0_0, om.K) if om.sigma[0, k].X > 0.5)

    v1 = om.vehicles[1]
    k0_1 = om.releases[1]
    x = 0.0
    checked_any = False
    for k in range(k0_1, om.K):
        p = om.p[1, k].X
        if k > finish_0:  # vehicle 0 is gone; pile fully available to vehicle 1
            # (16)'s cap: see README.md, "The discrete taper constant".
            taper_cap = (v1.R0 - x) / tau_delta
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
# 7. Pile symmetry breaking (27): rules out relabeled-pile duplicates
# ---------------------------------------------------------------------------


def _pile_symmetry_scenario():
    """Four identical vehicles, two identical piles (2 connectors each) --
    e.g. {veh0,veh3} on pile 0 / {veh1,veh2} on pile 1 is exactly as optimal
    as swapping which pile is which, or any other 2-2 split."""
    evs = [_make_ev(i, 0.0, 60.0, 0.2, 0.8) for i in range(4)]
    vehicles = vehicles_from_evs(evs)
    station = StationSpec(n_piles=2, n_connectors=2, n_modules=4, p_module=25.0)
    horizon = full_power_time(vehicles[0], S_THRESH) + 20.0
    return vehicles, station, horizon


def test_pile_symmetry_breaking_preserves_optimal_objective():
    """(27) must never change total_sojourn -- see README.md, "Pile
    symmetry" for the proof; check it holds on an instance built specifically
    to have real pile symmetry to break."""
    vehicles, station, horizon = _pile_symmetry_scenario()

    om_on = build_offline_model(
        vehicles, station, 1.0, horizon, TAU, break_pile_symmetry=True
    )
    solve_offline_model(om_on, mip_gap=1e-6)
    sol_on = extract_solution(om_on)

    om_off = build_offline_model(
        vehicles, station, 1.0, horizon, TAU, break_pile_symmetry=False
    )
    solve_offline_model(om_off, mip_gap=1e-6)
    sol_off = extract_solution(om_off)

    print(
        f"\nbreak_pile_symmetry=True:  total_sojourn={sol_on.total_sojourn:.3f}, "
        f"nodes={om_on.model.NodeCount:.0f}\n"
        f"break_pile_symmetry=False: total_sojourn={sol_off.total_sojourn:.3f}, "
        f"nodes={om_off.model.NodeCount:.0f}"
    )
    assert sol_on.total_sojourn == pytest.approx(sol_off.total_sojourn, abs=1e-4)


def test_pile_symmetry_breaking_forbids_relabeled_duplicate():
    """The solution (27) picks must respect the canonical ordering directly:
    for every pile m >= 1 that's used, its lowest-id occupant has a larger id
    than pile m-1's -- and no pile is used while an earlier one sits empty."""
    vehicles, station, horizon = _pile_symmetry_scenario()

    om = build_offline_model(
        vehicles, station, 1.0, horizon, TAU, break_pile_symmetry=True
    )
    solve_offline_model(om, mip_gap=1e-6)
    sol = extract_solution(om)

    min_id_by_pile: dict[int, int] = {}
    for _, row in sol.per_vehicle.iterrows():
        pile = row["pile"]
        vid = int(row["vehicle_id"])
        min_id_by_pile[pile] = min(min_id_by_pile.get(pile, vid), vid)
    print(f"\nlowest-id occupant per pile: {min_id_by_pile}")

    used_piles = sorted(min_id_by_pile)
    assert used_piles == list(range(len(used_piles))), "piles must be used with no gaps"
    for m in range(1, len(used_piles)):
        assert min_id_by_pile[m - 1] < min_id_by_pile[m]


# ---------------------------------------------------------------------------
# 8. Continuous relaxation bounds sandwich the true integer optimum
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
    station = StationSpec(n_piles=2, n_connectors=2, n_modules=5, p_module=25.0)
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
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)
    with pytest.raises(ValueError, match="outside the horizon"):
        compute_ip_bounds([ev], station, delta=1.0, horizon_minutes=5.0)


# ---------------------------------------------------------------------------
# 9. Taper cap (16) matches the exact discrete-time recursion (Sec. 1.3)
# ---------------------------------------------------------------------------


def test_taper_cap_matches_exact_discrete_recursion():
    """The exact discrete taper constant ``tau_delta`` (16) must reproduce
    the closed-form ODE solution ``R_{k+1} = R_k * exp(-delta/tau)`` at slot
    boundaries for an uncontended, always-taper-limited vehicle (write-up
    Sec. 1.3) -- not the backward-Euler shortcut ``p*(tau+delta) <= R0-x``
    an earlier version of this model used, which is a valid but needlessly
    loose cap (Lemma 1: ``tau < tau_delta < tau + delta``). Uses a coarse
    delta relative to tau and a vehicle already past s_th so the taper binds
    from the first slot, and checks both that the model matches the exact
    recursion and that it is strictly tighter than the backward-Euler cap
    in at least one slot (a regression guard against reverting the fix).
    """
    ev = _make_ev(0, 0.0, 50.0, 0.5, 0.95)  # s_i=0.5 > S_THRESH: taper from the start
    v = VehicleData.from_ev(ev)
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)  # uncontended
    delta = 15.0  # coarse relative to TAU (18 min under config defaults)
    horizon = full_power_time(v, S_THRESH) + 30.0
    tau_delta = discrete_taper_time_constant(TAU, delta)

    # tie_break=True: without it, the primary objective only cares *when*
    # the vehicle finishes, so the solver is free to return any of the many
    # power profiles that tie the true optimum (README.md, "Tie breaking")
    # -- this test needs the front-loaded one to check exact per-slot values.
    om = build_offline_model([v], station, delta, horizon, TAU, tie_break=True)
    solve_offline_model(om, mip_gap=1e-6)

    k0 = om.releases[0]
    x = 0.0
    checked_exact = False
    exceeded_backward_euler = False
    for k in range(k0, om.K):
        p = om.p[0, k].X
        residual = v.W - x
        room = v.R0 - x
        expected_p = room / tau_delta
        # Only check slots where the taper -- not the energy cap near
        # completion (18) -- is what's actually binding.
        if expected_p > 1e-9 and delta * expected_p <= residual + 1e-6:
            assert p == pytest.approx(expected_p, rel=1e-4, abs=1e-4), (
                f"slot {k}: delivered {p:.4f} kW, expected the exact taper "
                f"cap {expected_p:.4f} kW"
            )
            checked_exact = True
            backward_euler_p = room / (TAU + delta)
            if p > backward_euler_p + 1e-6:
                exceeded_backward_euler = True
        x += delta * p
        if om.sigma[0, k].X > 0.5:
            break

    assert checked_exact, "expected at least one fully taper-bound slot to check"
    assert exceeded_backward_euler, (
        "expected the exact taper constant to deliver strictly more than the "
        "backward-Euler (tau+delta) shortcut in at least one slot"
    )


if __name__ == "__main__":
    print("Running test_offline_optimization.py (direct mode)")
    test_tau_matches_taper_formula()
    test_single_ev_matches_theory()
    test_two_piles_no_contention_matches_independent_theory()
    test_offline_bound_never_exceeds_fifo_simulation()
    test_short_horizon_allows_unfinished_vehicles()
    test_tie_break_preserves_optimal_objective()
    test_tie_break_front_loads_after_contention_ends()
    test_pile_symmetry_breaking_preserves_optimal_objective()
    test_pile_symmetry_breaking_forbids_relabeled_duplicate()
    test_ip_bounds_sandwich_the_true_optimum()
    test_ip_bounds_release_outside_horizon_raises()
    test_taper_cap_matches_exact_discrete_recursion()
    print("\nAll tests in test_offline_optimization.py finished.")
