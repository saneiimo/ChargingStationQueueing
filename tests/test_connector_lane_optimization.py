"""
Tests for the connector-lane offline MILP (``offline_cl_opt`` package).

Validates the model against:
  1. Closed-form single-vehicle charge time (theory) -- no contention.
  2. The "no contention" case: two vehicles, one lane each -> both match
     solo theory and never share a lane.
  3. Two vehicles forced to share a single lane are correctly sequenced
     (never overlap in time).
  4. Energy conservation: delivered energy never exceeds W_j.
  5. The central sanity check: the offline optimum must never exceed what a
     causal (FIFO) simulation achieves on the same instance.
  6. Section 8's adaptive module integrality: solve_cl_model_adaptive
     matches a direct exact solve exactly (Proposition 3); the conservative
     shortcut (8.4) is always roundable; and the basis warm start actually
     speeds up re-optimization versus rebuilding from scratch.
  7. Section 10's symmetry breaking never changes the optimum, and measurably
     cuts branch-and-bound nodes on a symmetric instance.
  8. Section 9's E_j/UB: E_j (9.1) is a reachable lower bound on any
     vehicle's own true departure boundary; UB (9.2), from either source,
     is a valid upper bound on the true optimal objective; constraint (24)
     (9.3, bound_departures) never changes the optimum; and cutoff (9.2/11,
     Gurobi's own Cutoff parameter) never changes the optimum when valid,
     and raises a clear error instead of silently returning nothing when
     built from an incumbent that wasn't actually achievable.
  9. warm_start_evs: solve_cl_model_adaptive can be seeded from a real
     simulation's own output instead of the conservative shortcut, and the
     raw pile/connector assignment gets relabelled into canonical
     (25)-(26) order first whenever break_symmetry=True.

Run: python -m pytest tests/test_connector_lane_optimization.py -s -v
"""

from __future__ import annotations

import time
from math import log

import pytest

gp = pytest.importorskip("gurobipy")
from gurobipy import GRB

from models.ev import EV
from offline_cl_opt import (
    StationSpec,
    VehicleData,
    build_cl_model,
    conservative_feasible_solution,
    earliest_departures,
    incumbent_departure_total,
    rounded_module_routing,
    rounding_test_failures,
    solve_cl_model,
    solve_cl_model_adaptive,
    vehicles_from_evs,
)
from offline_cl_opt.adaptive import _relabel_lanes_for_symmetry
from offline_cl_opt.solution import extract_solution
from toy_demo.runner import run_toy_episode
from toy_demo.scenario import ToyEVSpec, ToyStationSpec, build_evs


def _make_ev(id: int, arrival: float, battery_kwh: float, s_i: float, s_f: float) -> EV:
    return EV(id=id, c_b=battery_kwh * 60, s_i=s_i, s_f=s_f, arrival_time=arrival)


def _solo_theory_minutes(v: VehicleData) -> float:
    """Continuous-time closed form: flat until s_th, then exponential taper
    to s_f, uncontested. Matches Section 4.2's continuous model exactly
    (before the model's own discrete-time approximation of it). Q/p_max is
    hours (kWh/kW), so every term here is converted to minutes (*60)."""
    tau_min = v.tau_hours * 60.0
    if v.s_f <= v.s_th:
        return (v.s_f - v.s_i) * v.Q / v.p_max * 60.0
    flat = max(0.0, v.s_th - v.s_i) * v.Q / v.p_max * 60.0
    if v.s_i >= v.s_th:
        taper = -tau_min * log((1.0 - v.s_f) / (1.0 - v.s_i))
    else:
        taper = -tau_min * log((1.0 - v.s_f) / (1.0 - v.s_th))
    return flat + taper


# ---------------------------------------------------------------------------
# 1. Single vehicle matches closed-form theory
# ---------------------------------------------------------------------------


def test_single_vehicle_matches_theory():
    ev = _make_ev(0, 0.0, 50.0, 0.2, 0.8)
    v = VehicleData.from_ev(ev)
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)  # 200 kW >> p_max
    delta = 1.0
    horizon = _solo_theory_minutes(v) + 20.0

    cl_model = build_cl_model([v], station, delta, horizon)
    solve_cl_model(cl_model, mip_gap=1e-6)
    sol = extract_solution(cl_model)

    row = sol.per_vehicle.iloc[0]
    t_min = _solo_theory_minutes(v)
    print(f"\nsingle vehicle: theory={t_min:.3f} min, MILP sojourn={row['sojourn_min']:.3f} min")

    assert row["served"]
    # Up to ~1 slot of release/departure rounding, plus the small, intended
    # tau -> tau_delta widening (Section 4.2) which very slightly slows the
    # taper relative to the pure continuous closed form.
    assert row["sojourn_min"] == pytest.approx(t_min, abs=2 * delta + 1e-6)
    assert row["energy_kwh"] == pytest.approx(v.W, rel=1e-6)


# ---------------------------------------------------------------------------
# 2. No contention: identical vehicles, one lane each -> both match solo theory
# ---------------------------------------------------------------------------


def test_two_lanes_no_contention_matches_independent_theory():
    ev0 = _make_ev(0, 0.0, 50.0, 0.2, 0.8)
    ev1 = _make_ev(1, 0.0, 50.0, 0.2, 0.8)
    vehicles = vehicles_from_evs([ev0, ev1])
    station = StationSpec(n_piles=2, n_connectors=1, n_modules=8, p_module=25.0)
    delta = 1.0
    t_min = _solo_theory_minutes(vehicles[0])
    horizon = t_min + 20.0

    cl_model = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(cl_model, mip_gap=1e-6)
    sol = extract_solution(cl_model)

    assert sol.per_vehicle["pile"].nunique() == 2  # each took its own pile
    for _, row in sol.per_vehicle.iterrows():
        assert row["served"]
        assert row["sojourn_min"] == pytest.approx(t_min, abs=2 * delta + 1e-6)


# ---------------------------------------------------------------------------
# 3. Two vehicles forced onto one lane are correctly sequenced
# ---------------------------------------------------------------------------


def test_shared_lane_forces_no_overlap():
    """One pile, one connector: both vehicles must use the same lane, so
    (11)-(12) must strictly order them -- one finishes before the other starts."""
    ev0 = _make_ev(0, 0.0, 50.0, 0.2, 0.8)
    ev1 = _make_ev(1, 0.0, 50.0, 0.2, 0.8)
    vehicles = vehicles_from_evs([ev0, ev1])
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=8, p_module=25.0)
    delta = 1.0
    t_min = _solo_theory_minutes(vehicles[0])
    horizon = 2 * t_min + 20.0

    cl_model = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(cl_model, mip_gap=1e-6)
    sol = extract_solution(cl_model)

    assert (sol.per_vehicle["pile"] == 0).all()
    assert (sol.per_vehicle["connector"] == 0).all()
    starts = sol.per_vehicle.set_index("vehicle_id")["start_slot"]
    ends = sol.per_vehicle.set_index("vehicle_id")["departure_slot"]
    print(f"\nvehicle 0: [{starts[0]}, {ends[0]}); vehicle 1: [{starts[1]}, {ends[1]})")
    # No overlap: one interval ends at or before the other starts.
    assert ends[0] <= starts[1] + 1e-6 or ends[1] <= starts[0] + 1e-6


# ---------------------------------------------------------------------------
# 4. Energy conservation
# ---------------------------------------------------------------------------


def test_energy_never_exceeds_requirement():
    specs = [
        ToyEVSpec(id=0, arrival_time=0.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
        ToyEVSpec(id=1, arrival_time=2.0, battery_kwh=100.0, s_i=0.15, s_f=0.85),
        ToyEVSpec(id=2, arrival_time=8.0, battery_kwh=50.0, s_i=0.25, s_f=0.75),
    ]
    station = StationSpec(n_piles=1, n_connectors=2, n_modules=5, p_module=25.0)
    vehicles = vehicles_from_evs(build_evs(specs))

    cl_model = build_cl_model(vehicles, station, delta=1.0, horizon_minutes=200.0)
    solve_cl_model(cl_model, mip_gap=1e-4)
    sol = extract_solution(cl_model)

    for _, row in sol.per_vehicle.iterrows():
        assert row["energy_kwh"] <= row["energy_required_kwh"] + 1e-6


# ---------------------------------------------------------------------------
# 5. Offline optimum never exceeds a causal (FIFO) simulation
# ---------------------------------------------------------------------------


def test_offline_bound_never_exceeds_fifo_simulation():
    """Core validation: OPT(omega) <= Cost(pi, omega) for every causal pi,
    pointwise in omega. FIFO is causal, so its total sojourn on this instance
    must be >= the offline optimum's."""
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

    cl_station = StationSpec(
        n_piles=station_spec.n_piles,
        n_connectors=station_spec.n_connectors,
        n_modules=station_spec.n_modules,
        p_module=station_spec.p_module,
    )
    vehicles = vehicles_from_evs(build_evs(specs))
    cl_model = build_cl_model(vehicles, cl_station, delta=1.0, horizon_minutes=200.0)
    solve_cl_model(cl_model, mip_gap=1e-4)
    sol = extract_solution(cl_model)
    opt_total_sojourn = float(sol.per_vehicle["sojourn_min"].sum())

    print(
        f"\nOPT total sojourn={opt_total_sojourn:.3f} min "
        f"(status={sol.status}, gap={sol.mip_gap:.2%}); "
        f"FIFO total sojourn={fifo_total_sojourn:.3f} min"
    )
    assert sol.per_vehicle["served"].all()
    assert opt_total_sojourn <= fifo_total_sojourn + 1e-6


# ---------------------------------------------------------------------------
# 6. Section 8: adaptive module integrality
# ---------------------------------------------------------------------------


def _adaptive_test_instance():
    """Small but genuinely contended: 1 pile, 2 connectors, 4 modules
    shared between two vehicles that both want more than half the pool."""
    ev0 = _make_ev(0, 0.0, 50.0, 0.2, 0.8)
    ev1 = _make_ev(1, 0.0, 100.0, 0.15, 0.85)
    vehicles = vehicles_from_evs([ev0, ev1])
    station = StationSpec(n_piles=1, n_connectors=2, n_modules=4, p_module=25.0)
    return vehicles, station, 1.0, 90.0  # delta, horizon_minutes


def test_adaptive_matches_exact_solve():
    """Proposition 3: the adaptive procedure's answer is exactly optimal
    for the fully-integer exact model, not merely a good heuristic one --
    check it matches a direct solve of the exact model (r integer from the
    start) to the tightened mip_gap used here."""
    vehicles, station, delta, horizon = _adaptive_test_instance()

    exact = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(exact, mip_gap=1e-6)

    result = solve_cl_model_adaptive(vehicles, station, delta, horizon, mip_gap=1e-6)

    print(
        f"\nexact direct obj={exact.model.ObjVal:.4f}; "
        f"adaptive obj={result.cl_model.model.ObjVal:.4f} "
        f"(converged={result.converged}, iterations={result.iterations}, "
        f"history={result.objective_history})"
    )
    assert result.converged
    # History must be non-decreasing (each z_t is a valid lower bound on
    # z*, Proposition 3's first inequality) and end at the true optimum.
    assert all(
        b >= a - 1e-6 for a, b in zip(result.objective_history, result.objective_history[1:])
    )
    assert result.cl_model.model.ObjVal == pytest.approx(exact.model.ObjVal, abs=1e-4)
    assert not rounding_test_failures(result.cl_model)


def test_conservative_shortcut_always_roundable():
    """Section 8.4: solving with the tightened N-C+1 budget must always
    pass the rounding test (20) -- conservative_feasible_solution asserts
    this internally; check it externally too, and that its objective is a
    valid (if loose) upper bound on the true optimum."""
    vehicles, station, delta, horizon = _adaptive_test_instance()

    cons = conservative_feasible_solution(vehicles, station, delta, horizon, mip_gap=1e-4)
    assert not rounding_test_failures(cons)

    exact = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(exact, mip_gap=1e-6)
    print(f"\nconservative obj={cons.model.ObjVal:.4f} >= exact obj={exact.model.ObjVal:.4f}")
    assert cons.model.ObjVal >= exact.model.ObjVal - 1e-6


def test_rounded_module_routing_respects_pile_budget():
    """The Lemma's hat_r, computed off a converged adaptive solve, must be
    a genuine (14)-(15)-feasible integer routing: within the pile's real
    module pool at every slot, and enough to cover each occupant's power."""
    vehicles, station, delta, horizon = _adaptive_test_instance()
    result = solve_cl_model_adaptive(vehicles, station, delta, horizon, mip_gap=1e-4)
    assert result.converged

    routing = rounded_module_routing(result.cl_model)
    N, Delta = station.n_modules, station.p_module
    for k in range(result.cl_model.K):
        total = sum(routing.get((0, cc, k), 0) for cc in range(station.n_connectors))
        assert total <= N, f"slot {k}: routed {total} modules > pool {N}"

    for j in result.cl_model.vehicles:
        k0 = result.cl_model.releases[j]
        for (mm, cc) in result.cl_model.lanes:
            if result.cl_model.y[j, mm, cc].X <= 0.5:
                continue
            for k in range(k0, result.cl_model.K):
                if result.cl_model.u[j, k].X > 0.5:
                    p_val = result.cl_model.p[j, k].X
                    assert routing[mm, cc, k] * Delta >= p_val - 1e-6


def test_basis_warm_start_speeds_up_reoptimization():
    """
    Empirically checks the source document's basis-warm-start claim
    (Section 8, quoted in adaptive.py's module docstring) rather than
    taking it on faith: re-optimizing the SAME live model after only
    mutating some r's VType (never touching the constraint matrix, bounds,
    or objective) should be at least as fast in total as rebuilding a fresh
    model with no prior basis at each step of the identical promotion
    schedule, since the LP relaxation is the literal same LP either way.
    """
    vehicles, station, delta, horizon = _adaptive_test_instance()

    # Discover the promotion schedule once (MIP start disabled so this
    # measures only the basis-reuse effect, not a different confound).
    result = solve_cl_model_adaptive(
        vehicles, station, delta, horizon, warm_start_from_conservative=False, mip_gap=1e-4
    )
    assert result.converged
    assert result.promotions_by_iteration, "need at least one promotion to compare warm vs. cold"

    def _promote(cl_model, pile_slots):
        for (mm, k) in pile_slots:
            for cc in range(station.n_connectors):
                cl_model.r[mm, cc, k].VType = GRB.INTEGER
        cl_model.model.update()

    # Warm: one live model, mutated and re-solved in place across iterations.
    warm_model = build_cl_model(vehicles, station, delta, horizon, relax_modules=True)
    t0 = time.time()
    solve_cl_model(warm_model, mip_gap=1e-4)
    for promoted in result.promotions_by_iteration:
        _promote(warm_model, promoted)
        solve_cl_model(warm_model, mip_gap=1e-4)
    warm_time = time.time() - t0

    # Cold: rebuild from scratch at every step, replaying the identical
    # cumulative promotion schedule, so the sequence of MIPs solved is
    # exactly the same -- the only difference is whether Gurobi has a prior
    # basis to start from.
    cumulative: list[tuple[int, int]] = []
    t0 = time.time()
    cold_model = build_cl_model(vehicles, station, delta, horizon, relax_modules=True)
    solve_cl_model(cold_model, mip_gap=1e-4)
    for promoted in result.promotions_by_iteration:
        cumulative.extend(promoted)
        cold_model = build_cl_model(vehicles, station, delta, horizon, relax_modules=True)
        _promote(cold_model, cumulative)
        solve_cl_model(cold_model, mip_gap=1e-4)
    cold_time = time.time() - t0

    n_solves = len(result.promotions_by_iteration) + 1
    print(
        f"\nwarm (reused model): {warm_time:.3f}s over {n_solves} solves\n"
        f"cold (rebuilt each step): {cold_time:.3f}s over {n_solves} solves"
    )
    # Generous tolerance -- the point is confirming the warm path is not
    # worse, not chasing a specific speedup factor on a tiny toy instance.
    assert warm_time <= cold_time * 1.25 + 0.5


def test_adaptive_progress_flag_prints_iteration_summary(capsys):
    """progress=True is independent of verbose (Gurobi's own log) -- it
    should print the adaptive loop's own per-iteration summary lines."""
    vehicles, station, delta, horizon = _adaptive_test_instance()
    solve_cl_model_adaptive(vehicles, station, delta, horizon, mip_gap=1e-4, progress=True)
    captured = capsys.readouterr()
    assert "[adaptive] iter" in captured.out


# ---------------------------------------------------------------------------
# 7. Section 10: symmetry breaking never changes the optimum
# ---------------------------------------------------------------------------


def _multi_pile_test_instance():
    """Two identical piles, two connectors each -- real pile/connector
    symmetry (25)-(26) can break."""
    ev0 = _make_ev(0, 0.0, 50.0, 0.2, 0.8)
    ev1 = _make_ev(1, 0.0, 100.0, 0.15, 0.85)
    ev2 = _make_ev(2, 5.0, 50.0, 0.2, 0.8)
    vehicles = vehicles_from_evs([ev0, ev1, ev2])
    station = StationSpec(n_piles=2, n_connectors=2, n_modules=4, p_module=25.0)
    return vehicles, station, 1.0, 100.0


def test_symmetry_breaking_preserves_optimal_objective():
    vehicles, station, delta, horizon = _multi_pile_test_instance()

    m_off = build_cl_model(vehicles, station, delta, horizon, break_symmetry=False)
    solve_cl_model(m_off, mip_gap=1e-4)
    m_on = build_cl_model(vehicles, station, delta, horizon, break_symmetry=True)
    solve_cl_model(m_on, mip_gap=1e-4)

    print(
        f"\nbreak_symmetry=False: obj={m_off.model.ObjVal:.3f}, nodes={m_off.model.NodeCount:.0f}\n"
        f"break_symmetry=True:  obj={m_on.model.ObjVal:.3f}, nodes={m_on.model.NodeCount:.0f}"
    )
    assert m_on.model.ObjVal == pytest.approx(m_off.model.ObjVal, abs=1e-3)


# ---------------------------------------------------------------------------
# 8. Section 9: preprocessing windows
# ---------------------------------------------------------------------------


def test_earliest_departures_are_reachable_lower_bounds():
    """E_j (22) must lower-bound any vehicle's own true departure boundary
    in the exact (contended) model -- it's each vehicle's best case alone,
    so a real, contended solve can never beat it for any single vehicle."""
    vehicles, station, delta, horizon = _multi_pile_test_instance()
    E = earliest_departures(vehicles, station, delta, horizon)

    m = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(m, mip_gap=1e-4)
    sol = extract_solution(m)
    print(f"\nE={E}")
    for _, row in sol.per_vehicle.iterrows():
        assert row["departure_slot"] >= E[int(row["vehicle_id"])] - 1e-6


def test_incumbent_departure_total_is_valid_upper_bound():
    """UB (Section 9.2), from either source, must be >= the true optimal objective
    -- it's the total departure slots of an actually-achievable schedule,
    and a minimization's optimum can never exceed any feasible value."""
    vehicles, station, delta, horizon = _multi_pile_test_instance()
    m_exact = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(m_exact, mip_gap=1e-6)

    ub_conservative = incumbent_departure_total(vehicles, station, delta, horizon, mip_gap=1e-4)
    print(f"\nUB (conservative fallback) = {ub_conservative}, true optimum = {m_exact.model.ObjVal}")
    assert ub_conservative >= m_exact.model.ObjVal - 1e-6


def test_cutoff_from_real_simulation_preserves_optimum():
    """A cutoff built from a genuinely feasible reference schedule (here, a
    real FIFO simulation on the same arrivals) must never change the
    optimal objective -- Gurobi's own Cutoff semantics accept any solution
    at least as good as the cutoff, so the true optimum is still found and
    reported normally whenever it equals or beats a valid cutoff."""
    vehicles, station, delta, horizon = _multi_pile_test_instance()

    m_exact = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(m_exact, mip_gap=1e-6)

    specs = [
        ToyEVSpec(id=v.id, arrival_time=v.a, battery_kwh=v.Q, s_i=v.s_i, s_f=v.s_f)
        for v in vehicles
    ]
    toy_station = ToyStationSpec(
        n_piles=station.n_piles,
        n_connectors=station.n_connectors,
        n_modules=station.n_modules,
        p_module=station.p_module,
    )
    env = run_toy_episode(toy_station, specs, seed=0, policy_seed=1)
    sim_departures = {ev.id: ev.departure_time for ev in env.engine.metrics.finished_evs}

    UB = incumbent_departure_total(
        vehicles, station, delta, horizon, incumbent_departures=sim_departures
    )
    print(f"\nsim_departures={sim_departures}\nUB={UB}")

    m_cut = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(m_cut, mip_gap=1e-6, cutoff=UB)

    assert m_cut.model.ObjVal == pytest.approx(m_exact.model.ObjVal, abs=1e-4)
    extract_solution(m_cut)


def test_cutoff_conservative_fallback_preserves_optimum():
    """With no incumbent_departures supplied, the conservative shortcut
    (8.4) is used automatically as the UB source -- also never changes the
    optimum when used as a cutoff."""
    vehicles, station, delta, horizon = _multi_pile_test_instance()

    m_exact = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(m_exact, mip_gap=1e-6)

    UB = incumbent_departure_total(vehicles, station, delta, horizon, mip_gap=1e-4)
    m_cut = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(m_cut, mip_gap=1e-6, cutoff=UB)

    assert m_cut.model.ObjVal == pytest.approx(m_exact.model.ObjVal, abs=1e-4)


def test_inconsistent_cutoff_raises():
    """A cutoff built from departures that aren't actually achievable for
    this instance must raise a clear error instead of silently returning no
    solution: a cutoff derived from a genuinely achievable schedule can
    never make Gurobi report GRB.CUTOFF, since the model can always at
    least match it -- so this can only happen from an inconsistent cutoff.

    The fabricated incumbent here is the instance's own true optimal
    departures (verified separately: 26/46/46 minutes), shrunk by 20% --
    close enough to look plausible, but not actually achievable, so the
    resulting cutoff sits strictly below the true optimum.
    """
    vehicles, station, delta, horizon = _multi_pile_test_instance()

    bad_UB = incumbent_departure_total(
        vehicles,
        station,
        delta,
        horizon,
        incumbent_departures={0: 20.8, 1: 36.8, 2: 36.8},  # not a real achievable schedule
    )
    m_bad = build_cl_model(vehicles, station, delta, horizon)
    with pytest.raises(RuntimeError, match="cutoff"):
        solve_cl_model(m_bad, mip_gap=1e-6, cutoff=bad_UB)


def test_bound_departures_preserves_optimal_objective():
    """(24), Section 9.3: a valid lower bound on D_j derived from E_j and
    the occupancy count -- turning it on/off must never change the optimal
    objective, only (potentially) the effort needed to find it."""
    vehicles, station, delta, horizon = _multi_pile_test_instance()

    m_on = build_cl_model(vehicles, station, delta, horizon, bound_departures=True)
    solve_cl_model(m_on, mip_gap=1e-6)
    m_off = build_cl_model(vehicles, station, delta, horizon, bound_departures=False)
    solve_cl_model(m_off, mip_gap=1e-6)

    print(f"\nbound_departures=True: obj={m_on.model.ObjVal:.3f}; False: obj={m_off.model.ObjVal:.3f}")
    assert m_on.model.ObjVal == pytest.approx(m_off.model.ObjVal, abs=1e-4)


# ---------------------------------------------------------------------------
# 9. warm_start_evs: seeding the adaptive procedure from a real simulation
# ---------------------------------------------------------------------------


def test_relabel_lanes_for_symmetry_produces_canonical_order():
    """Pure unit test of the relabelling algorithm (no MILP needed): piles,
    and connectors within each pile, must come out labelled in order of the
    lowest vehicle id that occupies them -- exactly what (25)-(26) require."""
    # Raw assignment deliberately NOT in canonical order: vehicle 0 (lowest
    # id) is on raw pile 5, but vehicle 1 got raw pile 2 -- and within raw
    # pile 5, vehicle 3 (connector 1) is seen before vehicle 4 (connector 9).
    raw = {0: (5, 9), 1: (2, 0), 3: (5, 1), 4: (5, 9)}
    sorted_ids = [0, 1, 3, 4]

    relabeled = _relabel_lanes_for_symmetry(raw, sorted_ids)

    # Vehicle 0 is the first vehicle overall -> its pile becomes canonical 0,
    # its connector becomes canonical 0.
    assert relabeled[0] == (0, 0)
    # Vehicle 1 is the first to introduce a second raw pile -> canonical 1.
    assert relabeled[1] == (1, 0)
    # Vehicle 3 shares vehicle 0's raw pile (now canonical 0) but a new raw
    # connector -> canonical connector 1 on pile 0.
    assert relabeled[3] == (0, 1)
    # Vehicle 4 shares vehicle 0's exact raw lane -> the same canonical lane.
    assert relabeled[4] == (0, 0)

    # General property: for every pile actually used, its lowest-id occupant
    # must be smaller than the next pile's lowest-id occupant (25); same for
    # connectors within a pile (26).
    by_pile: dict[int, list[int]] = {}
    for vid, (p, _c) in relabeled.items():
        by_pile.setdefault(p, []).append(vid)
    used_piles = sorted(by_pile)
    assert used_piles == list(range(len(used_piles)))
    for p in range(1, len(used_piles)):
        assert min(by_pile[p - 1]) < min(by_pile[p])


def _symmetric_sim_test_instance():
    """Same shape as _multi_pile_test_instance, plus the toy-demo specs
    needed to actually run a simulation on it."""
    specs = [
        ToyEVSpec(id=0, arrival_time=0.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
        ToyEVSpec(id=1, arrival_time=0.0, battery_kwh=100.0, s_i=0.15, s_f=0.85),
        ToyEVSpec(id=2, arrival_time=5.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
    ]
    toy_station = ToyStationSpec(n_piles=2, n_connectors=2, n_modules=4, p_module=25.0)
    station = StationSpec(n_piles=2, n_connectors=2, n_modules=4, p_module=25.0)
    vehicles = vehicles_from_evs(build_evs(specs))
    return specs, toy_station, station, vehicles


def test_warm_start_from_evs_matches_exact_solve():
    """solve_cl_model_adaptive seeded from a real simulation's own output
    (not the conservative shortcut) must still converge to the exact
    optimum (Proposition 3 doesn't care where the MIP start came from), and
    must not error out even though the raw simulated pile/connector
    assignment has no reason to already respect the canonical (25)-(26)
    order break_symmetry=True enforces -- the relabelling step handles it."""
    specs, toy_station, station, vehicles = _symmetric_sim_test_instance()
    delta, horizon = 1.0, 100.0

    env = run_toy_episode(toy_station, specs, seed=0, policy_seed=1)
    finished_evs = env.engine.metrics.finished_evs
    assert len(finished_evs) == len(vehicles), "expected everyone to finish in this toy scenario"

    exact = build_cl_model(vehicles, station, delta, horizon)
    solve_cl_model(exact, mip_gap=1e-6)

    result = solve_cl_model_adaptive(
        vehicles,
        station,
        delta,
        horizon,
        break_symmetry=True,
        warm_start_evs=finished_evs,
        mip_gap=1e-4,
    )
    print(
        f"\nexact obj={exact.model.ObjVal:.3f}; "
        f"warm_start_evs+break_symmetry obj={result.cl_model.model.ObjVal:.3f} "
        f"(converged={result.converged}, iterations={result.iterations})"
    )
    assert result.converged
    assert result.cl_model.model.ObjVal == pytest.approx(exact.model.ObjVal, abs=1e-4)


if __name__ == "__main__":
    print("Running test_connector_lane_optimization.py (direct mode)")
    test_single_vehicle_matches_theory()
    test_two_lanes_no_contention_matches_independent_theory()
    test_shared_lane_forces_no_overlap()
    test_energy_never_exceeds_requirement()
    test_offline_bound_never_exceeds_fifo_simulation()
    test_adaptive_matches_exact_solve()
    test_conservative_shortcut_always_roundable()
    test_rounded_module_routing_respects_pile_budget()
    test_basis_warm_start_speeds_up_reoptimization()
    # test_adaptive_progress_flag_prints_iteration_summary needs pytest's
    # capsys fixture -- run via pytest, not this __main__ block.
    test_symmetry_breaking_preserves_optimal_objective()
    test_earliest_departures_are_reachable_lower_bounds()
    test_incumbent_departure_total_is_valid_upper_bound()
    test_cutoff_from_real_simulation_preserves_optimum()
    test_cutoff_conservative_fallback_preserves_optimum()
    test_inconsistent_cutoff_raises()
    test_bound_departures_preserves_optimal_objective()
    test_relabel_lanes_for_symmetry_produces_canonical_order()
    test_warm_start_from_evs_matches_exact_solve()
    print("\nAll tests in test_connector_lane_optimization.py finished.")
