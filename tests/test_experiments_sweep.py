"""
The objective sweep's reporting semantics (``experiments`` package).

Validates what a sweep row's meaning rests on:

  1. Every model is the closed-system model, so a model schedule's
     ``completed`` and ``arrived`` groups coincide; a vehicle departing
     exactly at the horizon is complete and is counted in ``n_at_horizon``.
  2. ``_model_groups`` refuses a schedule short of some vehicle's energy
     (a solver bug, not a result) -- counting a boundary vehicle's
     pre-window energy, and exempting FIXED boundary vehicles, whose pinned
     power misses their final partial slot.
  3. The branch-and-price stage (``run_bp_model``) writes its ``bp_*``
     columns and agrees with the exact MILP wherever both prove optimality.
  4. The price-directed search stage (``run_pds_model``) does the same with
     its ``pds_*`` columns and ``pds_bound_history.csv``, and honours
     ``pds_tolerance_min``.

Run: python -m pytest tests/test_experiments_sweep.py -s -v
"""

from __future__ import annotations

import pandas as pd
import pytest

gp = pytest.importorskip("gurobipy")

from experiments.objective_sweep.trial import _model_groups
from offline_cl_opt import build_cl_model, solve_cl_model
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.solution import extract_solution


def test_vehicle_finishing_exactly_at_the_horizon_is_complete():
    """A horizon of exactly E_j slots forces D_j = K: the vehicle is complete
    (closed system), both groups hold it, and n_at_horizon flags the edge."""
    from offline_cl_opt.model import earliest_departures

    station = StationSpec(n_piles=1, n_connectors=1, n_modules=4, p_module=25.0)
    v = VehicleData(id=0, a=0.0, Q=100.0, s_i=0.15, s_f=0.82, s_th=0.4, p_max=100.0)
    E = earliest_departures([v], station, 5.0, 500.0)[v.id]
    cl = build_cl_model([v], station, delta=5.0, horizon_minutes=5.0 * E)
    solve_cl_model(cl, mip_gap=1e-6, verbose=False)
    sol = extract_solution(cl)
    row = sol.per_vehicle.iloc[0]
    assert row["departure_slot"] == pytest.approx(sol.K)
    assert row["energy_kwh"] == pytest.approx(row["energy_required_kwh"], abs=1e-6)

    completed, arrived, diag = _model_groups(sol.per_vehicle, sol.K)
    assert completed == arrived
    assert completed["all"]["n"] == 1.0
    assert diag == {"n_at_horizon": 1}


def _row(vid, cohort, mode, delivered, before, required, departure=5.0):
    return {
        "vehicle_id": vid, "cohort": cohort, "boundary_mode": mode, "sojourn_min": 10.0,
        "departure_slot": departure, "energy_kwh": delivered,
        "energy_delivered_before_kwh": before, "energy_required_kwh": required,
    }


def test_model_groups_reject_a_schedule_short_of_energy():
    pv = pd.DataFrame([_row(1, "measurement", None, 10.0, 0.0, 40.0)])
    with pytest.raises(RuntimeError, match="short of"):
        _model_groups(pv, K=12)


def test_boundary_energy_counts_and_fixed_vehicles_are_exempt():
    """An OPTIMIZE boundary vehicle with 30 kWh before t0 and 20 inside the
    window is complete (50 of 50); a FIXED one whose pinned power misses its
    final partial slot is exempt, as in the model."""
    pv = pd.DataFrame([
        _row(0, "boundary", "optimize", 20.0, 30.0, 50.0),
        _row(1, "boundary", "fixed", 9.0, 5.0, 15.0),
        _row(2, "measurement", None, 40.0, 0.0, 40.0, departure=12.0),
    ])
    completed, arrived, diag = _model_groups(pv, K=12)
    assert completed["all"]["n"] == 3.0 and completed["measurement"]["n"] == 1.0
    assert completed == arrived
    assert diag["n_at_horizon"] == 1


def test_saved_arrivals_replay_the_same_utilization():
    """The arrival specs written with a trial must rebuild the same episode.

    ``results.csv`` does not store the env. ``arrivals`` on the trial record
    plus ``TrialConfig.to_row()`` is the whole rebuild key: replaying them
    under FIFO must match the original connector utilization.
    """
    from experiments.objective_sweep import TrialConfig
    from experiments.objective_sweep.trial import (
        episode_utilization,
        replay_episode,
        run_episode,
    )

    cfg = TrialConfig(
        n_piles=1,
        n_connectors=1,
        n_modules=4,
        p_module=25.0,
        mean_interarrival=20.0,
        max_time=40.0,
        warmup_period=20.0,
        seed=3,
        policy_seed=1,
        battery_cap_kwh=(50.0,),
        delta_arr=None,
    )
    env, _, specs = run_episode(cfg)
    assert specs, "expected at least one saved arrival"
    replayed = replay_episode(cfg.to_row(), specs)

    original = episode_utilization(env, post_warmup=True)
    again = episode_utilization(replayed, post_warmup=True)
    assert again["rho_sim"] == pytest.approx(original["rho_sim"])
    assert again["rho_theory"] == pytest.approx(original["rho_theory"])
    assert again["n_finished"] == original["n_finished"]
    print(
        f"  PASS replay rho_sim={again['rho_sim']:.3f} "
        f"rho_theory={again['rho_theory']:.3f}"
    )


def _tiny_bp_config(**kw):
    from experiments.objective_sweep import TrialConfig

    base = dict(
        n_piles=1, n_connectors=2, n_modules=6, max_time=60, warmup_period=120, delta=5.0,
        seed=41, battery_cap_kwh=(75.0,), time_limit=120.0, dw_time_limit=120.0,
        bp_time_limit=120.0, gap_tolerance_target_min=None,
    )
    base.update(kw)
    return TrialConfig(**base)


def test_bp_stage_matches_exact_and_is_recorded(tmp_path):
    """Both exact stages on small closed systems: same proven optimum, bp_* columns,
    JSON record and run_meta stage flag all written. Every objective/bound
    column is total sojourn in minutes, so they line up with each other and
    with the per-vehicle sojourn columns directly."""
    import json

    from experiments.objective_sweep import comparison_table, run_sweep

    configs = [_tiny_bp_config(mean_interarrival=ia, bp_initial_schedule="both") for ia in (15.0, 25.0)]
    df = run_sweep(configs, "bp", out_dir=tmp_path, run_exact_model=True, run_bp_model=True,
                   run_dw_model=True, verbose=False)
    for _, row in df.iterrows():
        # Objective = total sojourn over objective_cohorts (all cohorts here).
        assert row["exact_objective"] == pytest.approx(row["exact_total_sojourn"])
        assert row["exact_objective"] == pytest.approx(row["exact_arrived_all_total_sojourn"])
        assert row["bp_objective"] == pytest.approx(row["bp_total_sojourn"])
        assert row["exact_mean_sojourn_LB"] == pytest.approx(row["exact_best_bound"] / row["exact_n_optimized"])
        assert row["dw_LB"] <= row["exact_objective"] + 1e-6
        assert row["dw_mean_sojourn_LB"] == pytest.approx(row["dw_LB"] / row["dw_n_optimized"])
        assert row["dw_sojourn_floor"] <= row["dw_LB"] + 1e-6
        assert row["exact_status"] == "OPTIMAL" and row["bp_status"] == "OPTIMAL"
        assert row["bp_objective"] == pytest.approx(row["exact_objective"])
        assert row["bp_best_bound"] == pytest.approx(row["bp_objective"])
        assert row["bp_compact_check_max_violation"] <= 1e-5
        assert row["bp_seed_source"] in ("simulation", "exact")
        assert row["bp_arrived_all_mean_sojourn"] == pytest.approx(row["exact_arrived_all_mean_sojourn"])
    table = comparison_table(df)
    assert {"bp", "bp_LB", "bp_status", "bp_runtime_s"} <= set(table.columns)
    meta = json.loads((tmp_path / "bp" / "run_meta.json").read_text())
    assert meta["stages"] == {"sim": True, "exact": True, "bp": True, "pds": False, "dw": True}
    record = json.loads((tmp_path / "bp" / "trials" / "trial_000.json").read_text())
    assert record["bp"]["status"] == "OPTIMAL" and "arrived_by_cohort" in record["bp"]


def test_bp_stage_alone_and_exact_seed_without_exact(tmp_path):
    """B&P without the MILP: an "exact" seed has nothing to use, so none is
    passed (B&P falls back to its own greedy schedule) -- not an error."""
    from experiments.objective_sweep import run_sweep

    df = run_sweep([_tiny_bp_config(mean_interarrival=15.0, bp_initial_schedule="exact")], "bp_only",
                   out_dir=tmp_path, run_exact_model=False, run_bp_model=True, run_dw_model=False,
                   verbose=False)
    row = df.iloc[0]
    assert "exact_objective" not in df.columns
    assert row["bp_status"] == "OPTIMAL"
    assert row["bp_seed_source"] == "none"
    assert row["bp_incumbent_source"] != "initial schedule"


def test_bp_bound_history_is_returned_saved_and_consistent(tmp_path):
    """bound_history: returned with return_history, written to
    bound_history.csv (not into results.csv), monotone per trial, and its
    last row is the trial's bp_best_bound / bp_objective."""
    from experiments.objective_sweep import load_bound_history, run_sweep

    # (12, delta=2) prunes its last open nodes only after the final incumbent:
    # the recorded lower bound must stop at the incumbent, not overshoot it.
    configs = [_tiny_bp_config(mean_interarrival=15.0, label="ia15"),
               _tiny_bp_config(mean_interarrival=12.0, delta=2.0, label="ia12")]
    df, history = run_sweep(configs, "hist", out_dir=tmp_path, run_exact_model=False, run_bp_model=True,
                            run_dw_model=False, verbose=False, return_history=True)
    assert not any("bound_history" in c for c in df.columns)
    saved = load_bound_history("hist", out_dir=tmp_path)
    pd.testing.assert_frame_equal(saved, history, check_dtype=False)
    assert set(history["trial_id"]) == {0, 1}
    for trial_id, h in history.groupby("trial_id"):
        row = df.loc[df["trial_id"] == trial_id].iloc[0]
        assert (h["label"] == row["label"]).all()
        lb, ub = h["lower_bound"].dropna(), h["upper_bound"].dropna()
        assert lb.is_monotonic_increasing and ub.is_monotonic_decreasing
        assert h["time_s"].is_monotonic_increasing
        last = h.iloc[-1]
        assert last["event"] == "end"
        assert last["lower_bound"] == pytest.approx(row["bp_best_bound"])
        assert last["upper_bound"] == pytest.approx(row["bp_objective"])
        assert last["upper_bound_min"] == pytest.approx(row["bp_mean_sojourn"])
        assert last["lower_bound_min"] == pytest.approx(row["bp_mean_sojourn_LB"])
        assert last["time_s"] <= row["bp_runtime_s"] + 1e-6
        valid = h.dropna(subset=["lower_bound", "upper_bound"])
        assert (valid["lower_bound"] <= valid["upper_bound"]).all() and (valid["gap"] >= 0).all()
        assert (valid["gap_pct"] - 100 * (valid["upper_bound"] - valid["lower_bound"]) / valid["upper_bound"]).abs().max() < 1e-9
    # Without B&P there is no history, and no file.
    run_sweep(configs[:1], "nohist", out_dir=tmp_path, run_exact_model=False, run_bp_model=False,
              run_dw_model=False, verbose=False)
    assert load_bound_history("nohist", out_dir=tmp_path).empty


def test_pds_stage_matches_exact_and_is_recorded(tmp_path):
    """Exact MILP and price-directed search on small closed systems: same
    proven optimum, pds_* columns, comparison table, JSON record, run_meta
    stage flag and pds_bound_history.csv all written."""
    import json

    pytest.importorskip("numba")
    from experiments.objective_sweep import comparison_table, load_bound_history, run_sweep

    configs = [_tiny_bp_config(mean_interarrival=ia, label=f"ia{ia:g}") for ia in (15.0, 25.0)]
    df = run_sweep(configs, "pds", out_dir=tmp_path, run_exact_model=True, run_pds_model=True,
                   run_dw_model=False, verbose=False)
    for _, row in df.iterrows():
        assert row["exact_status"] == "OPTIMAL" and row["pds_status"] == "OPTIMAL"
        assert row["pds_objective"] == pytest.approx(row["exact_objective"])
        assert row["pds_best_bound"] == pytest.approx(row["pds_objective"])
        assert row["pds_objective"] == pytest.approx(row["pds_arrived_all_total_sojourn"])
        assert row["pds_mean_sojourn_LB"] == pytest.approx(row["pds_best_bound"] / row["pds_n_optimized"])
        assert row["pds_master_bound"] <= row["pds_objective"] + 1e-6
        assert row["pds_root_lower_bound"] <= row["pds_objective"] + 1e-6
        assert row["pds_seed_objective"] >= row["pds_objective"] - 1e-6
        assert row["pds_compact_check_max_violation"] <= 1e-5
        assert row["pds_table_repairs"] == 0
        assert row["pds_within_tolerance"]
    table = comparison_table(df)
    assert {"pds", "pds_LB", "pds_status", "pds_runtime_s"} <= set(table.columns)
    meta = json.loads((tmp_path / "pds" / "run_meta.json").read_text())
    assert meta["stages"] == {"sim": True, "exact": True, "bp": False, "pds": True, "dw": False}
    record = json.loads((tmp_path / "pds" / "trials" / "trial_000.json").read_text())
    assert record["pds"]["status"] == "OPTIMAL" and "arrived_by_cohort" in record["pds"]
    history = load_bound_history("pds", out_dir=tmp_path, stage="pds")
    assert load_bound_history("pds", out_dir=tmp_path).empty  # no B&P in this run
    for trial_id, h in history.groupby("trial_id"):
        row = df.loc[df["trial_id"] == trial_id].iloc[0]
        lb, ub = h["lower_bound"].dropna(), h["upper_bound"].dropna()
        assert lb.is_monotonic_increasing and ub.is_monotonic_decreasing
        last = h.iloc[-1]
        assert last["event"] == "end"
        assert last["lower_bound"] == pytest.approx(row["pds_best_bound"])
        assert last["upper_bound"] == pytest.approx(row["pds_objective"])


def test_pds_tolerance_is_reported_like_the_other_solvers(tmp_path):
    """pds_tolerance_min = x stops within x minutes of mean sojourn and is
    reported as x * n_optimized minutes of total sojourn."""
    pytest.importorskip("numba")
    from experiments.objective_sweep import run_sweep

    cfg = _tiny_bp_config(mean_interarrival=12.0, delta=2.0, pds_tolerance_min=2.0)
    row = run_sweep([cfg], "pds_tol", out_dir=tmp_path, run_exact_model=False, run_pds_model=True,
                    run_dw_model=False, verbose=False).iloc[0]
    assert row["pds_status"] in ("OPTIMAL", "WITHIN_TOLERANCE")
    assert row["pds_tolerance_used"] == pytest.approx(2.0 * row["pds_n_optimized"])
    assert row["pds_within_tolerance"] and row["pds_mean_gap_min"] <= 2.0 + 1e-9
    assert row["pds_objective"] - row["pds_best_bound"] <= row["pds_tolerance_used"] + 1e-6
