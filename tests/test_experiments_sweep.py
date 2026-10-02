"""
The objective sweep's reporting semantics (``experiments`` package).

Validates the two things a sweep row's meaning rests on:

  1. Completion is decided by ENERGY, not by ``departure_slot < K``. The
     departure slot cannot tell a vehicle that finished exactly at the
     horizon (``D_j = K``, fully charged) from one that was never served
     (``D_j = K``, nothing delivered) -- see ``_model_groups``' own
     docstring for why, and Section 6.3 for the censoring that creates the
     ambiguity.
  2. A boundary vehicle's completion test must add back the energy it
     arrived with, or every completed boundary vehicle reads as short by
     exactly that amount.
  3. The branch-and-price stage (``run_bp_model``) writes its ``bp_*``
     columns and agrees with the exact MILP wherever both prove optimality.

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


def _solve_single(s_f: float, *, tie_break: bool):
    """One vehicle alone on a one-connector station, K = 12 slots."""
    station = StationSpec(n_piles=1, n_connectors=1, n_modules=4, p_module=25.0)
    v = VehicleData(id=0, a=0.0, Q=100.0, s_i=0.15, s_f=s_f, s_th=0.4, p_max=100.0)
    cl = build_cl_model(
        [v], station, delta=5.0, horizon_minutes=60.0, tie_break=tie_break
    )
    solve_cl_model(cl, mip_gap=1e-6, verbose=False)
    return extract_solution(cl)


def test_completed_at_horizon_is_not_censored():
    """A vehicle that finishes exactly at the horizon must count as
    completed, even though its ``departure_slot`` equals K.

    ``tie_break=True`` is what makes this reproducible: with the plain
    total-sojourn objective the solver is *indifferent* at ``D_j = K``
    (charging through the last slot and abandoning the vehicle cost the
    same), so it may return either. The tie-break's front-loading term
    settles it, and the resulting solution is a genuine counterexample to
    ``departure_slot < K`` as a completion test.
    """
    sol = _solve_single(0.82, tie_break=True)
    row = sol.per_vehicle.iloc[0]
    delivered = row["energy_kwh"] + row["energy_delivered_before_kwh"]

    # The setup only exercises anything if it really lands on the boundary.
    assert row["departure_slot"] == pytest.approx(sol.K), "no longer at the horizon"
    assert delivered == pytest.approx(row["energy_required_kwh"], abs=1e-6)

    completed, arrived, diag = _model_groups(sol.per_vehicle, sol.K)
    assert diag["n_completed"] == 1, "energy-complete vehicle was reported censored"
    assert diag["n_censored"] == 0
    assert diag["n_completed_at_horizon"] == 1  # the horizon is binding
    assert completed["all"]["n"] == 1.0
    assert arrived["all"]["n"] == 1.0
    assert arrived["all"]["n_censored"] == 0.0
    print("  PASS completed-at-horizon classified as completed (D_j == K)")


def test_unfinished_at_horizon_is_censored():
    """Same ``D_j = K``, but short of its energy -- must be censored. Pairs
    with the test above: identical departure slot, opposite verdict, so
    only the energy test can separate them."""
    sol = _solve_single(0.85, tie_break=True)
    row = sol.per_vehicle.iloc[0]
    delivered = row["energy_kwh"] + row["energy_delivered_before_kwh"]

    assert row["departure_slot"] == pytest.approx(sol.K)
    assert delivered < row["energy_required_kwh"] - 1e-6

    completed, arrived, diag = _model_groups(sol.per_vehicle, sol.K)
    assert diag["n_completed"] == 0
    assert diag["n_censored"] == 1
    assert diag["n_completed_at_horizon"] == 0
    assert completed["all"]["n"] == 0.0
    assert arrived["all"]["n"] == 1.0
    assert arrived["all"]["n_censored"] == 1.0
    # An empty group reports nan, not 0.0 -- a zero mean sojourn would read
    # as perfect service rather than as no data.
    assert completed["all"]["mean_sojourn"] != completed["all"]["mean_sojourn"]
    print("  PASS unfinished-at-horizon classified as censored (same D_j == K)")


def test_boundary_vehicle_completion_counts_pre_window_energy():
    """``energy_required_kwh`` is a boundary vehicle's FULL requirement while
    ``energy_kwh`` covers only the modeled window, so the test has to add
    ``energy_delivered_before_kwh`` back. Without that, a boundary vehicle
    that finished would read as short by exactly its pre-window energy."""
    per_vehicle = pd.DataFrame(
        [
            {  # boundary vehicle: 30 kWh before the window, 20 kWh inside it
                "vehicle_id": 0,
                "cohort": "boundary",
                "sojourn_min": 10.0,
                "departure_slot": 5.0,
                "energy_kwh": 20.0,
                "energy_delivered_before_kwh": 30.0,
                "energy_required_kwh": 50.0,
            },
            {  # ordinary vehicle, genuinely short
                "vehicle_id": 1,
                "cohort": "measurement",
                "sojourn_min": 20.0,
                "departure_slot": 12.0,
                "energy_kwh": 10.0,
                "energy_delivered_before_kwh": 0.0,
                "energy_required_kwh": 40.0,
            },
        ]
    )
    completed, arrived, diag = _model_groups(per_vehicle, K=12)

    assert diag["n_completed"] == 1 and diag["n_censored"] == 1
    # The boundary vehicle counts only at the "all" level, and it is the
    # completed one; the measurement-level vehicle is the censored one.
    assert completed["all"]["n"] == 1.0
    assert completed["measurement"]["n"] == 0.0
    assert arrived["measurement"]["n_censored"] == 1.0
    print("  PASS boundary completion adds pre-window energy back")


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
    """Both exact stages on small windows: same proven optimum, bp_* columns,
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
    assert meta["stages"] == {"sim": True, "exact": True, "bp": True, "dw": True}
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
