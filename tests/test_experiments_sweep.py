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
    ``sum_j D_j`` objective the solver is *indifferent* at ``D_j = K``
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
