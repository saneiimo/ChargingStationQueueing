"""
One sweep point: every knob a single benchmark trial needs.

``TrialConfig`` holds the station layout, the traffic/simulation settings and
the two solvers' tolerances in one flat, hashable-ish record, so a trial is
fully described (and reproducible) by the row it writes to ``results.csv``.
``config_grid`` builds the Cartesian product of whichever fields you want to
sweep; pass explicit ``TrialConfig`` objects instead when the combinations
are not a product.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from experiments.core.grid import config_grid
from offline_cl_opt.boundary import (
    COHORTS_ALL,
    BoundaryMode,
    Cohort,
)

HR2MIN = 60


@dataclass(frozen=True)
class TrialConfig:
    """
    A single (simulation, exact model, branch-and-price, DW model) run.

    Any field can be swept via ``config_grid``. Fields are grouped below by
    what they control; the defaults reproduce the ``sim_benchmark.ipynb``
    walkthrough.
    """

    # --- station layout (shared by the simulation and both offline models) --
    n_piles: int = 1
    n_connectors: int = 2
    n_modules: int = 6
    p_module: float = 25.0
    queue_capacity: int = 1000

    # --- traffic / simulation ----------------------------------------------
    mean_interarrival: float = 30.0
    max_time: float = 2 * HR2MIN  # length of the MEASURED phase (minutes)
    warmup_period: float = 6 * HR2MIN  # run before the measured phase
    flush_queue_at_warmup: bool = False
    battery_cap_kwh: tuple[float, ...] = (75.0,)  # kWh; converted on use
    seed: int = 39  # arrival stream
    policy_seed: int = 12  # queue-policy RNG
    # Oversampling factor for the one-shot arrival draw: the stream must cover
    # warm-up + measured phase with room to spare.
    #
    # Nothing here needs to pin the draw horizon for the sake of common
    # random numbers. `simulation.arrivals.generate_arrivals` gives gaps and
    # vehicle characteristics their own streams (see that module's docstring,
    # "One stream per source of randomness"), so vehicle k's battery/SoC
    # depend only on k. Widening the horizon therefore only appends arrivals:
    # every trial with the same `seed` already shares one EV population,
    # whatever `max_time`, `warmup_period` or `mean_interarrival` are swept
    # to. This replaces the old `arrival_horizon` knob, which existed solely
    # to work around the two sources sharing a stream.
    arrival_oversample: float = 1.5

    # --- discretization ----------------------------------------------------
    delta: float = 2.0  # slot length (minutes) for both offline models
    # Arrival-time grid. ``"delta"`` snaps arrivals to the model's own grid
    # (what the notebook does); ``None`` keeps continuous exponential times;
    # a number snaps to that grid instead.
    delta_arr: float | str | None = "delta"

    # --- measured-window boundary conditions (both offline models) ---------
    # Fold the boundary queue in as ordinary a=0 vehicles (cohort QUEUED).
    include_queued: bool = True
    # How to treat vehicles already plugged in at the boundary (cohort
    # BOUNDARY): None excludes them, FIXED pins them, OPTIMIZE lets the
    # optimizer control their future power/departure but not their lane.
    boundary_mode: BoundaryMode | None = BoundaryMode.OPTIMIZE
    # Which cohorts the objective is minimised over -- independent of the two
    # flags above: an excluded-from-objective vehicle is still modelled in
    # full (holds its connector, draws modules), its D_j just carries no
    # weight. Normally exclude BOUNDARY under FIXED, whose sojourn is a constant.
    objective_cohorts: frozenset[Cohort] = COHORTS_ALL

    # --- exact model -------------------------------------------------------
    # Gurobi's relative MIPGap. The objective is total sojourn (minutes), so
    # this is a fraction of total sojourn.
    mip_gap: float | None = 1e-4
    time_limit: float | None = 1200.0  # seconds; None for no limit
    break_symmetry: bool = False  # must stay False with boundary vehicles
    bound_departures: bool = True
    tie_break: bool = False

    # --- DW model ----------------------------------------------------------
    # Objective units: minutes of TOTAL sojourn. A tolerance of g minutes
    # corresponds to g/J minutes of mean-sojourn uncertainty, where J is
    # n_optimized (the vehicle count in objective_cohorts) -- see
    # gap_tolerance_target_min below for setting this from a per-vehicle
    # minutes target instead.
    gap_tolerance: float = 1e-6
    # Set this INSTEAD of gap_tolerance to target a mean-sojourn uncertainty
    # of m minutes directly. At solve time
    # gap_tolerance is computed as ``m * n_optimized``, where
    # n_optimized is read off the actual instance for this trial (it depends
    # on include_queued/boundary_mode/objective_cohorts, so it cannot be
    # known before the episode runs and the instance is built -- this field
    # exists precisely so you don't have to compute it by hand per trial).
    # None (default): use the raw gap_tolerance field above instead.
    gap_tolerance_target_min: float | None = 1
    dw_time_limit: float | None = 1200.0  # column-generation wall clock, seconds
    gamma: float = 0.0  # dual smoothing
    purge_every: int | None = None
    max_iterations: int = 500
    # Price-and-branch (the integer upper bound) is off by default: this sweep
    # compares against the certified LOWER bound, and the integer master is
    # the expensive stage.
    solve_integer_ub: bool = False

    # --- branch-and-price (offline_cl_PB) ----------------------------------
    # The SAME whole-module model as the exact MILP, solved by branch-and-
    # price: a proven optimum, or at the time limit a certified bracket
    # [bp_best_bound, bp_objective]. Only used when the sweep runs with
    # run_bp_model=True.
    bp_time_limit: float | None = 1800.0  # seconds; None for no limit
    # Initial incumbent handed to B&P (it always builds its own greedy list
    # schedule too, and keeps whichever is best). Every candidate is a full
    # schedule re-validated against the exact model, never a bare number, so
    # a bad seed can only be rejected -- it can never produce a false upper
    # bound:
    #   "simulation" -- this trial's FIFO episode rebuilt on the slot grid
    #                   (offline_cl_PB.schedule_from_simulation);
    #   "exact"      -- the exact MILP's incumbent from the same trial (needs
    #                   run_exact_model=True; otherwise nothing is seeded);
    #   "both"       -- the better of the two;
    #   "none"       -- only B&P's own greedy schedule.
    bp_initial_schedule: str = "simulation"

    # --- bookkeeping -------------------------------------------------------
    label: str = ""  # optional human-readable tag carried into the results
    notes: str = ""

    # ------------------------------------------------------------------ #
    @property
    def battery_cap_options(self) -> list[float]:
        """Battery capacities in the engine's native kW*min units."""
        return [c * HR2MIN for c in self.battery_cap_kwh]

    @property
    def resolved_delta_arr(self) -> float | None:
        """``delta_arr`` with the ``"delta"`` sentinel resolved."""
        return self.delta if self.delta_arr == "delta" else self.delta_arr  # type: ignore[return-value]

    @property
    def draw_horizon(self) -> float:
        """
        Horizon the arrival stream is sampled over: warm-up plus measured
        phase, with ``arrival_oversample`` headroom.

        Free to vary across a sweep -- a wider horizon only appends later
        arrivals, it never disturbs the ones already drawn or the vehicles
        attached to them. See ``arrival_oversample``.
        """
        return (self.warmup_period + self.max_time) * self.arrival_oversample

    @property
    def offline_horizon(self) -> float:
        """
        Completion horizon for both offline models: the MEASURED phase only,
        not warm-up + measured (the models start at the boundary, t=0).
        """
        return float(self.max_time)

    def to_row(self) -> dict[str, Any]:
        """Flat, CSV-friendly view -- one column per knob, enums as strings."""
        row = dataclasses.asdict(self)
        row["battery_cap_kwh"] = ",".join(f"{c:g}" for c in self.battery_cap_kwh)
        row["boundary_mode"] = (
            self.boundary_mode.value if self.boundary_mode is not None else "none"
        )
        row["objective_cohorts"] = "+".join(
            sorted(c.value for c in self.objective_cohorts)
        )
        row["delta_arr"] = "delta" if self.delta_arr == "delta" else self.delta_arr
        return row

# ``config_grid`` lives in experiments.core.grid (both pipelines use it);
# re-exported here so ``from experiments.objective_sweep.config import
# config_grid`` keeps working.
__all__ = ["TrialConfig", "config_grid", "HR2MIN"]
