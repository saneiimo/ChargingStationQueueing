"""
One sweep point: every knob a single benchmark trial needs.

``TrialConfig`` holds the station layout, the traffic/simulation settings and
the two solvers' tolerances in one flat, hashable-ish record, so a trial is
fully described (and reproducible) by the row it writes to ``results.csv``.
``config_grid`` builds the Cartesian product of whichever fields you want to
sweep; pass explicit ``TrialConfig`` objects instead when the combinations
are not a product.

Closed-system trials
--------------------
Every source -- the simulation, its grid replay and the offline models --
is run on the same closed set of vehicles from the same starting state:

1. FIFO warms the station up over ``[0, t0]``, ``t0 = warmup_period``.
2. The state at ``t0`` is snapshotted: the queue and the vehicles plugged
   in. The optimizer controls both (they are in the model with ``a=0``).
3. Vehicles arrive over ``[t0, T]``, ``T = t0 + max_time`` (an arrival at
   exactly ``T`` is admitted); none arrives after ``T``.
4. The simulation keeps serving until the station is empty (it is not cut
   off at ``T``), and records that end time.
5. The offline models get exactly that set: the queue and the plugged-in
   vehicles at ``t0`` plus every arrival in ``[t0, T]``. Their horizon is
   read off the discretized FIFO run -- the slot by which it empties the
   station -- plus ``horizon_slack``.

Mean sojourn is reported over every vehicle in the set. The offline models
are closed-system models (every vehicle served and fully charged within the
horizon), which is why they need the drained run's horizon rather than
``max_time``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from experiments.core.grid import config_grid
from metrics.load import mean_interarrival_for_power_load
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
    # Arrival rate: give EITHER mean_interarrival (minutes between arrivals,
    # lambda = 1 / mean_interarrival) OR offered_power_load (rho_P, the
    # theoretical counterpart of overall power utilization: the fraction of
    # the station's rated power the arriving demand needs; see
    # metrics/README.md, 6.3). With mean_interarrival=None it is derived from
    # offered_power_load as 60 * E[W] / (rho_P * R) (E[W]: mean requested
    # kWh of the battery/SoC distributions; R = n_piles * n_modules *
    # p_module); with mean_interarrival set, offered_power_load is ignored.
    # The value actually used is effective_mean_interarrival, and is what
    # the mean_interarrival column of results.csv records.
    mean_interarrival: float | None = 30.0
    offered_power_load: float | None = None
    # Length of the arrival window [t0, T] after warm-up (minutes). The
    # station is then drained, so the measured phase runs past T.
    max_time: float = 2 * HR2MIN
    warmup_period: float = 6 * HR2MIN  # FIFO warm-up before t0
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
    # Extra horizon, as a fraction of the FIFO horizon K_fifo
    # (K = ceil(K_fifo * (1 + horizon_slack))). Every model must finish
    # every vehicle by K, so K_fifo (which the discretized FIFO schedule
    # fits) keeps the models feasible; a longer K can only lower the optimum
    # (more schedules fit), and stops changing it once K reaches the
    # makespan of some optimal schedule. A model schedule departing at K
    # (*_n_at_horizon > 0) hints the horizon may bind. Larger K grows every
    # model roughly linearly.
    horizon_slack: float = 0.0
    # Arrival-time grid. ``"delta"`` snaps arrivals to the model's own grid
    # (what the notebook does); ``None`` keeps continuous exponential times;
    # a number snaps to that grid instead.
    delta_arr: float | str | None = "delta"

    # --- boundary conditions at t0 (every offline model) ------------------
    # A closed-system trial needs all of: include_queued=True, a boundary
    # mode (OPTIMIZE -- the optimizer controls them -- or FIXED), no flush,
    # and objective_cohorts=COHORTS_ALL; otherwise the models would not see
    # the same vehicle set as the simulation (checked in __post_init__).
    # The fields stay so the instance builder's options remain visible.
    #
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
    tie_break: bool = False
    # Stop as soon as the incumbent is proven within this many minutes of
    # MEAN sojourn of the optimum (converted per trial to x * n_optimized
    # minutes of total sojourn, Gurobi's MIPGapAbs). None: prove optimality.
    # Either way the free tightening applies (objective values are multiples
    # of delta), so only whole steps of delta / n_optimized count. Needs
    # tie_break=False. mip_gap (relative) still applies; Gurobi stops at
    # whichever gap is met first, so keep mip_gap small to let this govern.
    exact_tolerance_min: float | None = None

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
    # Same as exact_tolerance_min, for branch-and-price: stop as soon as its
    # schedule is proven within this many minutes of mean sojourn of the
    # optimum (status "WITHIN_TOLERANCE"). None: prove optimality.
    bp_tolerance_min: float | None = None

    # --- bookkeeping -------------------------------------------------------
    label: str = ""  # optional human-readable tag carried into the results
    notes: str = ""

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        # Neither rate input set is allowed here: a sweep's BASE leaves the
        # swept one to config_grid. effective_mean_interarrival raises if a
        # config is used with neither.
        if self.mean_interarrival is not None and self.mean_interarrival <= 0:
            raise ValueError(f"mean_interarrival must be > 0, got {self.mean_interarrival}")
        if self.offered_power_load is not None and self.offered_power_load <= 0:
            raise ValueError(f"offered_power_load must be > 0, got {self.offered_power_load}")
        if self.horizon_slack < 0:
            raise ValueError(f"horizon_slack must be >= 0, got {self.horizon_slack}")
        for name in ("exact_tolerance_min", "bp_tolerance_min"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} must be >= 0 or None, got {value}")
        if self.exact_tolerance_min is not None and self.tie_break:
            raise ValueError("exact_tolerance_min needs tie_break=False (one objective)")
        problems = []
        if not self.include_queued:
            problems.append("include_queued=False drops the queue at t0 from the models")
        if self.boundary_mode is None:
            problems.append("boundary_mode=None drops the vehicles plugged in at t0 from the models")
        if self.flush_queue_at_warmup:
            problems.append("flush_queue_at_warmup=True removes the queue at t0 from the simulation")
        if self.objective_cohorts != COHORTS_ALL:
            problems.append("objective_cohorts must be COHORTS_ALL so every vehicle is optimized")
        if problems:
            raise ValueError(
                "a closed-system trial runs every source on the same vehicle set, but: "
                + "; ".join(problems)
                + "."
            )

    @property
    def effective_mean_interarrival(self) -> float:
        """
        The mean inter-arrival time (minutes) the trial uses:
        ``mean_interarrival`` when given, else the one that makes the
        theoretical offered power load equal ``offered_power_load``
        (``metrics.load.mean_interarrival_for_power_load``). Computed from
        the inputs every time, so ``dataclasses.replace`` of either field
        always takes effect.
        """
        if self.mean_interarrival is not None:
            return float(self.mean_interarrival)
        if self.offered_power_load is None:
            raise ValueError("set mean_interarrival or offered_power_load (one of them is needed)")
        return mean_interarrival_for_power_load(
            self.offered_power_load,  # type: ignore[arg-type]
            rated_power=self.n_piles * self.n_modules * self.p_module,
            battery_cap_options=self.battery_cap_options,
        )

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

    def to_row(self) -> dict[str, Any]:
        """Flat, CSV-friendly view -- one column per knob, enums as strings."""
        row = dataclasses.asdict(self)
        # The rate actually used, whichever input set it.
        row["mean_interarrival"] = self.effective_mean_interarrival
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
