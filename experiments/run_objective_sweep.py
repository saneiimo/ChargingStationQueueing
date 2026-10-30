"""
Worked example: how does the realized objective move against its bounds as
the simulation configuration changes?

Run with ``python -m experiments.run_objective_sweep``. Edit ``SWEEP`` /
``main()``'s ``combos`` (or copy this file) to change what is run.

Each trial is a closed-system trial: FIFO warms the station up to ``t0``,
vehicles arrive over ``[t0, t0 + max_time]`` and none after, the simulation
drains the station, and every source -- simulation, grid replay, exact
MILP, branch-and-price, DW -- is run on that same closed set of vehicles.
The offline models are closed-system models (every vehicle served and fully
charged by the horizon); their horizon is read off the discretized FIFO run
(``inst_K``; ``horizon_slack`` adds to it). See ``experiments/README.md``,
"Closed-system trials".

Cost. Draining means ``K`` follows how long FIFO takes to empty the station,
not ``max_time``: an overloaded trial (a large warm-up backlog, arrivals
faster than service) can drain for many hours and give the models a horizon
several times longer than ``max_time``. The table below was measured on
this station layout for the older measured-window trials
(``K = max_time / delta``) -- read it as cost per slot of horizon:

    max_time  delta   J   K    exact      DW
        60     5.0    5   12   0.1-0.5s   0.2-0.5s
        60     2.0    5   30   0.8-4.3s   1.2-2.9s
       120     5.0   10   24   >120s *    2.5s
       120     2.0   10   60   >120s *    17s

``delta`` is the dominant knob (it sets K, hence the binary count), with the
vehicle count J close behind. (*) the exact MILP hit its ``time_limit`` on
both 120-minute rows and returned an incumbent plus a bound rather than a
proven optimum -- which the ``exact_status`` / ``exact_best_bound`` columns
report, so a limited run is still readable. On the ``delta=2`` row DW's
certified LB (401.8) was in fact *tighter* than the MILP's own bound (378.3)
after the same wall clock (figures measured when the objective was
``sum_j D_j`` in slots; every objective/bound column is now total sojourn in
minutes -- see ``run_meta.json``'s ``objective_units``).

Branch-and-price (``RUN_BP_MODEL`` below) solves the same exact model a
second way and fills ``bp_*`` columns: ``bp_objective`` (best schedule) and
``bp_best_bound`` (proven lower bound), equal when ``bp_status`` is OPTIMAL.
Its bound is typically far stronger than the MILP's at the same wall clock
(on a stored 12-vehicle, K=90 trial: 736.4 after 20 min vs the MILP's 717.4
after 30, both in the old slot units), but it does not always close the gap
on large instances; set
``bp_time_limit`` accordingly. ``bp_initial_schedule`` picks its starting
incumbent ("simulation", "exact", "both", "none" -- see ``TrialConfig``).

Price-directed state-space search (``RUN_PDS_MODEL``, ``offline_cl_PDS``)
solves the same exact model a third way and fills ``pds_*`` columns with the
same meaning: ``pds_objective`` / ``pds_best_bound``, equal when
``pds_status`` is OPTIMAL. Its bracket over time goes to
``pds_bound_history.csv`` (``load_bound_history(run, stage="pds")``).
``pds_time_limit`` and ``pds_tolerance_min`` control it.

``main()`` returns ``(results, history)``. ``history`` is B&P's certified
bracket on each trial's optimum over time (also saved as
``bound_history.csv``; empty unless ``RUN_BP_MODEL``): one row per change,
``trial_id``/``label`` plus ``time_s`` (since the bp stage started),
``lower_bound``/``upper_bound`` (total sojourn in minutes, NaN while unknown),
``lower_bound_min``/``upper_bound_min`` (the same as mean sojourn, minutes),
``gap`` and ``gap_pct`` (100*gap/upper_bound, i.e. relative to total
sojourn), ``nodes`` and ``event``.
Both bounds are step functions of ``time_s`` (constant between rows), and
the optimum of the horizon-K closed-system model always lies in
``[lower_bound, upper_bound]``.

.venv/Scripts/python.exe -m experiments.run_objective_sweep

from experiments.core import config_grid
from experiments.objective_sweep import (
    HR2MIN, TrialConfig, comparison_table, load_bound_history, load_results,
    run_sweep,
)

BASE = TrialConfig(n_piles=1, n_connectors=2, n_modules=6,
                   warmup_period=6*HR2MIN)
configs = config_grid(BASE, mean_interarrival=[20.0, 30.0], delta=[2.0, 5.0])

df = run_sweep(configs, "my_run")          # returns the DataFrame directly
df = run_sweep(configs, "my_bp_run", run_bp_model=True)   # + branch-and-price
print(comparison_table(df, cohort="all"))

df = load_results("my_run")                # read it back later

results, history = main()                  # or: from this module
df, history = run_sweep(configs, "my_bp_run", run_bp_model=True,
                        return_history=True)
history = load_bound_history("my_bp_run")  # read it back later
for label, h in history.groupby("label"):  # one bracket trajectory per trial
    print(label, h[["time_s", "lower_bound", "upper_bound", "gap_pct"]])


"""

from __future__ import annotations

from experiments.core import config_grid
from experiments.objective_sweep import (
    HR2MIN,
    TrialConfig,
    comparison_table,
    run_sweep,
)
import dataclasses

import pandas as pd


# Everything not swept: station layout, warm-up, seeds, boundary handling.
BASE = TrialConfig(
    n_piles=1,
    n_connectors=2,
    n_modules=6,
    p_module=25.0,
    battery_cap_kwh=[75],
    warmup_period=6 * HR2MIN,  # FIFO warm-up before t0, to reach steady state
    max_time=4 * HR2MIN,  # arrival window [t0, T]; the station is then drained
    # Closed-system trial: every source on the same vehicle set, simulation
    # drained, models' horizon K read off the discretized FIFO run. Needs
    # include_queued=True, a boundary_mode, objective_cohorts=COHORTS_ALL
    # (the defaults). horizon_slack lengthens K by that fraction; a longer K
    # can only lower the optimum (a schedule departing at K,
    # *_n_at_horizon > 0, hints the horizon may bind).
    # horizon_slack is a FRACTION of the FIFO horizon, not a horizon:
    #     K = ceil(K_fifo * (1 + horizon_slack))
    # where K_fifo is the slot by which the discretized FIFO schedule empties
    # the station. 0.0 -> exactly K_fifo; 0.2 -> 20% longer; 1.0 -> twice as
    # long. Any value >= 0 is valid (0.0-0.5 is a sensible range); each model
    # grows roughly linearly with K.
    horizon_slack=0.25,
    # Every trial with this seed sees the SAME EV population, whichever axis
    # is swept: generate_arrivals draws inter-arrival gaps and vehicle
    # characteristics from separate streams, so vehicle k's battery/SoC
    # depend only on k. A mean_interarrival sweep therefore varies the
    # arrival process alone, instead of also resampling the fleet under it.
    seed=37,
    # Cap both solvers so one hard trial cannot run away with the sweep. A
    # limit that bites is recorded, not hidden: see exact_status/dw_status.
    time_limit=1800.0,
    dw_time_limit=1200.0,
    # DW: lower bound only (price-and-branch is the expensive stage, and this
    # sweep compares against the LOWER bound).
    solve_integer_ub=False,
    mip_gap=1e-4,
    gap_tolerance=1e-6,
    gap_tolerance_target_min=None,
    # Branch-and-price (used only when RUN_BP_MODEL is True): wall-clock cap,
    # and where its first incumbent comes from. "both" takes the better of
    # the simulated FIFO schedule and the exact MILP's incumbent (the latter
    # only when RUN_EXACT_MODEL is also True).
    bp_time_limit=1800.0,
    bp_initial_schedule="none",
    # Optimality tolerances, in minutes of MEAN sojourn (None = prove the
    # optimum). The exact model / B&P stop as soon as their schedule is
    # proven within this many minutes of the true optimum (of the trial's
    # model: slot grid, whole modules, horizon K). Reported per stage as
    # exact_/bp_ tolerance_used (total minutes), mean_gap_min (proven gap)
    # and within_tolerance. A time limit that bites first leaves the stage
    # with within_tolerance=False.
    #
    # The tolerance only acts in whole steps of delta / n minutes, where n is
    # the number of vehicles in the trial (n_optimized). Every schedule's
    # total sojourn is a multiple of delta, so mean sojourns are delta / n
    # apart, and a tolerance x counts as floor(x / (delta / n)) steps:
    #   * x < delta / n   -> identical to None: the solver proves the optimum
    #     (it gains nothing from the tolerance);
    #   * otherwise       -> rounded DOWN to a whole number of steps, e.g. with
    #     delta = 2 min and n = 10 vehicles a step is 0.2 min, so x = 0.5
    #     behaves exactly like x = 0.4.
    # n is only known once the trial's instance is built (it depends on the
    # arrivals), so check exact_/bp_n_optimized in the results.
    exact_tolerance_min=None,
    bp_tolerance_min=1,
    # Price-directed state-space search (used only when RUN_PDS_MODEL is
    # True): wall-clock cap, and its optimality tolerance in minutes of MEAN
    # sojourn -- same meaning and same delta / n step rule as the two above
    # (pds_tolerance_used, pds_mean_gap_min, pds_within_tolerance). It starts
    # from the FIFO episode rebuilt on the slot grid.
    pds_time_limit=1800.0,
    pds_tolerance_min=None,
)

# Which stages run. The exact model can be solved as a compact MILP
# (RUN_EXACT_MODEL -> exact_* columns), by branch-and-price (RUN_BP_MODEL ->
# bp_* columns), by price-directed search (RUN_PDS_MODEL -> pds_* columns),
# or any combination; DW adds its continuous-module lower bound (dw_*).
RUN_EXACT_MODEL = False
RUN_BP_MODEL = False
RUN_PDS_MODEL = True
RUN_DW_MODEL = False

# The swept axes. First keyword varies slowest, so the cheap knobs cycle
# fastest and any early failure shows up on a cheap trial.
SWEEP = dict(
    mean_interarrival=[20.0, 30.0],
    delta=[2.0, 5.0],
)

# A heavier grid, once you have the cheap one working (see the cost table
# above -- expect the exact model to hit its time limit on the long windows):
#
# SWEEP = dict(
#     mean_interarrival=[15.0, 20.0, 30.0],
#     max_time=[1 * HR2MIN, 2 * HR2MIN, 4 * HR2MIN],
#     delta=[1.0, 2.0, 5.0],
# )

# .venv/Scripts/python.exe -m experiments.run_objective_sweep


def main() -> tuple[pd.DataFrame, pd.DataFrame]:
    # configs = config_grid(BASE, **SWEEP)
    coef = 1
    # One row per trial:
    #   (mean_interarrival, rho_P, delta, mip_gap, gap_tolerance_target_min,
    #    time_limit, dw_time_limit, max_time, label)
    # The arrival rate comes from EITHER of the first two fields:
    #   * mean_interarrival (minutes between arrivals): used as given; rho_P is
    #     then ignored (put None).
    #   * mean_interarrival = None and rho_P = target THEORETICAL offered power
    #     load (the theoretical counterpart of overall power utilization: the
    #     fraction of the station's rated power the arriving demand needs, e.g.
    #     0.8). The sweep sets mean_interarrival = 60 * E[W] / (rho_P * R),
    #     with E[W] the mean requested energy (kWh) of the battery/SoC
    #     distributions and R = n_piles * n_modules * p_module (kW). The value
    #     used is recorded in results.csv's mean_interarrival column, and the
    #     resulting load in sim_offered_power_load_theory (= rho_P).
    #   e.g. (None, 0.8, 2.0, 1e-4, None, 1800.0, 1200.0, 4 * HR2MIN, "rho0.8"),
    #  (mean_interarrival, rho_P, delta, mip_gap, bp_tolerance_min,
    #   time_limit, dw_time_limit, max_time, label, bp_initial_schedule)

    combos = [
        (None, 0.1, 1.0, 1e-4, None, 1800.0, 1200.0, 8 * HR2MIN, "rho0.1"),
        (None, 0.2, 1.0, 1e-4, None, 1800.0, 1200.0, 8 * HR2MIN, "rho0.2"),
        (None, 0.3, 1.0, 1e-4, None, 1800.0, 1200.0, 8 * HR2MIN, "rho0.3"),
        (None, 0.4, 1.0, 1e-4, None, 2 * 1800.0, 1200.0, 8 * HR2MIN, "rho0.4"),
        (None, 0.5, 1.0, 1e-4, None, 2 * 1800.0, 1200.0, 8 * HR2MIN, "rho0.5"),
        (None, 0.6, 1.0, 1e-4, 1, 2 * 1800.0, 1200.0, 8 * HR2MIN, "rho0.6"),
        (None, 0.7, 1.0, 1e-4, 1, 2 * 1800.0, 1200.0, 8 * HR2MIN, "rho0.7"),
        (None, 0.8, 1.0, 1e-4, 1, 3 * 1800.0, 1200.0, 8 * HR2MIN, "rho0.8"),
        (None, 0.9, 1.0, 1e-4, 1, 3 * 1800.0, 1200.0, 8 * HR2MIN, "rho0.9"),
        # (None, 0.93, 2.0, 1e-4, 1, 3 * 1800.0, 1200.0, 4 * HR2MIN, "rho0.93"),
        # (None, 0.96, 2.0, 1e-4, 1, 3 * 1800.0, 1200.0, 4 * HR2MIN, "rho0.96"),
    ]
    # To do: try with gap_tolerance_target_min = .1 (or remove it)
    # Set the time_limit for the first two to 2 hr.
    # Add more trials (in .5 veh intervals)
    # Try the case with 2piles
    RUN_NAME = "1pile_75kwh_pds"
    configs = [
        dataclasses.replace(
            BASE,
            mean_interarrival=ia,
            offered_power_load=rho_p,
            delta=d,
            mip_gap=mip_gap,
            bp_tolerance_min=bp_tol,
            time_limit=t_lim,
            dw_time_limit=dw_t_lim,
            max_time=mx_time,
            label=lab,
        )
        for ia, rho_p, d, mip_gap, bp_tol, t_lim, dw_t_lim, mx_time, lab in combos
    ]
    print(f"{len(configs)} trial(s) queued\n")
    # for cfg in configs:
    #     print(cfg)
    # `history`: B&P's lower/upper bound against time for every trial (see
    # the module docstring); also written to <run>/bound_history.csv.
    results, history = run_sweep(
        configs,
        run_sim=True,
        run_exact_model=RUN_EXACT_MODEL,
        run_bp_model=RUN_BP_MODEL,
        run_pds_model=RUN_PDS_MODEL,
        run_dw_model=RUN_DW_MODEL,
        run_name=RUN_NAME,
        verbose=True,
        solver_progress=True,
        return_history=True,
    )

    # # Headline view: every source side by side on the widest cohort.
    # print("\n=== mean sojourn (minutes), cohort='all' ===")
    # print(comparison_table(results, cohort="all").to_string(index=False))

    # # `grid` is the like-for-like comparison, not `sim` -- the offline models
    # # can only depart on slot boundaries, so the continuous-time mean is not
    # # a target any grid schedule can hit.
    # gap = results["grid_all_mean_sojourn"] - results["dw_mean_sojourn_LB"]
    # print(f"\ngrid-FIFO above the DW lower bound by {gap.mean():.2f} min on average")

    return results, history


if __name__ == "__main__":
    main()
