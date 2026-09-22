"""
Worked example: how does the realized objective move against its bounds as
the simulation configuration changes?

Run with ``python -m experiments.run_objective_sweep``. As written it is four
trials that finish in well under a minute; edit ``SWEEP`` (or copy this file)
to scale it up.

Cost, measured on this station layout -- worth knowing before widening the
grid, because the two models scale very differently:

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
after the same wall clock.

.venv/Scripts/python.exe -m experiments.run_objective_sweep

from experiments.core import config_grid
from experiments.objective_sweep import (
    HR2MIN, TrialConfig, comparison_table, load_results, run_sweep,
)

BASE = TrialConfig(n_piles=1, n_connectors=2, n_modules=6,
                   warmup_period=6*HR2MIN)
configs = config_grid(BASE, mean_interarrival=[20.0, 30.0], delta=[2.0, 5.0])

df = run_sweep(configs, "my_run")          # returns the DataFrame directly
print(comparison_table(df, cohort="all"))

df = load_results("my_run")                # read it back later


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


# Everything not swept: station layout, warm-up, seeds, boundary handling.
BASE = TrialConfig(
    n_piles=1,
    n_connectors=2,
    n_modules=6,
    p_module=25.0,
    battery_cap_kwh=[75],
    warmup_period=6 * HR2MIN,  # long relative to max_time, to reach steady state
    max_time=2 * HR2MIN,
    # Every trial with this seed sees the SAME EV population, whichever axis
    # is swept: generate_arrivals draws inter-arrival gaps and vehicle
    # characteristics from separate streams, so vehicle k's battery/SoC
    # depend only on k. A mean_interarrival sweep therefore varies the
    # arrival process alone, instead of also resampling the fleet under it.
    seed=41,
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
)

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


def main() -> None:
    # configs = config_grid(BASE, **SWEEP)
    combos = [
        (60.0 / 6, 2.0, 1e-3, 0.1, 2 * 3600.0, 1800.0, 2 * HR2MIN, "6veh"),
        (60.0 / 5.5, 2.0, 1e-3, 0.1, 2 * 3600.0, 1800.0, 2 * HR2MIN, "5.5veh"),
        (60.0 / 5, 2.0, 1e-3, 0.1, 2 * 3600.0, 1800.0, 2 * HR2MIN, "5veh"),
        (60.0 / 4.5, 2.0, 1e-3, 0.1, 1.5 * 3600.0, 1800.0, 2 * HR2MIN, "4.5veh"),
        (60.0 / 4, 2.0, 1e-3, 0.1, 3600.0, 1800.0, 2 * HR2MIN, "4veh"),
        (60.0 / 3.5, 2.0, 1e-4, None, 1800.0, 1200.0, 3 * HR2MIN, "3.5veh"),
        (60.0 / 3, 2.0, 1e-4, None, 1800.0, 1200.0, 3 * HR2MIN, "3veh"),
        (60.0 / 2.5, 1.0, 1e-4, None, 1800.0, 1200.0, 4 * HR2MIN, "2.5veh"),
        (60.0 / 2, 1.0, 1e-4, None, 1800.0, 1200.0, 4 * HR2MIN, "2veh"),
        (60.0 / 1.5, 1.0, 1e-4, None, 1800.0, 1200.0, 4 * HR2MIN, "1.5veh"),
        (60.0 / 1, 1.0, 1e-4, None, 1800.0, 1200.0, 4 * HR2MIN, "1veh"),
    ]
    # To do: try with gap_tolerance_target_min = .1 (or remove it)
    # Set the time_limit for the first two to 2 hr.
    # Add more trials (in .5 veh intervals)
    # Try the case with 2piles
    RUN_NAME = "1pile_75kwh"
    configs = [
        dataclasses.replace(
            BASE,
            mean_interarrival=ia,
            delta=d,
            mip_gap=mip_gap,
            gap_tolerance_target_min=gap_tol,
            time_limit=t_lim,
            dw_time_limit=dw_t_lim,
            max_time=mx_time,
            label=lab,
        )
        for ia, d, mip_gap, gap_tol, t_lim, dw_t_lim, mx_time, lab in combos
    ]
    print(f"{len(configs)} trial(s) queued\n")
    # for cfg in configs:
    #     print(cfg)
    results = run_sweep(
        configs,
        run_sim=True,
        run_exact_model=True,
        run_dw_model=True,
        run_name=RUN_NAME,
        verbose=True,
        solver_progress=True,
    )

    # # Headline view: the four sources side by side on the widest cohort.
    # print("\n=== mean sojourn (minutes), cohort='all' ===")
    # print(comparison_table(results, cohort="all").to_string(index=False))

    # # `grid` is the like-for-like comparison, not `sim` -- the offline models
    # # can only depart on slot boundaries, so the continuous-time mean is not
    # # a target any grid schedule can hit.
    # gap = results["grid_all_mean_sojourn"] - results["dw_mean_sojourn_LB"]
    # print(f"\ngrid-FIFO above the DW lower bound by {gap.mean():.2f} min on average")


if __name__ == "__main__":
    main()
