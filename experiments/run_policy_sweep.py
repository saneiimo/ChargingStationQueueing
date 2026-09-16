"""
Policy sweep: race queue/power policies across a parameter grid.

This is ``compare_policies.ipynb`` swept over configurations and recorded to
disk. Each config runs every policy over ``n_reps`` replications with common
random numbers, then reports per-policy confidence intervals and a paired
difference interval for every policy pair.

HOW TO RUN
----------
From the repo root::

    .venv/Scripts/python.exe -m experiments.run_policy_sweep

Edit ``BASE`` (everything held constant) and ``SWEEP`` (the axes) below
first. Results land in ``experiments/results/<RUN_NAME>/``; open
``policy_sweep_results.ipynb`` in the repo root to load and plot them.

Or drive it from a notebook / REPL instead::

    from experiments.core import config_grid
    from experiments.policy_sweep import PolicyConfig, run_policy_sweep, metric_table

    base = PolicyConfig(n_piles=1, n_connectors=2, n_modules=6, n_reps=30)
    configs = config_grid(base, mean_interarrival=[10.0, 15.0, 20.0])
    results = run_policy_sweep(configs, run_name="my_run")
    print(metric_table(results, "avg sys time"))

COST
----
Pure simulation, so this is cheap compared to the objective sweep: roughly
``n_configs x n_policies x n_reps`` episodes, at a few milliseconds each on
these station sizes. The 6-config x 2-policy x 30-rep example below runs in
a few seconds. Cost grows with ``max_time`` and with load (a busier station
processes more events), not with the number of metrics.

WHICH POLICIES
--------------
Named in ``experiments.policy_sweep.config``:
``QUEUE_POLICIES`` = FIFO, LSoCD (lowest SoC difference), PMatch (closest
power match); ``POWER_POLICIES`` = Prop (proportional), Static (equal split
among currently plugged EVs), Constant (fixed per-connector share from pile
geometry alone -- an idle connector's modules are never lent to a busy one).
Sweeping both axes labels runs compoundly (``FIFO_Prop``, ``FIFO_Static``);
with one power policy the labels stay the queue names.
"""

from __future__ import annotations

from experiments.core import config_grid
from experiments.policy_sweep import (
    PolicyConfig,
    metric_table,
    run_policy_sweep,
    HR2MIN,
)

RUN_NAME = "queue_sweep_PMatch"

# Everything held constant across the sweep.
BASE = PolicyConfig(
    # station layout
    n_piles=1,
    n_connectors=2,
    n_modules=6,
    p_module=25.0,
    battery_cap_kwh=[50, 75, 100],
    queue_capacity=1000,
    # traffic
    warmup_period=6 * HR2MIN,  # long relative to max_time, to reach steady state
    max_time=24 * HR2MIN,  # measured window (minutes); None -> config.MAX_TIME
    delta_arr=None,  # None keeps continuous exponential arrival times
    # who races
    queue_policies=("FIFO", "PMatch"),  # "LSoCD", "PMatch"
    power_policies=("Prop",),
    # max-wait override, as in compare_policies.ipynb: FIFO runs with none,
    # then every other queue policy gets FIFO's observed mean max wait x the
    # factor. Set `max_wait=<float>` instead for a fixed value applied to
    # everyone, or leave both unset to disable the override entirely.
    max_wait_from="FIFO",  # "FIFO"
    max_wait_factor=0.65,  # 0.75
    # replication budget -- 30 paired replications per policy per config
    n_reps=30,
    seed0=1,
    confidence=0.95,
)

# The swept axes. The FIRST keyword varies slowest.
SWEEP = dict(
    mean_interarrival=[HR2MIN / (n_veh / 2) for n_veh in range(2, 13)],
)

# Other grids worth trying (uncomment one):
#
# Capacity study -- does the better policy depend on how loaded the station is?
# SWEEP = dict(
#     mean_interarrival=[10.0, 15.0, 25.0],
#     n_connectors=[2, 4],
# )
#
# Three-way race including the power policy as an axis:
# BASE = dataclasses.replace(BASE, queue_policies=("FIFO", "LSoCD", "PMatch"),
#                            power_policies=("Prop", "Static"))
# SWEEP = dict(mean_interarrival=[12.0, 20.0])


def main() -> None:
    configs = config_grid(BASE, **SWEEP)
    n_runs = sum(len(c.queue_policies) * len(c.power_policies) for c in configs)
    print(
        f"{len(configs)} config(s) x policies = {n_runs} policy run(s), "
        f"{n_runs * BASE.n_reps} episodes total\n"
    )

    results = run_policy_sweep(configs, run_name=RUN_NAME)

    # Headline: one metric, configs x policies.
    swept = next(iter(SWEEP))
    print("\n=== mean sojourn (min), by policy ===")
    print(metric_table(results, "avg sys time", index=swept).to_string())

    print("\n=== mean queue wait (min), by policy ===")
    print(metric_table(results, "avg wait time", index=swept).to_string())

    print(
        f"\nPer-pair significance is in comparisons.csv; open "
        f"policy_sweep_results.ipynb and set RUN_NAME = {RUN_NAME!r} to plot."
    )


if __name__ == "__main__":
    main()
