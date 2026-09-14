"""
Experiment drivers. Two sweeps, one shared core.

* ``experiments.objective_sweep`` -- how far is a realized episode from the
  true optimum? One episode per configuration, bounded by the exact MILP
  and by Dantzig-Wolfe. Start at ``run_sweep``.
* ``experiments.policy_sweep`` -- which queue/power policy is better, and
  by how much? Monte-Carlo replications per configuration with common
  random numbers and paired confidence intervals. Start at
  ``run_policy_sweep``.
* ``experiments.core`` -- the grid builder and the on-disk run layout both
  of them write through.

Runnable entry points live at the top level so the module paths stay short::

    python -m experiments.run_objective_sweep
    python -m experiments.run_policy_sweep

The names below are re-exported for convenience, and are what the
result-loading notebooks import. Objective-sweep names are unprefixed for
backward compatibility; policy-sweep names keep their own module path when
they would otherwise collide (both sweeps have a ``load_results``).
"""

from __future__ import annotations

from . import core, objective_sweep, policy_sweep
from .core import DEFAULT_OUT_DIR, config_grid, list_runs, load_meta, load_table
from .objective_sweep import (
    HR2MIN,
    TrialConfig,
    TrialResult,
    build_instance,
    comparison_table,
    episode_counts,
    load_results,
    run_dw,
    run_episode,
    run_exact,
    run_sweep,
    run_trial,
    simulation_metrics,
)
from .policy_sweep import (
    PolicyConfig,
    compare_policies,
    policy_grid,
    run_policy_sweep,
    run_replications,
    summarize_ci,
)

__all__ = [
    # sub-packages
    "core",
    "objective_sweep",
    "policy_sweep",
    # shared
    "config_grid",
    "DEFAULT_OUT_DIR",
    "load_table",
    "load_meta",
    "list_runs",
    # objective sweep
    "TrialConfig",
    "HR2MIN",
    "TrialResult",
    "run_trial",
    "run_episode",
    "episode_counts",
    "simulation_metrics",
    "build_instance",
    "run_exact",
    "run_dw",
    "run_sweep",
    "load_results",
    "comparison_table",
    # policy sweep
    "PolicyConfig",
    "policy_grid",
    "run_policy_sweep",
    "run_replications",
    "summarize_ci",
    "compare_policies",
]
