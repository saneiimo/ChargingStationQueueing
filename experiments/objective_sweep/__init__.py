"""
Objective sweep: how far is a realized episode from the true optimum?

For each configuration, simulate a measured window and bound its objective
two ways -- the connector-lane MILP (an incumbent plus the solver's own
bound) and Dantzig-Wolfe (a certified lower bound) -- then store everything.
This is ``sim_benchmark.ipynb`` run headless and recorded.

Start at ``run_sweep``; see ``experiments/README.md`` for the column
conventions and the two reporting groups (``completed`` / ``arrived``).
"""

from __future__ import annotations

from experiments.core.grid import config_grid

from .config import HR2MIN, TrialConfig
from .sweep import comparison_table, flatten_trial, load_results, run_sweep
from .trial import (
    TrialResult,
    build_instance,
    episode_counts,
    episode_utilization,
    replay_episode,
    run_dw,
    run_episode,
    run_exact,
    run_trial,
    simulation_metrics,
)

__all__ = [
    "TrialConfig",
    "HR2MIN",
    "config_grid",
    "TrialResult",
    "run_trial",
    "run_episode",
    "replay_episode",
    "episode_utilization",
    "episode_counts",
    "simulation_metrics",
    "build_instance",
    "run_exact",
    "run_dw",
    "run_sweep",
    "flatten_trial",
    "load_results",
    "comparison_table",
]
