"""
Policy sweep: which queue/power policy wins, and by how much?

For each configuration, race the policies over Monte-Carlo replications
with common random numbers, then report per-policy confidence intervals
and paired-difference intervals for every pair. This is
``compare_policies.ipynb`` swept over a parameter grid and recorded.

Start at ``run_policy_sweep``; ``replications`` holds the underlying
single-scenario machinery (``run_replications``, ``summarize_ci``,
``compare_policies``), which is still usable on its own.
"""

from __future__ import annotations

from experiments.core.grid import config_grid

from .config import (
    HR2MIN,
    POWER_POLICIES,
    QUEUE_POLICIES,
    PolicyConfig,
    build_policies,
    policy_grid,
)
from .replications import (
    DEFAULT_METRICS,
    DEFAULT_METRICS_POST_WARMUP,
    compare_policies,
    labeled_policy_grid,
    run_replications,
    scenario_run_names,
    summarize_ci,
)
from .sweep import (
    load_results,
    metric_slug,
    metric_table,
    run_policy_sweep,
    run_policy_trial,
)

__all__ = [
    "PolicyConfig",
    "config_grid",
    "QUEUE_POLICIES",
    "POWER_POLICIES",
    "policy_grid",
    "build_policies",
    "run_policy_sweep",
    "run_policy_trial",
    "load_results",
    "metric_table",
    "metric_slug",
    # the underlying single-scenario machinery
    "run_replications",
    "summarize_ci",
    "compare_policies",
    "labeled_policy_grid",
    "scenario_run_names",
    "DEFAULT_METRICS",
    "DEFAULT_METRICS_POST_WARMUP",
]
