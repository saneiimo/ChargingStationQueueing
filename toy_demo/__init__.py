"""
Toy / presentation demos with fixed EVs and fixed arrival times.

This package does not change the core simulator. It builds a normal
``ChargingStationEnv``, then replaces the engine's random arrival sampler
with an explicit EV list before ``reset``.
"""

from __future__ import annotations

from .runner import finished_ev_table, metrics_table, run_toy_episode
from .scenario import ToyEVSpec, ToyStationSpec, build_evs, default_four_ev_specs

__all__ = [
    "ToyEVSpec",
    "ToyStationSpec",
    "build_evs",
    "default_four_ev_specs",
    "run_toy_episode",
    "metrics_table",
    "finished_ev_table",
]
