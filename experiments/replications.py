"""
Monte Carlo replications for a charging-station scenario.

Runs the same env settings many times with different seeds and reports
metric means with two-sided normal confidence intervals.
"""

from __future__ import annotations

from math import sqrt
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd

from env.charging_env import ChargingStationEnv
from policy.queue.fifo import FIFOQueuePolicy
from policy.queue.base import QueuePolicy


# Display name -> extractor(env) after one finished episode.
DEFAULT_METRICS: dict[str, Callable[[ChargingStationEnv], float]] = {
    "finished EVs": lambda e: float(len(e.engine.metrics.finished_evs)),
    "dropped EVs": lambda e: float(len(e.engine.metrics.dropped_evs)),
    "total energy delivered (kWh)": lambda e: e.engine.metrics.total_energy(),
    "average L": lambda e: e.engine.metrics.average_L(),
    "average Q": lambda e: e.engine.metrics.average_Q(),
    "avg wait time": lambda e: e.engine.metrics.mean_wait(),
    "max wait time": lambda e: e.engine.metrics.max_wait(),
    "avg sys time": lambda e: e.engine.metrics.mean_sojourn(),
}


def _mean_ci(samples: np.ndarray, confidence: float) -> tuple[float, float, float]:
    """Return (mean, CI lower, CI upper) using a normal critical value."""
    x = np.asarray(samples, dtype=float)
    n = x.size
    mean = float(x.mean()) if n else 0.0
    if n < 2:
        return mean, mean, mean
    se = float(x.std(ddof=1) / sqrt(n))
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    half = z * se
    return mean, mean - half, mean + half


def _run_episode(env: ChargingStationEnv, policy: QueuePolicy, seed: int) -> None:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        mask = env.action_masks()
        ev, pile_id = policy.decide(obs, mask, rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)


def run_replications(
    scenario: dict[str, Any],
    n_reps: int = 30,
    metrics: list[str] | None = None,
    confidence: float = 0.95,
    seed0: int = 0,
) -> pd.DataFrame:
    """
    Replicate one scenario and return a table: mean | ci_low | ci_high.

    Parameters
    ----------
    scenario :
        Keyword args for ``ChargingStationEnv``, plus optional ``policy``
        (a ``QueuePolicy`` instance). Defaults to ``FIFOQueuePolicy``.
    n_reps :
        Number of independent episodes.
    metrics :
        Display names from ``DEFAULT_METRICS`` (or any registered keys).
        Defaults to all default metrics.
    confidence :
        Two-sided confidence level in (0, 1), e.g. 0.95.
    seed0 :
        First replication seed; replication r uses ``seed0 + r``.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    metric_names = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    unknown = [m for m in metric_names if m not in DEFAULT_METRICS]
    if unknown:
        raise KeyError(
            f"Unknown metrics: {unknown}. Choose from {list(DEFAULT_METRICS)}"
        )

    scenario = dict(scenario)
    policy: QueuePolicy = scenario.pop("policy", None) or FIFOQueuePolicy()
    env = ChargingStationEnv(**scenario)

    samples = {name: [] for name in metric_names}
    for r in range(n_reps):
        _run_episode(env, policy, seed=seed0 + r)
        for name in metric_names:
            samples[name].append(DEFAULT_METRICS[name](env))

    rows = []
    for name in metric_names:
        mean, lo, hi = _mean_ci(np.asarray(samples[name]), confidence)
        rows.append({"metric": name, "mean": mean, "ci_low": lo, "ci_high": hi})

    return pd.DataFrame(rows).set_index("metric")


if __name__ == "__main__":
    table = run_replications(
        scenario={
            "n_piles": 2,
            "n_nozzles": 4,
            "n_bricks": 7,
            "p_brick": 25,
            "queue_capacity": 10,
            "lam": 5.0,
            "policy": FIFOQueuePolicy(),
        },
        n_reps=30,
        confidence=0.95,
    )
    print(table.to_string(float_format=lambda x: f"{x:.4f}"))
