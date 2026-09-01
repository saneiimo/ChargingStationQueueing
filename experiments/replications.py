"""
Monte Carlo replications for charging-station scenarios.

Typical workflow (common random numbers across policies via the same seed0):

1. ``labeled_policy_grid`` — names for a queue × power grid (compound names
   only when both lists have more than one entry).
2. ``run_replications`` — one scenario, raw metric value per replication.
3. ``summarize_ci`` — mean and two-sided CI over those replications.
4. ``compare_policies`` — paired differences; returns a sign table (-1/0/1)
   plus nested per-pair detail DataFrames with means and difference CIs.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt
from statistics import NormalDist
from typing import Any, Callable

import numpy as np
import pandas as pd

from env.charging_env import ChargingStationEnv
from policy.power.base import PowerPolicy
from policy.power.proportional import ProportionalPower
from policy.queue.fifo import FIFOQueuePolicy
from policy.queue.base import QueuePolicy


# Display name -> extractor(env) after one finished episode.
# Time metrics are minutes among finished EVs:
#   avg wait time     = mean queue wait W_q
#   avg charge time   = mean service / plug-in time S
#   avg sys time      = mean sojourn W = W_q + S
DEFAULT_METRICS: dict[str, Callable[[ChargingStationEnv], float]] = {
    "finished EVs": lambda e: float(len(e.engine.metrics.finished_evs)),
    "dropped EVs": lambda e: float(len(e.engine.metrics.dropped_evs)),
    "total energy delivered (kWh)": lambda e: e.engine.metrics.total_energy(),
    "average L": lambda e: e.engine.metrics.average_L(),
    "average Q": lambda e: e.engine.metrics.average_Q(),
    "avg wait time": lambda e: e.engine.metrics.mean_wait(),
    "max wait time": lambda e: e.engine.metrics.max_wait(),
    "avg charge time": lambda e: e.engine.metrics.mean_service(),
    "avg sys time": lambda e: e.engine.metrics.mean_sojourn(),
    "util_rho_sim": lambda e: e.engine.metrics.queueing_summary(
        e.engine.current_time,
        n_servers=e.engine.station.n_piles * e.engine.station.n_connectors,
    )["rho_sim"],
    "util_rho_theory": lambda e: e.engine.metrics.queueing_summary(
        e.engine.current_time,
        n_servers=e.engine.station.n_piles * e.engine.station.n_connectors,
    )["rho_theory"],
}


def _resolve_metrics(metrics: list[str] | None) -> list[str]:
    names = list(metrics) if metrics is not None else list(DEFAULT_METRICS)
    unknown = [m for m in names if m not in DEFAULT_METRICS]
    if unknown:
        raise KeyError(
            f"Unknown metrics: {unknown}. Choose from {list(DEFAULT_METRICS)}"
        )
    return names


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


def scenario_run_names(
    queue_names: list[str],
    power_names: list[str],
) -> list[str]:
    """
    Labels for a queue-policy × power-policy grid.

    * Several queue names, one power name → queue names only
      (``FIFO``, ``L_SoC_D``).
    * One queue name, several power names → power names only
      (``Prop``, ``Equal``).
    * Several of both → ``{queue}_{power}``
      (``FIFO_Prop``, ``FIFO_Equal``, ``L_SoC_D_Prop``, …).
    * One of each → the queue name (the only changing / primary label).
    """
    if not queue_names:
        raise ValueError("queue_names must be non-empty")
    if not power_names:
        raise ValueError("power_names must be non-empty")

    n_q, n_p = len(queue_names), len(power_names)
    labels: list[str] = []
    for q in queue_names:
        for p in power_names:
            if n_q > 1 and n_p > 1:
                labels.append(f"{q}_{p}")
            elif n_p > 1:
                labels.append(p)
            else:
                labels.append(q)
    return labels


def labeled_policy_grid(
    queue_policies: list[QueuePolicy],
    queue_names: list[str],
    power_policies: list[PowerPolicy] | None = None,
    power_names: list[str] | None = None,
) -> list[tuple[str, QueuePolicy, PowerPolicy, str, str]]:
    """
    Cartesian product of queue and power policies with display labels.

    Returns rows ``(label, queue_policy, power_policy, queue_name, power_name)``
    in queue-major order (all powers for queue 0, then queue 1, …).

    ``power_policies`` / ``power_names`` default to one ``ProportionalPower``
    named ``Prop``. Lengths of each pair of lists must match.
    """
    if len(queue_policies) != len(queue_names):
        raise ValueError(
            "queue_policies and queue_names must have the same length "
            f"({len(queue_policies)} vs {len(queue_names)})"
        )
    if not queue_policies:
        raise ValueError("Need at least one queue policy")

    if power_policies is None:
        power_policies = [ProportionalPower()]
        power_names = ["Prop"] if power_names is None else power_names
    if power_names is None:
        raise ValueError("power_names is required when power_policies is given")
    if len(power_policies) != len(power_names):
        raise ValueError(
            "power_policies and power_names must have the same length "
            f"({len(power_policies)} vs {len(power_names)})"
        )
    if not power_policies:
        raise ValueError("Need at least one power policy")

    labels = scenario_run_names(queue_names, power_names)
    rows: list[tuple[str, QueuePolicy, PowerPolicy, str, str]] = []
    k = 0
    for q_pol, q_name in zip(queue_policies, queue_names):
        for p_pol, p_name in zip(power_policies, power_names):
            rows.append((labels[k], q_pol, p_pol, q_name, p_name))
            k += 1
    return rows


def _run_episode(env: ChargingStationEnv, queue_policy: QueuePolicy, seed: int) -> None:
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    done = False
    while not done:
        mask = env.action_masks()
        ev, pile_id = queue_policy.decide(obs, mask, rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)


def run_replications(
    scenario: dict[str, Any],
    n_reps: int = 30,
    metrics: list[str] | None = None,
    seed0: int = 0,
) -> pd.DataFrame:
    """
    Run independent episodes and return one row per replication.

    Parameters
    ----------
    scenario :
        Keyword args for ``ChargingStationEnv``, plus optional ``queue_policy``
        (a ``QueuePolicy``; default ``FIFOQueuePolicy``) and optional
        ``power_policy`` (a ``PowerPolicy`` such as ``ProportionalPower`` or
        ``StaticPower``; default proportional sharing inside the env).
        Legacy key ``policy`` is accepted as an alias for ``queue_policy``
        and raises if both are set.
    n_reps :
        Number of episodes.
    metrics :
        Keys from ``DEFAULT_METRICS``. Defaults to all of them.
    seed0 :
        Replication ``r`` uses seed ``seed0 + r`` for arrivals and queue-policy RNG.

    Returns
    -------
    DataFrame
        Index ``0 .. n_reps-1``, columns ``seed`` plus each metric.
    """
    metric_names = _resolve_metrics(metrics)
    scenario = dict(scenario)
    if "policy" in scenario and "queue_policy" in scenario:
        raise ValueError(
            "Pass only 'queue_policy' (preferred). "
            "'policy' is a deprecated alias and cannot be combined with it."
        )
    if "policy" in scenario:
        scenario["queue_policy"] = scenario.pop("policy")
    queue_policy: QueuePolicy = (
        scenario.pop("queue_policy", None) or FIFOQueuePolicy()
    )
    env = ChargingStationEnv(**scenario)

    rows: list[dict[str, float]] = []
    for r in range(n_reps):
        seed = seed0 + r
        _run_episode(env, queue_policy, seed=seed)
        row: dict[str, float] = {"seed": float(seed)}
        for name in metric_names:
            row[name] = DEFAULT_METRICS[name](env)
        rows.append(row)

    return pd.DataFrame(rows)


def summarize_ci(
    reps: pd.DataFrame,
    confidence: float = 0.95,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """
    Summarize replication samples as mean | ci_low | ci_high.

    ``reps`` is the frame from ``run_replications``. Non-metric columns such as
    ``seed`` are ignored.
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    metric_names = (
        _resolve_metrics(metrics)
        if metrics is not None
        else [c for c in reps.columns if c != "seed"]
    )

    rows = []
    for name in metric_names:
        mean, lo, hi = _mean_ci(reps[name].to_numpy(dtype=float), confidence)
        rows.append({"metric": name, "mean": mean, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows).set_index("metric")


def compare_policies(
    results: dict[str, pd.DataFrame],
    confidence: float = 0.95,
    metrics: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, dict[str, pd.DataFrame]]]:
    """
    Pairwise paired comparison of replication outputs (common seeds).

    For each pair of policies (A, B) and each metric, forms
    ``D_i = A_i - B_i``, then a CI for ``E[D]``.

    Parameters
    ----------
    results :
        Map ``policy_name -> run_replications(...)`` output. Every frame must
        include a ``seed`` column, and all seed sequences must match (same
        ``seed0`` / ``n_reps``) for CRN pairing.
    confidence :
        Two-sided level for the CI on the paired difference.
    metrics :
        Metrics to compare. Defaults to shared metric columns across frames
        (excluding ``seed``), ordered like ``DEFAULT_METRICS`` when possible.

    Returns
    -------
    sign_table : DataFrame
        Rows are metrics; columns are pair labels ``\"A|B\"``. Cell values:
        ``0`` if the mean difference is not significant, ``1`` if B has the
        lower mean, ``-1`` if A has the lower mean.
    pair_details : dict[str, dict[str, DataFrame]]
        ``pair_details[A][B]`` is a metric-indexed table with columns
        ``A`` (mean), ``B`` (mean), ``significant``, ``ci_low``, ``ci_high``
        (CI bands for mean ``A - B``).
    """
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if len(results) < 2:
        raise ValueError("compare_policies needs at least two named result frames")

    names = list(results)
    metric_sets = [{c for c in df.columns if c != "seed"} for df in results.values()]
    shared = set.intersection(*metric_sets)
    if metrics is not None:
        metric_names = _resolve_metrics(metrics)
        missing = [m for m in metric_names if m not in shared]
        if missing:
            raise KeyError(f"Metrics missing from some results: {missing}")
    else:
        metric_names = [m for m in DEFAULT_METRICS if m in shared]
        metric_names += sorted(shared - set(metric_names))

    missing_seed = [name for name, df in results.items() if "seed" not in df.columns]
    if missing_seed:
        raise ValueError(
            "compare_policies requires a 'seed' column on every result frame "
            f"(from run_replications). Missing for: {missing_seed}"
        )

    aligned = {
        name: df.sort_values("seed").reset_index(drop=True)
        for name, df in results.items()
    }

    n0 = len(aligned[names[0]])
    ref_seeds = aligned[names[0]]["seed"].to_numpy()
    for name, df in aligned.items():
        if len(df) != n0:
            raise ValueError(
                f"Replication count mismatch: {name} has {len(df)}, expected {n0}"
            )
        if not np.array_equal(df["seed"].to_numpy(), ref_seeds):
            raise ValueError(
                f"Seed sequences differ between {names[0]!r} and {name!r}; "
                "use the same seed0/n_reps for CRN pairing."
            )

    sign_cols: dict[str, list[int]] = {}
    pair_details: dict[str, dict[str, pd.DataFrame]] = {}

    for a, b in combinations(names, 2):
        da, db = aligned[a], aligned[b]
        col_label = f"{a}|{b}"
        signs: list[int] = []
        detail_rows: list[dict[str, object]] = []

        for metric in metric_names:
            diff = da[metric].to_numpy(dtype=float) - db[metric].to_numpy(dtype=float)
            mean_d, lo, hi = _mean_ci(diff, confidence)
            significant = (lo > 0.0) or (hi < 0.0)
            mean_a = float(da[metric].mean())
            mean_b = float(db[metric].mean())

            if not significant:
                sign = 0
            elif mean_b < mean_a:
                sign = 1  # second policy (B) lower
            else:
                sign = -1  # first policy (A) lower
            signs.append(sign)

            detail_rows.append(
                {
                    "metric": metric,
                    a: mean_a,
                    b: mean_b,
                    "significant": significant,
                    "ci_low": lo,
                    "ci_high": hi,
                }
            )

        sign_cols[col_label] = signs
        detail_df = pd.DataFrame(detail_rows).set_index("metric")
        pair_details.setdefault(a, {})[b] = detail_df

    sign_table = pd.DataFrame(sign_cols, index=metric_names)
    sign_table.index.name = "metric"
    return sign_table, pair_details


if __name__ == "__main__":
    from policy.queue.lowest_soc_diff import LowestSoCDiffQueuePolicy

    base = {
        "n_piles": 2,
        "n_connectors": 4,
        "n_modules": 7,
        "p_module": 25,
        "queue_capacity": 10,
        "mean_interarrival": 5.0,
    }
    fifo_reps = run_replications({**base, "queue_policy": FIFOQueuePolicy()}, n_reps=5)
    soc_reps = run_replications(
        {**base, "queue_policy": LowestSoCDiffQueuePolicy()}, n_reps=5
    )
    print(summarize_ci(fifo_reps).to_string(float_format=lambda x: f"{x:.4f}"))
    signs, details = compare_policies({"FIFO": fifo_reps, "L_SoC": soc_reps})
    print(signs)
    print(details["FIFO"]["L_SoC"].to_string(float_format=lambda x: f"{x:.4f}"))
