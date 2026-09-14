"""
Run a list of ``PolicyConfig``s and store the comparison.

Each config races its queue x power grid over ``n_reps`` replications with
common random numbers (every policy sees the same seed sequence), then
summarises each policy with a confidence interval and every policy pair
with a paired-difference interval.

Four tables are written under ``experiments/results/<run_name>/``::

    results.csv        one row per (config, policy): every metric's
                       mean / ci_low / ci_high. Long in policy, so it
                       plots directly as one line per policy.
    results_wide.csv   one row per config, with {policy}_{metric}_{stat}
                       columns. Mirrors the objective sweep's shape; handy
                       for scanning a whole sweep at a glance.
    comparisons.csv    one row per (config, policy pair, metric): both
                       means, the CI on their difference, and which policy
                       won. This is the part a per-policy table cannot
                       express -- the pairing is what makes the difference
                       statistically meaningful.
    replications.csv   the raw per-replication metric values, so any of the
                       above can be recomputed (a different confidence
                       level, a different test) without re-simulating.

Every table is rewritten after each config, so an interrupted sweep still
leaves usable output.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from experiments.core.run_store import DEFAULT_OUT_DIR, RunStore, load_table

from .config import PolicyConfig, build_policies, policy_grid
from .replications import compare_policies, run_replications, summarize_ci

# The metric whose mean feeds the derived max-wait override (see
# PolicyConfig.max_wait_from). Must be a DEFAULT_METRICS key.
MAX_WAIT_SOURCE_METRIC = "max wait time"


def metric_slug(name: str) -> str:
    """
    ``"total energy delivered (kWh)"`` -> ``"total_energy_delivered_kwh"``.

    Metric names are human-readable with spaces and parentheses; column
    names should not be. ``"(measured)"`` becomes a ``_measured`` suffix,
    which is how the post-warm-up counterpart of each metric is spelled.
    """
    return re.sub(r"_+", "_", re.sub(r"[^0-9a-z]+", "_", name.lower())).strip("_")


# --------------------------------------------------------------------------- #
# One config
# --------------------------------------------------------------------------- #


def _resolve_order(cfg: PolicyConfig) -> list[tuple[str, str, str]]:
    """
    Grid order, with the max-wait reference policy first when there is one.

    The derived override needs the reference policy's result before the
    others can be configured, so it cannot simply run in grid order.
    """
    grid = policy_grid(cfg)
    if cfg.max_wait_from is None:
        return grid
    return sorted(grid, key=lambda row: row[1] != cfg.max_wait_from)


def run_policy_trial(cfg: PolicyConfig, *, verbose: bool = True) -> dict[str, Any]:
    """
    Race one config's policy grid and summarise it.

    Returns ``{"reps", "summaries", "signs", "pair_details", "max_wait_used",
    "runtime_s"}``. ``reps`` maps policy label -> the raw per-replication
    frame; everything else is derived from it.
    """
    t0 = time.perf_counter()
    reps: dict[str, pd.DataFrame] = {}
    max_wait_used: dict[str, float | None] = {}
    derived: float | None = None

    for label, q_name, p_name in _resolve_order(cfg):
        if cfg.max_wait is not None:
            mw = cfg.max_wait
        elif cfg.max_wait_from is not None and q_name != cfg.max_wait_from:
            mw = derived  # set by the reference policy, which ran first
        else:
            mw = None

        queue_policy, power_policy = build_policies(q_name, p_name, max_wait=mw)
        frame = run_replications(
            scenario={
                **cfg.scenario_kwargs,
                "queue_policy": queue_policy,
                "power_policy": power_policy,
            },
            n_reps=cfg.n_reps,
            seed0=cfg.seed0,
        )
        reps[label] = frame
        max_wait_used[label] = mw

        if cfg.max_wait_from is not None and q_name == cfg.max_wait_from:
            observed = float(frame[MAX_WAIT_SOURCE_METRIC].mean())
            derived = observed * cfg.max_wait_factor
            if verbose:
                print(
                    f"    {label}: observed {MAX_WAIT_SOURCE_METRIC}="
                    f"{observed:.2f} min -> max_wait={derived:.2f} for the rest"
                )
        if verbose:
            mw_txt = "off" if mw is None else f"{mw:.2f}"
            print(f"    {label}: {cfg.n_reps} reps done (max_wait={mw_txt})")

    summaries = {
        label: summarize_ci(frame, confidence=cfg.confidence)
        for label, frame in reps.items()
    }
    signs, pair_details = (
        compare_policies(reps, confidence=cfg.confidence)
        if len(reps) > 1
        else (None, None)
    )
    return {
        "reps": reps,
        "summaries": summaries,
        "signs": signs,
        "pair_details": pair_details,
        "max_wait_used": max_wait_used,
        "runtime_s": time.perf_counter() - t0,
    }


# --------------------------------------------------------------------------- #
# Flattening one config's result into the four tables
# --------------------------------------------------------------------------- #


def _result_rows(cfg: PolicyConfig, config_id: int, trial: dict) -> list[dict]:
    """One row per policy: config knobs + every metric's mean and CI."""
    base = {"config_id": config_id, **cfg.to_row()}
    grid = {label: (q, p) for label, q, p in policy_grid(cfg)}
    rows = []
    for label, summary in trial["summaries"].items():
        q_name, p_name = grid[label]
        row = {
            **base,
            "policy": label,
            "queue_policy": q_name,
            "power_policy": p_name,
            "max_wait_used": trial["max_wait_used"][label],
            "runtime_s": trial["runtime_s"],
        }
        for metric, stats in summary.iterrows():
            slug = metric_slug(str(metric))
            row[f"{slug}_mean"] = stats["mean"]
            row[f"{slug}_ci_low"] = stats["ci_low"]
            row[f"{slug}_ci_high"] = stats["ci_high"]
        rows.append(row)
    return rows


def _wide_row(cfg: PolicyConfig, config_id: int, trial: dict) -> dict:
    """One row per config: ``{policy}_{metric}_{stat}`` across all policies."""
    row: dict[str, Any] = {"config_id": config_id, **cfg.to_row()}
    row["runtime_s"] = trial["runtime_s"]
    for label, summary in trial["summaries"].items():
        row[f"{label}_max_wait_used"] = trial["max_wait_used"][label]
        for metric, stats in summary.iterrows():
            slug = metric_slug(str(metric))
            row[f"{label}_{slug}_mean"] = stats["mean"]
            row[f"{label}_{slug}_ci_low"] = stats["ci_low"]
            row[f"{label}_{slug}_ci_high"] = stats["ci_high"]
    # The paired verdicts, one column per (pair, metric).
    signs = trial["signs"]
    if signs is not None:
        for pair in signs.columns:
            for metric in signs.index:
                row[f"sign_{pair.replace('|', '_vs_')}_{metric_slug(str(metric))}"] = (
                    signs.loc[metric, pair]
                )
    return row


def _comparison_rows(cfg: PolicyConfig, config_id: int, trial: dict) -> list[dict]:
    """One row per (policy pair, metric), with the paired-difference CI."""
    details = trial["pair_details"]
    if not details:
        return []
    base = {"config_id": config_id, **cfg.to_row()}
    rows = []
    for a, inner in details.items():
        for b, frame in inner.items():
            for metric, stats in frame.iterrows():
                lo, hi = float(stats["ci_low"]), float(stats["ci_high"])
                significant = bool(stats["significant"])
                mean_a, mean_b = float(stats[a]), float(stats[b])
                # Mirrors compare_policies' own sign convention.
                sign = 0 if not significant else (1 if mean_b < mean_a else -1)
                rows.append({
                    **base,
                    "pair": f"{a}|{b}",
                    "policy_a": a,
                    "policy_b": b,
                    "metric": metric,
                    "metric_slug": metric_slug(str(metric)),
                    "mean_a": mean_a,
                    "mean_b": mean_b,
                    "diff_mean": mean_a - mean_b,
                    "diff_ci_low": lo,
                    "diff_ci_high": hi,
                    "significant": significant,
                    # -1: A lower (A wins); 0: not significant; 1: B lower.
                    "sign": sign,
                    # "n.s." rather than "" so it survives the CSV round-trip
                    # as a value instead of coming back as NaN.
                    "winner": (a if sign == -1 else b if sign == 1 else "n.s."),
                })
    return rows


def _replication_rows(config_id: int, trial: dict) -> list[dict]:
    """The raw per-replication values, keyed by ``config_id`` + policy."""
    rows = []
    for label, frame in trial["reps"].items():
        for _, rep in frame.iterrows():
            rows.append({"config_id": config_id, "policy": label, **rep.to_dict()})
    return rows


# --------------------------------------------------------------------------- #
# The sweep
# --------------------------------------------------------------------------- #


def run_policy_sweep(
    configs: Sequence[PolicyConfig] | Iterable[PolicyConfig],
    run_name: str | None = None,
    *,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run every config in order and write the four tables.

    Parameters
    ----------
    configs :
        From ``experiments.core.config_grid`` or built by hand.
    run_name :
        Sub-directory name under ``out_dir``; defaults to a timestamp.
        Reusing a name overwrites -- see ``RunStore``.

    Returns
    -------
    DataFrame
        ``results.csv``'s rows (one per config x policy).
    """
    configs = list(configs)
    store = RunStore(run_name, out_dir, prefix="policy_sweep")
    store.start_meta(
        n_configs=len(configs),
        n_reps=[c.n_reps for c in configs],
        configs=[c.to_row() for c in configs],
    )

    results: list[dict] = []
    wide: list[dict] = []
    comparisons: list[dict] = []
    replications: list[dict] = []

    for config_id, cfg in enumerate(configs):
        if verbose:
            tag = f" [{cfg.label}]" if cfg.label else ""
            print(
                f"\n[config {config_id + 1}/{len(configs)}]{tag} "
                f"interarrival={cfg.mean_interarrival:g} "
                f"piles={cfg.n_piles} conn={cfg.n_connectors} "
                f"modules={cfg.n_modules} n_reps={cfg.n_reps}"
            )
        trial = run_policy_trial(cfg, verbose=verbose)

        results.extend(_result_rows(cfg, config_id, trial))
        wide.append(_wide_row(cfg, config_id, trial))
        comparisons.extend(_comparison_rows(cfg, config_id, trial))
        replications.extend(_replication_rows(config_id, trial))

        # Rewritten each config so an interrupted sweep still leaves output.
        store.write_table("results", results)
        store.write_table("results_wide", wide)
        store.write_table("comparisons", comparisons)
        store.write_table("replications", replications)

    meta = store.finish_meta()
    if verbose:
        print(
            f"\nDone: {len(configs)} config(s), {len(results)} policy run(s) "
            f"in {meta['total_runtime_s']:.1f}s -> {store.dir / 'results.csv'}"
        )
    return pd.DataFrame(results)


def load_results(
    run_name: str,
    table: str = "results",
    out_dir: Path | str = DEFAULT_OUT_DIR,
) -> pd.DataFrame:
    """
    Read one of a policy sweep's tables back.

    ``table`` is ``"results"``, ``"results_wide"``, ``"comparisons"`` or
    ``"replications"``.
    """
    return load_table(run_name, table, out_dir)


def metric_table(
    results: pd.DataFrame,
    metric: str = "avg sys time",
    *,
    measured: bool = False,
    stat: str = "mean",
    index: str | list[str] = "mean_interarrival",
) -> pd.DataFrame:
    """
    Pivot one metric into configs x policies -- the headline view.

    ``metric`` is a human-readable ``DEFAULT_METRICS`` name; ``measured=True``
    selects its post-warm-up counterpart (identical unless the scenario set
    a ``warmup_period``). ``index`` is whichever knob you swept.
    """
    name = f"{metric} (measured)" if measured else metric
    column = f"{metric_slug(name)}_{stat}"
    if column not in results.columns:
        raise KeyError(
            f"{column!r} not in results. Available metric columns: "
            f"{sorted(c for c in results.columns if c.endswith('_' + stat))}"
        )
    return results.pivot_table(
        index=index, columns="policy", values=column, sort=False
    )
