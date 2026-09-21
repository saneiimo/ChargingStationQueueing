"""
Run a list of ``TrialConfig``s and store the results.

Layout written under ``out_dir / run_name``::

    results.csv            one row per trial: every config knob AND every
                           metric as its own column, so a row is
                           self-describing (no header needed to know which
                           delta / mip_gap / gap_tolerance produced it)
    trials/trial_000.json  the same trial in full, nested, incl. anything
                           that does not flatten well. Also stores
                           ``arrivals``: the EV specs needed to rebuild the
                           DES episode (not the env object, and not the
                           Gurobi / DW models)
    run_meta.json          the sweep definition, stage flags and timing

``results.csv`` is rewritten after every trial, so a sweep interrupted
halfway still leaves usable output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from experiments.core.run_store import DEFAULT_OUT_DIR, RunStore, load_table

from .config import TrialConfig
from .trial import COHORT_NAMES, TrialResult, run_trial


def _flatten_by_cohort(prefix: str, by_cohort: dict | None) -> dict[str, Any]:
    """
    ``{cohort: {stat: value}}`` -> ``{"{prefix}_{cohort}_{stat}": value}``.

    ``None`` (a stage that produced no per-cohort figures, e.g. DW without
    price-and-branch) yields nothing rather than a column of blanks.
    """
    if not by_cohort:
        return {}
    out: dict[str, Any] = {}
    for cohort in COHORT_NAMES:
        stats = by_cohort.get(cohort, {})
        for stat, value in stats.items():
            out[f"{prefix}_{cohort}_{stat}"] = value
    return out


def flatten_trial(result: TrialResult, trial_id: int) -> dict[str, Any]:
    """
    One flat CSV row: config knobs, then populations, then per-stage metrics.

    Stage scalars are prefixed by source (``exact_``, ``dw_``); per-cohort
    figures become ``{source}_{group}_{cohort}_{stat}`` columns, where
    ``group`` is ``completed`` (vehicles that got their full energy) or
    ``arrived`` (all of them, unfinished ones censored at the horizon).
    Comparing a ``completed`` figure against an ``arrived`` one averages
    different populations -- that mismatch is exactly what makes an optimum
    look worse than a feasible schedule. ``grid`` also carries
    ``total_departure_slots``, which is ``sum_j D_j`` in the offline
    objective's own units and so lines up directly with ``exact_objective``
    and ``dw_LB``.
    """
    row: dict[str, Any] = {"trial_id": trial_id}
    row.update(result.config.to_row())
    row.update(result.counts)
    row.update(
        {
            f"inst_{k}": v
            for k, v in result.instance.items()
            if k not in ("cohort_sizes", "late_arrival_ids")
        }
    )
    for cohort, n in result.instance.get("cohort_sizes", {}).items():
        row[f"inst_n_{cohort}"] = n

    for prefix, stage in (
        ("sim", result.sim),
        ("grid", result.grid),
        ("exact", result.exact),
        ("dw", result.dw),
    ):
        if stage is None:
            continue
        for key, value in stage.items():
            if key.endswith("_by_cohort"):
                # "completed_by_cohort" -> columns "{src}_completed_{cohort}_{stat}"
                group = key[: -len("_by_cohort")]
                row.update(_flatten_by_cohort(f"{prefix}_{group}", value))
            else:
                row[f"{prefix}_{key}"] = value
    return row


def _trial_to_json(result: TrialResult, trial_id: int) -> dict[str, Any]:
    """Nested record for the per-trial sidecar file."""
    return {
        "trial_id": trial_id,
        "config": result.config.to_row(),
        "counts": result.counts,
        "instance": result.instance,
        "sim": result.sim,
        "grid": result.grid,
        "exact": result.exact,
        "dw": result.dw,
        # Arrival specs only. Rebuilt via replay_episode; the env and the
        # solved MILP / DW objects are not serialized.
        "arrivals": result.arrivals,
    }


def run_sweep(
    configs: Sequence[TrialConfig] | Iterable[TrialConfig],
    run_name: str | None = None,
    *,
    out_dir: Path | str = DEFAULT_OUT_DIR,
    run_sim: bool = True,
    run_exact_model: bool = True,
    run_dw_model: bool = True,
    verbose: bool = True,
    solver_progress: bool = False,
) -> pd.DataFrame:
    """
    Run every config in order and write the results.

    Parameters
    ----------
    configs :
        From ``config_grid`` or built by hand.
    run_name :
        Sub-directory name under ``out_dir``; defaults to a UTC timestamp.
    run_sim, run_exact_model, run_dw_model :
        Which of the three stages to run. The DES episode runs regardless
        (both offline models are built from it) -- see ``run_trial``.
    solver_progress :
        Print each solver's own internal log (Gurobi's native solve log;
        column generation's per-iteration ``[colgen] iter N: ...`` line) --
        see ``run_trial``'s docstring. Meant for debugging one trial at a
        time; left on for a multi-trial sweep it floods the console with
        every solver's full log for every trial.

    Returns
    -------
    DataFrame
        The same rows as ``results.csv``, indexed by position.
    """
    configs = list(configs)
    store = RunStore(run_name, out_dir, prefix="sweep")
    store.start_meta(
        n_trials=len(configs),
        stages={"sim": run_sim, "exact": run_exact_model, "dw": run_dw_model},
        configs=[c.to_row() for c in configs],
    )

    rows: list[dict[str, Any]] = []
    for trial_id, cfg in enumerate(configs):
        if verbose:
            tag = f" [{cfg.label}]" if cfg.label else ""
            print(
                f"\n[trial {trial_id + 1}/{len(configs)}]{tag} "
                f"interarrival={cfg.mean_interarrival:g} max_time={cfg.max_time:g} "
                f"delta={cfg.delta:g} mip_gap={cfg.mip_gap} "
                f"gap_tol={cfg.gap_tolerance:g}"
            )
        result = run_trial(
            cfg,
            run_sim=run_sim,
            run_exact_model=run_exact_model,
            run_dw_model=run_dw_model,
            verbose=verbose,
            solver_progress=solver_progress,
        )

        rows.append(flatten_trial(result, trial_id))
        store.write_json(
            f"trials/trial_{trial_id:03d}.json", _trial_to_json(result, trial_id)
        )
        # Rewritten each trial so an interrupted sweep still leaves results.
        store.write_table("results", rows)

    meta = store.finish_meta()
    if verbose:
        print(
            f"\nDone: {len(rows)} trial(s) in {meta['total_runtime_s']:.1f}s "
            f"-> {store.dir / 'results.csv'}"
        )
    return pd.DataFrame(rows)


def load_results(run_name: str, out_dir: Path | str = DEFAULT_OUT_DIR) -> pd.DataFrame:
    """Read back a finished (or in-progress) sweep's ``results.csv``."""
    return load_table(run_name, "results", out_dir)


def comparison_table(
    results: pd.DataFrame, cohort: str = "all", group: str = "arrived"
) -> pd.DataFrame:
    """
    The headline view: mean sojourn from all four sources, side by side.

    ``sim`` is continuous-time FIFO, ``grid`` is that same schedule on the
    slot grid (the like-for-like target), ``exact`` is the MILP incumbent and
    ``dw_LB`` the certified lower bound. Missing columns (a stage that was
    switched off) are simply absent from the result.

    ``group`` picks which population every source is averaged over, and the
    default ``"arrived"`` is the one to compare on: it covers every vehicle
    the models were given, censoring unfinished ones at the horizon, so all
    four columns describe the same set. ``"completed"`` restricts to
    vehicles that got their full energy -- readable, but it conditions on an
    outcome, dropping precisely the longest sojourns, and the simulation and
    the models drop *different* vehicles, so an apparent "optimum worse than
    FIFO" there is a population artifact rather than a real inversion.

    ``n`` columns for each source are included so a residual mismatch stays
    visible instead of silently skewing the means.
    """
    wanted = {
        "sim": f"sim_{group}_{cohort}_mean_sojourn",
        "grid": f"grid_{group}_{cohort}_mean_sojourn",
        "exact": f"exact_{group}_{cohort}_mean_sojourn",
        "dw_LB": "dw_mean_sojourn_LB",
        "sim_n": f"sim_{group}_{cohort}_n",
        "grid_n": f"grid_{group}_{cohort}_n",
        "exact_n": f"exact_{group}_{cohort}_n",
        "exact_status": "exact_status",
        "dw_status": "dw_status",
        "sim_runtime_s": "sim_runtime_s",
        "exact_runtime_s": "exact_runtime_s",
        "dw_runtime_s": "dw_runtime_s",
    }
    keys = [
        c
        for c in ("mean_interarrival", "max_time", "delta", "mip_gap", "gap_tolerance")
        if c in results.columns
    ]
    cols = {out: src for out, src in wanted.items() if src in results.columns}
    table = results[keys + list(cols.values())].copy()
    table.columns = keys + list(cols)
    return table
