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
    bound_history.csv      (only with run_bp_model) branch-and-price's
                           certified bracket on the optimum over time: one
                           row per change per trial -- see
                           ``bound_history_rows`` / ``load_bound_history``

``results.csv`` and ``bound_history.csv`` are rewritten after every trial,
so a sweep interrupted halfway still leaves usable output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from experiments.core.run_store import DEFAULT_OUT_DIR, RunStore, load_table

from .config import TrialConfig

# run_meta.json's "objective_units": every objective/bound column is total
# sojourn sum_j (delta*D_j - a_j) over objective_cohorts, in minutes.
OBJECTIVE_UNITS = "total_sojourn_min"
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

    Stage scalars are prefixed by source (``exact_``, ``bp_``, ``dw_``);
    per-cohort figures become ``{source}_{group}_{cohort}_{stat}`` columns, where
    ``group`` is ``completed`` (vehicles that got their full energy) or
    ``arrived`` (all of them, unfinished ones censored at the horizon).
    Comparing a ``completed`` figure against an ``arrived`` one averages
    different populations -- that mismatch is exactly what makes an optimum
    look worse than a feasible schedule. The offline objective is total
    sojourn in minutes, so ``exact_objective``, ``bp_objective`` /
    ``bp_best_bound`` and ``dw_LB`` line up directly with
    ``grid_arrived_{cohort}_total_sojourn`` for the cohort level matching
    ``objective_cohorts`` (``grid`` also carries ``total_departure_slots``,
    ``sum_j D_j``).
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
        ("bp", result.bp),
        ("dw", result.dw),
    ):
        if stage is None:
            continue
        for key, value in stage.items():
            if key == "bound_history":
                continue  # a time series: written to bound_history.csv instead
            if key.endswith("_by_cohort"):
                # "completed_by_cohort" -> columns "{src}_completed_{cohort}_{stat}"
                group = key[: -len("_by_cohort")]
                row.update(_flatten_by_cohort(f"{prefix}_{group}", value))
            else:
                row[f"{prefix}_{key}"] = value
    return row


def bound_history_rows(result: TrialResult, trial_id: int) -> list[dict[str, Any]]:
    """
    Long-format rows of one trial's B&P bound history (empty if B&P did not
    run or failed): ``trial_id``, ``label`` and the knobs that usually tell
    trials apart, then ``time_s``, ``lower_bound``, ``upper_bound``,
    ``gap``, ``gap_pct``, ``lower_bound_min``, ``upper_bound_min``,
    ``nodes``, ``event``. Join on ``trial_id`` with ``results.csv`` for any
    other knob.
    """
    if not result.bp or "bound_history" not in result.bp:
        return []
    cfg = result.config
    ident = {
        "trial_id": trial_id,
        "label": cfg.label,
        "mean_interarrival": cfg.mean_interarrival,
        "max_time": cfg.max_time,
        "delta": cfg.delta,
        "n_piles": cfg.n_piles,
        "bp_status": result.bp.get("status"),
    }
    return [{**ident, **point} for point in result.bp["bound_history"]]


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
        "bp": result.bp,
        "dw": result.dw,
        # Arrival specs only. Rebuilt via replay_episode; the env and the
        # solved MILP / B&P / DW objects are not serialized.
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
    run_bp_model: bool = False,
    verbose: bool = True,
    solver_progress: bool = False,
    return_history: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run every config in order and write the results.

    Parameters
    ----------
    configs :
        From ``config_grid`` or built by hand.
    run_name :
        Sub-directory name under ``out_dir``; defaults to a UTC timestamp.
    run_sim, run_exact_model, run_bp_model, run_dw_model :
        Which stages to run. ``run_exact_model`` solves the exact model as a
        compact MILP, ``run_bp_model`` solves the same model by
        branch-and-price (``bp_*`` columns; off by default); either, both or
        neither. The DES episode runs regardless (every offline model is
        built from it) -- see ``run_trial``.
    solver_progress :
        Print each solver's own internal log (Gurobi's native solve log;
        B&P's per-node log; column generation's per-iteration
        ``[colgen] iter N: ...`` line) --
        see ``run_trial``'s docstring. Meant for debugging one trial at a
        time; left on for a multi-trial sweep it floods the console with
        every solver's full log for every trial.
    return_history :
        Also return the branch-and-price bound history (see below).

    Returns
    -------
    DataFrame
        The same rows as ``results.csv``, indexed by position.
    (DataFrame, DataFrame)
        With ``return_history=True``: the above, plus the same rows as
        ``bound_history.csv`` -- B&P's certified bracket ``[lower_bound,
        upper_bound]`` on each trial's optimum against ``time_s`` (see
        ``bound_history_rows``; empty if ``run_bp_model`` is off).
    """
    configs = list(configs)
    store = RunStore(run_name, out_dir, prefix="sweep")
    store.start_meta(
        n_trials=len(configs),
        # Units of every objective/bound column (exact_objective, *_best_bound,
        # dw_LB, bound_history's lower/upper_bound, ...). Runs written before
        # the models switched objective have no such key: there those columns
        # are sum_j D_j in slots (mean-sojourn columns were minutes either way).
        objective_units=OBJECTIVE_UNITS,
        stages={"sim": run_sim, "exact": run_exact_model, "bp": run_bp_model, "dw": run_dw_model},
        configs=[c.to_row() for c in configs],
    )

    rows: list[dict[str, Any]] = []
    history: list[dict[str, Any]] = []
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
            run_bp_model=run_bp_model,
            verbose=verbose,
            solver_progress=solver_progress,
        )

        rows.append(flatten_trial(result, trial_id))
        store.write_json(
            f"trials/trial_{trial_id:03d}.json", _trial_to_json(result, trial_id)
        )
        # Rewritten each trial so an interrupted sweep still leaves results.
        store.write_table("results", rows)
        trial_history = bound_history_rows(result, trial_id)
        if trial_history:
            history.extend(trial_history)
            store.write_table("bound_history", history)

    meta = store.finish_meta()
    if verbose:
        print(
            f"\nDone: {len(rows)} trial(s) in {meta['total_runtime_s']:.1f}s "
            f"-> {store.dir / 'results.csv'}"
        )
    results = pd.DataFrame(rows)
    if return_history:
        return results, pd.DataFrame(history)
    return results


def load_results(run_name: str, out_dir: Path | str = DEFAULT_OUT_DIR) -> pd.DataFrame:
    """Read back a finished (or in-progress) sweep's ``results.csv``."""
    return load_table(run_name, "results", out_dir)


def load_bound_history(run_name: str, out_dir: Path | str = DEFAULT_OUT_DIR) -> pd.DataFrame:
    """
    Read back ``bound_history.csv``: branch-and-price's certified bracket on
    the optimum over time, one row per change per trial (see
    ``bound_history_rows``). Empty if the run did not use ``run_bp_model``.

    Per trial, ``lower_bound`` is non-decreasing and ``upper_bound``
    non-increasing in ``time_s``; between two rows both are constant, so
    plot them as steps (``where="post"``).
    """
    path = Path(out_dir) / run_name / "bound_history.csv"
    if not path.exists():
        return pd.DataFrame()
    return load_table(run_name, "bound_history", out_dir)


def comparison_table(
    results: pd.DataFrame, cohort: str = "all", group: str = "arrived"
) -> pd.DataFrame:
    """
    The headline view: mean sojourn from every source, side by side.

    ``sim`` is continuous-time FIFO, ``grid`` is that same schedule on the
    slot grid (the like-for-like target), ``exact`` is the MILP incumbent,
    ``bp`` / ``bp_LB`` are branch-and-price's best schedule and proven lower
    bound for the same model (equal when ``bp_status == "OPTIMAL"``), and
    ``dw_LB`` is DW's certified lower bound. Missing columns (a stage that
    was switched off) are simply absent from the result.

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
        "bp": f"bp_{group}_{cohort}_mean_sojourn",
        "bp_LB": "bp_mean_sojourn_LB",
        "dw_LB": "dw_mean_sojourn_LB",
        "sim_n": f"sim_{group}_{cohort}_n",
        "grid_n": f"grid_{group}_{cohort}_n",
        "exact_n": f"exact_{group}_{cohort}_n",
        "bp_n": f"bp_{group}_{cohort}_n",
        "exact_status": "exact_status",
        "bp_status": "bp_status",
        "dw_status": "dw_status",
        "sim_runtime_s": "sim_runtime_s",
        "exact_runtime_s": "exact_runtime_s",
        "bp_runtime_s": "bp_runtime_s",
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
