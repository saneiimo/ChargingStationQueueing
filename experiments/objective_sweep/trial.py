"""
One benchmark trial: simulate a measured window, then bound it two ways.

A trial is the ``sim_benchmark.ipynb`` walkthrough, run headless and
recorded. Four numbers for the same objective come out of it, all reported
per cohort so they are comparable (see ``metrics.MetricsTracker`` and
``offline_cl_opt.boundary`` for the cohort definitions):

* ``sim``   -- the DES's own continuous-time sojourns, truncated at the
               warm-up boundary so the boundary cohorts are clocked the way
               the offline models clock them.
* ``grid``  -- that same FIFO schedule replayed on the ``delta`` slot grid.
               The like-for-like target: the offline models can only depart
               on slot boundaries (A1), so ``sim`` is not achievable by any
               grid schedule.
* ``exact`` -- the connector-lane MILP's incumbent (an upper bound on the
               optimum) plus the solver's own best bound.
* ``dw``    -- Dantzig-Wolfe's certified lower bound on the optimum.

The DES episode always runs, whatever ``run_sim`` says: both offline models
are built from the *realized* arrival stream and the boundary snapshot it
produces, so there is no instance without it. ``run_sim=False`` only skips
the sim/grid metric reporting.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from config import C_RATE, S_THRESH
from env.charging_env import ChargingStationEnv
from models.ev import EV
from offline_cl_dw import solve_by_decomposition
from offline_cl_opt import (
    StationSpec,
    build_cl_model,
    build_measurement_instance,
    solve_cl_model,
)
from offline_cl_opt.solution import extract_solution as extract_exact
from policy.power.proportional import ProportionalPower
from policy.queue.fifo import FIFOQueuePolicy
from simulation.arrivals import generate_arrivals

from .config import HR2MIN, TrialConfig

# Cohort levels, in the nesting order both packages report them.
COHORT_NAMES = ("measurement", "measurement_queued", "all")

# Which raw cohort tags each nested level covers.
COHORT_LEVEL_TAGS: dict[str, set[str]] = {
    "measurement": {"measurement"},
    "measurement_queued": {"measurement", "queued"},
    "all": {"measurement", "queued", "boundary"},
}

# The two reporting groups every source produces. See ``_model_groups`` and
# ``MetricsTracker.arrived_sojourn_by_cohort`` for why both are needed.
GROUP_NAMES = ("completed", "arrived")

NAN = float("nan")

# Tolerance on the model-side completion test. ``energy_kwh`` is a sum of
# continuous solver variables, so exact equality against W_j never holds;
# Gurobi's own default feasibility tolerance is 1e-6.
COMPLETION_EPS_KWH = 1e-6


@dataclass
class TrialResult:
    """
    Everything one trial produced, in nested dicts ready for JSON.

    ``sim``/``grid``/``exact``/``dw`` are ``None`` for a stage that was
    switched off, and carry an ``"error"`` key for one that was attempted
    and failed (a solve can hit an infeasibility or an assertion; a failed
    stage must not take the whole sweep down with it).
    """

    config: TrialConfig
    counts: dict[str, Any] = field(default_factory=dict)
    instance: dict[str, Any] = field(default_factory=dict)
    # Arrival specs the DES was actually fed (see ``arrival_specs``). Kept
    # so a later notebook can rebuild the env without re-drawing the stream.
    # Not copied into ``results.csv`` -- one row per EV would blow the table.
    arrivals: list[dict[str, Any]] = field(default_factory=list)
    sim: dict[str, Any] | None = None
    grid: dict[str, Any] | None = None
    exact: dict[str, Any] | None = None
    dw: dict[str, Any] | None = None


# --------------------------------------------------------------------------- #
# Stage 1 -- the simulated episode
# --------------------------------------------------------------------------- #


def arrival_specs(evs: list[EV]) -> list[dict[str, Any]]:
    """
    JSON-ready copy of an arrival list, before any DES state is written.

    These six fields are everything ``EV`` needs to start an episode
    (``s_th`` / ``c_rate`` default to ``config.py`` when omitted, but they
    are stored anyway so a later change of those defaults cannot silently
    redraw the same trial). Runtime fields (SoC, pile, energy, traces) are
    intentionally absent: the point is to rebuild, not to pickle the env.
    """
    return [
        {
            "id": int(ev.id),
            "c_b": float(ev.c_b),
            "s_i": float(ev.s_i),
            "s_f": float(ev.s_f),
            "arrival_time": float(ev.arrival_time),
            "s_th": float(ev.s_th),
            "c_rate": float(ev.c_rate),
        }
        for ev in evs
    ]


def evs_from_specs(specs: list[dict[str, Any]]) -> list[EV]:
    """Fresh ``EV`` objects from an ``arrival_specs`` record."""
    return [
        EV(
            id=int(row["id"]),
            c_b=float(row["c_b"]),
            s_i=float(row["s_i"]),
            s_f=float(row["s_f"]),
            arrival_time=float(row["arrival_time"]),
            s_th=float(row.get("s_th", S_THRESH)),
            c_rate=float(row.get("c_rate", C_RATE)),
        )
        for row in specs
    ]


def _battery_caps_kwmin(config: dict[str, Any]) -> list[float]:
    """``TrialConfig.to_row()`` stores kWh as a comma string; accept either."""
    raw = config["battery_cap_kwh"]
    if isinstance(raw, str):
        kwh = [float(part) for part in raw.split(",") if part.strip()]
    else:
        kwh = [float(part) for part in raw]
    return [cap * HR2MIN for cap in kwh]


def _resolved_delta_arr(config: dict[str, Any]) -> float | None:
    """Match ``TrialConfig.resolved_delta_arr`` on a ``to_row()`` dict."""
    delta_arr = config.get("delta_arr", "delta")
    if delta_arr == "delta":
        return float(config["delta"])
    if delta_arr is None:
        return None
    return float(delta_arr)


def _drive_fifo(env: ChargingStationEnv, policy_seed: int, obs) -> float:
    """FIFO loop after ``env.reset``. Returns the loop's wall-clock seconds."""
    rng = np.random.default_rng(policy_seed)
    policy = FIFOQueuePolicy()
    t0 = time.perf_counter()
    done = bool(env.engine.terminated)
    while not done:
        mask = env.action_masks()
        ev, pile_id = policy.decide(obs, mask, rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)
    return time.perf_counter() - t0


def _make_env(config: dict[str, Any], evs: list[EV]) -> ChargingStationEnv:
    """
    The station ``run_episode`` builds: proportional power, external arrivals.

    ``config`` is a ``TrialConfig.to_row()`` dict (or anything with the same
    keys). ``mean_interarrival`` is forced to ``None`` so the saved list is
    what the engine uses, not a fresh Poisson draw.
    """
    return ChargingStationEnv(
        n_piles=int(config["n_piles"]),
        n_connectors=int(config["n_connectors"]),
        n_modules=int(config["n_modules"]),
        p_module=float(config["p_module"]),
        queue_capacity=int(config["queue_capacity"]),
        power_policy=ProportionalPower(),
        mean_interarrival=None,
        arrivals=evs,
        max_time=float(config["max_time"]),
        battery_cap_options=_battery_caps_kwmin(config),
        delta_arr=_resolved_delta_arr(config),
        warmup_period=float(config["warmup_period"]),
        flush_queue_at_warmup=bool(config.get("flush_queue_at_warmup", False)),
    )


def run_episode(cfg: TrialConfig) -> tuple[ChargingStationEnv, float, list[dict[str, Any]]]:
    """
    Run one FIFO / proportional-power episode over warm-up + measured phase.

    The arrival stream is drawn once, over ``cfg.draw_horizon`` (a one-shot
    draw, not resampled per phase), and handed to the env with
    ``mean_interarrival=None`` so the external list is what gets used.
    Returns ``(env, wall_clock_s, arrival_specs)``. The specs are taken from
    the list *before* the episode mutates any EV, so ``replay_episode`` can
    rebuild the same env from disk.

    Note that the draw horizon itself is part of the random stream -- see
    ``TrialConfig.arrival_horizon`` for why that matters when sweeping
    ``max_time``.
    """
    evs = generate_arrivals(
        mean_interarrival=cfg.mean_interarrival,
        max_time=cfg.draw_horizon,
        rng=np.random.default_rng(cfg.seed),
        battery_cap_options=cfg.battery_cap_options,
        delta_arr=cfg.resolved_delta_arr,
    )
    specs = arrival_specs(evs)
    env = _make_env(cfg.to_row(), evs)
    obs, _ = env.reset(seed=cfg.seed)
    runtime = _drive_fifo(env, cfg.policy_seed, obs)
    return env, runtime, specs


def replay_episode(
    config: dict[str, Any], arrivals: list[dict[str, Any]]
) -> ChargingStationEnv:
    """
    Rebuild and re-run the FIFO episode stored with one objective-sweep trial.

    ``config`` is the ``config`` object in ``trials/trial_NNN.json``
    (``TrialConfig.to_row()``). ``arrivals`` is the ``arrivals`` list in that
    same file. The Gurobi / DW models are not stored and are not needed:
    this only reconstructs the discrete-event episode.

    ``policy_seed`` comes from ``config`` (default 12, matching
    ``TrialConfig``). ``seed`` is passed to ``env.reset`` for the same
    reason ``run_episode`` does; with an external arrival list it does not
    redraw vehicles.
    """
    if not arrivals:
        raise ValueError(
            "This trial JSON has no 'arrivals' list. Re-run the sweep to "
            "record the arrival specs needed to rebuild the env."
        )
    env = _make_env(config, evs_from_specs(arrivals))
    obs, _ = env.reset(seed=int(config.get("seed", 0)))
    _drive_fifo(env, int(config.get("policy_seed", 12)), obs)
    return env


def episode_utilization(
    env: ChargingStationEnv, *, post_warmup: bool = False
) -> dict[str, float]:
    """
    Theoretical and simulated connector utilization for a finished episode.

    ``rho_sim`` is the time-average fraction of connectors that were
    occupied. ``rho_theory`` is ``lambda_eff / (c * mu)`` with
    ``lambda_eff = n_finished / T`` and ``mu = 1 / E[S]``, over the same
    window. ``post_warmup=True`` restricts both to ``[warmup_period, T]``,
    which is the measured phase the objective sweep reports sojourns on.
    """
    metrics = env.engine.metrics
    sim_time = float(env.engine.current_time)
    n_servers = env.engine.station.n_piles * env.engine.station.n_connectors
    since = float(metrics.warmup_period) if post_warmup else 0.0
    summary = metrics.queueing_summary(sim_time, n_servers=n_servers, since=since)
    return {
        "since": float(summary["since"]),
        "T": float(summary["T"]),
        "n_finished": float(summary["n_finished"]),
        "lambda_eff": float(summary["lambda_eff"]),
        "mean_service_min": float(summary["S"]),
        "c": float(summary["c"]),
        "rho_sim": float(summary["rho_sim"]),
        "rho_theory": float(summary["rho_theory"]),
    }


def episode_counts(env: ChargingStationEnv) -> dict[str, Any]:
    """Population sizes: the boundary snapshot plus whole-run totals."""
    m = env.engine.metrics
    return {
        # The three cohorts present in the measured window.
        "n_arrived_post_warmup": len(m.arrived_post_warmup),
        "n_queued_at_warmup_end": len(m.queued_at_warmup_end),
        "n_in_service_at_warmup_end": len(m.in_service_at_warmup_end),
        # Whole-run context for those.
        "n_arrived_total": len(m.arrived_evs),
        "n_finished_total": len(m.finished_evs),
        "n_dropped_total": len(m.dropped_evs),
        "n_flushed_at_warmup": len(m.flushed_evs),
        "sim_clock_end": float(env.engine.current_time),
    }


def _cohort_stats(sojourns) -> dict[str, float]:
    """``{n, total_sojourn, mean_sojourn}``; ``nan`` mean on an empty set."""
    n = len(sojourns)
    total = float(sum(sojourns))
    return {
        "n": float(n),
        "total_sojourn": total,
        "mean_sojourn": float(total / n) if n else NAN,
    }


def _model_groups(per_vehicle, K: int) -> tuple[dict, dict, dict]:
    """
    Split a solved model's ``per_vehicle`` table into the two report groups.

    A vehicle counts as **completed** iff it received its full energy
    requirement -- for a boundary vehicle that means the in-window
    ``energy_kwh`` *plus* the ``energy_delivered_before_kwh`` it arrived
    with, since ``energy_required_kwh`` is always the full ``W_j`` (testing
    ``energy_kwh`` alone would read every completed boundary vehicle as
    short by exactly its pre-window energy).

    Why energy and not ``departure_slot < K``: that test is sound in one
    direction only. Constraint (19) forces ``x_j >= W_j`` at any departure
    edge strictly inside the horizon, so ``D_j < K`` does imply completion.
    The converse fails -- ``D_j = S_j + sum_k u_jk`` is an *exclusive*
    boundary, so a vehicle charging through the final slot ``K-1`` also
    lands on ``D_j = K``, precisely where an unserved vehicle sits
    (``S_j = K``, no occupancy). (19) has no row at ``k = K`` (Section
    6.3's deliberate censoring), so the departure slot cannot separate a
    buzzer-beater from a censored vehicle. ``served`` is no help either: it
    is ``any(u_jk > 0.5)``, i.e. "held a connector", and a censored vehicle
    can hold one while drawing zero power.

    ``D_j >= K`` is still worth reporting, as a *horizon-binding*
    diagnostic rather than a completion test -- see ``n_at_horizon``.

    Returns ``(completed_by_cohort, arrived_by_cohort, diagnostics)``.
    """
    pv = per_vehicle
    delivered = pv["energy_kwh"].fillna(0.0) + pv["energy_delivered_before_kwh"].fillna(0.0)
    is_completed = delivered >= pv["energy_required_kwh"] - COMPLETION_EPS_KWH
    at_horizon = pv["departure_slot"] >= K - 1e-9

    completed: dict[str, dict[str, float]] = {}
    arrived: dict[str, dict[str, float]] = {}
    for level, tags in COHORT_LEVEL_TAGS.items():
        in_level = pv["cohort"].isin(tags)
        sub_all = pv[in_level]
        sub_done = pv[in_level & is_completed]
        completed[level] = _cohort_stats(sub_done["sojourn_min"].tolist())
        arrived[level] = _cohort_stats(sub_all["sojourn_min"].tolist())
        arrived[level]["n_censored"] = float(len(sub_all) - len(sub_done))

    diagnostics = {
        "n_completed": int(is_completed.sum()),
        "n_censored": int((~is_completed).sum()),
        # Completed, but only just: finished exactly at the horizon edge.
        # A nonzero count means the horizon is binding on the optimum.
        "n_completed_at_horizon": int((is_completed & at_horizon).sum()),
        "n_at_horizon": int(at_horizon.sum()),
    }
    return completed, arrived, diagnostics


def simulation_metrics(
    env: ChargingStationEnv, delta: float, *, exclude_ids: set[int] | None = None
) -> tuple[dict, dict]:
    """
    Measured-window sojourns per cohort, continuous-time and grid-replayed,
    each in both reporting groups.

    All four use ``truncate_included=True`` semantics (the default): a
    vehicle already present at the boundary is clocked from the boundary,
    not from its real arrival, which is what makes these comparable with an
    optimizer that places the same vehicle at ``a=0``.

    ``exclude_ids`` drops vehicles absent from the offline instance (the
    closing-edge arrivals ``_arrivals_inside_horizon`` filters out) so the
    ``arrived`` group covers exactly the population the models were given.

    Returns ``(sim, grid)``, each ``{"completed_by_cohort": ...,
    "arrived_by_cohort": ...}``.
    """
    m = env.engine.metrics
    since = m.warmup_period
    censor_at = float(env.engine.max_time)  # warm-up + measured, the run's end
    sim = {
        "completed_by_cohort": m.sojourn_by_cohort(since=since),
        "arrived_by_cohort": m.arrived_sojourn_by_cohort(
            since=since, censor_at=censor_at, exclude_ids=exclude_ids
        ),
    }
    grid = {
        "completed_by_cohort": m.grid_sojourn_by_cohort(delta, since=since),
        "arrived_by_cohort": m.arrived_grid_sojourn_by_cohort(
            delta, since=since, censor_at=censor_at, exclude_ids=exclude_ids
        ),
    }
    return sim, grid


# --------------------------------------------------------------------------- #
# Stage 2 -- the offline instance shared by both models
# --------------------------------------------------------------------------- #


def _arrivals_inside_horizon(
    cfg: TrialConfig, env: ChargingStationEnv
) -> tuple[list, set[int]]:
    """
    Drop measured-window arrivals that have no slot to be released in.

    The offline model releases vehicle j at ``k_j = ceil(a_j / delta)`` and
    the horizon holds slots ``0..K-1``, so an arrival with ``k_j >= K`` has
    no slot at all and the model rejects the whole instance ("release slot
    falls outside the horizon"). That is not a misconfigured horizon here,
    it is the window's own closing edge: the DES runs to exactly
    ``warmup_period + max_time`` and records an EV arriving at that instant,
    which spends zero time inside the window. A continuous arrival stream
    (``delta_arr=None``) widens the same edge to anything after
    ``(K-1)*delta``.

    Returns the surviving EVs and the ids of those dropped (the sim-side
    report must exclude the same ones, or its ``arrived`` group would cover
    a population the models never saw).
    """
    m = env.engine.metrics
    K = math.ceil(round(cfg.offline_horizon / cfg.delta, 9))
    kept, dropped = [], set()
    for ev in m.arrived_post_warmup:
        k_j = math.ceil(
            round((float(ev.arrival_time) - env.engine.warmup_period) / cfg.delta, 9)
        )
        (kept.append(ev) if k_j < K else dropped.add(ev.id))
    return kept, dropped


def _n_optimized(cfg: TrialConfig, inst) -> int:
    """Vehicles inside ``cfg.objective_cohorts`` -- the J in delta*g/J."""
    return sum(1 for c in inst.cohorts.values() if c in cfg.objective_cohorts)


def build_instance(cfg: TrialConfig, env: ChargingStationEnv):
    """
    Turn the finished episode into the measured-window offline instance.

    Arrivals are shifted back to ``t=0``; the boundary queue and the
    already-charging vehicles are folded in per ``cfg.include_queued`` /
    ``cfg.boundary_mode``. Passing the station runs the whole-module check on
    FIXED vehicles here, where it can name the offending pile-slot.

    Arrivals landing at the window's closing edge are dropped first (see
    ``_arrivals_inside_horizon``) and counted as ``n_dropped_late_arrivals``.
    """
    m = env.engine.metrics
    station = StationSpec(
        n_piles=cfg.n_piles,
        n_connectors=cfg.n_connectors,
        n_modules=cfg.n_modules,
        p_module=cfg.p_module,
    )
    arrivals, late_ids = _arrivals_inside_horizon(cfg, env)
    inst = build_measurement_instance(
        arrived_post_warmup=arrivals,
        queued_at_warmup_end=m.queued_at_warmup_end,
        in_service_at_warmup_end=m.in_service_at_warmup_end,
        warmup_period=env.engine.warmup_period,
        delta=cfg.delta,
        horizon_minutes=cfg.offline_horizon,
        include_queued=cfg.include_queued,
        boundary_mode=cfg.boundary_mode,
        station=station,
    )
    summary = {
        "n_vehicles": len(inst.vehicles),
        "n_dropped_late_arrivals": len(late_ids),
        "late_arrival_ids": sorted(late_ids),
        "n_optimized": _n_optimized(cfg, inst),
        "n_boundary_vehicles": len(inst.boundary_vehicles),
        "n_discarded": len(inst.discarded),
        "cohort_sizes": {
            name: sum(1 for c in inst.cohorts.values() if c.value == name)
            for name in ("measurement", "queued", "boundary")
        },
    }
    return inst, station, summary


# --------------------------------------------------------------------------- #
# Stage 3 -- the exact MILP
# --------------------------------------------------------------------------- #


def run_exact(
    cfg: TrialConfig, inst, station: StationSpec, *, solver_progress: bool = False
) -> dict[str, Any]:
    """
    Build and solve the connector-lane MILP; report incumbent *and* bound.

    ``objective`` is the solver's incumbent (``sum_j D_j``, an upper bound on
    the optimum); ``best_bound`` is Gurobi's ``ObjBound`` (a lower bound), so
    a time-limited run still brackets the optimum. ``status``/``mip_gap``
    say whether the limit was what stopped it.

    ``solver_progress=True`` turns on Gurobi's own solve log (branch-and-bound
    node counts, incumbent/bound as they improve) -- see ``run_trial``'s
    docstring for when that is (and is not) a good idea.
    """
    t0 = time.perf_counter()
    cl = build_cl_model(
        inst.vehicles,
        station,
        delta=cfg.delta,
        horizon_minutes=cfg.offline_horizon,
        boundary_vehicles=inst.boundary_vehicles,
        cohorts=inst.cohorts,
        objective_cohorts=cfg.objective_cohorts,
        break_symmetry=cfg.break_symmetry,
        bound_departures=cfg.bound_departures,
        tie_break=cfg.tie_break,
    )
    build_s = time.perf_counter() - t0

    t1 = time.perf_counter()
    solve_cl_model(
        cl, mip_gap=cfg.mip_gap, time_limit=cfg.time_limit, verbose=solver_progress
    )
    solve_s = time.perf_counter() - t1

    sol = extract_exact(cl)
    # ObjBound is unavailable once a tie-break objective is set (Gurobi
    # raises), the same reason extract_solution reports mip_gap as nan there.
    try:
        best_bound = float(cl.model.ObjBound)
    except Exception:
        best_bound = NAN

    completed, arrived, diag = _model_groups(sol.per_vehicle, sol.K)
    return {
        "status": sol.status,
        "objective": sol.objective,  # incumbent sum_j D_j, slot units
        "best_bound": best_bound,  # solver's own lower bound, slot units
        "mip_gap_achieved": sol.mip_gap,
        "total_sojourn": sol.total_sojourn,
        "mean_sojourn": sol.mean_sojourn,
        "n_vehicles": sol.n_vehicles,
        "n_optimized": sol.n_optimized,
        "K": sol.K,
        "gurobi_runtime_s": sol.runtime,
        "build_runtime_s": build_s,
        "solve_runtime_s": solve_s,
        "runtime_s": build_s + solve_s,
        **diag,
        "completed_by_cohort": completed,
        "arrived_by_cohort": arrived,
    }


# --------------------------------------------------------------------------- #
# Stage 4 -- Dantzig-Wolfe
# --------------------------------------------------------------------------- #


def _dw_status(cfg: TrialConfig, cg, elapsed: float) -> str:
    """
    Why column generation stopped, as a single label.

    ``converged`` covers both exact-pricing stops (no improving column, and
    the gap-tolerance stop); otherwise the run was cut short, and the two
    ways that happens are distinguishable from the iteration count and the
    wall clock.
    """
    if cg.converged:
        return "CONVERGED"
    if cg.iterations >= cfg.max_iterations:
        return "ITERATION_LIMIT"
    if cfg.dw_time_limit is not None and elapsed >= cfg.dw_time_limit:
        return "TIME_LIMIT"
    return "NOT_CONVERGED"


def run_dw(
    cfg: TrialConfig, inst, station: StationSpec, *, solver_progress: bool = False
) -> dict[str, Any]:
    """
    Run the decomposition and report the whole bound family.

    ``LB`` is the certified anytime Lagrangian bound (valid even when column
    generation did not converge); ``z_rmp`` is the final restricted-master LP
    value and ``rmp_gap = z_rmp - LB`` its stopping-criterion gap -- a
    diagnostic of how close pricing got to proving ``z_RMP == z_MP``, *not* a
    certified bracket. ``UB``/``gap`` and the ``*_UB`` sojourns are ``nan``
    unless ``cfg.solve_integer_ub`` asked for price-and-branch.

    ``solver_progress=True`` turns on column generation's own per-iteration
    log (``[colgen] iter N: z_RMP=..., best_LB=..., gap=..., ...``) -- see
    ``run_trial``'s docstring for when that is (and is not) a good idea.

    ``gap_tolerance`` (raw ``sum_j D_j`` units, passed to column generation)
    comes from ``cfg.gap_tolerance_target_min`` when set: a target of ``m``
    mean-sojourn minutes becomes ``m * n_optimized / cfg.delta``, using this
    instance's own ``n_optimized`` (mirrors ``sim_benchmark.ipynb``'s
    ``gap_tolerance = m * len(vehicles) / delta``, but with ``n_optimized``
    -- the vehicles actually in ``objective_cohorts`` -- rather than every
    vehicle in the instance). Falls back to ``cfg.gap_tolerance`` otherwise.
    The value actually used is reported back as ``gap_tolerance_used``.
    """
    n_optimized = _n_optimized(cfg, inst)
    if cfg.gap_tolerance_target_min is not None:
        gap_tolerance = cfg.gap_tolerance_target_min * n_optimized / cfg.delta
    else:
        gap_tolerance = cfg.gap_tolerance

    t0 = time.perf_counter()
    sol, cg = solve_by_decomposition(
        inst.vehicles,
        station,
        delta=cfg.delta,
        horizon_minutes=cfg.offline_horizon,
        boundary_vehicles=inst.boundary_vehicles,
        cohorts=inst.cohorts,
        objective_cohorts=cfg.objective_cohorts,
        gamma=cfg.gamma,
        gap_tolerance=gap_tolerance,
        max_iterations=cfg.max_iterations,
        time_limit=cfg.dw_time_limit,
        purge_every=cfg.purge_every,
        solve_integer_ub=cfg.solve_integer_ub,
        progress=solver_progress,
    )
    elapsed = time.perf_counter() - t0

    # Only price-and-branch produces a schedule to split into groups.
    dw_completed = dw_arrived = dw_diag = None
    if cfg.solve_integer_ub:
        dw_completed, dw_arrived, dw_diag = _model_groups(sol.per_vehicle, sol.K)

    return {
        "status": _dw_status(cfg, cg, elapsed),
        "converged": sol.converged,
        "iterations": sol.iterations,
        "columns_purged": sol.columns_purged,
        "gap_tolerance_used": gap_tolerance,  # raw units, whichever mode set it
        # Lower bounds, raw objective units (sum_j D_j).
        "LB": sol.LB,
        "z_rmp": sol.LB + sol.rmp_gap,  # final restricted-master LP value
        "rmp_gap": sol.rmp_gap,
        "earliest_departure_sum": cg.earliest_departure_sum,  # cheap sanity floor
        # Lower bounds converted to sojourn minutes (the comparable numbers).
        "total_sojourn_LB": sol.total_sojourn_LB,
        "mean_sojourn_LB": sol.mean_sojourn_LB,
        # Upper-bound side: nan unless price-and-branch ran.
        "UB": sol.UB,
        "gap": sol.gap,
        "total_sojourn_UB": sol.total_sojourn_UB,
        "mean_sojourn_UB": sol.mean_sojourn_UB,
        "whole_module_feasible": (
            sol.whole_module_feasible if cfg.solve_integer_ub else None
        ),
        "n_vehicles": sol.n_vehicles,
        "n_optimized": sol.n_optimized,
        "K": sol.K,
        "runtime_s": elapsed,
        # Per-cohort figures describe the *schedule*, so they exist only when
        # price-and-branch built one. There is no per-cohort LB: the bound is
        # one scalar certifying the objective as a whole, which is why the
        # LB columns above carry no cohort/group split of their own.
        **(dw_diag or {}),
        "completed_by_cohort": dw_completed,
        "arrived_by_cohort": dw_arrived,
    }


# --------------------------------------------------------------------------- #
# The whole trial
# --------------------------------------------------------------------------- #


def run_trial(
    cfg: TrialConfig,
    *,
    run_sim: bool = True,
    run_exact_model: bool = True,
    run_dw_model: bool = True,
    verbose: bool = True,
    solver_progress: bool = False,
) -> TrialResult:
    """
    Simulate, then bound, one configuration.

    The three ``run_*`` flags switch off *reporting/solving* stages
    independently. The DES episode itself always runs -- both offline models
    are built from its realized arrivals and boundary snapshot -- so
    ``run_sim=False`` skips only the sim/grid sojourn tables, not the
    episode.

    ``verbose`` prints one line per trial/stage (this module's own summary --
    "instance: J=...", "exact: OPTIMAL obj=..."). ``solver_progress`` is a
    separate, much noisier switch: it turns on each solver's OWN internal
    log -- Gurobi's native branch-and-bound log for the exact model, column
    generation's own ``[colgen] iter N: z_RMP=..., best_LB=..., gap=...``
    line per iteration for DW. Meant for debugging a single slow/stuck
    trial, not for a sweep of many -- left on across a whole sweep it floods
    the console with every solver's full internal log for every trial.

    A solve that raises is recorded as ``{"error": ...}`` on its stage and
    the trial still returns, so one infeasible point cannot abort a sweep.
    """
    env, sim_runtime, arrivals = run_episode(cfg)
    result = TrialResult(config=cfg, counts=episode_counts(env), arrivals=arrivals)
    # On counts (not just on the sim stage) so the episode's cost is recorded
    # even for a run_sim=False sweep, where the episode still had to happen.
    result.counts["sim_runtime_s"] = sim_runtime

    # Built up front (cheap, no solver) even when only the sim is reported:
    # its late-arrival exclusion set is what keeps the sim's `arrived` group
    # over the same population the models would be given.
    inst, station, result.instance = build_instance(cfg, env)
    late_ids = set(result.instance.get("late_arrival_ids", ()))

    if run_sim:
        sim, grid = simulation_metrics(env, cfg.delta, exclude_ids=late_ids)
        result.sim = {"runtime_s": sim_runtime, **sim}
        result.grid = dict(grid)

    if not (run_exact_model or run_dw_model):
        return result

    if verbose:
        print(
            f"  instance: J={result.instance['n_vehicles']} "
            f"({result.instance['cohort_sizes']}), "
            f"optimized={result.instance['n_optimized']}"
        )

    for enabled, name, fn in (
        (run_exact_model, "exact", run_exact),
        (run_dw_model, "dw", run_dw),
    ):
        if not enabled:
            continue
        t0 = time.perf_counter()
        try:
            stage = fn(cfg, inst, station, solver_progress=solver_progress)
        except Exception as exc:  # a failed solve must not kill the sweep
            stage = {
                "error": f"{type(exc).__name__}: {exc}",
                "runtime_s": time.perf_counter() - t0,
            }
        setattr(result, name, stage)
        if verbose:
            print(f"  {name}: {_stage_line(stage)}")

    return result


def _stage_line(stage: dict[str, Any]) -> str:
    """One-line progress summary for a finished stage."""
    if "error" in stage:
        return f"FAILED {stage['error']}"
    if "LB" in stage:
        return (
            f"{stage['status']} LB={stage['LB']:.2f} "
            f"mean_sojourn_LB={stage['mean_sojourn_LB']:.2f} min "
            f"({stage['runtime_s']:.1f}s, {stage['iterations']} iters)"
        )
    return (
        f"{stage['status']} obj={stage['objective']:.2f} "
        f"bound={stage['best_bound']:.2f} gap={stage['mip_gap_achieved']:.2%} "
        f"mean_sojourn={stage['mean_sojourn']:.2f} min "
        f"({stage['runtime_s']:.1f}s)"
    )
