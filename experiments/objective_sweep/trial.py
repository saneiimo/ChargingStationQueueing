"""
One benchmark trial: simulate, then solve the same instance offline.

A trial is the ``sim_benchmark.ipynb`` walkthrough, run headless and
recorded, as a **closed-system** trial (see ``config.py``'s module
docstring): FIFO warms the station up to ``t0``, the state there is
snapshotted, vehicles arrive over ``[t0, T]`` and none after, and the
simulation drains the station instead of stopping at ``T``. Every source below is then run on exactly that closed
set of vehicles -- the queue and the plugged-in vehicles at ``t0`` plus
every arrival in ``[t0, T]`` -- and the offline models' horizon ``K`` is read
off the discretized FIFO run (``closed_system_horizon``). Nothing is
censored on the simulation side; a model is told nothing about FIFO beyond
``K`` (and, for B&P, an optional seed schedule).

Up to five numbers for the same objective come out of a trial, all
reported per cohort so they are comparable (see ``metrics.MetricsTracker``
and ``offline_cl_opt.boundary`` for the cohort definitions):

* ``sim``   -- the DES's own continuous-time sojourns, truncated at the
               warm-up boundary so the boundary cohorts are clocked the way
               the offline models clock them.
* ``grid``  -- that same FIFO schedule replayed on the ``delta`` slot grid.
               The like-for-like target: the offline models can only depart
               on slot boundaries (A1), so ``sim`` is not achievable by any
               grid schedule.
* ``exact`` -- the connector-lane MILP's incumbent (an upper bound on the
               optimum) plus the solver's own best bound.
* ``bp``    -- the SAME whole-module model solved by branch-and-price
               (``offline_cl_PB``): its best schedule plus a proven lower
               bound; the two coincide when ``bp_status == "OPTIMAL"``.
* ``dw``    -- Dantzig-Wolfe's certified lower bound on the optimum
               (continuous-module relaxation).

All three offline models minimize the same objective, total sojourn
``sum_j delta*(D_j - k_j)`` in minutes (``k_j = ceil(a_j/delta)``, the
release slot) over ``objective_cohorts``, so every
``objective`` / ``best_bound`` / ``LB`` / ``UB`` field is total sojourn in
minutes; ``*_mean_sojourn*`` divides it by ``n_optimized``.

What the model numbers mean. Every offline model is the closed-system
model: each vehicle is served and receives its full energy by the horizon
``K``, so every model schedule (``objective``, ``UB``) completes every
vehicle and every model lower bound (``best_bound``, ``LB``) bounds the
optimum over such schedules. ``K`` comes from the discretized FIFO run,
whose whole-module schedule fits inside it, so the models are always
feasible. Lengthening ``K`` (``horizon_slack``) can only lower the optimum,
and stops changing it once ``K`` reaches the makespan of some optimal
schedule. The optimum here is that of the discrete model -- slot grid,
whole modules -- over the closed set, the like-for-like counterpart of
``grid``, not of the continuous-time ``sim``.

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
from metrics.load import ArrivalModel, station_load_summary
from models.ev import EV
from offline_cl_dw import solve_by_decomposition
from offline_cl_opt import (
    StationSpec,
    build_cl_model,
    build_measurement_instance,
    solve_cl_model,
)
from offline_cl_opt.boundary import BoundaryMode, Cohort
from offline_cl_opt.model import lattice_round_up
from offline_cl_opt.solution import extract_solution as extract_exact
from offline_cl_PB import (
    BranchAndPrice,
    ScheduleValidationError,
    schedule_from_compact,
    schedule_from_simulation,
    validate_schedule,
)
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

# Tolerance on the model-side completion check. ``energy_kwh`` is a sum of
# continuous solver variables, so exact equality against W_j never holds;
# the models' own validation tolerances are 1e-6.
COMPLETION_EPS_KWH = 1e-6


@dataclass
class TrialResult:
    """
    Everything one trial produced, in nested dicts ready for JSON.

    ``sim``/``grid``/``exact``/``bp``/``dw`` are ``None`` for a stage that
    was switched off, and carry an ``"error"`` key for one that was
    attempted and failed (a solve can hit an infeasibility or an assertion;
    a failed stage must not take the whole sweep down with it).
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
    bp: dict[str, Any] | None = None
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
    what the engine uses, not a fresh Poisson draw. The engine runs in
    ``drain`` mode (arrivals close at ``T``, then the station is emptied).
    A trial JSON from the older measured-window sweep carries
    ``closed_system: false`` and replays as the window it was.
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
        drain=bool(config.get("closed_system", True)),
    )


def run_episode(cfg: TrialConfig) -> tuple[ChargingStationEnv, float, list[dict[str, Any]]]:
    """
    Run one FIFO / proportional-power episode over warm-up + measured phase.

    Arrivals close at ``T = warmup_period + max_time`` and the episode
    continues until the station is empty (``env.engine.end_time``).

    The arrival stream is drawn once, over ``cfg.draw_horizon`` (a one-shot
    draw, not resampled per phase), and handed to the env with
    ``mean_interarrival=None`` so the external list is what gets used.
    Returns ``(env, wall_clock_s, arrival_specs)``. The specs are taken from
    the list *before* the episode mutates any EV, so ``replay_episode`` can
    rebuild the same env from disk.

    The draw horizon is free to vary across a sweep: ``generate_arrivals``
    draws gaps and vehicle characteristics from separate streams, so a wider
    horizon only appends later arrivals and every trial sharing a ``seed``
    sees the same fleet regardless of ``max_time`` / ``warmup_period`` /
    ``mean_interarrival``. See ``simulation.arrivals``'s module docstring.
    """
    evs = generate_arrivals(
        mean_interarrival=cfg.effective_mean_interarrival,
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


def station_load(
    env: ChargingStationEnv, *, since: float, mean_interarrival: float | None = None
) -> dict[str, float]:
    """
    ``metrics.load.station_load_summary`` for a finished episode: connector
    occupancy (per connector and overall) and power utilization (per pile
    and overall) over ``[since, end of run]``; offered power / connector
    loads from the arrivals in ``[since, engine.max_time]`` (``*_realized``)
    and, given ``mean_interarrival`` (the env itself has none -- it is fed
    an arrival list), from the arrival generator (``*_theory``). Every key
    is explained in ``metrics/README.md``.
    """
    eng = env.engine
    now = float(eng.current_time)
    return station_load_summary(
        eng.metrics, eng.station, now, since=since,
        arrivals_until=min(float(eng.max_time), now),
        arrival_model=ArrivalModel.from_env(env, mean_interarrival=mean_interarrival),
    )


def episode_utilization(
    env: ChargingStationEnv, *, post_warmup: bool = False, mean_interarrival: float | None = None
) -> dict[str, float]:
    """
    Theoretical and simulated connector utilization for a finished episode,
    plus every ``station_load`` metric over the same window.

    ``rho_sim`` is the time-average fraction of connectors that were
    occupied. ``rho_theory`` is ``lambda_eff / (c * mu)`` with
    ``lambda_eff = n_finished / T`` and ``mu = 1 / E[S]``, over the same
    window. ``post_warmup=True`` restricts both to ``[warmup_period, end]``,
    the measured phase the objective sweep reports sojourns on (``end`` is
    when the station emptied, so the drain is included). See
    ``metrics/README.md`` for every key.
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
        **station_load(env, since=since, mean_interarrival=mean_interarrival),
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
        # When arrivals closed (T) and when the run ended; they differ by how
        # long draining the station took.
        "arrivals_close_at": float(env.engine.max_time),
        "sim_clock_end": float(env.engine.current_time),
        "drain_minutes": float(env.engine.current_time - env.engine.max_time),
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

    Every model is the closed-system model, so every vehicle is served and
    complete and the ``completed`` and ``arrived`` groups coincide; both are
    kept so a model's figures line up with the simulation's
    (``simulation_metrics``) column for column. Completion is checked here
    anyway -- a schedule short of some vehicle's energy is a solver bug, not
    a result -- for every vehicle except FIXED boundary ones, whose pinned
    power stops at the last whole slot before their real departure (the
    model exempts them from completion for that reason). A boundary
    vehicle's energy counts what it had at ``t0``
    (``energy_delivered_before_kwh``).

    ``n_at_horizon`` counts vehicles departing exactly at ``K``: nonzero
    means the horizon may bind the optimum (see ``horizon_slack``).

    Returns ``(completed_by_cohort, arrived_by_cohort, diagnostics)``.
    """
    pv = per_vehicle
    delivered = pv["energy_kwh"].fillna(0.0) + pv["energy_delivered_before_kwh"].fillna(0.0)
    # Every solution table carries boundary_mode; a bare table without it has
    # no FIXED vehicle.
    is_fixed = pv["boundary_mode"] == BoundaryMode.FIXED.value if "boundary_mode" in pv else False
    short = ~is_fixed & (delivered < pv["energy_required_kwh"] - COMPLETION_EPS_KWH)
    if short.any():
        raise RuntimeError(
            f"model schedule leaves vehicles {pv.loc[short, 'vehicle_id'].tolist()[:10]} short of "
            "their energy; the closed-system model forbids that."
        )

    completed: dict[str, dict[str, float]] = {}
    for level, tags in COHORT_LEVEL_TAGS.items():
        completed[level] = _cohort_stats(pv.loc[pv["cohort"].isin(tags), "sojourn_min"].tolist())
    arrived = {level: dict(stats) for level, stats in completed.items()}
    diagnostics = {"n_at_horizon": int((pv["departure_slot"] >= K - 1e-9).sum())}
    return completed, arrived, diagnostics


def simulation_metrics(env: ChargingStationEnv, delta: float) -> tuple[dict, dict]:
    """
    Measured-window sojourns per cohort, continuous-time and grid-replayed,
    each in both reporting groups.

    All four use ``truncate_included=True`` semantics (the default): a
    vehicle already present at the boundary is clocked from the boundary,
    not from its real arrival, which is what makes these comparable with an
    optimizer that places the same vehicle at ``a=0``.

    The station was drained, so every vehicle finished and the
    ``completed`` and ``arrived`` groups coincide (``arrived`` would censor
    an unfinished vehicle at the end of the run; there is none).

    Returns ``(sim, grid)``, each ``{"completed_by_cohort": ...,
    "arrived_by_cohort": ...}``.
    """
    m = env.engine.metrics
    since = m.warmup_period
    censor_at = float(env.engine.current_time)  # the run's end, after the drain
    sim = {
        "completed_by_cohort": m.sojourn_by_cohort(since=since),
        "arrived_by_cohort": m.arrived_sojourn_by_cohort(since=since, censor_at=censor_at),
    }
    grid = {
        "completed_by_cohort": m.grid_sojourn_by_cohort(delta, since=since),
        "arrived_by_cohort": m.arrived_grid_sojourn_by_cohort(delta, since=since, censor_at=censor_at),
    }
    return sim, grid


# --------------------------------------------------------------------------- #
# Stage 2 -- the offline instance shared by both models
# --------------------------------------------------------------------------- #

# How often closed_system_horizon doubles its provisional horizon looking for
# a slot grid long enough to fit the discretized FIFO schedule.
_HORIZON_DOUBLINGS = 8


def _closed_set_ids(env: ChargingStationEnv) -> set[int]:
    """
    The closed vehicle set of a drained episode: the queue and the plugged-in
    vehicles at ``t0`` plus every arrival in ``[t0, T]``.

    Raises unless every one of them finished. A vehicle turned away by a full
    queue never enters the station, so it would have no sojourn in the
    simulation while the models (which have no drop decision) would serve it
    -- the two would no longer share a vehicle set. A vehicle still present
    after draining would be an engine bug.
    """
    m = env.engine.metrics
    ids = {ev.id for ev in m.arrived_post_warmup} | m.cohort_ev_ids(queued=True, boundary=True)
    dropped = sorted(ev.id for ev in m.dropped_evs if ev.id in ids)
    if dropped:
        raise ValueError(
            f"{len(dropped)} vehicle(s) arriving in [t0, T] were turned away by a full queue "
            f"(ids {dropped[:10]}). A closed-system trial needs every arrival admitted: raise "
            "TrialConfig.queue_capacity."
        )
    unfinished = sorted(ids - {ev.id for ev in m.finished_evs})
    if unfinished:
        raise RuntimeError(
            f"vehicles {unfinished[:10]} are still in the station after draining "
            f"(clock {env.engine.current_time:g}); the drained episode is inconsistent."
        )
    return ids


def _measurement_instance(cfg: TrialConfig, env, station: StationSpec, arrivals, horizon_minutes: float):
    """``build_measurement_instance`` for this trial's snapshot and horizon."""
    m = env.engine.metrics
    return build_measurement_instance(
        arrived_post_warmup=arrivals,
        queued_at_warmup_end=m.queued_at_warmup_end,
        in_service_at_warmup_end=m.in_service_at_warmup_end,
        warmup_period=env.engine.warmup_period,
        delta=cfg.delta,
        horizon_minutes=horizon_minutes,
        include_queued=cfg.include_queued,
        boundary_mode=cfg.boundary_mode,
        station=station,
    )


def _objective_weights(cfg: TrialConfig, inst) -> dict[int, float]:
    """1.0 for vehicles in ``objective_cohorts``, else 0.0 (as the models use)."""
    return {
        v.id: 1.0 if inst.cohorts.get(v.id, Cohort.MEASUREMENT) in cfg.objective_cohorts else 0.0
        for v in inst.vehicles
    }


def closed_system_horizon(
    cfg: TrialConfig, env: ChargingStationEnv, station: StationSpec
) -> dict[str, Any]:
    """
    The offline models' horizon, read off the drained FIFO run.

    Two discretizations of the drained FIFO run give the slot by which it
    empties the station:

    * ``K_grid_fifo`` -- the grid replay reported as ``grid`` (each stay
      rounded up to whole slots along its lane's handover chain;
      ``MetricsTracker.grid_departure_slots``). It replays occupancy, not
      power, so it need not be a feasible schedule for the models.
    * ``K_feasible_fifo`` -- the whole-module schedule
      ``offline_cl_PB.schedule_from_simulation`` builds from the same run
      (simulated plug-in order and piles, rebuilt on the grid), validated
      here against the exact model (every vehicle served and completed). It
      is built on a provisional horizon, doubled until every vehicle fits.

    ``K = ceil(max(K_grid_fifo, K_feasible_fifo) * (1 + horizon_slack))``.
    Taking the larger guarantees the closed-system models are feasible --
    the feasible FIFO schedule is one of their schedules -- and it is the
    starting schedule of B&P (``"simulation"`` seed) and of DW.

    Returns the horizon (``K``, ``horizon_minutes``), both makespans and
    the feasible FIFO schedule's total sojourn.
    """
    m = env.engine.metrics
    t0 = env.engine.warmup_period
    delta = cfg.delta
    ids = _closed_set_ids(env)

    slots = m.grid_departure_slots(delta, since=t0)
    K_grid = max((slots[j] for j in ids if j in slots), default=1)

    # Provisional horizon for the feasible FIFO schedule: comfortably past
    # both the grid makespan and the continuous drain, doubled if needed.
    K_try = max(K_grid, math.ceil(round((env.engine.current_time - t0) / delta, 9)), 1)
    for _ in range(_HORIZON_DOUBLINGS):
        K_try *= 2
        inst = _measurement_instance(cfg, env, station, m.arrived_post_warmup, K_try * delta)
        sched, counts = schedule_from_simulation(
            m.arrived_evs, inst.vehicles, station, delta, K_try, inst.boundary_vehicles,
            warmup_period=t0,
        )
        if sched is not None:  # every vehicle placed and complete
            break
    else:
        raise RuntimeError(
            f"the discretized FIFO schedule still cannot finish every vehicle by K={K_try} "
            f"slots (delta={delta:g}); cannot derive a closed-system horizon."
        )
    check = validate_schedule(
        sched, inst.vehicles, station, delta, K_try, inst.boundary_vehicles,
        _objective_weights(cfg, inst),
    )
    K_feasible = max(p.departure for p in sched.values())

    K_fifo = max(K_grid, K_feasible)
    K = math.ceil(round(K_fifo * (1.0 + cfg.horizon_slack), 9))
    return {
        "K_grid_fifo": int(K_grid),
        "K_feasible_fifo": int(K_feasible),
        "K_fifo": int(K_fifo),
        "K": int(K),
        "horizon_minutes": float(K * delta),
        "feasible_fifo_total_sojourn": float(check.objective),
        "feasible_fifo_moved": int(counts["moved"]),
    }


def _n_optimized(cfg: TrialConfig, inst) -> int:
    """Vehicles inside ``cfg.objective_cohorts`` -- the J in (total sojourn)/J."""
    return sum(1 for c in inst.cohorts.values() if c in cfg.objective_cohorts)


def build_instance(cfg: TrialConfig, env: ChargingStationEnv):
    """
    Turn the finished episode into the offline instance every model solves.

    Arrivals are shifted back to ``t=0``; the boundary queue and the
    already-charging vehicles are folded in per ``cfg.include_queued`` /
    ``cfg.boundary_mode``. Passing the station runs the whole-module check on
    FIXED vehicles here, where it can name the offending pile-slot.

    The horizon is ``closed_system_horizon``'s, and the instance is
    exactly the closed vehicle set (checked here). The horizon is in the
    returned summary as ``horizon_minutes`` / ``K``; pass it to the stage
    functions.
    """
    station = StationSpec(
        n_piles=cfg.n_piles,
        n_connectors=cfg.n_connectors,
        n_modules=cfg.n_modules,
        p_module=cfg.p_module,
    )
    horizon = closed_system_horizon(cfg, env, station)
    horizon_minutes = horizon["horizon_minutes"]
    inst = _measurement_instance(cfg, env, station, env.engine.metrics.arrived_post_warmup, horizon_minutes)
    expected = _closed_set_ids(env)
    modelled = {v.id for v in inst.vehicles} | set(inst.discarded)
    if modelled != expected:
        raise RuntimeError(
            "closed-system instance does not match the simulated vehicle set: missing "
            f"{sorted(expected - modelled)[:10]}, extra {sorted(modelled - expected)[:10]}."
        )
    summary = {
        **horizon,
        "n_vehicles": len(inst.vehicles),
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


def fifo_schedule(cfg: TrialConfig, inst, station: StationSpec, env, horizon_minutes: float):
    """
    The drained FIFO run rebuilt on the slot grid as a whole-module schedule
    (``offline_cl_PB.schedule_from_simulation``) at this trial's horizon --
    the same schedule ``closed_system_horizon`` measured, so it completes
    every vehicle by ``K``. The starting schedule of DW and the
    ``"simulation"`` seed of B&P.
    """
    K = math.ceil(round(horizon_minutes / cfg.delta, 9))
    sched, _counts = schedule_from_simulation(
        env.engine.metrics.arrived_evs, inst.vehicles, station, cfg.delta, K,
        inst.boundary_vehicles, warmup_period=env.engine.warmup_period,
    )
    if sched is None:
        raise RuntimeError(f"the discretized FIFO schedule does not fit the horizon K={K}")
    return sched


def run_exact(
    cfg: TrialConfig,
    inst,
    station: StationSpec,
    *,
    horizon_minutes: float,
    solver_progress: bool = False,
) -> dict[str, Any]:
    """
    Build and solve the connector-lane MILP; report incumbent *and* bound.

    ``horizon_minutes`` is ``build_instance``'s ``horizon_minutes``.

    ``objective`` is the solver's incumbent (total sojourn in minutes, an
    upper bound on the optimum); ``best_bound`` is Gurobi's ``ObjBound``
    rounded up to a multiple of ``delta`` (the free tightening; the raw
    value is ``best_bound_raw``) -- a lower bound, same units;
    ``mean_sojourn_LB`` is it per optimized vehicle -- so a time-limited run
    still brackets the optimum. ``status``/``mip_gap`` say whether the limit
    was what stopped it -- ``mip_gap`` is relative to total sojourn.

    ``cfg.exact_tolerance_min`` (``x``, mean-sojourn minutes) becomes a
    total-sojourn tolerance ``x * n_optimized`` (``tolerance_used``); the
    solve stops once the incumbent is proven that close. ``mean_gap_min``
    is ``(objective - best_bound) / n_optimized``, the proven distance from
    the optimum in mean-sojourn minutes, and ``within_tolerance`` says
    whether it is within ``x`` (with no tolerance: whether optimality was
    proven).

    ``solver_progress=True`` turns on Gurobi's own solve log (branch-and-bound
    node counts, incumbent/bound as they improve) -- see ``run_trial``'s
    docstring for when that is (and is not) a good idea.
    """
    row, cl = _solve_exact(
        cfg, inst, station, horizon_minutes=horizon_minutes, solver_progress=solver_progress
    )
    cl.model.dispose()
    return row


def _solve_exact(
    cfg: TrialConfig,
    inst,
    station: StationSpec,
    *,
    horizon_minutes: float,
    solver_progress: bool,
):
    """``run_exact``'s work; also returns the solved model (for seeding B&P)."""
    t0 = time.perf_counter()
    cl = build_cl_model(
        inst.vehicles,
        station,
        delta=cfg.delta,
        horizon_minutes=horizon_minutes,
        boundary_vehicles=inst.boundary_vehicles,
        cohorts=inst.cohorts,
        objective_cohorts=cfg.objective_cohorts,
        break_symmetry=cfg.break_symmetry,
        tie_break=cfg.tie_break,
    )
    build_s = time.perf_counter() - t0

    n_opt = _n_optimized(cfg, inst)
    tolerance = cfg.exact_tolerance_min * n_opt if cfg.exact_tolerance_min is not None else None

    t1 = time.perf_counter()
    solve_cl_model(
        cl, mip_gap=cfg.mip_gap, time_limit=cfg.time_limit, tolerance=tolerance,
        verbose=solver_progress,
    )
    solve_s = time.perf_counter() - t1

    sol = extract_exact(cl)
    # ObjBound is unavailable once a tie-break objective is set (Gurobi
    # raises), the same reason extract_solution reports mip_gap as nan there.
    try:
        best_bound_raw = float(cl.model.ObjBound)
    except Exception:
        best_bound_raw = NAN
    best_bound = sol.best_bound  # rounded up to a multiple of delta

    completed, arrived, diag = _model_groups(sol.per_vehicle, sol.K)
    row = {
        "status": sol.status,
        "objective": sol.objective,  # incumbent total sojourn, minutes
        "best_bound": best_bound,  # proven lower bound, rounded up to a multiple of delta, minutes
        "best_bound_raw": best_bound_raw,  # Gurobi's ObjBound as reported
        "mip_gap_achieved": sol.mip_gap,  # relative to total sojourn
        **_tolerance_report(sol.objective, best_bound, n_opt, cfg.exact_tolerance_min),
        "total_sojourn": sol.total_sojourn,
        "mean_sojourn": sol.mean_sojourn,
        "mean_sojourn_LB": best_bound / sol.n_optimized if sol.n_optimized else NAN,
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
    return row, cl


def _tolerance_report(objective: float, bound: float, n_opt: int, tolerance_min: float | None) -> dict[str, Any]:
    """
    ``tolerance_used`` (total-sojourn minutes, ``nan`` without a tolerance),
    ``mean_gap_min`` (proven distance from the optimum, mean-sojourn
    minutes) and ``within_tolerance`` (that distance is within the
    tolerance; with none, that optimality was proven).
    """
    gap = (objective - bound) / n_opt if n_opt and math.isfinite(bound) else NAN
    allowed = tolerance_min if tolerance_min is not None else 0.0
    return {
        "tolerance_used": tolerance_min * n_opt if tolerance_min is not None else NAN,
        "mean_gap_min": gap,
        "within_tolerance": bool(math.isfinite(gap) and gap <= allowed + 1e-9),
    }


# --------------------------------------------------------------------------- #
# Stage 4 -- branch-and-price (the exact model, solved by offline_cl_PB)
# --------------------------------------------------------------------------- #

BP_INITIAL_SCHEDULES = ("simulation", "exact", "both", "none")


def _bp_seed(
    cfg: TrialConfig, inst, station: StationSpec, env, exact_model, horizon_minutes: float
) -> tuple[dict | None, str, float]:
    """
    The initial incumbent for B&P, per ``cfg.bp_initial_schedule``.

    Returns ``(schedule or None, source label, objective or nan)``. Each
    candidate is a complete schedule checked against the exact model here
    (and again inside B&P), never a bare objective value -- so a seed that
    does not survive discretization is dropped, not trusted.
    """
    choice = cfg.bp_initial_schedule
    if choice not in BP_INITIAL_SCHEDULES:
        raise ValueError(f"bp_initial_schedule must be one of {BP_INITIAL_SCHEDULES}, got {choice!r}")
    K = math.ceil(round(horizon_minutes / cfg.delta, 9))
    weights = _objective_weights(cfg, inst)
    candidates: list[tuple[str, dict]] = []
    if choice in ("simulation", "both"):
        candidates.append(("simulation", fifo_schedule(cfg, inst, station, env, horizon_minutes)))
    if choice in ("exact", "both") and exact_model is not None and exact_model.model.SolCount > 0:
        sched, _conns = schedule_from_compact(exact_model)
        candidates.append(("exact", sched))
    best: tuple[dict | None, str, float] = (None, "none", NAN)
    for label, sched in candidates:
        try:
            obj = validate_schedule(
                sched, inst.vehicles, station, cfg.delta, K, inst.boundary_vehicles, weights
            ).objective
        except ScheduleValidationError:
            continue
        if best[0] is None or obj < best[2]:
            best = (sched, label, obj)
    return best


def _bp_bound_history(sol, offset_s: float) -> list[dict[str, Any]]:
    """
    ``PBSolution.bound_history`` in sweep units: the certified bracket on the
    optimum over time, one row per change.

    ``time_s`` is measured from the start of the B&P stage (``offset_s`` adds
    the seed construction that precedes the solver), so the last row's time
    is ~``bp_runtime_s``. Bounds are given in objective units (total sojourn,
    minutes) and per vehicle as mean sojourn over ``objective_cohorts`` --
    the same figures as ``bp_objective`` / ``bp_best_bound`` and
    ``bp_mean_sojourn`` / ``bp_mean_sojourn_LB``. ``gap_pct`` is
    ``100 * (upper - lower) / upper``, the gap relative to total sojourn. A
    lower bound of ``-inf`` (before the root is first priced) is stored as
    NaN.

    These bracket the optimum of the horizon-``K`` closed-system model.
    """
    n = sol.n_optimized
    rows = []
    for h in sol.bound_history:
        lb = h["lower_bound"] if math.isfinite(h["lower_bound"]) else NAN
        ub = h["upper_bound"] if math.isfinite(h["upper_bound"]) else NAN
        gap = ub - lb
        rows.append(
            {
                "time_s": offset_s + h["time_s"],
                "lower_bound": lb,
                "upper_bound": ub,
                "gap": gap,
                "gap_pct": 100.0 * gap / ub if ub else NAN,
                "lower_bound_min": lb / n if n else NAN,
                "upper_bound_min": ub / n if n else NAN,
                "nodes": h["nodes"],
                "event": h["event"],
            }
        )
    return rows


def run_bp(
    cfg: TrialConfig,
    inst,
    station: StationSpec,
    env: ChargingStationEnv,
    *,
    exact_model=None,
    horizon_minutes: float,
    solver_progress: bool = False,
) -> dict[str, Any]:
    """
    Solve the exact whole-module model by branch-and-price (``offline_cl_PB``).

    ``horizon_minutes``: as in ``run_exact``.

    ``objective`` is the best schedule found (total sojourn in minutes, an
    upper bound on the optimum) and ``best_bound`` a proven lower bound, so a
    time-limited run still brackets the optimum -- the same two fields
    ``run_exact`` reports, for the same model. ``best_bound`` is rounded up
    to a multiple of ``delta`` (the free tightening). ``status`` is
    ``OPTIMAL`` when they coincide, ``WITHIN_TOLERANCE`` when
    ``cfg.bp_tolerance_min`` stopped it early with the bracket inside the
    tolerance, else ``TIME_LIMIT``. ``tolerance_used``, ``mean_gap_min``
    and ``within_tolerance`` are as in ``run_exact``. ``root_lower_bound``
    is the root node's bound (the whole-module Dantzig-Wolfe bound).

    The initial incumbent comes from ``cfg.bp_initial_schedule`` (see
    ``TrialConfig``); ``seed_source``/``seed_objective`` record which one
    was used and its value, ``incumbent_source`` where the final schedule
    came from. ``exact_model`` is the solved compact model from the same
    trial, needed only for the ``"exact"``/``"both"`` seeds.

    ``bound_history`` is the certified bracket over time (see
    ``_bp_bound_history``); the sweep writes it to ``bound_history.csv``,
    not to ``results.csv``.

    ``solver_progress=True`` prints B&P's own per-node log.
    """
    t0 = time.perf_counter()
    horizon = horizon_minutes
    seed, seed_source, seed_objective = _bp_seed(cfg, inst, station, env, exact_model, horizon)
    seed_s = time.perf_counter() - t0
    n_opt = _n_optimized(cfg, inst)
    sol = BranchAndPrice(
        inst.vehicles,
        station,
        cfg.delta,
        horizon,
        boundary_vehicles=inst.boundary_vehicles,
        cohorts=inst.cohorts,
        objective_cohorts=cfg.objective_cohorts,
        initial_schedule=seed,
        time_limit=cfg.bp_time_limit,
        gap_abs=cfg.bp_tolerance_min * n_opt if cfg.bp_tolerance_min is not None else 0.0,
        progress=solver_progress,
    ).solve()
    elapsed = time.perf_counter() - t0
    completed, arrived, diag = _model_groups(sol.per_vehicle, sol.K)
    return {
        "status": sol.status,
        "bound_history": _bp_bound_history(sol, seed_s),
        "objective": sol.objective,  # best schedule's total sojourn, minutes
        "best_bound": sol.lower_bound,  # proven lower bound, minutes
        "gap": sol.gap,  # objective - best_bound (0 when OPTIMAL)
        "rel_gap": sol.gap / sol.objective if sol.objective else NAN,  # relative to total sojourn
        **_tolerance_report(sol.objective, sol.lower_bound, n_opt, cfg.bp_tolerance_min),
        "root_lower_bound": sol.root_lower_bound,
        "total_sojourn": sol.total_sojourn,
        "mean_sojourn": sol.mean_sojourn,
        "total_sojourn_LB": sol.total_sojourn_LB,
        "mean_sojourn_LB": sol.mean_sojourn_LB,
        "n_vehicles": sol.n_vehicles,
        "n_optimized": sol.n_optimized,
        "K": sol.K,
        "nodes": sol.nodes_processed,
        "columns": sol.columns,
        "seed_source": seed_source,
        "seed_objective": seed_objective,
        "incumbent_source": sol.incumbent_source,
        "compact_check_max_violation": (
            sol.compact_check.max_violation if sol.compact_check is not None else NAN
        ),
        "runtime_s": elapsed,
        **diag,
        "completed_by_cohort": completed,
        "arrived_by_cohort": arrived,
    }


# --------------------------------------------------------------------------- #
# Stage 5 -- Dantzig-Wolfe
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
    cfg: TrialConfig,
    inst,
    station: StationSpec,
    *,
    horizon_minutes: float,
    initial_schedule: dict | None = None,
    solver_progress: bool = False,
) -> dict[str, Any]:
    """
    Run the decomposition and report the whole bound family.

    ``horizon_minutes``: as in ``run_exact``. ``initial_schedule`` is the
    complete schedule column generation starts from (``fifo_schedule`` in a
    trial); ``None`` lets DW build one by list scheduling.

    ``LB`` is the certified anytime Lagrangian bound (valid even when column
    generation did not converge); ``z_rmp`` is the final restricted-master LP
    value and ``rmp_gap = z_rmp - LB`` its stopping-criterion gap -- a
    diagnostic of how close pricing got to proving ``z_RMP == z_MP``, *not* a
    certified bracket. ``UB``/``gap`` and the ``*_UB`` sojourns are ``nan``
    unless ``cfg.solve_integer_ub`` asked for price-and-branch.

    ``solver_progress=True`` turns on column generation's own per-iteration
    log (``[colgen] iter N: z_RMP=..., best_LB=..., gap=..., ...``) -- see
    ``run_trial``'s docstring for when that is (and is not) a good idea.

    ``gap_tolerance`` (objective units: minutes of total sojourn, passed to
    column generation) comes from ``cfg.gap_tolerance_target_min`` when set:
    a target of ``m`` mean-sojourn minutes becomes ``m * n_optimized``, using
    this instance's own ``n_optimized`` (the vehicles actually in
    ``objective_cohorts``). Falls back to ``cfg.gap_tolerance`` otherwise.
    The value actually used is reported back as ``gap_tolerance_used``.
    """
    n_optimized = _n_optimized(cfg, inst)
    if cfg.gap_tolerance_target_min is not None:
        gap_tolerance = cfg.gap_tolerance_target_min * n_optimized
    else:
        gap_tolerance = cfg.gap_tolerance

    t0 = time.perf_counter()
    sol, cg = solve_by_decomposition(
        inst.vehicles,
        station,
        delta=cfg.delta,
        horizon_minutes=horizon_minutes,
        initial_schedule=initial_schedule,
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

    dw_lb = lattice_round_up(sol.LB, cfg.delta)

    # Only price-and-branch produces a schedule to split into groups.
    dw_completed = dw_arrived = dw_diag = None
    if cfg.solve_integer_ub:
        dw_completed, dw_arrived, dw_diag = _model_groups(sol.per_vehicle, sol.K)

    return {
        "status": _dw_status(cfg, cg, elapsed),
        "converged": sol.converged,
        "iterations": sol.iterations,
        "columns_purged": sol.columns_purged,
        "gap_tolerance_used": gap_tolerance,  # minutes of total sojourn, whichever mode set it
        # Lower bounds, objective units (total sojourn, minutes). LB is
        # rounded up to a multiple of delta (the free tightening: every
        # schedule's objective is one); LB_raw is column generation's own.
        "LB": dw_lb,
        "LB_raw": sol.LB,
        "z_rmp": sol.LB + sol.rmp_gap,  # final restricted-master LP value
        "rmp_gap": sol.rmp_gap,
        "sojourn_floor": cg.sojourn_floor,  # cheap sanity floor
        # The same lower bound per optimized vehicle.
        "total_sojourn_LB": dw_lb,
        "mean_sojourn_LB": dw_lb / sol.n_optimized if sol.n_optimized else NAN,
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
    run_bp_model: bool = False,
    verbose: bool = True,
    solver_progress: bool = False,
) -> TrialResult:
    """
    Simulate, then bound, one configuration.

    The ``run_*`` flags switch off *reporting/solving* stages independently.
    The DES episode itself always runs -- every offline model is built from
    its realized arrivals and boundary snapshot -- so ``run_sim=False``
    skips only the sim/grid sojourn tables, not the episode.

    ``run_exact_model`` and ``run_bp_model`` solve the same exact
    whole-module model two ways (compact MILP / branch-and-price); either,
    both or neither can run. When both run, exact goes first so B&P can be
    seeded from its incumbent (``bp_initial_schedule="exact"``/``"both"``).
    Stages run in the order exact, bp, dw.

    ``verbose`` prints one line per trial/stage (this module's own summary --
    "instance: J=...", "exact: OPTIMAL obj=..."). ``solver_progress`` is a
    separate, much noisier switch: it turns on each solver's OWN internal
    log -- Gurobi's native branch-and-bound log for the exact model, B&P's
    per-node log, column generation's own ``[colgen] iter N: z_RMP=...,
    best_LB=..., gap=...`` line per iteration for DW. Meant for debugging a
    single slow/stuck trial, not for a sweep of many -- left on across a
    whole sweep it floods the console with every solver's full internal log
    for every trial.

    A solve that raises is recorded as ``{"error": ...}`` on its stage and
    the trial still returns, so one failed point cannot abort a sweep.
    Building the instance is not a solve: a trial whose vehicle set cannot
    be matched (e.g. a full queue turned vehicles away) raises, since every
    number it produced would compare different sets.
    """
    env, sim_runtime, arrivals = run_episode(cfg)
    result = TrialResult(config=cfg, counts=episode_counts(env), arrivals=arrivals)
    # On counts (not just on the sim stage) so the episode's cost is recorded
    # even for a run_sim=False sweep, where the episode still had to happen.
    result.counts["sim_runtime_s"] = sim_runtime

    # Built up front (cheap, no solver) even when only the sim is reported:
    # it checks that the models and the simulation share one vehicle set and
    # records the horizon.
    inst, station, result.instance = build_instance(cfg, env)
    horizon = result.instance["horizon_minutes"]

    if run_sim:
        sim, grid = simulation_metrics(env, cfg.delta)
        # Station load over the measured phase (see metrics/README.md): its
        # keys become sim_* columns in results.csv.
        load = station_load(env, since=float(env.engine.warmup_period), mean_interarrival=cfg.effective_mean_interarrival)
        result.sim = {"runtime_s": sim_runtime, **load, **sim}
        result.grid = dict(grid)

    if not (run_exact_model or run_bp_model or run_dw_model):
        return result

    if verbose:
        drain = (
            f", drained {result.counts['drain_minutes']:.1f} min past T, "
            f"K={result.instance['K']} (grid FIFO {result.instance['K_grid_fifo']}, "
            f"feasible FIFO {result.instance['K_feasible_fifo']}, "
            f"slack {cfg.horizon_slack:g})"
        )
        print(
            f"  instance: J={result.instance['n_vehicles']} "
            f"({result.instance['cohort_sizes']}), "
            f"optimized={result.instance['n_optimized']}{drain}"
        )

    # The solved compact model is kept only if B&P may be seeded from it.
    keep_exact = run_bp_model and cfg.bp_initial_schedule in ("exact", "both")
    exact_model = None

    def solve_exact(cfg, inst, station, *, horizon_minutes, solver_progress):
        nonlocal exact_model
        row, cl = _solve_exact(
            cfg, inst, station, horizon_minutes=horizon_minutes, solver_progress=solver_progress
        )
        if keep_exact:
            exact_model = cl
        else:
            cl.model.dispose()
        return row

    def solve_bp(cfg, inst, station, *, horizon_minutes, solver_progress):
        return run_bp(
            cfg, inst, station, env, exact_model=exact_model,
            horizon_minutes=horizon_minutes, solver_progress=solver_progress,
        )

    def solve_dw(cfg, inst, station, *, horizon_minutes, solver_progress):
        return run_dw(
            cfg, inst, station, horizon_minutes=horizon_minutes,
            initial_schedule=fifo_schedule(cfg, inst, station, env, horizon_minutes),
            solver_progress=solver_progress,
        )

    try:
        for enabled, name, fn in (
            (run_exact_model, "exact", solve_exact),
            (run_bp_model, "bp", solve_bp),
            (run_dw_model, "dw", solve_dw),
        ):
            if not enabled:
                continue
            t0 = time.perf_counter()
            try:
                stage = fn(
                    cfg, inst, station, horizon_minutes=horizon, solver_progress=solver_progress
                )
            except Exception as exc:  # a failed solve must not kill the sweep
                stage = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "runtime_s": time.perf_counter() - t0,
                }
            setattr(result, name, stage)
            if verbose:
                print(f"  {name}: {_stage_line(stage)}")
    finally:
        if exact_model is not None:
            exact_model.model.dispose()
    return result


def _stage_line(stage: dict[str, Any]) -> str:
    """One-line progress summary for a finished stage."""
    if "error" in stage:
        return f"FAILED {stage['error']}"
    return _stage_summary(stage)


def _stage_summary(stage: dict[str, Any]) -> str:
    """The stage-specific part of ``_stage_line``."""
    if "LB" in stage:
        return (
            f"{stage['status']} LB={stage['LB']:.2f} "
            f"mean_sojourn_LB={stage['mean_sojourn_LB']:.2f} min "
            f"({stage['runtime_s']:.1f}s, {stage['iterations']} iters)"
        )
    if "nodes" in stage:  # branch-and-price
        return (
            f"{stage['status']} obj={stage['objective']:.2f} "
            f"bound={stage['best_bound']:.2f} gap={stage['rel_gap']:.2%} "
            f"mean_sojourn={stage['mean_sojourn']:.2f} min "
            f"({stage['runtime_s']:.1f}s, {stage['nodes']} nodes, "
            f"seed={stage['seed_source']})"
        )
    return (
        f"{stage['status']} obj={stage['objective']:.2f} "
        f"bound={stage['best_bound']:.2f} gap={stage['mip_gap_achieved']:.2%} "
        f"mean_sojourn={stage['mean_sojourn']:.2f} min "
        f"({stage['runtime_s']:.1f}s)"
    )
