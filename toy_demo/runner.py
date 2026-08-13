"""
Run one toy episode with fixed EVs / arrivals, then summarize metrics.

Flow
----
1. Build ``ChargingStationEnv`` from ``ToyStationSpec``.
2. Replace ``engine._generate_arrivals`` so ``reset`` pushes only your EVs
   (plus ``SIM_OVER`` at ``horizon``).
3. Roll out with a ``QueuePolicy`` via ``decide`` / ``step``, same as ``main.py``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from env.charging_env import ChargingStationEnv
from models.ev import EV
from policy.queue.base import QueuePolicy
from policy.queue.fifo import FIFOQueuePolicy
from simulation.event import Event, EventType

from .scenario import ToyEVSpec, ToyStationSpec, build_evs, suggest_horizon


def install_fixed_arrivals(
    env: ChargingStationEnv,
    evs: list[EV],
    *,
    horizon: float,
) -> None:
    """
    Make the next ``env.reset`` use ``evs`` instead of a random Poisson process.

    Also sets ``engine.max_time`` to ``horizon`` so SIM_OVER matches the toy day.
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be positive, got {horizon}")
    if any(ev.arrival_time > horizon for ev in evs):
        raise ValueError("Every EV arrival_time must be <= horizon")

    engine = env.engine
    engine.max_time = float(horizon)

    # Copy the list so later notebook edits to ``evs`` do not change a run
    # that already installed arrivals.
    fixed = list(evs)

    def _generate_arrivals() -> None:
        engine.event_heap.clear()
        for ev in fixed:
            engine.event_heap.push(
                Event(ev.arrival_time, EventType.ARRIVAL, obj=ev)
            )
        engine.event_heap.push(Event(engine.max_time, EventType.SIM_OVER))

    engine._generate_arrivals = _generate_arrivals  # type: ignore[method-assign]


def make_toy_env(station: ToyStationSpec) -> ChargingStationEnv:
    """Build a Gym env whose layout matches ``station`` (arrivals still random until install)."""
    return ChargingStationEnv(
        n_piles=station.n_piles,
        n_nozzles=station.n_nozzles,
        n_bricks=station.n_bricks,
        p_brick=station.p_brick,
        queue_capacity=station.queue_capacity,
        mean_interarrival=station.mean_interarrival,
    )


def run_toy_episode(
    station: ToyStationSpec,
    ev_specs: list[ToyEVSpec],
    *,
    policy: QueuePolicy | None = None,
    horizon: float | None = None,
    seed: int = 0,
    policy_seed: int = 1,
) -> ChargingStationEnv:
    """
    Run one customized episode and return the finished env (metrics + traces).

    Parameters
    ----------
    station :
        Pile / nozzle / brick layout.
    ev_specs :
        User-facing EV list (ids, arrivals, battery kWh, SoC targets).
    policy :
        Queue assignment rule. Defaults to FIFO + most-free-nozzle piles.
    horizon :
        SIM_OVER time in minutes. Default: last arrival + 90 min.
    seed, policy_seed :
        Seeds for env reset and policy RNG (tie-breaks only for most heuristics).
    """
    if policy is None:
        policy = FIFOQueuePolicy()

    evs = build_evs(ev_specs)
    t_end = float(horizon) if horizon is not None else suggest_horizon(evs)

    env = make_toy_env(station)
    install_fixed_arrivals(env, evs, horizon=t_end)

    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(policy_seed)
    done = bool(env.engine.terminated)
    while not done:
        mask = env.action_masks()
        ev, pile_id = policy.decide(obs, mask, rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)

    return env


def metrics_table(env: ChargingStationEnv) -> pd.DataFrame:
    """
    One-row-per-metric summary for slides (uses ``queueing_summary`` + counts).

    ``avg charge time`` is mean service time S (plug-in until departure)
    among finished EVs, in minutes.
    """
    m = env.engine.metrics
    T = float(env.engine.current_time)
    c = env.engine.station.n_piles * env.engine.station.n_nozzles
    q = m.queueing_summary(T, n_servers=c)

    rows: list[dict[str, Any]] = [
        {"metric": "sim time (min)", "value": q["T"]},
        {"metric": "arrived", "value": float(len(m.arrived_evs))},
        {"metric": "finished", "value": q["n_finished"]},
        {"metric": "dropped", "value": float(len(m.dropped_evs))},
        {"metric": "still queued", "value": float(len(env.engine.station.queue))},
        {
            "metric": "still plugged",
            "value": float(sum(len(p.evs) for p in env.engine.station.piles)),
        },
        {"metric": "total energy (kWh)", "value": m.total_energy()},
        {"metric": "average L", "value": q["L_sim"]},
        {"metric": "average Q", "value": q["Q_sim"]},
        {"metric": "mean wait W_q (min)", "value": q["W_q"]},
        {"metric": "avg charge time (min)", "value": q["S"]},
        {"metric": "mean service S (min)", "value": q["S"]},
        {"metric": "mean sojourn W (min)", "value": q["W"]},
        {"metric": "util_rho_sim", "value": q["rho_sim"]},
        {"metric": "util_rho_theory", "value": q["rho_theory"]},
        {"metric": "lambda_eff (1/min)", "value": q["lambda_eff"]},
    ]
    return pd.DataFrame(rows).set_index("metric")


def finished_ev_table(env: ChargingStationEnv) -> pd.DataFrame:
    """Per-finished-EV timeline for the presentation appendix."""
    from config import HR2MIN

    rows = []
    for ev in env.engine.metrics.finished_evs:
        pile = ev.pile_tracker.id if ev.pile_tracker is not None else None
        rows.append(
            {
                "ev_id": ev.id,
                "arrival": ev.arrival_time,
                "service_start": ev.service_start_time,
                "departure": ev.departure_time,
                "wait": (ev.service_start_time or 0.0) - ev.arrival_time,
                "service": (ev.departure_time - (ev.service_start_time or 0.0)),
                "s_i": ev.s_i,
                "s_f": ev.s_f,
                "s_end": ev.s_current,
                "battery_kwh": ev.c_b / HR2MIN,
                "pile": pile,
                "nozzle": ev.nozzle_id_tracker,
                "energy_kwh": ev.energy_received,
            }
        )
    return pd.DataFrame(rows).sort_values("ev_id").reset_index(drop=True)
