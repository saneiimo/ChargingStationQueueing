"""Resolve EV objects from a finished ChargingStationEnv."""

from __future__ import annotations

import numpy as np

from env.charging_env import ChargingStationEnv
from models.ev import EV


def all_known_evs(env: ChargingStationEnv) -> dict[int, EV]:
    """Map EV id -> object from arrivals, finished, and still-plugged EVs."""
    by_id: dict[int, EV] = {}
    metrics = env.engine.metrics
    for ev in metrics.arrived_evs:
        by_id[ev.id] = ev
    for ev in metrics.finished_evs:
        by_id[ev.id] = ev
    for pile in env.engine.station.piles:
        for ev in pile.evs:
            by_id[ev.id] = ev
    return by_id


def get_evs_from_env(env: ChargingStationEnv, ev_ids: list[int]) -> list[EV]:
    """Resolve EV objects by id. Raises KeyError with a clear message if missing."""
    known = all_known_evs(env)
    if not known:
        raise ValueError("No EVs found in env (did the episode run?).")

    ids_sorted = sorted(known.keys())
    lo, hi = ids_sorted[0], ids_sorted[-1]
    missing = [i for i in ev_ids if i not in known]
    if missing:
        print(
            f"Error: EV id(s) {missing} not in env. "
            f"Valid EV numbers must be from {lo} to {hi} "
            f"(known ids: {ids_sorted[:20]}{'...' if len(ids_sorted) > 20 else ''})."
        )
        raise KeyError(f"Unknown EV ids: {missing}")

    return [known[i] for i in ev_ids]


def ev_location_label(ev: EV) -> str:
    pile = ev.pile_tracker
    pile_id = pile.id if pile is not None else "?"
    dispenser = ev.dispenser_id_tracker if ev.dispenser_id_tracker is not None else "?"
    return f"pile {pile_id}, dispenser {dispenser}"


def ev_window_times(ev: EV, episode_time: float) -> tuple[float, float]:
    """Return (plug-in, departure-or-now) for window overlap tests."""
    t0 = ev.service_start_time if ev.service_start_time is not None else 0.0
    t1 = ev.departure_time if np.isfinite(ev.departure_time) else float(episode_time)
    return float(t0), float(t1)


def evs_for_pile(env: ChargingStationEnv, pile_id: int) -> list[EV]:
    """Finished EVs plus any still plugged on this pile."""
    pile = env.engine.station.piles[pile_id]
    seen: set[int] = set()
    out: list[EV] = []

    for ev in env.engine.metrics.finished_evs:
        if ev.pile_tracker is pile and ev.dispenser_id_tracker is not None:
            out.append(ev)
            seen.add(ev.id)

    for ev in pile.evs:
        if ev.id not in seen:
            out.append(ev)
            seen.add(ev.id)

    return out
