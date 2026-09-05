"""
Poisson arrival-process sampling, independent of the DES engine.

``SimulationEngine`` used to sample arrivals inline in ``_generate_arrivals``.
That sampling now lives here so it can be reused outside the engine: call
``generate_arrivals`` directly to get a list of ``EV`` objects (exactly the
format the engine itself pushes onto its event heap), then hand that list to
``SimulationEngine.set_arrivals`` / ``reset(arrivals=...)`` or
``ChargingStationEnv(arrivals=...)``.

Optional integer (grid) arrivals
--------------------------------
``delta_arr=None`` leaves the continuous exponential arrival times unchanged.

``delta_arr=d`` (minutes, typically a positive integer 1, 2, 3, 5, ...) snaps
each continuous time ``t`` to the nearest multiple of ``d``:

    snap(t) = round(t / d) * d

Examples at ``t = 3.69``:

- ``d = 1`` → 4
- ``d = 2`` → 4
- ``d = 3`` → 3  (3 is closer than 6)
- ``d = 5`` → 5.0

Ties at an exact halfway point follow Python 3 / NumPy banker's rounding
(round half to even). After snapping, times outside ``[0, max_time]`` are
dropped and the surviving list is re-sorted.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config import BATTERY_CAP_OPTIONS, SOC_I_BOUNDS, SOC_F_BOUNDS
from models.ev import EV


def validate_delta_arr(delta_arr: float | None) -> float | None:
    """Return ``None`` or a positive grid width; raise if ``delta_arr`` is <= 0."""
    if delta_arr is None:
        return None
    d = float(delta_arr)
    if d <= 0.0:
        raise ValueError(f"delta_arr must be positive when set, got {delta_arr}")
    return d


def snap_arrival_time(t: float, delta_arr: float) -> float:
    """Round ``t`` (minutes) to the nearest multiple of ``delta_arr``.

    ``snap_arrival_time(3.69, 1) == 4.0``, ``(..., 2) == 4.0``,
    ``(..., 3) == 3.0``, ``(..., 5) == 5.0``.
    """
    d = validate_delta_arr(delta_arr)
    if d is None:
        raise ValueError("snap_arrival_time requires a positive delta_arr")
    return float(round(t / d) * d)


def snap_arrivals(
    evs: list[EV],
    delta_arr: float | None,
    *,
    max_time: float | None = None,
) -> list[EV]:
    """
    Snap each EV's ``arrival_time`` onto the ``delta_arr`` minute grid.

    ``delta_arr is None`` returns ``evs`` unchanged (same objects). When set,
    returns fresh ``EV`` copies (ids, battery, SoC bounds preserved) with
    snapped times, dropping any that land outside ``[0, max_time]`` when
    ``max_time`` is given, then sorted by ``(arrival_time, id)``. Idempotent
    if the times are already on that grid.
    """
    d = validate_delta_arr(delta_arr)
    if d is None:
        return evs

    snapped: list[EV] = []
    for ev in evs:
        t = snap_arrival_time(ev.arrival_time, d)
        if t < 0.0:
            t = 0.0
        if max_time is not None and t > max_time:
            continue
        snapped.append(
            EV(
                id=ev.id,
                c_b=ev.c_b,
                s_i=ev.s_i,
                s_f=ev.s_f,
                arrival_time=t,
                s_th=ev.s_th,
                c_rate=ev.c_rate,
            )
        )
    snapped.sort(key=lambda e: (e.arrival_time, e.id))
    return snapped


def generate_arrivals(
    mean_interarrival: float,
    max_time: float,
    rng: np.random.Generator,
    battery_cap_options: Sequence[float] | None = None,
    soc_i_bounds: Sequence[float] | None = None,
    soc_f_bounds: Sequence[float] | None = None,
    start_id: int = 0,
    delta_arr: float | None = None,
) -> list[EV]:
    """
    Sample one Poisson(1 / mean_interarrival) arrival stream over [0, max_time].

    Exponential inter-arrival gaps; each EV's battery capacity, initial SoC,
    and target SoC are drawn i.i.d. from ``battery_cap_options`` /
    ``soc_i_bounds`` / ``soc_f_bounds`` (each defaults to the matching value in
    ``config.py`` when omitted -- note ``battery_cap_options`` is expected in
    the same units as ``config.BATTERY_CAP_OPTIONS``, i.e. kW*min, already
    converted from kWh via ``HR2MIN``).

    ``delta_arr``
        ``None`` (default): keep the continuous exponential times.
        Positive ``d``: snap each time to the nearest multiple of ``d`` minutes
        (see module docstring). Continuous draws in
        ``(max_time, max_time + d/2]`` can snap onto ``max_time`` and are
        kept; times that snap past ``max_time`` or below 0 are dropped.

    Returns EVs sorted by arrival time with sequential ids starting at
    ``start_id``.
    """
    if mean_interarrival <= 0:
        raise ValueError(
            f"mean_interarrival must be positive, got {mean_interarrival}"
        )
    if max_time <= 0:
        raise ValueError(f"max_time must be positive, got {max_time}")
    d = validate_delta_arr(delta_arr)

    if battery_cap_options is None:
        battery_cap_options = BATTERY_CAP_OPTIONS
    if soc_i_bounds is None:
        soc_i_bounds = SOC_I_BOUNDS
    if soc_f_bounds is None:
        soc_f_bounds = SOC_F_BOUNDS

    gaps = rng.exponential(mean_interarrival, int(max_time / mean_interarrival * 5))
    arrival_times = gaps.cumsum()
    if d is None:
        arrival_times = arrival_times[arrival_times <= max_time]
    else:
        # Keep draws that can still snap onto the horizon, then snap and clip.
        arrival_times = arrival_times[arrival_times <= max_time + 0.5 * d]
        snapped = np.array(
            [snap_arrival_time(float(t), d) for t in arrival_times],
            dtype=float,
        )
        snapped = np.clip(snapped, 0.0, None)
        arrival_times = snapped[snapped <= max_time]
        arrival_times.sort()

    lo_i, hi_i = soc_i_bounds
    lo_f, hi_f = soc_f_bounds

    evs: list[EV] = []
    for offset, t in enumerate(arrival_times):
        c_b = rng.choice(battery_cap_options)
        s_i = rng.uniform(lo_i, hi_i)
        s_f = rng.uniform(lo_f, hi_f)
        evs.append(
            EV(id=start_id + offset, c_b=c_b, s_i=s_i, s_f=s_f, arrival_time=float(t))
        )
    return evs


def clone_arrivals(evs: list[EV]) -> list[EV]:
    """
    Fresh ``EV`` copies with the same arrival spec (id, c_b, s_i, s_f, arrival_time).

    An externally-supplied arrival list may be installed once and reused
    across several ``SimulationEngine.reset`` calls; without cloning, the
    second reset would reuse EV objects still carrying runtime state (SoC,
    pile, energy) from the first episode instead of starting fresh, the way
    freshly-sampled arrivals do every reset.
    """
    return [
        EV(
            id=ev.id,
            c_b=ev.c_b,
            s_i=ev.s_i,
            s_f=ev.s_f,
            arrival_time=ev.arrival_time,
            s_th=ev.s_th,
            c_rate=ev.c_rate,
        )
        for ev in evs
    ]
