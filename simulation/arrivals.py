"""
Poisson arrival-process sampling, independent of the DES engine.

``SimulationEngine`` used to sample arrivals inline in ``_generate_arrivals``.
That sampling now lives here so it can be reused outside the engine: call
``generate_arrivals`` directly to get a list of ``EV`` objects (exactly the
format the engine itself pushes onto its event heap), then hand that list to
``SimulationEngine.set_arrivals`` / ``reset(arrivals=...)`` or
``ChargingStationEnv(arrivals=...)``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config import BATTERY_CAP_OPTIONS, SOC_I_BOUNDS, SOC_F_BOUNDS
from models.ev import EV


def generate_arrivals(
    mean_interarrival: float,
    max_time: float,
    rng: np.random.Generator,
    battery_cap_options: Sequence[float] | None = None,
    soc_i_bounds: Sequence[float] | None = None,
    soc_f_bounds: Sequence[float] | None = None,
    start_id: int = 0,
) -> list[EV]:
    """
    Sample one Poisson(1 / mean_interarrival) arrival stream over [0, max_time].

    Exponential inter-arrival gaps; each EV's battery capacity, initial SoC,
    and target SoC are drawn i.i.d. from ``battery_cap_options`` /
    ``soc_i_bounds`` / ``soc_f_bounds`` (each defaults to the matching value in
    ``config.py`` when omitted -- note ``battery_cap_options`` is expected in
    the same units as ``config.BATTERY_CAP_OPTIONS``, i.e. kW*min, already
    converted from kWh via ``HR2MIN``).

    Returns EVs sorted by arrival time with sequential ids starting at
    ``start_id``.
    """
    if mean_interarrival <= 0:
        raise ValueError(
            f"mean_interarrival must be positive, got {mean_interarrival}"
        )
    if max_time <= 0:
        raise ValueError(f"max_time must be positive, got {max_time}")

    if battery_cap_options is None:
        battery_cap_options = BATTERY_CAP_OPTIONS
    if soc_i_bounds is None:
        soc_i_bounds = SOC_I_BOUNDS
    if soc_f_bounds is None:
        soc_f_bounds = SOC_F_BOUNDS

    gaps = rng.exponential(mean_interarrival, int(max_time / mean_interarrival * 5))
    arrival_times = gaps.cumsum()
    arrival_times = arrival_times[arrival_times <= max_time]

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
