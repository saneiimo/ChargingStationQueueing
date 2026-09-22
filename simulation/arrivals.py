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

One stream per source of randomness
-----------------------------------
``generate_arrivals`` splits the ``rng`` it is given into two independent
child streams -- one for inter-arrival gaps, one for vehicle
characteristics (battery capacity, arrival SoC, target SoC). This matters
whenever a study sweeps an arrival-process knob.

Drawing both from a single stream couples them: the gap draw is sized
``int(max_time / mean_interarrival * 5)``, so changing *either* of those
changes how many numbers are consumed before the attribute draws begin,
and every vehicle comes out with different characteristics -- even the
ones whose arrival times did not move. A sweep over ``mean_interarrival``
then varies two things at once (how many cars arrive, *and* what kind of
cars exist), and point-to-point differences cannot be attributed to load.

With separate streams, vehicle *k*'s attributes depend only on ``k``, so a
sweep changes the arrival process alone and the realised fleet is held
fixed. This is ordinary common-random-numbers practice: give each
stochastic source its own stream so perturbing one input cannot ripple
into the others.

The gaps keep using the ``rng`` that was passed in and only the
characteristics move to a spawned child, so arrival times for a given seed
are unchanged from before the split -- see ``_attribute_stream``.

A rate sweep is coupled too, and for free: ``exponential(scale)`` is
``scale * standard_exponential``, so the *k*-th gap simply rescales with
the mean. Two runs at different ``mean_interarrival`` are one underlying
realisation stretched or compressed in time, carrying the same fleet.

The alignment is by arrival **index**, which is exact when only
``mean_interarrival`` / ``max_time`` change. Changing ``delta_arr`` can
snap a time across the ``max_time`` cut-off and add or drop an arrival,
which shifts every later index by one; hold it fixed when you want the
fleet held fixed.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from config import BATTERY_CAP_OPTIONS, SOC_I_BOUNDS, SOC_F_BOUNDS
from models.ev import EV


def _attribute_stream(rng: np.random.Generator) -> np.random.Generator:
    """
    A stream for vehicle characteristics, independent of ``rng`` itself.

    ``rng`` stays the inter-arrival gap stream and this child serves the
    battery/SoC draws. The asymmetry is deliberate rather than tidy: a child
    spawned off the parent's seed sequence is independent of the parent's
    own stream, so keeping gaps on the parent leaves every arrival time
    bit-identical to what a given seed produced before the two sources were
    separated. Only the characteristics move -- which is the whole point.
    Spawning reads the seed sequence and does not consume the parent's
    stream, so it is free to happen at any point.

    Requires a generator carrying a seed sequence, which is what
    ``np.random.default_rng(...)`` returns. A generator built around a bare
    bit-generator state cannot be spawned from; that raises rather than
    silently falling back to one shared stream, since the shared stream is
    exactly the coupling this exists to remove.
    """
    try:
        (attr_rng,) = rng.spawn(1)
    except (AttributeError, TypeError) as exc:  # no seed sequence to spawn from
        raise TypeError(
            "generate_arrivals needs an rng it can spawn an independent child "
            "stream from, to keep vehicle characteristics separate from the "
            "inter-arrival gaps. Pass np.random.default_rng(seed) rather than "
            "a Generator wrapped around a bare bit-generator state."
        ) from exc
    return attr_rng


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

    ``rng`` is split into two independent child streams -- gaps and vehicle
    characteristics -- so that vehicle *k*'s battery/SoC depend only on
    ``k``, never on how many gap draws preceded them. Two calls with the
    same seed but different ``mean_interarrival`` (or ``max_time``)
    therefore describe the *same fleet* arriving at a different rate,
    rather than a freshly resampled one. See this module's docstring, "One
    stream per source of randomness", for why that is worth having, and
    ``_split_streams`` for the mechanics.

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

    # rng itself stays the gap stream; characteristics get their own child.
    attr_rng = _attribute_stream(rng)

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
        # attr_rng, not rng: three draws per vehicle off a stream nothing
        # else touches, so vehicle k's characteristics depend only on k --
        # see this module's docstring, "One stream per source of randomness".
        c_b = attr_rng.choice(battery_cap_options)
        s_i = attr_rng.uniform(lo_i, hi_i)
        s_f = attr_rng.uniform(lo_f, hi_f)
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
