"""
Helpers for DES ``charge_trace`` samples: deduplication and densification.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from models.ev import EV
from visualization.charging_theory import theoretical_p_of_s


@dataclass(frozen=True)
class ChargeTraceSample:
    """One instantaneous sample from ``EV.charge_trace``."""

    t: float
    s: float
    p_req: float
    p_act: float
    p_allot: float
    n_bricks: int


def unique_charge_samples(
    ev: EV,
) -> list[tuple[float, float, float, float, float]]:
    """Return charge_trace samples with duplicate times collapsed (keep last)."""
    cleaned: list[tuple[float, float, float, float, float]] = []
    for sample in ev.charge_trace:
        if cleaned and abs(cleaned[-1][0] - sample[0]) < 1e-12:
            cleaned[-1] = sample
        else:
            cleaned.append(sample)
    return cleaned


def bricks_from_allotment(p_allot: float, p_brick: float) -> int:
    """Brick count implied by a kW allotment (allotment is n_bricks * p_brick)."""
    if p_brick <= 0:
        return 0
    return int(round(p_allot / p_brick))


def parsed_trace_samples(ev: EV, p_brick: float) -> list[ChargeTraceSample]:
    """Deduplicated ``charge_trace`` rows with brick counts."""
    out: list[ChargeTraceSample] = []
    for t, s, p_req, p_act, p_allot in unique_charge_samples(ev):
        out.append(
            ChargeTraceSample(
                t=float(t),
                s=float(s),
                p_req=float(p_req),
                p_act=float(p_act),
                p_allot=float(p_allot),
                n_bricks=bricks_from_allotment(float(p_allot), p_brick),
            )
        )
    return out


def trace_power_plateaus(
    ev: EV,
    p_brick: float,
    *,
    t_end: float | None = None,
) -> list[tuple[float, float, float, ChargeTraceSample]]:
    """
    Piecewise-constant actual-power plateaus between trace samples.

    Returns (t_start, t_end, p_act, start_sample) for each segment. The last
    segment ends at ``t_end`` when provided, otherwise at the final sample time.
    """
    samples = parsed_trace_samples(ev, p_brick)
    if not samples:
        return []

    plateaus: list[tuple[float, float, float, ChargeTraceSample]] = []
    for i, sample in enumerate(samples):
        t1 = samples[i + 1].t if i + 1 < len(samples) else sample.t
        if t_end is not None and i + 1 >= len(samples):
            t1 = max(t1, float(t_end))
        if t1 <= sample.t:
            continue
        plateaus.append((sample.t, t1, sample.p_act, sample))
    return plateaus


def trace_label_text(sample: ChargeTraceSample) -> str:
    """Compact multi-line label for a trace sample."""
    return (
        f"P={sample.p_act:.1f}\n"
        f"t={sample.t:.1f}\n"
        f"s={sample.s:.3f}\n"
        f"n={sample.n_bricks}"
    )


def charge_change_event_times(
    ev: EV,
    p_brick: float,
    *,
    t_lo: float | None = None,
    t_hi: float | None = None,
    exclude_times: set[float] | list[float] | None = None,
    time_tol: float = 1e-6,
) -> list[float]:
    """
    Times when this EV's brick allotment changes (redistribution / charge change).

    Skips the initial plug-in sample. Keeps only samples whose ``n_bricks``
    differs from the previous sample. Optional ``t_lo`` / ``t_hi`` window filter.

    ``exclude_times`` drops any candidate that falls within ``time_tol`` of a
    listed instant (use plug-in / departure times to keep *sole* charge changes).
    """
    samples = parsed_trace_samples(ev, p_brick)
    if len(samples) < 2:
        return []

    excluded = list(exclude_times) if exclude_times is not None else []

    def _near_excluded(t: float) -> bool:
        for t_ex in excluded:
            if abs(t - t_ex) <= time_tol:
                return True
        return False

    times: list[float] = []
    for prev, cur in zip(samples, samples[1:]):
        if cur.n_bricks == prev.n_bricks:
            continue
        if t_lo is not None and cur.t < t_lo:
            continue
        if t_hi is not None and cur.t > t_hi:
            continue
        # Never treat this EV's own plug-in / departure as a sole charge change.
        if ev.service_start_time is not None and abs(
            cur.t - ev.service_start_time
        ) <= time_tol:
            continue
        if np.isfinite(ev.departure_time) and abs(cur.t - ev.departure_time) <= time_tol:
            continue
        if _near_excluded(cur.t):
            continue
        times.append(cur.t)
    return times


def densify_charge_trace(
    ev: EV, dt: float = 0.25
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Rebuild dense (t, SoC, p_req, p_act) from sparse charge_trace segments.

    Between two logged samples we hold the brick allotment fixed and integrate
    SoC with p_act = min(p_req(s), p_allot), which matches the DES physics.
    """
    cleaned = unique_charge_samples(ev)
    if not cleaned:
        return (
            np.array([]),
            np.array([]),
            np.array([]),
            np.array([]),
        )

    ts: list[float] = []
    ss: list[float] = []
    preqs: list[float] = []
    pacts: list[float] = []

    for i, (t0, s0, _pr0, _pa0, allot0) in enumerate(cleaned):
        p_req0 = float(theoretical_p_of_s(ev, s0))
        ts.append(t0)
        ss.append(s0)
        preqs.append(p_req0)
        pacts.append(min(p_req0, allot0))

        if i + 1 >= len(cleaned):
            break

        t1, s1, _pr1, _pa1, _allot1 = cleaned[i + 1]
        if t1 <= t0:
            continue

        s = s0
        t = t0
        while t + dt < t1 - 1e-12:
            p_req = float(theoretical_p_of_s(ev, s))
            p_act = min(p_req, allot0)
            if p_act <= 0:
                break
            old_s, old_p = ev.s_current, ev.p_act
            ev.s_current = s
            ev.p_act = p_act
            ev.update_s_next(dt)
            s = ev.s_next
            ev.s_current, ev.p_act = old_s, old_p
            t += dt
            ts.append(t)
            ss.append(s)
            preqs.append(float(theoretical_p_of_s(ev, s)))
            pacts.append(min(float(theoretical_p_of_s(ev, s)), allot0))

        ts.append(t1)
        ss.append(s1)
        p_req1 = float(theoretical_p_of_s(ev, s1))
        preqs.append(p_req1)
        pacts.append(min(p_req1, allot0))

    return (
        np.asarray(ts, dtype=float),
        np.asarray(ss, dtype=float),
        np.asarray(preqs, dtype=float),
        np.asarray(pacts, dtype=float),
    )
