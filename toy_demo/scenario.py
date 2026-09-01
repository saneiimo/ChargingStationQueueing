"""
Editable specs for a presentation toy run.

Users think in kWh and minutes; ``build_evs`` converts battery size to the
simulator's internal kW·min units (see ``config.HR2MIN``).
"""

from __future__ import annotations

from dataclasses import dataclass

from config import HR2MIN, S_THRESH
from models.ev import EV


@dataclass
class ToyEVSpec:
    """One EV in the toy scenario (user-facing units)."""

    id: int
    arrival_time: float  # minutes from t = 0
    battery_kwh: float  # battery capacity in kWh
    s_i: float  # initial SoC in [0, 1]
    s_f: float  # target SoC in [0, 1]
    s_th: float = S_THRESH  # BMS taper start (optional override)


@dataclass
class ToyStationSpec:
    """Charging-station layout for the toy run."""

    n_piles: int = 2
    n_connectors: int = 2
    n_modules: int = 5
    p_module: float = 25.0  # kW per module
    queue_capacity: int = 10
    # Dummy mean gap: arrivals are replaced by the fixed EV list.
    mean_interarrival: float = 60.0


def build_ev(spec: ToyEVSpec) -> EV:
    """Turn a user-facing EV spec into a simulator ``EV`` (c_b in kW·min)."""
    if not (0.0 <= spec.s_i < spec.s_f <= 1.0):
        raise ValueError(
            f"EV {spec.id}: need 0 <= s_i < s_f <= 1, "
            f"got s_i={spec.s_i}, s_f={spec.s_f}"
        )
    if spec.battery_kwh <= 0:
        raise ValueError(f"EV {spec.id}: battery_kwh must be positive")
    if spec.arrival_time < 0:
        raise ValueError(f"EV {spec.id}: arrival_time must be >= 0")

    return EV(
        id=int(spec.id),
        c_b=float(spec.battery_kwh) * HR2MIN,
        s_i=float(spec.s_i),
        s_f=float(spec.s_f),
        arrival_time=float(spec.arrival_time),
        s_th=float(spec.s_th),
    )


def build_evs(specs: list[ToyEVSpec]) -> list[EV]:
    """Build EVs sorted by arrival time (stable for equal times by id)."""
    if not specs:
        raise ValueError("Need at least one ToyEVSpec")
    ids = [s.id for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate EV ids: {ids}")
    evs = [build_ev(s) for s in specs]
    evs.sort(key=lambda e: (e.arrival_time, e.id))
    return evs


def default_four_ev_specs() -> list[ToyEVSpec]:
    """
    Four illustrative EVs for slides.

    Staggered arrivals so pile sharing and charge changes show up clearly
    on a small station (edit freely in the notebook).
    """
    return [
        ToyEVSpec(id=0, arrival_time=0.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
        ToyEVSpec(id=1, arrival_time=2.0, battery_kwh=100.0, s_i=0.15, s_f=0.85),
        ToyEVSpec(id=2, arrival_time=8.0, battery_kwh=50.0, s_i=0.25, s_f=0.75),
        ToyEVSpec(id=3, arrival_time=12.0, battery_kwh=150.0, s_i=0.10, s_f=0.80),
    ]


def suggest_horizon(evs: list[EV], padding_min: float = 90.0) -> float:
    """
    Episode end time (minutes): last arrival plus padding.

    Padding should be long enough for the slowest EV to finish under sharing;
    raise it if anyone is still plugged at SIM_OVER.
    """
    last_arr = max(ev.arrival_time for ev in evs)
    return float(last_arr + padding_min)
