"""
Per-vehicle / station data for the offline MILP (see ``model.py``).

Unit convention
---------------
The simulator stores battery capacity (``EV.c_b``) as real kWh multiplied by
``HR2MIN`` (60) -- see ``config.py`` / ``models/ev.py`` -- chosen so that
``power [kW] * time [min]`` lands in the same units as ``c_b`` with no
conversion factor. ``VehicleData.Q`` keeps that convention (and hence ``W``,
``R0``) so ``delta`` (minutes) and the MILP's power variables combine
correctly, exactly the way ``EV.p_req`` / ``EV.energy_needed`` do. Use the
``*_kwh`` properties when you want human-readable kWh.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log
from typing import TYPE_CHECKING

from config import C_RATE, HR2MIN, S_THRESH

if TYPE_CHECKING:
    from models.ev import EV
    from models.station import ChargingStation


@dataclass(frozen=True)
class VehicleData:
    """One vehicle's static data."""

    id: int
    a: float  # a_j: arrival time (minutes)
    Q: float  # Q_j: battery capacity, internal kW*min units (see module docstring)
    s_i: float  # s_i^j: initial SoC
    s_f: float  # s_f^j: target SoC
    p_max: float  # P_j^max: peak BMS acceptance (kW)

    @property
    def W(self) -> float:
        """Energy required, W_j = Q_j (s_f^j - s_i^j), internal kW*min units."""
        return self.Q * (self.s_f - self.s_i)

    @property
    def R0(self) -> float:
        """Energy missing from a full battery on arrival, R_j^0, internal kW*min units."""
        return self.Q * (1.0 - self.s_i)

    @property
    def W_kwh(self) -> float:
        """Energy required, in real kWh (for reporting)."""
        return self.W / HR2MIN

    @classmethod
    def from_ev(cls, ev: "EV") -> "VehicleData":
        """Build from a simulator ``EV`` (arrival time, SoC targets, BMS peak)."""
        return cls(
            id=ev.id,
            a=float(ev.arrival_time),
            Q=float(ev.c_b),
            s_i=float(ev.s_i),
            s_f=float(ev.s_f),
            p_max=float(ev.p_req_max),
        )


def vehicles_from_evs(evs: list["EV"]) -> list[VehicleData]:
    """Build one ``VehicleData`` per EV, preserving order."""
    if not evs:
        raise ValueError("Need at least one EV")
    return [VehicleData.from_ev(ev) for ev in evs]


def taper_time_constant(s_th: float = S_THRESH, c_rate: float = C_RATE) -> float:
    """tau = (1 - s_th) / c: shared taper time constant (minutes), same for every vehicle."""
    return (1.0 - s_th) / c_rate


def full_power_time(v: VehicleData, s_th: float) -> float:
    """
    Minutes to charge vehicle ``v`` from s_i to s_f under uncontested peak
    power (flat until s_th, then exponential taper). Used only to seed a safe
    horizon guess (``bound.default_horizon_minutes``) -- not part of the MILP
    itself.
    """
    scale = v.Q / v.p_max
    s, s_i = v.s_f, v.s_i
    if s <= s_th:
        return scale * (s - s_i)
    if s_i >= s_th:
        return scale * (-(1.0 - s_th) * log((1.0 - s) / (1.0 - s_i)))
    return scale * ((s_th - s_i) - (1.0 - s_th) * log((1.0 - s) / (1.0 - s_th)))


@dataclass(frozen=True)
class StationSpec:
    """Station layout: M piles, each with N dispensers and B modules of Delta kW."""

    n_piles: int  # M
    n_dispensers: int  # N, dispensers per pile
    n_modules: int  # B, modules per pile
    p_module: float  # Delta, kW per module

    def __post_init__(self) -> None:
        if self.n_piles <= 0:
            raise ValueError(f"n_piles must be positive, got {self.n_piles}")
        if self.n_dispensers <= 0:
            raise ValueError(f"n_dispensers must be positive, got {self.n_dispensers}")
        if self.n_modules < self.n_dispensers:
            raise ValueError(
                f"n_modules ({self.n_modules}) must be >= n_dispensers ({self.n_dispensers})"
            )
        if self.p_module <= 0:
            raise ValueError(f"p_module must be positive, got {self.p_module}")

    @classmethod
    def from_station(cls, station: "ChargingStation") -> "StationSpec":
        return cls(
            n_piles=station.n_piles,
            n_dispensers=station.n_dispensers,
            n_modules=station.n_modules,
            p_module=station.p_module,
        )
