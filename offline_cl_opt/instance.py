"""
Per-vehicle / station data for the connector-lane offline MILP (see
``model.py``), matching the notation of ``connector_lane_model.html``
Section 3.

Unit convention
---------------
Real-world units throughout, as the source document specifies (Section 3.2):
power in kW, energy in kWh, time in minutes for inputs, and hours
(``h = delta / 60``) wherever time meets power/energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.ev import EV
    from models.station import ChargingStation


@dataclass(frozen=True)
class VehicleData:
    """One vehicle's static data, Section 3.4."""

    id: int
    a: float  # a_j: arrival time (minutes)
    Q: float  # Q_j: battery capacity (kWh)
    s_i: float  # s_i^j: initial SoC
    s_f: float  # s_f^j: target SoC
    s_th: float  # s_th^j: SoC where the BMS acceptance limit starts to fall
    p_max: float  # P_j^max: peak BMS acceptance (kW)

    def __post_init__(self) -> None:
        if self.Q <= 0:
            raise ValueError(f"Q must be positive, got {self.Q}")
        if not (0.0 <= self.s_i < self.s_f <= 1.0):
            raise ValueError(
                f"need 0 <= s_i < s_f <= 1, got s_i={self.s_i}, s_f={self.s_f}"
            )
        if not (0.0 <= self.s_th < 1.0):
            raise ValueError(f"need 0 <= s_th < 1, got s_th={self.s_th}")
        if self.p_max <= 0:
            raise ValueError(f"p_max must be positive, got {self.p_max}")

    @property
    def W(self) -> float:
        """Energy requested, W_j = Q_j (s_f^j - s_i^j), kWh."""
        return self.Q * (self.s_f - self.s_i)

    @property
    def R(self) -> float:
        """Energy missing from a full battery on arrival, R_j = Q_j (1 - s_i^j), kWh."""
        return self.Q * (1.0 - self.s_i)

    @property
    def tau_hours(self) -> float:
        """Taper time constant, tau_j = Q_j (1 - s_th^j) / P_j^max, hours (Section 4.2)."""
        return self.Q * (1.0 - self.s_th) / self.p_max

    def tau_delta_hours(self, delta_minutes: float) -> float:
        """
        Effective discrete time constant (Section 4.2, "Discretisation of the
        taper"): ``tau^delta_j = h / (1 - exp(-h/tau_j))``, ``h = delta/60``.

        Matches the continuous exponential taper exactly at every slot
        boundary whenever the taper binds throughout the slot -- tighter
        (always >= tau_j, and >= h) than using tau_j directly, which would
        understate how fast the battery actually decays and let the model
        claim power it couldn't really accept by a slot's end (the taper
        cap is set from the energy at the slot's start, while the
        instantaneous acceptance keeps falling through the slot).
        """
        h = delta_minutes / 60.0
        tau = self.tau_hours
        return h / (1.0 - exp(-h / tau))

    @classmethod
    def from_ev(cls, ev: "EV") -> "VehicleData":
        """Build from a simulator ``EV`` (converts out of its internal
        kW*min battery-capacity convention into real kWh)."""
        from config import HR2MIN

        return cls(
            id=ev.id,
            a=float(ev.arrival_time),
            Q=float(ev.c_b) / HR2MIN,
            s_i=float(ev.s_i),
            s_f=float(ev.s_f),
            s_th=float(ev.s_th),
            p_max=float(ev.p_req_max),
        )


def vehicles_from_evs(evs: list["EV"]) -> list[VehicleData]:
    """Build one ``VehicleData`` per EV, preserving order."""
    if not evs:
        raise ValueError("Need at least one EV")
    return [VehicleData.from_ev(ev) for ev in evs]


@dataclass(frozen=True)
class StationSpec:
    """Station layout, Section 3.3: M piles, each with C connectors and a
    shared pool of N power modules of Delta kW."""

    n_piles: int  # M
    n_connectors: int  # C, connectors per pile
    n_modules: int  # N, modules per pile
    p_module: float  # Delta, kW per module

    def __post_init__(self) -> None:
        if self.n_piles <= 0:
            raise ValueError(f"n_piles must be positive, got {self.n_piles}")
        if self.n_connectors <= 0:
            raise ValueError(f"n_connectors must be positive, got {self.n_connectors}")
        if self.n_modules < self.n_connectors:
            raise ValueError(
                f"n_modules ({self.n_modules}) must be >= n_connectors ({self.n_connectors})"
            )
        if self.p_module <= 0:
            raise ValueError(f"p_module must be positive, got {self.p_module}")

    @classmethod
    def from_station(cls, station: "ChargingStation") -> "StationSpec":
        """Mirrors ``station.n_connectors`` here, named to match this
        formulation's own notation."""
        return cls(
            n_piles=station.n_piles,
            n_connectors=station.n_connectors,
            n_modules=station.n_modules,
            p_module=station.p_module,
        )
