from __future__ import annotations
from dataclasses import dataclass
from math import ceil
import numpy as np
from collections import deque
from config import BRICK_CHECK_THRESH
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .ev import EV
    from .station import ChargingStation
    from simulation.engine import SimulationEngine


@dataclass
class ChargingPile:
    id: int
    n_nozzles: int  # Number of nozzles for this pile
    num_bricks: int  # Max number of power bricks for this pile
    p_brick: float  # kW per power brick
    station: ChargingStation = None
    # The threshold to check when an EV is utilizing less than
    # this ratio of its last brick, considering assigning that brick to another EV where they might beter benefit from that
    brickCheck_thresh: float = BRICK_CHECK_THRESH

    def __post_init__(self):
        if self.num_bricks < self.n:
            raise ValueError(
                f"num_bricks: {self.num_bricks} is less than the number of nozzles: {self.n}"
            )
        self.power_supp: float = self.num_bricks * self.p_brick
        self.evs: List[EV] = (
            []
        )  # EVs assigned to this charging pile; each EV is assigned to a nozzle
        self.finished_EVs: List[EV] = []  # EVs finished charging and left this pile
        self.ev_bricks: List[int] = []  # Number of power bricks assigned to each EV

    def reset(self):
        # Detach any EVs still linked to this pile
        for ev in self.evs:
            if ev.pile is self:
                ev.pile = None
            if ev.nozzle_id is not None:
                ev.nozzle_id = None

        # If any finished EV still points here via tracker, clear it
        for ev in self.finished_EVs:
            if ev.pile_tracker is self:
                ev.pile_tracker = None

        # Clear pile runtime state for a new episode
        self.evs.clear()
        self.finished_EVs.clear()
        self.ev_bricks.clear()

    @property
    def is_full(self) -> bool:  # If the pile doesn't have any empty nozzles
        if len(self.evs) < self.n_nozzles:
            return False
        return True

    def connect_ev(self, ev: EV):
        if ev.pile is not None:
            raise ValueError("EV already assigned to a pile.")
        if self.is_full:
            raise ValueError("Charging Pile is full.")
        ev.pile = self  # Link back
        ev.pile_tracker = self
        ev.nozzle_id = len(self.evs)
        ev.service_start_time = self.next_time  # Log Arrival
        self.evs.append(ev)

    def disconnect_ev(self, ev: EV):
        if ev.pile is None:
            raise ValueError("EV is not assigned to any pile.")
        if ev.pile != self:
            raise ValueError("EV is not assigned to this pile.")
        idx = self.evs.index(ev)
        self.evs.pop(idx)  # Remove EV
        ev.pile = None
        ev.nozzle_id = None
        ev.departure_time = self.next_time  # Log Departure

    #     self._update_power(engine)

    # def _update_power(self, engine: SimulationEngine):
    #     current_time = engine.current_time
    #     assignments = self.station.power_policy.update_power(self)
    #     for ev, power in assignments:
    #         ev.update_charging_power(power, current_time, engine.event_heap)

    # --------------------------------------------------

    @property
    def current_time(self) -> float:
        if self.station is None:
            return float("inf")
        return self.station.current_time

    @property
    def next_time(self) -> float:
        """
        This is the next global event (based on other EVs, Arrivals, ...)
        """
        if self.station is None:
            return float("inf")
        return self.station.next_time

    @property
    def is_active(self) -> bool:  # If any EV is connected to this pile
        if len(self.evs) > 0:
            return True
        return False

    @property
    def is_overloaded(
        self,
    ) -> bool:  # If power req is more than demand (in terms of power bricks)
        if sum(ceil(ev.p_req / self.p_brick) for ev in self.evs) > self.num_bricks:
            return True
        return False

    @property
    def power_reqs(self):
        # Keeps track of total power request of each EV put to the charging pile
        return [ev.p_req for ev in self.evs]

    @property
    def remaining_reqs(self):
        # Keep tracks of unfulfilled power request of each ev
        return [ev.p_req - ev.n_bricks * self.p_brick for ev in self.evs]
