"""
Electric vehicle with a nonlinear charging curve.

An EV arrives with battery size c_b, initial SoC s_i, and target SoC s_f.
While plugged into a ChargingPile it draws power p_act (capped by the pile's
brick allotment and by its own BMS request p_req).

p_req is flat up to s_th, then tapers linearly toward empty request at SoC=1.
Given p_act, we can compute:
  - how SoC grows over a time interval (update_s_next)
  - how much energy was delivered (compute_deltaE_power)
  - when the next DEPARTURE or CHARGE_CHANGE should fire (dt_next_candidate)

The SimulationEngine calls those helpers; this module does not own the clock.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import log, exp
import numpy as np
from typing import TYPE_CHECKING
from config import S_THRESH, C_RATE
from simulation.event import Event, EventQueue, EventType

if TYPE_CHECKING:
    from models.pile import ChargingPile


@dataclass
class EV:
    id: int
    c_b: float  # Battery capacity (kWh)
    s_i: float  # Initial SoC (0.0 - 1.0)
    s_f: float  # Target SoC (0.0 - 1.0)
    arrival_time: float
    s_th: float = S_THRESH
    c_rate: float = C_RATE  # C-rate: p_req_max = c_b * c_rate
    service_start_time: float = None
    pile: ChargingPile = None  # Set while plugged in; cleared on disconnect
    pile_tracker: ChargingPile = None  # Last pile used (kept after departure)
    nozzle_id: int = None  # Fixed slot index on the pile (stable if others leave)
    energy_received: float = 0  # Integrated from deltaE_power
    energy_received_2: float = 0  # Integrated from SoC deltas (sanity check)
    departure_time: float = float("inf")
    # Bump this whenever power changes so old DEPARTURE/CHARGE_CHANGE events die.
    event_generation: int = 0

    def __post_init__(self):
        self.p_req_max = self.c_b * self.c_rate  # Peak BMS request (kW)
        self.tan_B = self.p_req_max / (1 - self.s_th)  # Slope of the taper segment
        self.s_current = self.s_i
        self.t_th = self.c_b / self.p_req_max * (self.s_th - self.s_i)
        self.energy_needed = (self.s_f - self.s_i) * self.c_b
        self.s_next = self.s_current
        self.deltaE_power = 0.0
        self.p_act = 0.0

    def invalidate_pending_events(self) -> None:
        """Mark any previously scheduled EV-timed events as stale."""
        self.event_generation += 1

    def update_charging_power(
        self, power: float, schedule_time: float, event_heap: EventQueue
    ) -> None:
        """
        Apply a new allotted power (kW) and schedule the next EV event.

        schedule_time should be the instant power becomes active (usually the
        current event time). Old pending events are invalidated first.
        """
        self.p_act = min(self.p_req, power)
        self.invalidate_pending_events()

        if self.p_act <= 0:
            # Stuck until a later redistribution gives this EV some power.
            return

        event_id = self.event_generation
        t_next = self.dt_next_candidate + schedule_time
        event_heap.push(
            Event(
                t_next,
                self.event_type_next_candidate,
                event_id,
                self,
            )
        )

    @property
    def n_bricks(self) -> int | None:
        if self.pile is None or self.nozzle_id is None:
            return None
        return self.pile.ev_bricks[self.nozzle_id]

    @property
    def p_req(self) -> float:
        """BMS request at the current SoC (constant below s_th, then taper)."""
        return self.tan_B * (1 - max(self.s_th, self.s_current))

    @property
    def s_taper(self) -> float:
        """
        SoC where the constant-power segment would end under this p_act.
        If p_act == p_req_max then s_taper == s_th.
        """
        if self.p_act <= 0:
            return self.s_current
        return 1 - self.p_act / self.tan_B

    @property
    def dt_taper(self) -> float:
        """Minutes from now until we hit s_taper (0 if already in taper)."""
        if self.p_act <= 0:
            return 0.0
        return self.c_b / self.p_act * (self.s_taper - self.s_current)

    @property
    def next_state(self) -> tuple[float, EventType]:
        """
        Next SoC milestone and the event type that should fire there.

        If the pile is overloaded and this EV holds more than one brick, we
        watch for underutilization (CHARGE_CHANGE). Otherwise we aim for s_f
        (DEPARTURE).
        """
        if (
            self.pile is not None
            and self.n_bricks is not None
            and self.pile.is_overloaded
            and self.n_bricks > 1
        ):
            p_thresh = (
                self.n_bricks - 1 + self.pile.brickCheck_thresh
            ) * self.pile.p_brick
            p_thresh_2 = (self.n_bricks - 1) * self.pile.p_brick
            p_next = p_thresh if self.p_act > p_thresh else p_thresh_2
            local_next_s = 1 - p_next / self.tan_B
            if local_next_s < self.s_f:
                return (local_next_s, EventType.CHARGE_CHANGE)
            return (self.s_f, EventType.DEPARTURE)
        return (self.s_f, EventType.DEPARTURE)

    @property
    def s_next_candidate(self) -> float:
        return self.next_state[0]

    @property
    def event_type_next_candidate(self) -> EventType:
        return self.next_state[1]

    @property
    def dt_next_candidate(self) -> float:
        """Minutes until the next DEPARTURE or CHARGE_CHANGE under current p_act."""
        if self.p_act <= 0:
            return float("inf")
        linear_term = min(self.s_taper, self.s_next_candidate) - min(
            self.s_taper, self.s_current
        )
        log_term = -(1 - self.s_taper) * log(
            (1 - max(self.s_taper, self.s_next_candidate))
            / (1 - max(self.s_taper, self.s_current))
        )
        return self.c_b / self.p_act * (linear_term + log_term)

    def update_s_next(self, delta_t: float) -> None:
        """Project SoC forward by delta_t minutes into s_next (does not commit)."""
        if self.p_act <= 0 or delta_t <= 0:
            self.s_next = self.s_current
            return

        k_tmp = delta_t * self.p_act / self.c_b

        if np.isclose(self.dt_taper, 0, rtol=1e-6, atol=1e-9):
            self.s_next = 1 - (1 - self.s_current) * exp(-k_tmp / (1 - self.s_taper))
        else:
            if delta_t < self.dt_taper:
                self.s_next = k_tmp + self.s_current
            else:
                exp_term = exp(
                    (self.s_taper - self.s_current - k_tmp) / (1 - self.s_taper)
                )
                self.s_next = 1 - (1 - self.s_taper) * exp_term

    def compute_deltaE_power(self, delta_t: float) -> None:
        """Energy (kWh) delivered over delta_t under current p_act → deltaE_power."""
        if self.p_act <= 0 or delta_t <= 0:
            self.deltaE_power = 0.0
            return

        if np.isclose(self.dt_taper, 0, rtol=1e-6, atol=1e-9):
            expo_t_i = exp(-self.tan_B / self.c_b * (-self.dt_taper))
            expo_t_f = exp(-self.tan_B / self.c_b * (delta_t - self.dt_taper))
            self.deltaE_power = -self.c_b * (1 - self.s_taper) * (expo_t_f - expo_t_i)
        else:
            if delta_t < self.dt_taper:
                self.deltaE_power = self.p_act * delta_t
            else:
                lin_part = self.p_act * self.dt_taper
                expo_t_f = exp(-self.tan_B / self.c_b * (delta_t - self.dt_taper))
                expo_part = -self.c_b * (1 - self.s_taper) * (expo_t_f - 1)
                self.deltaE_power = lin_part + expo_part

    @property
    def deltaE_SoC(self) -> float:
        """Energy implied by the SoC change s_next - s_current."""
        return self.c_b * (self.s_next - self.s_current)

    @property
    def SoC_check(self) -> bool:
        return np.isclose(self.s_current, self.s_f, rtol=1e-6, atol=1e-9)

    @property
    def energy_received_check(self) -> bool:
        return np.isclose(
            self.energy_received, self.energy_needed, rtol=1e-6, atol=1e-9
        )

    @property
    def energy_received_check_2(self) -> bool:
        return np.isclose(
            self.energy_received, self.energy_received_2, rtol=1e-6, atol=1e-9
        )
