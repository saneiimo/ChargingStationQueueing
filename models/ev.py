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
    c_rate: float = C_RATE  # C-rate: p_m = C_b * C_rate
    service_start_time: float = None
    pile: ChargingPile = None  # Reference to assigned charging pile
    pile_tracker: ChargingPile = None
    nozzle_id: int = None  # ID of the nozzle when EV is connected to the pile
    energy_received: float = 0  # Energy received (kWh)
    energy_received_2: float = 0  # Energy received (kWh)
    departure_time: float = float("inf")
    # Used to invalidate CHARGE_CHANGE and DEPARTURE events for EVs;
    # when there is a power redistribution, there next events of EVs might change and we need to invalidate the previous events
    # For a DEPARTURE event to go through we must have event.event_id == ev.departure_event_id
    # Same for a CHARGE_CHANGE event
    departure_event_id: int = 0
    charge_change_event_id: int = 0

    def __post_init__(self):
        self.p_req_max = self.c_b * self.c_rate  # Peak charging request rate (kW)
        self.tan_B = self.p_req_max / (1 - self.s_th)  # Decay tangent
        self.s_current = self.s_i  # Current SoC
        self.t_th = (
            self.c_b / self.p_req_max * (self.s_th - self.s_i)
        )  # Time corresponding to s_thresh
        self.energy_needed = (self.s_f - self.s_i) * self.c_b  # Energy required (kWh)
        self.s_next = self.s_current

    # ------

    def update_charging_power(
        self, power: float, current_time: float, event_heap: EventQueue
    ):
        # The actual power drawn from the charging pile
        self.p_act = min(self.p_req, power)

        # invalidate previous event
        if self.event_type_next_candidate == EventType.CHARGE_CHANGE:
            self.charge_change_event_id += 1
            event_id = self.charge_change_event_id
        else:
            self.departure_event_id += 1
            event_id = self.departure_event_id
        t_next = self.dt_next_candidate + current_time
        event = Event(
            t_next,
            self.event_type_next_candidate,
            event_id,
            self,
        )

        event_heap.push(event)

    # ------

    # @property
    # def current_time(self) -> float:
    #     if self.pile is None:
    #         return float("inf")
    #     return self.pile.station.current_time

    # @property
    # def next_time(self) -> float:
    #     """
    #     This is the next global event (based on other EVs, Arrivals, ...)
    #     """
    #     if self.pile is None:
    #         return float("inf")
    #     return self.pile.station.next_time

    @property
    def n_bricks(self) -> int:
        if self.pile is None:
            return None
        return self.pile.ev_bricks[self.nozzle_id]

    @property
    def p_req(self) -> float:
        """
        The BMS power request based on the current SoC
        """
        # # This is handled in the ChargingStation logic right now:
        # # Valid only if EV is plugged and charging
        # if self.pile is None or self.p_act == 0:
        #     return None
        return self.tan_B * (1 - max(self.s_th, self.s_current))

    @property
    def s_taper(self) -> float:
        """
        Computes taper SoC based on the power received by the vehicle
        if p_act = p_req_max then s_taper = s_th
        """
        return 1 - self.p_act / self.tan_B

    @property
    def dt_taper(self) -> float:
        return self.c_b / self.p_act * (self.s_taper - self.s_current)

    @property
    def next_state(self) -> tuple[float, EventType]:
        """
        Next SoC of interest: if pile is overloaded, next event is when
        the power request of EV is less than the threshold (EV is underutilizing
        a power brick)
        else it is the final SoC
        """
        if self.pile.is_overloaded and self.n_bricks > 1:
            # Power corresponding to underutiizing a power brick
            p_thresh = (
                self.n_bricks - 1 + self.pile.brickCheck_thresh
            ) * self.pile.p_brick
            # Power corresponding to freeing up a power brick
            p_thresh_2 = (self.n_bricks - 1) * self.pile.p_brick
            # If self.p_act < p_thresh, next state is freeing up that power brick
            # else it is underutilizing that brick
            p_next = p_thresh if self.p_act > p_thresh else p_thresh_2
            # SoC corresponding to when underutilization happens
            local_next_s = 1 - p_next / self.tan_B
            if local_next_s < self.s_f:
                return (local_next_s, EventType.CHARGE_CHANGE)
            else:
                # This is for the highly unlikely case that the EV target SoC
                # is before tapering.
                return (self.s_f, EventType.DEPARTURE)
        else:
            return (self.s_f, EventType.DEPARTURE)

    @property
    def s_next_candidate(self) -> float:
        return self.next_state[0]

    @property
    def event_type_next_candidate(self) -> EventType:
        return self.next_state[1]

    @property
    def dt_next_candidate(self) -> float:
        """
        Computes the time of next charging event for this EV;
        """
        linear_term = min(self.s_taper, self.s_next_candidate) - min(
            self.s_taper, self.s_current
        )
        log_term = -(1 - self.s_taper) * log(
            (1 - max(self.s_taper, self.s_next_candidate))
            / (1 - max(self.s_taper, self.s_current))
        )
        return self.c_b / self.p_act * (linear_term + log_term)

    def compute_s_next(self, delta_t: float) -> float:
        """
        Computes the next SoC, after Δt, based on current SoC and received power
        """
        k_tmp = delta_t * self.p_act / self.c_b

        # Linear decay part
        if np.isclose(self.dt_taper, 0, rtol=1e-6, atol=1e-9):
            self.s_next = 1 - (1 - self.s_current) * exp(-k_tmp / (1 - self.s_taper))

        # Starting at the constant part
        else:
            # Not going in the linear decay part of P-S curve (s_next < s_taper)
            if delta_t < self.dt_taper:
                self.s_next = k_tmp + self.s_current
            # going in the linear decay
            else:
                exp_term = exp(
                    (self.s_taper - self.s_current - k_tmp) / (1 - self.s_taper)
                )
                self.s_next = 1 - (1 - self.s_taper) * exp_term

    @property
    def compute_deltaE_power(self, delta_t) -> float:
        # Linear decay (P-S curve)
        if np.isclose(self.dt_taper, 0, rtole=1e-6, atol=1e-9):
            expo_t_i = exp(-self.tan_B / self.c_b * (-self.dt_taper))
            expo_t_f = exp(-self.tan_B / self.c_b * (delta_t - self.dt_taper))
            self.deltaE_power = -self.c_b * (1 - self.s_taper) * (expo_t_f - expo_t_i)
        # Starting from the constant part
        else:
            # Not going in the linear decay part of P-S curve
            if delta_t < self.dt_taper:
                self.deltaE_power = self.p_act * delta_t
            # Going in the linear decay part
            else:
                lin_part = self.p_act * self.dt_taper
                expo_t_f = exp(-self.tan_B / self.c_b * (delta_t - self.dt_taper))
                expo_part = -self.c_b * (1 - self.s_taper) * (expo_t_f - 1)
                self.deltaE_power = lin_part + expo_part

    @property
    def deltaE_SoC(self):
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
