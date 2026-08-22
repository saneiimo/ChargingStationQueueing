"""
Electric vehicle with a nonlinear charging curve.

An EV arrives with battery size c_b, initial SoC s_i, and target SoC s_f.
While plugged into a ChargingPile it draws power p_act (capped by the pile's
module allotment and by its own BMS request p_req).

p_req is flat up to s_th, then tapers linearly toward empty request at SoC=1.
Given p_act, we can compute:
  - how SoC grows over a time interval (update_s_next)
  - how much energy was delivered (compute_deltaE_power)
  - when the next DEPARTURE or CHARGE_CHANGE should fire (dt_next_candidate).
    CHARGE_CHANGE is scheduled only if the station power policy sets
    ``supports_underuse_reallocation`` (see ``policy.power.base``).

SoC projection is piecewise (constant power to s_taper, then expo). After other
DES events split an interval, s_current may already be past s_taper; those
helpers continue expo from the current SoC so departure still lands on s_f.

The SimulationEngine calls those helpers; this module does not own the clock.
"""

from __future__ import annotations
from dataclasses import dataclass
from math import log, exp
import numpy as np
from typing import TYPE_CHECKING
from config import S_THRESH, C_RATE, HR2MIN
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
    service_start_time: float | None = None
    pile: ChargingPile | None = None  # Set while plugged in; cleared on disconnect
    pile_tracker: ChargingPile | None = None  # Last pile used (kept after departure)
    dispenser_id: int | None = (
        None  # Fixed slot index on the pile (cleared on disconnect)
    )
    dispenser_id_tracker: int | None = (
        None  # Same slot, kept after departure (for plots)
    )
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
        # Target energy in kWh (c_b is stored in kW·min).
        self.energy_needed = (self.s_f - self.s_i) * self.c_b / HR2MIN
        self.s_next = self.s_current
        self.deltaE_power = 0.0
        self.p_act = 0.0
        # Sparse log for visualization: (t, SoC, p_req, p_act, p_allot) at
        # each redistribution and at departure. Samples are instantaneous
        # (p_act <= p_req); interval energy uses the pre-event p_act in the
        # engine, not these rows. See visualization/pile_power.py.
        self.charge_trace: list[tuple[float, float, float, float, float]] = []

    def invalidate_pending_events(self) -> None:
        """Mark any previously scheduled EV-timed events as stale."""
        self.event_generation += 1

    def record_charge_sample(self, t: float, p_allot: float) -> None:
        """
        Append one instantaneous (time, SoC, BMS request, actual power, allotment)
        sample for plots / post-run inspection.

        Caller must set ``p_act`` consistently with ``p_req`` at this SoC before
        calling (``update_charging_power`` does; departure re-caps in
        ``ChargingPile.disconnect_ev``). This log is not used for energy metrics.
        """
        self.charge_trace.append(
            (
                float(t),
                float(self.s_current),
                float(self.p_req),
                float(self.p_act),
                float(p_allot),
            )
        )

    def update_charging_power(
        self, power: float, schedule_time: float, event_heap: EventQueue
    ) -> None:
        """
        Apply a new allotted power (kW) and schedule the next EV event.

        schedule_time should be the instant power becomes active (usually the
        current event time). Old pending events are invalidated first.
        """
        self.p_act = min(self.p_req, power)
        self.record_charge_sample(schedule_time, p_allot=power)
        self.invalidate_pending_events()

        if np.isclose(self.p_act, 0, rtol=1e-6, atol=1e-9):
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
    def n_modules(self) -> int | None:
        if self.pile is None or self.dispenser_id is None:
            return None
        return self.pile.ev_modules[self.dispenser_id]

    @property
    def p_req(self) -> float:
        """BMS request at the current SoC (constant below s_th, then taper)."""
        return self.tan_B * (1 - max(self.s_th, self.s_current))

    @property
    def s_taper(self) -> float:
        """
        SoC knee where constant-power charging at this ``p_act`` would end.

        Defined by ``p_act = tan_B * (1 - s_taper)``. If ``p_act == p_req_max``
        then ``s_taper == s_th``. Independent of ``s_current``; after several DES
        steps it is common that ``s_current > s_taper`` (already in expo region).
        """
        if np.isclose(self.p_act, 0, rtol=1e-6, atol=1e-9):
            return self.s_current
        return 1 - self.p_act / self.tan_B

    @property
    def dt_taper(self) -> float:
        """
        Minutes until ``s_taper`` under current ``p_act``.

        Returns 0 if already at or past the knee (``s_current >= s_taper``),
        including the usual case after intermediate DES projections in taper.
        Never returns a negative value.
        """
        if np.isclose(self.p_act, 0, rtol=1e-6, atol=1e-9):
            return 0.0
        gap = self.s_taper - self.s_current
        if gap <= 0:
            return 0.0
        return self.c_b / self.p_act * gap

    def _power_policy_supports_underuse_reallocation(self) -> bool:
        """
        True only if the station's power policy opts into CHARGE_CHANGE.

        Underuse events are for policies that free a module on underutilization
        (e.g. Proportional). Fixed-split policies (e.g. Static) leave the flag
        False so we never schedule CHARGE_CHANGE. Missing station/policy is
        treated as False (safe for unit tests that build a pile alone).
        """
        if self.pile is None or self.pile.station is None:
            return False
        policy = getattr(self.pile.station, "power_policy", None)
        if policy is None:
            return False
        return bool(getattr(policy, "supports_underuse_reallocation", False))

    @property
    def next_state(self) -> tuple[float, EventType]:
        """
        Next SoC milestone and the event type that should fire there.

        If the power policy supports underuse reallocation, the pile is
        overloaded, and this EV holds more than one module, we watch for
        underutilization (CHARGE_CHANGE). Otherwise we aim for s_f (DEPARTURE).
        """
        if (
            self.pile is not None
            and self.n_modules is not None
            and self._power_policy_supports_underuse_reallocation()
            and self.pile.is_overloaded
            and self.n_modules > 1
        ):
            p_thresh = (
                self.n_modules - 1 + self.pile.moduleCheck_thresh
            ) * self.pile.p_module
            p_thresh_2 = (self.n_modules - 1) * self.pile.p_module
            if self.p_act > p_thresh and not np.isclose(
                self.p_act, p_thresh, rtol=1e-6, atol=1e-9
            ):
                p_next = p_thresh
            else:
                p_next = p_thresh_2

            local_next_s = 1 - p_next / self.tan_B

            if local_next_s < self.s_f and not np.isclose(
                local_next_s, self.s_f, rtol=1e-6, atol=1e-9
            ):
                return (local_next_s, EventType.CHARGE_CHANGE)
            return (self.s_f, EventType.DEPARTURE)

            # If the target coincides with s_current, the last redistribution
            # handed the freed module straight back to this EV (still the
            # largest remaining request) and nothing actually changed.
            # Scheduling another CHARGE_CHANGE here would refire at the same
            # instant forever, so fall back to watching for DEPARTURE instead.
            # no_progress = local_next_s <= self.s_current or np.isclose(
            #     local_next_s, self.s_current, rtol=1e-6, atol=1e-9
            # )

            # if (
            #     not no_progress
            #     and local_next_s < self.s_f
            #     and not np.isclose(local_next_s, self.s_f, rtol=1e-6, atol=1e-9)
            # ):
            #   return (local_next_s, EventType.CHARGE_CHANGE)
            # return (self.s_f, EventType.DEPARTURE)
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
        if np.isclose(self.p_act, 0, rtol=1e-6, atol=1e-9):
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
        """
        Project SoC forward by ``delta_t`` minutes into ``s_next`` (no commit).

        Under fixed ``p_act``: constant power until ``s_taper``, then exponential
        taper. If ``s_current`` is already past ``s_taper`` (typical after other
        DES events split the interval), continue with expo-from-**current** SoC.
        The closed form that starts the expo at ``s_taper`` is only valid when
        ``s_current <= s_taper``; using it after the knee undershoots SoC and
        made finished EVs leave below ``s_f``.
        """
        if np.isclose(self.p_act, 0, rtol=1e-6, atol=1e-9) or np.isclose(
            delta_t, 0, rtol=1e-6, atol=1e-9
        ):
            self.s_next = self.s_current
            return

        k_tmp = delta_t * self.p_act / self.c_b
        s_tap = self.s_taper
        one_m_tap = 1.0 - s_tap

        # Already in expo region (or numerically at the knee).
        if self.s_current >= s_tap - 1e-9:
            self.s_next = 1.0 - (1.0 - self.s_current) * exp(-k_tmp / one_m_tap)

        # In the linear region
        else:
            # Compute time to taper
            dt_tap = self.c_b / self.p_act * (s_tap - self.s_current)
            # Still in the linear section after delta_t
            if delta_t <= dt_tap:
                self.s_next = self.s_current + k_tmp
            else:
                # Linear to s_taper, then expo for the remainder (from the knee).
                k_expo = k_tmp - (s_tap - self.s_current)
                self.s_next = 1.0 - one_m_tap * exp(-k_expo / one_m_tap)

    def compute_deltaE_power(self, delta_t: float) -> None:
        """
        Energy (kWh) delivered over ``delta_t`` under current ``p_act``.

        Matches ``update_s_next``: linear while below ``s_taper``, expo after.
        When already past the knee, integrate expo from ``s_current`` (not from
        ``s_taper``). ``c_b`` is stored in kW·min, so divide by ``HR2MIN`` for kWh.
        """
        if self.p_act <= 0 or delta_t <= 0:
            self.deltaE_power = 0.0
            return

        s_tap = self.s_taper
        rate = self.tan_B / self.c_b  # == p_act / (c_b * (1 - s_tap))

        # Already in expo region (or at the knee).
        if self.s_current >= s_tap - 1e-12:
            expo = exp(-rate * delta_t)
            self.deltaE_power = (
                -self.c_b * (1.0 - self.s_current) * (expo - 1.0) / HR2MIN
            )
            return

        dt_tap = self.c_b / self.p_act * (s_tap - self.s_current)
        if delta_t <= dt_tap:
            self.deltaE_power = self.p_act * delta_t / HR2MIN
        else:
            lin_part = self.p_act * dt_tap
            expo = exp(-rate * (delta_t - dt_tap))
            expo_part = -self.c_b * (1.0 - s_tap) * (expo - 1.0)
            self.deltaE_power = (lin_part + expo_part) / HR2MIN

    @property
    def deltaE_SoC(self) -> float:
        """Energy implied by the SoC change s_next - s_current.
        Note that as C_b is converted to kWmin, we need to
        do a unit conversion using HR2MIN"""
        return self.c_b * (self.s_next - self.s_current) / HR2MIN

    @property
    def SoC_check(self) -> bool:
        """True if committed SoC matches the departure target ``s_f``."""
        return np.isclose(self.s_current, self.s_f, rtol=1e-6, atol=1e-9)

    @property
    def energy_received_check(self) -> bool:
        """True if accrued energy (kWh) matches ``energy_needed`` (also kWh)."""
        return np.isclose(
            self.energy_received, self.energy_needed, rtol=1e-6, atol=1e-9
        )

    @property
    def energy_received_check_2(self) -> bool:
        """True if power-integral and SoC-delta energy tallies agree (both kWh)."""
        return np.isclose(
            self.energy_received, self.energy_received_2, rtol=1e-6, atol=1e-9
        )
