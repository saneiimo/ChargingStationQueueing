"""
Discrete-event engine: owns the clock, event heap, and step logic.

Typical step (advance_time):
  1. Pop the next valid event at time T1.
  2. Project every charging EV over [current_time, T1] (SoC + energy).
  3. Hand that interval to MetricsTracker.
  4. Commit s_current = s_next.
  5. Process the discrete event (queue arrival, departure, charge change, end).
     Redistribution here only updates p_act and schedules new EV events from T1.
  6. Set current_time = T1.

assign_ev(pile_id, ev=None) is called between events (by the Gym env or a
heuristic) to plug a waiting EV into a pile. ``ev=None`` means head-of-line.
"""

from __future__ import annotations
from models.station import ChargingStation
from models.pile import ChargingPile
from models.ev import EV
from metrics.metrics_tracker import MetricsTracker
from .event import EventQueue, Event, EventType
import numpy as np
from config import (
    BATTERY_CAP_OPTIONS,
    SOC_I_BOUNDS,
    SOC_F_BOUNDS,
    MAX_TIME,
    CHECK_INVARIANTS,
)


class SimulationEngine:
    """Runs one episode of the charging station from t=0 to MAX_TIME."""

    def __init__(
        self, station: ChargingStation, metrics: MetricsTracker, event_heap: EventQueue
    ):

        self.station = station
        self.metrics = metrics
        self.event_heap = event_heap
        self.station.engine = self

        self.current_time = 0.0
        self.max_time = MAX_TIME
        self.terminated = False

    def reset(self, seed=None):
        """Clear state, sample all arrivals, and process the first event."""
        self.rng = np.random.default_rng(seed)

        self.current_time: float = 0.0
        self.nxt_evnt: Event | None = None
        self.nxt_ev: EV | None = None
        self.nxt_evnt_typ: EventType | None = None
        self.next_time: float | None = None
        self.terminated: bool = False
        self.departure_happened: bool = False
        self.last_drops: int = 0
        self.last_event_type: EventType | None = None

        self.station.reset()
        self.metrics.reset()
        self.event_heap.clear()
        self._generate_arrivals()

        return self.advance_time()

    def _generate_arrivals(self):
        """Pre-sample the whole arrival process into the heap, plus SIM_OVER."""
        # station.lam is mean inter-arrival time (minutes); rate = 1/lam.
        arrivals = self.rng.exponential(
            self.station.lam, int(self.max_time / self.station.lam * 5)
        ).cumsum()

        arrivals = arrivals[arrivals <= self.max_time]

        for i, t in enumerate(arrivals):
            c_b = self.rng.choice(BATTERY_CAP_OPTIONS)
            s_i = self.rng.uniform(*SOC_I_BOUNDS)
            s_f = self.rng.uniform(*SOC_F_BOUNDS)
            ev = EV(id=i, c_b=c_b, s_i=s_i, s_f=s_f, arrival_time=t)
            self.event_heap.push(Event(t, EventType.ARRIVAL, obj=ev))

        self.event_heap.push(Event(self.max_time, EventType.SIM_OVER))

    def _is_valid_event(self, event: Event) -> bool:
        event_type = event.event_type

        if event_type in (EventType.ARRIVAL, EventType.SIM_OVER):
            return True

        ev = event.obj
        event_id = event.event_id

        if ev is None or event_id is None:
            raise ValueError(
                f"This shouldn't happen because ev is None: {ev is None} or there is no event id assigned to DEPARTURE or CHARGE_CHANGE event {event_id is None}."
            )

        if event_type in (EventType.DEPARTURE, EventType.CHARGE_CHANGE):
            return event_id == ev.event_generation

        raise ValueError(
            f"Simulation logic doesn't know how to handle {event_type.name}."
        )

    def _load_next_event(self) -> None:
        """Skip stale DEPARTURE/CHARGE_CHANGE events until a live one remains."""
        while True:
            event = self.event_heap.pop()
            if not self._is_valid_event(event):
                continue

            self.nxt_evnt = event
            self.nxt_evnt_typ = event.event_type
            self.nxt_ev = event.obj
            self.next_time = event.time
            return

    def _project_evs(self, delta_t: float) -> None:
        """Continuous dynamics for [current_time, next_time] under current p_act."""
        for pile in self.station.piles:
            for ev in pile.evs:
                ev.update_s_next(delta_t)
                ev.compute_deltaE_power(delta_t)

    def _update_metrics(self):
        delta_t = self.next_time - self.current_time
        self.metrics.update_metrics(self.station, delta_t, self.current_time)

    def _update_ev_state(self):
        """Make the projected SoC the new committed SoC at the event instant."""
        for pile in self.station.piles:
            for ev in pile.evs:
                ev.s_current = ev.s_next

    def assign_ev(self, pile_id: int, ev: EV | None = None) -> bool:
        """Plug a waiting EV into pile_id and redistribute that pile's power.

        ``ev=None`` assigns head-of-line (RL / default). Heuristics pass the EV
        chosen by ``QueuePolicy.select_ev``.
        """
        pile = self.station.piles[pile_id]
        assigned = self.station.assign_ev(pile, ev=ev)
        if assigned is None:
            return False
        self._redistribute_power(pile)
        return True

    def _schedule_time(self) -> float:
        """
        Time origin for newly scheduled EV events.

        While processing an event we have already committed SoC at next_time, but
        current_time is still the previous clock value — schedule from next_time.
        Between events (after advance_time returns) the two clocks match.
        """
        if self.next_time is not None:
            return self.next_time
        return self.current_time

    def _redistribute_power(self, pile: ChargingPile, ev: EV | None = None):
        """Ask the power policy for new brick shares; schedule EV events only."""
        assignments = self.station.power_policy.update_power(pile, ev)
        schedule_time = self._schedule_time()
        for connected, power in assignments:
            connected.update_charging_power(power, schedule_time, self.event_heap)
        if CHECK_INVARIANTS:
            pile.check_invariants()

    def _process_event(self):
        next_ev = self.nxt_ev
        event_type = self.nxt_evnt_typ
        self.last_event_type = event_type
        self.last_drops = 0

        if event_type == EventType.ARRIVAL:
            self.metrics.update_arrived_evs(next_ev)
            if not self.station.add_to_queue(next_ev):
                self.metrics.update_dropped_evs(next_ev)
                self.last_drops = 1

        elif event_type == EventType.DEPARTURE:
            pile = next_ev.pile
            self.station.remove_ev(next_ev)
            self.metrics.update_finished_evs(next_ev)
            self._redistribute_power(pile=pile)
            self.departure_happened = True

        elif event_type == EventType.CHARGE_CHANGE:
            self._redistribute_power(pile=next_ev.pile, ev=next_ev)

        elif event_type == EventType.SIM_OVER:
            self.terminated = True

        else:
            raise ValueError(
                f"Simulation Logic doesn't know how to handle {event_type.name} Type"
            )

    def advance_time(self) -> bool:
        """
        One DES step. Returns True when the episode has ended (SIM_OVER).
        """
        self._load_next_event()

        delta_t = self.next_time - self.current_time
        self._project_evs(delta_t)
        self._update_metrics()
        self._update_ev_state()
        self._process_event()
        self.current_time = self.next_time

        if CHECK_INVARIANTS:
            for pile in self.station.piles:
                pile.check_invariants()

        return self.terminated

    def needs_assignment_decision(self) -> bool:
        """True when someone is waiting and at least one nozzle is free."""
        if self.terminated:
            return False
        if not self.station.queue:
            return False
        return any(not pile.is_full for pile in self.station.piles)
