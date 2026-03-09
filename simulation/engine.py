from models.station import ChargingStation
from models.pile import ChargingPile
from models.ev import EV
from metrics.metrics_tracker import MetricsTracker
from .event import EventQueue, Event, EventType
import numpy as np
from config import BATTERY_CAP_OPTIONS, SOC_I_BOUNDS, SOC_F_BOUNDS, MAX_TIME


class SimulationEngine:

    def __init__(
        self, station: ChargingStation, metrics: MetricsTracker, event_heap: EventQueue
    ):

        self.station = station
        self.metrics = metrics
        self.event_heap = event_heap

        self.current_time = 0
        self.max_time = MAX_TIME
        self.terminated = False

    def reset(self, seed=None):

        self.rng = np.random.default_rng(seed)

        self.current_time: float = 0
        self.next_event: Event = None
        self.next_ev: EV = None
        self.next_event_type: EventType = None
        self.next_event_id: int = None
        self.next_time: float = None
        self.terminated: bool = False
        self.departure_happened: bool = False

        self.station.reset()
        self.metrics.reset()

        self.event_heap.clear()

        self._generate_arrivals()

        return self.advance_time()

    def _generate_arrivals(self):
        """Generates all EV arrivals for the simulation duration
        and stores them as min-heap."""
        arrivals = self.rng.exponential(
            self.station.lam, int(self.max_time / self.station.lam * 1.5)
        ).cumsum()

        arrivals = arrivals[arrivals <= self.max_time]

        for i, t in enumerate(arrivals):

            c_b = self.rng.choice(*BATTERY_CAP_OPTIONS)
            s_i = self.rng.uniform(*SOC_I_BOUNDS)
            s_f = self.rng.uniform(*SOC_F_BOUNDS)
            ev = EV(id=i, c_b=c_b, s_i=s_i, s_f=s_f, arrival_time=t)
            self.event_heap.push(Event(t, EventType.ARRIVAL, obj=ev))

        # Push the simulation end time to the heap
        self.event_heap.push(Event(self.max_time, EventType.SIM_OVER))

    def assign_ev(self, pile_id: int):

        pile = self.station.piles[pile_id]

        ev = self.station.assign_ev(pile)

        if ev is None:
            return False

        self._redistribute_power(pile)

        return True

    def _redistribute_power(
        self, pile: ChargingPile, ev: EV | None = None, kind: str | None = None
    ):
        assignments = self.station.power_policy.update_power(pile, ev, kind)
        for each_ev, power in assignments:
            each_ev.update_charging_power(power, self.current_time, self.event_heap)

    def _update_metrics(self):
        delta_t = self.next_time - self.current_time
        self.metrics.update_metrics(self.station, delta_t, self.current_time)
        pass

    # --------------------------------------------------

    def _update_state(self):
        """
        Update the state of EVs and ChargingPiles
        """
        # ---- process event ----
        next_ev = self.next_ev
        event_type = self.next_event_type
        event_id = self.next_event_id
        if event_type == EventType.ARRIVAL:
            self.metrics.update_arrived_evs(next_ev)
            if not self.station.add_to_queue(next_ev):
                self.metrics.update_dropped_evs(next_ev)

        elif event_type == EventType.DEPARTURE:
            # If this is an invalid event (later overwritten by another event)
            if event_id != next_ev.departure_event_id:
                return False
            self.station.remove_ev(next_ev)
            self.metrics.update_finished_evs(next_ev)
            self._redistribute_power(pile=next_ev.pile)
            self.departure_happened = True

        elif event_type == EventType.CHARGE_CHANGE:
            # If this is an invalid event (later overwritten by another event)
            if event_id != next_ev.charge_change_event_id:
                return False
            self._redistribute_power(pile=next_ev.pile, ev=next_ev)

        elif event_type == EventType.SIM_OVER:

            self.terminated = True

        return True

    def _load_next_event(self):
        """
        Loads the next event by popping the event_heap
        """
        self.next_event = self.event_heap.pop()
        self.next_ev = self.next_event.obj
        self.next_event_type = self.next_event.event_type
        self.next_event_id = self.next_event.event_id
        self.next_time = self.next_event.time

    # --------------------------------------------------

    def advance_time(self):

        # Used to control if we are processing a valid event
        # By checking next_event_id with the latest event_id of the EV
        # We are processing
        is_valid_event = False

        # Keep advancing time, until we find a valid event.
        while not is_valid_event:

            self._load_next_event()

            # ---- update metrics ----
            self._update_metrics()

            # ---- process event ----
            is_valid_event = self._update_state()

            # ---- advance clock ----
            self.current_time = self.next_time

        return self.terminated
