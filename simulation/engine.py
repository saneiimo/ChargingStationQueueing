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

Warm-up period
--------------
``warmup_period`` (minutes, default 0.0) extends the episode: the DES runs
``warmup_period`` minutes *before* what would otherwise be t=0 of the
measured window, then continues for the usual ``max_time`` (which keeps its
original meaning -- the length of the *measured* phase, now renamed
internally to ``measurement_horizon`` for clarity -- see that attribute).
``self.max_time`` itself becomes the *total* horizon
(``warmup_period + measurement_horizon``), since that is what every
existing internal consumer (the SIM_OVER sentinel, arrival generation,
``snap_arrivals``) actually needs; with the default ``warmup_period=0.0``
this is identical to ``measurement_horizon``, i.e. behaviour is unchanged
unless a warm-up period is requested.

Implemented as one more sentinel event (``EventType.WARMUP_END``, mirroring
``SIM_OVER``), scheduled at ``t=warmup_period`` when that is positive. See
``_handle_warmup_end`` for exactly what fires at that instant: it always
snapshots who was queued / mid-service into ``self.metrics`` (see
``MetricsTracker``'s own module docstring), and additionally empties the
live queue if ``flush_queue_at_warmup`` is set. Vehicles already being
served are *never* touched by the flush, only the queue.

The arrival process itself needs no warm-up-specific handling: it is a
single continuous, memoryless Poisson stream sampled once over
``[0, max_time]`` (the *total* horizon) at ``reset()``, exactly as before --
splitting it into "warm-up" and "measured" arrivals is purely a reporting
concern (``MetricsTracker``'s ``since`` parameter), not a generation one.
"""

from __future__ import annotations
from typing import Sequence
from models.station import ChargingStation
from models.pile import ChargingPile
from models.ev import EV
from metrics.metrics_tracker import InServiceAtBoundary, MetricsTracker
from .event import EventQueue, Event, EventType
from .arrivals import generate_arrivals, clone_arrivals, snap_arrivals, validate_delta_arr
import numpy as np
from config import MAX_TIME, WARMUP_PERIOD, FLUSH_QUEUE_AT_WARMUP, CHECK_INVARIANTS


class SimulationEngine:
    """Runs one episode of the charging station from t=0 to max_time."""

    def __init__(
        self,
        station: ChargingStation,
        metrics: MetricsTracker,
        event_heap: EventQueue,
        max_time: float | None = None,
        battery_cap_options: Sequence[float] | None = None,
        delta_arr: float | None = None,
        warmup_period: float | None = None,
        flush_queue_at_warmup: bool | None = None,
    ):
        """
        Parameters
        ----------
        max_time :
            Length of the *measured* phase (minutes) -- i.e. everything
            after any warm-up period. Defaults to ``config.MAX_TIME``. See
            the module docstring, "Warm-up period", for how this combines
            with ``warmup_period`` into ``self.max_time`` (the *total*
            horizon / actual SIM_OVER instant).
        battery_cap_options :
            Battery capacities (kW*min, same units as
            ``config.BATTERY_CAP_OPTIONS``) sampled from when generating
            arrivals internally (``station.mean_interarrival`` set). Defaults
            to ``config.BATTERY_CAP_OPTIONS``. Unused when arrivals are
            supplied externally (see ``set_arrivals``).
        delta_arr :
            Arrival-time grid in minutes. ``None`` leaves continuous Poisson
            (or externally supplied) times unchanged. A positive ``d`` snaps
            every arrival to the nearest multiple of ``d`` (see
            ``simulation.arrivals.snap_arrival_time``). Applied both to
            internally sampled streams and to lists installed via
            ``set_arrivals``.
        warmup_period :
            Minutes to run *before* the measured phase begins. ``None``
            (the default) uses ``config.WARMUP_PERIOD`` -- ``0.0`` out of
            the box, i.e. no warm-up, fully backward compatible -- the same
            "``None`` means read the config default" convention
            ``max_time`` already uses for ``config.MAX_TIME``. Pass an
            explicit value (``0.0`` included) to override the config
            default for one run without touching ``config.py``. Must be
            ``>= 0``. See the module docstring, "Warm-up period".
        flush_queue_at_warmup :
            If True, empty the live queue at the instant ``t=warmup_period``
            (only the queue -- EVs already plugged in are left alone).
            ``None`` (the default) uses ``config.FLUSH_QUEUE_AT_WARMUP``
            (``False`` out of the box) -- same "``None`` reads the config
            default" convention as ``warmup_period``/``max_time``. Flushed
            EVs are recorded in ``metrics.flushed_evs`` and excluded from
            every reported statistic, but remain in ``metrics.arrived_evs``
            (historical fact: they did arrive) -- see ``MetricsTracker``'s
            module docstring. No effect when ``warmup_period`` is ``0``
            (there is no boundary to flush at).
        """

        self.station = station
        self.metrics = metrics
        self.event_heap = event_heap
        self.station.engine = self

        self.current_time = 0.0
        resolved_warmup_period = (
            float(warmup_period) if warmup_period is not None else WARMUP_PERIOD
        )
        if resolved_warmup_period < 0:
            raise ValueError(
                f"warmup_period must be >= 0, got {resolved_warmup_period} "
                f"(passed {warmup_period!r}; config.WARMUP_PERIOD={WARMUP_PERIOD})"
            )
        self.warmup_period = resolved_warmup_period
        self.flush_queue_at_warmup = (
            bool(flush_queue_at_warmup)
            if flush_queue_at_warmup is not None
            else FLUSH_QUEUE_AT_WARMUP
        )
        # measurement_horizon: what `max_time` meant before warm-up existed
        # (length of the measured phase). max_time: reused as the *total*
        # episode horizon -- see the module docstring for why.
        self.measurement_horizon = float(max_time) if max_time is not None else MAX_TIME
        self.max_time = self.warmup_period + self.measurement_horizon
        self.battery_cap_options = battery_cap_options
        self.delta_arr = validate_delta_arr(delta_arr)
        self._external_arrivals: list[EV] | None = None
        self.terminated = False

    def set_arrivals(self, arrivals: list[EV] | None) -> None:
        """
        Install a fixed external arrival list for future ``reset`` calls.

        Only used while ``station.mean_interarrival`` is None -- when it is
        set, the engine samples its own Poisson process on every ``reset``
        instead (mean_interarrival takes priority over an installed list).
        Pass ``None`` to clear a previously installed list.
        """
        self._external_arrivals = list(arrivals) if arrivals is not None else None

    def reset(self, seed=None, arrivals: list[EV] | None = None):
        """Clear state, sample/install all arrivals, and process the first event.

        ``arrivals``, if given, is installed via ``set_arrivals`` before this
        reset (and remains installed for subsequent resets too).
        """
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

        if arrivals is not None:
            self.set_arrivals(arrivals)

        self.station.reset()
        self.metrics.reset()
        # Known immediately (0.0 if no warm-up), regardless of whether/when
        # WARMUP_END later fires -- see MetricsTracker's module docstring.
        self.metrics.warmup_period = self.warmup_period
        self.event_heap.clear()
        self._generate_arrivals()

        return self.advance_time()

    def _generate_arrivals(self):
        """Populate the event heap with ARRIVAL events, plus SIM_OVER (and
        WARMUP_END when a warm-up period was requested).

        ``station.mean_interarrival`` takes priority: if set, arrivals are
        sampled fresh (any installed external list is ignored). Otherwise an
        external list must have been installed via ``set_arrivals`` /
        ``reset(arrivals=...)``. ``delta_arr`` snaps times onto a minute
        grid when set (see ``simulation.arrivals``). ``self.max_time`` is
        the *total* horizon (warm-up included, see the module docstring),
        so the single continuous Poisson stream sampled below already spans
        the warm-up phase too -- no special-casing needed there.
        """
        if self.station.mean_interarrival is not None:
            evs = generate_arrivals(
                mean_interarrival=self.station.mean_interarrival,
                max_time=self.max_time,
                rng=self.rng,
                battery_cap_options=self.battery_cap_options,
                delta_arr=self.delta_arr,
            )
        elif self._external_arrivals is not None:
            # Clone so a list reused across several resets always starts each
            # EV fresh, the way freshly-sampled arrivals do.
            evs = clone_arrivals(self._external_arrivals)
        else:
            raise ValueError(
                "No arrival process: set station.mean_interarrival, or "
                "install an external list via engine.set_arrivals(...) / "
                "reset(arrivals=...)."
            )

        # External lists are snapped here; internally sampled lists are
        # already on the grid (idempotent if delta_arr is set twice).
        evs = snap_arrivals(evs, self.delta_arr, max_time=self.max_time)

        for ev in evs:
            self.event_heap.push(Event(ev.arrival_time, EventType.ARRIVAL, obj=ev))

        if self.warmup_period > 0:
            self.event_heap.push(Event(self.warmup_period, EventType.WARMUP_END))
        self.event_heap.push(Event(self.max_time, EventType.SIM_OVER))

    def _is_valid_event(self, event: Event) -> bool:
        event_type = event.event_type

        if event_type in (EventType.ARRIVAL, EventType.SIM_OVER, EventType.WARMUP_END):
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
        """Ask the power policy for new module shares; schedule EV events only."""
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

        elif event_type == EventType.WARMUP_END:
            self._handle_warmup_end()

        else:
            raise ValueError(
                f"Simulation Logic doesn't know how to handle {event_type.name} Type"
            )

    def _snapshot_in_service(self) -> list[InServiceAtBoundary]:
        """
        Per-connector snapshot of every EV currently plugged in, at the
        warm-up/measurement boundary.

        Called only from ``_handle_warmup_end``, at the instant
        ``current_time`` has just been committed to ``warmup_period`` (see
        ``advance_time``: ``_update_ev_state`` runs before ``_process_event``
        on every step, so ``ev.s_current`` here is already the state exactly
        at the boundary, not one step stale). See ``InServiceAtBoundary``'s
        own docstring for the frozen-fields-vs-live-``ev``-reference split
        (it needs both: a frozen copy of the state *at* the boundary, and a
        live reference for whatever this vehicle does *after* it).
        """
        snapshot: list[InServiceAtBoundary] = []
        for pile in self.station.piles:
            for ev in pile.evs:
                snapshot.append(
                    InServiceAtBoundary(
                        ev_id=ev.id,
                        pile_id=pile.id,
                        connector_id=ev.connector_id,
                        n_modules=pile.ev_modules[ev.connector_id],
                        p_act=ev.p_act,
                        c_b=ev.c_b,
                        s_i=ev.s_i,
                        s_f=ev.s_f,
                        s_th=ev.s_th,
                        c_rate=ev.c_rate,
                        s_current=ev.s_current,
                        arrival_time=ev.arrival_time,
                        service_start_time=ev.service_start_time,
                        boundary_time=self.warmup_period,
                        ev=ev,
                    )
                )
        return snapshot

    def _handle_warmup_end(self) -> None:
        """
        Fires exactly once, at t = warmup_period (only when warmup_period >
        0 -- see ``_generate_arrivals``). See the module docstring, "Warm-up
        period", and ``MetricsTracker``'s own module docstring for the full
        picture.

        Always records who was queued / mid-service at this instant
        (regardless of ``flush_queue_at_warmup``) -- this is the boundary
        condition a future optimization run over just the measured window
        would need, and/or the post-warm-up-only reporting cohort. Only
        *removes* the queued EVs from the live queue -- and only the queue,
        never in-service EVs -- when ``flush_queue_at_warmup`` is set.
        """
        self.metrics.record_warmup_snapshot(
            queued=list(self.station.queue),
            in_service=self._snapshot_in_service(),
        )
        if self.flush_queue_at_warmup:
            flushed = self.station.clear_queue()
            for ev in flushed:
                self.metrics.update_flushed_evs(ev)

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
        """True when someone is waiting and at least one connector is free."""
        if self.terminated:
            return False
        if not self.station.queue:
            return False
        return any(not pile.is_full for pile in self.station.piles)
