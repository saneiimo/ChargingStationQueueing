"""
Timed events for the discrete-event simulator.

The engine keeps a min-heap of Event objects. ARRIVAL, SIM_OVER, and
WARMUP_END are always kept. DEPARTURE and CHARGE_CHANGE carry an event_id
that must match the EV's current event_generation; otherwise they are stale
(power was redistributed and a newer event replaced them) and get skipped.

CHARGE_CHANGE is scheduled only when the station's power policy sets
``supports_underuse_reallocation`` (Proportional yes, Static no).

WARMUP_END fires once, at t = engine.warmup_period, only when a warm-up
period was requested (warmup_period > 0). It carries no ``obj``/``event_id``
(like SIM_OVER) -- it is not tied to any one EV. See
``SimulationEngine._handle_warmup_end`` for what it does.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from enum import Enum
import heapq

if TYPE_CHECKING:
    from models.ev import EV


@dataclass
class Event:
    """One scheduled instant in the simulation."""

    time: float
    event_type: EventType
    # Matches EV.event_generation for DEPARTURE / CHARGE_CHANGE; unused otherwise.
    event_id: int | None = None
    # The EV this event belongs to (arrivals, departures, charge changes).
    obj: EV | None = None

    def __lt__(self, other: Event) -> bool:
        # heapq needs a total order; earliest time wins.
        return self.time < other.time


class EventType(Enum):
    ARRIVAL = 1  # EV shows up and tries to join the station queue
    DEPARTURE = 2  # EV reaches target SoC and leaves its connector
    # Opt-in via PowerPolicy.supports_underuse_reallocation (Prop yes, Static no).
    CHARGE_CHANGE = 3  # EV underuses a module; policy may free/redistribute it
    SIM_OVER = 4  # Hard stop at engine.max_time (= warmup_period + measurement_horizon)
    WARMUP_END = 5  # Boundary between the warm-up phase and the measured phase


class EventQueue:
    """Thin wrapper around heapq so the engine can push/pop/clear events."""

    def __init__(self):
        self.heap = []

    def push(self, event: Event):
        heapq.heappush(self.heap, event)

    def pop(self):
        return heapq.heappop(self.heap)

    def peek(self):
        return self.heap[0]

    def empty(self):
        return len(self.heap) == 0

    def clear(self):
        self.heap.clear()
