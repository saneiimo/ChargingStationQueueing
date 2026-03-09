from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
from enum import Enum
import heapq

if TYPE_CHECKING:
    from models.ev import EV


@dataclass
class Event:
    time: float
    event_type: EventType
    # Used to invalidate CHARGE_CHANGE and DEPARTURE events for EVs; when there is a power redistribution, there next events of EVs might change and we need to invalidate the previous events
    event_id: int | None = None
    obj: EV | None = None  # optional; if provided, must be EV

    def __lt__(self, other: Event) -> bool:
        return self.time < other.time


class EventType(Enum):
    ARRIVAL = 1
    DEPARTURE = 2
    CHARGE_CHANGE = 3
    SIM_OVER = 4


class EventQueue:

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
