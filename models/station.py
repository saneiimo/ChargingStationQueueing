"""
Charging station: a finite waiting queue plus a list of piles.

This is the "yard" the SimulationEngine drives. Arrivals are offered to the
queue (or dropped if full). Assignment removes a chosen waiting EV (head-of-line
by default) and plugs it into a chosen pile. The station does not decide
*which* EV or pile — that comes from a queue policy or the RL agent via the
engine's ``assign_ev(pile_id, ev=...)``.

Clock properties (current_time / next_time) are delegated to the engine so
piles and EVs can stamp service start / departure without importing the engine
directly in hot paths.
"""

from __future__ import annotations
from collections import deque
from .pile import ChargingPile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.ev import EV
    from policy.power.base import PowerPolicy
    from simulation.engine import SimulationEngine


class ChargingStation:
    """Owns the queue, piles, arrival gap, and the active power policy."""

    def __init__(
        self,
        n_piles: int,
        n_nozzles: int,
        n_bricks: int,
        p_brick: float,
        queue_capacity: int,
        power_policy: PowerPolicy,
        mean_interarrival: float,
    ):

        self.n_piles = n_piles
        self.n_nozzles = n_nozzles
        self.n_bricks = n_bricks
        self.p_brick = p_brick
        self.queue_capacity = queue_capacity
        self.power_policy = power_policy
        self.piles = [
            ChargingPile(i, n_nozzles, n_bricks, p_brick, self) for i in range(n_piles)
        ]
        # Mean inter-arrival time (minutes). Arrival rate λ = 1 / mean_interarrival.
        # Count in an interval of length t is Poisson(λ t); gaps are Exp(mean).
        if mean_interarrival <= 0:
            raise ValueError(
                f"mean_interarrival must be positive, got {mean_interarrival}"
            )
        self.mean_interarrival = float(mean_interarrival)
        self.queue = deque()
        # Wired by SimulationEngine.__init__.
        self.engine: SimulationEngine | None = None

    @property
    def arrival_rate(self) -> float:
        """Arrival rate λ (customers per minute) = 1 / mean_interarrival."""
        return 1.0 / self.mean_interarrival

    # --------------------------------------------------
    # Clock (delegates to SimulationEngine)
    # --------------------------------------------------

    @property
    def current_time(self) -> float:
        if self.engine is None:
            return float("inf")
        return self.engine.current_time

    @property
    def next_time(self) -> float:
        if self.engine is None:
            return float("inf")
        return self.engine.next_time

    # --------------------------------------------------
    # Reset / queue / assignment
    # --------------------------------------------------

    def reset(self):
        self.queue.clear()
        for pile in self.piles:
            pile.reset()

    def add_to_queue(self, ev: EV) -> bool:
        """Return False if the queue is full (arrival is dropped)."""
        if len(self.queue) >= self.queue_capacity:
            return False
        self.queue.append(ev)
        return True

    def pop_from_queue(self):
        if not self.queue:
            return None
        return self.queue.popleft()

    def assign_ev(self, pile: ChargingPile, ev: EV | None = None):
        """
        Move a waiting EV onto ``pile``.

        If ``ev`` is None, assigns the head-of-line customer. Otherwise removes
        that EV from the queue (must currently be waiting) and connects it.
        Returns the EV, or None if the move is illegal.
        """
        if not self.queue:
            return None
        if pile.is_full:
            return None

        if ev is None:
            ev = self.pop_from_queue()
        else:
            if ev not in self.queue:
                raise ValueError(f"EV {ev.id} is not in the waiting queue")
            self.queue.remove(ev)

        pile.connect_ev(ev)
        return ev

    def remove_ev(self, ev: EV):
        """Unplug an EV that has finished (or is otherwise leaving)."""
        pile = ev.pile
        pile.disconnect_ev(ev)

    # --------------------------------------------------
    # Snapshots used by metrics / heuristics
    # --------------------------------------------------

    def vehicles_in_system(self):
        charging = sum(len(pile.evs) for pile in self.piles)
        return charging + len(self.queue)

    def vehicles_charging(self):
        return sum(len(pile.evs) for pile in self.piles)

    def queue_length(self):
        return len(self.queue)

    def available_piles(self):
        return [pile for pile in self.piles if not pile.is_full]

    def is_queue_full(self):
        return len(self.queue) >= self.queue_capacity

    def is_idle(self):
        return self.vehicles_charging() == 0 and len(self.queue) == 0
