from __future__ import annotations
from collections import deque
from .pile import ChargingPile
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ev import EV
    from policy.power.base import PowerPolicy
    from simulation.engine_2 import SimulationEngine


class ChargingStation:

    def __init__(
        self,
        n_piles: int,
        n_nozzles: int,
        n_bricks: int,
        p_brick: float,
        queue_capacity: int,
        power_policy: PowerPolicy,
        lam: float,
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
        self.lam = lam  # Arrival rate (mean minutes); Number of arrivals in any interval of length t is Poisson(t / lambda)
        # and each inter-arrival time is Exp(1 / lambda)
        self.queue = deque()

    # --------------------------------------------------
    # Reset station state
    # --------------------------------------------------

    def reset(self):

        self.queue.clear()
        self.dropped_cars.clear()

        for pile in self.piles:
            pile.reset()

    # --------------------------------------------------
    # Queue management
    # --------------------------------------------------

    def add_to_queue(self, ev: EV):
        # Queue full
        if len(self.queue) >= self.queue_capacity:
            return False
        # Append to queue
        self.queue.append(ev)
        return True

    def pop_from_queue(self):

        if not self.queue:
            return None

        return self.queue.popleft()

    # --------------------------------------------------
    # Assignment logic
    # --------------------------------------------------

    def assign_ev(self, pile: ChargingPile):

        # Penalty for trying to assign from empty queue
        if not self.queue:
            return None

        # Penalty for trying to assign to a full pile
        if pile.is_full:
            return None

        ev = self.pop_from_queue()
        # Connect ev to the target pile
        pile.connect_ev(ev)

        return ev

    def remove_ev(self, ev: EV):
        pile = ev.pile
        pile.disconnect_ev(ev)

    # --------------------------------------------------
    # System information helpers
    # --------------------------------------------------

    def vehicles_in_system(self):

        charging = sum(len(pile.evs) for pile in self.piles)
        return charging + len(self.queue)

    def vehicles_charging(self):

        return sum(len(pile.evs) for pile in self.piles)

    def queue_length(self):

        return len(self.queue)

    def available_piles(self):

        return [pile for pile in self.piles if not pile.is_full()]

    def is_queue_full(self):

        return len(self.queue) >= self.queue_capacity

    def is_idle(self):

        return self.vehicles_charging() == 0 and len(self.queue) == 0
