"""
Episode statistics collected while the SimulationEngine runs.

Before each event is processed, the engine asks us to record the just-finished
interval [current_time, next_time]: system size L, queue size Q, busy time per
pile/nozzle, and energy sold. We also keep lists of arrived / finished / dropped
EVs for end-of-episode summaries.
"""

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.station import ChargingStation
    from models.ev import EV


class MetricsTracker:

    def __init__(self, n_piles: int, n_nozzles: int):
        self.n_piles = n_piles
        self.n_nozzles = n_nozzles
        self.reset()

    def reset(self):
        self.finished_evs = []
        self.dropped_evs = []
        self.arrived_evs = []

        # Time-weighted occupancy samples (one entry per inter-event interval).
        self.history_L = []
        self.history_Q = []
        self.event_times = []
        self.event_durations = []

        self.pile_active_minutes = np.zeros(self.n_piles)
        self.nozzle_active_minutes = np.zeros((self.n_piles, self.n_nozzles))

        self.pile_energy_sold = np.zeros(self.n_piles)
        self.nozzle_energy_sold = np.zeros((self.n_piles, self.n_nozzles))

    def update_finished_evs(self, ev: EV):
        self.finished_evs.append(ev)

    def update_arrived_evs(self, ev: EV):
        self.arrived_evs.append(ev)

    def update_dropped_evs(self, ev: EV):
        self.dropped_evs.append(ev)

    def update_metrics(
        self, station: ChargingStation, delta_t: float, current_time: float
    ):
        """Accumulate interval stats using each EV's already-projected deltaE_power."""
        queue = station.queue
        piles = station.piles

        L = len(queue) + sum(len(pile.evs) for pile in piles)
        Q = len(queue)

        self.history_L.append(L)
        self.history_Q.append(Q)
        self.event_times.append(current_time)
        self.event_durations.append(delta_t)

        for pile_idx, pile in enumerate(piles):
            if not pile.is_active:
                continue

            self.pile_active_minutes[pile_idx] += delta_t
            for ev in pile.evs:
                nozzle_idx = ev.nozzle_id
                self.nozzle_active_minutes[pile_idx][nozzle_idx] += delta_t
                self.nozzle_energy_sold[pile_idx][nozzle_idx] += ev.deltaE_power
                self.pile_energy_sold[pile_idx] += ev.deltaE_power
                ev.energy_received += ev.deltaE_power
                ev.energy_received_2 += ev.deltaE_SoC

    def average_L(self):
        durations = np.array(self.event_durations)
        L = np.array(self.history_L)
        if durations.sum() == 0:
            return 0
        return np.sum(L * durations) / durations.sum()

    def average_Q(self):
        durations = np.array(self.event_durations)
        Q = np.array(self.history_Q)
        if durations.sum() == 0:
            return 0
        return np.sum(Q * durations) / durations.sum()

    def pile_utilization(self, sim_time):
        return self.pile_active_minutes / sim_time

    def nozzle_utilization(self, sim_time):
        return self.nozzle_active_minutes / sim_time

    def total_energy(self):
        return np.sum(self.pile_energy_sold)
