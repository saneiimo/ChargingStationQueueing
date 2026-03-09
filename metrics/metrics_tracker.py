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

        # ---- system level metrics ----
        self.finished_evs = []
        self.dropped_evs = []
        self.arrived_evs = []

        # time-weighted metrics
        self.history_L = []  # vehicles in system
        self.history_Q = []  # vehicles in queue
        self.event_times = []  # event timestamps
        self.event_durations = []  # delta_t between events

        # ---- utilization metrics ----
        self.pile_active_minutes = np.zeros(self.n_piles)
        self.nozzle_active_minutes = np.zeros((self.n_piles, self.n_nozzles))

        # ---- energy metrics ----
        self.pile_energy_sold = np.zeros(self.n_piles)
        self.nozzle_energy_sold = np.zeros((self.n_piles, self.n_nozzles))

    # --------------------------------------------------
    # Core update function called by SimulationEngine
    # --------------------------------------------------
    def update_finished_evs(self, ev: EV):
        self.finished_evs.append(ev)

    def update_arrived_evs(self, ev: EV):
        self.arrived_evs.append(ev)

    def update_dropped_evs(self, ev: EV):
        self.dropped_evs.append(ev)

    def update_metrics(
        self, station: ChargingStation, delta_t: float, current_time: float
    ):

        queue = station.queue
        piles = station.piles

        # ---- system occupancy ----

        L = len(queue) + sum(len(pile.evs) for pile in piles)
        Q = len(queue)

        self.history_L.append(L)
        self.history_Q.append(Q)
        self.event_times.append(current_time)
        self.event_durations.append(delta_t)

        # ---- utilization tracking ----

        for pile_idx, pile in enumerate(piles):

            if pile.is_active:

                self.pile_active_minutes[pile_idx] += delta_t
                # Iterate over EVs and their 'nozzle' index
                for nozzle_idx, ev in enumerate(pile.evs):
                    # Nozzle tracking
                    self.nozzle_active_minutes[pile_idx][nozzle_idx] += delta_t
                    self.nozzle_energy_sold[pile_idx][nozzle_idx] += ev.deltaE_power
                    self.pile_energy_sold[pile_idx] += ev.deltaE_power
                    # EV Tracking
                    ev.energy_received += ev.deltaE_power
                    ev.energy_received_2 += ev.deltaE_SoC

    # --------------------------------------------------
    # Convenience statistics
    # --------------------------------------------------

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
