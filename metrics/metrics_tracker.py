"""
Episode statistics for the charging-station DES.

``MetricsTracker`` is the single place for simulation metrics:

* **Online** (called by ``SimulationEngine`` each inter-event interval):
  time-average system size L and queue size Q, connector/pile busy time,
  energy sold, and lists of arrived / finished / dropped EVs.

* **Offline** (derived after the run from finished-EV timestamps):
  waits, service times, sojourns, mean / max W_q, mean W / S, total and
  average energy (station and per pile), pile utilization, Little's law.
  Callers should use these methods instead of re-deriving the same
  quantities in notebooks or tests.

After an episode, ``metrics.validate`` can assert invariants and optionally
print the queueing-law summary via ``report_queueing_laws`` (which delegates
here).
"""

from __future__ import annotations

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.station import ChargingStation
    from models.ev import EV


class MetricsError(ValueError):
    """Raised when finished-EV timestamps are inconsistent or missing."""


class MetricsTracker:
    """Collects online interval stats and exposes offline customer-level metrics."""

    def __init__(self, n_piles: int, n_connectors: int):
        self.n_piles = n_piles
        self.n_connectors = n_connectors
        self.reset()

    def reset(self) -> None:
        """Clear all episode accumulators (called on env / engine reset)."""
        self.finished_evs: list[EV] = []
        self.dropped_evs: list[EV] = []
        self.arrived_evs: list[EV] = []

        # Time-weighted occupancy samples (one entry per inter-event interval).
        self.history_L: list[int] = []
        self.history_Q: list[int] = []
        self.event_times: list[float] = []
        self.event_durations: list[float] = []

        self.pile_active_minutes = np.zeros(self.n_piles)
        self.connector_active_minutes = np.zeros((self.n_piles, self.n_connectors))

        self.pile_energy_sold = np.zeros(self.n_piles)
        self.connector_energy_sold = np.zeros((self.n_piles, self.n_connectors))

    # ------------------------------------------------------------------
    # Online updates (DES)
    # ------------------------------------------------------------------

    def update_finished_evs(self, ev: EV) -> None:
        self.finished_evs.append(ev)

    def update_arrived_evs(self, ev: EV) -> None:
        self.arrived_evs.append(ev)

    def update_dropped_evs(self, ev: EV) -> None:
        self.dropped_evs.append(ev)

    def update_metrics(
        self, station: ChargingStation, delta_t: float, current_time: float
    ) -> None:
        """Accumulate interval stats using each EV's already-projected deltaE_power."""
        queue = station.queue
        piles = station.piles
        # Len system
        L = len(queue) + sum(len(pile.evs) for pile in piles)
        # Len queue
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
                connector_idx = ev.connector_id
                self.connector_active_minutes[pile_idx][connector_idx] += delta_t
                self.connector_energy_sold[pile_idx][connector_idx] += ev.deltaE_power
                self.pile_energy_sold[pile_idx] += ev.deltaE_power
                ev.energy_received += ev.deltaE_power
                ev.energy_received_2 += ev.deltaE_SoC

    # ------------------------------------------------------------------
    # Online-derived time averages
    # ------------------------------------------------------------------

    def average_L(self) -> float:
        """Time-average number of EVs in system (queue + charging)."""
        durations = np.array(self.event_durations)
        L = np.array(self.history_L)
        if durations.sum() == 0:
            return 0.0
        return float(np.sum(L * durations) / durations.sum())

    def average_Q(self) -> float:
        """Time-average queue length (waiting only)."""
        durations = np.array(self.event_durations)
        Q = np.array(self.history_Q)
        if durations.sum() == 0:
            return 0.0
        return float(np.sum(Q * durations) / durations.sum())

    def pile_utilization(self, sim_time: float) -> np.ndarray:
        """Fraction of time each pile had at least one EV plugged in."""
        if sim_time <= 0:
            return np.zeros(self.n_piles)
        return self.pile_active_minutes / sim_time

    def connector_utilization(self, sim_time: float) -> np.ndarray:
        """Fraction of time each connector was occupied (shape n_piles x n_connectors)."""
        if sim_time <= 0:
            return np.zeros((self.n_piles, self.n_connectors))
        return self.connector_active_minutes / sim_time

    def mean_connector_utilization(self, sim_time: float) -> float:
        """Mean connector busy fraction across all connectors."""
        util = self.connector_utilization(sim_time)
        return float(np.mean(util)) if util.size else 0.0

    def total_energy(self) -> float:
        """Total energy sold (kWh) at the station (sum over piles)."""
        return float(np.sum(self.pile_energy_sold))

    def pile_energy(self) -> np.ndarray:
        """Total energy sold (kWh) by each pile."""
        return self.pile_energy_sold.copy()

    def average_energy(self) -> float:
        """Mean energy delivered (kWh) among finished EVs (0 if none)."""
        if not self.finished_evs:
            return 0.0
        return float(np.mean([ev.energy_received for ev in self.finished_evs]))

    def average_pile_energy(self) -> np.ndarray:
        """
        Mean energy delivered (kWh) per finished EV that used each pile.

        Piles with no finished EVs return 0. Uses ``pile_tracker`` kept after
        departure.
        """
        buckets: list[list[float]] = [[] for _ in range(self.n_piles)]
        for ev in self.finished_evs:
            pile = ev.pile_tracker
            if pile is None:
                continue
            buckets[pile.id].append(ev.energy_received)

        out = np.zeros(self.n_piles)
        for i, vals in enumerate(buckets):
            if vals:
                out[i] = float(np.mean(vals))
        return out

    # ------------------------------------------------------------------
    # Offline: finished-EV time samples
    # ------------------------------------------------------------------

    def finished_time_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-finished-EV wait, service, and sojourn times (minutes).

        Definitions
        -----------
        * wait W_q = service_start_time - arrival_time
        * service S = departure_time - service_start_time
        * sojourn W = departure_time - arrival_time

        Raises
        ------
        MetricsError
            If any finished EV is missing ``service_start_time`` or has a
            non-finite ``departure_time`` (logical bug: finished implies served).
        """
        waits: list[float] = []
        services: list[float] = []
        sojourns: list[float] = []

        for ev in self.finished_evs:
            if ev.service_start_time is None or not np.isfinite(ev.departure_time):
                raise MetricsError(
                    f"EV {ev.id} is finished (served) but "
                    f"service_start_time={ev.service_start_time}, "
                    f"departure_time={ev.departure_time}. Check!"
                )
            waits.append(ev.service_start_time - ev.arrival_time)
            services.append(ev.departure_time - ev.service_start_time)
            sojourns.append(ev.departure_time - ev.arrival_time)

        return (
            np.asarray(waits, dtype=float),
            np.asarray(services, dtype=float),
            np.asarray(sojourns, dtype=float),
        )

    def mean_wait(self) -> float:
        """Mean queue wait W_q among finished EVs (0 if none)."""
        waits, _, _ = self.finished_time_arrays()
        return float(np.mean(waits)) if waits.size else 0.0

    def max_wait(self) -> float:
        """Maximum queue wait W_q among finished EVs (0 if none)."""
        waits, _, _ = self.finished_time_arrays()
        return float(np.max(waits)) if waits.size else 0.0

    def mean_service(self) -> float:
        """
        Mean service / charge time S among finished EVs (0 if none).

        S = departure_time - service_start_time (minutes plugged in).
        Replication tables report this as ``avg charge time``.
        """
        _, services, _ = self.finished_time_arrays()
        return float(np.mean(services)) if services.size else 0.0

    def mean_sojourn(self) -> float:
        """Mean sojourn W among finished EVs (0 if none)."""
        _, _, sojourns = self.finished_time_arrays()
        return float(np.mean(sojourns)) if sojourns.size else 0.0

    def service_squared_cv(self) -> float:
        """
        Squared coefficient of variation of service times, c_s^2 = Var(S)/E[S]^2.

        Uses sample variance (ddof=1). Returns NaN if fewer than two finished EVs
        or E[S] = 0.
        """
        _, services, _ = self.finished_time_arrays()
        if services.size < 2:
            return float("nan")
        S = float(np.mean(services))
        if S <= 0:
            return float("nan")
        return float(np.var(services, ddof=1) / (S**2))

    # ------------------------------------------------------------------
    # Offline: Little's law + utilization summary
    # ------------------------------------------------------------------

    def queueing_summary(
        self,
        sim_time: float,
        *,
        n_servers: int | None = None,
    ) -> dict[str, float]:
        """
        Little's law (system and queue) and utilization for the episode.

        Uses effective throughput ``lambda_eff = (# finished) / T`` and
        customer averages from ``finished_time_arrays``. Simulated L and Q
        come from the online time averages; theory sides are
        ``lambda_eff * W`` and ``lambda_eff * W_q``.

        Parameters
        ----------
        sim_time :
            Episode length T (usually ``engine.current_time``).
        n_servers :
            Number of parallel servers for rho theory. Defaults to
            ``n_piles * n_connectors``.

        Returns
        -------
        dict
            ``T``, ``n_finished``, ``lambda_eff``, ``W``, ``W_q``, ``W_q_max``,
            ``S``, ``mu``, ``c``, ``L_sim``, ``L_theory``, ``Q_sim``,
            ``Q_theory``, ``rho_sim``, ``rho_theory``, ``E_total``, ``E_avg``,
            ``c_s2``.

            Per-pile energy / utilization arrays are available via
            ``pile_energy``, ``average_pile_energy``, and ``pile_utilization``.
        """
        T = float(sim_time) if sim_time > 0 else 1.0
        n_finished = len(self.finished_evs)
        lambda_eff = n_finished / T

        waits, services, sojourns = self.finished_time_arrays()
        W = float(np.mean(sojourns)) if sojourns.size else 0.0
        W_q = float(np.mean(waits)) if waits.size else 0.0
        W_q_max = float(np.max(waits)) if waits.size else 0.0
        S = float(np.mean(services)) if services.size else 0.0
        mu = (1.0 / S) if S > 0 else 0.0
        c_s2 = self.service_squared_cv()

        c = int(n_servers) if n_servers is not None else self.n_piles * self.n_connectors
        L_sim = self.average_L()
        Q_sim = self.average_Q()
        L_theory = lambda_eff * W
        Q_theory = lambda_eff * W_q

        rho_sim = self.mean_connector_utilization(T)
        rho_theory = (lambda_eff / (c * mu)) if (c > 0 and mu > 0) else 0.0

        return {
            "T": T,
            "n_finished": float(n_finished),
            "lambda_eff": lambda_eff,
            "W": W,
            "W_q": W_q,
            "W_q_max": W_q_max,
            "S": S,
            "mu": mu,
            "c": float(c),
            "L_sim": L_sim,
            "L_theory": L_theory,
            "Q_sim": Q_sim,
            "Q_theory": Q_theory,
            "rho_sim": rho_sim,
            "rho_theory": rho_theory,
            "E_total": self.total_energy(),
            "E_avg": self.average_energy(),
            "c_s2": float(c_s2) if np.isfinite(c_s2) else float("nan"),
        }
