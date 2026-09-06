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

Warm-up period
--------------
``SimulationEngine`` can be given a ``warmup_period > 0`` (minutes), in
which case the DES runs an extra ``warmup_period`` minutes *before* t=0 of
what would otherwise be the episode, then continues for the usual
``max_time``. At the instant the boundary is crossed (t = warmup_period),
the engine calls ``record_warmup_snapshot`` once, which fills in:

* ``warmup_period`` -- the boundary time (also set at ``reset()``, from
  ``engine.warmup_period``, so it is defined even before/without the
  boundary event firing -- 0.0 means "no warm-up").
* ``queued_at_warmup_end`` -- plain ``EV`` references for whoever was
  waiting in the queue at that instant. Safe to store as live references:
  a queued EV's state never changes while it waits, so nothing here can
  drift out of date.
* ``in_service_at_warmup_end`` -- ``InServiceAtBoundary`` snapshots (see
  that class) for whoever was already plugged in and charging. Each entry
  carries both a frozen copy of that EV's state *at* the boundary
  (pile/connector/module allotment, ``s_current``, etc. -- the boundary
  condition if a future optimization run takes over this vehicle's power)
  and a live reference for whatever it does *after* the boundary
  (``actual_departure_time``, ``post_boundary_trace()`` -- the boundary
  condition if that vehicle is instead left exactly as the causal
  simulation actually charged it). See that class's own docstring for why
  it needs both.

A third list, ``arrived_post_warmup`` (a property, not separately stored --
see its own docstring for why), completes the picture: together with
``queued_at_warmup_end`` it is the full set of vehicles a re-optimization
confined to the measurement window would need to consider (whoever was
already waiting at the boundary, plus whoever arrived after it) --
``in_service_at_warmup_end`` is the third, already-mid-charge case.

If ``SimulationEngine.flush_queue_at_warmup`` is also set, every EV in
``queued_at_warmup_end`` is additionally removed from the live queue and
appended to ``flushed_evs`` -- a bucket kept distinct from both
``finished_evs`` and ``dropped_evs`` so it can be excluded from every
summary statistic while still preserving a historical record (it stays in
``arrived_evs`` too -- they really did arrive) and keeping
``metrics.validate.validate_episode``'s conservation check sound:
``arrived = finished + dropped + flushed + queued + plugged``.

Every "offline" method below accepts an optional ``since`` cutoff (default
``0.0``, i.e. the whole run) -- passing ``since=metrics.warmup_period``
restricts it to the post-warm-up window, via one of two conventions
depending on what kind of statistic it is:

* **Per-customer** statistics (``finished_time_arrays`` and everything
  built on it -- waits, services, sojourns, ``n_finished``, and per-EV
  energy averages ``average_energy``/``average_pile_energy``) filter the
  *cohort*: only finished EVs with ``arrival_time >= since`` count (the
  standard DES warm-up convention -- exclude customers who were already in
  the system during the transient, even if they happen to finish after
  it).
* **Station-level, time-weighted** statistics (``average_L``/``average_Q``,
  ``pile_utilization``/``connector_utilization``/
  ``mean_connector_utilization``, ``total_energy``/``pile_energy``) are
  re-integrated over only the ``[since, T]`` portion of the per-interval
  logs ``update_metrics`` keeps (an interval straddling ``since`` is
  apportioned by the fraction of it at/after ``since``, not wholly included
  or excluded -- exact whenever ``since`` lines up with an actual logged
  event, which it always does for ``since=metrics.warmup_period`` since
  ``WARMUP_END`` is itself such an event).

``queueing_summary`` (and ``metrics.validate.report_queueing_laws``'s
``post_warmup_only`` flag) combines both conventions in one call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.station import ChargingStation
    from models.ev import EV


class MetricsError(ValueError):
    """Raised when finished-EV timestamps are inconsistent or missing."""


@dataclass(frozen=True)
class InServiceAtBoundary:
    """
    Snapshot of one EV already plugged in and charging at the warm-up/
    measurement boundary (t = warmup_period), carrying enough information
    to define either kind of boundary condition a future optimization run
    might need for it -- the decision between them is deliberately not
    made here (see ``SimulationEngine._snapshot_in_service``'s caller):

    * "Optimizer takes over" -- this vehicle stays plugged into its
      current pile/connector (unplugging it mid-charge is unrealistic),
      but its *future* power is now a decision variable. Needs only the
      frozen fields below: which pile/connector/module allotment it
      already occupies, and its state *at* the boundary (``s_current``,
      i.e. how much energy it already has and how much it still needs to
      reach ``s_f``).
    * "Left as-is" -- the optimizer has no control over this vehicle; it
      keeps charging exactly as the causal simulation actually charged it,
      and departs whenever it actually departed. Needs ``actual_
      departure_time``/``post_boundary_trace()`` below -- necessarily a
      *live* read (see their own docstrings), since that trajectory is
      still in the future at the instant this snapshot is taken.

    Frozen fields (plain copies taken at the boundary instant -- unlike
    ``queued_at_warmup_end``, safe as live ``EV`` references there, an
    in-service EV's ``s_current``/``p_act`` keep evolving afterward, so a
    live read would silently drift out of date for *these* quantities):

    ``c_b`` is in the EV's own native unit (kW*min, i.e. kWh * HR2MIN --
    see ``models.ev.EV`` / ``config.HR2MIN``), matching every other place
    in this codebase that reads ``ev.c_b`` directly.

    Not consumed anywhere yet -- nothing currently reads this list besides
    ``MetricsTracker`` storing it.
    """

    ev_id: int
    pile_id: int
    connector_id: int
    n_modules: int  # module allotment on that connector, at the boundary
    p_act: float  # actual charging power (kW) at the boundary
    c_b: float  # battery capacity, kW*min (EV's native unit)
    s_i: float  # original arrival SoC
    s_f: float  # target SoC
    s_th: float  # BMS taper knee
    c_rate: float  # C-rate (p_req_max = c_b * c_rate)
    s_current: float  # SoC AT the boundary -- the state to resume from
    arrival_time: float  # this EV's real arrival time (may predate warm-up)
    service_start_time: float  # when plugged in (may predate warm-up)
    boundary_time: float  # = warmup_period; duplicated here so this object
    # is self-contained (no need to also carry metrics.warmup_period around)

    # LIVE reference, deliberately not copied -- see actual_departure_time
    # and post_boundary_trace() below. Everything ABOVE this field is a
    # frozen copy and will never change; this one field's target keeps
    # mutating for as long as the simulation keeps running.
    ev: "EV"

    @property
    def actual_departure_time(self) -> float:
        """
        ``ev.departure_time``, read live. ``inf`` until this vehicle
        actually departs -- e.g. still ``inf`` if read before the episode
        ends, or if the vehicle never finishes within the horizon. This is
        the "left as-is" boundary condition's departure time: whatever the
        causal simulation actually produced, not a decision variable.
        """
        return self.ev.departure_time

    def post_boundary_trace(
        self,
    ) -> list[tuple[float, float, float, float, float]]:
        """
        This EV's ``charge_trace`` rows at/after the boundary -- ``(t,
        SoC, p_req, p_act, p_allot)`` instantaneous samples (see
        ``models.ev.EV.record_charge_sample``), a LIVE read of
        ``ev.charge_trace`` as of whenever this is called. Typically call
        this after the episode ends, when the trace is complete through
        ``actual_departure_time``; if the vehicle hasn't departed yet this
        only returns whatever has been sampled so far (a partial
        trajectory, not the full one).

        This is the "left as-is" boundary condition's power profile: what
        this vehicle actually drew, sample by sample, if the optimizer is
        never given control over it. Note the samples are *instantaneous*
        (event-triggered), not a dense per-slot series -- turning this
        into whatever time discretization an optimizer expects is left to
        that optimizer's own preprocessing, not done here.
        """
        return [row for row in self.ev.charge_trace if row[0] >= self.boundary_time]


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
        # See the module docstring, "Warm-up period". flushed_evs holds EVs
        # discarded from the queue at the warm-up boundary (only nonempty
        # when SimulationEngine.flush_queue_at_warmup is set); they also
        # remain in arrived_evs (historical fact: they did arrive) but are
        # excluded from every summary statistic below. warmup_period is set
        # here to 0.0 (i.e. "no warm-up") and overwritten by the engine
        # immediately on reset() from its own warmup_period, so it is always
        # defined even before/without the WARMUP_END event firing.
        self.flushed_evs: list[EV] = []
        self.warmup_period: float = 0.0
        self.queued_at_warmup_end: list[EV] = []
        self.in_service_at_warmup_end: list[InServiceAtBoundary] = []

        # Time-weighted occupancy samples (one entry per inter-event interval).
        self.history_L: list[int] = []
        self.history_Q: list[int] = []
        self.event_times: list[float] = []
        self.event_durations: list[float] = []

        # Per-interval increments (NOT running totals -- see
        # pile_active_minutes/connector_active_minutes/pile_energy_sold/
        # connector_energy_sold below for those) backing the windowed
        # (`since`-aware) utilization/energy methods, the same way
        # history_L/history_Q back windowed average_L/average_Q. One
        # array per interval, parallel to event_times/event_durations.
        self.history_pile_active_dt: list[np.ndarray] = []
        self.history_connector_active_dt: list[np.ndarray] = []
        self.history_pile_energy: list[np.ndarray] = []
        self.history_connector_energy: list[np.ndarray] = []

        # Running (whole-run) totals -- unchanged from before the warm-up
        # feature existed; still the fast path for since=0.0.
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

    def update_flushed_evs(self, ev: EV) -> None:
        """Record one EV discarded from the queue by the warm-up flush.
        The EV also stays in ``arrived_evs`` -- this is an additional
        bucket, not a replacement -- see the module docstring."""
        self.flushed_evs.append(ev)

    def record_warmup_snapshot(
        self, *, queued: list[EV], in_service: list[InServiceAtBoundary]
    ) -> None:
        """
        Called exactly once by ``SimulationEngine._handle_warmup_end`` (only
        when ``warmup_period > 0``), right at the warm-up/measurement
        boundary. Stores who was queued / mid-service at that instant -- see
        the module docstring, "Warm-up period", for why these need
        different storage (plain references vs. frozen snapshots).
        """
        self.queued_at_warmup_end = list(queued)
        self.in_service_at_warmup_end = list(in_service)

    @property
    def arrived_post_warmup(self) -> list[EV]:
        """
        Vehicles that arrived strictly during the measurement window
        (``arrival_time >= warmup_period``), i.e. after
        ``queued_at_warmup_end``/``in_service_at_warmup_end`` were captured.

        Unlike those two (a point-in-time snapshot that cannot be
        reconstructed later -- once an EV leaves the live queue, it's
        gone), this needs no separate bookkeeping: ``arrived_evs`` already
        has every arrival with its own ``arrival_time``, so this is just
        computed on access, always trivially correct and never at risk of
        drifting out of sync with the log it's derived from. Includes EVs
        that were later dropped (queue full) -- same convention as
        ``arrived_evs`` itself, which is a superset later partitioned into
        finished/dropped/still-active, not pre-filtered.

        Together with ``queued_at_warmup_end``, this is the full set of
        vehicles a re-optimization confined to the measurement window would
        need to consider: whoever was already waiting at the boundary, plus
        whoever arrived after it. (``in_service_at_warmup_end`` covers the
        third case -- vehicles already mid-charge at the boundary -- see
        that class's own docstring.)
        """
        return [ev for ev in self.arrived_evs if ev.arrival_time >= self.warmup_period]

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

        # This interval's own contribution (not the running total) to each
        # per-pile/connector quantity -- logged below so pile_utilization/
        # connector_utilization/total_energy/etc. can be windowed by
        # `since`, the same way history_L/history_Q back average_L/average_Q.
        pile_active_dt = np.zeros(self.n_piles)
        connector_active_dt = np.zeros((self.n_piles, self.n_connectors))
        pile_energy = np.zeros(self.n_piles)
        connector_energy = np.zeros((self.n_piles, self.n_connectors))

        for pile_idx, pile in enumerate(piles):
            if not pile.is_active:
                continue

            pile_active_dt[pile_idx] = delta_t
            for ev in pile.evs:
                connector_idx = ev.connector_id
                connector_active_dt[pile_idx][connector_idx] = delta_t
                connector_energy[pile_idx][connector_idx] += ev.deltaE_power
                pile_energy[pile_idx] += ev.deltaE_power
                ev.energy_received += ev.deltaE_power
                ev.energy_received_2 += ev.deltaE_SoC

        # Running (whole-run) totals -- unchanged arithmetic from before the
        # warm-up feature existed.
        self.pile_active_minutes += pile_active_dt
        self.connector_active_minutes += connector_active_dt
        self.pile_energy_sold += pile_energy
        self.connector_energy_sold += connector_energy

        # Per-interval logs for windowing (see the note above).
        self.history_pile_active_dt.append(pile_active_dt)
        self.history_connector_active_dt.append(connector_active_dt)
        self.history_pile_energy.append(pile_energy)
        self.history_connector_energy.append(connector_energy)

    # ------------------------------------------------------------------
    # Online-derived time averages
    # ------------------------------------------------------------------

    def _windowed_time_average(self, values: list[int], since: float) -> float:
        """
        Time-weighted average of ``values`` (one entry per logged
        inter-event interval, see ``update_metrics``), restricted to the
        portion of each interval at or after ``since``.

        An interval straddling ``since`` is partially counted (only its
        post-``since`` slice), not wholly included or excluded -- exact
        re-integration over ``[since, T]``, not a coarse per-interval
        filter, since ``update_metrics`` logs one sample per interval
        regardless of that interval's length.
        """
        if since <= 0:
            durations = np.array(self.event_durations)
            vals = np.array(values)
            total = durations.sum()
            if total == 0:
                return 0.0
            return float(np.sum(vals * durations) / total)

        weighted_sum = 0.0
        total_weight = 0.0
        for t_end, dt, v in zip(self.event_times, self.event_durations, values):
            eff_start = max(t_end - dt, since)
            eff_dt = t_end - eff_start
            if eff_dt <= 0:
                continue
            weighted_sum += v * eff_dt
            total_weight += eff_dt
        if total_weight == 0:
            return 0.0
        return weighted_sum / total_weight

    def average_L(self, since: float = 0.0) -> float:
        """Time-average number of EVs in system (queue + charging).

        ``since``: restrict the average to ``[since, T]`` -- see the module
        docstring, "Warm-up period" (e.g. ``since=metrics.warmup_period``
        for the post-warm-up-only figure). Default ``0.0`` is the whole run,
        matching the original (pre-warm-up-feature) behaviour exactly."""
        return self._windowed_time_average(self.history_L, since)

    def average_Q(self, since: float = 0.0) -> float:
        """Time-average queue length (waiting only). See ``average_L`` for
        the ``since`` windowing convention."""
        return self._windowed_time_average(self.history_Q, since)

    def _windowed_sum(self, values: list[np.ndarray], shape: tuple[int, ...], since: float) -> np.ndarray:
        """
        Sum of ``values`` (one already ``delta_t``-scaled array per logged
        interval -- e.g. active-minutes or energy *contributed in that
        interval*, not a running total -- see ``update_metrics``),
        restricted to ``[since, T]``.

        An interval straddling ``since`` has its contribution apportioned
        by the fraction of that interval at/after ``since`` (linear in
        time -- exact whenever ``since`` coincides with an actual logged
        event boundary, which it always does for
        ``since=metrics.warmup_period``: ``WARMUP_END`` is itself a logged
        event, so a fresh interval starts exactly there, and no interval
        straddles it. Only an approximation -- assuming a roughly uniform
        rate within the straddled interval -- for an arbitrary ``since``
        that does not line up with a real event).
        """
        if since <= 0:
            total = np.zeros(shape)
            for v in values:
                total = total + v
            return total

        total = np.zeros(shape)
        for t_end, dt, v in zip(self.event_times, self.event_durations, values):
            if dt <= 0:
                continue
            eff_start = max(t_end - dt, since)
            eff_dt = t_end - eff_start
            if eff_dt <= 0:
                continue
            total = total + v * (eff_dt / dt)
        return total

    def pile_active_minutes_since(self, since: float = 0.0) -> np.ndarray:
        """Per-pile active minutes restricted to ``[since, T]`` -- see
        ``_windowed_sum``. ``since=0.0`` returns the same array as the
        running total ``self.pile_active_minutes``."""
        return self._windowed_sum(self.history_pile_active_dt, (self.n_piles,), since)

    def connector_active_minutes_since(self, since: float = 0.0) -> np.ndarray:
        """Per-connector active minutes restricted to ``[since, T]``
        (shape n_piles x n_connectors) -- see ``_windowed_sum``."""
        return self._windowed_sum(
            self.history_connector_active_dt, (self.n_piles, self.n_connectors), since
        )

    def pile_energy_sold_since(self, since: float = 0.0) -> np.ndarray:
        """Per-pile energy sold (kWh) restricted to ``[since, T]`` -- see
        ``_windowed_sum``."""
        return self._windowed_sum(self.history_pile_energy, (self.n_piles,), since)

    def connector_energy_sold_since(self, since: float = 0.0) -> np.ndarray:
        """Per-connector energy sold (kWh) restricted to ``[since, T]``
        (shape n_piles x n_connectors) -- see ``_windowed_sum``."""
        return self._windowed_sum(
            self.history_connector_energy, (self.n_piles, self.n_connectors), since
        )

    def pile_utilization(self, sim_time: float, since: float = 0.0) -> np.ndarray:
        """
        Fraction of time each pile had at least one EV plugged in, over
        ``[since, sim_time]``.

        ``sim_time`` is always the *whole-run* clock value (e.g.
        ``engine.current_time``), matching ``queueing_summary``'s own
        ``sim_time``/``since`` convention -- the window length used as the
        denominator is ``sim_time - since`` (or ``sim_time`` when
        ``since=0.0``, the original pre-warm-up-feature behaviour exactly).
        """
        window = sim_time - since if since > 0 else sim_time
        if window <= 0:
            return np.zeros(self.n_piles)
        active = self.pile_active_minutes if since <= 0 else self.pile_active_minutes_since(since)
        return active / window

    def connector_utilization(self, sim_time: float, since: float = 0.0) -> np.ndarray:
        """Fraction of time each connector was occupied (shape n_piles x
        n_connectors), over ``[since, sim_time]``. See ``pile_utilization``
        for the ``sim_time``/``since`` convention."""
        window = sim_time - since if since > 0 else sim_time
        if window <= 0:
            return np.zeros((self.n_piles, self.n_connectors))
        active = (
            self.connector_active_minutes
            if since <= 0
            else self.connector_active_minutes_since(since)
        )
        return active / window

    def mean_connector_utilization(self, sim_time: float, since: float = 0.0) -> float:
        """Mean connector busy fraction across all connectors, over
        ``[since, sim_time]``."""
        util = self.connector_utilization(sim_time, since)
        return float(np.mean(util)) if util.size else 0.0

    def total_energy(self, since: float = 0.0) -> float:
        """Total energy sold (kWh) at the station (sum over piles), over
        ``[since, T]`` -- station-level, time-windowed (unlike
        ``average_energy`` below, which is per-customer and cohort-filtered
        instead). ``since=0.0`` is the whole run."""
        pile_e = self.pile_energy_sold if since <= 0 else self.pile_energy_sold_since(since)
        return float(np.sum(pile_e))

    def pile_energy(self, since: float = 0.0) -> np.ndarray:
        """Total energy sold (kWh) by each pile, over ``[since, T]``."""
        if since <= 0:
            return self.pile_energy_sold.copy()
        return self.pile_energy_sold_since(since)

    def average_energy(self, since: float = 0.0) -> float:
        """
        Mean energy delivered (kWh) among finished EVs (0 if none).

        Unlike ``total_energy``/``pile_energy`` (station-level, time-
        windowed), this is a per-customer statistic, so ``since`` filters
        the *cohort* the same way ``finished_time_arrays`` does
        (``arrival_time >= since``), not the energy itself -- a given
        finished EV's own ``energy_received`` is its total over its whole
        stay regardless of when the warm-up boundary fell.
        """
        evs = [ev for ev in self.finished_evs if ev.arrival_time >= since]
        if not evs:
            return 0.0
        return float(np.mean([ev.energy_received for ev in evs]))

    def average_pile_energy(self, since: float = 0.0) -> np.ndarray:
        """
        Mean energy delivered (kWh) per finished EV that used each pile.

        Piles with no (cohort-matching) finished EVs return 0. Uses
        ``pile_tracker`` kept after departure. ``since`` is a cohort filter
        (``arrival_time >= since``), same convention as ``average_energy``.
        """
        buckets: list[list[float]] = [[] for _ in range(self.n_piles)]
        for ev in self.finished_evs:
            if ev.arrival_time < since:
                continue
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

    def cohort_ev_ids(self, *, queued: bool = False, boundary: bool = False) -> set[int]:
        """
        Ids of the boundary cohorts, for use as ``include_ids`` below.

        ``queued`` adds ``queued_at_warmup_end`` (waiting at the boundary),
        ``boundary`` adds ``in_service_at_warmup_end`` (already plugged in
        at it). Both arrived *before* ``warmup_period``, so the ordinary
        ``since=warmup_period`` filter excludes them -- these are exactly
        the ids you pass to add them back. Mirrors
        ``offline_cl_opt.boundary.Cohort``'s QUEUED / BOUNDARY, so the
        simulation and the optimizer can be made to report over the same
        vehicle set.
        """
        ids: set[int] = set()
        if queued:
            ids.update(ev.id for ev in self.queued_at_warmup_end)
        if boundary:
            ids.update(snap.ev_id for snap in self.in_service_at_warmup_end)
        return ids

    def finished_time_arrays(
        self,
        since: float = 0.0,
        *,
        include_ids: set[int] | None = None,
        truncate_included: bool = True,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Per-finished-EV wait, service, and sojourn times (minutes).

        Definitions
        -----------
        * wait W_q = service_start_time - arrival_time
        * service S = departure_time - service_start_time
        * sojourn W = departure_time - arrival_time

        ``since`` : restrict to the post-warm-up cohort, ``arrival_time >=
        since`` (the standard DES warm-up convention -- exclude customers
        who were already in the system during the warm-up transient, even
        if they finish after it; see the module docstring). Default
        ``0.0`` includes everyone, matching the original behaviour exactly.

        ``include_ids`` : additionally keep these EVs even though they
        arrived before ``since`` -- build it with ``cohort_ev_ids`` to fold
        the boundary queue and/or the already-charging vehicles back into a
        measured-window report.

        ``truncate_included`` : measure the added EVs' times from ``since``
        rather than from their real arrival, i.e. count only the part of
        their stay that falls inside the measured window. On by default
        because that is what makes the figures comparable with the offline
        models, which place these vehicles at ``a=0`` and so measure them
        from the boundary too (see ``offline_cl_opt.boundary``). Set
        ``False`` for their true end-to-end times, which are the right
        thing for a "how long did this driver actually wait" question but
        *not* comparable against an optimizer's mean sojourn.

        Raises
        ------
        MetricsError
            If any finished EV is missing ``service_start_time`` or has a
            non-finite ``departure_time`` (logical bug: finished implies served).
        """
        include_ids = include_ids or set()
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
            added = ev.arrival_time < since and ev.id in include_ids
            if ev.arrival_time < since and not added:
                continue
            # Clock each added EV from the window start, not its real
            # arrival, so its numbers mean the same thing the optimizer's do.
            origin = since if (added and truncate_included) else ev.arrival_time
            waits.append(max(0.0, ev.service_start_time - origin))
            services.append(ev.departure_time - max(ev.service_start_time, origin))
            sojourns.append(ev.departure_time - origin)

        return (
            np.asarray(waits, dtype=float),
            np.asarray(services, dtype=float),
            np.asarray(sojourns, dtype=float),
        )

    def sojourn_by_cohort(
        self, since: float = 0.0, *, truncate_included: bool = True
    ) -> dict[str, dict[str, float]]:
        """
        Total/mean sojourn over the same three nested cohorts the offline
        models report (``offline_cl_opt.boundary.COHORT_LEVELS``):
        ``"measurement"`` (arrived after ``since``), ``"measurement_queued"``
        (plus those queued at the boundary) and ``"all"`` (plus those
        already charging at it). Each value is ``{"n", "total_sojourn",
        "mean_sojourn"}``.

        Pass ``since=metrics.warmup_period`` to get the measured-window
        figures directly comparable with ``ConnectorLaneSolution.by_cohort``
        / ``DWSolution.by_cohort`` on the same run.
        """
        levels = (
            ("measurement", {"queued": False, "boundary": False}),
            ("measurement_queued", {"queued": True, "boundary": False}),
            ("all", {"queued": True, "boundary": True}),
        )
        out: dict[str, dict[str, float]] = {}
        for name, flags in levels:
            _, _, sojourns = self.finished_time_arrays(
                since,
                include_ids=self.cohort_ev_ids(**flags),  # type: ignore[arg-type]
                truncate_included=truncate_included,
            )
            out[name] = {
                "n": float(sojourns.size),
                "total_sojourn": float(np.sum(sojourns)) if sojourns.size else 0.0,
                "mean_sojourn": float(np.mean(sojourns)) if sojourns.size else 0.0,
            }
        return out

    def grid_sojourn_by_cohort(
        self, delta: float, since: float = 0.0
    ) -> dict[str, dict[str, float]]:
        """
        This run's own schedule replayed on a ``delta`` slot grid, reported
        over the same three nested cohorts as ``sojourn_by_cohort``.

        Why this exists. The DES is continuous-time: a connector passes to
        the next vehicle the *instant* the previous one finishes. The
        offline models put every plug-in and departure on a slot boundary
        (assumption A1), so each handover costs the successor up to
        ``delta`` minutes. Comparing an optimum against the raw
        continuous-time mean is therefore apples-to-oranges -- that mean is
        not achievable by any grid schedule. This replays the *same*
        schedule on the grid, which is the like-for-like number.

        Method. Group finished vehicles by the physical lane they used
        (pile *and* connector -- a connector index alone is ambiguous once
        there is more than one pile), walk each lane's handover chain in
        service order, and give every vehicle ``ceil(service_minutes /
        delta)`` slots, starting no earlier than both its own release slot
        and its predecessor's discretized departure. Only the part of a
        stay inside the window counts, so a vehicle already charging at
        ``since`` is not charged for its warm-up time -- the same
        convention the offline models use when they place such a vehicle at
        ``a=0``.

        Note this rounds each *stay* up once. Rounding a stay's two
        endpoints up independently instead would compress it (an 18.21
        minute stay squeezed into 18 slots) and produce departure times no
        policy can actually achieve -- which the offline model rejects via
        its earliest-departure bound (24).

        Caveats. Like ``sojourn_by_cohort`` this covers finished vehicles
        only. And it replays *occupancy*, not power: it assumes each
        vehicle still needs the same charging time after being shifted onto
        the grid. Shifting changes which vehicles overlap, hence how the
        pile's modules are shared, so the result is a close reconstruction
        rather than a guaranteed-feasible schedule for the offline model.

        Returns, per cohort level, ``{"n", "total_sojourn", "mean_sojourn",
        "total_departure_slots"}``. The last is ``sum_j D_j`` in the offline
        objective's own slot units, directly comparable with
        ``ConnectorLaneSolution.objective`` when the objective covers the
        matching cohorts.
        """
        if delta <= 0:
            raise ValueError(f"delta must be positive, got {delta}")

        # Vehicles that actually held a lane inside the window.
        active = [
            ev
            for ev in self.finished_evs
            if ev.service_start_time is not None
            and np.isfinite(ev.departure_time)
            and ev.departure_time > since
            and ev.pile_tracker is not None
            and ev.connector_id_tracker is not None
        ]

        lanes: dict[tuple[int, int], list["EV"]] = {}
        for ev in active:
            lanes.setdefault(
                (ev.pile_tracker.id, ev.connector_id_tracker), []  # type: ignore[union-attr]
            ).append(ev)

        departure_slot: dict[int, int] = {}
        for group in lanes.values():
            group.sort(key=lambda e: e.service_start_time)  # type: ignore[arg-type,return-value]
            lane_free = 0
            for ev in group:
                a_rel = max(0.0, ev.arrival_time - since)
                served_from = max(float(ev.service_start_time), since)  # type: ignore[arg-type]
                need = math.ceil(round((ev.departure_time - served_from) / delta, 9))
                start = max(math.ceil(round(a_rel / delta, 9)), lane_free)
                departure_slot[ev.id] = lane_free = start + max(need, 1)

        by_id = {ev.id: ev for ev in active}
        levels = (
            ("measurement", {"queued": False, "boundary": False}),
            ("measurement_queued", {"queued": True, "boundary": False}),
            ("all", {"queued": True, "boundary": True}),
        )
        out: dict[str, dict[str, float]] = {}
        for name, flags in levels:
            keep = self.cohort_ev_ids(**flags)  # type: ignore[arg-type]
            ids = [
                j
                for j in departure_slot
                if by_id[j].arrival_time >= since or j in keep
            ]
            sojourns = [
                delta * departure_slot[j] - max(0.0, by_id[j].arrival_time - since)
                for j in ids
            ]
            out[name] = {
                "n": float(len(ids)),
                "total_sojourn": float(sum(sojourns)),
                "mean_sojourn": float(sum(sojourns) / len(sojourns)) if sojourns else 0.0,
                "total_departure_slots": float(sum(departure_slot[j] for j in ids)),
            }
        return out

    def mean_wait(
        self,
        since: float = 0.0,
        *,
        include_ids: set[int] | None = None,
        truncate_included: bool = True,
    ) -> float:
        """Mean queue wait W_q among finished EVs (0 if none). See
        ``finished_time_arrays`` for the ``since`` cohort convention and the
        ``include_ids``/``truncate_included`` keywords forwarded here."""
        waits, _, _ = self.finished_time_arrays(
            since, include_ids=include_ids, truncate_included=truncate_included
        )
        return float(np.mean(waits)) if waits.size else 0.0

    def max_wait(
        self,
        since: float = 0.0,
        *,
        include_ids: set[int] | None = None,
        truncate_included: bool = True,
    ) -> float:
        """Maximum queue wait W_q among finished EVs (0 if none)."""
        waits, _, _ = self.finished_time_arrays(
            since, include_ids=include_ids, truncate_included=truncate_included
        )
        return float(np.max(waits)) if waits.size else 0.0

    def mean_service(
        self,
        since: float = 0.0,
        *,
        include_ids: set[int] | None = None,
        truncate_included: bool = True,
    ) -> float:
        """
        Mean service / charge time S among finished EVs (0 if none).

        S = departure_time - service_start_time (minutes plugged in).
        Replication tables report this as ``avg charge time``.
        """
        _, services, _ = self.finished_time_arrays(
            since, include_ids=include_ids, truncate_included=truncate_included
        )
        return float(np.mean(services)) if services.size else 0.0

    def mean_sojourn(
        self,
        since: float = 0.0,
        *,
        include_ids: set[int] | None = None,
        truncate_included: bool = True,
    ) -> float:
        """Mean sojourn W among finished EVs (0 if none). Pass
        ``include_ids=metrics.cohort_ev_ids(queued=True, boundary=True)`` to
        fold the boundary cohorts in -- see ``finished_time_arrays``."""
        _, _, sojourns = self.finished_time_arrays(
            since, include_ids=include_ids, truncate_included=truncate_included
        )
        return float(np.mean(sojourns)) if sojourns.size else 0.0

    def service_squared_cv(
        self,
        since: float = 0.0,
        *,
        include_ids: set[int] | None = None,
        truncate_included: bool = True,
    ) -> float:
        """
        Squared coefficient of variation of service times, c_s^2 = Var(S)/E[S]^2.

        Uses sample variance (ddof=1). Returns NaN if fewer than two finished EVs
        or E[S] = 0.
        """
        _, services, _ = self.finished_time_arrays(
            since, include_ids=include_ids, truncate_included=truncate_included
        )
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
        since: float = 0.0,
        include_ids: set[int] | None = None,
        truncate_included: bool = True,
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
            Total elapsed episode time (usually ``engine.current_time``) --
            always the *whole-run* clock value, regardless of ``since``.
        n_servers :
            Number of parallel servers for rho theory. Defaults to
            ``n_piles * n_connectors``.
        since :
            Restrict to the post-``since`` window (see the module
            docstring, "Warm-up period" -- typically
            ``since=metrics.warmup_period``). ``T`` (the window length used
            for ``lambda_eff``/L/Q/``rho_sim``/``E_total``) becomes
            ``sim_time - since``; finished EVs feeding the per-customer
            statistics (``W``, ``W_q``, ``S``, ``n_finished``, ``E_avg``)
            are filtered to ``arrival_time >= since``. Default ``0.0`` is
            the whole run, matching the original behaviour exactly.
        include_ids, truncate_included :
            Forwarded to ``finished_time_arrays`` -- use
            ``cohort_ev_ids(queued=..., boundary=...)`` to fold the
            boundary cohorts back into the per-customer statistics.
            Note these affect only the customer averages (``W``, ``W_q``,
            ``S``, ``n_finished`` and hence ``lambda_eff``/``L_theory``/
            ``Q_theory``); the time-average ``L_sim``/``Q_sim``/``rho_sim``
            come from the online occupancy history and already count every
            vehicle physically present in the window, whichever cohort it
            belongs to.

        Returns
        -------
        dict
            ``T``, ``since``, ``n_finished``, ``lambda_eff``, ``W``, ``W_q``,
            ``W_q_max``, ``S``, ``mu``, ``c``, ``L_sim``, ``L_theory``,
            ``Q_sim``, ``Q_theory``, ``rho_sim``, ``rho_theory``,
            ``E_total``, ``E_avg``, ``c_s2``.

            Per-pile energy / utilization arrays are available via
            ``pile_energy``, ``average_pile_energy``, and ``pile_utilization``.
        """
        window = float(sim_time) - since
        T = window if window > 0 else 1.0

        waits, services, sojourns = self.finished_time_arrays(
            since, include_ids=include_ids, truncate_included=truncate_included
        )
        n_finished = int(sojourns.size)
        lambda_eff = n_finished / T
        W = float(np.mean(sojourns)) if sojourns.size else 0.0
        W_q = float(np.mean(waits)) if waits.size else 0.0
        W_q_max = float(np.max(waits)) if waits.size else 0.0
        S = float(np.mean(services)) if services.size else 0.0
        mu = (1.0 / S) if S > 0 else 0.0
        c_s2 = self.service_squared_cv(since)

        c = int(n_servers) if n_servers is not None else self.n_piles * self.n_connectors
        L_sim = self.average_L(since)
        Q_sim = self.average_Q(since)
        L_theory = lambda_eff * W
        Q_theory = lambda_eff * W_q

        # rho_sim is windowed by `since` too, over the same [since, sim_time]
        # window as everything else (pile_utilization/connector_utilization
        # divide by sim_time - since internally). rho_theory is unaffected
        # either way (it only depends on lambda_eff/mu/c, already windowed).
        rho_sim = self.mean_connector_utilization(float(sim_time), since)
        rho_theory = (lambda_eff / (c * mu)) if (c > 0 and mu > 0) else 0.0

        return {
            "T": T,
            "since": float(since),
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
            "E_total": self.total_energy(since),
            "E_avg": self.average_energy(since),
            "c_s2": float(c_s2) if np.isfinite(c_s2) else float("nan"),
        }
