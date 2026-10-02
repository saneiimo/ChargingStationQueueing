"""
Boundary conditions and reporting cohorts for the offline models.

Everything here turns ``SimulationEngine``'s warm-up boundary snapshot
(``metrics.queued_at_warmup_end`` / ``metrics.in_service_at_warmup_end`` --
see ``metrics/metrics_tracker.py``'s own module docstring, "Warm-up
period") into the data the offline models need. ``offline_cl_dw`` reuses
all of it rather than duplicating it (matching this repo's existing one-way
dependency: offline_cl_dw depends on offline_cl_opt, never the reverse).

Three cohorts, three kinds of vehicle
-------------------------------------
A measured-window model can contain vehicles from three disjoint sources,
and ``Cohort`` tags which one each came from:

* ``MEASUREMENT`` -- arrived after the warm-up boundary. Ordinary vehicles
  in every respect; their ``a`` is just shifted back by ``warmup_period``.
* ``QUEUED`` -- waiting in the queue at the boundary, not yet plugged in.
  Also ordinary vehicles: an EV that never started charging has the same
  SoC it arrived with, so only ``a=0`` ("already here") is needed.
* ``BOUNDARY`` -- already plugged in and mid-charge at the boundary. These
  are the ones that need real model changes; ``BoundaryVehicle`` /
  ``BoundaryMode`` describe them.

The cohort tags matter for two independent things: which vehicles the
objective is minimised over (``model.build_cl_model``'s
``objective_cohorts``), and which vehicles each reported sojourn figure
covers (``solution.extract_solution`` reports all three nested cohorts
regardless of what was optimised).

Boundary modes
--------------
* ``FIXED``: the optimizer has no control -- occupancy and power are
  pinned to exactly what the simulation realized, discretized to the
  target ``delta``. Contributes a *constant* sojourn, so it is normally left
  out of the objective (see ``objective_cohorts``) while still consuming
  connectors and modules, which is the whole point of modelling it.
* ``OPTIMIZE``: the optimizer controls this vehicle's future power and
  departure -- but never its pile/connector (unplugging a mid-charge
  vehicle is unrealistic), so only the lane is pinned; energy accounting
  starts from how much it already has, not from zero.

Reconstructing the realized power profile
-----------------------------------------
``realized_slot_power`` is the one piece of real numerical work here. It
converts an event-sampled ``charge_trace`` into per-slot power by
computing the *exact* energy delivered in each slot and dividing by the
slot length -- the energy-equivalent constant power, in exactly the spirit
of ``instance.tau_delta_hours``. See that function's own docstring for why
a zero-order hold (holding each sample's ``p_act`` across the slot) is not
merely less accurate but actively unsafe here.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from enum import Enum
from math import exp
from typing import TYPE_CHECKING

from config import HR2MIN

from .instance import StationSpec, VehicleData

if TYPE_CHECKING:
    from metrics.metrics_tracker import InServiceAtBoundary
    from models.ev import EV


class Cohort(Enum):
    """Which of the three sources a vehicle in the measured window came from."""

    MEASUREMENT = "measurement"  # arrived after the warm-up boundary
    QUEUED = "queued"  # waiting in the queue at the boundary
    BOUNDARY = "boundary"  # already plugged in at the boundary


# The three nested cohort selections worth reporting/optimising over, in
# increasing order of inclusiveness. Named so callers never have to
# assemble a frozenset by hand (and so the nesting is obvious at a glance).
COHORTS_MEASUREMENT = frozenset({Cohort.MEASUREMENT})
COHORTS_MEASUREMENT_QUEUED = frozenset({Cohort.MEASUREMENT, Cohort.QUEUED})
COHORTS_ALL = frozenset({Cohort.MEASUREMENT, Cohort.QUEUED, Cohort.BOUNDARY})

# Ordered for reporting: every solution reports all three of these.
COHORT_LEVELS: tuple[tuple[str, frozenset[Cohort]], ...] = (
    ("measurement", COHORTS_MEASUREMENT),
    ("measurement_queued", COHORTS_MEASUREMENT_QUEUED),
    ("all", COHORTS_ALL),
)


class BoundaryMode(Enum):
    """How an in-service (already-plugged-in) boundary vehicle is modeled."""

    FIXED = "fixed"  # optimizer has no control -- trajectory taken as-is
    OPTIMIZE = "optimize"  # optimizer controls power/departure; lane is pinned


@dataclass(frozen=True)
class BoundaryVehicle:
    """
    One vehicle already occupying a connector when the modeled horizon
    begins (t=0 of the measured phase / warm-up boundary).

    ``vehicle_id`` must match a ``VehicleData.id`` in the same
    ``vehicles`` list passed to the model builder -- this record only
    carries the *extra* boundary-specific information VehicleData has no
    field for; the vehicle's own battery/taper parameters (c_b/Q, s_i,
    s_f, s_th, p_max) still come from its ordinary VehicleData entry.
    """

    vehicle_id: int
    pile: int
    connector: int
    mode: BoundaryMode
    # OPTIMIZE only: energy (kWh) already delivered before t=0, relative
    # to the vehicle's own arrival SoC s_i -- the x_j0 recursion must
    # start here instead of the usual 0 a freshly-arriving vehicle gets.
    initial_energy_kwh: float = 0.0
    # FIXED only: its already-known trajectory -- departure slot (floor-
    # discretized to the target delta, see _floor_slot) and per-slot power
    # (kW), both relative to t=0 of the modeled window.
    departure_slot: int = 0
    power: dict[int, float] = field(default_factory=dict)
    # OPTIMIZE only: the realized trajectory, kept as a ready-made seed
    # column for the Dantzig-Wolfe pricer (see offline_cl_dw.preprocess.
    # boundary_seed_plan). Departure rounds *up* here, unlike FIXED's
    # floor -- see realized_slot_power for why the two differ.
    seed_power: dict[int, float] = field(default_factory=dict)
    seed_departure_slot: int = 0


@dataclass
class MeasurementInstance:
    """
    Everything a measured-window offline model needs, assembled by
    ``build_measurement_instance``.

    ``discarded`` lists in-service EVs dropped because their floor-
    discretized departure landed at slot 0 (they left within the first
    ``delta`` of the window, so they occupy no whole slot) -- they are
    absent from ``vehicles``, ``boundary_vehicles`` and ``cohorts`` alike,
    which is what makes the removal safe; see ``build_measurement_instance``.
    """

    vehicles: list[VehicleData]
    boundary_vehicles: dict[int, BoundaryVehicle]
    cohorts: dict[int, Cohort]
    discarded: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Exact charging physics, mirrored from the simulator
# ---------------------------------------------------------------------------


def _advance(
    s: float, p_act: float, tan_B: float, c_b: float, dt: float
) -> tuple[float, float]:
    """
    State after charging for ``dt`` minutes at allotment ``p_act`` (kW),
    starting from SoC ``s``: returns ``(s_next, energy_kWh)``.

    Mirrors ``models.ev.EV.update_s_next`` and ``EV.compute_deltaE_power``
    exactly -- constant power while below the knee ``s_taper = 1 -
    p_act/tan_B``, exponential decay after it, and the mixed case when the
    interval crosses the knee. Both outputs are computed here together
    precisely so they can never drift apart from each other.

    ``c_b`` is in the simulator's native kW*min; the returned energy is
    kWh (hence the ``HR2MIN`` divisions), matching this package's units.
    """
    if p_act <= 0.0 or dt <= 0.0:
        return s, 0.0

    s_tap = 1.0 - p_act / tan_B  # SoC at which the BMS request meets p_act
    one_m_tap = 1.0 - s_tap
    k_tmp = dt * p_act / c_b
    rate = tan_B / c_b

    # Already at/past the knee: pure exponential decay from the current SoC.
    if s >= s_tap - 1e-12:
        expo = exp(-rate * dt)
        return 1.0 - (1.0 - s) * expo, -c_b * (1.0 - s) * (expo - 1.0) / HR2MIN

    dt_tap = c_b / p_act * (s_tap - s)  # time remaining at constant power
    if dt <= dt_tap:
        # Never reaches the knee: constant power throughout.
        return s + k_tmp, p_act * dt / HR2MIN

    # Crosses the knee: constant power to it, exponential decay after.
    k_expo = k_tmp - (s_tap - s)
    expo = exp(-rate * (dt - dt_tap))
    energy = (p_act * dt_tap - c_b * one_m_tap * (expo - 1.0)) / HR2MIN
    return 1.0 - one_m_tap * exp(-k_expo / one_m_tap), energy


def realized_slot_power(
    snap: "InServiceAtBoundary", delta: float, K: int
) -> tuple[dict[int, float], int, int]:
    """
    The energy-equivalent per-slot power (kW) this vehicle actually drew
    after the boundary, plus its floor- and ceil-discretized departure
    slots: ``(power, departure_floor, departure_ceil)``.

    Method. The vehicle's post-boundary trajectory is a sequence of
    segments, each starting at a known ``(time, SoC, p_act)``: the first
    from the boundary snapshot itself, the rest from ``post_boundary_
    trace()``. Within a segment the allotment is constant and the exact
    energy follows the closed form the simulator itself integrates
    (``_advance``). Slot ``k``'s power is the total energy delivered in
    ``[k*delta, (k+1)*delta)`` divided by the slot length in hours -- the
    constant power that delivers exactly the same energy, in the same
    spirit as ``instance.tau_delta_hours``.

    Why not a zero-order hold. Holding each sample's ``p_act`` across the
    slot is not merely a little optimistic -- it is unsound in two
    specific ways this construction avoids outright:

    * ``p_act`` is the power at the *start* of a segment and decays within
      it, so holding it overstates the slot's energy. Averaging instead
      gives ``p_bar <= y_k(1-e^{-h/tau})/h = y_k/tau^delta``, i.e. the
      taper constraint (18) is satisfied automatically.
    * Traces are event-sampled *per EV*, so two vehicles' most recent
      samples generally come from different instants. Holding each one
      independently can combine powers from allocation regimes that never
      coexisted, and the reconstructed pile total can exceed ``N*Delta``
      even though the real station never did. Averaging cannot: the
      per-instant total respects the pile limit, and an average of
      values in a convex set stays in it.

    Departure. ``departure_floor`` is the last fully-elapsed slot (the
    convention for a FIXED vehicle: never claim energy past the instant it
    was actually delivered). ``departure_ceil`` rounds up instead, which
    is what an OPTIMIZE-mode *seed column* needs, since the departure rule
    (19) requires the vehicle to have received its full ``W_j`` by the
    slot it unplugs in -- flooring would truncate energy and make the seed
    infeasible. Both are capped at ``K``; a vehicle that never departed
    within the run (``actual_departure_time`` is ``inf``) is censored at
    ``K``, matching the model's own horizon convention.
    """
    h = delta / 60.0
    t0 = snap.boundary_time
    p_req_max = snap.c_b * snap.c_rate
    tan_B = p_req_max / (1.0 - snap.s_th)

    # Departure relative to the boundary, censored at the horizon -- also
    # what keeps a still-charging vehicle (departure_time == inf) from
    # reaching math.floor() as an infinity.
    rel_departure = snap.actual_departure_time - t0
    if not math.isfinite(rel_departure):
        rel_departure = float(K) * delta
    rel_departure = max(0.0, min(rel_departure, float(K) * delta))

    # Segment starts: the boundary state first (the traces need not carry a
    # sample at the boundary instant -- typically they do not), then every
    # recorded post-boundary sample.
    segments: list[tuple[float, float, float]] = [
        (t0, snap.s_current, snap.p_act)
    ]
    for row in snap.post_boundary_trace():
        t_row = float(row[0])
        if t_row > t0 + 1e-9:
            segments.append((t_row, float(row[1]), float(row[3])))  # (t, SoC, p_act)
    segments.sort(key=lambda seg: seg[0])

    # Walk the segments once, splitting each at slot boundaries and at the
    # departure instant, accumulating exact energy per slot.
    energy_by_slot: dict[int, float] = {}
    for idx, (t_start, s_start, p_act) in enumerate(segments):
        t_end = segments[idx + 1][0] if idx + 1 < len(segments) else t0 + rel_departure
        t_end = min(t_end, t0 + rel_departure)
        if t_end <= t_start + 1e-12 or p_act <= 1e-9:
            continue
        s = s_start
        t = t_start
        while t < t_end - 1e-12:
            k = int(math.floor(round((t - t0) / delta, 9)))
            slot_end = t0 + (k + 1) * delta
            step_end = min(slot_end, t_end)
            s, e = _advance(s, p_act, tan_B, snap.c_b, step_end - t)
            if e > 0.0 and 0 <= k < K:
                energy_by_slot[k] = energy_by_slot.get(k, 0.0) + e
            t = step_end

    power = {k: e / h for k, e in energy_by_slot.items() if e / h > 1e-9}
    departure_floor = min(K, max(0, math.floor(round(rel_departure / delta, 9))))
    departure_ceil = min(K, max(0, math.ceil(round(rel_departure / delta, 9))))
    return power, departure_floor, departure_ceil


# ---------------------------------------------------------------------------
# Building the model's inputs
# ---------------------------------------------------------------------------


def vehicles_from_boundary(
    arrived: list[VehicleData],
    queued_at_warmup_end: list["EV"] | None,
    *,
    include_queued: bool,
) -> list[VehicleData]:
    """
    Vehicle list for the measured-window offline model (low-level; see
    ``build_measurement_instance`` for the one-call version that also
    produces boundary vehicles and cohort tags).

    ``arrived`` is the normal, already arrival-time-shifted vehicle set.
    When ``include_queued`` is True, every EV in ``queued_at_warmup_end``
    is appended as an ordinary vehicle with ``a=0``: a queued vehicle's
    SoC hasn't moved since its real arrival, so no other field needs
    adjusting -- it just needs to be modeled as "already here" at the
    start of the window. When False, the model assumes an empty queue at
    the start of the measured phase.
    """
    vehicles = list(arrived)
    if include_queued and queued_at_warmup_end:
        for ev in queued_at_warmup_end:
            vehicles.append(dataclasses.replace(VehicleData.from_ev(ev), a=0.0))
    return vehicles


def boundary_vehicles_from_in_service(
    in_service: list["InServiceAtBoundary"],
    delta: float,
    mode: BoundaryMode,
    K: int,
) -> tuple[list[VehicleData], dict[int, BoundaryVehicle], list[int]]:
    """
    Convert ``metrics.in_service_at_warmup_end`` into ``(vehicles,
    boundary, discarded)``: ordinary ``VehicleData`` entries to add to the
    model's vehicle list (``a=0``, same battery/taper parameters as the
    vehicle's real original arrival), the matching ``BoundaryVehicle``
    records, and the ids dropped entirely.

    A FIXED-mode vehicle whose floor-discretized departure lands at slot 0
    is **discarded**: it left within the first ``delta`` of the window, so
    under the floor convention it occupies no whole slot, and keeping it
    would force the two solvers to disagree (the compact model reads an
    all-zero occupancy as "never served", contributing ``D_j = K``, while
    a Dantzig-Wolfe column with ``start == departure == 0`` contributes
    ``0`` and trips ``postprocess.validate_schedule``'s degenerate-interval
    check). It is removed from *both* returned structures together --
    dropping it from only ``boundary`` would silently turn it into an
    ordinary vehicle with ``a=0`` demanding its full ``W_j``, which is far
    worse than the problem being avoided.

    ``mode`` applies uniformly to every in-service vehicle passed in --
    call this separately per subset (and merge the results) if you want a
    mix, e.g. some vehicles FIXED and others OPTIMIZE.
    """
    vehicles: list[VehicleData] = []
    boundary: dict[int, BoundaryVehicle] = {}
    discarded: list[int] = []

    for snap in in_service:
        power, dep_floor, dep_ceil = realized_slot_power(snap, delta, K)

        if mode is BoundaryMode.FIXED and dep_floor == 0:
            discarded.append(snap.ev_id)
            continue

        # Same battery/taper data the vehicle arrived with -- a=0 places
        # it at the very start of the modeled (measured) window.
        v = VehicleData(
            id=snap.ev_id,
            a=0.0,
            Q=snap.c_b / HR2MIN,
            s_i=snap.s_i,
            s_f=snap.s_f,
            s_th=snap.s_th,
            p_max=snap.c_b * snap.c_rate,
        )
        vehicles.append(v)

        if mode is BoundaryMode.FIXED:
            # No decision to make: pin exactly what happened, cut to the
            # last fully-elapsed slot.
            boundary[snap.ev_id] = BoundaryVehicle(
                vehicle_id=snap.ev_id,
                pile=snap.pile_id,
                connector=snap.connector_id,
                mode=BoundaryMode.FIXED,
                departure_slot=dep_floor,
                power={k: p for k, p in power.items() if k < dep_floor},
            )
        else:
            # Energy already delivered before t=0, relative to s_i (the
            # same baseline W_j/R_j use), clipped at 0 for safety. The
            # realized profile is kept as a ready-made seed column.
            initial_energy_kwh = (snap.c_b / HR2MIN) * (snap.s_current - snap.s_i)
            boundary[snap.ev_id] = BoundaryVehicle(
                vehicle_id=snap.ev_id,
                pile=snap.pile_id,
                connector=snap.connector_id,
                mode=BoundaryMode.OPTIMIZE,
                initial_energy_kwh=max(0.0, min(initial_energy_kwh, v.W)),
                seed_power={k: p for k, p in power.items() if k < dep_ceil},
                seed_departure_slot=dep_ceil,
            )

    return vehicles, boundary, discarded


def assert_whole_module_feasible(
    boundary: dict[int, BoundaryVehicle], station: StationSpec
) -> None:
    """
    Check condition (20) -- ``sum_c ceil(p/Delta) <= N`` -- on the pinned
    power of every FIXED-mode boundary vehicle, and raise with the
    offending pile-slot if it fails.

    ``realized_slot_power``'s averaging guarantees the *continuous* module
    condition ``sum_c p <= N*Delta`` unconditionally, which is all that
    rows (26)/(15)-with-relaxed-``r`` require. It does not guarantee the
    stricter *whole-module* condition: averaging can leave two connectors
    each just above a module boundary (76 + 74 kW needs 4 + 3 modules on a
    6-module pile). That needs an allotment change inside the slot and two
    vehicles landing near module boundaries at once, so it is a narrow
    residual rather than a likely one -- but a pinned power that violates
    it makes the model flatly infeasible with no hint as to why, so it is
    worth one cheap check that names the cause.
    """
    Delta = station.p_module
    by_pile_slot: dict[tuple[int, int], list[tuple[int, float]]] = {}
    for bv in boundary.values():
        if bv.mode is not BoundaryMode.FIXED:
            continue
        for k, p in bv.power.items():
            if p > 1e-9:
                by_pile_slot.setdefault((bv.pile, k), []).append((bv.vehicle_id, p))

    for (pile, k), items in sorted(by_pile_slot.items()):
        needed = sum(math.ceil(p / Delta - 1e-9) for _, p in items)
        if needed > station.n_modules:
            raise ValueError(
                f"FIXED boundary vehicles need {needed} whole modules at pile {pile}, "
                f"slot {k}, but the pile owns only {station.n_modules} -- condition (20) "
                f"fails for {[(vid, round(p, 2)) for vid, p in items]} kW at Delta="
                f"{Delta} kW. The reconstructed powers respect the continuous pile limit "
                "but not the whole-module one; solve with relax_modules=True (or the "
                "conservative module budget) for this instance, or coarsen delta so the "
                "allotment change falls on a slot boundary."
            )


def build_measurement_instance(
    *,
    arrived_post_warmup: list["EV"],
    queued_at_warmup_end: list["EV"] | None,
    in_service_at_warmup_end: list["InServiceAtBoundary"] | None,
    warmup_period: float,
    delta: float,
    horizon_minutes: float,
    include_queued: bool = True,
    boundary_mode: BoundaryMode | None = BoundaryMode.OPTIMIZE,
    station: StationSpec | None = None,
) -> MeasurementInstance:
    """
    One call from a finished simulation to a ready-to-solve measured-window
    instance: shifts arrivals back to ``t=0``, optionally folds in the
    boundary queue and the in-service vehicles, and tags every vehicle
    with its ``Cohort``.

    Parameters
    ----------
    arrived_post_warmup :
        ``metrics.arrived_post_warmup`` -- arrival times are shifted back
        by ``warmup_period`` here, so pass them exactly as the tracker
        reports them.
    include_queued :
        Fold ``queued_at_warmup_end`` in as ordinary ``a=0`` vehicles
        (cohort ``QUEUED``). ``False`` models an empty queue at the start
        of the measured phase -- matching ``flush_queue_at_warmup=True`` in
        the simulation, or simply choosing to ignore the warm-up backlog.
    boundary_mode :
        ``BoundaryMode.OPTIMIZE`` / ``FIXED`` to include the in-service
        vehicles (cohort ``BOUNDARY``) under that mode, or ``None`` to
        leave them out entirely, i.e. model the measured window as
        starting with every connector free.
    station :
        When given, ``assert_whole_module_feasible`` is run on FIXED-mode
        vehicles so a whole-module violation is reported here, by name,
        rather than surfacing later as a bare ``INFEASIBLE``. Pass it
        unless you are deliberately solving with relaxed modules.
    """
    K = math.ceil(round(horizon_minutes / delta, 9))

    vehicles: list[VehicleData] = []
    cohorts: dict[int, Cohort] = {}

    for ev in arrived_post_warmup:
        v = dataclasses.replace(
            VehicleData.from_ev(ev), a=float(ev.arrival_time) - warmup_period
        )
        vehicles.append(v)
        cohorts[v.id] = Cohort.MEASUREMENT

    if include_queued and queued_at_warmup_end:
        for ev in queued_at_warmup_end:
            v = dataclasses.replace(VehicleData.from_ev(ev), a=0.0)
            vehicles.append(v)
            cohorts[v.id] = Cohort.QUEUED

    boundary: dict[int, BoundaryVehicle] = {}
    discarded: list[int] = []
    if boundary_mode is not None and in_service_at_warmup_end:
        bvs, boundary, discarded = boundary_vehicles_from_in_service(
            in_service_at_warmup_end, delta, boundary_mode, K
        )
        for v in bvs:
            vehicles.append(v)
            cohorts[v.id] = Cohort.BOUNDARY
        # Catch a whole-module violation here, where it can name the pile-slot,
        # instead of letting it surface as an unexplained INFEASIBLE later.
        if station is not None:
            assert_whole_module_feasible(boundary, station)

    return MeasurementInstance(
        vehicles=vehicles,
        boundary_vehicles=boundary,
        cohorts=cohorts,
        discarded=discarded,
    )


def cohort_totals(
    per_vehicle_rows: list[dict[str, object]],
    cohorts: dict[int, Cohort],
) -> dict[str, dict[str, float]]:
    """
    Total/mean sojourn for each of ``COHORT_LEVELS``, computed by summing
    the per-vehicle rows rather than from the aggregate objective.

    Summing per-vehicle sojourns is deliberate: the models' objective is
    total sojourn over ``objective_cohorts`` only, so it cannot be split
    into cohort figures -- a cohort level outside it needs its own sum.
    Rows are ``{"vehicle_id": int, "sojourn_min": float, ...}`` as built by
    either package's ``extract_solution``.
    """
    out: dict[str, dict[str, float]] = {}
    for name, selection in COHORT_LEVELS:
        vals = [
            float(row["sojourn_min"])  # type: ignore[arg-type]
            for row in per_vehicle_rows
            if cohorts.get(int(row["vehicle_id"]), Cohort.MEASUREMENT) in selection  # type: ignore[arg-type]
        ]
        out[name] = {
            "n": float(len(vals)),
            "total_sojourn": float(sum(vals)),
            # nan, not 0.0: a cohort with no vehicles has no mean sojourn, and
            # 0.0 would read as perfect service rather than as no data.
            "mean_sojourn": float(sum(vals) / len(vals)) if vals else float("nan"),
        }
    return out


def _floor_slot(t_relative: float, delta: float) -> int:
    """
    Discretize a continuous, boundary-relative time down to the last fully
    elapsed slot -- e.g. t=12.43, delta=1 -> 12. Always floors, never
    rounds/ceils: rounding up would claim energy was delivered past the
    instant it actually was.
    """
    return max(0, math.floor(round(t_relative / delta, 9)))
