"""
One charging pile: fixed nozzles and a shared pool of power bricks.

A pile belongs to a ChargingStation. EVs plug into nozzle slots. Slots are
fixed-length (None = free) so when EV A leaves, EV B keeps the same nozzle_id
and the matching entry in ev_bricks stays aligned.

Brick counts themselves are written by a PowerPolicy (see policy/power/).
This class only stores the allotment and answers questions like "are we full?"
or "would isolated demand exceed our brick pool?" (is_overloaded).
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from config import BRICK_CHECK_THRESH, CHECK_INVARIANTS
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .ev import EV
    from .station import ChargingStation


@dataclass
class ChargingPile:
    id: int
    n_nozzles: int  # Physical plugs on this pile
    num_bricks: int  # Shared discrete power chunks
    p_brick: float  # kW per brick
    station: ChargingStation = None
    # Free a brick when utilization of the last brick falls below this fraction.
    brickCheck_thresh: float = BRICK_CHECK_THRESH

    def __post_init__(self):
        if self.num_bricks < self.n_nozzles:
            raise ValueError(
                f"num_bricks: {self.num_bricks} is less than the number of nozzles: {self.n_nozzles}"
            )
        self.power_supp: float = self.num_bricks * self.p_brick
        # Index i always means nozzle i — never compact this list on disconnect.
        self.nozzles: List[EV | None] = [None] * self.n_nozzles
        self.ev_bricks: List[int] = [0] * self.n_nozzles

    def reset(self):
        """Clear all plugs for a new episode."""
        for ev in self.evs:
            if ev.pile is self:
                ev.pile = None
            if ev.nozzle_id is not None:
                ev.nozzle_id = None
            if ev.pile_tracker is self:
                ev.pile_tracker = None

        self.nozzles = [None] * self.n_nozzles
        self.ev_bricks = [0] * self.n_nozzles

    @property
    def evs(self) -> List[EV]:
        """Currently plugged EVs, ordered by nozzle index."""
        return [ev for ev in self.nozzles if ev is not None]

    @property
    def is_full(self) -> bool:
        return all(slot is not None for slot in self.nozzles)

    @property
    def free_nozzles(self) -> int:
        return sum(1 for slot in self.nozzles if slot is None)

    def connect_ev(self, ev: EV):
        """Plug EV into the first free nozzle and link both sides."""
        if ev.pile is not None:
            raise ValueError("EV already assigned to a pile.")
        if self.is_full:
            raise ValueError("Charging Pile is full.")

        for i, slot in enumerate(self.nozzles):
            if slot is None:
                self.nozzles[i] = ev
                ev.pile = self
                ev.pile_tracker = self
                ev.nozzle_id = i
                ev.nozzle_id_tracker = i  # kept after departure for plots
                # Assignment happens between events; clock is already at decision time.
                ev.service_start_time = self.current_time
                return

        raise ValueError("Charging Pile is full.")

    def disconnect_ev(self, ev: EV):
        """
        Unplug EV, free its nozzle/bricks, and invalidate its pending events.

        Called after the engine has already projected energy over the last
        interval and committed ``s_current`` (typically to ``s_f``). Energy
        metrics for that interval used the pre-departure ``p_act``; they are
        not recomputed here.
        """
        if ev.pile is None:
            raise ValueError("EV is not assigned to any pile.")
        if ev.pile != self:
            raise ValueError("EV is not assigned to this pile.")
        if ev.nozzle_id is None or self.nozzles[ev.nozzle_id] is not ev:
            raise ValueError("EV nozzle slot is inconsistent.")

        idx = ev.nozzle_id
        depart_t = self.next_time
        last_allot = ev.charge_trace[-1][4] if ev.charge_trace else float(ev.p_act)
        # Trace-only: SoC is already at the departure value, so p_req may have
        # dropped (taper) while p_act still holds the last redistribution
        # setpoint. Re-cap so the final sample is instantaneous draw at s_f
        # under the same allotment — does not affect energy already accrued.
        ev.p_act = min(ev.p_req, last_allot)
        ev.record_charge_sample(depart_t, p_allot=last_allot)

        self.nozzles[idx] = None
        self.ev_bricks[idx] = 0
        ev.invalidate_pending_events()
        ev.pile = None
        ev.nozzle_id = None
        # nozzle_id_tracker / pile_tracker kept for post-run visualization
        # Engine still has current_time < next_time while processing this departure.
        ev.departure_time = depart_t

    @property
    def current_time(self) -> float:
        if self.station is None:
            return float("inf")
        return self.station.current_time

    @property
    def next_time(self) -> float:
        if self.station is None:
            return float("inf")
        return self.station.next_time

    @property
    def is_active(self) -> bool:
        return any(slot is not None for slot in self.nozzles)

    @property
    def bricks_used(self) -> int:
        return sum(self.ev_bricks)

    @property
    def is_overloaded(self) -> bool:
        """True if giving every EV its isolated ceil(p_req/p_brick) needs more bricks than we have."""
        if not self.evs:
            return False
        return sum(ceil(ev.p_req / self.p_brick) for ev in self.evs) > self.num_bricks

    @property
    def power_reqs(self) -> List[float]:
        return [ev.p_req for ev in self.evs]

    @property
    def remaining_reqs(self) -> List[float]:
        """
        How much requested power is still unmet at each nozzle, in kW.
        Empty nozzles get -inf so argmax in the power policy skips them.
        """
        reqs: List[float] = []
        for i, ev in enumerate(self.nozzles):
            if ev is None:
                reqs.append(float("-inf"))
            else:
                reqs.append(ev.p_req - self.ev_bricks[i] * self.p_brick)
        return reqs

    def check_invariants(self) -> None:
        """Cheap sanity checks used when CHECK_INVARIANTS is on."""
        if not CHECK_INVARIANTS:
            return
        if len(self.nozzles) != self.n_nozzles:
            raise AssertionError("nozzles length mismatch")
        if len(self.ev_bricks) != self.n_nozzles:
            raise AssertionError("ev_bricks length mismatch")
        if sum(self.ev_bricks) > self.num_bricks:
            raise AssertionError(
                f"brick over-allocation: {sum(self.ev_bricks)} > {self.num_bricks}"
            )
        seen = set()
        for i, ev in enumerate(self.nozzles):
            if ev is None:
                if self.ev_bricks[i] != 0:
                    raise AssertionError(f"empty nozzle {i} has bricks")
                continue
            if ev.nozzle_id != i:
                raise AssertionError(f"EV {ev.id} nozzle_id {ev.nozzle_id} != slot {i}")
            if ev.pile is not self:
                raise AssertionError(f"EV {ev.id} pile link broken")
            if i in seen:
                raise AssertionError(f"duplicate nozzle index {i}")
            seen.add(i)
