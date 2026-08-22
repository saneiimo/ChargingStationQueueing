"""
One charging pile: fixed dispensers and a shared pool of power modules.

A pile belongs to a ChargingStation. EVs plug into dispenser slots. Slots are
fixed-length (None = free) so when EV A leaves, EV B keeps the same dispenser_id
and the matching entry in ev_modules stays aligned.

Module counts themselves are written by a PowerPolicy (see policy/power/).
This class only stores the allotment and answers questions like "are we full?"
or "would isolated demand exceed our module pool?" (is_overloaded).
"""

from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from config import MODULE_CHECK_THRESH, CHECK_INVARIANTS
from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .ev import EV
    from .station import ChargingStation


@dataclass
class ChargingPile:
    id: int
    n_dispensers: int  # Physical plugs on this pile
    num_modules: int  # Shared discrete power chunks
    p_module: float  # kW per module
    station: ChargingStation = None
    # Free a module when utilization of the last module falls below this fraction.
    moduleCheck_thresh: float = MODULE_CHECK_THRESH

    def __post_init__(self):
        if self.num_modules < self.n_dispensers:
            raise ValueError(
                f"num_modules: {self.num_modules} is less than the number of dispensers: {self.n_dispensers}"
            )
        self.power_supp: float = self.num_modules * self.p_module
        # Index i always means dispenser i — never compact this list on disconnect.
        self.dispensers: List[EV | None] = [None] * self.n_dispensers
        self.ev_modules: List[int] = [0] * self.n_dispensers

    def reset(self):
        """Clear all plugs for a new episode."""
        for ev in self.evs:
            if ev.pile is self:
                ev.pile = None
            if ev.dispenser_id is not None:
                ev.dispenser_id = None
            if ev.pile_tracker is self:
                ev.pile_tracker = None

        self.dispensers = [None] * self.n_dispensers
        self.ev_modules = [0] * self.n_dispensers

    @property
    def evs(self) -> List[EV]:
        """Currently plugged EVs, ordered by dispenser index."""
        return [ev for ev in self.dispensers if ev is not None]

    @property
    def is_full(self) -> bool:
        return all(slot is not None for slot in self.dispensers)

    @property
    def free_dispensers(self) -> int:
        return sum(1 for slot in self.dispensers if slot is None)

    def connect_ev(self, ev: EV):
        """Plug EV into the first free dispenser and link both sides."""
        if ev.pile is not None:
            raise ValueError("EV already assigned to a pile.")
        if self.is_full:
            raise ValueError("Charging Pile is full.")

        for i, slot in enumerate(self.dispensers):
            if slot is None:
                self.dispensers[i] = ev
                ev.pile = self
                ev.pile_tracker = self
                ev.dispenser_id = i
                ev.dispenser_id_tracker = i  # kept after departure for plots
                # Assignment happens between events; clock is already at decision time.
                ev.service_start_time = self.current_time
                return

        raise ValueError("Charging Pile is full.")

    def disconnect_ev(self, ev: EV):
        """
        Unplug EV, free its dispenser/modules, and invalidate its pending events.

        Called after the engine has already projected energy over the last
        interval and committed ``s_current`` (typically to ``s_f``). Energy
        metrics for that interval used the pre-departure ``p_act``; they are
        not recomputed here.
        """
        if ev.pile is None:
            raise ValueError("EV is not assigned to any pile.")
        if ev.pile != self:
            raise ValueError("EV is not assigned to this pile.")
        if ev.dispenser_id is None or self.dispensers[ev.dispenser_id] is not ev:
            raise ValueError("EV dispenser slot is inconsistent.")

        idx = ev.dispenser_id
        depart_t = self.next_time
        last_allot = ev.charge_trace[-1][4] if ev.charge_trace else float(ev.p_act)
        # Trace-only: SoC is already at the departure value, so p_req may have
        # dropped (taper) while p_act still holds the last redistribution
        # setpoint. Re-cap so the final sample is instantaneous draw at s_f
        # under the same allotment — does not affect energy already accrued.
        ev.p_act = min(ev.p_req, last_allot)
        ev.record_charge_sample(depart_t, p_allot=last_allot)

        self.dispensers[idx] = None
        self.ev_modules[idx] = 0
        ev.invalidate_pending_events()
        ev.pile = None
        ev.dispenser_id = None
        # dispenser_id_tracker / pile_tracker kept for post-run visualization
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
        return any(slot is not None for slot in self.dispensers)

    @property
    def modules_used(self) -> int:
        return sum(self.ev_modules)

    @property
    def is_overloaded(self) -> bool:
        """True if giving every EV its isolated ceil(p_req/p_module) needs more modules than we have."""
        if not self.evs:
            return False
        return sum(ceil(ev.p_req / self.p_module) for ev in self.evs) > self.num_modules

    @property
    def power_reqs(self) -> List[float]:
        return [ev.p_req for ev in self.evs]

    @property
    def remaining_reqs(self) -> List[float]:
        """
        How much requested power is still unmet at each dispenser, in kW.
        Empty dispensers get -inf so argmax in the power policy skips them.
        """
        reqs: List[float] = []
        for i, ev in enumerate(self.dispensers):
            if ev is None:
                reqs.append(float("-inf"))
            else:
                reqs.append(ev.p_req - self.ev_modules[i] * self.p_module)
        return reqs

    def check_invariants(self) -> None:
        """Cheap sanity checks used when CHECK_INVARIANTS is on."""
        if not CHECK_INVARIANTS:
            return
        if len(self.dispensers) != self.n_dispensers:
            raise AssertionError("dispensers length mismatch")
        if len(self.ev_modules) != self.n_dispensers:
            raise AssertionError("ev_modules length mismatch")
        if sum(self.ev_modules) > self.num_modules:
            raise AssertionError(
                f"module over-allocation: {sum(self.ev_modules)} > {self.num_modules}"
            )
        seen = set()
        for i, ev in enumerate(self.dispensers):
            if ev is None:
                if self.ev_modules[i] != 0:
                    raise AssertionError(f"empty dispenser {i} has modules")
                continue
            if ev.dispenser_id != i:
                raise AssertionError(f"EV {ev.id} dispenser_id {ev.dispenser_id} != slot {i}")
            if ev.pile is not self:
                raise AssertionError(f"EV {ev.id} pile link broken")
            if i in seen:
                raise AssertionError(f"duplicate dispenser index {i}")
            seen.add(i)
