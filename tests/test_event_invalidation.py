"""
Checks that bumping event_generation kills stale DEPARTURE/CHARGE_CHANGE events,
and that zero power does not schedule or divide by zero.
"""

from __future__ import annotations

from models.ev import EV
from models.pile import ChargingPile
from simulation.event import Event, EventQueue, EventType
from policy.power.proportional import ProportionalPower


def test_event_generation_invalidates_both_event_types():
    heap = EventQueue()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=4, p_brick=25.0)
    ev = EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=0.0)
    pile.connect_ev(ev)

    # Force a DEPARTURE schedule first (not overloaded, single EV).
    ProportionalPower().update_power(pile)
    ev.update_charging_power(pile.ev_bricks[ev.nozzle_id] * pile.p_brick, 0.0, heap)
    assert heap.heap
    first = heap.heap[0]
    assert first.event_type == EventType.DEPARTURE
    old_id = first.event_id

    # Add second EV to create overload so next schedule may be CHARGE_CHANGE.
    ev2 = EV(id=1, c_b=150.0, s_i=0.2, s_f=0.9, arrival_time=0.0)
    pile.connect_ev(ev2)
    ProportionalPower().update_power(pile)
    # Manually set high brick count on ev so CHARGE_CHANGE is eligible.
    if pile.is_overloaded and pile.ev_bricks[ev.nozzle_id] <= 1:
        pile.ev_bricks[ev.nozzle_id] = 2
        pile.ev_bricks[ev2.nozzle_id] = max(1, pile.num_bricks - 2)

    ev.update_charging_power(pile.ev_bricks[ev.nozzle_id] * pile.p_brick, 0.0, heap)

    # Old DEPARTURE id must no longer match generation.
    assert old_id != ev.event_generation
    assert first.event_id != ev.event_generation


def test_disconnect_invalidates_pending_events():
    heap = EventQueue()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=5, p_brick=25.0)
    ev = EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=0.0)
    pile.connect_ev(ev)
    ProportionalPower().update_power(pile)
    ev.update_charging_power(100.0, 0.0, heap)
    gen_before = ev.event_generation
    pile.disconnect_ev(ev)
    assert ev.event_generation == gen_before + 1


def test_zero_power_does_not_schedule_or_crash():
    heap = EventQueue()
    ev = EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=0.0)
    ev.update_charging_power(0.0, 0.0, heap)
    assert ev.p_act == 0.0
    assert heap.empty()
    ev.update_s_next(10.0)
    ev.compute_deltaE_power(10.0)
    assert ev.s_next == ev.s_current
    assert ev.deltaE_power == 0.0
