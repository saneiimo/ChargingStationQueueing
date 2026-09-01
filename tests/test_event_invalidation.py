"""
Checks that bumping event_generation kills stale DEPARTURE/CHARGE_CHANGE events,
and that zero power does not schedule or divide by zero.

Each test prints what it is checking and the numbers it sees.

Run from the repo root (prints need -s):

    python -m pytest tests/test_event_invalidation.py -s -v

Or run this file directly:

    python tests/test_event_invalidation.py

In a notebook:

    !python -m pytest tests/test_event_invalidation.py -s -v
"""

from __future__ import annotations

from models.ev import EV
from models.pile import ChargingPile
from simulation.event import EventQueue, EventType
from policy.power.proportional import ProportionalPower


def test_event_generation_invalidates_both_event_types():
    """A new schedule must invalidate an older DEPARTURE even if the new type differs."""
    print("\n=== test_event_generation_invalidates_both_event_types ===")
    print(
        "Intent: after redistributing power, the old DEPARTURE event_id must no "
        "longer match EV.event_generation (stale events must be ignored)."
    )

    heap = EventQueue()
    pile = ChargingPile(id=0, n_connectors=2, num_modules=4, p_module=25.0)
    ev = EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=0.0)
    pile.connect_ev(ev)

    ProportionalPower().update_power(pile)
    ev.update_charging_power(pile.ev_modules[ev.connector_id] * pile.p_module, 0.0, heap)
    assert heap.heap
    first = heap.heap[0]
    print(
        f"  First schedule: type={first.event_type.name}, "
        f"event_id={first.event_id}, generation={ev.event_generation}"
    )
    assert first.event_type == EventType.DEPARTURE
    old_id = first.event_id

    ev2 = EV(id=1, c_b=150.0, s_i=0.2, s_f=0.9, arrival_time=0.0)
    pile.connect_ev(ev2)
    ProportionalPower().update_power(pile)
    if pile.is_overloaded and pile.ev_modules[ev.connector_id] <= 1:
        pile.ev_modules[ev.connector_id] = 2
        pile.ev_modules[ev2.connector_id] = max(1, pile.num_modules - 2)
        print(
            f"  Forced modules for CHARGE_CHANGE eligibility: "
            f"ev_modules={pile.ev_modules}, overloaded={pile.is_overloaded}"
        )

    ev.update_charging_power(pile.ev_modules[ev.connector_id] * pile.p_module, 0.0, heap)
    print(
        f"  After reschedule: generation={ev.event_generation}, "
        f"old DEPARTURE id={old_id}, heap_size={len(heap.heap)}"
    )

    assert old_id != ev.event_generation
    assert first.event_id != ev.event_generation
    print("  PASS (old event is stale)")


def test_disconnect_invalidates_pending_events():
    """Unplugging an EV must bump its generation so leftover timed events die."""
    print("\n=== test_disconnect_invalidates_pending_events ===")
    print(
        "Intent: disconnect_ev should invalidate any pending DEPARTURE/CHARGE_CHANGE "
        "for that EV by incrementing event_generation."
    )

    heap = EventQueue()
    pile = ChargingPile(id=0, n_connectors=2, num_modules=5, p_module=25.0)
    ev = EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=0.0)
    pile.connect_ev(ev)
    ProportionalPower().update_power(pile)
    ev.update_charging_power(100.0, 0.0, heap)
    gen_before = ev.event_generation
    print(f"  Before disconnect: generation={gen_before}, heap_size={len(heap.heap)}")

    pile.disconnect_ev(ev)
    print(f"  After disconnect: generation={ev.event_generation}")

    assert ev.event_generation == gen_before + 1
    print("  PASS")


def test_zero_power_does_not_schedule_or_crash():
    """p_act=0 must not push an event or blow up SoC/energy helpers."""
    print("\n=== test_zero_power_does_not_schedule_or_crash ===")
    print(
        "Intent: allotting 0 kW should leave the heap empty and keep SoC/energy "
        "unchanged over a projection step."
    )

    heap = EventQueue()
    ev = EV(id=0, c_b=50.0, s_i=0.2, s_f=0.8, arrival_time=0.0)
    ev.update_charging_power(0.0, 0.0, heap)
    print(
        f"  After update_charging_power(0): p_act={ev.p_act}, heap_empty={heap.empty()}"
    )

    assert ev.p_act == 0.0
    assert heap.empty()

    s_before = ev.s_current
    ev.update_s_next(10.0)
    ev.compute_deltaE_power(10.0)
    print(
        f"  After 10-min projection: s_next={ev.s_next:.4f} "
        f"(was {s_before:.4f}), deltaE_power={ev.deltaE_power}"
    )

    assert ev.s_next == ev.s_current
    assert ev.deltaE_power == 0.0
    print("  PASS")


if __name__ == "__main__":
    print("Running test_event_invalidation.py (direct mode)")
    test_event_generation_invalidates_both_event_types()
    test_disconnect_invalidates_pending_events()
    test_zero_power_does_not_schedule_or_crash()
    print("\nAll tests in test_event_invalidation.py finished.")
