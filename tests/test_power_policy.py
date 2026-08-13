"""
Checks that ProportionalPower respects brick caps and fixed nozzle slots.

These tests build a pile directly (no full DES) so failures point at the
power policy or pile indexing, not at the event engine.

Each test prints what it is checking and the numbers it sees.

Run from the repo root (prints need -s):

    python -m pytest tests/test_power_policy.py -s -v

Or run this file directly:

    python tests/test_power_policy.py

In a notebook:

    !python -m pytest tests/test_power_policy.py -s -v
"""

from __future__ import annotations

from math import ceil

from models.ev import EV
from models.pile import ChargingPile
from policy.power.proportional import ProportionalPower
from policy.power.static import StaticPower
from simulation.event import EventType


def _make_ev(
    ev_id: int,
    c_b: float,
    s_i: float = 0.2,
    s_f: float = 0.8,
    s_th: float = 0.5,
    c_rate: float | None = None,
) -> EV:
    kwargs = dict(id=ev_id, c_b=c_b, s_i=s_i, s_f=s_f, arrival_time=0.0, s_th=s_th)
    if c_rate is not None:
        kwargs["c_rate"] = c_rate
    return EV(**kwargs)


def test_non_overloaded_ceil_allocation():
    """When demand fits, each EV gets ceil(p_req / p_brick) bricks."""
    print("\n=== test_non_overloaded_ceil_allocation ===")
    print(
        "Intent: with enough bricks, allotment should be ceil of each EV's "
        "isolated request (no sharing fight)."
    )

    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=10, p_brick=10.0)
    # With s_th=0.5 and c_rate=1, p_req equals c_b while SoC < s_th.
    ev1 = _make_ev(1, 25)
    ev2 = _make_ev(2, 35)
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)
    print(
        f"  Setup: p_brick={pile.p_brick}, num_bricks={pile.num_bricks}, "
        f"EV1 p_req={ev1.p_req:.1f}, EV2 p_req={ev2.p_req:.1f}"
    )

    assignments = policy.update_power(pile)
    power_by_id = {e.id: p for e, p in assignments}
    expected_1 = ceil(ev1.p_req / pile.p_brick)
    expected_2 = ceil(ev2.p_req / pile.p_brick)

    print(
        f"  Result: ev_bricks={pile.ev_bricks}, "
        f"expected=[{expected_1}, {expected_2}], "
        f"powers={power_by_id}, overloaded={pile.is_overloaded}"
    )

    assert pile.ev_bricks[ev1.nozzle_id] == expected_1
    assert pile.ev_bricks[ev2.nozzle_id] == expected_2
    assert sum(pile.ev_bricks) <= pile.num_bricks
    assert power_by_id[1] == pile.ev_bricks[ev1.nozzle_id] * 10
    assert power_by_id[2] == pile.ev_bricks[ev2.nozzle_id] * 10
    print("  PASS")


def test_overloaded_respects_brick_cap():
    """When isolated demand exceeds bricks, total allotment must equal num_bricks."""
    print("\n=== test_overloaded_respects_brick_cap ===")
    print(
        "Intent: if ceil requests sum above the brick pool, policy must still "
        "assign exactly num_bricks and give each EV at least one."
    )

    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=4, p_brick=50.0)
    # Isolated demand: ceil(105/50)+ceil(90/50) = 3+2 = 5 > 4
    ev1 = _make_ev(1, 105, s_th=0.5)
    ev2 = _make_ev(2, 90, s_th=0.5)
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)

    isolated = [
        ceil(ev1.p_req / pile.p_brick),
        ceil(ev2.p_req / pile.p_brick),
    ]
    print(
        f"  Setup: p_reqs=[{ev1.p_req:.1f}, {ev2.p_req:.1f}], "
        f"isolated ceils={isolated}, pool={pile.num_bricks}, "
        f"is_overloaded={pile.is_overloaded}"
    )
    assert pile.is_overloaded

    policy.update_power(pile)
    print(
        f"  Result: ev_bricks={pile.ev_bricks}, sum={sum(pile.ev_bricks)}, "
        f"each >= 1? {all(pile.ev_bricks[e.nozzle_id] >= 1 for e in pile.evs)}"
    )

    assert sum(pile.ev_bricks) == pile.num_bricks
    assert all(pile.ev_bricks[e.nozzle_id] >= 1 for e in pile.evs)
    print("  PASS")


def test_disconnect_middle_nozzle_keeps_slots():
    """Removing the EV in slot 0 must not renumber the EV still in slot 1."""
    print("\n=== test_disconnect_middle_nozzle_keeps_slots ===")
    print(
        "Intent: nozzle indices are fixed slots. After unplugging EV1 from slot 0, "
        "EV2 should stay on nozzle_id=1 and brick vector stays aligned."
    )

    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=5, p_brick=25.0)
    ev1 = _make_ev(1, 50)
    ev2 = _make_ev(2, 50)
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)
    policy.update_power(pile)
    print(
        f"  Before disconnect: nozzles="
        f"{[None if e is None else e.id for e in pile.nozzles]}, "
        f"ev2.nozzle_id={ev2.nozzle_id}, ev_bricks={pile.ev_bricks}"
    )

    slot1 = ev2.nozzle_id
    pile.disconnect_ev(ev1)
    print(
        f"  After disconnect EV1: nozzles="
        f"{[None if e is None else e.id for e in pile.nozzles]}, "
        f"ev2.nozzle_id={ev2.nozzle_id}, ev_bricks={pile.ev_bricks}"
    )

    assert ev2.nozzle_id == slot1
    assert pile.nozzles[0] is None
    assert pile.nozzles[1] is ev2
    assert pile.ev_bricks[0] == 0

    policy.update_power(pile)
    print(
        f"  After redistribute: ev_bricks={pile.ev_bricks}, "
        f"EV2 bricks={pile.ev_bricks[ev2.nozzle_id]}"
    )
    assert pile.ev_bricks[ev2.nozzle_id] >= 1
    assert sum(pile.ev_bricks) <= pile.num_bricks
    pile.check_invariants()
    print("  PASS")


def test_micro_distribute_frees_brick_from_trigger_ev():
    """CHARGE_CHANGE path: free one brick from the trigger EV, then refill leftovers."""
    print("\n=== test_micro_distribute_frees_brick_from_trigger_ev ===")
    print(
        "Intent: update_power(pile, ev=EV1) should drop one brick from EV1 then "
        "reassign any free bricks; total must stay at num_bricks."
    )

    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=4, p_brick=25.0)
    ev1 = _make_ev(1, 100)
    ev2 = _make_ev(2, 100)
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)
    policy.update_power(pile)
    before = list(pile.ev_bricks)
    before_ev1 = pile.ev_bricks[ev1.nozzle_id]
    print(f"  Before micro: ev_bricks={before}, EV1 bricks={before_ev1}")
    assert before_ev1 >= 1

    policy.update_power(pile, ev=ev1)
    after = list(pile.ev_bricks)
    print(f"  After micro (trigger=EV1): ev_bricks={after}, sum={sum(after)}")

    assert sum(pile.ev_bricks) == pile.num_bricks
    print("  PASS")


def test_static_equal_split_with_leftovers_by_unmet_request():
    """
    Static: equal base bricks, leftovers to largest unmet p_req.

    Matches the documented example: 5 bricks, p_brick=25, requests
    75 / 250 / 200 kW -> base 1 each, leftovers to 250 then 200 -> [1, 2, 2].
    """
    print("\n=== test_static_equal_split_with_leftovers_by_unmet_request ===")
    print(
        "Intent: with 5 bricks and three EVs at 75/250/200 kW, Static should "
        "give 1 each then hand leftovers by unmet request -> bricks [1, 2, 2]."
    )

    policy = StaticPower()
    pile = ChargingPile(id=0, n_nozzles=3, num_bricks=5, p_brick=25.0)
    # c_rate=1 so flat-region p_req equals c_b (kW) for the documented numbers.
    ev_low = _make_ev(0, 75.0, s_i=0.2, s_th=0.6, c_rate=1.0)
    ev_hi = _make_ev(1, 250.0, s_i=0.2, s_th=0.6, c_rate=1.0)
    ev_mid = _make_ev(2, 200.0, s_i=0.2, s_th=0.6, c_rate=1.0)
    pile.connect_ev(ev_low)
    pile.connect_ev(ev_hi)
    pile.connect_ev(ev_mid)

    print(
        f"  Setup: p_reqs=[{ev_low.p_req:.1f}, {ev_hi.p_req:.1f}, {ev_mid.p_req:.1f}], "
        f"pool={pile.num_bricks}, p_brick={pile.p_brick}"
    )

    policy.update_power(pile)
    bricks = [
        pile.ev_bricks[ev_low.nozzle_id],
        pile.ev_bricks[ev_hi.nozzle_id],
        pile.ev_bricks[ev_mid.nozzle_id],
    ]
    print(f"  Result: ev_bricks by EV order={bricks}, sum={sum(bricks)}")

    assert bricks == [1, 2, 2]
    assert sum(pile.ev_bricks) == pile.num_bricks
    print("  PASS")


def test_static_single_ev_gets_full_pool():
    """One plugged EV receives the entire brick pool under equal split."""
    print("\n=== test_static_single_ev_gets_full_pool ===")
    print("Intent: with one EV, num_bricks // 1 assigns the full pool to that EV.")

    policy = StaticPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=5, p_brick=25.0)
    ev = _make_ev(1, 100.0, s_i=0.2, s_th=0.6, c_rate=1.0)
    pile.connect_ev(ev)
    policy.update_power(pile)

    print(f"  Result: ev_bricks={pile.ev_bricks}")
    assert pile.ev_bricks[ev.nozzle_id] == pile.num_bricks
    assert sum(pile.ev_bricks) == pile.num_bricks
    print("  PASS")


def test_underuse_reallocation_flag_and_charge_change_gating():
    """
    Prop opts into CHARGE_CHANGE; Static opts out so next_state stays DEPARTURE.

    Without this gate, Static rebuilds the equal split on underuse, hands the
    brick back, and the DES can loop forever on CHARGE_CHANGE.
    """
    print("\n=== test_underuse_reallocation_flag_and_charge_change_gating ===")
    print(
        "Intent: Static.supports_underuse_reallocation is False and schedules "
        "DEPARTURE even when overloaded with n_bricks>1; Prop can schedule "
        "CHARGE_CHANGE."
    )

    assert ProportionalPower.supports_underuse_reallocation is True
    assert StaticPower.supports_underuse_reallocation is False

    class _StubStation:
        def __init__(self, policy):
            self.power_policy = policy
            self.current_time = 0.0
            self.next_time = 0.0

    # Shared overloaded setup: two EVs, 4 bricks, p_brick=25, high p_req.
    def _overloaded_pile(policy):
        station = _StubStation(policy)
        pile = ChargingPile(id=0, n_nozzles=2, num_bricks=4, p_brick=25.0, station=station)
        ev1 = _make_ev(1, 200.0, s_i=0.2, s_f=0.95, s_th=0.6, c_rate=1.0)
        ev2 = _make_ev(2, 200.0, s_i=0.2, s_f=0.95, s_th=0.6, c_rate=1.0)
        pile.connect_ev(ev1)
        pile.connect_ev(ev2)
        policy.update_power(pile)
        # Ensure CHARGE_CHANGE eligibility shape: >1 brick on an overloaded pile.
        if pile.ev_bricks[ev1.nozzle_id] <= 1:
            pile.ev_bricks[ev1.nozzle_id] = 2
            pile.ev_bricks[ev2.nozzle_id] = pile.num_bricks - 2
        ev1.p_act = pile.ev_bricks[ev1.nozzle_id] * pile.p_brick
        return pile, ev1

    pile_s, ev_s = _overloaded_pile(StaticPower())
    assert pile_s.is_overloaded
    assert ev_s.n_bricks > 1
    typ_s = ev_s.event_type_next_candidate
    print(f"  Static: overloaded={pile_s.is_overloaded}, bricks={ev_s.n_bricks}, next={typ_s.name}")
    assert typ_s == EventType.DEPARTURE

    pile_p, ev_p = _overloaded_pile(ProportionalPower())
    assert pile_p.is_overloaded
    assert ev_p.n_bricks > 1
    typ_p = ev_p.event_type_next_candidate
    print(f"  Prop: overloaded={pile_p.is_overloaded}, bricks={ev_p.n_bricks}, next={typ_p.name}")
    assert typ_p == EventType.CHARGE_CHANGE
    print("  PASS")


if __name__ == "__main__":
    print("Running test_power_policy.py (direct mode)")
    test_non_overloaded_ceil_allocation()
    test_overloaded_respects_brick_cap()
    test_disconnect_middle_nozzle_keeps_slots()
    test_micro_distribute_frees_brick_from_trigger_ev()
    test_static_equal_split_with_leftovers_by_unmet_request()
    test_static_single_ev_gets_full_pool()
    test_underuse_reallocation_flag_and_charge_change_gating()
    print("\nAll tests in test_power_policy.py finished.")
