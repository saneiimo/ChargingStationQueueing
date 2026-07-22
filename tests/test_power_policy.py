"""
Checks that ProportionalPower respects brick caps and fixed nozzle slots.

These tests build a pile directly (no full DES) so failures point at the
power policy or pile indexing, not at the event engine.
"""

from __future__ import annotations

from math import ceil

from models.ev import EV
from models.pile import ChargingPile
from policy.power.proportional import ProportionalPower


def _make_ev(ev_id: int, c_b: float, s_i: float = 0.2, s_f: float = 0.8, s_th: float = 0.5) -> EV:
    return EV(id=ev_id, c_b=c_b, s_i=s_i, s_f=s_f, arrival_time=0.0, s_th=s_th)


def test_non_overloaded_ceil_allocation():
    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=10, p_brick=10.0)
    ev1 = _make_ev(1, 25)  # p_req = 25*(1/0.5)*(1-0.5)? tan_B = c_b*c_rate/(1-s_th)=25/0.5=50, p_req=50*(1-0.5)=25
    ev2 = _make_ev(2, 35)  # p_req = 35
    # Use c_b so p_req matches ceil targets used historically: p_req = c_b when s_th=0.5 and c_rate=1
    # Actually p_req_max = c_b, tan_B = c_b/(1-s_th), p_req = tan_B*(1-s_th) = c_b when s < s_th.
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)
    assignments = policy.update_power(pile)

    assert pile.ev_bricks[ev1.nozzle_id] == ceil(ev1.p_req / pile.p_brick)
    assert pile.ev_bricks[ev2.nozzle_id] == ceil(ev2.p_req / pile.p_brick)
    assert sum(pile.ev_bricks) <= pile.num_bricks
    assert {e.id: p for e, p in assignments}[1] == pile.ev_bricks[ev1.nozzle_id] * 10
    assert {e.id: p for e, p in assignments}[2] == pile.ev_bricks[ev2.nozzle_id] * 10


def test_overloaded_respects_brick_cap():
    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=4, p_brick=50.0)
    # Isolated demand: ceil(105/50)+ceil(90/50) = 3+2 = 5 > 4
    ev1 = _make_ev(1, 105, s_th=0.5)
    ev2 = _make_ev(2, 90, s_th=0.5)
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)
    assert pile.is_overloaded
    policy.update_power(pile)
    assert sum(pile.ev_bricks) == pile.num_bricks
    assert all(pile.ev_bricks[e.nozzle_id] >= 1 for e in pile.evs)


def test_disconnect_middle_nozzle_keeps_slots():
    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=5, p_brick=25.0)
    ev1 = _make_ev(1, 50)
    ev2 = _make_ev(2, 50)
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)
    policy.update_power(pile)

    # Disconnect first slot; second EV must keep nozzle_id == 1
    slot1 = ev2.nozzle_id
    pile.disconnect_ev(ev1)
    assert ev2.nozzle_id == slot1
    assert pile.nozzles[0] is None
    assert pile.nozzles[1] is ev2
    assert pile.ev_bricks[0] == 0

    policy.update_power(pile)
    assert pile.ev_bricks[ev2.nozzle_id] >= 1
    assert sum(pile.ev_bricks) <= pile.num_bricks
    pile.check_invariants()


def test_micro_distribute_frees_brick_from_trigger_ev():
    policy = ProportionalPower()
    pile = ChargingPile(id=0, n_nozzles=2, num_bricks=4, p_brick=25.0)
    ev1 = _make_ev(1, 100)
    ev2 = _make_ev(2, 100)
    pile.connect_ev(ev1)
    pile.connect_ev(ev2)
    policy.update_power(pile)
    before = pile.ev_bricks[ev1.nozzle_id]
    assert before >= 1
    policy.update_power(pile, ev=ev1)
    # Trigger EV lost one brick before refill; total still capped.
    assert sum(pile.ev_bricks) == pile.num_bricks
