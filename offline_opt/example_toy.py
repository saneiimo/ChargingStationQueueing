"""
Toy example: one station/EV configuration, solved two ways.

1. Simulate it with the DES (FIFO queue policy, proportional power sharing).
2. Solve the same configuration with the offline MILP in this package.

Because the offline model optimizes with full knowledge of every arrival,
its total sojourn time should never exceed the (causal) FIFO simulation's --
that inequality is the whole point of computing it, and this script prints
both so you can see the gap directly.

Run: python -m offline_opt.example_toy
"""

from __future__ import annotations

from offline_opt import StationSpec, compute_offline_bound
from toy_demo.runner import run_toy_episode
from toy_demo.scenario import ToyEVSpec, ToyStationSpec, build_evs

# --- The one configuration, shared by both solves --------------------------

STATION = ToyStationSpec(n_piles=2, n_dispensers=2, n_modules=5, p_module=25.0)

EV_SPECS = [
    ToyEVSpec(id=0, arrival_time=0.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
    ToyEVSpec(id=1, arrival_time=2.0, battery_kwh=100.0, s_i=0.15, s_f=0.85),
    ToyEVSpec(id=2, arrival_time=8.0, battery_kwh=50.0, s_i=0.25, s_f=0.75),
    ToyEVSpec(id=3, arrival_time=12.0, battery_kwh=150.0, s_i=0.10, s_f=0.80),
]


def main() -> None:
    # 1. Simulate: FIFO queue policy, default (proportional) power sharing.
    env = run_toy_episode(STATION, EV_SPECS, seed=0, policy_seed=1)
    fifo_sojourns = {
        ev.id: ev.departure_time - ev.arrival_time
        for ev in env.engine.metrics.finished_evs
    }
    fifo_total = sum(fifo_sojourns.values())

    print("=== FIFO simulation ===")
    for vid in sorted(fifo_sojourns):
        print(f"  vehicle {vid}: sojourn = {fifo_sojourns[vid]:.2f} min")
    print(f"  total sojourn = {fifo_total:.2f} min\n")

    # 2. Solve the same configuration offline (fresh EV objects; build_evs is
    # a pure function of EV_SPECS, so this instance matches the simulation's).
    offline_station = StationSpec(
        n_piles=STATION.n_piles,
        n_dispensers=STATION.n_dispensers,
        n_modules=STATION.n_modules,
        p_module=STATION.p_module,
    )
    solution = compute_offline_bound(
        build_evs(EV_SPECS), offline_station, delta=1.0, mip_gap=1e-4
    )

    print(f"=== Offline MILP (status={solution.status}, gap={solution.mip_gap:.2%}) ===")
    print(solution.per_vehicle.to_string(index=False))
    print(f"  total sojourn = {solution.total_sojourn:.2f} min\n")

    print(
        f"OPT ({solution.total_sojourn:.2f}) <= FIFO ({fifo_total:.2f}): "
        f"{solution.total_sojourn <= fifo_total}"
    )


if __name__ == "__main__":
    main()
