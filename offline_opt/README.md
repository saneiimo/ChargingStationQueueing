# `offline_opt` — offline (clairvoyant) lower-bound MILP

Given every vehicle's arrival time and charging requirement up front, this
package finds the pile-assignment / module-schedule that minimizes total
sojourn time, solved with [gurobipy](https://www.gurobi.com/) (an academic
license is already active in this project's `.venv`). Because it optimizes
with information no deployable (causal) policy could actually have, its
optimum is a valid lower bound on the expected cost of any causal
queue/power policy — FIFO, heuristics, RL — on the same instance.

## Quickstart

```bash
python -m offline_opt.example_toy
```

or from a notebook:

```python
from offline_opt import compute_offline_bound, StationSpec
from toy_demo.scenario import ToyEVSpec, build_evs

specs = [
    ToyEVSpec(id=0, arrival_time=0.0, battery_kwh=50.0, s_i=0.20, s_f=0.80),
    ToyEVSpec(id=1, arrival_time=2.0, battery_kwh=100.0, s_i=0.15, s_f=0.85),
]
station = StationSpec(n_piles=2, n_dispensers=2, n_modules=5, p_module=25.0)

solution = compute_offline_bound(build_evs(specs), station, delta=1.0)
print(solution.status, solution.total_sojourn, solution.mip_gap)
print(solution.per_vehicle)
```

`compute_offline_bound` also accepts a live `ChargingStation` (e.g.
`env.engine.station`) in place of `StationSpec`, and a list of `EV` objects
pulled straight out of `engine.metrics.arrived_evs` after a simulated
episode — so you can compare a heuristic's realized cost against the offline
bound *on the exact same instance* it just ran. See `example_toy.py` for a
full worked example that runs a FIFO simulation and the offline MILP on the
identical station/EV configuration and prints both.

## Formulation

Slots: the horizon `T` is split into `K = ceil(T/delta)` half-open slots,
slot `k = [k*delta, (k+1)*delta)`, `k = 0, ..., K-1`. Every decision is made
at a slot boundary.

| Eq. | Code (`model.py`) | Meaning |
|---|---|---|
| (3) | objective | minimize `sum_j (c_j - a_j)` (total sojourn) |
| (4) | `C4_alpha_mono` | once plugged in, stays plugged in |
| (5) | `C5_sigma_mono` | once finished, stays finished |
| (6) | `C6_sigma_le_alpha` | can't finish before starting |
| (7) | *(implicit)* | `alpha[j,k]` for `k < k_j` is never created — provably 0 |
| (8) | *(omitted)* | finish-by-horizon is **not** enforced; unfinished vehicles are allowed |
| (9) | `C9_one_pile` | each vehicle uses exactly one pile |
| (10) | `C10_occupancy` | occupies exactly one dispenser while plugged-in-and-unfinished |
| (11) | `C11_pile_link` | that dispenser is on the vehicle's assigned pile |
| (12) | `C12_dispenser_cap` | `C` physical dispensers per pile |
| (13) | `C13_module_pool` | `N` power modules per pile |
| (14) | `C14_module_link` | modules only held on the pile the vehicle occupies |
| (15) | `C15_power_from_modules` | power comes from the modules actually held |
| (16) | `C16_completion` | can only be marked finished once fully charged (`x >= W·sigma`) |
| (17) | `C17_energy_cap` | delivered energy at most `W_j` (`x[j,K] <= W_j`) |
| (18) | `C18_flat_cap` | flat part of the BMS curve, `p <= P_max` |
| (19) | `C19_taper_cap` | taper part of the BMS curve, `p <= (R0 - x) / tau` |
| (20) | `C20_no_idle_modules` | at most one held module may go unused (not in the write-up — see "Module efficiency") |

Only variables for `k >= k_j` (a vehicle's release slot) are created —
everything earlier is forced to zero by (7)/(18) anyway, so skipping them
keeps instances with staggered arrivals far smaller than a dense `0..K-1`
grid would be. Without (8), a short horizon no longer makes the model
infeasible: vehicles that cannot finish by `T` simply stay unfinished
(`sigma` stays 0) and contribute sojourn through the end of the horizon.

## `z` is continuous, not binary

Fix `j` and `k`. By (6) the right-hand side of (10) is 0 or 1. If 0, (10)
plus `z >= 0` forces `z[j,m,k] = 0` for every `m`. If 1, (9) picks exactly
one pile `m*` with `y[j,m*]=1`; (11) forces `z[j,m,k]=0` for every `m !=
m*`, and (10) then forces `z[j,m*,k]=1`. Either way `z` lands in `{0,1}` on
its own — it's an exact linearization of `y[j,m]*(alpha[j,k]-sigma[j,k])`,
pinned by (9)-(11) regardless of its declared type. Declaring it continuous
on `[0,1]` (constraint 23) therefore leaves the feasible set and the optimal
value unchanged, while removing `z`'s `J*M*K` variables — the largest binary
block in the formulation — from what the solver branches on.

## Release slot `k_j`

`k_j = ceil(a_j/delta)`: the earliest slot whose start (`k_j*delta`) is at or
after the arrival `a_j`. Since a slot's occupancy decision applies to the
whole `[k*delta, (k+1)*delta)` interval, this is the smallest `k_j` that
never treats the vehicle as present before it actually arrives — a vehicle
arriving exactly on a slot boundary gets zero slack (`k_j*delta == a_j`); one
arriving mid-slot loses at most `delta` waiting out the remainder of the slot
containing its arrival. That's the minimum slack achievable while only
allowing whole-slot occupancy decisions — it never delays a vehicle by a full
extra slot the way `floor(a_j/delta) + 1` would.

## Tie breaking

The primary objective (minimize total sojourn) only cares *when* a vehicle
finishes (and who finishes), never how its power is distributed within its
own charging window. So whenever a vehicle has slack — e.g. it's alone on a
pile with the taper already binding well below the pile's module cap — many
different power profiles tie the true optimum exactly, and Gurobi is free to
return any of them. In practice this can look like a vehicle briefly drawing
less than it could, then making it up later: not a bug, just one of several
equally-valid optimal schedules, and `total_sojourn` is identical across all
of them.

`build_offline_model(..., tie_break=True)` (or `compute_offline_bound(...,
tie_break=True)`) adds a second, lower-priority objective — maximize
`sum(x[j,k])`, the cumulative energy delivered before each slot, already
built for constraints (16)/(19) — which prefers front-loaded profiles among
the tied solutions, so the plotted power trace looks "maxed out whenever
physically possible" instead of arbitrary. Under Gurobi's minimize model
sense this secondary is entered as `-sum(x)` so that minimizing it is
equivalent to maximizing cumulative energy.

The guarantee that this can't change the real answer comes from *how* it's
implemented, not from picking a small-enough weight: it uses Gurobi's native
hierarchical multi-objective mode (`Model.setObjectiveN` with `priority=1`
on the sojourn objective, `priority=0` on the tie-break). Gurobi solves the
priority-1 objective to its true optimum first, then re-optimizes
priority-0 *holding that value fixed* (within `abstol=1e-6`). Distinct
finish-slot patterns change sojourn by multiples of `delta`, far above that
tolerance for typical slot lengths, so the tie-break provably cannot alter
which schedules count as optimal.

Costs roughly 2x solve time (two hierarchical phases); off by default.
Also, `Model.MIPGap` isn't retrievable at all once a second objective is
set (Gurobi raises `AttributeError`), so `OfflineSolution.mip_gap` reports
`nan` when `tie_break=True` — see `extract_solution`'s docstring.

## Module efficiency

Constraint (20) — added, not part of the original write-up:

```
p[j,k] >= Delta * (sum_m n[j,m,k] - 1)      for all j, k
```

Without it, nothing stops the solver from holding modules a vehicle isn't
actually drawing power from: C13/C14/C15 are all *upper* bounds on modules
and power, never a lower bound tying the two together. That's the same kind
of degeneracy "Tie breaking" above describes for power profiles, one level
down — if no other vehicle on the pile needs those modules at that slot, an
allocation that wastes them is exactly as optimal as one that doesn't, so
the solver has no reason to avoid it.

(20) closes that off: with `n` modules held, delivered power must be at
least `(n-1)*Delta` — i.e. every module beyond the first must actually be in
use. Note this is **not** because the objective "already prevents"
inefficient allocation; the objective doesn't see `n` at all except through
constraint (15). It's that inefficient allocation is never *necessary* for
optimality, so ruling it out costs nothing:

For any feasible solution, rebuilding the module assignment as
`n[j,m,k] = ceil(p[j,k] / Delta)` (on whichever pile `y[j,m]=1`) leaves `p`,
`alpha`, `sigma`, `y` untouched — so `total_sojourn` is identical — and
satisfies (20) automatically, since `ceil(x) <= x + 1` for any `x >= 0`.
Because this reconstruction only ever *reduces* modules held per vehicle, it
can't violate (13) (`sum_j n <= B`, a pile-wide upper bound) or (14)
(`n <= B*z`, also an upper bound) either. So every optimal `p`-trajectory
remains achievable under (20); only the redundant module-allocation slack is
pruned. The `-1` is exactly the discrete-module rounding allowance: holding
1 module supports any `p` in `[0, Delta]` (modules aren't divisible), but a
2nd module must actually be used, not just reserved.

## Continuous relaxation bounds

The discrete module count `n[j,m,k]` is what makes
`build_offline_model` an integer program: it can take any of `B+1` values
per vehicle/pile/slot, and the solver has to branch on all of them. That's
the dominant cost of solving it exactly, and it only gets worse with more
piles, vehicles, or a finer `delta`.

`relaxed_model.build_relaxed_model` drops that discreteness: it replaces the
module count with a continuous power variable `q[j,m,k]` (power delivered to
vehicle `j` *by pile `m`*) and the module-count constraints (13)-(15) with a
single aggregate pile-capacity constraint. Everything else — timeline, pile
assignment, dispenser capacity, the BMS curve — is untouched. Call this
`RP(cap)`, parameterized by the pile capacity `cap` it's given.

`RP` is cheaper to solve than the true integer program `IP`, but its
optimum, `RP(N*Delta)`, is not itself a valid stand-in for `IP(N*Delta)` —
it's only a relaxation, so it can (and generally will) do strictly better.
What *is* provably true is a two-sided bracket on total sojourn (both models
minimize `sum_j (c_j - a_j)`):

```
RP(N*Delta).total_sojourn <= IP(N*Delta).total_sojourn <= RP((N-C+1)*Delta).total_sojourn
```

- **Lower bound, `RP(N*Delta)`** — solve the relaxation at the pile's real
  capacity. Any integer-module allocation is also a valid continuous one, so
  `RP` optimizes over a superset of what `IP` can reach; its optimum can only
  be at least as good, i.e. its `total_sojourn` can only be at least as low.
- **Upper bound, `RP((N-C+1)*Delta)`** — solve the same relaxation, but at
  a *reduced* capacity of `(N-C+1)*Delta` (`N` power modules, `C`
  dispensers). Any optimal solution here can be rounded up to whole modules,
  `n[j,m,k] := ceil(q[j,m,k]/Delta)`, without ever exceeding the pile's real
  `N`-module budget or delaying any vehicle. The reason the reduced capacity
  is enough headroom: at most `C` vehicles can share a pile at once
  (constraint 12), so at most `C` of the roundings at any pile/slot are
  simultaneously "in progress," each wasting less than one module — capacity
  `N-C+1` already covers that worst case, leaving exactly `N` modules after
  rounding. Because the rounded solution is feasible for `IP`, its
  `total_sojourn` upper-bounds `IP`'s.

`compute_ip_bounds(evs, station, ...)` takes the same inputs as
`compute_offline_bound` and solves both `RP` instances, returning
`(lower_bound, upper_bound)` as two `OfflineSolution`s. Neither call touches
the integer program at all, so this is a way to get a (looser, but
two-sided and cheap) read on `total_sojourn` when solving `IP` itself would
be too slow.

Not implemented here: the write-up's further step of repairing an `RP(N*Delta)`
solution directly into a tighter, instance-specific feasible `IP` bound (by
locally re-rounding only where a pile/slot actually overflows) rather than
using the uniform worst-case capacity reduction above. That's a valid lower
bound the same way `RP((N-C+1)*Delta)` is an upper bound, just tighter and
without a closed form — out of scope for now.

## Units

The simulator stores battery capacity (`EV.c_b`) as real kWh times `HR2MIN`
(60), chosen so that `power [kW] * time [min]` lands directly in the same
units with no conversion factor (see `config.py`, `models/ev.py`). This
module reuses that convention (`VehicleData.Q`, `.W`, `.R0`) so `delta`
(minutes) and the MILP's power variables combine correctly exactly the way
`EV.p_req` / `EV.energy_needed` do. Use `VehicleData.W_kwh` /
`OfflineSolution.per_vehicle["energy_kwh"]` for human-readable kWh.

`s_th` and `tau` are shared, station-wide parameters (not per-vehicle):
`tau = (1 - s_th) / c_rate`, computed by `taper_time_constant`, defaulting to
`config.S_THRESH` / `config.C_RATE` (`tau = 18` min at those defaults).

## Practical notes

- `delta` and `horizon_minutes` set the number of slots `K`; the model is
  `O(J * M * K)` in variables/constraints. For a handful of vehicles over an
  hour or two at `delta=0.5-1` min this solves in seconds; a full simulated
  day with many vehicles will be much slower — use a coarser `delta`, or a
  `time_limit` / looser `mip_gap` to trade exactness for speed.
- A short `horizon_minutes` no longer makes the model infeasible (constraint
  (8) is omitted): unfinished vehicles stay unfinished and contribute sojourn
  through `T`. Use `default_horizon_minutes` when you want a horizon long
  enough that the optimum can finish everyone. Build-time errors still occur
  if a vehicle's release slot falls *outside* the horizon (arrival after `T`).
- `OfflineSolution.status` is `"OPTIMAL"` only when the solver actually
  closed the gap; check `mip_gap` when you pass a `time_limit` (`nan` when
  `tie_break=True` — see "Tie breaking" above). `objective` is the primary
  Gurobi value (total sojourn in minutes) and should match `total_sojourn`.

See `tests/test_offline_optimization.py` for worked examples, including the
central check that the offline optimum never exceeds a FIFO simulation's
total sojourn time on the same instance.
