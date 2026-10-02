# `offline_cl_opt` — connector-lane offline MILP

Implements **Section 4 ("The optimisation model")** of the connector-lane
formulation for offline optimal scheduling of an EV charging station
(`connector_lane_model.html`): constraints (2)-(19), plus one strengthening
from Section 9 (preprocessing) that the source document treats as part of the
model proper — see "Preprocessing" below. The objective is **total sojourn in
minutes**, `sum_j (delta*D_j - a_j)` over `objective_cohorts` — the
document's objective (1), `sum_j D_j`, scaled by `delta` and shifted by the
constant `sum_j a_j`. Same optimal schedules, same ranking of every feasible
schedule; but `ObjVal`/`ObjBound`/`solution.objective` read directly as total
sojourn, and Gurobi's relative `MIPGap` is a fraction of total sojourn rather
than of a sum inflated by every vehicle's arrival time. Attainable objective
values are `delta` apart (`sojourn_objective_round_up`).
Given every vehicle's arrival time and charging requirement up front, finds
the lane assignment and module-routing schedule that minimizes total
sojourn time, exactly as in `offline_opt` — but via a structurally
different formulation. See "How this differs from `offline_opt`" below for
when to reach for which.

## Quickstart

```python
from offline_cl_opt import StationSpec, VehicleData, build_cl_model, solve_cl_model
from offline_cl_opt.solution import extract_solution

station = StationSpec(n_piles=2, n_connectors=2, n_modules=5, p_module=25.0)
vehicles = [
    VehicleData(id=0, a=0.0, Q=50.0, s_i=0.20, s_f=0.80, s_th=0.4, p_max=100.0),
    VehicleData(id=1, a=2.0, Q=100.0, s_i=0.15, s_f=0.85, s_th=0.4, p_max=100.0),
]

cl_model = build_cl_model(vehicles, station, delta=1.0, horizon_minutes=120.0)
solve_cl_model(cl_model, mip_gap=1e-4)
solution = extract_solution(cl_model)

print(solution.mean_sojourn, solution.per_vehicle)
```

`vehicles_from_evs` / `VehicleData.from_ev` / `StationSpec.from_station`
build these from this repo's simulator objects (`models.ev.EV`,
`models.station.ChargingStation`), the same convenience `offline_opt`
offers. See `cl_model.ipynb` (repo root) for a full worked example against
a FIFO simulation.

For the exact model (`build_cl_model`/`solve_cl_model`), solving is a
single call as above. For Section 8's adaptive module-integrality
procedure — usually much faster, and *exactly* as optimal, not merely a
good heuristic (see "Adaptive module integrality" below) — use
`solve_cl_model_adaptive` instead, which builds and solves internally:

```python
from offline_cl_opt import solve_cl_model_adaptive
from offline_cl_opt.solution import extract_solution

result = solve_cl_model_adaptive(vehicles, station, delta=1.0, horizon_minutes=120.0)
solution = extract_solution(result.cl_model)
print(result.converged, result.iterations, solution.mean_sojourn)
```

Pass `progress=True` for a one-line-per-iteration summary of the adaptive
loop itself (objective, gap, solve time, pile-slots promoted) — independent
of `verbose`, which instead toggles each individual Gurobi solve's own
console log.

`solve_cl_model`/`solve_cl_model_adaptive` also expose:

- `presolve`/`pre_passes` (Gurobi's own `Presolve`/`PrePasses` parameters)
  for the case where presolve itself, not the branch-and-bound search
  after it, dominates solve time on a large instance;
- `cutoff` (Section 9.2/11's recommended technique): pass a known feasible
  schedule's objective value as Gurobi's own `Cutoff` parameter, so the
  solver can prune a node the moment its own bound reaches that value
  without needing to find a matching incumbent first — see "Preprocessing"
  below for where to get that value and `solve_cl_model`'s own docstring
  for the exact semantics (and what happens if it wasn't actually
  achievable).

See each function's own docstring for when these are worth trying and how
to tell from Gurobi's log.

## Formulation

Slots: horizon `T` split into `K = ceil(T/delta)` half-open slots, slot
`k = [k*delta, (k+1)*delta)`, `k = 0, ..., K-1`. `h = delta/60` is the slot
length in hours (Section 3.2) — power (kW) and energy (kWh) meet only
through `h`, never `delta` directly.

| Eq. | Code (`model.py`) | Meaning |
|---|---|---|
| (2) | *(implicit)* | `u[j,k]` for `k < k_j` is never created — provably 0 |
| (3)-(5) | `C3_eta_lb`, `C4_eta_le_u`, `C5_eta_le_1mprev` | pin `eta[j,k]` to the plug-in (rising) edge indicator |
| (6) | `C6_one_start` | at most one plug-in per vehicle |
| (7)-(9) | `S[j]`, `D[j]` (dependent expressions, not variables) | served indicator, start slot, departure boundary |
| (10) | `C10_one_lane` | each vehicle gets exactly one lane |
| (11)-(12) | `C11_seq`, `C12_seq` | two vehicles sharing a lane never overlap in time |
| (13) | `C13_power_cap` | power only while connected, capped at `P_bar_j = min(P_max_j, N*Delta)` |
| (14) | `C14_power_from_modules` | power capped by modules routed to the vehicle's own lane |
| (15) | `C15_module_pool` | a pile routes at most `N` modules total (`module_pool_cap` when tightened) |
| (16) | `C16_station_power` | station-wide power bound (valid inequality, tightens the LP relaxation) |
| (17) | `C17_energy_recursion` | `x[j,k+1] = x[j,k] + h*p[j,k]` — a real variable, not an inline sum (see "Why `x` is a real variable" below) |
| (18) | `C18_taper_cap` | BMS taper cap, in terms of the *effective discrete* time constant `tau^delta_j` |
| (19) | `C19_departure_rule` | can't unplug before receiving the full `W_j` — the raw occupancy difference, no extra indicator variable needed |
| (24) | `C24_departure_lower_bound` | valid lower bound on `D_j` from `E_j` and the occupancy count (Section 9.3; on by default, `bound_departures`) |
| (25)-(26) | `C25_pile_symmetry`, `C26_connector_symmetry` | optional pile/connector symmetry breaking (`break_symmetry`) |

Every vehicle's variables/rows are generated over the *whole* `[k_j, K)` —
only the left (arrival) edge is used to narrow anything; see
"Preprocessing" below for why there's no per-vehicle right edge.

`eta` is declared continuous `[0,1]`, not binary — (3)-(5) pin it to the
exact rising-edge indicator regardless of declared type (see the source
document's Proposition 2), so leaving it continuous drops `J*K` variables
from branching for free, same spirit as `z` in `offline_opt` (see that
package's README, "`z` is continuous, not binary"). There is no equivalent
falling-edge variable — see below.

`S[j]`, `D[j]` (start slot, departure boundary) are Gurobi `LinExpr`
objects built from `eta`/`u`, not decision variables — they're already
exactly pinned by (7)-(9), so materializing them as Vars would only add
branching surface for nothing.

### No `theta` variable

An earlier revision of both the source document and this module used a
falling-edge indicator `theta[j,k]` (mirroring the rising-edge `eta[j,k]`)
to state the departure rule. The current document's Section 5.6 proves this
was never necessary: because the delivered-energy variable `x[j,k]` is
already non-negative and upper-bounded by `W_j`, the raw difference
`x[j,k] >= W_j*(u[j,k-1] - u[j,k])` (19) cuts off exactly the same points,
in the linear relaxation as well as at integer points, as the version
written with an explicit indicator. Dropping `theta` removes `O(J*K)`
variables and three constraint families for no loss of correctness.

### Why `x` is a real variable

The delivered-energy state `x[j,k]` (energy delivered to vehicle `j`
strictly before slot `k`) is an explicit Gurobi variable, governed by the
two-term recursion (17), `x[j,k+1] = x[j,k] + h*p[j,k]` — *not* accumulated
as a growing Python-side linear expression substituted directly into
(18)/(19). Section 7.1 of the source document explains why this matters:
writing the cumulative sum out inline gives slot `k`'s row `O(k)` nonzeros,
`O(K^2)` per vehicle in total — "enough to make presolve alone run for
minutes" on a realistic instance. An earlier revision of this module did
exactly that (an incrementally-grown `LinExpr` folded straight into the
taper-cap and departure rows), and it is very likely what caused presolve
times of several minutes observed on real notebook runs before this fix.
Carrying `x` as a variable keeps every row of (17)/(18)/(19) at a constant
few nonzeros regardless of `k` — verified directly: nonzero count now
scales linearly with `K` (confirmed empirically: ~25 nonzeros per slot,
flat, from `K=100` through `K=800`), where the old approach scaled
quadratically. The vehicle's total energy requirement is enforced by `x`'s
own upper bound (`W_j`, a per-variable bound, cheaper than a row) rather
than by a separate summed constraint — `offline_opt`'s `x[j,k]` uses the
same trick (see that package's README).

## Censoring and the horizon (Section 6.3)

There is no hard "must finish" variant of this model. The objective is
always the **censored** one: a vehicle that cannot finish within the
horizon departs at boundary `K`, i.e. contributes a sojourn of
`delta*K - a_j` (the model may even prefer
leaving it unserved entirely if that frees a lane for someone else). An
earlier revision of the source document had a `complete_service` variant
that forced full service via a hard constraint; the current document
removes it in favor of a simpler practice — choose `horizon_minutes`
generous enough that censoring doesn't bind at the optimum (e.g. longer
than a simple FIFO simulation's makespan on the same arrivals), then
**verify that afterwards** rather than enforcing it structurally:
`extract_solution`'s `per_vehicle` reports both `departure_slot` and
`energy_kwh`/`energy_required_kwh` per vehicle, so checking every vehicle
has `departure_slot < K` and `energy_kwh == energy_required_kwh` confirms
the horizon was generous enough for this instance.

## Adaptive module integrality (`adaptive.py`, current Section 8)

`r`, the whole-module-routing variable, is the only part of the model
that's genuinely combinatorial once everything else is fixed — and it
usually resolves itself: either a lane holds one vehicle (any routing that
covers its power works), or the powers on a pile leave enough headroom
that rounding each connector's requirement up to whole modules still fits.
Declaring all `M*C*K` of them integer up front (the default,
`relax_modules=False`) makes the solver branch on something that, in most
places, was never actually in question.

`solve_cl_model_adaptive(vehicles, station, delta, horizon_minutes, ...)`
implements the alternative:

1. Solve `CL_R` — the exact model with `r` relaxed to continuous
   (`build_cl_model(..., relax_modules=True)`; `u`, `y`, `b` stay binary,
   so `CL_R` is still a MIP, just without the module block's
   combinatorics).
2. `rounding_test_failures` checks every pile-slot: does
   `sum_c ceil(p_occupant/Delta)` (the *real* whole-module requirement)
   still fit the pile's *real* module count `N`? (This is exactly
   `offline_opt`'s `n[j,m,k]=ceil(p[j,k]/Delta)` reconstruction argument
   for its own constraint (20), applied here per pile-slot instead of
   globally.)
3. If every pile-slot passes: stop. `rounded_module_routing` gives the
   Lemma's `hat_r = ceil(p/Delta)` directly — a genuine `(14)`-`(15)`-valid
   integer routing, proven exactly optimal for the fully-integer exact
   model (not just a good answer) by the source document's Proposition 3.
4. Otherwise: promote the failing pile-slots' `r[m,c,k]` (all `C`
   connectors of that pile-slot) from continuous to integer in place —
   `cl_model.r[m,c,k].VType = GRB.INTEGER` on the live model, never a
   rebuild — and go to 2.

`extract_solution` never reads `r` at all (only `p`/`u`/`y`), so the final
schedule's reported sojourns/energies are correct regardless of whether
every `r` ended up literally integer-typed in Gurobi's eyes; call
`rounded_module_routing` if you want to inspect or plot the actual module
routing.

**MIP start.** The first solve of `CL_R` needs a starting point, from one
of two sources:

- **The conservative shortcut** (default, `warm_start_from_conservative=True`):
  `conservative_feasible_solution` — Section 8.4's shortcut, solving `CL_R`
  with the pile budget tightened from `N` to `N-C+1` (21). Rounding up at
  most `C` positive numbers can't inflate their sum by `C` or more, so this
  solution's power values are *always* roundable into a feasible schedule
  for the exact model, no repair needed — a cheap, guaranteed-feasible
  incumbent, at the cost of one extra MILP solve.
- **A real simulation's own output** (`warm_start_evs=<list of EV objects>`,
  e.g. `env.engine.metrics.finished_evs`): `_seed_values_from_evs`
  reconstructs a discretized schedule directly from each vehicle's own
  recorded pile/connector, timing, and power trace (`EV.pile_tracker`,
  `.connector_id_tracker`, `.service_start_time`, `.departure_time`,
  `.charge_trace`) — no MILP solve needed to build it at all, only as good
  a starting point as the simulation itself was. `x`'s `.Start` values are
  reconstructed from the same seeded `p` trajectory via the model's own
  recursion (17). Takes priority over `warm_start_from_conservative` when
  given.

As the document notes, a *previous adaptive iteration's* own relaxed
optimum is deliberately **not** used as a later MIP start (from either
source): it's typically infeasible exactly at the pile-slots just
promoted, so it would mislead rather than help.

**Objective cutoff.** `solve_cl_model_adaptive(..., cutoff=UB)` applies a
known upper bound to *every* iteration, not just the final exact solve.
This is always safe even though most iterations solve a relaxation `M(I_t)`
rather than the exact model: `z(I_t) <= z*` for every `I_t` (Section 8.2's
own monotonicity argument), so any valid upper bound on the true optimum
`z*` is automatically also a valid cutoff for every intermediate
relaxation solved along the way.

**Symmetry-aware seeding.** A real simulation has no reason to already
label its piles/connectors in the canonical order `break_symmetry` enforces
((25)-(26) — see "Symmetry breaking" below) — nothing about the DES cares
which pile is called "0". So whenever `break_symmetry=True`,
`_seed_values_from_evs` first relabels the raw simulated assignment into
that canonical order (`_relabel_lanes_for_symmetry`: piles, then
connectors within each pile, labelled in order of the lowest-id vehicle
occupying them — the exact same constructive relabelling the
symmetry-breaking correctness proof uses) *before* setting `.Start` values.
Without this, Gurobi would likely find the raw assignment infeasible for
(25)-(26) and simply discard the seed rather than benefit from it. The
conservative shortcut doesn't need this step: it's built with the same
`break_symmetry` value as the target model, so its own solve already
respects the canonical order.

**Basis warm start.** Promoting a pile-slot changes only a few variables'
declared type — never the constraint matrix, bounds, or objective — so the
LP relaxation at the root is the literal same LP before and after.
Gurobi reuses the previous basis across successive `optimize()` calls on
the same live `Model` object automatically (nothing to configure, as long
as the model is never `.reset()`), so `solve_cl_model_adaptive` builds one
`ConnectorLaneModel` and mutates it in place across iterations rather than
rebuilding from scratch each time. This is checked empirically, not just
asserted, by `tests/test_connector_lane_optimization.py::
test_basis_warm_start_speeds_up_reoptimization`, which replays the same
promotion schedule through a deliberately-cold, rebuilt-every-time control
and compares total solve time.

`AdaptiveSolveResult` (`cl_model`, `iterations`, `objective_history`,
`integer_pile_slots`, `promotions_by_iteration`, `converged`) exposes the
whole trace for diagnostics; `objective_history` is non-decreasing by
Proposition 3's first inequality (each intermediate `z_t` is itself a
valid lower bound on the true optimum, since relaxing integrality only
enlarges the feasible set) and its own solving instructions cap the loop
at `M*K` iterations (Proposition 3's own finiteness bound) purely as a
defensive guard — `converged=False` there would indicate a bug, not
expected behavior.

## Symmetry breaking (current Section 10, constraints (25)-(26))

All piles are identical, and within a pile all connectors are identical,
so any solution has up to `M! * (C!)^M` relabelled twins the solver may
waste effort re-proving are no better than each other. `build_cl_model(...,
break_symmetry=True)` adds:

```
(25)  sum_{i<j} sum_c y[i,m-1,c]  >=  sum_c y[j,m,c]     for m = 1..M-1
(26)  sum_{i<j} y[i,m,c-1]        >=  y[j,m,c]           for m = 0..M-1, c = 1..C-1
```

(25) says vehicle `j` may use pile `m` only if some vehicle with a
strictly smaller id already uses pile `m-1` (on any connector); (26) says
the same for connectors within a pile. Given any feasible solution,
relabelling piles by the smallest-id occupant, then relabelling connectors
within each pile the same way, produces an equivalent solution satisfying
both — the same relabelling-based proof as `offline_opt`'s
`break_pile_symmetry` (see that package's README, "Pile symmetry"), so
this never excludes the true optimum. `i < j` here uses ascending vehicle
`id`, matching `(11)`-`(12)`'s own sequencing-pair order, not the source
document's arrival-time order — the proof only needs *some* fixed total
order over vehicles, not that specific one.

Off by default, unlike `offline_opt`'s analogous flag (on by default
there): this document explicitly warns "aggressive symmetry breaking can
interfere with warm starts" (Section 8.2's warm-start note), which matters
more here since `adaptive.py`'s whole strategy leans on warm-starting.
Measure the effect on your own instance — for `solve_cl_model_adaptive`,
`break_symmetry` is applied identically to the main model and its
conservative MIP-start seed, so the two stay consistent either way.

## Preprocessing (`preprocess.py`, current Section 9)

Section 9.1 computes each vehicle's earliest possible departure, and
Section 9.2 explains what it's for — which is *not* narrowing which
variables get built.

- **`earliest_departures`** (9.1, eq. 22): `E_j`, the departure boundary
  vehicle `j` would achieve alone, with the whole module pool to itself,
  charging at its own acceptance limit throughout — computed with the
  model's own discrete recursion (`P_bar_j`, `tau^delta_j`), *not* a
  continuous closed form, since only the discrete version is guaranteed
  valid for the discretized model being preprocessed. Lives in `model.py`
  now (re-exported here), since `build_cl_model` itself also needs `E_j`
  internally for (24) below.
- **`incumbent_objective`** (9.2): `UB`, the objective value (total
  sojourn, minutes) of *any* known feasible schedule. Two sources, matching the
  document's own suggestions:
  - **a reference schedule you already have** — pass
    `incumbent_departures={vehicle_id: departure_time_minutes}`, e.g. from
    a finished simulation: `{ev.id: ev.departure_time for ev in
    env.engine.metrics.finished_evs}`. A vehicle missing from the dict
    counts as never served (departs at slot `K`, matching (9)'s own
    convention). Departure times are rounded *up* to the next slot
    boundary — always a safe upper bound, since a schedule that finishes
    at continuous time `t` can equally be read as "done" by the next slot
    boundary at or after `t`.
  - **Section 8.4's conservative shortcut** — used automatically when
    `incumbent_departures` is omitted (`None`).

### No per-vehicle right-edge windowing

An earlier revision of both the source document and this module narrowed
each vehicle's slot range on *both* ends — left (arrival, `k_j`) and right
(an incumbent-derived `kappa_j`) — combining `E_j` and `UB` into
`Kset_j = [k_j, min(K, E_j + (UB - sum_i E_i)))`. The current document's
Section 9.2 drops the right edge entirely: the slack any incumbent leaves
above the sum of individual best cases, `UB - sum_i E_i`, is shared across
*every* vehicle, so it only narrows anything when that slack is smaller
than the horizon itself — which does not happen in the congested regime
this model targets (a numeric example in the document: at `K=1440`,
`J=60`, `delta=1`, the slack would need to be under 24 minutes of excess
mean sojourn to bind at all). "Carrying a right edge that is always the
horizon adds notation and a boundary case without removing a single
variable." Every vehicle's variables/rows are therefore always generated
over the *whole* `[k_j, K)` — there is no `vehicle_windows` parameter
anymore, and no analogue of the old restored-obligation constraint that
per-vehicle windowing used to require (it existed only to patch a hole
that windowing itself created).

`UB` is still worth computing, for two cheaper uses instead (Section
9.2/11):

```python
from offline_cl_opt import incumbent_objective, build_cl_model, solve_cl_model_adaptive

# From a finished simulation:
sim_departures = {ev.id: ev.departure_time for ev in env.engine.metrics.finished_evs}
UB = incumbent_objective(vehicles, station, delta=1.0, horizon_minutes=120.0,
                         incumbent_departures=sim_departures)

# Or, with no reference schedule, fall back to Section 8.4 automatically:
UB = incumbent_objective(vehicles, station, delta=1.0, horizon_minutes=120.0)

result = solve_cl_model_adaptive(vehicles, station, delta=1.0, horizon_minutes=120.0, cutoff=UB)
```

- **an objective cutoff** — `solve_cl_model(..., cutoff=UB)` /
  `solve_cl_model_adaptive(..., cutoff=UB)`, Gurobi's own `Cutoff`
  parameter (see the Quickstart section above and `solve_cl_model`'s own
  docstring);
- **a MIP start** — the underlying schedule itself, already how
  `conservative_feasible_solution`'s own result and `warm_start_evs` are
  used in `solve_cl_model_adaptive`.

### Bounding the departure variables directly (9.3)

Independent of the above, `bound_departures=True` (`build_cl_model`'s
default) adds constraint (24), a cheap and always-valid lower bound on
each `D_j`:

```
sum_{k in Kset_j} u[j,k]  >=  n_min_j * (v_j - u[j, K-1])      n_min_j = E_j - k_j
```

`(v_j - u[j, K-1])` is 1 exactly when vehicle `j` genuinely departed within
the horizon (served, and not still occupying the last slot), 0 otherwise
(never served, or still present at the horizon edge) — so this only binds
for a vehicle that actually departs, requiring it to have occupied at
least `n_min_j` slots, the fewest any departing trajectory could have
needed, from `E_j` (9.1). An upper bound on `D_j` is never worth adding,
by contrast (the document is explicit about this): under a minimization of
total sojourn (increasing in every `D_j`), an upper bound on `D_j` can never
be active at the optimum.
(24) is cheap (`J` extra rows) and, per the source document, worth having
because the LP relaxation can otherwise report a `D_j` — which drives the
objective — well below what's actually achievable, weakening the root
bound.

## Scope

This package implements the exact model ("(2)-(19)" and the objective,
total sojourn) exactly, plus Section 8 (adaptive module integrality), Section 9's
`E_j`/`UB` (used for an objective cutoff, a MIP start, and the departure
lower bound (24) — never for windowing, see above), and Section 10's
symmetry breaking — nothing from the remaining refinement sections of the
source document:

- **Section 8.5** ("harvesting incumbents" via a solver callback on every
  integer-feasible solution found, and stopping an iteration before it
  proves optimality while still keeping a valid bracket) — not
  implemented; every iteration in `solve_cl_model_adaptive` runs to
  whatever `mip_gap`/`time_limit` allow, and only each iteration's *final*
  solution is tested/harvested, not every incumbent Gurobi finds along the
  way.
- **The recommended staged bracket-then-solve computational sequence**
  (bracket, preprocess, solve compact, adapt, refine) — not implemented
  end-to-end as a single call; `solve_cl_model_adaptive` covers "adapt",
  `conservative_feasible_solution`/`incumbent_objective` cover
  "bracket" and "preprocess", and `break_symmetry` covers "refine", but
  nothing here wires the stages together automatically -- compose them
  yourself, as in "Preprocessing" above.

`(14)` is `O(J*M*C*K)` rows, the largest family in the model once the
energy block is sparse (see "Why `x` is a real variable" above) — this
implementation is practical mainly for small-to-moderate instances,
similar in scale to `offline_opt`'s toy examples, for stations/horizons
where that count stays manageable.

## How this differs from `offline_opt`

Both packages solve the same underlying question (offline-optimal EV
charging-station scheduling) and both produce a valid performance-ceiling
lower bound, but via different formulations:

- **`offline_opt`** tracks per-slot completion state (`alpha`/`sigma`) and
  module counts per (vehicle, pile, slot); pile assignment is implicit
  through which pile's connector capacity a vehicle's modules draw from.
- **`offline_cl_opt`** (this package) assigns each vehicle to one *lane*
  (a specific pile+connector pair) for its whole stay, and sequences
  vehicles sharing a lane via an explicit disjunctive precedence variable
  `b_ij`. It has no `offline_opt`-style tapering safety margin to retrofit
  — the taper constraint (18) already uses the *effective discrete* time
  constant `tau^delta_j` from the start (see `instance.py`,
  `tau_delta_hours`), which is exactly the fix `offline_opt/README.md`
  documents having to add after the fact ("Taper cap looks ahead to slot
  end") — here it's part of the source formulation itself.
- Units: `offline_cl_opt` uses real kWh/kW/hours throughout (`h=delta/60`);
  `offline_opt` folds a kW*min scaling into battery capacity so `delta` in
  minutes combines with power directly. The two packages' `VehicleData`
  are **not** interchangeable — always use each package's own `instance.py`
  helpers to build vehicles for it.

Neither model is a special case of the other, so their optimal
`total_sojourn` / `mean_sojourn` values are not guaranteed to be numerically
identical on the same instance in general — both are valid lower bounds
on any causal policy's cost, which is what `test_offline_bound_never_exceeds_fifo_simulation`
in each package's test suite checks directly.
