# `offline_cl_dw` — Dantzig-Wolfe decomposition for the connector-lane MILP

Implements `dantzig_wolfe_decomposition.html`: a decomposition-by-vehicle of
`offline_cl_opt`'s compact model (`connector_lane_model.html`, equations
(1)-(19), identical to what `offline_cl_opt/model.py` implements). Where
`offline_cl_opt` solves that model directly, this package splits it into a
**master problem** (which whole-schedule "plan" each vehicle uses) and a
**pricing subproblem** per (vehicle, pile) pair (generating good plans on
demand) — **column generation**. See "Why bother" below for when this is
worth reaching for.

**Deliverable implemented here** (the source document's own Section 0
recommendation): column generation for a certified lower bound, paired with
**price-and-branch** — one small integer solve over the columns already
generated — for a feasible upper bound. This is a *bracket*
`[lower_bound, upper_bound]`, not necessarily a single certified optimum;
see "Scope" for what that means in practice and when you'd want more.

## Why bother

`offline_cl_opt`'s compact model has a **weak** linear relaxation, and it's
weak in two specific, fixable places:

- The pairwise disjunctive rows (11)-(12) that keep two vehicles off the
  same lane at once are big-*M* — at a fractional point the big-*M* term
  swallows the constraint, so the relaxation sees almost no conflict
  between vehicles.
- A vehicle's occupancy can decay gradually in the relaxation rather than
  dropping cleanly, letting the departure rule (19) demand only a small
  fraction of the requested energy at each step — so the relaxation can
  report a departure time far below anything physically achievable.
  Departure times *are* the objective, so this directly destroys the bound.
  This is exactly the same phenomenon `offline_cl_opt`'s own
  `bound_departures` (constraint (26)) exists to blunt — decomposition
  removes it structurally instead of patching around it.

Decomposing by vehicle deletes the big-*M* rows entirely (replaced by a
plain packing constraint) and makes every column a genuine, physically
valid single-vehicle schedule, so neither weakness can occur. If
`offline_cl_opt.solve_cl_model_adaptive` (which relaxes only the module
variable `r`) still isn't fast enough at your scale, this is the next
lever — it relaxes the parts of the model that were actually causing the
weak bound, not the part (`r`) that Section 8 of the compact-model document
already shows is usually easy.

## Quickstart

```python
from offline_cl_opt.instance import StationSpec, vehicles_from_evs
from offline_cl_dw import solve_by_decomposition

station = StationSpec(n_piles=2, n_connectors=2, n_modules=4, p_module=25.0)
vehicles = vehicles_from_evs(evs)  # or build VehicleData directly, same as offline_cl_opt

solution, colgen = solve_by_decomposition(
    vehicles, station, delta=1.0, horizon_minutes=1440.0, progress=True,
)
print(f"bracket: [{solution.lower_bound:.3f}, {solution.upper_bound:.3f}]  gap={solution.gap:.3f}")
print(solution.per_vehicle)
```

`solution.lower_bound` is a certified lower bound on the true optimum —
valid even if `solution.converged` is `False` (an *anytime* bound, see
"Column generation" below). `solution.upper_bound` is `price-and-branch`'s
own objective — a genuine feasible schedule, always valid regardless of
convergence. Both are total sojourn in minutes, the same objective as
`offline_cl_opt` (a column costs its vehicle's sojourn `delta*D - a_j`), so a
gap of `g` is `g/n_optimized` minutes of mean-sojourn-time uncertainty.

`colgen` (a `ColGenResult`) keeps the full iteration history
(`z_rmp_history`, `lower_bound_history`) and the live master/pricers, if
you want to inspect convergence or add more columns and re-solve yourself.

## Column generation (`colgen.py`, `master.py`, `pricer.py`)

The restricted master (`master.py`) has one row per (pile, slot) for
connector capacity (25) and module capacity (26) — Propositions 1-2 of the
source document, replacing (10)-(16) of the compact model entirely — plus
one convexity row (27) per vehicle ("exactly one plan"). Columns
(`columns.Plan`) are added incrementally via gurobipy's own `Column`
object, the standard idiom.

The pricing subproblem (`pricer.py`) is, for one vehicle and one *fixed*
pile, exactly the compact model's own Groups A/D/E + (13) restricted to
that vehicle alone — no lane/module/sequencing constructs at all, since
Propositions 1-2 move that coupling entirely into the master's price
signal. One live `VehiclePricer` (a persistent gurobipy `Model`) is built
per (vehicle, pile) pair and reused for the whole run; only its objective
coefficients change between rounds (the document's own implementation
checklist, item 4: "the single largest speed factor in the loop"), which
also gets Gurobi's own LP basis warm-start for free — same principle
`offline_cl_opt.adaptive` already relies on.

**Pricers are solved concurrently** across a thread pool (`max_workers`),
not sequentially. Pricing every `(vehicle, pile)` pair is embarrassingly
parallel -- each one only depends on that round's duals, never on any
other pair's result -- and at scale (many vehicles/piles, or a long
horizon making each individual pricer itself slow) it dominates iteration
wall time far more than the master LP does. Gurobi's own `optimize()`
releases the GIL for the duration of the solve, so plain threads give real
concurrency across independent `Model` objects without the overhead --
and the loss of each pricer's persistent, basis-warm-started `Model` --
that process-based parallelism would force. Measured on a 30-vehicle,
4-pile synthetic instance (`K=240`, 120 pricers/round): 84s → 24s for the
same 3 iterations, ~3.5x on an 8-core machine, with byte-identical
`z_RMP_history` (parallelism changes wall time, never the answer). Each
individual pricer's own thread budget (`pricer_threads`) is capped small
(default `1`) precisely *because* many run at once -- letting every one of
them also claim every core would thrash rather than help; lower
`max_workers` and raise `pricer_threads` instead if you'd rather run fewer,
fatter pricers.

**Exact-MILP pricer only.** The source document offers faster
dynamic-programming/enumeration pricers (Section 6.3) as an optional
speed-up for later; this implementation always re-solves each pricer to
proven optimality. Simpler, always correct, no new approximation to tune —
profile your own instance before reaching for the faster pricer if pricing
itself (not the master, not price-and-branch) turns out to dominate wall
time.

**Stabilised** with dual price smoothing (`gamma`, eq. 33) — the source
document is explicit that unstabilised column generation on this
particular master "will oscillate", since the connector-capacity rows are
highly degenerate and identical piles make whole groups of columns
interchangeable. A genuine subtlety worth knowing if you read `colgen.py`:
the Lagrangian bound (32) is only valid when built from pricer optima at
the *true* (unsmoothed) master duals, so it's only updated on iterations
where pricing actually happened at true duals — a smoothed round still
looks for columns to add (checked against the *true* reduced cost before
being accepted, Section 7.3's own caution), it just doesn't move
`best_lower_bound`. Set `gamma=0` to disable smoothing outright (every
round then prices on true duals, and the bound updates every iteration).

**Deduplication** (Section 8.3): `master.add_column` silently skips a
column whose `(pile, start, departure, rounded power profile)` already
exists for that vehicle — "regenerating an existing column is a symptom of
a dual cycling problem, not a harmless waste."

**Two stopping criteria** (Section 7.4), whichever comes first: an exact
pricing pass finds no column with reduced cost below `-eps_rc` (`z_RMP ==
z_MP` exactly), or the bracket `z_RMP - best_lower_bound` closes to within
`gap_tolerance` (minutes of total sojourn). Either way, `max_iterations`/`time_limit` are a soft
backstop; `converged=False` there still leaves `best_lower_bound` valid.

## Getting an integer schedule: price-and-branch (`master.solve_integer`)

Section 9.1: re-solve the *current* restricted master (whatever columns
column generation produced) with every `lambda` forced to `{0,1}`. Always
feasible (the null plan is present for every vehicle), so its objective is
a genuine feasible schedule — a valid upper bound, but **not in general the
true optimum**: the best integer solution may need a column that was never
generated, since column generation only ever chased the *continuous*
relaxation's own optimum.

<details>
<summary>How loose can this actually be? (a real example)</summary>

On the 3-vehicle instance `tests/test_offline_cl_dw.py::test_bracket_contains_known_compact_model_optimum`
uses (true compact-model optimum: 113.0 minutes of total sojourn), column
generation (2000 iterations, not formally converged) ends with a certified
lower bound of 105.42 and a restricted-master LP value of 106.09 — correctly
below 113.0, since it's a continuous-module relaxation — but
price-and-branch's own upper bound comes out at 152.0,
dropping one vehicle entirely (choosing its null plan) rather than serving
all three. This is exactly the failure mode the source document names: the
LP-optimal columns can rely on a *fractional blend* across several plans
per vehicle that no single whole plan replicates, and price-and-branch
can't fall back to anything better than what was already generated. The
bracket `[105.42, 152.0]` is still correct (it contains 113.0), just not
tight on this particular instance.
</details>

If the gap price-and-branch leaves you with is too wide to support the
conclusion you're drawing, that is precisely the situation the source
document's Section 0 says to build full branch-and-price (Section 9.2) for
— not implemented here, see "Scope".

## Post-processing (`postprocess.py`)

- **`assign_connectors`**: the greedy left-edge sweep (10.1), exact for
  interval graphs (Proposition 1's own constructive proof) — a plan only
  names a *pile*, never a connector, so this is always needed to get a
  fully physical schedule.
- **`whole_module_failures`** / **`rounded_module_routing`**: the master
  only enforces the *continuous*-module condition (Proposition 2); these
  check/repair the stricter whole-module condition (37), same Lemma
  `offline_cl_opt.adaptive` already uses. Pass
  `conservative_modules=True` to `solve_by_decomposition` if you want this
  guaranteed feasible by construction instead (Section 8.4's tightened
  `(N-C+1)*Delta` budget, at the cost of reserving `C-1` modules per
  pile-slot regardless of load) rather than checked after the fact.
- **`validate_schedule`**: Section 10.3's full checklist, run as
  assertions — on by default (`solve_by_decomposition(validate=True)`).
  Cheap relative to the solve itself; leave it on.

## Seeding (`preprocess.py`)

`seed_columns` (used automatically) gives every vehicle the null plan
(mandatory — this is what makes the restricted master feasible from
iteration one, no artificial variables or Farkas pricing ever needed) plus
one "charge at the acceptance limit from `k_j`" plan per pile — cheap (no
MILP), individually optimal for that (vehicle, pile) pair alone.
`columns_from_evs` is the document's other suggested source: convert a
real simulation's own output into plans, the same idea as
`offline_cl_opt.adaptive`'s `warm_start_evs`, passed in via
`solve_by_decomposition(extra_seed_columns=...)`.

## Scope

Implements Sections 1-8, 10 (master, pricing, column generation,
stabilisation, post-processing) and Section 9.1 (price-and-branch) of the
source document. **Not implemented:**

- **Section 6.3's faster pricers** (dynamic programming / fixed-interval
  enumeration) — exact MILP only, see "Column generation" above.
- **Section 9.2, full branch-and-price** — the source document's own
  recommendation is to build this only if price-and-branch's gap turns out
  too wide for your purposes (see the worked example above, where it can
  be substantial). Branching would act on the pricer's own feasible region
  (fixing a vehicle's pile, or bounding `S_j`/`D_j`, per the document's
  three branching rules B1-B3), never on `lambda` directly — not built
  here.
- **A conservative-modules-by-default posture**: off by default (see
  `postprocess.repair_modules` above) rather than always paying the
  `(N-C+1)*Delta` reservation.

## Relationship to `offline_cl_opt`

Both packages solve the same underlying compact model
(`connector_lane_model.html`) and share its `VehicleData`/`StationSpec`
input types directly (imported, not duplicated) — build vehicles the same
way you would for `offline_cl_opt` (`vehicles_from_evs`, `VehicleData.from_ev`,
etc.). They differ only in *how* they search for a solution:
`offline_cl_opt` solves the compact MILP directly (exactly, or via its own
adaptive module-integrality relaxation); this package decomposes it by
vehicle first. Reach for this one specifically when
`offline_cl_opt.solve_cl_model_adaptive` is still too slow at your scale —
see "Why bother" above for exactly which weakness that relaxation doesn't
address but this decomposition does.
