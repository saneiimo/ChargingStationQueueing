# `offline_cl_PB` — exact branch-and-price for the connector-lane model

Solves the **whole-module** compact MILP of `offline_cl_opt` (`build_cl_model`, `relax_modules=False`) to proven optimality. It uses the same `VehicleData` / `StationSpec` / `BoundaryVehicle` / cohort inputs, so a sweep can hand it the instance it already builds for the other two solvers.

```python
from offline_cl_PB import solve_branch_and_price

sol = solve_branch_and_price(
    inst.vehicles, station, delta=2.0, horizon_minutes=120.0,
    boundary_vehicles=inst.boundary_vehicles, cohorts=inst.cohorts,
    objective_cohorts=COHORTS_ALL,
    time_limit=3600, max_workers=4, progress=True,
)
sol.status            # "OPTIMAL" | "TIME_LIMIT" | "NODE_LIMIT"
sol.objective         # total sojourn of the best schedule, minutes: sum_j (delta*D_j - a_j)
sol.lower_bound       # proven lower bound, same units (== objective when OPTIMAL)
sol.mean_sojourn      # minutes, over objective_cohorts
sol.per_vehicle       # same columns as offline_cl_opt / offline_cl_dw
sol.compact_check     # every compact-model row evaluated at the schedule
sol.bound_history     # [{time_s, lower_bound, upper_bound, nodes, event}, ...]
```

`bound_history` records the certified bracket each time either end moves: `upper_bound` is the incumbent (non-increasing), `lower_bound` the best proven global bound so far (non-decreasing; `-inf` until the root is first priced). The global bound is the minimum bound over open nodes; while a node is being solved it is `min(that node's bound, other open nodes' minimum)`. Dive nodes are not tree nodes and never move it. The last entry (`event == "end"`) equals `(lower_bound, objective)`.

In the objective sweep it is its own stage: `run_sweep(..., run_bp_model=True)`, configured by `TrialConfig.bp_time_limit` and `TrialConfig.bp_initial_schedule`, reporting `bp_*` columns (see `experiments/README.md`).

## Formulation

**Columns.** A column (`columns.PBPlan`, an `offline_cl_dw.columns.Plan` plus `modules`) is one vehicle's whole stay: pile, start `S`, departure `D`, power `p_k`, and the **whole modules `q_k`** its connector holds in each occupied slot. It satisfies the vehicle's own rows of the compact model: (3)–(9), (13), (17)–(19) (no departure row at `K`, i.e. censoring) and `x ≤ W`. It also satisfies `p_k ≤ Δ q_k`, `q_k ≤ N u_k`, `q_k ∈ ℤ`. The null plan (`S = D = K`) is the never-served column.

**Master.**

```
min  Σ_j w_j D_ω λ_ω
     Σ α_ωmk λ_ω ≤ C        (pile m, slot k)   connectors
     Σ q_ωk λ_ω  ≤ N        (pile m, slot k)   whole modules
     Σ_{ω∈Ω_j} λ_ω = 1      (vehicle j)
```

`w_j = 1` for vehicles in `objective_cohorts`, else 0. With binary `λ` this is **exactly** the compact model. In one direction, a compact solution gives each vehicle its lane's `r` as `q`. In the other, an integer master solution gets connectors by interval colouring (`validation.assign_connectors`, which pins boundary vehicles to their physical connector) and each connector's `r` is its occupant's `q`. `validation.compact_model_check` verifies the second direction row by row on every returned schedule.

The DW package (`offline_cl_dw`) differs: its module row sums kW, a relaxation, which is why its schedules can fail the whole-module test.

**Boundary vehicles.**
- FIXED: one mandatory column, the pinned trajectory with `ceil(p/Δ)` modules.
- OPTIMIZE: priced on its own pile only, occupying slot 0, with energy starting from `initial_energy_kwh`. It has no null plan. Its simulated trajectory is added as a seed column only if it is whole-module valid on its own.

## Algorithm (`bp.py`)

- **Pricing**: one pricer per (vehicle, pile). It prices **served plans only**. The null plan is priced in closed form, and only when the node allows it: at a node that forbids the null plan, a pricer allowed to return `u = 0` could hide an improving served plan.
  - `labeling.py` (default): exact forward labelling over (slot, delivered energy, cost). For a fixed module profile, charging as fast as the modules allow is optimal, because the one-slot energy update is non-decreasing in energy. So a label with more energy and no more cost dominates. On a label explosion it falls back to the MILP for that call.
  - `pricer.py` (`pricer="milp"`): the pricing MILP, mirroring the compact model's single-vehicle rows plus `p ≤ Δq`, `q ≤ N u`. It reports Gurobi's `ObjBound`, and `BestBdStop` ends a solve once the bound proves "no improving column here".

  The tests check that the two agree on random duals and restrictions.
- **Node column generation**:
  - Phase II. If the filtered restricted master is infeasible (branching removed every column of some vehicle), run Phase I with artificials rather than prune the node.
  - Phase I proves infeasibility only through a Phase-I Lagrangian bound.
- **Bounds**: each round computes the node's Lagrangian bound, `C Σπ + N Σμ + Σ_j min(allowed plans' pricing objective)`, from the pricers' `ObjBound` and duals clamped to `≤ 0`. The objective is an integer, so a node closes when `ceil(bound − slack) ≥ UB`. The restricted master's LP value is never used as a bound.
- **Incumbents**:
  - Greedy list schedule (always feasible, needs no simulation).
  - Integer-recoverable node LPs.
  - Column-fixing dives: from the best open node, fix one fractional vehicle to a heavy served column's timetable and re-price, backtracking on failure. Runs before the root and every `dive_every` nodes.
  - The restricted integer master over generated columns: once after the root, then every `rim_every_nodes`.
  - An optional `initial_schedule`, for example `heuristics.schedule_from_compact` on a time-limited compact solve.

  Every incumbent passes `validation.validate_schedule` before it can lower `UB`.
- **Branching** (`branching.py`): on the first attribute the positive-weight columns disagree on, in order departure, start, pile, module count. It never branches on `λ`. Columns that share `(pile, S, D)` but differ in `q` have their power averaged and get `ceil(p̄/Δ)` modules. If that fits every pile, the node is integral without branching.
- **Search**: best bound first, deeper first on ties.
- **Speed knobs that never change the answer** (`BranchAndPrice` arguments):
  - `pricing_gap_abs`: a pricing MILP may stop within this absolute gap (minutes; default half a slot, `0.5*delta`). It is re-solved exactly whenever that leaves "improving column or certificate?" open.
  - `columns_per_pricer`: extra improving columns taken from Gurobi's solution pool.
  - `early_branching`: stop a node's column generation once its Lagrangian bound and restricted-master value round up to the same attainable objective value.
  - `dive_every`, `dive_time_limit`, `dive_backtracks`: the column-fixing heuristic.
  - `max_workers`: parallel pricing, one Gurobi environment per worker. This only helps `pricer="milp"`: the labelling pricer is pure Python and holds the GIL.

## Performance (measured, Windows laptop, i5-1145G7)

- **Contended 12-slot instances** (4–6 vehicles, scarce modules): proves the compact optimum in 0.1–27 s. On one instance the compact MILP needed 107 s.
- **Stored sweep trial `1pile_75kwh` #6** (12 vehicles incl. 2 in-service, K = 90, δ = 2; measured when the objective was `sum_j D_j`, so the figures below are in slots — total sojourn is `2·(figure) − sum_j a_j`):

  | Method | Lower bound | Best schedule |
  |---|---|---|
  | Compact MILP, 30 min (stored) | 717.4 | 742 |
  | DW, continuous modules | 730.6 | none |
  | This package, root node | 733.2 (in 9 s) | none |
  | This package, 20 min seeded with a 2-min compact incumbent (746) | 736.4 | 746 |

  So it gives by far the strongest proven bound. It did not close this instance in 20 minutes, and its own heuristics did not beat the compact model's incumbents here. On instances like this, run the compact model briefly and pass its incumbent: `initial_schedule=schedule_from_compact(cl)[0]`.
- **The 3-vehicle, 100-slot test instance**: the root LP bound (108 min) is far below the optimum (113 min of total sojourn). A long tree is needed, while the compact MILP solves it quickly.

With `time_limit`, `[lower_bound, objective]` is a rigorous bracket at any stop.

## Why the answer is exact

Two things could make it wrong, and each is guarded:

1. **Accepting an infeasible schedule.** Every incumbent passes the independent validator, and the final one also passes the compact model's own rows.
2. **Pruning on something that is not a lower bound.** Pruning only uses Lagrangian bounds, which are valid for any non-positive duals and any lower bounds on the pricing minima. So an early stop, a tolerance-level borderline column, or an `ObjBound`-terminated pricer can only weaken a bound, never invalidate it.

Numerical inconsistencies raise instead of being worked around. Examples: a pricer "improving" a column that is already active, a positive capacity dual, or Phase I stalling. See `tolerances.py`.

## Tests

```
python -m pytest offline_cl_PB/tests -v
```

- Labelling pricer vs MILP pricer: identical optimum under random duals and restrictions.
- Root column generation vs the LP of the **full** master built by brute-force enumeration of every whole-module column (`tests/enumeration.py`, independent of the pricer).
- Node bounds under random restrictions vs the filtered full master.
- Phase I revival and Phase I infeasibility.
- The null-plan trap.
- Branch partitions.
- Support disagreement.
- Power averaging.
- Optimum equality with the compact MILP and the full integer master, with and without boundary vehicles and cohorts, and with any incumbent, with or without the heuristic, sequential or parallel.
