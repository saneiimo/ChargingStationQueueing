# `offline_cl_PDS` — exact price-directed state-space search

Solves the **same model** as `offline_cl_opt` (compact MILP, whole modules, closed system) and `offline_cl_PB` (branch-and-price). It takes the same inputs, and returns either a proven optimum or a certified bracket `[lower_bound, objective]`. The method is the one in *An exact method for the clairvoyant benchmark of electric vehicle charging with shared power modules* ("the paper" below; equation numbers refer to it).

```python
from offline_cl_PDS import solve_price_directed_search

sol = solve_price_directed_search(
    inst.vehicles, station, delta=2.0, horizon_minutes=K * 2.0,
    boundary_vehicles=inst.boundary_vehicles, cohorts=inst.cohorts,
    objective_cohorts=COHORTS_ALL,
    initial_schedule=fifo_schedule,      # optional, e.g. offline_cl_PB.schedule_from_simulation
    time_limit=600, progress=True,
)
sol.status          # "OPTIMAL" | "WITHIN_TOLERANCE" | "TIME_LIMIT" | "EXPANSION_LIMIT" | "MEMORY_LIMIT"
sol.objective       # total sojourn of the best schedule, minutes: sum_j w_j delta*(D_j - k_j)
sol.lower_bound     # certified lower bound, same units (== objective when OPTIMAL)
sol.per_vehicle     # same columns as offline_cl_opt / offline_cl_PB
sol.to_cl_model()   # a solved ConnectorLaneModel: extract_solution and every offline_cl_opt plot
sol.passes          # one row per search pass; sol.bound_history: the bracket over time
```

`cl_model_PDS.ipynb` (repo root) runs it next to the compact model and B&P on `cl_model.ipynb`'s instance, and on a larger one the compact model cannot close.

Requires `numba` (the search kernel is compiled; the first call compiles for about a minute, and later calls use the on-disk cache).

## Notation

| Symbol | Meaning | Code |
|---|---|---|
| `M, C, N, Δ` | piles, connectors per pile, modules per pile, kW per module | `StationSpec` |
| `δ, h = δ/60, K` | slot length (min), in hours; number of slots | `delta`, `inst.K` |
| `k_j` | release slot `ceil(a_j/δ)` | `inst.release` |
| `W_j, R_j, P̄_j` | energy to deliver, energy to a full battery, `min(P_max, NΔ)` | `VehicleData` |
| `φ_j` | `exp(-h/τ_j)`; the compact model's `τ^δ_j = h/(1-φ_j)` | `params[:, P_PHI]` |
| `w_j` | 1 if the vehicle is in `objective_cohorts`, else 0 | `inst.weight` |
| `z` | search objective `Σ_j w_j D_j` (slots, an integer); total sojourn = `δ z − δ Σ_j w_j k_j` | `inst.to_minutes` |
| `μ[k,t]` | price of a vehicle holding at least `t` modules in slot `k` (`t = 0`: a connector) | `SlotPrices.mu` |
| `rent[k,q]` | `Σ_{t≤q} μ[k,t]`, what holding `q` modules in slot `k` costs | `Tables.rent` |
| `γ[k]` | most rent the whole station can pay in slot `k` | `Tables.gamma` |
| `V_j(k,x)`, `U_j(k)` | cost to go connected with energy `x` / not yet connected | `Tables.off/bp/val`, `Tables.U` |
| `f(s)` | bound of a partial schedule (state) `s` | `search.py` |

## Is the paper's model the compact model?

Yes, once the paper's search is made to respect the horizon (`D_j ≤ K`, which this package does). Term by term:

| Paper | `offline_cl_opt.model` | Same? |
|---|---|---|
| (5) `g(x,q) = min{x + hΔq, x + hP̄, φx + (1−φ)R, W}` | (14) `p ≤ Δ r`, (13) `p ≤ P̄`, (18) `τ^δ p + x ≤ R`, `x ≤ W`, (17) `x' = x + hp` | yes: `τ̃ = h/(1−φ)` is `tau_delta_hours`, so `x + h(R−x)/τ̃ = φx + (1−φ)R`. The compact model may draw less power, never more; drawing the most is never worse (`g` is non-decreasing, Proposition 1). |
| (12) at most `C` vehicles per pile and slot | lanes `y`, sequencing (11)–(12) | yes: intervals that overlap in at most `C` vehicles can be coloured with `C` connectors (`offline_cl_PB.validation.assign_connectors` does it, keeping boundary vehicles on their own connector). |
| (13) `Σ q ≤ N` per pile and slot, `q` integer | (15) `Σ_c r ≤ N`, `r` integer, (14) `p ≤ Δ r` | yes (the `r` of a lane is its occupant's `q`). |
| (10), (16) connect after arrival, one uninterrupted stay, depart at the first boundary with `x ≥ W` | (3)–(9), (19), completion `x_K ≥ W` | yes: the compact model may let a full vehicle stay longer, which never helps. |
| objective (8) `Σ D_j` | `Σ_j w_j δ (D_j − k_j)` | same ranking (scaled by `δ`, shifted by a constant; `w_j` handled as below). |
| no horizon | horizon `K` | this package forbids departures after `K` (`V_j(K, x<W) = ∞`), so it solves exactly the compact model. |

What the paper does not cover and this package adds (all exact):

* **Objective cohorts** (`w_j ∈ {0,1}`): departure costs `w_j D_j`; a zero-weight vehicle still holds connectors and modules.
* **OPTIMIZE boundary vehicles**: the search starts with them plugged in at boundary 0 on their pile, holding `initial_energy_kwh`.
* **FIXED boundary vehicles**: not searched; they reserve their connector and `ceil(p/Δ)` modules per slot (`free_connectors`, `free_modules`), their known rent is subtracted from the station term, and their departure is a constant. Piles are treated as interchangeable only from the first slot with no FIXED vehicle left (`symmetric_from`).
* **The horizon**: see "Vehicles that must connect now" below.

## The method

| Step | What | Where | Paper |
|---|---|---|---|
| 1 | Incumbents: `offline_cl_PB`'s greedy list schedule and `initial_schedule`, each validated | `solver.py` | Alg. 1 step 1 |
| 2 | Column generation → prices `μ` with the best Lagrangian bound `L(μ)` | `master.py`, `pricing.py`, `colgen.py` | (20)–(30), (26)–(27) |
| 3 | Prices rounded to `2^-20`; cost-to-go tables `V_j`, `U_j`, each step verified | `cost_to_go.py` | (31)–(33), (45), (47) |
| 4 | Beam passes (widths 200, 2000) for a good incumbent | `search.py`, `solver.py` | Alg. 1 step 4 |
| 5 | Exact passes with rising thresholds; the last proves optimality | `search.py`, `solver.py` | (35)–(44), Theorem 1 |

### Step 2: masters and prices

Two masters, both written for the **whole station** (`master.py`), so their duals are the same on every pile. That symmetry is what the bound needs: one rent per slot, whichever pile a vehicle is on.

* **Aggregate** (28)–(29): connectors in use ≤ `MC`, modules in use ≤ `MN` per slot. Its LP value equals `offline_cl_PB`'s pile-by-pile root LP (split each plan evenly over the piles); `tests` check it against the brute-force enumerated master.
* **Configuration** (22)–(23): for each threshold `t`, the vehicles holding ≥ `t` modules must fit into the piles' configurations. This is stronger: two piles of 6 modules cannot carry three vehicles with 4 modules each, which the aggregate rows allow. Run after the aggregate master, starting from all its plans. `master="auto"` uses it when `M ≥ 2`.

Each round prices every vehicle exactly by labelling (`pricing.py`: labels `(energy, rent)` per boundary; one with more energy and less rent dominates) and evaluates

```
L(μ) = Σ_j min_plans [w_j D + Σ_k rent[k, q_k]]  +  Σ_FIXED [w_j D_j + its rent]  −  Σ_k γ[k]
```

which is a lower bound on every schedule **for any `μ ≥ 0`** (Proposition 3(c)). The reason: on each pile, the vehicles connected in a slot form a configuration, whose rent is at most `γ[k]/M`. So the prices only steer how good the bound is, never whether it is valid. The best ones are kept. Dual smoothing (Wentges) speeds convergence.

### Step 3: cost-to-go tables

`V_j(k,x)` is the cheapest departure-plus-rent of vehicle `j` from slot `k` with energy `x` (32), and `U_j(k) = min_{S ≥ max(k,k_j)} V_j(S,0)` (33). It is a non-increasing step function of `x` (Proposition 4). Breakpoints come from mapping the next slot's breakpoints back through each module count.

**Verification (47).** `H(x)`, the right-hand side of (32) using the already-checked next table, is non-increasing. So if `H` at a step's last grid energy is still ≥ the step's value, the step is exact. Otherwise a breakpoint was missed: bisection finds it and a new step starts there with a value no higher than `H`. By induction from `K`, every table is ≤ the true cost to go at every grid energy. `PDSSolution.table_repairs` counts corrections; it has been 0 on every instance so far, as in the paper.

### Steps 4–5: the search

A **state** at boundary `k` is (departed set, connected vehicles with pile and energy, `g = Σ w_j D_j` so far). A transition over slot `k` connects a set of waiting vehicles to free connectors, each with ≥ 1 module (Lemma 1(c)), and gives every pile a *maximal* module vector. Maximal means each vehicle gets at most its useful modules `q̂`, and the pile's free modules are all used unless everyone already has `q̂` (Lemma 1(a)–(b)). Vehicles reaching `W` depart at `k+1`.

Every state carries the bound (35)

```
f(s) = g − tail[k] + Σ_connected V_j(k, x_j) + Σ_unconnected U_j(k),     tail[k] = Σ_{k'≥k} γ[k'] − FIXED rent in slots ≥ k
```

which is ≤ the value of every schedule through `s` (Proposition 5). A **pass** with threshold `T` keeps a successor only if `ceil(f) < min(T, v)`, where `v` is the best value the pass has found. Two shortcuts discard whole families of successors at once. Connection sets are enumerated over waiting vehicles sorted by their least bound increase `Δ_i` (Lemma 2(a), (40)). Module vectors are enumerated per pile, sorted by their contribution, depth first (Lemma 2(b)).

**Dominance** (Proposition 6): with the same departed set and the same vehicles on the same piles, a state with `g` no larger and every energy at least as large is at least as good, so the other is dropped. With identical piles, states are stored under a canonical pile numbering (piles sorted by their smallest connected vehicle).

**What a pass proves** (Theorem 1). A pass gives up regions in two ways. A memory cut (more than `layer_cap` states in a slot: the worse half dropped and an admission threshold set) or a beam cut. An expansion budget or time limit (the open states are left unexpanded). Let `β` be the smallest bound of anything given up. No schedule is below `min(T, v, ceil(β))` (`PassResult.certified`). A **complete** pass (`β = ∞`) that finds nothing proves `z* ≥ T`. One that finds `v` proves `v` optimal.

**Thresholds** (44): `T_1 = LB + ceil((target − LB)/8)`, steps growing by 1.5×, up to `target = UB − gap_steps`. With an expansion budget, the step is shortened so the next pass's predicted effort stays within half the budget.

**Vehicles that must connect now.** With a horizon, a waiting vehicle can reach a point where waiting one more slot makes finishing by `K` impossible (`U_j(k+1) = ∞`). Such a vehicle is part of every decision of that state. Without this, the bound would add `∞ = 1e18` for waiting and subtract it again for connecting. In floating point, `(x + 1e18) − 1e18` is off by up to ~100, which could prune a state wrongly. The paper has no horizon and does not need this.

## Why the answer is exact

Two things could make it wrong, and each is guarded:

1. **Pruning on something that is not a lower bound.** Every prune uses `f`, built from tables that are verified lower bounds (step 3) under prices for which the bound holds whatever they are (step 2). All of it lives on exact grids, so no rounding can push a bound up:
   * **Energies** are multiples of `2^-30` kWh, and every term of the dynamics is rounded *up* with a `2^-40` kWh margin (`dynamics.py`, (46)). So the search's model never delivers less energy than the compact model, and its optimum is ≤ the compact optimum (Proposition 8). Bounds computed on it are valid for the compact model.
   * **Prices** are multiples of `2^-20` slots, so every rent, table value and bound is an exact double, and `ceil(f)` is exact.
2. **Accepting an infeasible schedule.** A schedule found on the grid is replayed with the compact model's own dynamics (`solver.exact_plan`: `p = min(Δq, P̄, (R−x)/τ^δ, (W−x)/h)`) and must pass `offline_cl_PB.validate_schedule` before it can lower the upper bound. The returned schedule also passes every row of the compact model (`compact_check`). The grid adds at most about `10^-8` kWh per slot. A replay could fail only if a vehicle reached `W` within about `10^-5` kWh, and then the status says `REPLAY_FAILED`; it has not happened.

## Parameters worth knowing

| Argument | Default | Effect |
|---|---|---|
| `master` | `"auto"` | `"aggregate"`, `"configuration"`; auto = configuration when `M ≥ 2` |
| `time_limit` | `None` | whole solve; the bracket is valid at any stop |
| `cg_time_limit`, `cg_rel_tol` | `None`, `1e-5` | column generation stop; weaker prices only lengthen the search |
| `beam_widths` | `(200, 2000)` | heuristic passes; `()` skips them |
| `layer_cap` | `2_000_000` | states per slot before a memory cut; memory ≈ `2 × layer_cap × (60 + 11MC)` bytes |
| `max_expand` | `0` (none) | expanded states per exact pass |
| `gap_abs` | `0` | stop within this many minutes of total sojourn (`x · n_optimized` for `x` minutes of mean sojourn) |

## Tests

```
python -m pytest offline_cl_PDS/tests -v
```

The tests check:

* that the grid dynamics bound the compact dynamics from above;
* the aggregate master against the brute-force full master;
* the configuration bound against the aggregate bound and the optimum;
* tables and pricing against brute force over all module sequences;
* the optimum against the compact MILP, on one and two piles, two and three connectors, FIXED and OPTIMIZE boundary vehicles and objective cohorts, with and without beams and deepening;
* that early stops (memory, budget, tolerance) and the bound history give valid brackets;
* that a too-short horizon raises, as in the compact model.

## Limits and extensions

* **The bound is the master's bound.** The search must close the gap between the master's bound and the optimum, and its effort grows about geometrically with that gap. More vehicles in a busy period or several piles at heavy load can leave a bracket instead of a proof. The paper reports 0.1–0.7% on its hardest instances.
* **Boundary vehicles and symmetric prices.** With OPTIMIZE vehicles pinned to piles the true master is no longer pile-symmetric. The symmetric prices stay valid, but may be a little weaker than pile-specific ones.
* **The paper's further ideas are not implemented:** re-optimised prices at intermediate slots (the bound is valid for any prices, so the maximum of two bounds is valid too), and a horizon-free mode with the certificate (43).
