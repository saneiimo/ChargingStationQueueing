# experiments

Two sweeps over a shared core. They answer different questions and are
complementary — one tells you which policy to run, the other tells you how
much that policy still leaves on the table.

```
experiments/
  core/                     shared: config_grid, RunStore (run dir + tables)
  objective_sweep/          how far is a realized episode from the optimum?
    config.py               TrialConfig
    trial.py                one episode -> instance -> exact MILP + DW
    sweep.py                run_sweep, comparison_table, load_results
  policy_sweep/             which policy wins, and by how much?
    config.py               PolicyConfig + the policy name registries
    replications.py         run_replications / summarize_ci / compare_policies
    sweep.py                run_policy_sweep, metric_table, load_results
  run_objective_sweep.py    editable entry point  (python -m experiments.run_objective_sweep)
  run_policy_sweep.py       editable entry point  (python -m experiments.run_policy_sweep)
  results/                  output (gitignored)
```

| | objective sweep | policy sweep |
|---|---|---|
| question | how far from optimal is this run? | which policy is better? |
| per config | 1 episode + 2 solves | `n_reps` episodes per policy |
| cost | seconds–minutes per trial (Gurobi) | milliseconds per episode |
| statistics | none — one episode, bounded | paired CIs across common random numbers |
| notebook | `objective_sweep_results.ipynb` | `policy_sweep_results.ipynb` |

Both write through `core.RunStore`, so their output directories follow the
same conventions and reusing a `run_name` overwrites the same way (see
**Output** at the end).

Importing is unprefixed for the objective sweep (backward compatible) and
explicit for the policy sweep:

```python
from experiments import TrialConfig, run_sweep, comparison_table   # objective
from experiments.policy_sweep import PolicyConfig, run_policy_sweep  # policy
from experiments.core import config_grid                            # shared
```

## The policy sweep

`compare_policies.ipynb` swept over a parameter grid and recorded. For each
configuration, every policy runs `n_reps` replications **sharing one seed
sequence** (common random numbers), so all policies face identical arrival
streams; differences are then paired per seed, which removes most of the
between-replication noise.

```python
from experiments.core import config_grid
from experiments.policy_sweep import PolicyConfig, run_policy_sweep, metric_table

base = PolicyConfig(
    n_piles=1, n_connectors=2, n_modules=6,
    queue_policies=("FIFO", "LSoCD"), power_policies=("Prop",),
    max_wait_from="FIFO", max_wait_factor=0.75,   # see below
    n_reps=30, seed0=0,
)
results = run_policy_sweep(config_grid(base, mean_interarrival=[10.0, 15.0, 20.0]))
print(metric_table(results, "avg sys time"))
```

Or `python -m experiments.run_policy_sweep` after editing `BASE`/`SWEEP` in
that file.

### Which policies

Named in `policy_sweep/config.py` so a config row stays CSV-serializable:

* `QUEUE_POLICIES` — `FIFO`, `LSoCD` (lowest SoC difference), `PMatch`
  (closest power match)
* `POWER_POLICIES` — `Prop` (proportional), `Static` (equal split among
  currently plugged EVs), `Constant` (fixed per-connector share, from pile
  geometry alone — an idle connector's modules are never lent to a busy one)

Sweeping both axes labels runs compoundly (`FIFO_Prop`, `FIFO_Static`); with
a single power policy the labels stay the queue names. Adding a policy to a
registry is all it takes to make it sweepable.

### The `max_wait` override

`QueuePolicy.max_wait` serves anyone waiting longer than the threshold ahead
of the subclass rule. Three ways to set it:

| | effect |
|---|---|
| neither field set | off for everyone (default) |
| `max_wait=<float>` | that value, for every policy |
| `max_wait_from="FIFO"` | FIFO runs unconstrained, then every *other* queue policy gets FIFO's observed mean max wait × `max_wait_factor` |

The third mirrors `compare_policies.ipynb`. It makes the run **sequentially
dependent** — the threshold comes out of the reference policy's own
replications, so the config row alone does not determine it. The resolved
number is recorded per policy as `max_wait_used`.

### Output tables

| file | one row per | use |
|---|---|---|
| `results.csv` | (config, policy) | every metric's `mean`/`ci_low`/`ci_high`. Long in policy, so it plots as one line per policy |
| `results_wide.csv` | config | `{policy}_{metric}_{stat}` columns + `sign_*` verdicts; mirrors the objective sweep's shape |
| `comparisons.csv` | (config, pair, metric) | both means, the CI on their **difference**, `significant`, `winner` |
| `replications.csv` | (config, policy, seed) | raw per-episode values, so any summary can be recomputed without re-simulating |

Metric names are slugified for columns: `"total energy delivered (kWh)"` →
`total_energy_delivered_kwh`, and the post-warm-up counterpart
`"avg sys time (measured)"` → `avg_sys_time_measured`.

**Read `comparisons.csv`, not the per-policy CIs, to judge a difference.**
Per-policy confidence bands routinely overlap while the paired difference is
decisively significant — that is the whole point of the common-random-numbers
pairing. `sign` is `-1` when the pair's first policy is lower, `+1` when the
second is, `0` when not significant; lower is better for the time metrics and
for `dropped EVs`, higher for `finished EVs` and energy delivered.

## The objective sweep

### What a trial produces

One trial = one `TrialConfig` = one simulated episode plus up to two solves.
Four numbers for the same objective come out, each reported per cohort:

| source  | what it is | comparable? |
|---------|-----------|-------------|
| `sim`   | the DES's continuous-time sojourns, truncated at the warm-up boundary | not achievable by any grid schedule — see below |
| `grid`  | that *same* FIFO schedule replayed on the `delta` slot grid | **yes** — the like-for-like target |
| `exact` | the connector-lane MILP's incumbent (an upper bound on the optimum), plus the solver's own `best_bound` | yes |
| `dw`    | Dantzig-Wolfe's certified **lower** bound on the optimum | yes |

The offline models can only depart on slot boundaries (assumption A1), so
each connector handover costs the successor up to `delta` minutes.
Comparing an optimum against `sim` directly is apples-to-oranges and makes
the optimum look worse for no real reason — compare against `grid`.

### Two reporting groups: `completed` and `arrived`

Every source reports each cohort over **two** populations, because "who
counts" is exactly where a naive comparison goes wrong:

| group | who it covers | use it for |
|-------|---------------|-----------|
| `arrived` | every vehicle the models were given; unfinished ones censored at the horizon | **comparing sources** — all four cover the same set |
| `completed` | only vehicles that received their full energy | reading realized service quality |

Columns are `{source}_{group}_{cohort}_{stat}`, e.g.
`grid_arrived_all_mean_sojourn`. `arrived` additionally carries
`..._n_censored`.

**Why both.** The simulation can only average vehicles that *finished*; the
offline models cannot drop anyone, so they censor the unfinished at
`D_j = K`. Mixing the two averages different populations, and the bias is
one-directional — the simulation drops precisely the longest sojourns while
the models keep them at their censored value, so the optimizer looks worse
than it is. Badly enough, at small `n`, to invert the ordering entirely and
make a *proven optimum* read as worse than feasible FIFO. `comparison_table`
defaults to `group="arrived"` for this reason, and prints each source's `n`
so any residual mismatch stays visible.

Note `dw_LB` is always an `arrived`-group quantity (it bounds the objective
over `objective_cohorts` as a whole), so it is not a valid bound for the
`completed` subset — comparing them will look like a bound violation
without being one.

### What counts as "completed"

A vehicle is completed iff it **received its full energy**:

```
energy_kwh + energy_delivered_before_kwh >= energy_required_kwh - 1e-6
```

Not `departure_slot < K`. That test is sound in one direction only —
constraint (19) forces `x_j >= W_j` at any departure edge strictly inside
the horizon, so `D_j < K` *does* imply completion — but the converse fails.
`D_j = S_j + sum_k u_jk` is an **exclusive** boundary, so a vehicle charging
through the final slot `K-1` lands on `D_j = K`, exactly where an unserved
vehicle sits (`S_j = K`, no occupancy). (19) has no row at `k = K` (Section
6.3's deliberate censoring), so the departure slot cannot separate the two.
`tests/test_experiments_sweep.py` pins both cases: same `D_j = K`, opposite
verdicts, distinguishable only by energy.

`served` is not a completion test either — it is `any(u_jk > 0.5)`, i.e.
"held a connector", and a censored vehicle can hold one while drawing zero
power.

`exact_n_completed_at_horizon` counts vehicles that finished *exactly* at
the horizon edge. Nonzero means the horizon is binding on the optimum.

**Watch the `n` columns.** Even within `arrived`, lengthen `max_time` until
the censored share is small before reading a gap as "what the policy leaves
on the table" — the two sides never agree on `n` for the `completed` group,
and a short window is dominated by the censoring convention rather than by
scheduling quality.

### Quick start

```python
from experiments import TrialConfig, config_grid, run_sweep, comparison_table

base = TrialConfig(
    n_piles=1, n_connectors=2, n_modules=6,
    warmup_period=360,
    arrival_horizon=900,  # pinned: this sweep varies max_time -- see below
)
configs = config_grid(
    base,
    mean_interarrival=[20.0, 30.0],
    max_time=[120, 240],
    delta=[1.0, 2.0],
)
results = run_sweep(configs, run_name="interarrival_x_delta")
print(comparison_table(results, cohort="all"))
```

Or run the worked example: `python -m experiments.run_objective_sweep`.

Either way, the results are written to disk as soon as they're produced (see
"Output" below) — running through the console is enough, nothing extra is
needed to persist them. `objective_sweep_results.ipynb` (repo root) loads a
finished run back and gives you the headline table, a plot against whichever
knob you swept, and a way to drop into one trial's full JSON detail.

### Sweeping `max_time`: pin the arrival draw

`simulation.arrivals.generate_arrivals` draws
`int(max_time / mean_interarrival * 5)` inter-arrival gaps up front and only
*then* draws each EV's battery and SoC. So a different draw horizon moves the
RNG position where the attribute draws begin, and **every EV comes out
different — including the ones whose arrival times are unchanged.** Two trials
differing only in `max_time` therefore see two different EV populations, and
the sweep would credit that variation to `max_time`.

Set `arrival_horizon` to a fixed value (comfortably above
`warmup_period` + the largest `max_time` swept) whenever the sweep varies
`max_time`, `warmup_period` or `arrival_oversample`:

```python
BASE = TrialConfig(warmup_period=6 * HR2MIN, arrival_horizon=15 * HR2MIN, ...)
```

Then every trial with the same `seed` and `mean_interarrival` shares one EV
population, and `max_time` changes only how much of the window is measured.
(There is no such thing as common random numbers *across* different
`mean_interarrival` values — a different arrival rate is a different stream.)

### The window's two edges

Both edges of the measured window are handled explicitly, because an arrival
landing exactly on one is routine with a gridded arrival stream (`delta_arr`,
whose grid normally divides `warmup_period`):

* **Opening edge.** An EV arriving exactly at `warmup_period` can be in the
  boundary snapshot *and* pass `arrived_post_warmup`'s `>=` test.
  `MetricsTracker.arrived_post_warmup` now excludes anyone already in the
  snapshot, so the three cohorts stay disjoint — otherwise
  `build_measurement_instance` emits the vehicle twice and the model dies on
  `KeyError: 'Duplicate keys in Model.addVars()'`.
* **Closing edge.** An EV arriving at (or within the last slot of) the
  window's end has release slot `k_j >= K` — no slot to be released in, and
  zero time inside the window. Those are dropped from the instance and
  counted as `inst_n_dropped_late_arrivals`.

### Which stages run

```python
run_sweep(configs, run_sim=True, run_exact_model=False, run_dw_model=True)
```

The DES episode **always** runs, whatever `run_sim` says: both offline models
are built from its realized arrival stream and its boundary snapshot, so
there is no instance without it. `run_sim=False` skips only the sim/grid
sojourn tables.

A solve that raises (an infeasibility, a failed assertion) is recorded as
`{"error": ...}` on its stage and the sweep carries on.

### Watching a solver think: `solver_progress`

`verbose` (above) prints this module's own one-line-per-stage summary
(`instance: J=...`, `exact: OPTIMAL obj=...`). `solver_progress` is a
separate, much noisier switch: it turns on each solver's OWN internal log.

```python
run_sweep(configs, solver_progress=True)   # or run_trial(cfg, solver_progress=True)
```

* **Exact model** — Gurobi's native solve log (presolve, root relaxation,
  the branch-and-bound node table as the incumbent/bound improve).
* **DW** — column generation's own per-iteration line:
  `[colgen] iter 19: z_RMP=231.756, best_LB=145.967, gap=85.789, exact=True, candidates=5, columns_added=5, total_columns=98`

This is meant for debugging one slow or stuck trial at a time, not for a
sweep of many — left on across a multi-trial sweep it prints each solver's
full internal log, in full, for every single trial.

### Boundary conditions

Two independent knobs decide who is in the measured-window instance:

* `include_queued` — fold the vehicles waiting in the queue at the boundary in
  as ordinary `a=0` vehicles (cohort `QUEUED`).
* `boundary_mode` — the vehicles already plugged in at the boundary (cohort
  `BOUNDARY`): `None` leaves them out (every connector starts free),
  `BoundaryMode.FIXED` pins them to what the simulation did, and
  `BoundaryMode.OPTIMIZE` lets the optimizer control their future power and
  departure but never their lane.

A third, `objective_cohorts`, decides who the optimizer *works for* — separate
from who is in the model. A vehicle outside it is still modelled in full (it
holds its connector, draws its modules, constrains everyone else); its `D_j`
just carries no weight. Normally exclude `BOUNDARY` under `FIXED`, whose
`D_j` is a constant; but under `OPTIMIZE`, excluding them means the optimizer
has no incentive to finish them and will let them linger.

### DW: lower bound only, by default

`solve_integer_ub=False` (the default here) stops after column generation and
skips price-and-branch. The sweep compares against the certified **lower**
bound, and the integer master is the expensive stage. In that mode `dw_UB`,
`dw_gap` and the `*_UB` sojourns are `nan` and there are no per-cohort DW
columns — there is no schedule to describe. Set `solve_integer_ub=True` to
get the full bracket back.

Three lower-bound numbers are recorded either way:

* `dw_LB` — the certified anytime Lagrangian bound (32). Valid even when
  column generation did not converge. **This is the bound to plot.**
* `dw_z_rmp` — the final restricted-master LP value.
* `dw_rmp_gap` — `z_rmp - LB`, column generation's own stopping criterion.
  A diagnostic of how close pricing got to proving `z_RMP == z_MP`; *not* a
  certified bracket the way `gap = UB - LB` is.

`dw_LB` is in raw objective units (`sum_j D_j`, absolute departure slots).
`dw_total_sojourn_LB` / `dw_mean_sojourn_LB` are the same bound converted to
minutes — those are what compares against a simulation's sojourn figures.

### DW: setting `gap_tolerance` from a minutes target

`gap_tolerance` is in raw objective units, and the minutes-uncertainty it
implies depends on `delta` *and* on `n_optimized` (the vehicle count inside
`objective_cohorts` — see the boundary-conditions section above, since that
count depends on `include_queued`/`boundary_mode`/`objective_cohorts`
together). Picking a raw value that means the same thing across trials whose
`n_optimized` differs is awkward, so there's a second way in:

```python
TrialConfig(delta=2.0, gap_tolerance_target_min=1.0, ...)  # 1 minute of mean-sojourn uncertainty
```

When `gap_tolerance_target_min` (`m`, minutes) is set, it takes priority over
the raw `gap_tolerance` field, and the actual value column generation runs
with is computed **per trial**, from that trial's own instance:

```
gap_tolerance = m * n_optimized / delta
```

(mirrors `sim_benchmark.ipynb`'s `gap_tolerance = m * len(vehicles) / delta`,
but uses `n_optimized` rather than every vehicle in the instance — the two
coincide only when `objective_cohorts=COHORTS_ALL`.) This can't be computed
before the episode runs — `n_optimized` isn't known until the boundary
snapshot and cohort assignment happen — so it's resolved inside `run_dw`,
not in `TrialConfig` itself. The value actually used, in raw units, is always
recorded as `dw_gap_tolerance_used`, whichever of the two fields set it.

### Output

Written under `experiments/results/<run_name>/`:

```
results.csv            one row per trial: every config knob AND every metric
                       as its own column, so a row is self-describing
trials/trial_000.json  the same trial in full, nested, plus `arrivals`:
                       the EV specs (id, battery, SoCs, arrival time) the
                       DES was fed. Enough to rebuild that episode with
                       `replay_episode`; the env object and the Gurobi / DW
                       models are not saved
run_meta.json          the sweep definition, stage flags, timing
```

`delta`, `mip_gap`, `time_limit`, `gap_tolerance`, `include_queued`,
`boundary_mode`, `objective_cohorts`, `seed` and every other knob are
**columns on each row**, not a file header — so a row read in isolation
still says what produced it, and you can filter/pivot on them directly.

Status and tolerance columns for reading a limited run honestly:

| column | meaning |
|--------|---------|
| `exact_status` | `OPTIMAL`, `TIME_LIMIT`, `SUBOPTIMAL`, … |
| `exact_mip_gap_achieved` | gap actually reached (vs. the requested `mip_gap` column) |
| `exact_objective` / `exact_best_bound` | incumbent and the solver's own bound — a time-limited run still brackets the optimum |
| `dw_status` | `CONVERGED`, `TIME_LIMIT`, `ITERATION_LIMIT`, `NOT_CONVERGED` |
| `dw_converged` / `dw_iterations` | column generation's own stopping detail |
| `*_runtime_s` | wall clock per stage (`exact_` also splits build vs. solve, and carries Gurobi's own `exact_gurobi_runtime_s`) |

`results.csv` is rewritten after every trial, so an interrupted sweep still
leaves usable output.
