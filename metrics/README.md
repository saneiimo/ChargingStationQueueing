# Simulation metrics: what is recorded, how, and how they relate

This is the reference for every metric the charging-station simulation
records or reports. The code lives in `metrics/metrics_tracker.py`
(`MetricsTracker`) and `metrics/load.py` (station load). This file covers
where each metric is reported (section 9), what it means, how it is
computed, and how the metrics constrain each other (section 8).

Metrics of the offline *optimization models* (objective values, bounds,
gaps, solver statuses) are documented in `experiments/README.md`.

---

## Notation

**Time** (all in simulation minutes)

| symbol | meaning | in code |
|---|---|---|
| `t0` | end of warm-up; the measured phase starts here (0 if no warm-up) | `metrics.warmup_period`, `engine.warmup_period` |
| `T` | when arrivals close: `t0 + max_time` (an arrival at exactly `T` is admitted) | `engine.max_time` |
| `t_end` | when the run ended: `T`, or later in a drained run (the station is served until empty) | `engine.current_time` at the end |
| `since` | start of the reporting window: `0` (whole run) or `t0` (measured phase only) | the `since=` argument |
| window | `[since, t_end]` for time averages; `[since, T]` for arrival rates | |

**Station**

| symbol | meaning | in code |
|---|---|---|
| `P` | number of piles | `station.n_piles` |
| `C` | connectors per pile; `P·C` connectors in total | `station.n_connectors` |
| `N` | power modules per pile | `pile.num_modules` |
| `Δ` | kW per module | `pile.p_module` |
| `N·Δ` | rated power of one pile, kW | `pile.power_supp` |
| `R = P·N·Δ` | rated power of the station, kW | |

**Vehicle `j`**

| symbol | meaning | in code |
|---|---|---|
| `a_j` | arrival time | `ev.arrival_time` |
| `b_j` | service start (plug-in) time | `ev.service_start_time` |
| `d_j` | departure time (charge complete) | `ev.departure_time` |
| `c_b` | battery capacity, kW·min (the simulator's unit; kWh = `c_b/60`) | `ev.c_b` |
| `s_i`, `s_f` | initial and target state of charge (SoC) | `ev.s_i`, `ev.s_f` |
| `s_th` | SoC where the charging request starts to taper | `ev.s_th` (= `config.S_THRESH`) |
| `p_req_max` | peak charging request, kW: `c_b · c_rate` | `ev.p_req_max` |
| `W_j` | energy the vehicle asks for, kWh: `(s_f − s_i) · c_b / 60` | `ev.energy_needed` |
| `S_j^solo` | solo charging time: from `s_i` to `s_f` alone on an empty pile (section 6.3) | `metrics.load.solo_charging_minutes` |

**Rates and averages**

| symbol | meaning |
|---|---|
| `λ` | arrival rate, vehicles per minute (theory: `1/mean_interarrival`) |
| `λ̂` | realized arrival rate: arrivals counted in `[since, T]` divided by `T − since` |
| `λ_eff` | throughput: finished vehicles of the cohort divided by the window length |
| `E[·]` | mean: over a set of vehicles (realized), or over the arrival generator's distributions (theory) |

**Realized vs theory.** A *realized* figure is computed from what happened
in this run (the vehicles that actually arrived, the energy actually
delivered). A *theory* figure is computed from the arrival generator's
distributions alone (`metrics.load.ArrivalModel`: Poisson arrivals, battery
uniform over the options, `s_i` and `s_f` uniform on their ranges, all
independent). It is the same for every run of a configuration.

---

## 1. The two windowing rules

Every method takes `since`, but it is applied in one of two ways:

* **Per-vehicle statistics** (waits, services, sojourns, finished counts,
  average energy per vehicle) filter the *vehicles*: only those with
  `a_j ≥ since` count. This is the standard warm-up convention.
  `include_ids` adds specific earlier vehicles back, namely the boundary
  cohorts (section 2). By default those are clocked from `since` rather than
  their real arrival (`truncate_included=True`).
* **Time averages** (L, Q, occupancy, power utilization, energy delivered)
  integrate only the part of the run in `[since, t_end]`. Everything
  physically present then counts, whenever it arrived.

---

## 2. Populations and cohorts

| metric | meaning |
|---|---|
| arrived | every vehicle that entered the station (`arrived_evs`), including later-dropped ones |
| finished | served and departed (`finished_evs`) |
| dropped | turned away because the queue was full (`dropped_evs`) |
| flushed | removed from the queue at `t0` by `flush_queue_at_warmup` (`flushed_evs`) |
| in system | still queued or plugged in when the run ended |

Identity (checked by `metrics.validate.validate_episode`):
`arrived = finished + dropped + flushed + queued + plugged`.

Cohorts at the boundary `t0` (used to compare with the offline models):

| cohort | who |
|---|---|
| measurement | arrived at or after `t0` (`arrived_post_warmup`) |
| queued | waiting in the queue at `t0` (`queued_at_warmup_end`) |
| boundary | plugged in and charging at `t0` (`in_service_at_warmup_end`) |

The nested levels reported are `measurement`, `measurement_queued`
(measurement + queued) and `all` (all three).

---

## 3. Per-vehicle times

For a finished vehicle (`finished_time_arrays`):

* **wait** `W_q = b_j − a_j`
* **service (charge) time** `S = d_j − b_j`, minutes plugged in
* **sojourn** `W = d_j − a_j = W_q + S`

| reported as | meaning |
|---|---|
| `W_q`, `S`, `W` (means) / `W_q_max` | means / maximum over finished vehicles of the cohort |
| `c_s2` | squared coefficient of variation of service times, `Var(S)/E[S]²` |
| policy sweep: `avg wait time`, `max wait time`, `avg charge time`, `avg sys time` | the same means, finished vehicles only |
| policy sweep: `... (all)` | also counts vehicles still in the system at the end, clocked up to the end (`in_system_time_arrays`). This is a lower bound on the truth, since they had not finished. A vehicle still queued has no service time and is skipped in the charge-time mean. |

Finished-only is optimistic (it drops the stragglers); `(all)` is
pessimistically clipped. The truth lies between them. In a drained run
nobody is left, so the two agree.

---

## 4. Time averages: L and Q

Logged online: the engine calls `update_metrics` at every event, so the
state is constant between consecutive log entries.

* **`L_sim`**: time-average number of vehicles in the station (queued +
  plugged in) over the window.
* **`Q_sim`**: time-average number queued.

---

## 5. Little's law and utilization

| key | formula | meaning |
|---|---|---|
| `lambda_eff` | finished vehicles of the cohort / window length | throughput |
| `L_theory` | `λ_eff · E[W]` | Little's law prediction of `L_sim` |
| `Q_theory` | `λ_eff · E[W_q]` | Little's law prediction of `Q_sim` |
| `rho_sim` | mean over connectors of (minutes occupied / window length) | measured connector utilization (= connector occupancy, section 6.1) |
| `rho_theory` | `λ_eff · E[S] / (P·C)` | utilization implied by throughput and the **actual** (shared, throttled) service times |
| `mu` | `1/E[S]` | service rate |

`L_sim ≈ L_theory`, `Q_sim ≈ Q_theory` and `rho_sim ≈ rho_theory` hold in a
long run. Over a short window they differ by edge effects: vehicles present
at the window's start or end are counted by the time averages but only
partly, or not at all, by the finished-vehicle means.

---

## 6. Station load

Computed by `metrics.load.station_load_summary`. 6.1 and 6.2 say what the
station *did*: time averages over `[since, t_end]`. 6.3 and 6.4 say what it
was *asked* to do: rate × per-vehicle demand. Those two come in a realized
and a theory version.

### 6.1 Connector occupancy

The fraction of time a connector held a vehicle. A vehicle leaves as soon
as it reaches `s_f`, so a held connector is always charging.

* **Per connector** `(m, c)`: minutes occupied in `[since, t_end]` /
  `(t_end − since)`. Key: `connector_occupancy_p{m}_c{c}`.
* **Overall**: plain mean over the `P·C` connectors (each weighs the same).
  Key: `connector_occupancy`. This is identical to `rho_sim`.
* Gathered from the per-interval busy log
  (`MetricsTracker.connector_utilization`).

### 6.2 Power utilization

The fraction of a pile's rated power that actually reached vehicles,
averaged over time:

    U_P(m) = energy pile m delivered in [since, t_end]  /  (N·Δ · (t_end − since)/60)

That is, delivered kWh over the kWh the modules would deliver running flat
out for the whole window.

* **Per pile**: key `power_utilization_p{m}`.
* **Overall**: plain mean over piles (each weighs the same). Key:
  `power_utilization`. With identical piles this equals station energy /
  (`R` × window).
* Gathered from the per-interval energy log (`MetricsTracker.pile_energy`,
  exact charging-curve integration) via
  `MetricsTracker.pile_power_utilization`.

### 6.3 Offered power load `ρ_P`: the fraction of rated module power needed to meet the charging demand

The average power the arriving vehicles ask for, as a fraction of the
station's rated power:

    ρ_P = 60 · λ · E[W] / R

`λ` is in vehicles per minute and `E[W]` in kWh, so `60·λ·E[W]` is in kW.

* **Realized** (`offered_power_load_realized`):
  * `λ̂` = vehicles that arrived in `[since, T]` divided by `T − since`.
  * `E[W]` = their mean requested energy.
  * Counted: every arrival in the window, including any turned away by a
    full queue (they are demand too) and any arriving exactly at `t0`.
  * Not counted: vehicles that arrived during warm-up and are still in the
    station at `t0`. They are the warm-up's demand, and they appear in the
    identity of section 8 instead.
* **Theory** (`offered_power_load_theory`):
  * `λ = 1/mean_interarrival`.
  * `E[W] = E[c_b]/60 · (E[s_f] − E[s_i])`. This is exact, because the
    battery and the two SoCs are drawn independently.

The components are reported too: `arrival_rate_*` (per minute) and
`mean_energy_request_*` (kWh).

### 6.4 Offered connector load `a`: the average number of busy connectors without throttling

How many connectors the arriving demand would keep busy on average if every
vehicle charged alone on an empty pile:

    a = λ · E[S^solo]     (Erlangs: an average count of busy connectors)
    a / (P·C)             (offered occupancy, comparable with 6.1)

**Solo charging time** `S^solo` (`solo_charging_minutes`): the vehicle
draws `min(charging request, N·Δ)` throughout, which is the simulator's own
charging curve. The request is `p_req_max` up to `s_th`, then falls
linearly in SoC to 0 at SoC 1, with slope `tan_B = p_req_max / (1 − s_th)`.
With power capped at `p̂ = min(p_req_max, N·Δ)` the vehicle charges at
constant `p̂` up to the knee `s_k = 1 − p̂/tan_B`. (The knee is `s_th`
when the pile does not cap it.) After the knee its power decays
exponentially in time:

    S^solo = (c_b / p̂) · (min(s_f, s_k) − s_i)⁺  +  (c_b / tan_B) · ln((1 − max(s_i, s_k)) / (1 − s_f))

The second term applies only when `s_f > max(s_i, s_k)`. The test suite
checks this against the simulator to 1e-6, with and without the pile
binding.

* **Realized** (`offered_connector_load_realized`,
  `offered_occupancy_realized`): `λ̂` as in 6.3 and `E[S^solo]` = the mean
  over those same vehicles.
* **Theory** (`offered_connector_load_theory`, `offered_occupancy_theory`):
  `λ = 1/mean_interarrival`. `E[S^solo]` is averaged over the battery
  options and the two uniform SoC ranges by a 400 × 400 midpoint rule
  (relative error well below 1e-4).

The component `mean_solo_time_*` (minutes) is reported too.

---

## 7. Sojourns by cohort and the grid replay (objective sweep)

These exist to compare the simulation with the offline models on equal
terms.

| method | meaning |
|---|---|
| `sojourn_by_cohort` | `n`, `total_sojourn`, `mean_sojourn` per nested cohort level, **finished** vehicles, clocked from `t0` for queued/boundary ones (the models place them at `a = 0`) |
| `arrived_sojourn_by_cohort` | the same over **every arrived** vehicle; an unfinished one counts `censor_at − origin` (`n_censored` says how many). The objective sweep calls this the `arrived` group; the finished-only one is the `completed` group. In a drained run they coincide. |
| `grid_sojourn_by_cohort` / `arrived_grid_...` | the run's own schedule replayed on the models' `δ`-minute slot grid: each vehicle keeps its lane and order, and its stay is rounded **up** to `⌈S/δ⌉` slots, starting no earlier than its release slot and its predecessor's departure. This is the like-for-like target for the models, which can only change vehicles on slot boundaries. Like the models, a vehicle's sojourn here runs from its release slot `k_j = ⌈a_j/δ⌉` to its departure slot: `δ·(D_j − k_j)`. `total_departure_slots` is `Σ_j D_j`. |
| `grid_departure_slots`, `grid_replay_intervals` | the replay's departure slot, and `(start slot, departure slot)`, per vehicle |

The grid replay keeps timing only, not power. `offline_cl_opt.grid_fifo_cl_model`
gives it a feasible power profile, following the simulator's rule (see
`offline_cl_opt/README.md`).

---

## 8. How the metrics relate

* **Occupancy and `rho_sim`** are the same number: `connector_occupancy ≡ rho_sim`.
* **Measured vs offered occupancy.** Sharing a pile only slows a vehicle
  down, so its actual service time is at least its solo time. Over a long,
  steady run, `connector_occupancy ≈ offered_occupancy × E[S]/E[S^solo]`,
  so occupancy ≥ offered occupancy. The ratio `E[S]/E[S^solo]` is the
  average stretch caused by sharing modules. Within one trial the two use
  different windows (`[t0, t_end]` vs `[t0, T]`), so treat this as a
  tendency, not an identity.
* **`rho_theory` vs offered occupancy.** Both are rate × time / `(P·C)`, but
  `rho_theory` uses throughput and the *actual* service times (throttling
  included), while offered occupancy uses arrivals and *solo* times
  (demand only).
* **Power utilization vs offered power load (exact).** In a drained run
  where nobody is turned away, every kWh asked for is eventually delivered:

      U_P · R · (t_end − t0)/60  =  ρ_P(realized) · R · (T − t0)/60  +  E_backlog(t0)

  `E_backlog(t0)` is the energy the vehicles present at `t0` still needed:
  queued vehicles' full `W_j`, plus `(s_f − s_current)·c_b/60` for those
  already charging. This holds to rounding (tested). Over a long steady run
  the backlog term and the drain become negligible, and `U_P ≈ ρ_P`.
* **Stability.** A station can only keep up in the long run if
  `ρ_P < 1` (enough power) **and** `a < P·C` (enough connectors). Sharing
  slows charging, so meeting both is necessary but not sufficient.
* **Realized vs theory.** The realized figures converge to the theoretical
  ones as the arrival window grows (law of large numbers; tested on a long
  stream). Their difference in one trial is the sampling noise of that
  arrival stream.

---

## 9. Where each metric is reported

**`MetricsTracker.queueing_summary(sim_time, since=..., station=..., arrivals_until=..., arrival_model=...)`**
returns `T`, `since`, `n_finished`, `lambda_eff`, `W`, `W_q`, `W_q_max`,
`S`, `mu`, `c`, `L_sim`, `L_theory`, `Q_sim`, `Q_theory`, `rho_sim`,
`rho_theory`, `E_total` (kWh delivered in the window), `E_avg` (mean kWh
per finished vehicle) and `c_s2`. With `station` given it also returns
every station-load key below. Theory keys appear only when `arrival_model`
is given.

**Station-load keys** (`metrics.load.station_load_summary`):

| key | section |
|---|---|
| `connector_occupancy_p{m}_c{c}`, `connector_occupancy` | 6.1 |
| `power_utilization_p{m}`, `power_utilization` | 6.2 |
| `arrival_rate_{realized,theory}`, `mean_energy_request_{…}`, `offered_power_load_{…}` | 6.3 |
| `mean_solo_time_{…}`, `offered_connector_load_{…}`, `offered_occupancy_{…}` | 6.4 |
| `n_arrivals_realized` | arrivals counted in `[since, T]` |

**Validation printout**: `metrics.validate.report_queueing_laws(env, verbose=True, post_warmup_only=..., mean_interarrival=...)`
prints the Little's-law table followed by the station-load block (overall
and per-connector occupancy, overall and per-pile power utilization,
offered loads realized vs theory). For an env fed an arrival list, pass
`mean_interarrival` to get the theory column.

**Policy sweep** (`experiments/policy_sweep/replications.py`): one row per
replication. Each metric appears twice: under its own name for the whole
run (`since = 0`, arrivals `[0, T]`), and as `"{name} (measured)"` for the
measured phase (`since = t0`). Columns: `finished EVs`, `dropped EVs`,
`total energy delivered (kWh)`, `average L`, `average Q`, `avg wait time`,
`max wait time`, `avg charge time`, `avg sys time`, `util_rho_sim`,
`util_rho_theory`, `connector occupancy`, `power utilization`,
`offered power load (realized|theory)`,
`offered connector load (realized|theory)`,
`offered occupancy (realized|theory)`, `in-system EVs` and the `(all)`
family. The theory columns are NaN when the scenario feeds its own arrival
list (no `mean_interarrival`).

**Objective sweep** (`experiments/objective_sweep`, one row of
`results.csv` per trial; window = measured phase `[t0, t_end]`, arrivals
`[t0, T]`):

| columns | content |
|---|---|
| `sim_` + every station-load key | sections 6.1–6.4 (`sim_connector_occupancy_p0_c1`, `sim_power_utilization`, `sim_offered_power_load_theory`, …); theory from the trial's `mean_interarrival` and battery options |
| `sim_{completed,arrived}_{cohort}_{n,total_sojourn,mean_sojourn,n_censored}` | section 7, continuous time |
| `grid_{completed,arrived}_{cohort}_{…,total_departure_slots}` | section 7, grid replay |
| `n_arrived_post_warmup`, `n_queued_at_warmup_end`, `n_in_service_at_warmup_end`, `n_arrived_total`, `n_finished_total`, `n_dropped_total`, `n_flushed_at_warmup` | section 2 |
| `arrivals_close_at` (`T`), `sim_clock_end` (`t_end`), `drain_minutes` (`t_end − T`), `sim_runtime_s` | run clock |

`experiments.objective_sweep.episode_utilization(env, post_warmup=..., mean_interarrival=...)`
returns `rho_sim`, `rho_theory`, `lambda_eff`, `mean_service_min` plus every
station-load key, for a replayed trial (`replay_episode`).
