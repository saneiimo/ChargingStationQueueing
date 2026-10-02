# Charging Station Queueing Simulator

Discrete-event simulator of a multi-pile EV charging station, wrapped as a
Gymnasium environment for learning **which pile** should take the next vehicle
in queue.

## How the pieces fit together

Read the code in this order the first time through:

1. `config.py` — shared constants (taper threshold, module check, episode length, rewards).
2. `models/` — physical objects: station, piles, EVs.
3. `simulation/event.py` — timed events on a min-heap.
4. `simulation/arrivals.py` — Poisson sampling and optional `delta_arr` time grid.
5. `simulation/engine.py` — the clock: advance time, project SoC, process events.
6. `policy/power/` — how a pile splits its power modules among plugged EVs.
7. `policy/queue/` — baseline rules for choosing a pile (e.g. FIFO / join-shortest).
8. `metrics/` — L, Q, energy, utilization collected while the engine runs.
9. `env/charging_env.py` — Gym API: agent only acts at assignment decision points.
10. `main.py` — short script that rolls out the FIFO baseline.
11. Notebooks at the repo root (run with the project root as the working
    directory / kernel cwd so package imports resolve):
    - `simulate_episode_cl.ipynb` — one DES episode + validation / power plots
      against the connector-lane model
    - `compare_policies.ipynb` — Monte Carlo replications across queue/power policies
    - `cl_model.ipynb` — the offline connector-lane model (compact MILP,
      branch-and-price, Dantzig-Wolfe) on one simulated window
12. Offline (clairvoyant) optimum for total sojourn time, solved with
    gurobipy. Given full knowledge of arrivals up front, its optimum
    lower-bounds every causal queue/power policy's cost on the same
    instance — the benchmark to compare FIFO / heuristics / RL against:
    - `offline_cl_opt/` — the connector-lane compact MILP. See
      `offline_cl_opt/README.md`.
    - `offline_cl_PB/` — the same model solved exactly by branch-and-price.
    - `offline_cl_dw/` — its Dantzig-Wolfe lower bound (continuous modules).

```text
                    +------------------+
                    | ChargingStationEnv|
                    |  (Gym wrapper)    |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | SimulationEngine |  <-- owns the clock and event heap
                    +---+----------+---+
                        |          |
           +------------+          +-------------+
           v                                    v
  +----------------+                   +----------------+
  | ChargingStation|                   | MetricsTracker |
  |  queue + piles |                   +----------------+
  +--------+-------+
           |
           v
  +----------------+     update_power()     +----------------+
  |  ChargingPile  | <--------------------> |  PowerPolicy   |
  | connectors/modules |                        | (e.g. proport.)|
  +--------+-------+                        +----------------+
           |
           v
  +----------------+
  |      EV        |  SoC curve, p_req, p_act, next event
  +----------------+
```

## Station layout (in words)

- A **station** has a waiting **queue** and several **piles**.
- Each **pile** has a fixed number of **connectors** (physical plugs) and a pool of
  **power modules** (discrete chunks of kW). Modules are shared by all EVs on that pile.
- An **EV** arrives, waits in queue, gets assigned to one pile, charges until its
  target SoC, then leaves. Charging power follows a constant-then-taper curve.

Arrival times are a Poisson process (exponential gaps). Pass
`delta_arr=None` (the default) to keep those continuous times. Set
`delta_arr=d` on `generate_arrivals`, `ChargingStationEnv`, or
`SimulationEngine` to snap each arrival to the **nearest multiple of `d`
minutes** (`round(t / d) * d`). For example at `t = 3.69`: `d=1` → 4,
`d=2` → 4, `d=3` → 3, `d=5` → 5.0. The same snap is applied to an
externally supplied arrival list.

Power on a pile is **redistributed** when someone plugs in, leaves, or starts
under-using a module (`CHARGE_CHANGE`). That logic lives in `policy/power/`, not
in the RL agent. The agent only chooses the pile for the head-of-line EV.

## Running

```bash
python main.py

# All tests (use -s so print statements show)
python -m pytest tests/ -s -v

# Individual suites
python -m pytest tests/test_simulation_env.py -s -v
python -m pytest tests/test_arrivals.py -s -v
python -m pytest tests/test_queueing_laws.py -s -v
python tests/test_simulation_env.py
python tests/test_queueing_laws.py
```

In a notebook:

```python
!python -m pytest tests/test_simulation_env.py -s -v
!python -m pytest tests/test_arrivals.py -s -v
!python -m pytest tests/test_queueing_laws.py -s -v
```

## Visualization

After a run (or via the CLI helper), plot BMS request vs actual power for every
connector on one pile:

```bash
python -m visualization.pile_power --pile 0 --t-start 0 --t-end 240 --seed 42 --save pile0.png
```

Optional `--show` opens an interactive window. In a notebook:

```python
from visualization.pile_power import run_fifo_episode, plot_pile_connector_power
import matplotlib.pyplot as plt

env = run_fifo_episode(seed=42)
fig = plot_pile_connector_power(env, pile_id=0, t_start=0, t_end=200)
plt.show()
```

Per-EV theory vs simulation (`P-S`, `T-S`, `P-T`):

```python
from visualization.ev_curves import plot_ev_theory_vs_sim

figs = plot_ev_theory_vs_sim(env, ev_ids=[0, 3], charts=["P-S", "T-S", "P-T"])
plt.show()
```

```bash
python -m visualization.ev_curves --evs 0,1,2 --charts P-S,T-S,P-T --seed 42
```


The env does **not** ask the agent to advance time. It auto-advances the
simulator until either:

- there is a vehicle in queue and at least one free connector, or
- the episode ends (`SIM_OVER`).

At a decision point the action is a pile index. Use `env.action_masks()` so
full piles are never chosen. Reward penalizes time spent waiting in queue and
dropped arrivals (see `QUEUE_HOLDING_COST` and `DROP_PENALTY` in `config.py`).
