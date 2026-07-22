# Charging Station Queueing Simulator

Discrete-event simulator of a multi-pile EV charging station, wrapped as a
Gymnasium environment for learning **which pile** should take the next vehicle
in queue.

## How the pieces fit together

Read the code in this order the first time through:

1. `config.py` — shared constants (taper threshold, brick check, episode length, rewards).
2. `models/` — physical objects: station, piles, EVs.
3. `simulation/event.py` — timed events on a min-heap.
4. `simulation/engine.py` — the clock: advance time, project SoC, process events.
5. `policy/power/` — how a pile splits its power bricks among plugged EVs.
6. `policy/queue/` — baseline rules for choosing a pile (e.g. FIFO / join-shortest).
7. `metrics/` — L, Q, energy, utilization collected while the engine runs.
8. `env/charging_env.py` — Gym API: agent only acts at assignment decision points.
9. `main.py` — short script that rolls out the FIFO baseline.

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
  | nozzles/bricks |                        | (e.g. proport.)|
  +--------+-------+                        +----------------+
           |
           v
  +----------------+
  |      EV        |  SoC curve, p_req, p_act, next event
  +----------------+
```

## Station layout (in words)

- A **station** has a waiting **queue** and several **piles**.
- Each **pile** has a fixed number of **nozzles** (physical plugs) and a pool of
  **power bricks** (discrete chunks of kW). Bricks are shared by all EVs on that pile.
- An **EV** arrives, waits in queue, gets assigned to one pile, charges until its
  target SoC, then leaves. Charging power follows a constant-then-taper curve.

Power on a pile is **redistributed** when someone plugs in, leaves, or starts
under-using a brick (`CHARGE_CHANGE`). That logic lives in `policy/power/`, not
in the RL agent. The agent only chooses the pile for the head-of-line EV.

## Running

```bash
python main.py
pytest tests/
```

## RL loop (decision-point MDP)

The env does **not** ask the agent to advance time. It auto-advances the
simulator until either:

- there is a vehicle in queue and at least one free nozzle, or
- the episode ends (`SIM_OVER`).

At a decision point the action is a pile index. Use `env.action_masks()` so
full piles are never chosen. Reward penalizes time spent waiting in queue and
dropped arrivals (see `QUEUE_HOLDING_COST` and `DROP_PENALTY` in `config.py`).
