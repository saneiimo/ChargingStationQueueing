"""
Gymnasium env for learning head-of-line pile assignment.

The agent does not advance the DES clock. We auto-advance until either a
decision is needed (queue nonempty and a free nozzle exists) or the episode
ends. Actions are pile indices; call action_masks() to hide full piles.

Observation (all scaled roughly to [0, 1]):
  for each pile:
    occupancy fraction, free-nozzle fraction, overload flag,
    total p_req / pile power, bricks used / bricks, mean SoC of plugged EVs
  head-of-line EV (or zeros if queue empty):
    battery / max, s_current, s_f, p_req / max power, energy_needed / max
  queue length / capacity

Reward over auto-advanced time:
  - QUEUE_HOLDING_COST * (queue length) * Δt
  - DROP_PENALTY per blocked arrival
Illegal pile choices also get a small instantaneous -1.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from models.station import ChargingStation
from metrics.metrics_tracker import MetricsTracker
from simulation.event import EventQueue
from simulation.engine import SimulationEngine
from policy.power.proportional import ProportionalPower
from config import (
    BATTERY_CAP_OPTIONS,
    DROP_PENALTY,
    HR2MIN,
    QUEUE_HOLDING_COST,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from policy.power.base import PowerPolicy


class ChargingStationEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_piles: int = 4,
        n_nozzles: int = 2,
        n_bricks: int = 5,
        p_brick: float = 25.0,
        queue_capacity: int = 10,
        lam: float = 5.0,
        power_policy: PowerPolicy | None = None,
        queue_holding_cost: float = QUEUE_HOLDING_COST,
        drop_penalty: float = DROP_PENALTY,
    ):
        super().__init__()

        if power_policy is None:
            power_policy = ProportionalPower()

        self.n_piles = n_piles
        self.n_nozzles = n_nozzles
        self.n_bricks = n_bricks
        self.p_brick = p_brick
        self.queue_capacity = queue_capacity
        self.queue_holding_cost = queue_holding_cost
        self.drop_penalty = drop_penalty
        self._max_battery = float(max(BATTERY_CAP_OPTIONS))
        self._max_power = float(n_bricks * p_brick)

        station = ChargingStation(
            n_piles=n_piles,
            n_nozzles=n_nozzles,
            n_bricks=n_bricks,
            p_brick=p_brick,
            queue_capacity=queue_capacity,
            power_policy=power_policy,
            lam=lam,
        )
        metrics = MetricsTracker(n_piles=n_piles, n_nozzles=n_nozzles)
        self.engine = SimulationEngine(station, metrics, EventQueue())

        self.action_space = spaces.Discrete(n_piles)

        # 6 features per pile + 5 HOL features + queue fraction
        obs_dim = 6 * n_piles + 5 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def action_masks(self) -> np.ndarray:
        """True where the pile still has a free nozzle."""
        return np.array(
            [not pile.is_full for pile in self.engine.station.piles], dtype=np.bool_
        )

    def _hol_ev(self):
        queue = self.engine.station.queue
        return queue[0] if queue else None

    def _get_obs(self) -> np.ndarray:
        piles = self.engine.station.piles
        features: list[float] = []

        for pile in piles:
            occ = len(pile.evs)
            features.append(occ / pile.n_nozzles)
            features.append(pile.free_nozzles / pile.n_nozzles)
            features.append(1.0 if pile.is_overloaded else 0.0)
            total_req = sum(ev.p_req for ev in pile.evs)
            features.append(
                min(total_req / self._max_power, 1.0) if self._max_power else 0.0
            )
            features.append(pile.bricks_used / pile.num_bricks)
            if pile.evs:
                features.append(float(np.mean([ev.s_current for ev in pile.evs])))
            else:
                features.append(0.0)

        hol = self._hol_ev()
        if hol is None:
            features.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        else:
            features.append(hol.c_b / self._max_battery)
            features.append(hol.s_current)
            features.append(hol.s_f)
            features.append(
                min(hol.p_req / self._max_power, 1.0) if self._max_power else 0.0
            )
            # energy_needed is kWh; _max_battery is kW·min → convert to kWh.
            features.append(hol.energy_needed / (self._max_battery / HR2MIN))

        features.append(len(self.engine.station.queue) / max(self.queue_capacity, 1))
        return np.array(features, dtype=np.float32)

    def _info(self, extra: dict | None = None) -> dict:
        info = {
            "time": self.engine.current_time,
            "queue_len": len(self.engine.station.queue),
            "terminated": self.engine.terminated,
            "last_event_type": (
                self.engine.last_event_type.name
                if self.engine.last_event_type is not None
                else None
            ),
            "finished": len(self.engine.metrics.finished_evs),
            "dropped": len(self.engine.metrics.dropped_evs),
            "action_mask": self.action_masks(),
        }
        if extra:
            info.update(extra)
        return info

    def _auto_advance_to_decision(self) -> tuple[float, bool, dict]:
        """Run the DES until the agent must choose a pile (or the day ends)."""
        reward = 0.0
        drops = 0
        steps = 0

        while (
            not self.engine.needs_assignment_decision() and not self.engine.terminated
        ):
            q_before = len(self.engine.station.queue)
            t_before = self.engine.current_time
            self.engine.advance_time()
            dt = self.engine.current_time - t_before
            reward -= self.queue_holding_cost * q_before * dt
            drops += self.engine.last_drops
            steps += 1

        reward -= self.drop_penalty * drops
        return (
            reward,
            self.engine.terminated,
            self._info({"auto_steps": steps, "drops": drops}),
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.engine.reset(seed)
        # engine.reset already processed one event; keep going to a decision if needed.
        if self.engine.needs_assignment_decision() or self.engine.terminated:
            return self._get_obs(), self._info({"auto_steps": 0, "drops": 0})

        _, _, info = self._auto_advance_to_decision()
        return self._get_obs(), info

    def step(self, action: int):
        mask = self.action_masks()
        illegal = False

        if not self.engine.needs_assignment_decision():
            reward, done, info = self._auto_advance_to_decision()
            info["illegal_action"] = True
            return self._get_obs(), reward, done, False, info

        if action < 0 or action >= self.n_piles or not mask[action]:
            illegal = True
            success = False
        else:
            success = self.engine.assign_ev(int(action))

        reward = -1.0 if illegal or not success else 0.0

        adv_reward, done, info = self._auto_advance_to_decision()
        reward += adv_reward
        info["illegal_action"] = illegal
        info["assign_success"] = success and not illegal
        return self._get_obs(), reward, done, False, info
