"""
Gymnasium env for learning pile assignment (typically head-of-line).

The agent does not advance the DES clock. We auto-advance until either a
decision is needed (queue nonempty and a free connector exists) or the episode
ends. Actions are pile indices; call action_masks() to hide full piles.
Heuristic baselines may also pass ``ev=`` into ``step`` to assign a chosen
waiting vehicle (see ``policy.queue.base.QueuePolicy.decide``).

Observation (all scaled roughly to [0, 1]):
  for each pile:
    occupancy fraction, free-connector fraction, overload flag,
    total p_req / pile power, modules used / modules, mean SoC of plugged EVs
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

from models.ev import EV
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
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from policy.power.base import PowerPolicy


class ChargingStationEnv(gym.Env):

    metadata = {"render_modes": []}

    def __init__(
        self,
        n_piles: int = 4,
        n_connectors: int = 2,
        n_modules: int = 5,
        p_module: float = 25.0,
        queue_capacity: int = 10,
        mean_interarrival: float | None = 5.0,
        power_policy: PowerPolicy | None = None,
        queue_holding_cost: float = QUEUE_HOLDING_COST,
        drop_penalty: float = DROP_PENALTY,
        max_time: float | None = None,
        battery_cap_options: Sequence[float] | None = None,
        arrivals: list[EV] | None = None,
        delta_arr: float | None = None,
        warmup_period: float | None = None,
        flush_queue_at_warmup: bool | None = None,
    ):
        """
        Parameters
        ----------
        mean_interarrival :
            Mean gap between arrivals in minutes. Arrival rate
            λ = 1 / mean_interarrival (customers per minute). Takes priority
            over ``arrivals`` when both are given. Pass ``None`` to run only
            off an externally-supplied ``arrivals`` list.
        max_time :
            Length of the *measured* phase (minutes) -- i.e. everything
            after any warm-up period; defaults to ``config.MAX_TIME``. See
            ``warmup_period`` below and ``SimulationEngine``'s own module
            docstring, "Warm-up period", for how the two combine.
        battery_cap_options :
            Battery capacities (kW*min, same units as
            ``config.BATTERY_CAP_OPTIONS``) sampled when generating arrivals
            internally. Defaults to ``config.BATTERY_CAP_OPTIONS``.
        arrivals :
            Pre-built EV arrival list (see
            ``simulation.arrivals.generate_arrivals``), used only while
            ``mean_interarrival`` is None. Installed once here; also
            overridable per call via ``reset(options={"arrivals": [...]})``.
        delta_arr :
            Arrival-time grid in minutes. ``None`` keeps continuous times.
            A positive ``d`` snaps every arrival (sampled or external) to the
            nearest multiple of ``d``; see ``simulation.arrivals``.
        warmup_period :
            Minutes to run *before* the measured phase begins. ``None``
            (the default) uses ``config.WARMUP_PERIOD`` -- ``0.0`` out of
            the box, i.e. no warm-up, fully backward compatible -- same
            "``None`` reads the config default" convention as ``max_time``.
            The DES runs ``[0, warmup_period]`` first, then ``max_time``
            more. Passed straight through to ``SimulationEngine`` -- see
            its module docstring for the full mechanics (the WARMUP_END
            sentinel event, the queued/in-service snapshot it always
            records on ``self.engine.metrics``, and how post-warm-up-only
            reporting works via ``metrics.validate.report_queueing_laws
            (post_warmup_only=True)``). The *resolved* value (never
            ``None``) is read back from ``self.engine.warmup_period`` after
            construction -- ``self.warmup_period`` here just mirrors
            whatever was passed in, same as ``self.mean_interarrival``/
            ``self.delta_arr`` do.
        flush_queue_at_warmup :
            If True, empty the live queue at t=warmup_period (only the
            queue -- EVs already plugged in are left alone; see
            ``SimulationEngine``). ``None`` (the default) uses
            ``config.FLUSH_QUEUE_AT_WARMUP`` (``False`` out of the box).
            No effect when ``warmup_period`` is 0.
        """
        super().__init__()

        if power_policy is None:
            power_policy = ProportionalPower()

        self.n_piles = n_piles
        self.n_connectors = n_connectors
        self.n_modules = n_modules
        self.p_module = p_module
        self.queue_capacity = queue_capacity
        self.mean_interarrival = mean_interarrival
        self.delta_arr = delta_arr
        self.warmup_period = warmup_period
        self.flush_queue_at_warmup = flush_queue_at_warmup
        self.queue_holding_cost = queue_holding_cost
        self.drop_penalty = drop_penalty
        self.battery_cap_options = (
            battery_cap_options if battery_cap_options is not None else BATTERY_CAP_OPTIONS
        )
        self._max_battery = float(max(self.battery_cap_options))
        self._max_power = float(n_modules * p_module)

        station = ChargingStation(
            n_piles=n_piles,
            n_connectors=n_connectors,
            n_modules=n_modules,
            p_module=p_module,
            queue_capacity=queue_capacity,
            power_policy=power_policy,
            mean_interarrival=mean_interarrival,
        )
        metrics = MetricsTracker(n_piles=n_piles, n_connectors=n_connectors)
        self.engine = SimulationEngine(
            station,
            metrics,
            EventQueue(),
            max_time=max_time,
            battery_cap_options=self.battery_cap_options,
            delta_arr=delta_arr,
            warmup_period=warmup_period,
            flush_queue_at_warmup=flush_queue_at_warmup,
        )
        if arrivals is not None:
            self.engine.set_arrivals(arrivals)

        self.action_space = spaces.Discrete(n_piles)

        # 6 features per pile + 5 HOL features + queue fraction
        obs_dim = 6 * n_piles + 5 + 1
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

    def action_masks(self) -> np.ndarray:
        """True where the pile still has a free connector."""
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
            features.append(occ / pile.n_connectors)  # occ fract
            features.append(pile.free_connectors / pile.n_connectors)  # free-connector frac
            features.append(1.0 if pile.is_overloaded else 0.0)  # overload flag
            total_req = sum(ev.p_req for ev in pile.evs)
            features.append(
                min(total_req / self._max_power, 1.0) if self._max_power else 0.0
            )  # total p_req / max_power
            features.append(pile.modules_used / pile.num_modules)  # module used modules
            if pile.evs:
                features.append(
                    float(np.mean([ev.s_current for ev in pile.evs]))
                )  # mean SoC of plugged EVs
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
        """
        ``options`` may carry ``{"arrivals": [EV, ...]}`` to (re)install an
        externally-supplied arrival list for this and future resets, used
        only while ``mean_interarrival`` is None (see
        ``simulation.arrivals.generate_arrivals``).
        """
        super().reset(seed=seed)
        arrivals = options.get("arrivals") if options else None
        self.engine.reset(seed, arrivals=arrivals)
        # engine.reset already processed one event; keep going to a decision if needed.
        if self.engine.needs_assignment_decision() or self.engine.terminated:
            return self._get_obs(), self._info({"auto_steps": 0, "drops": 0})

        _, _, info = self._auto_advance_to_decision()
        return self._get_obs(), info

    def step(self, action: int, ev=None):
        """
        Assign a waiting EV to ``action`` (pile id), then auto-advance.

        ``ev=None`` assigns head-of-line (RL default). Heuristics pass the EV
        from ``QueuePolicy.decide``.
        """
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
            success = self.engine.assign_ev(int(action), ev=ev)

        reward = -1.0 if illegal or not success else 0.0

        adv_reward, done, info = self._auto_advance_to_decision()
        reward += adv_reward
        info["illegal_action"] = illegal
        info["assign_success"] = success and not illegal
        return self._get_obs(), reward, done, False, info
