"""
One sweep point for the policy comparison: a scenario plus the policies to
race against each other in it.

``PolicyConfig`` is flat and CSV-serializable, like
``objective_sweep.TrialConfig`` -- which is why policies are named
(``"FIFO"``, ``"LSoCD"``) and resolved through the registries below rather
than stored as objects. A results row then fully describes the run that
produced it.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any

from config import BATTERY_CAP_OPTIONS, HR2MIN
from metrics.load import mean_interarrival_for_power_load
from policy.power.base import PowerPolicy
from policy.power.constant import ConstantPower
from policy.power.proportional import ProportionalPower
from policy.power.static import StaticPower
from policy.queue.base import QueuePolicy
from policy.queue.closest_power_match import ClosestPowerMatchQueuePolicy
from policy.queue.fifo import FIFOQueuePolicy
from policy.queue.lowest_soc_diff import LowestSoCDiffQueuePolicy

from .replications import scenario_run_names

# Name -> class. Names are what appear in configs and results, so keep them
# short and stable; adding a policy here is all it takes to make it sweepable.
QUEUE_POLICIES: dict[str, type[QueuePolicy]] = {
    "FIFO": FIFOQueuePolicy,
    "LSoCD": LowestSoCDiffQueuePolicy,
    "PMatch": ClosestPowerMatchQueuePolicy,
}

POWER_POLICIES: dict[str, type[PowerPolicy]] = {
    "Prop": ProportionalPower,
    "Static": StaticPower,
    "Constant": ConstantPower,
}


@dataclass(frozen=True)
class PolicyConfig:
    """
    A scenario, the policies to compare in it, and the replication budget.

    Any field can be swept via ``experiments.core.config_grid``. The
    defaults reproduce ``compare_policies.ipynb``'s own scenario.
    """

    # --- station layout ----------------------------------------------------
    n_piles: int = 1
    n_connectors: int = 2
    n_modules: int = 6
    p_module: float = 25.0
    queue_capacity: int = 100

    # --- traffic -----------------------------------------------------------
    # Arrival rate: give EITHER mean_interarrival (minutes between arrivals,
    # lambda = 1 / mean_interarrival) OR offered_power_load (rho_P, the
    # theoretical counterpart of overall power utilization: the fraction of
    # the station's rated power the arriving demand needs; see
    # metrics/README.md, 6.3). With mean_interarrival=None it is derived from
    # offered_power_load as 60 * E[W] / (rho_P * R) (E[W]: mean requested
    # kWh of the battery/SoC distributions; R = n_piles * n_modules *
    # p_module); with mean_interarrival set, offered_power_load is ignored.
    # The value actually used is effective_mean_interarrival, and is what
    # the mean_interarrival column of the results records.
    mean_interarrival: float | None = 15.0
    offered_power_load: float | None = None
    # None defers to config.MAX_TIME / config.WARMUP_PERIOD respectively.
    max_time: float | None = None
    warmup_period: float | None = None
    flush_queue_at_warmup: bool = False
    # Arrival-time grid: None keeps continuous exponential times.
    delta_arr: float | None = None
    # kWh; None defers to config.BATTERY_CAP_OPTIONS.
    battery_cap_kwh: tuple[float, ...] | None = None

    # --- who is racing -----------------------------------------------------
    # Names from QUEUE_POLICIES / POWER_POLICIES. Labels follow
    # ``scenario_run_names``: compound ("FIFO_Static") only when BOTH lists
    # have more than one entry, otherwise just the varying side's name.
    queue_policies: tuple[str, ...] = ("FIFO", "LSoCD")
    power_policies: tuple[str, ...] = ("Prop",)

    # --- the max-wait override --------------------------------------------
    # QueuePolicy.max_wait serves any EV waiting longer than this ahead of
    # the subclass rule. Three ways to set it:
    #
    #   max_wait=None, max_wait_from=None  -> off for everyone (default).
    #   max_wait=<float>                   -> that value, for every policy.
    #   max_wait_from="FIFO"               -> run FIFO first with no override,
    #                                         then give every OTHER queue
    #                                         policy FIFO's observed mean max
    #                                         wait * max_wait_factor.
    #
    # The third mirrors compare_policies.ipynb. Note it makes the run
    # sequentially dependent: the value handed to the other policies comes
    # out of the reference policy's own replications, so the config row
    # alone does not determine it -- the resolved number is recorded per
    # policy as ``max_wait_used``.
    max_wait: float | None = None
    max_wait_from: str | None = None
    max_wait_factor: float = 0.75

    # --- replication budget ------------------------------------------------
    n_reps: int = 30
    seed0: int = 0  # replication r uses seed0 + r, shared across policies (CRN)
    confidence: float = 0.95

    # --- bookkeeping -------------------------------------------------------
    label: str = ""
    notes: str = ""

    # ------------------------------------------------------------------ #
    def __post_init__(self) -> None:
        # Neither rate input set is allowed here: a sweep's BASE leaves the
        # swept one to config_grid. effective_mean_interarrival raises if a
        # config is used with neither.
        if self.mean_interarrival is not None and self.mean_interarrival <= 0:
            raise ValueError(f"mean_interarrival must be > 0, got {self.mean_interarrival}")
        if self.offered_power_load is not None and self.offered_power_load <= 0:
            raise ValueError(f"offered_power_load must be > 0, got {self.offered_power_load}")
        unknown_q = set(self.queue_policies) - set(QUEUE_POLICIES)
        unknown_p = set(self.power_policies) - set(POWER_POLICIES)
        if unknown_q:
            raise KeyError(
                f"Unknown queue policies {sorted(unknown_q)}; "
                f"choose from {sorted(QUEUE_POLICIES)}"
            )
        if unknown_p:
            raise KeyError(
                f"Unknown power policies {sorted(unknown_p)}; "
                f"choose from {sorted(POWER_POLICIES)}"
            )
        if self.max_wait_from is not None:
            if self.max_wait_from not in self.queue_policies:
                raise ValueError(
                    f"max_wait_from={self.max_wait_from!r} is not among "
                    f"queue_policies={list(self.queue_policies)}; it has to run "
                    "for its max wait to be observable."
                )
            if self.max_wait is not None:
                raise ValueError(
                    "Set either max_wait (explicit, same for everyone) or "
                    "max_wait_from (derived from a reference policy), not both."
                )

    @property
    def effective_mean_interarrival(self) -> float:
        """
        The mean inter-arrival time (minutes) the scenario uses:
        ``mean_interarrival`` when given, else the one that makes the
        theoretical offered power load equal ``offered_power_load``
        (``metrics.load.mean_interarrival_for_power_load``; batteries default
        to ``config.BATTERY_CAP_OPTIONS`` like the arrival generator's).
        Computed from the inputs every time, so ``dataclasses.replace`` /
        ``config_grid`` of either field always takes effect.
        """
        if self.mean_interarrival is not None:
            return float(self.mean_interarrival)
        if self.offered_power_load is None:
            raise ValueError("set mean_interarrival or offered_power_load (one of them is needed)")
        return mean_interarrival_for_power_load(
            self.offered_power_load,  # type: ignore[arg-type]
            rated_power=self.n_piles * self.n_modules * self.p_module,
            battery_cap_options=self.battery_cap_options or BATTERY_CAP_OPTIONS,
        )

    @property
    def battery_cap_options(self) -> list[float] | None:
        """Battery capacities in the engine's native kW*min units."""
        if self.battery_cap_kwh is None:
            return None
        return [c * HR2MIN for c in self.battery_cap_kwh]

    @property
    def scenario_kwargs(self) -> dict[str, Any]:
        """The ``ChargingStationEnv`` keyword arguments this config implies."""
        return {
            "n_piles": self.n_piles,
            "n_connectors": self.n_connectors,
            "n_modules": self.n_modules,
            "p_module": self.p_module,
            "queue_capacity": self.queue_capacity,
            "mean_interarrival": self.effective_mean_interarrival,
            "max_time": self.max_time,
            "battery_cap_options": self.battery_cap_options,
            "delta_arr": self.delta_arr,
            "warmup_period": self.warmup_period,
            "flush_queue_at_warmup": self.flush_queue_at_warmup,
        }

    def to_row(self) -> dict[str, Any]:
        """Flat, CSV-friendly view -- one column per knob, tuples joined."""
        row = dataclasses.asdict(self)
        # The rate actually used, whichever input set it.
        row["mean_interarrival"] = self.effective_mean_interarrival
        row["queue_policies"] = "+".join(self.queue_policies)
        row["power_policies"] = "+".join(self.power_policies)
        row["battery_cap_kwh"] = (
            "" if self.battery_cap_kwh is None
            else ",".join(f"{c:g}" for c in self.battery_cap_kwh)
        )
        return row


def policy_grid(cfg: PolicyConfig) -> list[tuple[str, str, str]]:
    """
    The queue x power grid for one config, as ``(label, queue, power)``.

    Queue-major (all powers for queue 0, then queue 1, ...), with labels
    from ``scenario_run_names`` so a one-power sweep stays labelled by the
    queue rule alone.
    """
    labels = scenario_run_names(list(cfg.queue_policies), list(cfg.power_policies))
    rows: list[tuple[str, str, str]] = []
    k = 0
    for q in cfg.queue_policies:
        for p in cfg.power_policies:
            rows.append((labels[k], q, p))
            k += 1
    return rows


def build_policies(
    queue_name: str, power_name: str, *, max_wait: float | None
) -> tuple[QueuePolicy, PowerPolicy]:
    """Instantiate one grid point's policy pair."""
    return (
        QUEUE_POLICIES[queue_name](max_wait=max_wait),
        POWER_POLICIES[power_name](),
    )
