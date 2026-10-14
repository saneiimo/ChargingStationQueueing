"""
Station load: how busy the station was, and how much it was asked to do.

Four families, explained in full (notation, formulas, how they relate) in
``metrics/README.md``:

* connector occupancy  -- fraction of time each connector held a vehicle;
* power utilization    -- fraction of each pile's rated power that reached
                          vehicles;
* offered power load   -- rated power needed to meet the arriving demand,
                          as a fraction of the station's rated power;
* offered connector load -- average number of connectors the arriving
                          demand would keep busy if every vehicle charged
                          alone on an empty pile (Erlangs), also per
                          connector.

The first two describe what the simulation *did* (time averages over a
window); the last two describe what it was *asked* to do (arrival rate
times per-vehicle demand), in a realized version (the vehicles that
actually arrived) and a theoretical one (the arrival generator's own
distributions, ``ArrivalModel``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from config import BATTERY_CAP_OPTIONS, C_RATE, HR2MIN, S_THRESH, SOC_F_BOUNDS, SOC_I_BOUNDS

if TYPE_CHECKING:
    from metrics.metrics_tracker import MetricsTracker
    from models.station import ChargingStation


# ---------------------------------------------------------------------------
# Solo charging time
# ---------------------------------------------------------------------------


def solo_charging_minutes(
    c_b: float | np.ndarray,
    s_i: float | np.ndarray,
    s_f: float | np.ndarray,
    *,
    s_th: float,
    c_rate: float,
    pile_power: float,
) -> float | np.ndarray:
    """
    Minutes to charge from ``s_i`` to ``s_f`` alone on an empty pile.

    The vehicle draws ``min(BMS request, pile_power)`` throughout -- the
    simulator's own charging curve (``models.ev.EV``): the BMS request is
    ``p_req_max = c_b * c_rate`` up to ``s_th``, then falls linearly to 0 at
    SoC 1 (slope ``tan_B = p_req_max / (1 - s_th)``). With the power capped
    at ``p_hat = min(p_req_max, pile_power)`` the vehicle charges at
    constant ``p_hat`` until the knee ``s_k = 1 - p_hat / tan_B`` (``= s_th``
    when nothing caps it), then follows the request, which decays
    exponentially in time:

        t = (c_b / p_hat) * (min(s_f, s_k) - s_i)^+
          + (c_b / tan_B) * ln((1 - max(s_i, s_k)) / (1 - s_f))   [if s_f > max(s_i, s_k)]

    ``c_b`` is in the simulator's kW*min units (so ``c_b / p_hat`` is in
    minutes per unit SoC). Accepts scalars or numpy arrays.
    """
    c_b = np.asarray(c_b, dtype=float)
    s_i = np.asarray(s_i, dtype=float)
    s_f = np.asarray(s_f, dtype=float)
    p_req_max = c_b * c_rate
    tan_B = p_req_max / (1.0 - s_th)
    p_hat = np.minimum(p_req_max, pile_power)
    s_k = 1.0 - p_hat / tan_B
    flat = (c_b / p_hat) * np.clip(np.minimum(s_f, s_k) - s_i, 0.0, None)
    s_a = np.maximum(s_i, s_k)
    with np.errstate(divide="ignore", invalid="ignore"):
        taper = np.where(s_f > s_a, (c_b / tan_B) * np.log((1.0 - s_a) / (1.0 - s_f)), 0.0)
    out = flat + taper
    return float(out) if out.ndim == 0 else out


def _rated_pile_power(station: "ChargingStation") -> float:
    """Rated power of one pile (``N * Delta``, kW); the piles must be identical."""
    powers = {float(p.num_modules * p.p_module) for p in station.piles}
    if len(powers) != 1:
        raise ValueError(f"piles have different rated powers {sorted(powers)}; offered loads assume identical piles")
    return powers.pop()


# ---------------------------------------------------------------------------
# The arrival generator, for the theoretical versions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ArrivalModel:
    """
    The distributions ``simulation.arrivals.generate_arrivals`` draws from:
    Poisson arrivals with mean gap ``mean_interarrival`` (minutes); battery
    capacity uniform over ``battery_cap_options`` (kW*min, the simulator's
    units); initial and target SoC uniform on ``soc_i_bounds`` /
    ``soc_f_bounds``; all independent. ``s_th`` / ``c_rate`` set every
    vehicle's charging curve (``config.S_THRESH`` / ``config.C_RATE``).
    """

    mean_interarrival: float
    battery_cap_options: tuple[float, ...] = tuple(BATTERY_CAP_OPTIONS)
    soc_i_bounds: tuple[float, float] = (SOC_I_BOUNDS[0], SOC_I_BOUNDS[1])
    soc_f_bounds: tuple[float, float] = (SOC_F_BOUNDS[0], SOC_F_BOUNDS[1])
    s_th: float = S_THRESH
    c_rate: float = C_RATE

    @classmethod
    def from_env(cls, env: Any, *, mean_interarrival: float | None = None) -> "ArrivalModel | None":
        """
        The model behind a ``ChargingStationEnv``: its ``mean_interarrival``
        (or the one passed here -- needed when the env was fed an external
        arrival list and so has ``mean_interarrival=None``) and its
        ``battery_cap_options``; SoC bounds and curve from ``config``.
        ``None`` when no mean inter-arrival time is known.
        """
        mia = mean_interarrival if mean_interarrival is not None else getattr(env, "mean_interarrival", None)
        if mia is None:
            return None
        caps = getattr(env, "battery_cap_options", None) or BATTERY_CAP_OPTIONS
        return cls(mean_interarrival=float(mia), battery_cap_options=tuple(float(c) for c in caps))


def expected_energy_request(
    battery_cap_options: Sequence[float],
    soc_i_bounds: Sequence[float] = (SOC_I_BOUNDS[0], SOC_I_BOUNDS[1]),
    soc_f_bounds: Sequence[float] = (SOC_F_BOUNDS[0], SOC_F_BOUNDS[1]),
) -> float:
    """
    ``E[W]``, kWh: the arrival generator's mean requested energy,
    ``E[c_b] / 60 * (E[s_f] - E[s_i])`` -- exact, since the battery and the
    two uniform SoCs are drawn independently. ``battery_cap_options`` in the
    simulator's kW*min units.
    """
    (i_lo, i_hi), (f_lo, f_hi) = soc_i_bounds, soc_f_bounds
    return float(np.mean(battery_cap_options)) / HR2MIN * ((f_lo + f_hi) / 2.0 - (i_lo + i_hi) / 2.0)


def mean_interarrival_for_power_load(
    offered_power_load: float,
    *,
    rated_power: float,
    battery_cap_options: Sequence[float],
    soc_i_bounds: Sequence[float] = (SOC_I_BOUNDS[0], SOC_I_BOUNDS[1]),
    soc_f_bounds: Sequence[float] = (SOC_F_BOUNDS[0], SOC_F_BOUNDS[1]),
) -> float:
    """
    The mean inter-arrival time (minutes) that gives a target theoretical
    offered power load ``rho_P`` -- the inverse of ``rho_P = 60 * lambda *
    E[W] / R`` (metrics/README.md, 6.3):

        lambda            = rho_P * R / (60 * E[W])     vehicles per minute
        mean_interarrival = 1 / lambda = 60 * E[W] / (rho_P * R)

    ``rated_power`` is the station's ``R = P * N * Delta`` (kW);
    ``battery_cap_options`` in kW*min, as the generator draws them.
    """
    if offered_power_load <= 0:
        raise ValueError(f"offered_power_load must be > 0, got {offered_power_load}")
    e_w = expected_energy_request(battery_cap_options, soc_i_bounds, soc_f_bounds)
    return HR2MIN * e_w / (offered_power_load * rated_power)


def theoretical_offered_load(
    model: ArrivalModel, *, n_servers: int, pile_power: float, rated_power: float, grid: int = 400
) -> dict[str, float]:
    return dict(_theoretical_offered_load(model, n_servers, pile_power, rated_power, grid))


@lru_cache(maxsize=64)
def _theoretical_offered_load(
    model: ArrivalModel, n_servers: int, pile_power: float, rated_power: float, grid: int
) -> dict[str, float]:
    """
    Offered loads from the arrival generator's distributions (no sample).

    * ``lambda = 1 / mean_interarrival`` (vehicles per minute);
    * ``E[W] = E[c_b] / 60 * (E[s_f] - E[s_i])`` kWh -- exact, since the
      battery and the two SoCs are independent;
    * ``E[S_solo]`` -- ``solo_charging_minutes`` averaged over the battery
      options and over the two uniform SoC ranges by a ``grid x grid``
      midpoint rule (relative error far below 1e-4 at the default).
    """
    lam = 1.0 / model.mean_interarrival
    caps = np.asarray(model.battery_cap_options, dtype=float)
    (i_lo, i_hi), (f_lo, f_hi) = model.soc_i_bounds, model.soc_f_bounds
    e_w = expected_energy_request(model.battery_cap_options, model.soc_i_bounds, model.soc_f_bounds)
    u = (np.arange(grid) + 0.5) / grid
    s_i, s_f = np.meshgrid(i_lo + (i_hi - i_lo) * u, f_lo + (f_hi - f_lo) * u, indexing="ij")
    e_s = float(np.mean([
        np.mean(solo_charging_minutes(c, s_i, s_f, s_th=model.s_th, c_rate=model.c_rate, pile_power=pile_power))
        for c in caps
    ]))
    return _offered(lam, e_w, e_s, n_servers=n_servers, rated_power=rated_power, suffix="theory")


def realized_offered_load(
    metrics: "MetricsTracker",
    *,
    since: float,
    arrivals_until: float,
    n_servers: int,
    pile_power: float,
    rated_power: float,
) -> dict[str, float]:
    """
    Offered loads from the vehicles that actually arrived in
    ``[since, arrivals_until]`` -- every arrival in that window, including
    any turned away by a full queue (they are demand too) and any caught in
    the warm-up snapshot at exactly ``since``. Vehicles that arrived during
    warm-up and are still queued or charging at ``since`` are not counted:
    they are the warm-up's demand, not this window's.

    ``lambda_hat = n / (arrivals_until - since)``; ``E[W]`` and ``E[S_solo]``
    are those vehicles' own means (each with its own ``s_th`` / ``c_rate``).
    """
    evs = [ev for ev in metrics.arrived_evs if since <= ev.arrival_time <= arrivals_until]
    length = arrivals_until - since
    if length <= 0:
        raise ValueError(f"arrival window [{since}, {arrivals_until}] is empty")
    lam = len(evs) / length
    if evs:
        e_w = float(np.mean([ev.energy_needed for ev in evs]))
        e_s = float(np.mean([
            float(solo_charging_minutes(ev.c_b, ev.s_i, ev.s_f, s_th=ev.s_th, c_rate=ev.c_rate, pile_power=pile_power))
            for ev in evs
        ]))
    else:
        e_w = e_s = 0.0
    out = _offered(lam, e_w, e_s, n_servers=n_servers, rated_power=rated_power, suffix="realized")
    out["n_arrivals_realized"] = float(len(evs))
    return out


def _offered(lam: float, e_w: float, e_s: float, *, n_servers: int, rated_power: float, suffix: str) -> dict[str, float]:
    a = lam * e_s
    return {
        f"arrival_rate_{suffix}": lam,  # vehicles per minute
        f"mean_energy_request_{suffix}": e_w,  # kWh
        f"mean_solo_time_{suffix}": e_s,  # minutes
        f"offered_power_load_{suffix}": HR2MIN * lam * e_w / rated_power,
        f"offered_connector_load_{suffix}": a,  # Erlangs
        f"offered_occupancy_{suffix}": a / n_servers if n_servers else math.nan,
    }


# ---------------------------------------------------------------------------
# Everything in one flat dict
# ---------------------------------------------------------------------------


def station_load_summary(
    metrics: "MetricsTracker",
    station: "ChargingStation",
    sim_time: float,
    *,
    since: float = 0.0,
    arrivals_until: float | None = None,
    arrival_model: ArrivalModel | None = None,
) -> dict[str, float]:
    """
    All four metric families as one flat ``{name: float}`` dict.

    Time averages (occupancy, power utilization) cover ``[since,
    sim_time]``; offered loads use the arrivals in ``[since,
    arrivals_until]`` (default ``sim_time``; pass ``engine.max_time``, the
    instant arrivals close). Keys:

    * ``connector_occupancy_p{m}_c{c}``, ``connector_occupancy`` (mean);
    * ``power_utilization_p{m}``, ``power_utilization`` (mean);
    * ``*_realized`` offered loads (always) and ``*_theory`` ones (when
      ``arrival_model`` is given): ``arrival_rate``,
      ``mean_energy_request``, ``mean_solo_time``, ``offered_power_load``,
      ``offered_connector_load``, ``offered_occupancy``.
    """
    n_piles, n_conn = len(station.piles), station.piles[0].n_connectors
    n_servers = n_piles * n_conn
    pile_power = _rated_pile_power(station)
    rated_power = pile_power * n_piles
    out: dict[str, float] = {}

    occ = metrics.connector_utilization(sim_time, since)
    for m in range(n_piles):
        for c in range(n_conn):
            out[f"connector_occupancy_p{m}_c{c}"] = float(occ[m, c])
    out["connector_occupancy"] = float(np.mean(occ)) if occ.size else math.nan

    util = metrics.pile_power_utilization(sim_time, pile_power, since)
    for m in range(n_piles):
        out[f"power_utilization_p{m}"] = float(util[m])
    out["power_utilization"] = float(np.mean(util)) if util.size else math.nan

    until = float(sim_time) if arrivals_until is None else float(arrivals_until)
    out.update(realized_offered_load(
        metrics, since=since, arrivals_until=until, n_servers=n_servers,
        pile_power=pile_power, rated_power=rated_power,
    ))
    if arrival_model is not None:
        out.update(theoretical_offered_load(
            arrival_model, n_servers=n_servers, pile_power=pile_power, rated_power=rated_power,
        ))
    return out
