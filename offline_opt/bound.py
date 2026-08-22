"""
High-level entry point: EVs + station -> offline lower-bound solution.

Mirrors the ergonomics of ``toy_demo.runner.run_toy_episode`` elsewhere in
this project, so a notebook can compute FIFO/heuristic/RL costs and the
offline bound side by side with the same EV list and station layout.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .instance import (
    C_RATE,
    S_THRESH,
    StationSpec,
    VehicleData,
    full_power_time,
    taper_time_constant,
    vehicles_from_evs,
)
from .model import build_offline_model, solve_offline_model
from .relaxed_model import build_relaxed_model
from .solution import OfflineSolution, extract_solution

if TYPE_CHECKING:
    from models.ev import EV
    from models.station import ChargingStation


def default_horizon_minutes(
    vehicles: list[VehicleData],
    station: StationSpec,
    s_th: float = S_THRESH,
    *,
    congestion_factor: float = 2.0,
    min_buffer: float = 30.0,
) -> float:
    """
    A generous, safe-starting-point horizon: last arrival, plus the slowest
    vehicle's own solo charge time, plus a congestion allowance scaled by
    total work over total servers, plus a flat buffer.

    Heuristic for seeding ``compute_offline_bound``, not a proof of
    feasibility -- if the solver still reports INFEASIBLE, pass a larger
    ``horizon_minutes`` explicitly. Per-vehicle solo time accounts for the
    station's module cap as well as the BMS taper, since a pile can bind below
    a vehicle's own peak acceptance.
    """
    if not vehicles:
        raise ValueError("Need at least one vehicle")

    pile_cap = station.n_modules * station.p_module

    def solo_time(v: VehicleData) -> float:
        capped = v.W / min(v.p_max, pile_cap)
        return max(full_power_time(v, s_th), capped)

    last_arrival = max(v.a for v in vehicles)
    total_work = sum(solo_time(v) for v in vehicles)
    slowest = max(solo_time(v) for v in vehicles)
    servers = max(station.n_piles * station.n_dispensers, 1)

    return last_arrival + slowest + congestion_factor * total_work / servers + min_buffer


def compute_offline_bound(
    evs: list["EV"],
    station: "ChargingStation | StationSpec",
    *,
    delta: float = 1.0,
    horizon_minutes: float | None = None,
    s_th: float = S_THRESH,
    c_rate: float = C_RATE,
    tie_break: bool = False,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
    verbose: bool = False,
) -> OfflineSolution:
    """
    Build, solve, and extract the offline lower bound for one instance.

    Parameters
    ----------
    evs :
        The vehicles in the instance (e.g. ``toy_demo.scenario.build_evs``
        output, or ``engine.metrics.arrived_evs`` from a simulated episode).
    station :
        A ``ChargingStation`` (its layout is read off) or a ``StationSpec``.
    delta :
        Slot length, minutes.
    horizon_minutes :
        T, the horizon. Defaults to ``default_horizon_minutes``.
    s_th, c_rate :
        Shared BMS taper parameters used to compute tau = (1-s_th)/c_rate.
        Default to this project's ``config.S_THRESH`` / ``config.C_RATE``.
    tie_break :
        See ``build_offline_model``. Produces a deterministic, front-loaded
        power profile without changing ``total_sojourn``; costs roughly 2x
        solve time.
    mip_gap, time_limit, threads, verbose :
        See ``solve_offline_model``.
    """
    vehicles = vehicles_from_evs(evs)
    spec = station if isinstance(station, StationSpec) else StationSpec.from_station(station)
    horizon = (
        horizon_minutes
        if horizon_minutes is not None
        else default_horizon_minutes(vehicles, spec, s_th)
    )
    tau = taper_time_constant(s_th, c_rate)

    offline_model = build_offline_model(vehicles, spec, delta, horizon, tau, tie_break=tie_break)
    solve_offline_model(
        offline_model,
        mip_gap=mip_gap,
        time_limit=time_limit,
        threads=threads,
        verbose=verbose,
    )
    return extract_solution(offline_model)


def compute_ip_bounds(
    evs: list["EV"],
    station: "ChargingStation | StationSpec",
    *,
    delta: float = 1.0,
    horizon_minutes: float | None = None,
    s_th: float = S_THRESH,
    c_rate: float = C_RATE,
    tie_break: bool = False,
    mip_gap: float | None = 1e-4,
    time_limit: float | None = None,
    threads: int | None = None,
    verbose: bool = False,
) -> tuple[OfflineSolution, OfflineSolution]:
    """
    Sandwich the true integer program's ``total_sojourn`` between two cheap
    continuous-relaxation bounds, without solving the (much harder) integer
    problem at all -- see ``relaxed_model.py``'s module docstring and
    ``README.md``, "Continuous relaxation bounds":

        RP(N*Delta).total_sojourn <= IP(N*Delta).total_sojourn
                                   <= RP([N-C+1]*Delta).total_sojourn

    Both calls solve ``relaxed_model.build_relaxed_model`` (no discrete
    module variable to branch on -- just the pile-assignment/timeline
    combinatorics), which is why this is cheaper than ``compute_offline_bound``,
    at the cost of only bracketing, not pinning down, the true value.

    Parameters
    ----------
    evs, station, delta, horizon_minutes, s_th, c_rate, tie_break, mip_gap,
    time_limit, threads, verbose :
        See ``compute_offline_bound``.

    Returns
    -------
    (lower_bound, upper_bound) :
        ``lower_bound`` solves RP at the station's real capacity (N*Delta);
        ``upper_bound`` solves RP at the reduced capacity (N-C+1)*Delta.
        ``lower_bound.total_sojourn <= upper_bound.total_sojourn`` always,
        and the true integer optimum lies in between.
    """
    vehicles = vehicles_from_evs(evs)
    spec = station if isinstance(station, StationSpec) else StationSpec.from_station(station)
    horizon = (
        horizon_minutes
        if horizon_minutes is not None
        else default_horizon_minutes(vehicles, spec, s_th)
    )
    tau = taper_time_constant(s_th, c_rate)

    full_capacity = spec.n_modules * spec.p_module
    reduced_capacity = (spec.n_modules - spec.n_dispensers + 1) * spec.p_module

    lower_model = build_relaxed_model(
        vehicles, spec, delta, horizon, tau, pile_capacity_kw=full_capacity, tie_break=tie_break
    )
    solve_offline_model(
        lower_model, mip_gap=mip_gap, time_limit=time_limit, threads=threads, verbose=verbose
    )
    lower_bound = extract_solution(lower_model)

    upper_model = build_relaxed_model(
        vehicles, spec, delta, horizon, tau, pile_capacity_kw=reduced_capacity, tie_break=tie_break
    )
    solve_offline_model(
        upper_model, mip_gap=mip_gap, time_limit=time_limit, threads=threads, verbose=verbose
    )
    upper_bound = extract_solution(upper_model)

    return lower_bound, upper_bound
