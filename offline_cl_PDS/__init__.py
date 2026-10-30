"""
Exact price-directed state-space search for the connector-lane model.

Solves ``offline_cl_opt``'s compact closed-system model (whole modules, every
vehicle served and fully charged by the horizon) -- the same instance
objects, boundary vehicles and objective cohorts as ``offline_cl_opt`` and
``offline_cl_PB`` -- to proven optimality, or returns a certified bracket.
Column generation supplies slot prices; the prices become exact cost-to-go
tables that bound every partial schedule; a layered search over the
station's slot-by-slot states builds schedules, pruning with the bound and
merging dominated states. See ``README.md``.

Typical usage::

    from offline_cl_PDS import solve_price_directed_search

    sol = solve_price_directed_search(
        inst.vehicles, station, delta=2.0, horizon_minutes=120.0,
        boundary_vehicles=inst.boundary_vehicles, cohorts=inst.cohorts,
        progress=True,
    )
    print(sol.status, sol.objective, sol.lower_bound, sol.mean_sojourn)
    cl = sol.to_cl_model()   # for extract_solution and the offline_cl_opt plots
"""

from __future__ import annotations

from .colgen import SlotPrices, slot_prices
from .instance import SearchInstance, build_search_instance
from .master import pile_configurations
from .solution import PDSSolution
from .solver import PriceDirectedSearch, exact_plan, solve_price_directed_search

__all__ = [
    "PriceDirectedSearch",
    "solve_price_directed_search",
    "PDSSolution",
    "SearchInstance",
    "build_search_instance",
    "SlotPrices",
    "slot_prices",
    "pile_configurations",
    "exact_plan",
]
