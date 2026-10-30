"""
Column generation for the slot prices (step 2 of the method).

Each round solves the restricted master (``master.SlotMaster``), prices every
searched vehicle exactly (``pricing.price_vehicle``) and evaluates the
Lagrangian bound (26) at the prices it priced with:

    L(mu) = sum_j min_plan [ w_j*D + sum_k rent[k, q_k] ]  +  sum_FIXED [ w_j*D_j + sum_k rent[k, q_jk] ]
            - sum_k gamma[k]

(slots). ``L(mu)`` is a lower bound on every schedule for *any* ``mu >= 0``
(Proposition 3(c)): on each pile, the vehicles connected in slot ``k`` form a
configuration, whose rent is at most ``gamma[k]/M``. The prices with the best
``L`` are kept; they are what the search uses. At convergence ``L`` equals
the master's LP value ``z_LM``.

Dual smoothing (Wentges, 1997): vehicles are priced at
``alpha * best + (1 - alpha) * current`` duals. If that finds no column that
improves the master at the *current* duals ("mispricing"), the next round
prices with the current duals alone; column generation ends when a round at
the current duals finds nothing (or the gap ``z - L`` is below ``rel_tol``,
or time runs out -- the prices found so far are still valid, just weaker).

``"configuration"`` first runs the aggregate master and starts the
configuration master from all of its plans, as in Section 5.3.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from offline_cl_PB import PBPlan

from .dynamics import fastest_trajectory
from .instance import SearchInstance
from .master import MASTER_KINDS, SlotMaster, rent_and_gamma
from .pricing import price_vehicle


@dataclass
class SlotPrices:
    """Prices from one column generation, in slot units (``mu`` is not rounded yet)."""

    kind: str
    mu: np.ndarray  # [K, N+1] >= 0, the prices with the best Lagrangian bound
    configs: np.ndarray  # pile configurations n(c) of this master (rows)
    lagrangian_bound: float  # L(mu), slots
    master_value: float  # last restricted-master LP value, slots
    converged: bool
    iterations: int
    n_plans: int
    runtime: float
    aggregate_bound: float = float("nan")  # the aggregate master's L, when kind == "configuration"
    history: list[dict] = field(default_factory=list)  # one row per round


def seed_plans(schedules: list[dict[int, PBPlan]]) -> list[tuple[int, int, int, tuple[int, ...]]]:
    """``(vehicle id, S, D, modules)`` of every plan of the given schedules."""
    out = []
    for sched in schedules:
        for j, p in sched.items():
            out.append((j, p.start, p.departure, tuple(p.q(k) for k in p.occupied_slots())))
    return out


def _lagrangian_fixed_part(inst: SearchInstance, rent: np.ndarray) -> float:
    """FIXED vehicles have one plan each: departure plus the rent of its module profile."""
    total = 0.0
    for j, plan in inst.fixed_plans.items():
        total += inst.weights[j] * plan.departure + sum(rent[k, plan.q(k)] for k in plan.occupied_slots())
    return total


def _run(
    inst: SearchInstance,
    kind: str,
    seeds: list[tuple[int, int, int, tuple[int, ...]]],
    *,
    deadline: float | None,
    rel_tol: float,
    smoothing: float,
    columns_per_vehicle: int,
    max_labels: int,
    progress: bool,
    t0: float,
) -> tuple[SlotPrices, list]:
    st = inst.station
    K, N = inst.K, st.n_modules
    index = {int(j): i for i, j in enumerate(inst.ids)}
    fixed_ids = list(inst.fixed_plans)
    master = SlotMaster(inst.J + len(fixed_ids), K, st.n_piles, st.n_connectors, N, kind)
    row_of = {**index, **{j: inst.J + r for r, j in enumerate(fixed_ids)}}
    for j, S, D, q in seeds:
        if j in row_of and (j not in inst.fixed_plans):
            master.add_plan(row_of[j], S, D, q, float(inst.weights[j] * D))
    for j, plan in inst.fixed_plans.items():
        master.add_plan(row_of[j], plan.start, plan.departure, tuple(plan.q(k) for k in plan.occupied_slots()),
                        float(inst.weights[j] * plan.departure))

    trajs = [fastest_trajectory(N, inst.kappa, inst.params[i], K) for i in range(inst.J)]
    best_L, best_mu, best_sigma = -np.inf, None, None
    mispriced = False
    history: list[dict] = []
    converged = False
    it = 0
    z = np.inf
    try:
        while True:
            it += 1
            z, mu, sigma = master.solve()
            if best_mu is not None and best_sigma is not None and not mispriced and smoothing > 0:
                mu_p = smoothing * best_mu + (1 - smoothing) * mu
                sigma_p = smoothing * best_sigma + (1 - smoothing) * sigma
            else:
                mu_p, sigma_p = mu, sigma
            rent_p, gamma_p = rent_and_gamma(mu_p, master.configs, st.n_piles)
            rent_true, _ = rent_and_gamma(mu, master.configs, st.n_piles)

            L = _lagrangian_fixed_part(inst, rent_p) - gamma_p.sum()
            added = 0
            for i in range(inst.J):
                cutoff = sigma_p[i] + 0.5  # plans costing more cannot have negative reduced cost
                best, n, S, D, _costs, q, ok = price_vehicle(
                    inst.release[i], K, inst.start_pile[i] >= 0, inst.start_energy[i], N, inst.kappa,
                    inst.params[i], inst.weight[i], trajs[i], rent_p, cutoff, columns_per_vehicle, max_labels,
                )
                if not ok:
                    raise RuntimeError(f"pricing vehicle {inst.ids[i]} needed more than max_labels={max_labels} labels")
                L += min(best, cutoff)
                for c in range(n):
                    mods = tuple(int(x) for x in q[c, S[c] : D[c]])
                    w_cost = float(inst.weight[i] * D[c])
                    reduced = w_cost + sum(rent_true[k, m] for k, m in zip(range(S[c], D[c]), mods)) - sigma[i]
                    if reduced < -1e-7:
                        added += master.add_plan(i, int(S[c]), int(D[c]), mods, w_cost)
            if L > best_L:
                best_L, best_mu, best_sigma = L, mu_p.copy(), sigma_p.copy()
            elapsed = time.perf_counter() - t0
            history.append({"time_s": elapsed, "kind": kind, "master": z, "lagrangian": L, "best": best_L, "added": added})
            if progress and it % 10 == 0:
                print(f"[PDS {elapsed:7.1f}s] {kind} CG it {it}: z_RMP={z:.3f} L={L:.3f} best L={best_L:.3f} added={added}", flush=True)
            if added == 0:
                if mu_p is not mu:  # smoothed prices found nothing: retry at the current duals
                    mispriced = True
                    continue
                converged = True
                break
            mispriced = False
            if z - best_L <= rel_tol * max(1.0, abs(z)) or (deadline is not None and time.perf_counter() >= deadline):
                break
        assert best_mu is not None  # the first round always sets it
        prices = SlotPrices(
            kind=kind, mu=best_mu, configs=master.configs, lagrangian_bound=best_L, master_value=z,
            converged=converged, iterations=it, n_plans=len(master.plans), runtime=time.perf_counter() - t0,
            history=history,
        )
        plans = [(int(inst.ids[j]) if j < inst.J else fixed_ids[j - inst.J], S, D, q) for (j, S, D, q) in master.plans]
        return prices, plans
    finally:
        master.dispose()


def slot_prices(
    inst: SearchInstance,
    seeds: list[dict[int, PBPlan]],
    *,
    kind: str = "aggregate",
    time_limit: float | None = None,
    rel_tol: float = 1e-5,
    smoothing: float = 0.8,
    columns_per_vehicle: int = 3,
    max_labels: int = 2_000_000,
    progress: bool = False,
) -> SlotPrices:
    """
    Column generation on the ``kind`` master, started from the plans of
    ``seeds`` (complete schedules). Returns the best prices found.
    """
    if kind not in MASTER_KINDS:
        raise ValueError(f"kind must be one of {MASTER_KINDS}, got {kind!r}")
    t0 = time.perf_counter()
    deadline = t0 + time_limit if time_limit is not None else None

    def run(master_kind: str, plans: list) -> tuple[SlotPrices, list]:
        return _run(
            inst, master_kind, plans, deadline=deadline, rel_tol=rel_tol, smoothing=smoothing,
            columns_per_vehicle=columns_per_vehicle, max_labels=max_labels, progress=progress, t0=t0,
        )

    prices, plans = run("aggregate", seed_plans(seeds))
    if kind == "configuration":
        aggregate_bound, history = prices.lagrangian_bound, prices.history
        prices, _ = run("configuration", plans)
        prices.aggregate_bound = aggregate_bound
        prices.history = history + prices.history
    return prices
