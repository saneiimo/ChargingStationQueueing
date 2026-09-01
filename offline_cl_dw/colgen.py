"""
Column generation -- Section 7 of ``dantzig_wolfe_decomposition.html``.

Alternates solving the restricted master (``master.py``) for duals and the
per-``(vehicle, pile)`` pricing subproblems (``pricer.py``) for new
columns, tracking the Lagrangian bound (32) so a valid lower bound is
available at every iteration, not only at convergence (Section 6.4: "what
makes the procedure anytime"). Exact-MILP pricer only -- see ``pricer.py``'s
own docstring for why (Section 6.3's faster DP pricer is not built here).

Stabilised with dual price smoothing (eq. 33, Section 7.3): unstabilised
column generation on this particular master "will oscillate" per the
source document, since the connector-capacity rows are highly degenerate
and identical piles make whole groups of columns interchangeable.

A subtlety worth being explicit about, since it is easy to get wrong: the
Lagrangian bound (32) is valid only when built from ``zeta_jm`` values that
are genuine pricer *optima at the current true (unsmoothed) master duals*
(Section 6.4: "with duals (pi,mu,sigma) from the restricted master and
*exact* subproblem optima zeta_jm"). A plan found by pricing on *smoothed*
duals, even re-evaluated against the true duals afterwards, is only a
feasible point of the true-dual pricing problem, not necessarily its
optimum -- its true-dual objective is an upper bound on the true zeta_jm,
and plugging an upper bound into (32) in place of the true minimum can make
the resulting number too high to be a valid lower bound. So the bound is
only updated on iterations where pricing was actually done at the true
duals (matching Section 7.1's pseudocode: ``if pricing was exact this
round: LB = ...``) -- smoothed-dual rounds still price for candidate
columns (checked against the *true* reduced cost before being added, per
Section 7.3's own caution) but simply leave ``best_lower_bound`` where it
was.

Pricing every ``(vehicle, pile)`` pair is embarrassingly parallel -- each
one only depends on this round's duals, never on any other pair's result
-- and at scale (many vehicles and/or a long horizon, so each pricer MILP
itself takes real time) it dominates iteration wall time far more than the
master LP does. Pricers are therefore solved across a thread pool
(``max_workers``), not sequentially: Gurobi's own ``optimize()`` releases
the GIL for the duration of the C-level solve, so real concurrency across
independent ``Model`` objects works from plain threads, without the
overhead -- and the loss of each pricer's persistent, basis-warm-started
``Model`` -- that process-based parallelism would force. Each individual
pricer's own thread budget (``pricer_threads``) is capped small by default
specifically *because* many are running at once; letting every one of them
also claim every core would thrash rather than help.

Column purging (Section 8.3, ``master.purge_columns``): nothing in
``master.py`` ever removes a column on its own, so the restricted master
only ever grows across a run, and every ``solve_lp`` gets slower as it
does. Every ``purge_every`` iterations, this loop sweeps non-basic columns
whose reduced cost (at that round's own fresh LP solve) exceeds
``purge_threshold`` -- purging is therefore done immediately after each
round's ``solve_lp``, *before* that round's new columns are added, so every
column considered was actually part of the solve its ``.RC``/``.VBasis``
are read from (see ``purge_columns``'s own docstring for why this is a
deliberate, safe deviation from Section 7.1's pseudocode ordering). A
purged column is never gone for good: the pricer always searches the full
per-vehicle plan space, not the historical pool, so it can always come back
later if it becomes attractive again -- ``purge_columns`` clears the purged
plan's own dedup key precisely so that re-addition isn't silently refused.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

from offline_cl_opt.instance import StationSpec, VehicleData

from .columns import Plan
from .master import RestrictedMaster, add_column, build_master, purge_columns, solve_lp
from .pricer import VehiclePricer, _release_slot, build_pricer, price
from .preprocess import earliest_departures, seed_columns


@dataclass
class ColGenResult:
    """Outcome of ``run_column_generation``."""

    master: RestrictedMaster
    pricers: dict[tuple[int, int], VehiclePricer]  # (vehicle_id, pile) -> pricer
    iterations: int
    z_rmp_history: list[float]
    lower_bound_history: list[float]  # running max of the Lagrangian bound (32)
    best_lower_bound: float
    converged: bool  # True iff an exact pricing pass found no improving column
    earliest_departure_sum: float  # sum_j E_j -- a cheap sanity floor on the optimum
    columns_purged: int  # cumulative count removed by purge_columns (Section 8.3), 0 if disabled


def _log(progress: bool, message: str) -> None:
    if progress:
        print(message)


def _horizon_slots(delta: float, horizon_minutes: float) -> int:
    return math.ceil(round(horizon_minutes / delta, 9))


def _plan_reduced_cost(
    plan: Plan,
    *,
    pi: dict[tuple[int, int], float],
    mu: dict[tuple[int, int], float],
    pile: int,
) -> float:
    """Evaluate ``D_omega - sum(pi*alpha) - sum(mu*beta)`` (28) for a
    concrete plan directly off its own (start, departure, power), against
    whichever duals are passed in -- used both to price on the true duals
    directly and to re-check a smoothed-dual plan against the true ones."""
    if plan.is_null:
        return float(plan.departure)
    total = float(plan.departure)
    for k in plan.occupied_slots():
        total -= pi.get((pile, k), 0.0)
        total -= mu.get((pile, k), 0.0) * plan.power.get(k, 0.0)
    return total


def run_column_generation(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    extra_seed_columns: dict[int, list[Plan]] | None = None,
    conservative_modules: bool = False,
    gamma: float = 0.5,
    eps_rc: float = 1e-6,
    gap_tolerance: float = 1e-6,
    max_iterations: int = 500,
    max_columns_per_round: int | None = None,
    time_limit: float | None = None,
    pricer_mip_gap: float | None = 1e-4,
    max_workers: int | None = None,
    pricer_threads: int | None = 1,
    exact_mip_gap: float | None = 0.0,
    purge_every: int | None = 25,
    purge_threshold: float = 10.0,
    progress: bool = False,
) -> ColGenResult:
    """
    Section 7.1's loop. Builds the master and one persistent pricer per
    ``(vehicle, pile)`` pair, then alternates master-LP / pricing rounds
    until either of Section 7.4's stopping criteria is met: an exact
    (true-dual) pricing pass finds no column with reduced cost below
    ``-eps_rc`` (eq. 31) -- ``z_RMP == z_MP`` exactly -- or the bracket
    ``z_RMP - best_lower_bound`` closes to within ``gap_tolerance`` (only
    ever checked on an exact-pricing round, for the same reason the
    Lagrangian bound itself is only updated then -- see this module's own
    docstring). Also stops on ``max_iterations``/``time_limit``, in which
    case ``converged`` is False but ``best_lower_bound`` remains a valid
    certified lower bound regardless (Section 6.4's "anytime" property).

    Parameters
    ----------
    extra_seed_columns :
        Additional starting columns beyond ``preprocess.seed_columns``'s
        own defaults (null plan + per-pile greedy solo plan for every
        vehicle) -- e.g. from ``preprocess.columns_from_evs`` on a real
        simulation. Merged in before the loop starts; better seeds mean
        fewer iterations, never a different answer.
    conservative_modules :
        See ``master.build_master``.
    gamma :
        Dual smoothing weight (33); ``0`` disables smoothing entirely (then
        every round prices on the true duals, and the Lagrangian bound is
        updated every iteration).
    eps_rc :
        Reduced-cost threshold (31) for accepting a column as improving.
    gap_tolerance :
        Section 7.4's second stopping criterion: stop once ``z_RMP -
        best_lower_bound`` is at or below this (objective units, i.e. slot
        units -- ``delta*gap_tolerance/n_vehicles`` minutes of mean-sojourn
        uncertainty). The tight default (``1e-6``, effectively "exactly
        zero") only ever fires once the bound has genuinely closed; raise
        it (e.g. to ``n_vehicles`` for a roughly one-slot-of-delta
        tolerance) to stop earlier once *some* residual uncertainty is
        acceptable, which the source document notes can matter on a large,
        highly degenerate master where the last fraction of the gap is
        expensive to certify exactly.
    max_columns_per_round :
        Cap on how many improving columns to add per iteration (keeping
        the most negative reduced cost first); ``None`` adds all of them.
        Section 8.2 suggests ``J`` to ``3*J``.
    time_limit :
        Wall-clock budget for the whole loop, checked once per iteration
        (not mid-iteration) -- a soft, not hard, limit.
    pricer_mip_gap :
        MIP gap used only for smoothed-dual rounds hunting for candidate
        columns (``exact=False``) -- fine to leave loose since any column
        found there is re-checked against the *true* reduced cost before
        being added (Section 7.3), so a suboptimal pricer solve here can
        at worst miss a candidate, never corrupt anything. Rounds that
        determine ``exact_this_round`` (true, unsmoothed duals) always use
        ``exact_mip_gap`` instead, never this one -- see that parameter.
    exact_mip_gap :
        MIP gap for pricer solves on ``exact_this_round`` iterations --
        i.e. the ones whose ``zeta_jm`` values feed the Lagrangian bound
        (32) and decide convergence. Section 6.3's own warning: "the final
        exact pricing pass must use a gap of zero" -- a nonzero gap here
        lets Gurobi return an incumbent that is only an *upper bound* on
        the true pricer optimum, which silently invalidates ``best_lb`` as
        a lower bound (it can end up above the true optimum). Defaults to
        ``0.0``; do not loosen this without understanding that tradeoff.
    max_workers :
        How many ``(vehicle, pile)`` pricers to solve *concurrently* per
        round, via a thread pool -- see this module's own docstring for
        why threads (not processes) are the right tool here. ``None`` uses
        ``ThreadPoolExecutor``'s own default (typically
        ``min(32, os.cpu_count() + 4)``); pass ``1`` to solve strictly
        sequentially (e.g. for reproducible timing, or if you suspect a
        threading-related issue and want to rule it out). This is the
        single biggest lever for wall-clock time on a large instance
        (many vehicles/piles, or a long horizon making each individual
        pricer itself slow) -- pricing dominates iteration cost far more
        than the master LP does at that scale.
    pricer_threads :
        Gurobi's own ``Threads`` cap on each *individual* pricer solve.
        Left small (default ``1``) precisely because many pricers run at
        once under ``max_workers > 1``; raise it only if you also lower
        ``max_workers`` (e.g. fewer, fatter pricers instead of many thin
        ones), or set both to ``None`` to let Gurobi decide freely on each
        (reasonable only when solving pricers sequentially).
    purge_every :
        Section 8.3/8.2: sweep ``master.purge_columns`` every this many
        iterations (Section 8.2's suggested default, ``25``); ``None``
        disables purging entirely (the master then only ever grows, the
        behaviour before this parameter existed). Purging never changes the
        final answer -- a purged column can always be regenerated later if
        it becomes attractive again (see ``purge_columns``'s own
        docstring) -- it only keeps ``solve_lp``'s own cost from growing
        unbounded over a long run.
    purge_threshold :
        Reduced-cost cutoff (Section 8.2's suggested default, ``10``, in
        the same objective/slot units as the master's own objective) above
        which a non-basic column is swept by ``purge_columns``. Only
        consulted when ``purge_every`` is not ``None``.
    progress :
        Print one line per iteration if True. Also prints one extra line
        whenever a purge round actually removes at least one column.
    """
    K = _horizon_slots(delta, horizon_minutes)
    vehicle_ids = [v.id for v in vehicles]
    E = earliest_departures(vehicles, station, delta, horizon_minutes)
    e_sum = float(sum(E.values()))

    seeds = seed_columns(vehicles, station, delta, K)
    if extra_seed_columns:
        for j, plans in extra_seed_columns.items():
            seeds.setdefault(j, [])
            seeds[j].extend(p for p in plans if not p.is_null)  # null already seeded

    k_lo = min(_release_slot(v.a, delta) for v in vehicles)
    rm = build_master(
        vehicle_ids, station, delta, K, k_lo, seeds, conservative_modules=conservative_modules
    )

    pricers: dict[tuple[int, int], VehiclePricer] = {}
    for v in vehicles:
        for mm in range(station.n_piles):
            pricers[v.id, mm] = build_pricer(v, mm, station, delta, K, E[v.id])

    # Dual smoothing centre (33), zero-initialised; only meaningful once
    # gamma>0 and it's actually been set (see "use_smoothing" below --
    # smoothing is skipped on iteration 1, when there's nothing to smooth
    # toward yet, rather than blending real duals toward an artificial 0).
    center_pi: dict[tuple[int, int], float] = {}
    center_mu: dict[tuple[int, int], float] = {}

    z_rmp_history: list[float] = []
    lower_bound_history: list[float] = []
    best_lb = float("-inf")
    converged = False
    it = 0
    t_start = time.time()
    total_purged = 0

    def _price_one(key, price_pi, price_mu, mip_gap):
        z, plan = price(
            pricers[key], price_pi, price_mu, mip_gap=mip_gap, threads=pricer_threads
        )
        return key, z, plan

    # One pool for the whole run, not re-created per iteration -- pricer
    # solves are cheap-ish to dispatch but not free, and there's no reason
    # to pay thread-pool startup/teardown cost every round.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:

        def _price_round(price_pi, price_mu, *, exact: bool):
            """One pricing pass over every (vehicle, pile), dispatched
            concurrently across ``executor``. Returns (candidates, zeta)
            where zeta[j][m] is only populated -- and only meaningful for
            the Lagrangian bound -- when exact=True (i.e. price_pi/price_mu
            are the true, unsmoothed duals). ``exact`` also selects the MIP
            gap: candidate-hunting rounds use the loose ``pricer_mip_gap``,
            but a round whose zeta values may feed the Lagrangian bound
            must be solved to (near-)proven optimality -- ``exact_mip_gap``
            -- since a nonzero gap would make ``zeta`` merely an upper
            bound on the true pricer optimum, silently invalidating (32)."""
            mip_gap = exact_mip_gap if exact else pricer_mip_gap
            candidates: list[tuple[float, int, Plan]] = []
            zeta: dict[int, dict[int, float]] = {j: {} for j in vehicle_ids}
            futures = [
                executor.submit(_price_one, key, price_pi, price_mu, mip_gap) for key in pricers
            ]
            for future in futures:
                (j, mm), z, plan = future.result()
                if exact:
                    rc = z - lp.sigma[j]
                    zeta[j][mm] = z
                else:
                    true_z = _plan_reduced_cost(plan, pi=lp.pi, mu=lp.mu, pile=mm)
                    rc = true_z - lp.sigma[j]
                if not plan.is_null and rc < -eps_rc:
                    candidates.append((rc, j, plan))
            return candidates, zeta

        while it < max_iterations:
            it += 1
            if time_limit is not None and time.time() - t_start > time_limit:
                _log(progress, f"[colgen] time_limit={time_limit}s reached at iteration {it}.")
                break

            lp = solve_lp(rm)
            z_rmp_history.append(lp.objective)

            # Section 8.3: sweep non-basic, unattractive columns every
            # purge_every iterations, right after this solve (not after
            # this round's new columns are added -- see purge_columns's own
            # docstring for why) so every .RC/.VBasis read is meaningful.
            if purge_every is not None and it % purge_every == 0:
                n_purged = purge_columns(rm, threshold=purge_threshold)
                total_purged += n_purged
                if n_purged:
                    _log(
                        progress,
                        f"[colgen] iter {it}: purged {n_purged} column(s) "
                        f"(RC > {purge_threshold:g}), "
                        f"total_columns={sum(len(v) for v in rm.columns.values())}",
                    )

            use_smoothing = gamma > 0.0 and it > 1
            if use_smoothing:
                price_pi = {
                    k: gamma * center_pi.get(k, 0.0) + (1 - gamma) * v for k, v in lp.pi.items()
                }
                price_mu = {
                    k: gamma * center_mu.get(k, 0.0) + (1 - gamma) * v for k, v in lp.mu.items()
                }
                candidates, zeta = _price_round(price_pi, price_mu, exact=False)
                exact_this_round = False
                if not candidates:
                    # Section 7.3: smoothed pricing found nothing -- re-price
                    # once on the true duals before concluding convergence.
                    candidates, zeta = _price_round(lp.pi, lp.mu, exact=True)
                    exact_this_round = True
            else:
                candidates, zeta = _price_round(lp.pi, lp.mu, exact=True)
                exact_this_round = True

            if exact_this_round:
                lb = (
                    rm.station.n_connectors * sum(lp.pi.values())
                    + station.n_modules * station.p_module * sum(lp.mu.values())
                    + sum(min(zeta[j].values()) for j in vehicle_ids)
                )
                if lb > best_lb:
                    best_lb = lb
                    center_pi, center_mu = dict(lp.pi), dict(lp.mu)
            lower_bound_history.append(best_lb)

            candidates.sort(key=lambda t: t[0])
            if max_columns_per_round is not None:
                candidates = candidates[:max_columns_per_round]
            # add_column returns None for a duplicate (Section 8.3) -- track
            # how many candidates actually became new master columns, since a
            # round where every "candidate" turns out to already exist carries
            # no new information either, regardless of how it's logged.
            n_added = sum(1 for _rc, j, plan in candidates if add_column(rm, plan) is not None)

            _log(
                progress,
                f"[colgen] iter {it}: z_RMP={lp.objective:.3f}, best_LB={best_lb:.3f}, "
                f"gap={lp.objective - best_lb:.3f}, exact={exact_this_round}, "
                f"candidates={len(candidates)}, columns_added={n_added}, "
                f"total_columns={sum(len(v) for v in rm.columns.values())}",
            )

            if exact_this_round and lp.objective - best_lb <= gap_tolerance:
                # Section 7.4's second stopping criterion: the bracket has
                # closed to within tolerance, whether or not z_RMP == z_MP
                # exactly -- checked only on an exact-pricing round, the same
                # gating the Lagrangian bound itself uses (see this module's
                # own docstring on why a smoothed round's bound can't be
                # trusted for this).
                converged = True
                _log(
                    progress,
                    f"[colgen] converged after {it} iteration(s): gap "
                    f"{lp.objective - best_lb:.6g} <= gap_tolerance={gap_tolerance:g}.",
                )
                break

            if not candidates or (exact_this_round and n_added == 0):
                # Either an exact pricing pass found nothing (genuine
                # convergence, z_RMP == z_MP), or it found only columns already
                # present (no new information, even though eps_rc nominally
                # called them improving -- treat the same way).
                converged = True
                _log(progress, f"[colgen] converged after {it} iteration(s): z_RMP == z_MP.")
                break

    return ColGenResult(
        master=rm,
        pricers=pricers,
        iterations=it,
        z_rmp_history=z_rmp_history,
        lower_bound_history=lower_bound_history,
        best_lower_bound=best_lb,
        converged=converged,
        earliest_departure_sum=e_sum,
        columns_purged=total_purged,
    )
