"""
Exact branch-and-price for the connector-lane model with whole modules.

Driver: ``BranchAndPrice(...).solve()`` (or ``solve_branch_and_price``).
The tree is managed here; Gurobi only ever solves one master LP or one
pricing MILP at a time.

Per node
--------
1. Activate the columns the node's restrictions allow; impose the same
   restrictions in every pricer.
2. Phase II column generation. If the restricted master is infeasible
   (e.g. every column of some vehicle was cut by the branch), run Phase I
   instead of pruning: minimise the artificial total, pricing with the
   Phase-I duals, until either it reaches zero (feasible; back to Phase II)
   or a Phase-I Lagrangian bound proves it cannot (the node is infeasible).
3. Each Phase-II round prices every allowed (vehicle, pile) and computes
   the node's Lagrangian bound from the pricers' ``ObjBound``s. The node is
   pruned as soon as its bound, rounded up to the next attainable objective
   value, reaches ``UB`` -- without waiting for convergence. The objective
   is total sojourn ``sum_j w_j (delta*D_j - a_j)`` (minutes), which with
   0/1 weights is ``delta * n - sum_j w_j a_j`` for an integer ``n``, so
   attainable values are ``delta`` apart (``objective_index``).
4. At convergence: if the LP solution is integer-recoverable, it is an
   incumbent and the node closes; otherwise branch (``branching``).

The pruning bound is always a Lagrangian bound, never the restricted
master's LP value, and every incumbent passes ``validation`` -- see
``tolerances.py`` for why these two rules are what keep the answer exact.
"""

from __future__ import annotations

import heapq
import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import gurobipy as gp

from offline_cl_opt.boundary import COHORTS_ALL, BoundaryMode, BoundaryVehicle, Cohort
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import _release_slot, earliest_departures

from .branching import Branch, positive_support, select_branch, try_recover
from .columns import PBPlan, fixed_boundary_plan, modules_needed, null_pb_plan
from .heuristics import greedy_list_schedule, greedy_plan
from .labeling import LabelingPricer
from .master import MasterLP, MasterNumericalError, PBMaster
from .pricer import INFEASIBLE, TIME_LIMIT, PBPricer, PricingResult, reduced_cost
from .restrictions import NodeRestrictions
from .tolerances import EPS_FEAS, EPS_RC, prune_slack
from .validation import ScheduleValidationError, validate_plan, validate_schedule


class BranchAndPriceError(RuntimeError):
    """An internal inconsistency; raised instead of returning a doubtful answer."""


@dataclass(order=True)
class _QueueItem:
    lower_bound: float
    neg_depth: int
    node_id: int
    node: "Node" = field(compare=False)


@dataclass
class Node:
    node_id: int
    parent_id: int | None
    depth: int
    restrictions: NodeRestrictions
    lower_bound: float  # valid lower bound for every schedule in this node
    branch: str = "root"


@dataclass
class NodeRecord:
    node_id: int
    parent_id: int | None
    depth: int
    branch: str
    status: str  # "branched" | "pruned_bound" | "integer" | "infeasible" | "aborted"
    lower_bound: float
    lp_value: float | None
    phase2_iterations: int
    phase1_iterations: int
    columns_after: int
    upper_bound_after: float
    seconds: float


@dataclass
class _NodeOutcome:
    status: str  # "converged" | "pruned_bound" | "infeasible" | "aborted"
    lower_bound: float
    lp: MasterLP | None = None
    phase2_iterations: int = 0
    phase1_iterations: int = 0


@dataclass
class BPStats:
    nodes_processed: int = 0
    nodes_pruned_before_solve: int = 0
    pricing_calls: int = 0
    phase2_iterations: int = 0
    phase1_iterations: int = 0
    borderline_columns: int = 0
    labeling_fallbacks: int = 0  # labelling calls settled by the MILP pricer instead
    dive_node_solves: int = 0  # column-generation solves spent inside heuristic dives
    time_master: float = 0.0
    time_pricing: float = 0.0
    time_heuristic: float = 0.0
    time_validation: float = 0.0


class BranchAndPrice:
    """
    Parameters
    ----------
    vehicles, station, delta, horizon_minutes, boundary_vehicles, cohorts,
    objective_cohorts :
        Exactly as for ``offline_cl_opt.model.build_cl_model``; the model
        solved is that compact model with whole modules
        (``relax_modules=False``).
    initial_schedule :
        Optional ``{vehicle_id: PBPlan}`` known to be feasible (e.g.
        ``heuristics.schedule_from_compact`` on a time-limited compact
        solve). Validated first; an invalid one raises.
    time_limit, node_limit :
        Stop early; the returned bracket ``[lower_bound, objective]`` is
        still valid, only ``status`` is no longer ``"OPTIMAL"``.
    rim_time_limit, rim_every_nodes :
        Restricted integer master heuristic (binary master over the columns
        generated so far): once after the root and then every
        ``rim_every_nodes`` processed nodes (``None``: root only). Only
        ever improves the incumbent; ``rim_time_limit=0`` disables it.
    max_workers :
        Pricing threads. Each worker owns its own Gurobi environment and
        only ever solves its own pricers (Gurobi environments are not
        thread-safe). ``1`` prices sequentially in the default environment.
    pricer_threads :
        Gurobi ``Threads`` for each pricing MILP.
    pricer :
        ``"labeling"`` (default): exact forward labelling (``labeling.py``),
        falling back to the MILP on label explosion. ``"milp"``: always the
        pricing MILP (``pricer.py``). Both return the exact pricing minimum.
    compact_check :
        Evaluate every row of ``offline_cl_opt``'s compact model at the
        returned schedule (``PBSolution.compact_check``).
    pricing_gap_abs, columns_per_pricer :
        Speed knobs that never affect the answer. A pricing MILP may stop
        once within ``pricing_gap_abs`` minutes of optimal (default ``None``:
        half a departure slot, ``0.5 * delta``; it is re-solved exactly
        whenever that leaves the pricing question open), and up to
        ``columns_per_pricer`` improving columns are taken from its solution
        pool per round.
    early_branching :
        Stop a node's column generation as soon as its Lagrangian bound and
        its restricted-master value round up to the same attainable
        objective value (further pricing could not raise the node's rounded
        bound), and branch.
        Never affects the answer.
    dive_every, dive_time_limit, dive_backtracks :
        Primal heuristic: before the root and then every ``dive_every``
        processed nodes, a column-fixing dive from the best open node
        (repeatedly fix one fractional vehicle to a heavy served column's
        timetable and re-price; on an infeasible or hopeless fixing, back
        up and try the next candidate, at most ``dive_backtracks`` times)
        looks for an integer schedule within ``dive_time_limit`` seconds.
        ``dive_every=None`` disables it. It only ever offers validated
        incumbents; never affects the answer.
    """

    def __init__(
        self,
        vehicles: list[VehicleData],
        station: StationSpec,
        delta: float,
        horizon_minutes: float,
        *,
        boundary_vehicles: dict[int, BoundaryVehicle] | None = None,
        cohorts: dict[int, Cohort] | None = None,
        objective_cohorts: frozenset[Cohort] | None = None,
        initial_schedule: dict[int, PBPlan] | None = None,
        time_limit: float | None = None,
        node_limit: int | None = None,
        rim_time_limit: float = 30.0,
        rim_every_nodes: int | None = 50,
        max_workers: int = 1,
        pricer_threads: int | None = 1,
        pricer: str = "labeling",
        compact_check: bool = True,
        pricing_gap_abs: float | None = None,
        columns_per_pricer: int = 3,
        early_branching: bool = True,
        dive_every: int | None = 20,
        dive_time_limit: float = 60.0,
        dive_backtracks: int = 6,
        progress: bool = False,
        log_every_iterations: int = 10,
    ) -> None:
        if not vehicles:
            raise ValueError("need at least one vehicle")
        if delta <= 0 or horizon_minutes <= 0:
            raise ValueError("delta and horizon_minutes must be positive")
        ids = [v.id for v in vehicles]
        if len(set(ids)) != len(ids):
            raise ValueError("vehicle ids must be unique")
        self.vehicles = list(vehicles)
        self.by_id = {v.id: v for v in vehicles}
        self.station = station
        self.delta = delta
        self.horizon_minutes = horizon_minutes
        self.K = math.ceil(round(horizon_minutes / delta, 9))
        self.boundary = dict(boundary_vehicles or {})
        unknown = set(self.boundary) - set(self.by_id)
        if unknown:
            raise ValueError(f"boundary_vehicles lists unknown vehicle ids {sorted(unknown)}")
        self.cohorts = dict(cohorts or {})
        self.objective_cohorts = objective_cohorts if objective_cohorts is not None else COHORTS_ALL
        self.weights = {
            v.id: 1.0 if self.cohorts.get(v.id, Cohort.MEASUREMENT) in self.objective_cohorts else 0.0
            for v in vehicles
        }
        if not any(self.weights.values()):
            raise ValueError("objective_cohorts selects no vehicle; the objective would be empty")
        # Objective = delta * sum_j w_j D_j - arrival_total (minutes).
        self.arrival_total = float(sum(self.weights[v.id] * v.a for v in vehicles))
        self.releases = {v.id: _release_slot(v.a, delta) for v in vehicles}
        for j, k0 in self.releases.items():
            if k0 > self.K - 1:
                raise ValueError(f"Vehicle {j}: release slot {k0} is outside the horizon (K={self.K}).")
        for j, bv in self.boundary.items():
            if self.releases[j] != 0:
                raise ValueError(f"boundary vehicle {j} must have a=0 (release slot 0)")
            if not (0 <= bv.pile < station.n_piles and 0 <= bv.connector < station.n_connectors):
                raise ValueError(f"boundary vehicle {j} sits on a lane outside the station")

        self.time_limit = time_limit
        self.node_limit = node_limit
        self.rim_time_limit = rim_time_limit
        self.rim_every_nodes = rim_every_nodes
        self.pricer_threads = pricer_threads
        self.run_compact_check = compact_check
        if pricing_gap_abs is None:
            pricing_gap_abs = 0.5 * delta  # half a departure slot, in minutes
        if pricing_gap_abs < 0 or columns_per_pricer < 1:
            raise ValueError("pricing_gap_abs must be >= 0 and columns_per_pricer >= 1")
        self.pricing_gap_abs = float(pricing_gap_abs)
        self.columns_per_pricer = int(columns_per_pricer)
        self.early_branching = early_branching
        self.dive_every = dive_every if dive_every else None
        self.dive_time_limit = float(dive_time_limit)
        self.dive_backtracks = max(0, int(dive_backtracks))
        self.progress = progress
        self.log_every_iterations = max(1, int(log_every_iterations))
        self.n_workers = max(1, int(max_workers))
        self.slack = prune_slack(len(vehicles))
        self.stats = BPStats()
        self.records: list[NodeRecord] = []

        self._t0 = time.perf_counter()
        # Bound history: the certified bracket [global lower bound, incumbent]
        # over time -- see ``_note_bounds``.
        self.bound_history: list[dict] = []
        self._global_lb = -math.inf
        self._open_min = math.inf  # least bound among OTHER open nodes, while one is being solved
        self._deadline = self._t0 + time_limit if time_limit is not None else None

        # Environments: the master (and a sequential pricer set) use Gurobi's
        # default environment; parallel workers each get their own.
        self._envs: list[gp.Env] = []
        self._executor: ThreadPoolExecutor | None = None
        if self.n_workers > 1:
            for _ in range(self.n_workers):
                env = gp.Env(empty=True)
                env.setParam("OutputFlag", 0)
                env.start()
                self._envs.append(env)
            self._executor = ThreadPoolExecutor(max_workers=self.n_workers)

        self.fixed_plans: dict[int, PBPlan] = {
            j: fixed_boundary_plan(bv, station) for j, bv in self.boundary.items() if bv.mode is BoundaryMode.FIXED
        }
        E = earliest_departures(self.vehicles, station, delta, horizon_minutes)
        k_lo = min(self.releases.values())
        self.master = PBMaster(
            ids, self.weights, station, self.K, k_lo, delta=delta, arrivals={v.id: v.a for v in vehicles}
        )

        if pricer not in ("labeling", "milp"):
            raise ValueError(f"pricer must be 'labeling' or 'milp', got {pricer!r}")
        pricer_cls = LabelingPricer if pricer == "labeling" else PBPricer
        self.pricers: dict[tuple[int, int], PBPricer | LabelingPricer] = {}
        self.pricer_group: dict[tuple[int, int], int] = {}
        for v in self.vehicles:
            bv = self.boundary.get(v.id)
            if bv is not None and bv.mode is BoundaryMode.FIXED:
                continue
            piles = [bv.pile] if bv is not None else list(range(station.n_piles))
            for m in piles:
                key = (v.id, m)
                group = len(self.pricers) % self.n_workers
                self.pricers[key] = pricer_cls(
                    v,
                    m,
                    station,
                    delta,
                    self.K,
                    0 if bv is not None else E[v.id],
                    initial_energy=bv.initial_energy_kwh if bv is not None else 0.0,
                    forced_start=bv is not None,
                    weight=self.weights[v.id],
                    env=self._envs[group] if self._envs else None,
                    threads=pricer_threads,
                )
                self.pricer_group[key] = group

        self.upper_bound = math.inf
        self.incumbent: dict[int, PBPlan] | None = None
        self.incumbent_connectors: dict[int, int] = {}
        self.incumbent_source = ""
        self._seed(initial_schedule)

    # ------------------------------------------------------------------ #
    # set-up helpers
    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.progress:
            print(f"[B&P {time.perf_counter() - self._t0:8.1f}s] {msg}", flush=True)

    def _time_left(self) -> float | None:
        return None if self._deadline is None else self._deadline - time.perf_counter()

    def _out_of_time(self) -> bool:
        left = self._time_left()
        return left is not None and left <= 0.0

    def _check_plan(self, plan: PBPlan) -> list[str]:
        return validate_plan(
            plan, self.by_id[plan.vehicle_id], self.boundary.get(plan.vehicle_id), self.station, self.delta, self.K
        )

    def _add_column(self, plan: PBPlan, *, active: bool) -> tuple[int, bool]:
        errs = self._check_plan(plan)
        if errs:
            raise BranchAndPriceError(f"refusing an invalid column: {errs[:3]}")
        return self.master.add_column(plan, active=active)

    def _seed(self, initial_schedule: dict[int, PBPlan] | None) -> None:
        """Mandatory columns, cheap seed columns, and the initial incumbent."""
        N = self.station.n_modules
        for v in self.vehicles:
            j = v.id
            bv = self.boundary.get(j)
            if bv is not None and bv.mode is BoundaryMode.FIXED:
                self._add_column(self.fixed_plans[j], active=True)
                continue
            if bv is None:
                self._add_column(null_pb_plan(j, self.K), active=True)
                for m in range(self.station.n_piles):
                    plan = greedy_plan(v, m, self.releases[j], self.station, self.delta, self.K, lambda k: N)
                    if plan is not None:
                        self._add_column(plan, active=True)
            else:
                plan = greedy_plan(
                    v, bv.pile, 0, self.station, self.delta, self.K, lambda k: N,
                    initial_energy=bv.initial_energy_kwh,
                )
                if plan is not None:
                    self._add_column(plan, active=True)
                # The simulation's realized trajectory, with the whole modules
                # it needs, when that is a valid column on its own.
                if bv.seed_power and bv.seed_departure_slot > 0:
                    dep = min(self.K, bv.seed_departure_slot)
                    seed = PBPlan(
                        vehicle_id=j,
                        pile=bv.pile,
                        start=0,
                        departure=dep,
                        power={k: p for k, p in bv.seed_power.items() if k < dep and p > 0.0},
                        modules={k: modules_needed(bv.seed_power.get(k, 0.0), self.station.p_module) for k in range(dep)},
                    )
                    if not self._check_plan(seed):
                        self._add_column(seed, active=True)

        greedy = greedy_list_schedule(self.vehicles, self.station, self.delta, self.K, self.boundary)
        self._offer(greedy, "greedy list schedule", strict=True)
        if initial_schedule is not None:
            self._offer(dict(initial_schedule), "initial schedule", strict=True)

    # ------------------------------------------------------------------ #
    # incumbents
    # ------------------------------------------------------------------ #
    def _offer(self, schedule: dict[int, PBPlan], source: str, *, strict: bool) -> bool:
        """Validate ``schedule``; adopt it if it beats ``UB``. Its columns join the pool."""
        t = time.perf_counter()
        try:
            check = validate_schedule(
                schedule, self.vehicles, self.station, self.delta, self.K, self.boundary, self.weights
            )
        except ScheduleValidationError as exc:
            self.stats.time_validation += time.perf_counter() - t
            if strict:
                raise ScheduleValidationError(f"{source} is not feasible: {exc}") from exc
            return False
        self.stats.time_validation += time.perf_counter() - t
        for plan in schedule.values():
            self.master.add_column(plan, active=False)
        # Attainable objective values are delta apart: a genuinely better
        # schedule is at least one lattice step below UB.
        if check.objective < self.upper_bound - 0.5 * self.delta:
            self.upper_bound = check.objective
            self.incumbent = dict(schedule)
            self.incumbent_connectors = check.connectors
            self.incumbent_source = source
            self._log(f"incumbent {check.objective:.6g} from {source}")
            self._note_bounds(f"incumbent: {source}")
            return True
        return False

    def _note_bounds(self, event: str, lower: float | None = None, *, force: bool = False) -> None:
        """
        Append ``(time, global lower bound, incumbent)`` to ``bound_history``.

        ``lower`` is a lower bound valid for the WHOLE problem at this moment
        (the least bound over every open node, or the root's bound while it
        is the only node). A lower bound stays valid forever, so the recorded
        value is the running maximum: the history is a non-decreasing lower
        curve and a non-increasing upper curve. Unchanged points are skipped
        unless ``force``.

        The recorded lower bound is capped at the incumbent: once the
        incumbent beats every open node's bound, those nodes are only waiting
        to be pruned, and the certified bound is ``min(open nodes, incumbent)``
        -- never above the incumbent (which would be a negative gap).
        """
        if lower is not None and lower > self._global_lb:
            self._global_lb = lower
        ub = self.upper_bound
        lb = min(self._global_lb, ub)
        if not force and self.bound_history:
            last = self.bound_history[-1]
            if last["lower_bound"] == lb and last["upper_bound"] == ub:
                return
        self.bound_history.append(
            {
                "time_s": time.perf_counter() - self._t0,
                "lower_bound": lb,
                "upper_bound": ub,
                "nodes": self.stats.nodes_processed,
                "event": event,
            }
        )

    def objective_index(self, value: float, *, bound: bool) -> int:
        """
        Position of ``value`` on the lattice of attainable objective values
        ``delta * n - arrival_total`` (``n = sum_j w_j D_j``, an integer).

        ``bound=True``: ``value`` is a lower bound; returns the smallest
        ``n`` it allows, ``ceil((value - slack + arrival_total) / delta)``.
        The slack absorbs floating-point noise (``tolerances.prune_slack``).
        ``bound=False``: ``value`` is an attained objective (a schedule's
        total sojourn); returns its own ``n`` (rounded to the nearest).
        """
        n = (value + self.arrival_total) / self.delta
        if bound:
            return math.ceil(round(n - self.slack / self.delta, 9))
        return round(n)

    def _prunable(self, lower_bound: float) -> bool:
        if lower_bound == math.inf:
            return True
        if not math.isfinite(lower_bound) or not math.isfinite(self.upper_bound):
            return False
        return self.objective_index(lower_bound, bound=True) >= self.objective_index(self.upper_bound, bound=False)

    # ------------------------------------------------------------------ #
    # pricing
    # ------------------------------------------------------------------ #
    def _price_all(self, lp: MasterLP, *, phase: int) -> tuple[dict[tuple[int, int], PricingResult], bool]:
        """Price every active pricer; returns (results, aborted_on_time)."""
        keys = [key for key, pr in self.pricers.items() if pr.active]
        include_departure = phase == 2

        def run(batch: list[tuple[int, int]]) -> list[tuple[tuple[int, int], PricingResult]]:
            out = []
            for key in batch:
                left = self._time_left()
                if left is not None and left <= 0.0:
                    out.append((key, PricingResult(key[0], key[1], TIME_LIMIT, -math.inf, None, None, None, False)))
                    continue
                pr = self.pricers[key]
                out.append(
                    (
                        key,
                        pr.price(
                            lp.pi,
                            lp.mu,
                            lp.sigma[key[0]],
                            include_departure=include_departure,
                            eps_rc=EPS_RC,
                            time_limit=left,
                            gap_abs=self.pricing_gap_abs,
                            max_columns=self.columns_per_pricer,
                        ),
                    )
                )
            return out

        t = time.perf_counter()
        results: dict[tuple[int, int], PricingResult] = {}
        if self._executor is None:
            for key, res in run(keys):
                results[key] = res
        else:
            batches = [[key for key in keys if self.pricer_group[key] == g] for g in range(self.n_workers)]
            futures = [self._executor.submit(run, batch) for batch in batches if batch]
            for fut in futures:
                for key, res in fut.result():
                    results[key] = res
        self.stats.time_pricing += time.perf_counter() - t
        self.stats.pricing_calls += len(keys)
        aborted = any(res.status == TIME_LIMIT for res in results.values())
        return results, aborted

    def _lagrangian_bound(
        self, lp: MasterLP, results: dict[tuple[int, int], PricingResult], restrictions: NodeRestrictions, *, phase: int
    ) -> float:
        """
        ``C*sum(pi) + N*sum(mu) + sum_j min_{allowed plans} (cost - pi*alpha - mu*q)``.
        Valid for any pi, mu <= 0 (the master clamps them) when each minimum
        is replaced by a lower bound -- here the pricers' ``ObjBound``.
        """
        C, N = self.station.n_connectors, self.station.n_modules
        total = C * sum(lp.pi.values()) + N * sum(lp.mu.values())
        for v in self.vehicles:
            j = v.id
            if j in self.fixed_plans:
                plan = self.fixed_plans[j]
                cost = self.master.column_cost(plan) if phase == 2 else 0.0
                total += reduced_cost(plan, cost, lp.pi, lp.mu, 0.0)
                continue
            best = math.inf
            r = restrictions.get(j)
            if j not in self.boundary and r.allows_null(self.K, j):
                best = self.master.column_cost(null_pb_plan(j, self.K)) if phase == 2 else 0.0
            for m in range(self.station.n_piles):
                res = results.get((j, m))
                if res is not None and res.status != INFEASIBLE:
                    best = min(best, res.bound)
            if best == math.inf:
                return math.inf
            total += best
        return total

    def _add_improving(self, key: tuple[int, int], plan: PBPlan, rc: float, restrictions: NodeRestrictions) -> None:
        """Insert (or re-activate) a column with reduced cost < -EPS_RC."""
        if not restrictions.column_satisfies(plan, self.K):
            raise BranchAndPriceError(f"pricer {key} returned a column the node forbids")
        index, new = self._add_column(plan, active=True)
        if not new:
            if self.master.is_active(index):
                # Same signature = same master coefficients = same reduced cost,
                # which the LP optimum says is >= 0 for every active column.
                raise BranchAndPriceError(
                    f"pricer {key} found reduced cost {rc:.3g} for a column already active in "
                    "the master -- master and pricer disagree."
                )
            # Stored earlier (e.g. with an incumbent) but not yet activated at
            # this node; it satisfies the node, so use it.
            self.master.set_active(index, True)

    def _add_priced_columns(self, results: dict[tuple[int, int], PricingResult], restrictions: NodeRestrictions) -> int:
        """Add improving columns; returns how many went in (new or re-activated)."""
        added = 0
        for key, res in results.items():
            if res.status == INFEASIBLE:
                continue
            plan, rc = res.plan, res.reduced_cost
            for other, other_rc in res.extra:
                self._add_improving(key, other, other_rc, restrictions)
                added += 1
            if plan is not None and rc is not None and rc < -EPS_RC:
                self._add_improving(key, plan, rc, restrictions)
                added += 1
            elif res.certified:
                continue
            elif plan is not None and rc is not None and rc < 0.0:
                # Within tolerance of the threshold: add if new (progress), else
                # accept as converged. The node bound never relies on this.
                index, new = self._add_column(plan, active=True)
                self.stats.borderline_columns += 1
                if new:
                    added += 1
                elif not self.master.is_active(index):
                    self.master.set_active(index, True)
                    added += 1
            else:
                raise BranchAndPriceError(
                    f"pricer {key}: neither an improving column nor a certificate "
                    f"(status={res.status}, bound={res.bound}, objective={res.objective})"
                )
        return added

    # ------------------------------------------------------------------ #
    # node solve
    # ------------------------------------------------------------------ #
    def _solve_master(self) -> MasterLP:
        t = time.perf_counter()
        lp = self.master.solve()
        self.stats.time_master += time.perf_counter() - t
        return lp

    def _phase1(self, node: Node) -> tuple[str, int]:
        """Returns ("feasible" | "infeasible" | "aborted", iterations)."""
        self.master.set_phase(1)
        iters = 0
        try:
            while True:
                if self._out_of_time():
                    return "aborted", iters
                lp = self._solve_master()
                iters += 1
                self.stats.phase1_iterations += 1
                if lp.objective <= EPS_FEAS:
                    return "feasible", iters
                results, aborted = self._price_all(lp, phase=1)
                if aborted:
                    return "aborted", iters
                bound = self._lagrangian_bound(lp, results, node.restrictions, phase=1)
                if bound > EPS_FEAS:
                    return "infeasible", iters
                if self._add_priced_columns(results, node.restrictions) == 0:
                    raise BranchAndPriceError(
                        f"Phase I at node {node.node_id} stalled: artificial total {lp.objective:.3g} > "
                        f"{EPS_FEAS} but no improving column and Lagrangian bound {bound:.3g}."
                    )
        finally:
            self.master.set_phase(2)

    def _solve_node(self, node: Node) -> _NodeOutcome:
        self.master.activate(node.restrictions)
        self.master.set_phase(2)
        for (j, _m), pr in self.pricers.items():
            pr.apply(node.restrictions.get(j))
        lower = node.lower_bound
        p2 = p1 = 0
        just_left_phase1 = False
        while True:
            if self._out_of_time():
                return _NodeOutcome("aborted", lower, None, p2, p1)
            lp = self._solve_master()
            if not lp.feasible:
                if just_left_phase1:
                    raise MasterNumericalError(f"node {node.node_id}: Phase I succeeded but Phase II is infeasible")
                status, iters = self._phase1(node)
                p1 += iters
                if status != "feasible":
                    return _NodeOutcome(status, lower if status == "aborted" else math.inf, None, p2, p1)
                just_left_phase1 = True
                continue
            just_left_phase1 = False
            p2 += 1
            self.stats.phase2_iterations += 1

            recovery = try_recover(
                positive_support(lp, self.master), self.vehicles, self.station, self.delta, self.K, self.boundary
            )
            if recovery.schedule is not None:
                self._offer(recovery.schedule, f"node {node.node_id} LP", strict=not recovery.averaged)

            t_price = time.perf_counter()
            results, aborted = self._price_all(lp, phase=2)
            t_price = time.perf_counter() - t_price
            if aborted:
                return _NodeOutcome("aborted", lower, lp, p2, p1)
            lower = max(lower, self._lagrangian_bound(lp, results, node.restrictions, phase=2))
            if node.node_id >= 0:  # tree nodes only: a dive's bound is not a global bound
                self._note_bounds("node", min(lower, self._open_min))
            if self._prunable(lower):
                return _NodeOutcome("pruned_bound", lower, lp, p2, p1)
            added = self._add_priced_columns(results, node.restrictions)
            # The node's true LP value lies in [lower, z_RMP], so once both round
            # up to the same attainable objective value no further pricing can
            # raise the node's rounded bound: branch now (the bound used stays
            # ``lower``).
            early = (
                added > 0
                and self.early_branching
                and self.objective_index(lower, bound=True) >= self.objective_index(lp.objective, bound=True)
            )
            if self.progress and (p2 == 1 or p2 % self.log_every_iterations == 0 or added == 0 or early):
                self._log(
                    f"  node {node.node_id} iter {p2}: z_RMP={lp.objective:.4f} LB={lower:.4f} "
                    f"UB={self.upper_bound:.6g} added={added} cols={len(self.master.columns)} "
                    f"pricing={t_price:.2f}s" + (" (early branch)" if early else "")
                )
            if added == 0 or early:
                return _NodeOutcome("converged", lower, lp, p2, p1)

    # ------------------------------------------------------------------ #
    # restricted integer master
    # ------------------------------------------------------------------ #
    def _run_rim(self, label: str) -> None:
        if self.rim_time_limit is not None and self.rim_time_limit <= 0:
            return
        limit = self.rim_time_limit
        left = self._time_left()
        if left is not None:
            if left <= 0:
                return
            limit = left if limit is None else min(limit, left)
        t = time.perf_counter()
        self.master.activate(NodeRestrictions())
        chosen = self.master.solve_integer_restricted(
            time_limit=limit,
            cutoff=self.upper_bound - 0.5 * self.delta if math.isfinite(self.upper_bound) else None,
        )
        self.stats.time_heuristic += time.perf_counter() - t
        status, solcount, runtime = self.master.last_rim_status
        self._log(
            f"restricted integer master ({label}): {len(self.master.columns)} columns, "
            f"Gurobi status {status}, {solcount} solution(s), {runtime:.1f}s"
        )
        if chosen is not None:
            self._offer(chosen, f"restricted integer master ({label})", strict=True)

    # ------------------------------------------------------------------ #
    # the tree
    # ------------------------------------------------------------------ #
    def solve(self):
        from .solution import build_solution

        try:
            status = self._search()
        finally:
            self._shutdown()
        return build_solution(self, status, run_compact_check=self.run_compact_check)

    def _search(self) -> str:
        """
        Best-bound search (deeper first on ties). Before the root and then
        every ``dive_every`` processed nodes, a column-fixing dive
        (``_column_fixing_dive``) is run from the best open node to find
        incumbents; it never changes the tree.
        """
        root = Node(0, None, 0, NodeRestrictions(), -math.inf)
        queue: list[_QueueItem] = [_QueueItem(root.lower_bound, 0, 0, root)]
        last_dive = -(self.dive_every or 0)  # so a dive runs before the root
        next_id = 1
        self.root_lower_bound = -math.inf
        self.root_lp_value = math.nan
        self.open_bounds_at_stop: list[float] = []

        while queue:
            # Every open node is in the heap here, so its least bound is global.
            self._note_bounds("tree", queue[0].lower_bound)
            if self._out_of_time():
                self.open_bounds_at_stop = [it.lower_bound for it in queue]
                return "TIME_LIMIT"
            if self.node_limit is not None and self.stats.nodes_processed >= self.node_limit:
                self.open_bounds_at_stop = [it.lower_bound for it in queue]
                return "NODE_LIMIT"
            if self.dive_every and self.stats.nodes_processed - last_dive >= self.dive_every:
                last_dive = self.stats.nodes_processed
                self._column_fixing_dive(queue[0].node.restrictions, f"from node {queue[0].node.node_id}")
                continue  # re-check limits; the queue is unchanged
            item = heapq.heappop(queue)
            node = item.node
            if self._prunable(node.lower_bound):
                self.stats.nodes_pruned_before_solve += 1
                continue

            t = time.perf_counter()
            self._open_min = queue[0].lower_bound if queue else math.inf
            outcome = self._solve_node(node)
            self.stats.nodes_processed += 1
            status = outcome.status
            lp_value = outcome.lp.objective if outcome.lp is not None and outcome.lp.feasible else None
            if node.node_id == 0 and status != "aborted":
                self.root_lower_bound = outcome.lower_bound
                self.root_lp_value = lp_value if lp_value is not None else math.nan

            if status == "aborted":
                heapq.heappush(queue, _QueueItem(outcome.lower_bound, -node.depth, node.node_id, node))
                self._record(node, "aborted", outcome, lp_value, t)
                self.open_bounds_at_stop = [it.lower_bound for it in queue]
                return "TIME_LIMIT"
            if status == "infeasible":
                self._record(node, "infeasible", outcome, lp_value, t)
                continue
            if status == "pruned_bound":
                self._record(node, "pruned_bound", outcome, lp_value, t)
                continue

            # converged
            if node.node_id == 0 or (self.rim_every_nodes and self.stats.nodes_processed % self.rim_every_nodes == 0):
                self._run_rim("root" if node.node_id == 0 else f"after {self.stats.nodes_processed} nodes")
                # RIM re-activated the root; this node's LP must be re-read under
                # its own restrictions before branching on it.
                self.master.activate(node.restrictions)
                self.master.set_phase(2)
                outcome.lp = self._solve_master()

            lp = outcome.lp
            assert lp is not None and lp.feasible
            support = positive_support(lp, self.master)
            recovery = try_recover(support, self.vehicles, self.station, self.delta, self.K, self.boundary)
            if recovery.schedule is not None:
                self._offer(recovery.schedule, f"node {node.node_id} LP", strict=not recovery.averaged)
            if self._prunable(outcome.lower_bound):
                self._record(node, "integer" if recovery.schedule is not None else "pruned_bound", outcome, lp_value, t)
                continue

            branch = select_branch(support, node.restrictions, self.K, preferred_pile_slots=recovery.failing_pile_slots)
            if branch is None:
                raise BranchAndPriceError(
                    f"node {node.node_id}: every vehicle's support agrees (recovered schedule "
                    f"{'found' if recovery.schedule is not None else 'not found'}), yet the node bound "
                    f"{outcome.lower_bound:.6f} does not close against UB={self.upper_bound}."
                )
            self._record(node, "branched", outcome, lp_value, t, branch)
            for child_restr, side in ((branch.left, "L"), (branch.right, "R")):
                child = Node(
                    next_id, node.node_id, node.depth + 1, child_restr, outcome.lower_bound,
                    f"{branch.description} [{side}]",
                )
                heapq.heappush(queue, _QueueItem(child.lower_bound, -child.depth, child.node_id, child))
                next_id += 1
        return "OPTIMAL"

    # ------------------------------------------------------------------ #
    # primal heuristic: column-fixing dive
    # ------------------------------------------------------------------ #
    def _column_fixing_dive(self, restrictions: NodeRestrictions, label: str) -> None:
        """
        Repeatedly solve column generation and fix one vehicle to the
        timetable ``(pile, S, D)`` of its largest-weight column (or, once all
        timetables agree, to that column's module profile), until the LP is
        integer-recoverable (-> incumbent), infeasible, or cannot beat UB.
        Only ever offers validated incumbents; the tree is untouched.
        """
        t0 = time.perf_counter()
        K = self.K
        best_before = self.upper_bound
        restr = restrictions
        # Each level: (restrictions before fixing, ranked candidates, index tried).
        stack: list[list] = []
        backtracks = 0
        steps = 0
        end = "limit"
        while steps < 3 * len(self.vehicles) + self.dive_backtracks:
            steps += 1
            if self._out_of_time() or time.perf_counter() - t0 > self.dive_time_limit:
                end = "time"
                break
            outcome = self._solve_node(Node(-1, None, steps, restr, -math.inf, "dive"))
            self.stats.dive_node_solves += 1
            if outcome.status == "aborted":
                end = "time"
                break
            if outcome.status == "converged":
                support = positive_support(outcome.lp, self.master)
                recovery = try_recover(support, self.vehicles, self.station, self.delta, K, self.boundary)
                if recovery.schedule is not None:
                    self._offer(recovery.schedule, f"dive {label}", strict=not recovery.averaged)
                    end = "integer"
                    break
                candidates = self._dive_candidates(support)
                if not candidates:
                    end = "no-candidate"
                    break
                stack.append([restr, candidates, 0])
            else:
                # Infeasible, or provably unable to beat UB: undo the last fixing
                # and try the next candidate at the deepest level that has one.
                while stack and stack[-1][2] + 1 >= len(stack[-1][1]):
                    stack.pop()
                if not stack or backtracks >= self.dive_backtracks:
                    end = outcome.status
                    break
                backtracks += 1
                stack[-1][2] += 1
            base, candidates, idx = stack[-1]
            restr = self._fix_to(base, candidates[idx])
        self._log(
            f"dive {label}: {steps} step(s), {backtracks} backtrack(s), ended {end}, "
            f"UB {best_before:.6g} -> {self.upper_bound:.6g} ({time.perf_counter() - t0:.1f}s)"
        )

    def _dive_candidates(self, support) -> list[tuple[int, PBPlan, int]]:
        """
        Up to three ``(vehicle, column, level)`` to fix, best first: vehicles
        whose support is not integral, timetables (level 0) before module
        profiles (level 1), served columns before the null plan (fixing a
        vehicle to "never served" costs K), then larger LP weight.
        """
        K = self.K
        ranked = []
        for j, items in support.items():
            timetables = {(p.pile, p.start_slot(K), p.departure) for p, _ in items}
            profiles = {p.q_profile() for p, _ in items}
            if len(timetables) == 1 and len(profiles) == 1:
                continue
            level = 0 if len(timetables) > 1 else 1
            for plan, lam in items:
                ranked.append(((-level, 0 if plan.is_null else 1, lam), j, plan, level))
        ranked.sort(key=lambda t: t[0], reverse=True)
        return [(j, plan, level) for _, j, plan, level in ranked[:3]]

    def _fix_to(self, restrictions: NodeRestrictions, candidate: tuple[int, PBPlan, int]) -> NodeRestrictions:
        """Restrict vehicle ``j`` to ``plan``'s timetable (level 1: and module profile)."""
        j, plan, level = candidate
        K = self.K
        r = restrictions.get(j)
        if plan.is_null:
            r = r.with_s_min(K)
        else:
            S, D = plan.start_slot(K), plan.departure
            r = r.with_s_min(S).with_s_max(S).with_d_min(D).with_d_max(D)
            if r.pile_only is None:
                r = r.with_pile_only(plan.pile)
            if level == 1:
                for k in plan.occupied_slots():
                    r = r.with_q_min(k, plan.q(k)).with_q_max(k, plan.q(k))
        return restrictions.with_vehicle(j, r)

    def _record(self, node: Node, status: str, outcome: _NodeOutcome, lp_value, t0: float, branch: Branch | None = None):
        rec = NodeRecord(
            node_id=node.node_id,
            parent_id=node.parent_id,
            depth=node.depth,
            branch=node.branch,
            status=status,
            lower_bound=outcome.lower_bound,
            lp_value=lp_value,
            phase2_iterations=outcome.phase2_iterations,
            phase1_iterations=outcome.phase1_iterations,
            columns_after=len(self.master.columns),
            upper_bound_after=self.upper_bound,
            seconds=time.perf_counter() - t0,
        )
        self.records.append(rec)
        self._log(
            f"node {node.node_id} (depth {node.depth}, {node.branch}): {status}, "
            f"LB={outcome.lower_bound:.4f}, UB={self.upper_bound:.6g}, "
            f"iters={outcome.phase2_iterations}+{outcome.phase1_iterations}(PI), cols={len(self.master.columns)}"
            + (f" -> branch {branch.description}" if branch else "")
        )

    def _shutdown(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None
        for pr in self.pricers.values():
            if isinstance(pr, LabelingPricer):
                self.stats.labeling_fallbacks += pr.fallbacks
            pr.dispose()
        for env in self._envs:
            env.dispose()
        self._envs = []


def solve_branch_and_price(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    **kwargs,
):
    """Convenience wrapper: ``BranchAndPrice(...).solve()``. See ``BranchAndPrice``."""
    return BranchAndPrice(vehicles, station, delta, horizon_minutes, **kwargs).solve()
