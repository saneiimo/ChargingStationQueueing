"""
Price-directed state-space search: the driver (the paper's Algorithm 1).

1. **Incumbents.** ``offline_cl_PB``'s greedy list schedule and the caller's
   ``initial_schedule`` (e.g. FIFO from the simulation); each passes
   ``offline_cl_PB.validate_schedule`` before it can set the upper bound.
2. **Prices.** Column generation on the aggregate or configuration master
   (``colgen.slot_prices``); keeps the prices with the best Lagrangian bound.
3. **Tables.** Prices rounded to the price grid, then every vehicle's
   cost-to-go table (``cost_to_go.build_tables``). The bound at the empty
   schedule (``root_lower_bound``) is the master's bound up to rounding.
4. **Beam passes** (widths ``beam_widths``): passes that keep only the best
   states per slot. They find good schedules fast, and still certify a lower
   bound (a beam cut counts as a memory cut).
5. **Deepening passes.** Exact passes with rising thresholds
   ``T_1 = LB + ceil((target - LB)/8)``, ``T_{i+1} = T_i + ceil(1.5 * step)``
   (44); a complete pass that finds nothing raises the lower bound to its
   threshold. The last threshold is ``target = UB - gap_steps`` (``UB`` without
   a tolerance): a complete pass there proves the incumbent optimal (or within
   ``gap_abs``). Deepening stops after the first pass that finds a schedule
   (complete: it is optimal) or that is not complete (memory cut, expansion
   budget, time limit: the bracket is reported).

Every schedule a pass finds lives on the grid relaxation; it is replayed with
the compact model's own dynamics (``exact_plan``) and validated before it can
lower ``UB``. Every lower bound comes from Theorem 1 (``PassResult.certified``)
and is valid whatever the replay does. Bounds are tracked in slots
(``z = sum_j w_j D_j``, an integer) and reported in minutes.
"""

from __future__ import annotations

import math
import time

from offline_cl_opt.boundary import BoundaryVehicle, Cohort
from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_PB import (
    PBPlan,
    ScheduleValidationError,
    compact_model_check,
    greedy_list_schedule,
    validate_schedule,
)

from .colgen import slot_prices
from .cost_to_go import INF, build_tables, table_value
from .instance import build_search_instance
from .search import BUDGET, COMPLETE, MEMORY_CUT, TIME_LIMIT, PassResult, run_search_pass
from .solution import PDSSolution, pass_row, per_vehicle_table

STOP_STATUS = {MEMORY_CUT: "MEMORY_LIMIT", BUDGET: "EXPANSION_LIMIT", TIME_LIMIT: "TIME_LIMIT"}
# A pass found a schedule on the grid relaxation that the exact dynamics could not replay
# (a vehicle reaching W within ~1e-5 kWh); the paper never saw it, nor should you.
REPLAY_FAILED = "REPLAY_FAILED"


def exact_plan(
    v: VehicleData, bv: BoundaryVehicle | None, station: StationSpec, delta: float,
    pile: int, start: int, modules: list[int],
) -> PBPlan:
    """
    Replay a module profile with the compact model's dynamics (5): each slot
    draws ``min(Delta*q, P_bar, (R - x)/tau_d, (W - x)/h)``, as
    ``offline_cl_PB``'s greedy plans do. The plan departs after its last
    module slot; ``validate_schedule`` checks that it really completes there.
    """
    h = delta / 60.0
    Delta = station.p_module
    p_bar = min(v.p_max, station.n_modules * Delta)
    tau_d = v.tau_delta_hours(delta)
    x = bv.initial_energy_kwh if bv is not None else 0.0
    power, mods = {}, {}
    for k, q in zip(range(start, start + len(modules)), modules):
        p = max(0.0, min(Delta * q, p_bar, (v.R - x) / tau_d, (v.W - x) / h))
        if p > 0.0:
            power[k] = p
        mods[k] = int(q)
        x += h * p
    return PBPlan(vehicle_id=v.id, pile=pile, start=start, departure=start + len(modules), power=power, modules=mods)


class PriceDirectedSearch:
    """
    Parameters
    ----------
    vehicles, station, delta, horizon_minutes, boundary_vehicles, cohorts,
    objective_cohorts :
        Exactly as for ``offline_cl_opt.model.build_cl_model`` /
        ``offline_cl_PB.BranchAndPrice``: the model solved is that compact
        model (whole modules, every vehicle departing by the horizon).
    initial_schedule :
        Optional ``{vehicle_id: PBPlan}`` (e.g.
        ``offline_cl_PB.schedule_from_simulation``); validated, and an invalid
        one raises.
    master :
        ``"aggregate"``, ``"configuration"``, or ``"auto"`` (configuration
        when the station has two or more piles, as in the paper). The
        configuration master's bound is at least as strong; with one pile the
        difference is usually small and the aggregate master is faster.
    time_limit :
        Seconds for the whole solve; the bracket is valid at any stop.
    cg_time_limit, cg_rel_tol :
        Column generation stops at this many seconds or when
        ``z_RMP - L <= cg_rel_tol * z_RMP`` (prices stay valid either way;
        weaker prices only make the search longer).
    beam_widths :
        Widths of the heuristic beam passes run before the exact ones.
    deepen :
        Rising thresholds before the final pass (step 5). ``False``: one exact
        pass at the target.
    layer_cap :
        States kept per slot before a memory cut. Memory is about
        ``2 * layer_cap * (60 + 11*M*C)`` bytes.
    max_expand :
        Expanded states allowed per exact pass (0: no limit).
    gap_abs :
        Stop once the incumbent is proven within this many minutes of total
        sojourn of the optimum (``x * n_optimized`` for ``x`` minutes of mean
        sojourn). Only whole steps of ``delta`` count.
    compact_check :
        Evaluate every row of the compact model at the returned schedule.
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
        master: str = "auto",
        time_limit: float | None = None,
        cg_time_limit: float | None = None,
        cg_rel_tol: float = 1e-5,
        beam_widths: tuple[int, ...] = (200, 2000),
        deepen: bool = True,
        layer_cap: int = 2_000_000,
        max_expand: int = 0,
        gap_abs: float = 0.0,
        compact_check: bool = True,
        progress: bool = False,
    ) -> None:
        self.inst = build_search_instance(
            vehicles, station, delta, horizon_minutes, boundary_vehicles=boundary_vehicles,
            cohorts=cohorts, objective_cohorts=objective_cohorts,
        )
        if master == "auto":
            master = "configuration" if station.n_piles >= 2 else "aggregate"
        if gap_abs < 0:
            raise ValueError("gap_abs must be >= 0")
        self.master = master
        self.time_limit, self.cg_time_limit, self.cg_rel_tol = time_limit, cg_time_limit, cg_rel_tol
        self.beam_widths, self.deepen = tuple(beam_widths), deepen
        self.layer_cap, self.max_expand = int(layer_cap), int(max_expand)
        self.gap_steps = math.floor(gap_abs / delta + 1e-9)  # one slot of z is delta minutes
        self.run_compact_check, self.progress = compact_check, progress
        self.initial_schedule = initial_schedule

        inst = self.inst
        # Any schedule has z <= sum_j w_j K + FIXED constant; passes use this
        # as their threshold until a schedule is known.
        self.cap = int(inst.weight.sum()) * inst.K + inst.fixed_departure_total + 1
        self.upper = math.inf  # incumbent value, slots
        self.lower = -math.inf  # certified lower bound, slots (an integer once set)
        self.incumbent: dict[int, PBPlan] | None = None
        self.connectors: dict[int, int] = {}
        self.source = ""
        self.passes: list[dict] = []
        self.bound_history: list[dict] = []
        self.last_stop = COMPLETE
        self.replay_failed = False
        self.timings = {"prices": 0.0, "tables": 0.0, "beam": 0.0, "search": 0.0}
        self._t0 = time.perf_counter()
        self._deadline = self._t0 + time_limit if time_limit is not None else None

    # ------------------------------------------------------------------ #
    def _log(self, msg: str) -> None:
        if self.progress:
            print(f"[PDS {time.perf_counter() - self._t0:7.1f}s] {msg}", flush=True)

    def _out_of_time(self) -> bool:
        return self._deadline is not None and time.perf_counter() >= self._deadline

    def _note(self, event: str) -> None:
        inst = self.inst
        lb = min(self.lower, self.upper)
        self.bound_history.append({
            "time_s": time.perf_counter() - self._t0,
            "lower_bound": inst.to_minutes(lb) if math.isfinite(lb) else -math.inf,
            "upper_bound": inst.to_minutes(self.upper) if math.isfinite(self.upper) else math.inf,
            "event": event,
        })

    def _raise_lower(self, value: float, event: str) -> None:
        if value > self.lower:
            self.lower = value
            self._note(event)

    def _offer(self, schedule: dict[int, PBPlan] | None, source: str, *, strict: bool) -> None:
        """Validate ``schedule``; adopt it if it beats the incumbent."""
        if schedule is None:
            return
        inst = self.inst
        try:
            check = validate_schedule(
                schedule, inst.vehicles, inst.station, inst.delta, inst.K, inst.boundary_vehicles, inst.weights
            )
        except ScheduleValidationError as exc:
            if strict:
                raise ScheduleValidationError(f"{source} is not feasible: {exc}") from exc
            self._log(f"{source}: replay with the exact dynamics failed ({exc}); not used")
            self.replay_failed = True
            return
        z = round((check.objective + inst.arrival_total) / inst.delta)
        if z < self.upper:
            self.upper, self.incumbent, self.connectors, self.source = z, dict(schedule), check.connectors, source
            self._log(f"incumbent {check.objective:.6g} min from {source}")
            self._note(f"incumbent: {source}")

    def _schedule_from_pass(self, res: PassResult) -> dict[int, PBPlan] | None:
        """The pass's schedule (search indices) as plans of every vehicle, replayed exactly."""
        if res.schedule is None:
            return None
        inst = self.inst
        sched = dict(inst.fixed_plans)
        for i, (pile, start, modules) in res.schedule.items():
            v = inst.vehicle(i)
            sched[v.id] = exact_plan(v, inst.boundary_vehicles.get(v.id), inst.station, inst.delta, pile, start, modules)
        return sched

    def _pass(self, threshold: int, *, beam_width: int, keep_history: bool, label: str) -> PassResult:
        res = run_search_pass(
            self.inst, self.tables, threshold, beam_width=beam_width, layer_cap=self.layer_cap,
            keep_history=keep_history, max_expand=0 if beam_width else self.max_expand, deadline=self._deadline,
        )
        self.passes.append(pass_row(res, self.inst))
        self._log(
            f"{label} T={self.inst.to_minutes(threshold):.6g}: "
            f"{'found ' + format(self.inst.to_minutes(res.value), '.6g') if res.found else 'nothing below T'}, "
            f"{res.status_name}, certified LB {self.inst.to_minutes(res.certified):.6g}, "
            f"{res.expanded} expanded, peak {res.peak} states/slot ({res.runtime:.1f}s)"
        )
        if res.found:
            if res.schedule is None:
                # Value-only pass: run again with history to rebuild the schedule.
                again = run_search_pass(
                    self.inst, self.tables, int(res.value) + 1, layer_cap=self.layer_cap, keep_history=True,
                    max_expand=self.max_expand, deadline=self._deadline,
                )
                self._offer(self._schedule_from_pass(again), f"{label} (rebuilt)", strict=False)
                self._raise_lower(again.certified, f"{label} (rebuilt)")
            else:
                self._offer(self._schedule_from_pass(res), label, strict=False)
        self._raise_lower(res.certified, label)
        if res.status != COMPLETE:
            self.last_stop = res.status
        return res

    def _target(self) -> int:
        """The last threshold: prove no schedule is below ``UB - gap_steps``."""
        return int(self.upper) - self.gap_steps if math.isfinite(self.upper) else self.cap

    def _done(self) -> bool:
        return self.lower >= self._target()

    # ------------------------------------------------------------------ #
    def solve(self) -> PDSSolution:
        inst = self.inst
        # 1. incumbents
        self._offer(greedy_list_schedule(inst.vehicles, inst.station, inst.delta, inst.K, inst.boundary_vehicles),
                    "greedy list schedule", strict=True)
        if self.initial_schedule is not None:
            self._offer(dict(self.initial_schedule), "initial schedule", strict=True)

        # 2. prices
        t = time.perf_counter()
        cg_limit = self.cg_time_limit
        if self._deadline is not None:
            left = self._deadline - t
            cg_limit = left if cg_limit is None else min(cg_limit, left)
        seeds = [self.incumbent] if self.incumbent is not None else []
        self.prices = slot_prices(inst, seeds, kind=self.master, time_limit=cg_limit, rel_tol=self.cg_rel_tol,
                                  progress=self.progress)
        self.timings["prices"] = time.perf_counter() - t
        pr = self.prices
        self._log(f"{self.master} master: Lagrangian bound {inst.to_minutes(pr.lagrangian_bound):.6g} min after "
                  f"{pr.iterations} rounds ({'converged' if pr.converged else 'stopped'}: z_RMP - L = "
                  f"{inst.delta * (pr.master_value - pr.lagrangian_bound):.3g} min)")

        # 3. tables and the bound at the empty schedule
        t = time.perf_counter()
        self.tables = build_tables(inst, self.prices.mu, self.prices.configs)
        self.timings["tables"] = time.perf_counter() - t
        tb = self.tables
        root = inst.fixed_departure_total - tb.tail[0]
        for i in range(inst.J):
            if inst.start_pile[i] >= 0:
                root += table_value(tb.off, tb.cnt, tb.bp, tb.val, i, 0, inst.start_energy[i])
            else:
                root += tb.U[i, 0]
        if root >= INF / 2:
            raise RuntimeError(
                f"No schedule serves and completes every vehicle within the horizon (K={inst.K} slots of "
                f"{inst.delta:g} min): some vehicle cannot finish in time even alone. Lengthen the horizon."
            )
        self.root_bound = math.ceil(root)
        self._raise_lower(self.root_bound, "root bound")
        self._log(f"tables built ({tb.build_s:.1f}s, {len(tb.bp)} breakpoints, {tb.repairs} repairs); "
                  f"root bound {inst.to_minutes(self.root_bound):.6g} min")

        # 4. beam passes
        t = time.perf_counter()
        for w in self.beam_widths:
            if self._done() or self._out_of_time():
                break
            threshold = int(self.upper) if math.isfinite(self.upper) else self.cap
            self._pass(threshold, beam_width=w, keep_history=True, label=f"beam {w}")
        self.timings["beam"] = time.perf_counter() - t

        # 5. deepening passes
        t = time.perf_counter()
        self._deepen()
        self.timings["search"] = time.perf_counter() - t
        return self._solution()

    def _deepen(self) -> None:
        step = max(1, math.ceil((self._target() - self.lower) / 8)) if self.deepen else 0
        T = self.lower + step if self.deepen else self._target()
        tried: list[tuple[int, int]] = []  # (threshold, expanded) of passes that found nothing
        while not self._done():
            if self._out_of_time():
                self.last_stop = TIME_LIMIT
                return
            T = int(min(T, self._target()))
            res = self._pass(T, beam_width=0, keep_history=False, label="exact pass")
            if res.found or res.status != COMPLETE or T >= self._target():
                return
            tried.append((T, res.expanded))
            step = math.ceil(1.5 * step)
            if self.max_expand and len(tried) >= 2 and tried[-2][1] > 0 and tried[-1][1] > tried[-2][1]:
                # Effort grows about geometrically with T: aim the next pass at half the budget.
                rate = (tried[-1][1] / tried[-2][1]) ** (1.0 / max(1, tried[-1][0] - tried[-2][0]))
                room = self.max_expand / 2 / tried[-1][1]
                step = max(1, min(step, int(math.log(room) / math.log(rate)))) if room > 1 else max(1, step // 3)
            T = int(self.lower) + step

    # ------------------------------------------------------------------ #
    def _solution(self) -> PDSSolution:
        inst = self.inst
        if self.incumbent is None:
            if self.lower >= self.cap:
                raise RuntimeError(
                    f"The search proved that no schedule serves and completes every vehicle within the "
                    f"horizon (K={inst.K} slots of {inst.delta:g} min). Lengthen the horizon."
                )
            raise RuntimeError("The search stopped before finding any schedule; pass initial_schedule or raise the limits.")
        lower = min(self.lower, self.upper)
        if lower >= self.upper:
            status = "OPTIMAL"
        elif lower >= self.upper - self.gap_steps:
            status = "WITHIN_TOLERANCE"
        elif self.last_stop != COMPLETE:
            status = STOP_STATUS[self.last_stop]
        else:
            status = REPLAY_FAILED if self.replay_failed else "TIME_LIMIT"
        self.lower = lower
        self._note("end")

        objective, lower_bound = inst.to_minutes(self.upper), inst.to_minutes(lower)
        per_vehicle, by_cohort, routing = per_vehicle_table(inst, self.incumbent, self.connectors)
        check = None
        if self.run_compact_check:
            check = compact_model_check(
                self.incumbent, self.connectors, inst.vehicles, inst.station, inst.delta, inst.horizon_minutes,
                boundary_vehicles=inst.boundary_vehicles, cohorts=inst.cohorts,
                objective_cohorts=inst.objective_cohorts,
            )
        n = inst.n_optimized
        return PDSSolution(
            status=status,
            objective=objective,
            lower_bound=lower_bound,
            gap=objective - lower_bound,
            master_bound=inst.to_minutes(self.prices.lagrangian_bound),
            root_lower_bound=inst.to_minutes(self.root_bound),
            total_sojourn=objective,
            mean_sojourn=objective / n,
            total_sojourn_LB=lower_bound,
            mean_sojourn_LB=lower_bound / n,
            n_vehicles=len(inst.vehicles),
            n_optimized=n,
            delta=inst.delta,
            K=inst.K,
            runtime=time.perf_counter() - self._t0,
            incumbent_source=self.source,
            schedule=dict(self.incumbent),
            connectors=dict(self.connectors),
            module_routing=routing,
            by_cohort=by_cohort,
            per_vehicle=per_vehicle,
            compact_check=check,
            prices=self.prices,
            table_repairs=self.tables.repairs,
            table_breakpoints=int(len(self.tables.bp)),
            timings=dict(self.timings),
            passes=list(self.passes),
            bound_history=list(self.bound_history),
            inputs=dict(
                vehicles=inst.vehicles, station=inst.station, delta=inst.delta, horizon_minutes=inst.horizon_minutes,
                boundary_vehicles=inst.boundary_vehicles, cohorts=inst.cohorts,
                objective_cohorts=inst.objective_cohorts,
            ),
        )


def solve_price_directed_search(
    vehicles: list[VehicleData],
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    **kwargs,
) -> PDSSolution:
    """``PriceDirectedSearch(...).solve()``; see that class for the keyword arguments."""
    return PriceDirectedSearch(vehicles, station, delta, horizon_minutes, **kwargs).solve()

