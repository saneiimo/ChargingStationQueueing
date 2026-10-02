"""
Single-vehicle pricing MILP with whole modules.

One ``PBPricer`` per (vehicle, pile) pair, built once and re-used for the
whole solve: between calls only the objective (new duals) and the
restriction bounds (new node) change, never the constraint matrix.

What the pricer's feasible set is -- exactly
--------------------------------------------
The *served* plans of this vehicle on this pile that satisfy the node's
restrictions. The vehicle's own constraints mirror
``offline_cl_opt.model.build_cl_model`` row for row -- occupancy block
(3)-(9), power cap (13), recursion (17), taper (18), departure rule (19)
with no row at ``k = K`` (censoring), ``x <= W`` as a bound -- plus the
whole-module link ``p <= Delta*q``, ``q <= N*u``, ``q`` integer, which is
what the compact model's (14)-(15) say for one connector's occupant.

Two deliberate differences from ``offline_cl_dw.pricer``:

* ``sum_k eta_k == 1`` (served only). The null plan is priced by the caller
  in closed form, and only when the node allows it. Letting the MILP return
  "u = 0" would be unsound under branching: at a node that forbids the null
  plan its convexity dual can exceed ``K``, the MILP would return the
  (inactive) null plan as its optimum, and a genuinely improving served
  plan would never be seen.
* Every result carries Gurobi's ``ObjBound``. The caller's Lagrangian bound
  is built from bounds, never from incumbents, so it stays a valid lower
  bound even when a solve stops early on ``BestBdStop``.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import _release_slot, sojourn_minutes

from .columns import PBPlan, column_key, modules_needed
from .restrictions import EMPTY_RESTRICTION, VehicleRestriction
from .tolerances import (
    EPS_CERT,
    PRICER_FEASIBILITY_TOL,
    PRICER_INT_FEAS_TOL,
    VALIDATION_TOL,
)


class PricingNumericalError(RuntimeError):
    """A pricing solve returned something inconsistent (never silently ignored)."""


# Status labels on PricingResult.
INFEASIBLE = "infeasible"  # no served plan on this pile satisfies the node
SOLVED = "solved"  # optimum or bound available (see certified / plan)
TIME_LIMIT = "time_limit"  # stopped by the caller's time limit, not certified


@dataclass
class PricingResult:
    vehicle_id: int
    pile: int
    status: str
    bound: float  # valid lower bound on this pricer's optimum (inf if infeasible)
    objective: float | None  # incumbent objective, if any
    plan: PBPlan | None  # cleaned incumbent plan, only when it looked improving
    reduced_cost: float | None  # reduced cost of ``plan``, recomputed from its coefficients
    certified: bool  # bound proves no plan here has reduced cost < -eps_rc
    # Further improving columns from Gurobi's solution pool (cleaned,
    # distinct signatures, reduced cost < -eps_rc), best first. Excludes ``plan``.
    extra: list[tuple[PBPlan, float]] = field(default_factory=list)


def reduced_cost(
    plan: PBPlan,
    cost: float,
    pi: dict[tuple[int, int], float],
    mu: dict[tuple[int, int], float],
    sigma: float,
) -> float:
    """``cost - sum pi*alpha - sum mu*q - sigma`` straight off the plan's coefficients."""
    rc = cost - sigma
    if plan.is_null:
        return rc
    for k in plan.occupied_slots():
        rc -= pi.get((plan.pile, k), 0.0) + mu.get((plan.pile, k), 0.0) * plan.q(k)  # type: ignore[arg-type]
    return rc


class PBPricer:
    """Reusable pricing MILP for one (vehicle, pile)."""

    def __init__(
        self,
        v: VehicleData,
        pile: int,
        station: StationSpec,
        delta: float,
        K: int,
        earliest_departure: int,
        *,
        initial_energy: float = 0.0,
        forced_start: bool = False,
        weight: float = 1.0,
        env: gp.Env | None = None,
        threads: int | None = 1,
    ) -> None:
        self.vehicle = v
        self.vehicle_id = v.id
        self.pile = pile
        self.station = station
        self.delta = delta
        self.K = K
        self.h = delta / 60.0
        self.k0 = _release_slot(v.a, delta)
        if self.k0 > K - 1:
            raise ValueError(f"Vehicle {v.id}: release slot {self.k0} is outside the horizon K={K}.")
        self.earliest_departure = int(earliest_departure)
        self.x0 = float(initial_energy)
        self.forced_start = forced_start
        self.weight = float(weight)
        self.N = station.n_modules
        self.Delta = station.p_module
        self.p_bar = min(v.p_max, self.N * self.Delta)
        self.tau_d = v.tau_delta_hours(delta)
        self.restriction: VehicleRestriction = EMPTY_RESTRICTION
        self.active = True

        m = gp.Model(f"pb_pricer_v{v.id}_m{pile}", env=env) if env is not None else gp.Model(
            f"pb_pricer_v{v.id}_m{pile}"
        )
        m.Params.OutputFlag = 0
        if threads is not None:
            m.Params.Threads = threads
        m.Params.MIPGap = 0.0
        m.Params.FeasibilityTol = PRICER_FEASIBILITY_TOL
        m.Params.IntFeasTol = PRICER_INT_FEAS_TOL
        # A bounded MIP can only be optimal or infeasible; DualReductions=0
        # makes Gurobi say which instead of INF_OR_UNBD.
        m.Params.DualReductions = 0
        self.model = m

        ks = range(self.k0, K)
        self.ks = ks
        self.u = m.addVars(ks, vtype=GRB.BINARY, name="u")
        self.eta = m.addVars(ks, lb=0.0, ub=1.0, name="eta")
        self.p = m.addVars(ks, lb=0.0, ub=self.p_bar, name="p")
        self.q = m.addVars(ks, lb=0, ub=self.N, vtype=GRB.INTEGER, name="q")
        self.x = m.addVars(range(self.k0 + 1, K + 1), lb=0.0, ub=v.W, name="x")

        if forced_start:
            self.u[self.k0].lb = self.u[self.k0].ub = 1.0

        # (3)-(6) with "served" in place of "at most one start".
        for k in ks:
            u_prev = self.u[k - 1] if k > self.k0 else 0.0
            m.addConstr(self.eta[k] >= self.u[k] - u_prev, name=f"eta_lb[{k}]")
            m.addConstr(self.eta[k] <= self.u[k], name=f"eta_le_u[{k}]")
            m.addConstr(self.eta[k] <= 1 - u_prev, name=f"eta_le_1mprev[{k}]")
        m.addConstr(gp.quicksum(self.eta[k] for k in ks) == 1, name="served")
        # (8)-(9) with v_j = 1.
        self.S = gp.quicksum(k * self.eta[k] for k in ks)
        self.D = self.S + gp.quicksum(self.u[k] for k in ks)

        for k in ks:
            m.addConstr(self.p[k] <= self.p_bar * self.u[k], name=f"power_cap[{k}]")  # (13)
            m.addConstr(self.q[k] <= self.N * self.u[k], name=f"modules_need_occupancy[{k}]")
            m.addConstr(self.p[k] <= self.Delta * self.q[k], name=f"power_from_modules[{k}]")

        for k in ks:
            x_prev = self.x[k] if k > self.k0 else self.x0
            m.addConstr(self.x[k + 1] == x_prev + self.h * self.p[k], name=f"energy_recursion[{k}]")  # (17)
            m.addConstr(self.tau_d * self.p[k] + x_prev <= v.R, name=f"taper_cap[{k}]")  # (18)
            if k > self.k0:
                m.addConstr(
                    self.x[k] >= v.W * (self.u[k - 1] - self.u[k]), name=f"departure_rule[{k}]"
                )  # (19); no row at k = K (censoring)

        # Departure window. The lower row also carries D >= E_j, valid for
        # every served plan (E_j is the fastest possible solo departure, or K).
        self.d_lb = m.addConstr(self.D >= self.earliest_departure, name="d_lb")
        self.d_ub = m.addConstr(self.D <= K, name="d_ub")
        # The compact model's (24) with v_j = 1: a plan that departs inside
        # the horizon occupies at least n_min = E_j - k_j slots (the taper
        # depends on delivered energy, not on when charging starts). Only
        # for vehicles that start from s_i, exactly as build_cl_model does.
        if not forced_start and self.earliest_departure > self.k0:
            n_min = self.earliest_departure - self.k0
            m.addConstr(
                gp.quicksum(self.u[k] for k in ks) >= n_min * (1 - self.u[K - 1]),
                name="min_occupancy_if_departed",
            )
        m.update()

    def dispose(self) -> None:
        self.model.dispose()

    # ------------------------------------------------------------------ #
    def _q_lo(self, k: int) -> int:
        return int(self.restriction.q_min.get(k, 0))

    def _q_hi(self, k: int) -> int:
        return int(min(self.N, self.restriction.q_max.get(k, self.N)))

    def plan_cost(self, departure: int, weight: float) -> float:
        """A plan's objective cost: ``weight * (delta*D - a)``, its sojourn in minutes (0 for weight 0)."""
        return weight * sojourn_minutes(departure, self.vehicle.a, self.delta) if weight else 0.0

    def apply(self, restriction: VehicleRestriction) -> None:
        """Impose a node's restrictions for this vehicle (bounds/RHS only)."""
        self.restriction = restriction
        self.active = restriction.allows_pile(self.pile)
        # A positive module lower bound outside this vehicle's slot range
        # cannot be met by any plan of it.
        for k, lo in restriction.q_min.items():
            if lo > 0 and not (self.k0 <= k < self.K):
                self.active = False
        d_lo = max(self.earliest_departure, restriction.d_min if restriction.d_min is not None else 0)
        d_hi = restriction.d_max if restriction.d_max is not None else self.K
        self.d_lb.RHS = d_lo
        self.d_ub.RHS = d_hi
        s_lo = restriction.s_min if restriction.s_min is not None else self.k0
        s_hi = restriction.s_max if restriction.s_max is not None else self.K - 1
        for k in self.ks:
            self.eta[k].ub = 1.0 if s_lo <= k <= s_hi else 0.0
            lo, hi = self._q_lo(k), self._q_hi(k)
            if lo > hi:
                self.active = False
                lo = hi = 0  # keep Gurobi's bounds consistent; the pricer is skipped anyway
            self.q[k].lb = lo
            self.q[k].ub = hi
        if d_lo > d_hi or s_lo > s_hi:
            self.active = False

    def price(
        self,
        pi: dict[tuple[int, int], float],
        mu: dict[tuple[int, int], float],
        sigma: float,
        *,
        include_departure: bool,
        eps_rc: float,
        time_limit: float | None = None,
        gap_abs: float = 0.0,
        max_columns: int = 1,
    ) -> PricingResult:
        """
        Minimise ``w*(delta*D - a) - sum pi*u - sum mu*q`` -- the vehicle's
        sojourn in minutes minus rent (``include_departure=False`` drops the
        sojourn term, for Phase I) -- over this pricer's feasible set.
        ``sigma`` is the vehicle's convexity dual; it only sets the
        ``BestBdStop`` threshold and the "improving" test.

        ``gap_abs`` > 0 lets the MILP stop once its incumbent is within
        ``gap_abs`` of its bound. If that leaves neither an improving column
        nor a certificate, the same MILP is re-solved to optimality, so the
        result always settles the pricing question. ``max_columns`` > 1 also
        returns further improving columns found along the way (``extra``).
        """
        if not self.active:
            return PricingResult(self.vehicle_id, self.pile, INFEASIBLE, math.inf, None, None, None, True)

        m = self.model
        pile = self.pile
        cost_weight = self.weight if include_departure else 0.0
        obj = gp.LinExpr()
        if cost_weight != 0.0:
            obj += cost_weight * (self.delta * self.D - self.vehicle.a)
        for k in self.ks:
            pk = pi.get((pile, k), 0.0)
            mk = mu.get((pile, k), 0.0)
            if pk != 0.0:
                obj += -pk * self.u[k]
            if mk != 0.0:
                obj += -mk * self.q[k]
        m.setObjective(obj, GRB.MINIMIZE)
        # Stop as soon as the bound proves nothing here has reduced cost
        # below -eps_rc; the bound is still read and used in that case.
        m.Params.BestBdStop = sigma - eps_rc
        m.Params.MIPGapAbs = gap_abs if gap_abs > 0.0 else 1e-10
        start = time.perf_counter()
        m.Params.TimeLimit = time_limit if time_limit is not None else GRB.INFINITY
        m.optimize()
        status = self._check_status()
        if (
            gap_abs > 0.0
            and status == GRB.OPTIMAL
            and not (m.SolCount > 0 and m.ObjVal - sigma < -eps_rc)
            and not (m.ObjBound - sigma >= -eps_rc - EPS_CERT)
        ):
            # Gap reached without deciding the question: finish exactly.
            m.Params.MIPGapAbs = 1e-10
            if time_limit is not None:
                m.Params.TimeLimit = max(0.0, time_limit - (time.perf_counter() - start))
            m.optimize()
            status = self._check_status()

        if status == GRB.INFEASIBLE:
            return PricingResult(self.vehicle_id, pile, INFEASIBLE, math.inf, None, None, None, True)

        bound = float(m.ObjBound)
        objective = float(m.ObjVal) if m.SolCount > 0 else None
        plan = rc = None
        extra: list[tuple[PBPlan, float]] = []
        if objective is not None and objective - sigma < -eps_rc:
            plan = self._extract_clean_plan(0)
            rc = reduced_cost(plan, self.plan_cost(plan.departure, cost_weight), pi, mu, sigma)
            # Clean-up only rounds u/q and clips p (and may lower q to what the
            # power needs, which can only lower the reduced cost since mu <= 0).
            if rc > (objective - sigma) + 1e-6:
                raise PricingNumericalError(
                    f"Pricer v{self.vehicle_id} pile {pile}: cleaned plan's reduced cost {rc:.9g} "
                    f"is worse than the solver's {objective - sigma:.9g}."
                )
            seen = {column_key(plan)}
            for i in range(1, m.SolCount):
                if len(extra) >= max_columns - 1:
                    break
                m.Params.SolutionNumber = i
                if m.PoolObjVal - sigma >= -eps_rc:
                    continue
                try:
                    other = self._extract_clean_plan(i)
                except PricingNumericalError:
                    continue  # a secondary pool solution is optional
                key = column_key(other)
                other_rc = reduced_cost(other, self.plan_cost(other.departure, cost_weight), pi, mu, sigma)
                if key not in seen and other_rc < -eps_rc:
                    seen.add(key)
                    extra.append((other, other_rc))
        certified = bound - sigma >= -eps_rc - EPS_CERT
        if status == GRB.TIME_LIMIT and not certified and (rc is None or rc >= -eps_rc):
            return PricingResult(self.vehicle_id, pile, TIME_LIMIT, bound, objective, plan, rc, False)
        return PricingResult(self.vehicle_id, pile, SOLVED, bound, objective, plan, rc, certified, extra)

    def _check_status(self) -> int:
        status = self.model.Status
        if status not in (GRB.OPTIMAL, GRB.USER_OBJ_LIMIT, GRB.TIME_LIMIT, GRB.INFEASIBLE):
            raise PricingNumericalError(
                f"Pricer v{self.vehicle_id} pile {self.pile}: unexpected Gurobi status {status} "
                f"under restriction [{self.restriction.describe()}]."
            )
        return status

    # ------------------------------------------------------------------ #
    def _extract_clean_plan(self, solution: int = 0) -> PBPlan:
        """
        Read pool solution ``solution`` (0 = incumbent) and make it exactly
        feasible: round ``u``/``q``, clip ``p`` to every cap (modules,
        ``P_bar``, taper, ``W - x``) along a recomputed energy trajectory,
        and lower ``q`` to what the clipped power needs (never below the
        node's lower bound). Each change is of the order of the solver's
        1e-9 tolerances.
        """
        if solution == 0:
            val = lambda var: var.X  # noqa: E731
        else:
            self.model.Params.SolutionNumber = solution
            val = lambda var: var.Xn  # noqa: E731
        occupied = [k for k in self.ks if val(self.u[k]) > 0.5]
        if not occupied:
            raise PricingNumericalError(f"Pricer v{self.vehicle_id}: 'served' solution has no occupied slot.")
        S, D = occupied[0], occupied[-1] + 1
        if occupied != list(range(S, D)):
            raise PricingNumericalError(f"Pricer v{self.vehicle_id}: occupancy {occupied} is not contiguous.")

        v, Delta, h = self.vehicle, self.Delta, self.h
        x = self.x0
        power: dict[int, float] = {}
        modules: dict[int, int] = {}
        for k in range(S, D):
            lo, hi = self._q_lo(k), self._q_hi(k)
            q_raw = min(max(int(round(val(self.q[k]))), lo), hi)
            p_k = min(
                max(0.0, val(self.p[k])),
                Delta * q_raw,
                self.p_bar,
                (v.R - x) / self.tau_d,
                (v.W - x) / h,
            )
            p_k = max(0.0, p_k)
            q_k = max(modules_needed(p_k, Delta), lo)
            if q_k > q_raw:
                raise PricingNumericalError(
                    f"Pricer v{self.vehicle_id} slot {k}: power {p_k} needs {q_k} modules > solver's {q_raw}."
                )
            p_k = min(p_k, Delta * q_k)
            if p_k > 0.0:
                power[k] = p_k
            modules[k] = q_k
            x += h * p_k
        if D < self.K and x < v.W - VALIDATION_TOL:
            raise PricingNumericalError(
                f"Pricer v{self.vehicle_id}: plan departs at {D} with {x:.9g} kWh < W={v.W:.9g} after clean-up."
            )
        return PBPlan(
            vehicle_id=self.vehicle_id,
            pile=self.pile,
            start=S,
            departure=D,
            power=power,
            modules=modules,
        )
