"""
The restricted master LP with whole-module rows, Phase-I artificials, and
node activation.

Rows (one Gurobi model for the whole solve):

    connector[m,k]:  sum_omega alpha_{omega,m,k} lambda_omega           <= C
    modules[m,k]:    sum_omega q_{omega,k} [m_omega = m] lambda_omega   <= N
    convexity[j]:    sum_{omega in Omega_j} lambda_omega + a_j          == 1

``a_j`` are Phase-I artificials: objective 1 and unbounded in Phase I, fixed
to 0 in Phase II. A column is never deleted; a node *activates* the columns
its restrictions allow (no upper bound on ``lambda``; convexity caps it at
1) and deactivates the rest (upper bound 0), so a column priced at one node
is available to every later node it satisfies.
"""

from __future__ import annotations

from dataclasses import dataclass

import gurobipy as gp
from gurobipy import GRB

from offline_cl_opt.instance import StationSpec
from offline_cl_opt.model import sojourn_minutes

from .columns import PBPlan, column_key
from .restrictions import NodeRestrictions
from .tolerances import (
    DUAL_SIGN_TOL,
    EPS_LAMBDA,
    MASTER_FEASIBILITY_TOL,
    MASTER_OPTIMALITY_TOL,
)


class MasterNumericalError(RuntimeError):
    pass


# Upper bound of an active column. Deliberately *not* 1: convexity already
# implies lambda <= 1, and an explicit bound would let the LP optimum hold a
# column at that bound with a negative reduced cost (the bound's own dual
# absorbing it). Column generation relies on every active column having
# reduced cost >= 0 at the LP optimum, which only holds without the bound.
ACTIVE_UB = GRB.INFINITY


@dataclass
class MasterColumn:
    index: int
    plan: PBPlan
    var: gp.Var
    key: tuple


@dataclass
class MasterLP:
    feasible: bool
    objective: float
    pi: dict[tuple[int, int], float]  # connector-row duals, clamped to <= 0
    mu: dict[tuple[int, int], float]  # module-row duals, clamped to <= 0
    sigma: dict[int, float]  # convexity duals (free)
    lam: dict[int, float]  # column index -> value (active columns with value > EPS_LAMBDA)
    artificial_total: float


class PBMaster:
    def __init__(
        self,
        vehicle_ids: list[int],
        weights: dict[int, float],
        station: StationSpec,
        K: int,
        k_lo: int,
        *,
        delta: float,
        arrivals: dict[int, float],
        env: gp.Env | None = None,
    ) -> None:
        self.station = station
        self.K = K
        self.delta = float(delta)
        self.arrivals = {j: float(arrivals[j]) for j in vehicle_ids}
        self.k_lo = k_lo
        self.vehicle_ids = list(vehicle_ids)
        self.weights = dict(weights)
        m = gp.Model("pb_master", env=env) if env is not None else gp.Model("pb_master")
        m.Params.OutputFlag = 0
        m.Params.OptimalityTol = MASTER_OPTIMALITY_TOL
        m.Params.FeasibilityTol = MASTER_FEASIBILITY_TOL
        # The LP is bounded (every column's sojourn is >= 0); make Gurobi report a clean
        # INFEASIBLE rather than INF_OR_UNBD.
        m.Params.DualReductions = 0
        self.model = m

        C, N = station.n_connectors, station.n_modules
        self.connector_rows: dict[tuple[int, int], gp.Constr] = {}
        self.module_rows: dict[tuple[int, int], gp.Constr] = {}
        for mm in range(station.n_piles):
            for k in range(k_lo, K):
                self.connector_rows[mm, k] = m.addConstr(gp.LinExpr() <= C, name=f"connector[{mm},{k}]")
                self.module_rows[mm, k] = m.addConstr(gp.LinExpr() <= N, name=f"modules[{mm},{k}]")
        self.convexity: dict[int, gp.Constr] = {}
        self.artificial: dict[int, gp.Var] = {}
        for j in self.vehicle_ids:
            self.convexity[j] = m.addConstr(gp.LinExpr() == 1, name=f"convexity[{j}]")
        m.update()
        for j in self.vehicle_ids:
            col = gp.Column()
            col.addTerms(1.0, self.convexity[j])
            self.artificial[j] = m.addVar(lb=0.0, ub=0.0, obj=0.0, column=col, name=f"artificial[{j}]")
        m.update()

        self.columns: list[MasterColumn] = []
        self.by_key: dict[tuple, int] = {}
        self.phase = 2
        self.last_rim_status: tuple[int, int, float] = (0, 0, 0.0)  # (Gurobi status, SolCount, runtime)

    # ------------------------------------------------------------------ #
    def column_cost(self, plan: PBPlan) -> float:
        """``w_j * (delta*D - a_j)``: the vehicle's sojourn in minutes (0 outside the objective)."""
        j = plan.vehicle_id
        return self.weights[j] * sojourn_minutes(plan.departure, self.arrivals[j], self.delta)

    def add_column(self, plan: PBPlan, *, active: bool) -> tuple[int, bool]:
        """Add ``plan`` (dedup by signature). Returns ``(index, is_new)``."""
        key = column_key(plan)
        if key in self.by_key:
            return self.by_key[key], False
        col = gp.Column()
        col.addTerms(1.0, self.convexity[plan.vehicle_id])
        if not plan.is_null:
            for k in plan.occupied_slots():
                if (plan.pile, k) not in self.connector_rows:
                    raise ValueError(f"Column for vehicle {plan.vehicle_id} occupies slot {k} outside [k_lo, K).")
                col.addTerms(1.0, self.connector_rows[plan.pile, k])
                q = plan.q(k)
                if q:
                    col.addTerms(float(q), self.module_rows[plan.pile, k])
        obj = self.column_cost(plan) if self.phase == 2 else 0.0
        var = self.model.addVar(lb=0.0, ub=ACTIVE_UB if active else 0.0, obj=obj, column=col, name=f"lambda[{plan.vehicle_id}]")
        index = len(self.columns)
        self.columns.append(MasterColumn(index, plan, var, key))
        self.by_key[key] = index
        return index, True

    def activate(self, restrictions: NodeRestrictions) -> None:
        """No upper bound on every column the node allows, 0 on the rest."""
        self.model.update()
        vars_ = [c.var for c in self.columns]
        ubs = [ACTIVE_UB if restrictions.column_satisfies(c.plan, self.K) else 0.0 for c in self.columns]
        if vars_:
            self.model.setAttr("UB", vars_, ubs)

    def is_active(self, index: int) -> bool:
        self.model.update()
        return self.columns[index].var.UB > 0.5

    def set_active(self, index: int, active: bool) -> None:
        self.columns[index].var.UB = ACTIVE_UB if active else 0.0

    def set_phase(self, phase: int) -> None:
        """Phase 1: minimise the artificial total. Phase 2: minimise total sojourn sum_j w_j (delta*D_j - a_j)."""
        if phase not in (1, 2):
            raise ValueError(phase)
        self.model.update()
        vars_ = [c.var for c in self.columns]
        if vars_:
            costs = [0.0 if phase == 1 else self.column_cost(c.plan) for c in self.columns]
            self.model.setAttr("Obj", vars_, costs)
        arts = list(self.artificial.values())
        self.model.setAttr("Obj", arts, [1.0 if phase == 1 else 0.0] * len(arts))
        self.model.setAttr("UB", arts, [GRB.INFINITY if phase == 1 else 0.0] * len(arts))
        self.phase = phase

    # ------------------------------------------------------------------ #
    def solve(self) -> MasterLP:
        m = self.model
        m.optimize()
        if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
            if self.phase == 1:
                raise MasterNumericalError("Phase-I master reported infeasible; it always has a solution.")
            return MasterLP(False, float("inf"), {}, {}, {}, {}, float("nan"))
        if m.Status != GRB.OPTIMAL:
            raise MasterNumericalError(f"Master LP ended with status {m.Status}.")

        pi_raw = {key: c.Pi for key, c in self.connector_rows.items()}
        mu_raw = {key: c.Pi for key, c in self.module_rows.items()}
        bad = [(key, v) for key, v in list(pi_raw.items()) + list(mu_raw.items()) if v > DUAL_SIGN_TOL]
        if bad:
            raise MasterNumericalError(
                f"Capacity-row duals must be <= 0 in a minimisation with <= rows; got {bad[:5]}."
            )
        pi = {key: min(v, 0.0) for key, v in pi_raw.items()}
        mu = {key: min(v, 0.0) for key, v in mu_raw.items()}
        sigma = {j: c.Pi for j, c in self.convexity.items()}
        values = m.getAttr("X", [c.var for c in self.columns]) if self.columns else []
        lam = {i: float(x) for i, x in enumerate(values) if x > EPS_LAMBDA}
        art_total = float(sum(a.X for a in self.artificial.values()))
        return MasterLP(True, float(m.ObjVal), pi, mu, sigma, lam, art_total)

    # ------------------------------------------------------------------ #
    def solve_integer_restricted(
        self, *, time_limit: float | None, cutoff: float | None, threads: int | None = None
    ) -> dict[int, PBPlan] | None:
        """
        Restricted integer master (a primal heuristic only): the currently
        *active* columns as binaries, on a copy so the LP model and its basis
        are untouched. Returns one plan per vehicle, or ``None`` if no
        solution better than ``cutoff`` was found.
        """
        self.model.update()
        was_phase = self.phase
        if was_phase != 2:
            self.set_phase(2)
        copy = self.model.copy()
        if was_phase != 2:
            self.set_phase(was_phase)
        try:
            copy_vars = copy.getVars()
            for c in self.columns:
                cv = copy_vars[c.var.index]
                cv.UB = min(cv.UB, 1.0)
                cv.VType = GRB.BINARY
            for a in self.artificial.values():
                copy_vars[a.index].UB = 0.0
            copy.Params.OutputFlag = 0
            copy.Params.MIPGap = 0.0
            if threads is not None:
                copy.Params.Threads = threads
            if time_limit is not None:
                copy.Params.TimeLimit = time_limit
            if cutoff is not None:
                copy.Params.Cutoff = cutoff
            copy.optimize()
            self.last_rim_status = (int(copy.Status), int(copy.SolCount), float(copy.Runtime))
            if copy.SolCount == 0:
                return None
            chosen: dict[int, PBPlan] = {}
            for c in self.columns:
                if copy_vars[c.var.index].X > 0.5:
                    j = c.plan.vehicle_id
                    if j in chosen:
                        raise MasterNumericalError(f"Restricted integer master chose two columns for vehicle {j}.")
                    chosen[j] = c.plan
            missing = [j for j in self.vehicle_ids if j not in chosen]
            if missing:
                raise MasterNumericalError(f"Restricted integer master left vehicles {missing} without a column.")
            return chosen
        finally:
            copy.dispose()
