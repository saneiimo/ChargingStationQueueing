"""
Exact pricing by forward labelling (no MILP), with the MILP as fallback.

The pricing problem for (vehicle j, pile m) with duals ``pi, mu <= 0`` is

    min  w*(delta*D - a_j) + sum_{k=S}^{D-1} (a_k + b_k * q_k),   a_k = -pi_mk >= 0, b_k = -mu_mk >= 0

(``w*(delta*D - a_j)`` is the vehicle's sojourn in minutes; it depends only on D.)

over served plans (``S < D``, ``q_k`` integer in the node's bounds) whose
power can deliver ``W`` by ``D`` (no requirement if ``D = K``: censoring).

Why labelling is exact here
---------------------------
1. For a fixed ``(S, D, q)`` the cost does not depend on the power profile,
   and the one-slot energy update with the most power the slot allows,
   ``x -> x + h*min(Delta*q, P_bar, (R-x)/tau_d, (W-x)/h)``, is
   non-decreasing in ``x`` (its taper branch has slope ``1 - h/tau_d >= 0``
   because ``tau_d >= h``). So charging as fast as the modules allow is
   optimal: it reaches every energy level no later than any other profile,
   and never exceeds ``W``. The signature is feasible iff this greedy
   trajectory reaches ``W`` by ``D`` (or ``D = K``).
2. Hence the state after slot ``k`` is just the delivered energy ``x``; the
   start slot matters only through the restrictions already applied when
   the label was created. A label ``(x1, c1)`` dominates ``(x2, c2)`` at the
   same slot if ``x1 >= x2`` and ``c1 <= c2``: by (1) every continuation of
   the second is available to the first at no higher cost.
3. More modules than the power can use cost ``b_k >= 0`` more and add no
   energy, so only ``q <= ceil(P_max_possible/Delta)`` (or the node's lower
   bound, if higher) is ever extended.

Keeping only non-dominated labels per slot therefore returns the exact
minimum. If the number of labels ever exceeds ``max_labels``, the call
returns ``None`` and the caller falls back to the pricing MILP.
"""

from __future__ import annotations

import math

import gurobipy as gp

from offline_cl_opt.instance import StationSpec, VehicleData
from offline_cl_opt.model import _release_slot, sojourn_minutes

from .columns import PBPlan, column_key, modules_needed
from .pricer import INFEASIBLE, SOLVED, PBPricer, PricingNumericalError, PricingResult, reduced_cost
from .restrictions import EMPTY_RESTRICTION, VehicleRestriction
from .tolerances import EPS_CERT

# Departure test: greedy energy within this of W counts as complete. The
# greedy step caps at (W - x)/h, so a completing trajectory lands on W up to
# float rounding; 1e-9 kWh is far below every validation tolerance.
_COMPLETE_TOL = 1e-9
# Safety margin subtracted from the exact labelling minimum before it is
# reported as a lower bound (float summation noise).
_BOUND_MARGIN = 1e-9


class LabelingPricer:
    """Same interface as ``pricer.PBPricer`` (``apply`` / ``price`` / ``active``)."""

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
        max_labels: int = 200_000,
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
        self.x0 = float(initial_energy)
        self.forced_start = forced_start
        self.weight = float(weight)
        self.N = station.n_modules
        self.Delta = station.p_module
        self.p_bar = min(v.p_max, self.N * self.Delta)
        self.tau_d = v.tau_delta_hours(delta)
        self.max_labels = max_labels
        self.restriction: VehicleRestriction = EMPTY_RESTRICTION
        self.active = True
        # Built on first need only (labelling almost always suffices).
        self._milp_args = (v, pile, station, delta, K, earliest_departure)
        self._milp_kwargs = dict(
            initial_energy=initial_energy, forced_start=forced_start, weight=weight, env=env, threads=threads
        )
        self._milp: PBPricer | None = None
        self.fallbacks = 0
        self._apply_bounds()

    # ------------------------------------------------------------------ #
    def plan_cost(self, departure: int, weight: float) -> float:
        """A plan's objective cost: ``weight * (delta*D - a)``, its sojourn in minutes (0 for weight 0)."""
        return weight * sojourn_minutes(departure, self.vehicle.a, self.delta) if weight else 0.0

    def apply(self, restriction: VehicleRestriction) -> None:
        self.restriction = restriction
        self._apply_bounds()
        if self._milp is not None:
            self._milp.apply(restriction)

    def _apply_bounds(self) -> None:
        r, K, k0 = self.restriction, self.K, self.k0
        self.active = r.allows_pile(self.pile)
        self.s_lo = max(k0, r.s_min if r.s_min is not None else k0)
        self.s_hi = min(K - 1, r.s_max if r.s_max is not None else K - 1)
        if self.forced_start:
            if not (self.s_lo <= k0 <= self.s_hi):
                self.active = False
            self.s_lo = self.s_hi = k0
        self.d_lo = max(1, r.d_min if r.d_min is not None else 1)
        self.d_hi = min(K, r.d_max if r.d_max is not None else K)
        self.q_lo = {k: int(r.q_min.get(k, 0)) for k in range(k0, K)}
        self.q_hi = {k: int(min(self.N, r.q_max.get(k, self.N))) for k in range(k0, K)}
        must = [k for k, lo in r.q_min.items() if lo > 0]
        for k in must:
            if not (k0 <= k < K):
                self.active = False
        # A positive module lower bound in slot k means the vehicle is
        # plugged in during k: it started no later and departs after.
        self.must_first = min(must) if must else None
        self.must_last = max(must) if must else None
        if any(self.q_lo[k] > self.q_hi[k] for k in range(k0, K)):
            self.active = False
        if self.s_lo > self.s_hi or self.d_lo > self.d_hi:
            self.active = False

    def dispose(self) -> None:
        if self._milp is not None:
            self._milp.model.dispose()

    @property
    def model(self):  # used only for disposal by callers that expect a MILP pricer
        return self._milp.model if self._milp is not None else None

    # ------------------------------------------------------------------ #
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
        if not self.active:
            return PricingResult(self.vehicle_id, self.pile, INFEASIBLE, math.inf, None, None, None, True)
        res = self._label(pi, mu, sigma, include_departure=include_departure, eps_rc=eps_rc, max_columns=max_columns)
        if res is not None:
            return res
        # Label explosion: settle this call with the exact MILP instead.
        self.fallbacks += 1
        if self._milp is None:
            self._milp = PBPricer(*self._milp_args, **self._milp_kwargs)
            self._milp.apply(self.restriction)
        milp = self._milp
        return milp.price(
            pi, mu, sigma, include_departure=include_departure, eps_rc=eps_rc,
            time_limit=time_limit, gap_abs=gap_abs, max_columns=max_columns,
        )

    def _label(self, pi, mu, sigma, *, include_departure: bool, eps_rc: float, max_columns: int) -> PricingResult | None:
        v, K, h, Delta = self.vehicle, self.K, self.h, self.Delta
        R, W, tau_d, p_bar = v.R, v.W, self.tau_d, self.p_bar
        pile = self.pile
        w = self.weight if include_departure else 0.0
        last_start = self.s_hi if self.must_first is None else min(self.s_hi, self.must_first)
        first_departure = self.d_lo if self.must_last is None else max(self.d_lo, self.must_last + 1)

        # Label store: energy after the label's last slot, cost so far,
        # parent label id (-1 for a start), modules used in its last slot,
        # and that slot (for a start label: the start slot, q = -1).
        X: list[float] = []
        C: list[float] = []
        PAR: list[int] = []
        Q: list[int] = []
        SLOT: list[int] = []

        current: list[int] = []  # labels plugged in at the start of slot k
        candidates: list[tuple[float, int, int]] = []  # (objective, label id, departure)

        for k in range(self.k0, K):
            if self.s_lo <= k <= last_start:
                X.append(self.x0)
                C.append(0.0)
                PAR.append(-1)
                Q.append(-1)
                SLOT.append(k)
                current.append(len(X) - 1)
            if not current:
                continue
            a_k = -pi.get((pile, k), 0.0)
            b_k = -mu.get((pile, k), 0.0)
            lo, hi = self.q_lo[k], self.q_hi[k]
            children: list[tuple[float, float, int, int]] = []
            for lid in current:
                x, c = X[lid], C[lid]
                p_cap = max(0.0, min(p_bar, (R - x) / tau_d, (W - x) / h))
                q_top = min(hi, max(lo, modules_needed(p_cap, Delta)))
                for q in range(lo, q_top + 1):
                    p = min(Delta * q, p_cap)
                    children.append((x + h * p, c + a_k + b_k * q, lid, q))
            # Keep the non-dominated children: sort by energy desc, cost asc,
            # and keep each one strictly cheaper than everything with >= energy.
            children.sort(key=lambda t: (-t[0], t[1]))
            kept: list[int] = []
            best_cost = math.inf
            for x2, c2, parent, q in children:
                if c2 < best_cost:
                    best_cost = c2
                    X.append(x2)
                    C.append(c2)
                    PAR.append(parent)
                    Q.append(q)
                    SLOT.append(k)
                    kept.append(len(X) - 1)
            if len(X) > self.max_labels:
                return None
            current = kept
            D = k + 1
            if first_departure <= D <= self.d_hi:
                for lid in kept:
                    if D == K or X[lid] >= W - _COMPLETE_TOL:
                        candidates.append((C[lid] + self.plan_cost(D, w), lid, D))

        if not candidates:
            return PricingResult(self.vehicle_id, pile, INFEASIBLE, math.inf, None, None, None, True)

        candidates.sort(key=lambda t: t[0])
        best_obj = candidates[0][0]
        bound = best_obj - _BOUND_MARGIN
        certified = bound - sigma >= -eps_rc - EPS_CERT
        plan = rc = None
        extra: list[tuple[PBPlan, float]] = []
        if best_obj - sigma < -eps_rc:
            seen: set = set()
            for obj, lid, D in candidates:
                if obj - sigma >= -eps_rc or len(extra) + (plan is not None) >= max_columns:
                    break
                cand = self._reconstruct(lid, D, PAR, SLOT, Q)
                key = column_key(cand)
                if key in seen:
                    continue
                seen.add(key)
                cand_rc = reduced_cost(cand, self.plan_cost(cand.departure, w), pi, mu, sigma)
                if abs(cand_rc - (obj - sigma)) > 1e-6:
                    raise PricingNumericalError(
                        f"labelling v{self.vehicle_id} pile {pile}: rebuilt plan's reduced cost {cand_rc:.9g} "
                        f"differs from its label's {obj - sigma:.9g}"
                    )
                if plan is None:
                    plan, rc = cand, cand_rc
                elif cand_rc < -eps_rc:
                    extra.append((cand, cand_rc))
        return PricingResult(self.vehicle_id, pile, SOLVED, bound, best_obj, plan, rc, certified, extra)

    def _reconstruct(self, lid: int, D: int, PAR: list[int], SLOT: list[int], Q: list[int]) -> PBPlan:
        """Walk the label chain back to its start; recompute the greedy power."""
        qs: list[tuple[int, int]] = []
        cur = lid
        while PAR[cur] != -1:
            qs.append((SLOT[cur], Q[cur]))
            cur = PAR[cur]
        qs.reverse()
        S = SLOT[cur]
        assert qs and qs[0][0] == S and qs[-1][0] == D - 1, (S, D, qs[:2], qs[-2:])
        v, h, Delta = self.vehicle, self.h, self.Delta
        x = self.x0
        power: dict[int, float] = {}
        modules: dict[int, int] = {}
        for k, q in qs:
            p = min(Delta * q, max(0.0, min(self.p_bar, (v.R - x) / self.tau_d, (v.W - x) / h)))
            if p > 0.0:
                power[k] = p
            modules[k] = q
            x += h * p
        return PBPlan(vehicle_id=self.vehicle_id, pile=self.pile, start=S, departure=D, power=power, modules=modules)
