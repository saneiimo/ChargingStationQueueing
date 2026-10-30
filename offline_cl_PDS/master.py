"""
The two master problems of the paper (Sections 5.2-5.3), as one restricted
master LP over a growing set of plans.

Both pool the ``M`` identical piles into station-wide rows, so their duals
are the same for every pile -- exactly the symmetric prices the search's
bound needs. (``offline_cl_PB``'s master is written pile by pile instead.)

Rows, for every slot ``k = 0..K-1``:

``"aggregate"`` (28)-(29)::

    sum_plans alpha_k * lam       <= M*C        connectors in use
    sum_plans q_k * lam           <= M*N        modules in use

``"configuration"`` (22)-(23)::

    sum_plans [q_k >= t] alpha_k * lam - sum_c n_t(c) psi_kc <= 0    t = 0..N
    sum_c psi_kc                                             <= M

plus one convexity row per vehicle, ``sum_{plans of j} lam + a_j = 1``. A
*pile configuration* ``c`` is the list of module counts of the vehicles on
one pile in one slot (at most ``C`` of them, sum at most ``N``);
``n_t(c)`` counts its vehicles holding at least ``t`` modules (``n_0`` counts
all of them). Only configurations whose vector ``n(c)`` is not dominated by
another's are kept (``pile_configurations``). ``psi_kc`` is how many piles
run ``c`` in slot ``k``. The configuration rows say that, for every
threshold ``t``, the vehicles holding ``>= t`` modules fit into the piles'
configurations -- e.g. two piles of 6 modules cannot carry three vehicles
with 4 modules each, which the aggregate rows allow. The configuration
master is therefore at least as strong (Proposition 3(b)).

``a_j`` is an artificial column with cost ``ARTIFICIAL_COST`` so the LP is
feasible before every vehicle has a real plan; it only steers the duals.
Plan costs are in slots: ``w_j * D``.

Prices
------
Both masters' capacity duals become a price table ``mu[k, t] >= 0``
(``t = 0..N``): a vehicle holding ``q`` modules in slot ``k`` pays the rent
``rent[k, q] = sum_{t <= q} mu[k, t]``, and one slot's rent over the whole
station is at most ``gamma[k] = M * max_c sum_t mu[k, t] * n_t(c)`` (25).
For the aggregate master ``mu[k, 0]`` is the connector price and
``mu[k, t>=1]`` the module price, and its only "configuration" is ``C``
vehicles holding ``N`` modules in total (30). Any ``mu >= 0`` gives a valid
Lagrangian bound (Proposition 3(c)); see ``colgen.py``.
"""

from __future__ import annotations

from itertools import combinations_with_replacement

import gurobipy as gp
import numpy as np
from gurobipy import GRB

MASTER_KINDS = ("aggregate", "configuration")
ARTIFICIAL_COST = 1e6  # slots; far above any plan's cost w_j * D <= K


def pile_configurations(C: int, N: int) -> np.ndarray:
    """
    The non-dominated pile configurations as rows ``n(c) = (n_0, ..., n_N)``.
    For ``C=2, N=6``: {0,6}, {1,5}, {2,4}, {3,3}.
    """
    vectors = set()
    for size in range(1, C + 1):
        for counts in combinations_with_replacement(range(N + 1), size):
            if sum(counts) <= N:
                vectors.add(tuple(sum(1 for q in counts if q >= t) for t in range(N + 1)))
    vecs = np.array(sorted(vectors), dtype=float)
    keep = [
        i
        for i in range(len(vecs))
        if not any(np.all(vecs[o] >= vecs[i]) and np.any(vecs[o] > vecs[i]) for o in range(len(vecs)))
    ]
    return vecs[keep]


def rent_and_gamma(mu: np.ndarray, configs: np.ndarray, n_piles: int) -> tuple[np.ndarray, np.ndarray]:
    """``rent[k, q] = sum_{t<=q} mu[k, t]`` and ``gamma[k] = M * max_c mu[k] . n(c)`` (25)."""
    return np.cumsum(mu, axis=1), n_piles * (mu @ configs.T).max(axis=1)


class SlotMaster:
    """Restricted master LP of kind ``"aggregate"`` or ``"configuration"``."""

    def __init__(self, n_vehicles: int, K: int, n_piles: int, n_connectors: int, n_modules: int, kind: str) -> None:
        if kind not in MASTER_KINDS:
            raise ValueError(f"kind must be one of {MASTER_KINDS}, got {kind!r}")
        self.kind, self.K, self.M, self.N = kind, K, n_piles, n_modules
        M, C, N = n_piles, n_connectors, n_modules
        if kind == "aggregate":
            cfg = np.zeros((1, N + 1))
            cfg[0, 0], cfg[0, 1] = C, N  # C connectors and N modules (only the totals matter, (30))
            self.configs = cfg
        else:
            self.configs = pile_configurations(C, N)

        m = gp.Model("pds_master")
        m.Params.OutputFlag = 0
        m.Params.Method = 0  # primal simplex: re-optimises warm after columns are added
        m.Params.OptimalityTol = 1e-9
        m.Params.FeasibilityTol = 1e-9
        self.model = m
        if kind == "aggregate":
            self.connector_rows = [m.addConstr(gp.LinExpr() <= M * C, name=f"connectors[{k}]") for k in range(K)]
            self.module_rows = [m.addConstr(gp.LinExpr() <= M * N, name=f"modules[{k}]") for k in range(K)]
        else:
            self.threshold_rows = [
                [m.addConstr(gp.LinExpr() <= 0.0, name=f"threshold[{k},{t}]") for t in range(N + 1)] for k in range(K)
            ]
            pile_rows = [m.addConstr(gp.LinExpr() <= M, name=f"piles[{k}]") for k in range(K)]
            m.update()
            for k in range(K):
                for c, n in enumerate(self.configs):
                    col = gp.Column()
                    for t in range(N + 1):
                        if n[t] > 0:
                            col.addTerms(-float(n[t]), self.threshold_rows[k][t])
                    col.addTerms(1.0, pile_rows[k])
                    m.addVar(lb=0.0, obj=0.0, column=col, name=f"psi[{k},{c}]")
        self.convexity = [m.addConstr(gp.LinExpr() == 1.0, name=f"convexity[{j}]") for j in range(n_vehicles)]
        m.update()
        for j in range(n_vehicles):
            m.addVar(lb=0.0, obj=ARTIFICIAL_COST, column=gp.Column([1.0], [self.convexity[j]]), name=f"artificial[{j}]")
        self.plans: list[tuple[int, int, int, tuple[int, ...]]] = []  # (j, S, D, modules S..D-1)
        self._keys: set[tuple] = set()

    def add_plan(self, j: int, start: int, departure: int, modules: tuple[int, ...], cost: float) -> bool:
        """Add one plan of vehicle ``j`` (modules in slots ``start..departure-1``); False if already present."""
        key = (j, start, departure, modules)
        if key in self._keys:
            return False
        self._keys.add(key)
        col = gp.Column()
        col.addTerms(1.0, self.convexity[j])
        for k, q in zip(range(start, departure), modules):
            if self.kind == "aggregate":
                col.addTerms(1.0, self.connector_rows[k])
                if q:
                    col.addTerms(float(q), self.module_rows[k])
            else:
                for t in range(q + 1):
                    col.addTerms(1.0, self.threshold_rows[k][t])
        self.model.addVar(lb=0.0, obj=cost, column=col, name=f"lam[{j}]")
        self.plans.append(key)
        return True

    def solve(self) -> tuple[float, np.ndarray, np.ndarray]:
        """Solve the LP; returns ``(value, mu[K, N+1] >= 0, sigma[J])``."""
        m = self.model
        m.optimize()
        if m.Status != GRB.OPTIMAL:
            raise RuntimeError(f"master LP ended with Gurobi status {m.Status}")
        mu = np.zeros((self.K, self.N + 1))
        if self.kind == "aggregate":
            mu[:, 0] = [-r.Pi for r in self.connector_rows]
            mu[:, 1:] = np.array([-r.Pi for r in self.module_rows])[:, None]
        else:
            mu[:, :] = [[-r.Pi for r in row] for row in self.threshold_rows]
        sigma = np.array([r.Pi for r in self.convexity])
        # <= rows of a minimisation have duals <= 0; clamp float noise.
        return float(m.ObjVal), np.maximum(mu, 0.0), sigma

    def dispose(self) -> None:
        self.model.dispose()
