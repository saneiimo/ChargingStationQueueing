"""
The restricted master problem -- Section 5 of ``dantzig_wolfe_decomposition.html``.

Rows (25) connector capacity and (26) module capacity replace the compact
model's big-M sequencing (11)-(12) and module-link (14)-(15) entirely
(Propositions 1-2 there), leaving only ordinary packing/knapsack rows per
(pile, slot) plus one convexity row (27) per vehicle. Columns (``Plan``
objects, see ``columns.py``) are added incrementally via gurobipy's own
``Column`` object, the standard column-generation idiom -- each new
variable is attached directly to the existing constraint objects, with no
need to touch already-built rows.

The master LP (``solve_lp``) is what column generation iterates on; the
master IP (``solve_integer``, Section 9.1's price-and-branch) is solved
exactly once, at the end, over whatever columns were generated.

``purge_columns`` (Section 8.3) is the counterpart to ``add_column``: since
nothing in this module ever removes a column on its own, the master's
column count -- and hence every subsequent ``solve_lp``'s size -- only ever
grows across a column generation run unless something periodically sweeps
it. See that function's own docstring for the removal rule and, in
particular, why it also clears the removed column's ``seen_keys`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import gurobipy as gp
from gurobipy import GRB

from offline_cl_opt.instance import StationSpec

from .columns import Plan


@dataclass
class RestrictedMaster:
    """Live gurobipy model plus the bookkeeping column generation needs to
    add columns and read duals back out."""

    model: gp.Model
    station: StationSpec
    delta: float
    K: int
    k_lo: int  # rows (25)/(26) only built for k in [k_lo, K) -- Section 8.1's own suggested omission
    vehicle_ids: list[int]
    connector_cap: dict[tuple[int, int], gp.Constr]  # (pile, k) -> row (25)
    module_cap: dict[tuple[int, int], gp.Constr]  # (pile, k) -> row (26)
    convexity: dict[int, gp.Constr]  # vehicle_id -> row (27)
    columns: dict[int, list[tuple[Plan, gp.Var]]] = field(default_factory=dict)  # vehicle_id -> [(plan, lambda_var)]
    # Section 8.3's deduplication key set, per vehicle -- see _column_key
    # and add_column. Regenerating an existing column is a symptom of dual
    # cycling, not a harmless waste, so add_column silently skips it
    # rather than growing the master with redundant variables.
    seen_keys: dict[int, set[tuple]] = field(default_factory=dict)


def _column_key(plan: Plan, ndigits: int = 3) -> tuple:
    """Section 8.3: ``(pile, start, departure, rounded power profile)`` --
    the null plan's key is simply ``(None, None, K)`` (it's the same
    column for every vehicle by construction)."""
    if plan.is_null:
        return (None, None, plan.departure)
    power_key = tuple(sorted((k, round(p, ndigits)) for k, p in plan.power.items() if p > 1e-9))
    return (plan.pile, plan.start, plan.departure, power_key)


@dataclass
class LPResult:
    objective: float
    pi: dict[tuple[int, int], float]  # (pile, k) -> dual of (25)
    mu: dict[tuple[int, int], float]  # (pile, k) -> dual of (26)
    sigma: dict[int, float]  # vehicle_id -> dual of (27)


@dataclass
class IntegerResult:
    objective: float
    chosen: dict[int, Plan]  # vehicle_id -> the one plan selected


def build_master(
    vehicle_ids: list[int],
    station: StationSpec,
    delta: float,
    K: int,
    k_lo: int,
    initial_columns: dict[int, list[Plan]],
    *,
    conservative_modules: bool = False,
    model_name: str = "connector_lane_dw_master",
) -> RestrictedMaster:
    """
    Build the restricted master LP with ``initial_columns`` (must include
    the null plan for every vehicle -- see ``preprocess.seed_columns``,
    which guarantees this) and no others; column generation adds the rest.

    ``conservative_modules``: use ``(N-C+1)*Delta`` instead of ``N*Delta``
    as the RHS of every module-capacity row (26), matching the "What
    Proposition 2 does and does not give" warning box -- guarantees the
    resulting schedule is whole-module-feasible by construction (Section
    8.4 of the compact-model document, same argument as
    ``offline_cl_opt.adaptive.conservative_feasible_solution``), at the
    cost of reserving ``C-1`` modules per pile-slot regardless of load.
    Off by default: ``postprocess.repair_modules`` checks and repairs
    after the fact instead, which is usually enough and never pays that
    reservation cost up front.
    """
    m = gp.Model(model_name)
    m.Params.OutputFlag = 0

    N, Delta, C = station.n_modules, station.p_module, station.n_connectors
    module_rhs = (N - C + 1) * Delta if conservative_modules else N * Delta

    connector_cap: dict[tuple[int, int], gp.Constr] = {}
    module_cap: dict[tuple[int, int], gp.Constr] = {}
    for mm in range(station.n_piles):
        for k in range(k_lo, K):
            # (25): at most C vehicles on this pile in this slot.
            connector_cap[mm, k] = m.addConstr(gp.LinExpr(0.0) <= C, name=f"connector_cap[{mm},{k}]")
            # (26): total power on this pile in this slot at most module_rhs.
            module_cap[mm, k] = m.addConstr(gp.LinExpr(0.0) <= module_rhs, name=f"module_cap[{mm},{k}]")

    convexity: dict[int, gp.Constr] = {}
    for j in vehicle_ids:
        # (27): exactly one plan per vehicle. Built as == 0 initially (no
        # columns yet) and satisfied as soon as the null plan is added
        # below via add_column, which is required for every vehicle.
        convexity[j] = m.addConstr(gp.LinExpr(0.0) == 1, name=f"convexity[{j}]")

    rm = RestrictedMaster(
        model=m,
        station=station,
        delta=delta,
        K=K,
        k_lo=k_lo,
        vehicle_ids=list(vehicle_ids),
        connector_cap=connector_cap,
        module_cap=module_cap,
        convexity=convexity,
        columns={j: [] for j in vehicle_ids},
        seen_keys={j: set() for j in vehicle_ids},
    )

    for j, plans in initial_columns.items():
        for plan in plans:
            add_column(rm, plan)

    missing = [j for j in vehicle_ids if not any(p.is_null for p, _ in rm.columns.get(j, []))]
    if missing:
        raise ValueError(
            f"initial_columns must include the null plan for every vehicle -- missing for "
            f"{missing}. Use preprocess.seed_columns to build a valid starting set."
        )

    m.update()
    return rm


def add_column(rm: RestrictedMaster, plan: Plan) -> gp.Var | None:
    """
    Add one new lambda variable for ``plan``, wired into the existing rows
    (25)/(26)/(27) via a gurobipy ``Column`` object -- the standard
    column-generation idiom: no existing constraint is touched, only new
    nonzeros are attached to it.

    Returns ``None`` without adding anything if an equivalent column for
    this vehicle is already present (Section 8.3's deduplication key --
    ``(pile, start, departure, rounded power profile)``): "regenerating an
    existing column is a symptom of a dual cycling problem, not a harmless
    waste."
    """
    j = plan.vehicle_id
    key = _column_key(plan)
    seen = rm.seen_keys.setdefault(j, set())
    if key in seen:
        return None
    seen.add(key)

    col = gp.Column()
    col.addTerms(1.0, rm.convexity[j])  # (27): this plan counts toward vehicle j's total
    if not plan.is_null:
        pile = plan.pile
        assert pile is not None  # guaranteed by `not plan.is_null`
        for k in plan.occupied_slots():
            if (pile, k) in rm.connector_cap:
                col.addTerms(1.0, rm.connector_cap[pile, k])  # (25)
                p_val = plan.power.get(k, 0.0)
                if p_val > 1e-9:
                    col.addTerms(p_val, rm.module_cap[pile, k])  # (26)
    var = rm.model.addVar(lb=0.0, obj=float(plan.departure), column=col, name=f"lambda[{j}]")
    rm.columns.setdefault(j, []).append((plan, var))
    return var


def purge_columns(rm: RestrictedMaster, *, threshold: float = 10.0) -> int:
    """
    Section 8.3/8.2: remove non-basic columns whose reduced cost, at the
    master's own last LP solve, exceeds ``threshold`` (Section 8.2's
    suggested default, ``PURGE_THRESHOLD = 10``, in the same objective/slot
    units as the master's own objective). Never removes the null plan
    (mandatory for feasibility from iteration one, Section 7.2) or a basic
    column (still load-bearing for the current solution -- removing it
    would invalidate the very basis the reduced costs were just read from).

    Precondition: ``rm.model`` must be in the state left by a just-completed
    ``solve_lp`` call -- ``.RC``/``.VBasis`` are LP-only Gurobi attributes,
    populated for exactly the variables that took part in that solve. Don't
    call this after ``add_column`` has added variables the model hasn't
    been re-optimized with yet (they'd have no meaningful ``.RC``), and
    don't call it after ``solve_integer`` has flipped every variable to
    ``BINARY`` (there is no LP basis at all at that point). This is why
    ``run_column_generation`` purges immediately after each round's
    ``solve_lp``, before that round's new columns are added -- not after,
    despite Section 7.1's pseudocode showing it that way; solving right
    after guarantees every ``.RC``/``.VBasis`` read here is meaningful,
    which matters more than matching the pseudocode's exact position in the
    loop (the two are equivalent up to a one-round delay in when a given
    column becomes eligible to be swept).

    Crucially, a purged plan is not gone from the underlying search space:
    ``pricer.py``'s pricing subproblem always searches the *entire*
    per-vehicle plan space Omega_j (eq. 30), never just the pool of
    already-generated columns, so a plan purged here can always be
    regenerated later if it becomes attractive again at a future
    iteration's duals -- exactly as if it had never been generated the
    first time. Making that possible is why this function also drops the
    purged plan's key from ``rm.seen_keys``: leaving it there would make
    ``add_column`` silently refuse to re-add a plan this function just
    removed, permanently -- an easy, quiet bug to introduce for a purge
    routine that only remembers to touch ``rm.columns``. If the same
    column keeps getting purged and immediately regenerated across many
    purge cycles, that is Section 8.3's own diagnostic for dual cycling
    ("regenerating an existing column is a symptom of a dual cycling
    problem, not a harmless waste") -- a sign to tighten dual smoothing
    (``gamma``), not evidence purging itself is misbehaving.

    Returns the number of columns removed (``0`` if none qualified).
    """
    removed = 0
    for j, entries in rm.columns.items():
        kept: list[tuple[Plan, gp.Var]] = []
        for plan, var in entries:
            if not plan.is_null and var.VBasis != 0 and var.RC > threshold:
                rm.seen_keys[j].discard(_column_key(plan))
                rm.model.remove(var)
                removed += 1
            else:
                kept.append((plan, var))
        rm.columns[j] = kept
    if removed:
        rm.model.update()
    return removed


def solve_lp(rm: RestrictedMaster, *, check_dual_signs: bool = True) -> LPResult:
    """
    Solve the master as a continuous LP (all lambda relaxed to ``>= 0``,
    the convexity rows alone pin each vehicle's total to 1) and read back
    the duals column generation needs for pricing (28)-(29).

    ``check_dual_signs``: enforce the sign convention the source document's
    own "Sign convention" warning box calls out -- ``pi[m,k] <= 0`` and
    ``mu[m,k] <= 0`` for every row (both are ``<=`` rows in a minimisation,
    so Gurobi reports non-positive duals). On a master this size (hundreds
    of rows, many of them degenerate whenever a capacity isn't actually
    binding), Gurobi routinely reports values like ``+1e-8`` on a row that
    is mathematically exactly zero -- ordinary floating-point noise around
    a degenerate optimum, not a sign error. Anything within
    ``dual_sign_tol`` of zero is silently clamped to ``0.0`` (the
    mathematically correct value for a non-binding row) rather than left
    as a small positive number that would otherwise flip the sign of a
    rent term fed into the pricer; anything *larger* than that raises
    ``AssertionError``, since that magnitude of violation means something
    is structurally wrong with the master (e.g. a row built with the wrong
    sense), not a numerical fluke.
    """
    m = rm.model
    for var in m.getVars():
        var.VType = GRB.CONTINUOUS
    m.optimize()
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(
            f"Master LP did not solve to optimality (status={m.Status}) -- the null plan for "
            "every vehicle should always keep this feasible and bounded; this indicates a bug "
            "in how the master or its columns were built."
        )

    pi = {key: c.Pi for key, c in rm.connector_cap.items()}
    mu = {key: c.Pi for key, c in rm.module_cap.items()}
    sigma = {j: c.Pi for j, c in rm.convexity.items()}

    if check_dual_signs:
        dual_sign_tol = 1e-6
        bad_pi = [k for k, val in pi.items() if val > dual_sign_tol]
        bad_mu = [k for k, val in mu.items() if val > dual_sign_tol]
        if bad_pi or bad_mu:
            raise AssertionError(
                "Master LP duals have the wrong sign for rows (25)/(26) -- expected pi,mu<=0 "
                "for <=-constraints in a minimisation (see the source document's 'Sign "
                f"convention' box), by more than the {dual_sign_tol:g} noise tolerance. "
                f"Violating rows: pi={bad_pi[:5]}, mu={bad_mu[:5]}. This means every "
                "rent/price formula downstream would have the wrong sign; do not proceed "
                "without finding the cause."
            )
        pi = {k: min(val, 0.0) for k, val in pi.items()}
        mu = {k: min(val, 0.0) for k, val in mu.items()}

    return LPResult(objective=float(m.ObjVal), pi=pi, mu=mu, sigma=sigma)


def solve_integer(
    rm: RestrictedMaster,
    *,
    mip_gap: float | None = 1e-6,
    time_limit: float | None = None,
) -> IntegerResult:
    """
    Section 9.1's price-and-branch: re-solve the *current* restricted
    master (whatever columns column generation has generated so far) with
    every lambda forced to ``{0,1}``. Always feasible (the null plans are
    present), so its optimum is a genuine feasible schedule -- a valid
    upper bound on the true optimum, not in general the true optimum
    itself, since the best integer solution may need a column that was
    never generated.
    """
    m = rm.model
    for var in m.getVars():
        var.VType = GRB.BINARY
    if mip_gap is not None:
        m.Params.MIPGap = mip_gap
    if time_limit is not None:
        m.Params.TimeLimit = time_limit
    m.optimize()

    if m.SolCount == 0:
        raise RuntimeError(
            f"Price-and-branch found no feasible integer solution (status={m.Status}) -- "
            "the null plan for every vehicle should always make lambda=null-only feasible; "
            "this indicates a bug, not a genuinely infeasible instance."
        )

    chosen: dict[int, Plan] = {}
    for j, plans in rm.columns.items():
        picked = [plan for plan, var in plans if var.X > 0.5]
        if len(picked) != 1:
            raise RuntimeError(
                f"Vehicle {j} has {len(picked)} selected plans in the integer master solution "
                "(expected exactly 1) -- convexity row (27) should make this impossible."
            )
        chosen[j] = picked[0]

    return IntegerResult(objective=float(m.ObjVal), chosen=chosen)
