"""
Every numerical tolerance the branch-and-price solver uses, in one place.

How exactness is kept despite floating point
--------------------------------------------
Two things, and only two, can make the solver report a wrong optimum: a
schedule accepted as an incumbent that is not really feasible, or a node
pruned on a bound that is not really a lower bound. The code guards each
separately:

* Every incumbent passes ``validation.validate_schedule`` (physics, whole
  modules, connector counts) with ``VALIDATION_TOL`` before it can lower
  ``UB``, whatever produced it.
* Every pruning bound is a Lagrangian bound (``colgen``) built from the
  pricing solver's own ``ObjBound`` and from master duals clamped to the
  sign the Lagrangian argument needs -- never from a restricted-master LP
  value. Pruning then also subtracts ``prune_slack`` before rounding up to
  the next attainable objective value (the objective, total sojourn, takes
  values ``delta * n - sum_j w_j a_j`` for integer ``n`` -- see
  ``bp.BranchAndPrice.objective_index``), so float noise can only make
  pruning *more* conservative.

Everything else below (``EPS_RC``, ``EPS_LAMBDA``, ...) only steers how fast
the search converges, not whether its answer is right.
"""

from __future__ import annotations

# A column is "improving" if its reduced cost is below -EPS_RC. Kept above
# the master LP's optimality tolerance (MASTER_OPTIMALITY_TOL) so an
# already-present column can never look improving through LP noise alone.
EPS_RC = 1e-6

# Slack when reading a pricer's ObjBound as a "no improving column"
# certificate: bound - sigma >= -EPS_RC - EPS_CERT.
EPS_CERT = 1e-7

# A master variable is in the positive support if lambda > EPS_LAMBDA.
EPS_LAMBDA = 1e-7

# Phase I: the node master is feasible once the artificial total is at or
# below EPS_FEAS; infeasible once a Phase-I Lagrangian bound exceeds it.
EPS_FEAS = 1e-7

# Dual-sign guard: a <= row's dual above this is treated as a sign bug and
# raised; anything between 0 and this is clamped to 0.
DUAL_SIGN_TOL = 1e-6

# Base slack (objective units, minutes) subtracted from a lower bound before
# rounding it up to the next attainable objective value (attainable values
# are delta apart -- see the module docstring). See ``prune_slack`` for the
# J-dependent part.
PRUNE_SLACK_BASE = 1e-4

# ceil(p / Delta - MODULE_EPS): whole modules needed for power p. Same
# 1e-9 guard ``offline_cl_opt.boundary.assert_whole_module_feasible`` uses,
# so a power that is a module multiple up to float noise needs exactly that
# many modules.
MODULE_EPS = 1e-9

# Absolute tolerance (kW for power, kWh for energy) for every feasibility
# check on a finished schedule, including the compact-model row check.
# Gurobi's own default FeasibilityTol is 1e-6; this is one decade looser so
# a schedule read back from the compact model's own solution also passes.
VALIDATION_TOL = 1e-5

# Gurobi parameters for the master LP and the pricing MILPs. Tighter than
# Gurobi's defaults so the clean-up of a pricer's solution (rounding u/q,
# clipping p) moves it by ~1e-9, far below VALIDATION_TOL and EPS_RC.
MASTER_OPTIMALITY_TOL = 1e-9
MASTER_FEASIBILITY_TOL = 1e-9
PRICER_FEASIBILITY_TOL = 1e-9
PRICER_INT_FEAS_TOL = 1e-9


def prune_slack(n_vehicles: int) -> float:
    """
    Slack subtracted from a node's lower bound before rounding it up to the
    next attainable objective value.

    A converged node's Lagrangian bound can sit below its restricted-master
    LP value by up to ``J * (EPS_RC + EPS_CERT)`` (each vehicle's pricing
    minimum is only certified to that tolerance). If the slack were smaller
    than that, a node whose LP value is exactly an attainable value ``z`` could fail
    to prune against ``UB = z``. A larger slack is always safe: it only ever
    keeps a node open that could have been closed.
    """
    return max(PRUNE_SLACK_BASE, 10.0 * n_vehicles * (EPS_RC + EPS_CERT))
