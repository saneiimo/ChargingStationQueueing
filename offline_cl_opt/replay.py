"""
A simulation's own FIFO schedule as a solved ``ConnectorLaneModel``.

``grid_fifo_cl_model`` turns the grid replay of a finished simulation
(``MetricsTracker.grid_replay_intervals`` -- each stay rounded up to whole
slots on the lane FIFO used, in FIFO's order) into a solved compact model,
so everything that reads a solved model (``extract_solution``,
``plot_pile_power_and_modules``, ``plot_vehicle_power_and_modules``) works
on FIFO too.

The replay fixes occupancy and lanes only; it has no power. Power follows
the simulator's own rule: in every slot but a vehicle's last, it draws

    p_k = min(P_bar, (R - x_k) / tau^delta, Delta * r_k)

-- its acceptance limit or the whole modules routed to its connector,
whichever binds -- exactly as the DES draws ``min(BMS request, allotment)``.
(The last slot may draw less: the vehicle only needs what is left of
``W``.) The simulator also never leaves a module idle while a plugged-in
vehicle could use it: a vehicle holds only the modules its power needs, and
a pile hands out all of its modules whenever someone is module-limited.
That pile side is applied too when the replayed timetable admits it
(``no_idle_modules``; ``info`` says whether it held).

What remains free is the module routing ``r``, chosen so the delivered
power is as close as possible to what FIFO's power sharing actually
delivered: the target for a vehicle's ``i``-th slot is the energy the
simulation delivered to it in the ``i``-th ``delta``-window of its own
stay, divided by the slot length -- computed exactly with the simulator's
own integrator (``boundary._advance``) over its charge trace (least total
absolute deviation, reported back). Those targets are window averages, so
where FIFO's allotment changed mid-window they sit between two allotments;
fitting them without the rules above would leave a vehicle drawing less
than both its BMS request and its modules allow.

Shifting stays onto the grid changes who overlaps whom on a pile, so no
routing may satisfy even the per-vehicle rule, whole modules and completion
together; then the replay is not a schedule of the model and this raises.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import gurobipy as gp
from gurobipy import GRB

from .boundary import COHORTS_ALL, BoundaryMode, Cohort, MeasurementInstance, _advance
from .instance import StationSpec
from .model import ConnectorLaneModel, build_cl_model

if TYPE_CHECKING:
    from metrics.metrics_tracker import MetricsTracker
    from models.ev import EV


def _window_energy(ev: "EV", served_from: float, delta: float, n_windows: int) -> list[float]:
    """
    kWh the simulation delivered to ``ev`` in each window
    ``[served_from + i*delta, served_from + (i+1)*delta)``, ``i < n_windows``.

    Walks the charge trace's segments -- each starts at a recorded
    ``(time, SoC, p_act)`` with a constant allotment until the next sample
    or the departure -- splitting them at window boundaries; energy before
    ``served_from`` is skipped (a vehicle already charging at the boundary).
    """
    tan_B = ev.c_b * ev.c_rate / (1.0 - ev.s_th)
    rows = sorted(ev.charge_trace, key=lambda row: row[0])
    segments = [(float(r[0]), float(r[1]), float(r[3])) for r in rows]  # (t, SoC, p_act)
    end = float(ev.departure_time)
    energy = [0.0] * n_windows
    for idx, (t_start, s, p_act) in enumerate(segments):
        t_end = min(segments[idx + 1][0] if idx + 1 < len(segments) else end, end)
        if t_end <= t_start + 1e-12 or p_act <= 1e-9:
            continue
        t = t_start
        while t < t_end - 1e-12:
            if t < served_from - 1e-12:
                step_end = min(served_from, t_end)
                s, _ = _advance(s, p_act, tan_B, ev.c_b, step_end - t)
                t = step_end
                continue
            i = int(math.floor(round((t - served_from) / delta, 9)))
            step_end = min(served_from + (i + 1) * delta, t_end)
            s, e = _advance(s, p_act, tan_B, ev.c_b, step_end - t)
            if 0 <= i < n_windows:
                energy[i] += e
            t = step_end
    return energy


def grid_fifo_cl_model(
    metrics: "MetricsTracker",
    inst: MeasurementInstance,
    station: StationSpec,
    delta: float,
    horizon_minutes: float,
    *,
    objective_cohorts: frozenset[Cohort] = COHORTS_ALL,
    since: float | None = None,
    no_idle_modules: bool | None = None,
) -> tuple[ConnectorLaneModel, dict[str, Any]]:
    """
    The grid-replayed FIFO schedule as a solved compact model (see this
    module's docstring).

    Parameters
    ----------
    metrics :
        ``env.engine.metrics`` of a finished (drained) simulation.
    inst :
        The offline instance built from that run
        (``build_measurement_instance``), on ``delta`` and ``horizon_minutes``.
    station, delta, horizon_minutes, objective_cohorts :
        As for ``build_cl_model``; ``horizon_minutes`` must cover the replay.
    since :
        The replay's time origin; defaults to ``metrics.warmup_period`` (the
        boundary the instance was built at).
    no_idle_modules :
        The pile side of the simulator's rule, on top of the per-vehicle
        ``p = min(...)``: a vehicle holds no more modules than its power
        needs, an empty connector holds none, and a vehicle may be limited
        by its modules only in a slot where its pile hands out all ``N``.
        ``True`` requires it (raises if impossible), ``False`` skips it, and
        ``None`` (default) tries it and falls back to the per-vehicle rule
        alone when no routing satisfies it on the replayed timetable.

    Returns
    -------
    ``(cl_model, info)``: the solved model (``extract_solution`` reports the
    replay's total sojourn as its objective; the power fit is a secondary
    objective, set up like ``build_cl_model``'s ``tie_break``), and ``info``
    with ``total_abs_deviation_kwh`` (energy moved away from FIFO's own
    per-window deliveries to fit the rules), ``max_abs_deviation_kw``
    (largest single-slot power change) and ``no_idle_modules`` (whether the
    pile-side rule held).
    """
    since = float(metrics.warmup_period) if since is None else float(since)
    intervals = metrics.grid_replay_intervals(delta, since)
    by_id = {ev.id: ev for ev in metrics.finished_evs}
    K = math.ceil(round(horizon_minutes / delta, 9))
    h = delta / 60.0

    targets: dict[tuple[int, int], float] = {}
    plan: dict[int, tuple[int, int, int, int]] = {}  # vid -> (S, D, pile, connector)
    for v in inst.vehicles:
        bv = inst.boundary_vehicles.get(v.id)
        if bv is not None and bv.mode is BoundaryMode.FIXED:
            continue  # its trajectory is pinned by the model itself
        if v.id not in intervals:
            raise ValueError(f"vehicle {v.id} has no grid-replay interval (did the simulation finish it?)")
        S, D = intervals[v.id]
        if D > K:
            raise ValueError(f"the grid replay departs vehicle {v.id} at slot {D}, after the horizon K={K}")
        ev = by_id[v.id]
        served_from = max(float(ev.service_start_time), since)  # type: ignore[arg-type]
        for i, e in enumerate(_window_energy(ev, served_from, delta, D - S)):
            targets[v.id, S + i] = e / h
        plan[v.id] = (S, D, ev.pile_tracker.id, ev.connector_id_tracker)  # type: ignore[union-attr,assignment]

    def solve(pile_rule: bool):
        return _solve_replay(
            inst, station, delta, horizon_minutes, objective_cohorts, plan, targets, pile_rule=pile_rule
        )

    if no_idle_modules is None:
        cl, dev = solve(True)
        pile_rule = cl is not None
        if cl is None:
            cl, dev = solve(False)
    else:
        pile_rule = no_idle_modules
        cl, dev = solve(no_idle_modules)
    if cl is None or dev is None:
        raise ValueError(
            "the grid-replayed FIFO timetable is not a schedule of the model: no whole-module "
            "routing completes every vehicle on it with each drawing min(acceptance limit, "
            "modules) until its last slot"
            + (" and no module idle while a vehicle could use it" if pile_rule else "")
            + " (shifting stays onto the grid changed how the piles' modules are shared)."
        )
    info = {
        "total_abs_deviation_kwh": float(h * sum(d.X for d in dev.values())),
        "max_abs_deviation_kw": float(max((d.X for d in dev.values()), default=0.0)),
        "no_idle_modules": pile_rule,
    }
    return cl, info


def _solve_replay(inst, station, delta, horizon_minutes, objective_cohorts, plan, targets, *, pile_rule: bool):
    """Build and solve one replay model: ``(cl_model, dev_vars)``, or ``(None, None)`` if infeasible."""
    cl = build_cl_model(
        inst.vehicles,
        station,
        delta,
        horizon_minutes,
        boundary_vehicles=inst.boundary_vehicles,
        cohorts=inst.cohorts,
        objective_cohorts=objective_cohorts,
        tie_break=False,
        model_name="grid_fifo_replay",
    )
    m = cl.model
    for j, (S, D, pile, conn) in plan.items():
        for k in range(cl.releases[j], cl.K):
            cl.u[j, k].LB = cl.u[j, k].UB = 1.0 if S <= k < D else 0.0
        for lane in cl.lanes:
            on = 1.0 if lane == (pile, conn) else 0.0
            cl.y[j, lane[0], lane[1]].LB = cl.y[j, lane[0], lane[1]].UB = on

    # Work conservation: in every slot but its last, a vehicle draws
    # min(P_bar, taper cap, Delta * modules on its connector). The model's
    # own rows already give p <= each term; one binary per term picks which
    # one p also reaches.
    N, Delta = station.n_modules, station.p_module
    by_vid = {v.id: v for v in inst.vehicles}
    module_bound = {}  # (j, k) -> binary: 1 when the modules are what limit p_jk
    for j, (S, D, pile, conn) in plan.items():
        v = by_vid[j]
        bv = inst.boundary_vehicles.get(j)
        x0 = bv.initial_energy_kwh if bv is not None else 0.0
        p_bar = min(v.p_max, N * Delta)
        tau_d = v.tau_delta_hours(delta)
        big_m = max(N * Delta, p_bar, v.R / tau_d) + 1.0
        for k in range(S, D - 1):
            x_prev = cl.x[j, k] if k > cl.releases[j] else x0
            caps = (Delta * cl.r[pile, conn, k], p_bar, (v.R - x_prev) / tau_d)
            z = m.addVars(len(caps), vtype=GRB.BINARY, name=f"binding[{j},{k}]")
            m.addConstr(z.sum() == 1, name=f"one_binding[{j},{k}]")
            for i, cap in enumerate(caps):
                m.addConstr(cl.p[j, k] >= cap - big_m * (1 - z[i]), name=f"work_conserving[{j},{k},{i}]")
            module_bound[j, k] = z[0]

    if pile_rule:
        # The pile side of the simulator's policy: a vehicle holds no more
        # modules than its power needs (r <= ceil(p/Delta)); an empty lane
        # holds none; and a vehicle may be module-limited only in a slot
        # where its pile hands out all N modules.
        occupant = {}
        for j, (S, D, pile, conn) in plan.items():
            for k in range(S, D):
                occupant[pile, conn, k] = j
        for (pile, conn, k), r_var in cl.r.items():
            j = occupant.get((pile, conn, k))
            if j is None:
                r_var.UB = 0.0
            else:
                m.addConstr(Delta * r_var <= cl.p[j, k] + Delta - 1e-4, name=f"no_hoarding[{pile},{conn},{k}]")
        saturated = {}
        for (j, k), z0 in module_bound.items():
            pile = plan[j][2]
            if (pile, k) not in saturated:
                w = m.addVar(vtype=GRB.BINARY, name=f"pile_saturated[{pile},{k}]")
                lanes_m = [lane for lane in cl.lanes if lane[0] == pile]
                m.addConstr(
                    gp.quicksum(cl.r[pile, c, k] for _, c in lanes_m) >= N * w, name=f"saturation[{pile},{k}]"
                )
                saturated[pile, k] = w
            m.addConstr(z0 <= saturated[pile, k], name=f"module_limited_only_if_saturated[{j},{k}]")

    # One solve, two objectives, as with build_cl_model's tie_break: total
    # sojourn first (a constant here -- occupancy is fixed -- so it is the
    # replay's own value), then the power fit: the routing whose power is
    # closest to FIFO's own deliveries. extract_solution reads index 0 when
    # tie_break is set, so it reports the sojourn, not the fit.
    sojourn_objective = m.getObjective()
    dev = m.addVars(list(targets), lb=0.0, name="dev")
    for (j, k), target in targets.items():
        m.addConstr(dev[j, k] >= cl.p[j, k] - target, name=f"dev_up[{j},{k}]")
        m.addConstr(dev[j, k] >= target - cl.p[j, k], name=f"dev_dn[{j},{k}]")
    m.ModelSense = GRB.MINIMIZE
    m.setObjectiveN(
        sojourn_objective, index=0, priority=1, weight=1.0, abstol=1e-6, reltol=0.0, name="total_sojourn"
    )
    m.setObjectiveN(dev.sum(), index=1, priority=0, weight=1.0, name="fifo_power_fit")
    cl.tie_break = True
    m.Params.OutputFlag = 0
    m.optimize()
    if m.Status in (GRB.INFEASIBLE, GRB.INF_OR_UNBD):
        m.dispose()
        return None, None
    if m.Status != GRB.OPTIMAL:
        raise RuntimeError(f"grid replay power fit ended with Gurobi status {m.Status}")
    return cl, dev
