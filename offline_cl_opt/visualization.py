"""
Plots for a *solved* connector-lane MILP (``ConnectorLaneModel``):
per-vehicle power/module curves, and per-pile connector panels -- all read
directly off the model's own decision variables, not the DES simulator.
Call ``solve_cl_model`` (or ``solve_cl_model_adaptive``, and pass
``result.cl_model``) first -- these read ``.X`` values.

Adapted from ``offline_opt/visualization.py``'s ``plot_vehicle_power_and_modules``
/ ``plot_pile_power_and_modules`` for this package's different variable set:

  - No ``alpha``/``sigma`` pair. Occupancy is a single ``u[j,k]``, and
    ``S[j]``/``D[j]`` (dependent expressions, see ``model.py``) already give
    the occupied interval directly -- no reconstruction from two variables
    needed.
  - No per-vehicle module-count variable. ``offline_opt``'s ``n[j,m,k]`` has
    no analogue here -- module routing (``r[m,c,k]``) lives per lane-slot,
    not per vehicle. The "module power capacity" curve shown here is
    ``ceil(p[j,k]/Delta)*Delta`` -- ``adaptive.py``'s Lemma (Section 8.1):
    the modules this vehicle's own delivered power genuinely requires, not
    a stored decision variable.
  - Connector identity **is** a real decision here (``y[j,m,c]``), unlike
    ``offline_opt`` where a pile's connectors aren't individually named by
    the model and have to be assigned for display only
    (``assign_display_connectors``). There is no equivalent step in this
    file -- vehicles are grouped by the connector the model actually chose.

Styling reuses ``visualization.style``, same as ``offline_opt/visualization.py``.
"""

from __future__ import annotations

import math
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import HR2MIN
from models.ev import EV
from visualization import style as viz_style
from visualization.charging_theory import theoretical_power_vs_global_time

from .instance import VehicleData
from .model import ConnectorLaneModel

DEFAULT_LABEL_POWER = "Power drawn (p_jk)"
DEFAULT_LABEL_BMS = "BMS max request"
DEFAULT_LABEL_MODULES = "Module power capacity"
DEFAULT_LABEL_OTHER_POWER = "Other connector (context)"
DEFAULT_LABEL_PILE_CAPACITY = "Pile power capacity"


# ---------------------------------------------------------------------------
# Data extraction (shared by both plots)
# ---------------------------------------------------------------------------


def vehicle_slot_series(
    cl_model: ConnectorLaneModel,
    vehicle_id: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-slot ``(t, p_act, p_req, module_power)`` for one vehicle, read
    directly off the solved MILP.

    ``t`` is each slot's start time (``k*delta``), for ``k = k_j`` through
    the slot the vehicle actually departs in (``D[j].getValue()``, capped
    at the model's own horizon ``K``) -- includes any slots spent waiting
    before plug-in (``p_act`` is correctly 0 there via (13)), matching
    ``offline_opt``'s equivalent. Empty arrays if the vehicle was never
    served.

    ``p_act`` is ``p[j,k].X`` (kW). ``p_req`` reconstructs the BMS
    acceptance curve from the vehicle's own energy trajectory, using its
    *own* ``s_th`` (a per-vehicle field here, unlike ``offline_opt`` which
    takes a shared ``s_th`` parameter since its vehicles don't carry one).
    ``module_power`` is ``ceil(p_act/Delta)*Delta`` -- see this module's
    docstring.
    """
    cm = cl_model
    if vehicle_id not in cm.vehicles:
        raise KeyError(
            f"Vehicle {vehicle_id} not in this connector-lane model "
            f"(known ids: {sorted(cm.vehicles)})"
        )
    v = cm.vehicles[vehicle_id]
    k0 = cm.releases[vehicle_id]
    D_val = cm.D[vehicle_id].getValue()
    k_stop = max(k0, min(cm.K, int(round(D_val))))

    ks = list(range(k0, k_stop))
    t = np.array([k * cm.delta for k in ks])
    p_act = np.array([cm.p[vehicle_id, k].X for k in ks])

    h = cm.delta / 60.0
    Delta = cm.station.p_module
    p_req = np.zeros(len(ks))
    module_power = np.zeros(len(ks))
    x = 0.0  # cumulative energy delivered so far, kWh
    for i, p in enumerate(p_act):
        s = v.s_i + x / v.Q
        p_req[i] = v.p_max if s <= v.s_th else v.p_max * (1.0 - s) / (1.0 - v.s_th)
        module_power[i] = math.ceil(p / Delta - 1e-9) * Delta if p > 1e-9 else 0.0
        x += h * p

    return t, p_act, p_req, module_power


def vehicle_lane(cl_model: ConnectorLaneModel, vehicle_id: int) -> tuple[int, int] | None:
    """``(pile, connector)`` that ``y[j,m,c].X > 0.5`` picked for this
    vehicle (``None`` if unsolved, or never served -- (10)'s nominal
    assignment for an unserved vehicle is still reported here since it's a
    real, if physically meaningless, decision; callers that care should
    check ``_occupancy_interval`` / ``served`` first)."""
    for (mm, cc) in cl_model.lanes:
        if cl_model.y[vehicle_id, mm, cc].X > 0.5:
            return mm, cc
    return None


def _occupancy_interval(cl_model: ConnectorLaneModel, vehicle_id: int) -> tuple[int, int] | None:
    """``[k_start, k_end)`` vehicle_id actually occupies a lane, read
    straight off ``S[j]``/``D[j]`` -- no alpha/sigma reconstruction needed
    here, unlike ``offline_opt``. ``None`` if never served (``S_j = K``)."""
    cm = cl_model
    S_val = cm.S[vehicle_id].getValue()
    D_val = cm.D[vehicle_id].getValue()
    if S_val >= cm.K - 1e-6:
        return None
    return int(round(S_val)), int(round(D_val))


def pile_vehicle_intervals(
    cl_model: ConnectorLaneModel, pile_id: int
) -> list[tuple[int, int, int, int]]:
    """``(vehicle_id, connector, k_start, k_end)`` for every vehicle
    assigned to ``pile_id``, sorted by ``k_start``. ``connector`` is the
    model's own real decision (``y[j,m,c]``) -- see the module docstring."""
    cm = cl_model
    out: list[tuple[int, int, int, int]] = []
    for vid in cm.vehicles:
        lane = vehicle_lane(cm, vid)
        if lane is None or lane[0] != pile_id:
            continue
        interval = _occupancy_interval(cm, vid)
        if interval is None:
            continue
        out.append((vid, lane[1], interval[0], interval[1]))
    out.sort(key=lambda item: item[2])
    return out


def _color_by_start(intervals: list[tuple[int, int, int, int]]) -> dict[int, str]:
    ordered = sorted(intervals, key=lambda item: (item[2], item[0]))
    colors = viz_style.palette()
    return {item[0]: colors[i % len(colors)] for i, item in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Per-vehicle figure
# ---------------------------------------------------------------------------


def _plot_one_vehicle(
    cl_model: ConnectorLaneModel,
    vehicle_id: int,
    *,
    show_bms_request: bool,
    show_modules: bool,
    bar_width: float,
    label_power: str,
    label_bms: str,
    label_modules: str,
) -> plt.Figure:
    v = cl_model.vehicles[vehicle_id]
    t, p_act, p_req, module_power = vehicle_slot_series(cl_model, vehicle_id)

    fig, ax = viz_style.new_figure(figsize=viz_style.FIGSIZE_WIDE)
    c_act = viz_style.series_color("sim_actual")
    c_bms = viz_style.series_color("sim_bms")
    c_brk = viz_style.series_color("modules")

    ax.bar(t, p_act, width=bar_width, align="edge", color=c_act, alpha=0.85, label=label_power, zorder=3)
    if show_bms_request and t.size:
        ax.plot(
            t + bar_width / 2,
            p_req,
            ls="--",
            lw=viz_style.LINEWIDTH_SIM,
            color=c_bms,
            marker="o",
            markersize=3,
            label=label_bms,
            zorder=4,
        )
    if show_modules and t.size:
        # A "post" step draws each segment from one x-point to the *next*,
        # at the current point's height -- so the last slot's value has
        # nothing to step to and would otherwise never be drawn at all.
        # Close it off with one extra point at that slot's own end time.
        step_t = np.append(t, t[-1] + cl_model.delta)
        step_y = np.append(module_power, module_power[-1])
        ax.step(
            step_t,
            step_y,
            where="post",
            ls="--",
            lw=viz_style.LINEWIDTH_SIM * 0.9,
            color=c_brk,
            label=label_modules,
            zorder=2,
        )
    ax.set_xlabel("Time (min)")
    ax.set_ylabel("Power (kW)")
    ax.set_ylim(bottom=0.0)
    viz_style.style_axes(ax, title=None)
    ax.grid(True, axis="y", alpha=viz_style.GRID_ALPHA, linewidth=viz_style.GRID_LINEWIDTH)

    handles, labels = ax.get_legend_handles_labels()
    if handles:
        viz_style.style_legend(ax, handles=handles, labels=labels, loc="best")

    lane = vehicle_lane(cl_model, vehicle_id)
    lane_str = f"pile {lane[0]}, connector {lane[1]}" if lane else "unserved"
    fig_title = (
        f"EV {vehicle_id} (connector-lane OPT)  |  {lane_str}  |  "
        f"Q={v.Q:.0f} kWh, s_i={v.s_i:.2f}, s_f={v.s_f:.2f}, "
        f"a={v.a:.2f}m  |  delta={cl_model.delta:g} min"
    )
    viz_style.style_figure_title(fig, fig_title)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    plt.close(fig)
    return fig


def plot_vehicle_power_and_modules(
    cl_model: ConnectorLaneModel,
    vehicle_ids: list[int],
    *,
    show_bms_request: bool = True,
    show_modules: bool = True,
    bar_width: float = 0.9,
    label_power: str | None = None,
    label_bms: str | None = None,
    label_modules: str | None = None,
) -> list[plt.Figure]:
    """
    One figure per vehicle id, built from the connector-lane MILP's own
    solution: power drawn each slot (bars, ``p[j,k].X``), the reconstructed
    BMS max request (dashed), and the power capacity of the modules that
    power genuinely requires (dashed step, ``ceil(p/Delta)*Delta``) -- all
    three share the same kW axis, so it's easy to see which constraint (BMS
    taper vs. module allotment) is actually binding ``p_act`` at any slot.

    Parameters
    ----------
    cl_model :
        A solved ``ConnectorLaneModel`` (call ``solve_cl_model`` or
        ``solve_cl_model_adaptive`` first -- for the latter, pass
        ``result.cl_model``).
    vehicle_ids :
        Vehicle numbers to plot.
    show_bms_request, show_modules :
        Toggle the two dashed overlays.
    bar_width :
        Bar width in minutes (``<= delta`` leaves a visible gap between bars).
    label_power, label_bms, label_modules :
        Optional legend text overrides.

    Returns
    -------
    list of matplotlib Figures (one per vehicle id).
    """
    lbl_power = label_power if label_power is not None else DEFAULT_LABEL_POWER
    lbl_bms = label_bms if label_bms is not None else DEFAULT_LABEL_BMS
    lbl_modules = label_modules if label_modules is not None else DEFAULT_LABEL_MODULES

    figs: list[plt.Figure] = []
    for vid in vehicle_ids:
        figs.append(
            _plot_one_vehicle(
                cl_model,
                vid,
                show_bms_request=show_bms_request,
                show_modules=show_modules,
                bar_width=bar_width,
                label_power=lbl_power,
                label_bms=lbl_bms,
                label_modules=lbl_modules,
            )
        )
    return figs


# ---------------------------------------------------------------------------
# Per-pile figure
# ---------------------------------------------------------------------------


def plot_pile_power_and_modules(
    cl_model: ConnectorLaneModel,
    pile_id: int,
    t_start: float | None = None,
    t_end: float | None = None,
    *,
    stack_other_connectors: bool = False,
    other_alpha: float = 0.35,
    show_bms_request: bool = True,
    show_modules: bool = True,
    show_pile_capacity: bool = False,
    bar_width: float = 0.9,
    show_ev_ids: bool = True,
    show_legend: bool = True,
    figsize: tuple[float, float] | None = None,
    label_own_power: str | None = None,
    label_bms: str | None = None,
    label_modules: str | None = None,
    label_other: str | None = None,
    label_pile_capacity: str | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    One figure for ``pile_id``, built from the connector-lane MILP's own
    solution: vertical subplots = the pile's real connectors (``y[j,m,c]``
    -- no display-only reassignment needed, unlike ``offline_opt``), x = time.

    Each subplot bars its occupant(s)' power drawn each slot, with the
    reconstructed BMS max request and the power capacity the delivered
    power genuinely requires overlaid as dashed lines -- all sharing the
    same kW axis as the power bars. Optionally also a flat reference line
    at the pile's total power capacity (``show_pile_capacity``).

    Parameters
    ----------
    cl_model :
        A solved ``ConnectorLaneModel`` (call ``solve_cl_model`` or
        ``solve_cl_model_adaptive`` first -- for the latter, pass
        ``result.cl_model``).
    pile_id :
        Which pile to plot (0 .. M-1).
    t_start, t_end :
        Optional time window (minutes). Defaults to ``[0, last occupancy end
        on this pile]`` -- not the MILP's full (deliberately padded) horizon.
    stack_other_connectors :
        If True, stack every other connector's power on top of this one's
        own bars (low alpha), so total bar height at any minute reads as
        the pile's total draw at that slot.
    other_alpha :
        Opacity of the stacked "other connector" layers, in ``[0, 1]``.
    show_bms_request, show_modules :
        Toggle the two dashed overlays, drawn for the OWN connector only.
    show_pile_capacity :
        If True, draw a flat horizontal line at the pile's total power
        capacity (``n_modules * Delta``) on every subplot.
    bar_width :
        Bar width in minutes.
    show_ev_ids :
        Annotate each vehicle's own bar segment with "EV {id}".
    show_legend :
        Single shared figure-level legend.
    figsize :
        Optional ``(width, height)``. Defaults from ``visualization.style``.
    label_own_power, label_bms, label_modules, label_other, label_pile_capacity :
        Optional legend text overrides.
    title :
        Optional figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    cm = cl_model
    if pile_id < 0 or pile_id >= cm.station.n_piles:
        raise ValueError(f"pile_id={pile_id} out of range [0, {cm.station.n_piles - 1}]")

    viz_style.apply_visualization_style()

    n_connectors = cm.station.n_connectors
    delta = cm.delta
    intervals = pile_vehicle_intervals(cm, pile_id)

    t_lo = 0.0 if t_start is None else float(t_start)
    if t_end is not None:
        t_hi = float(t_end)
    elif intervals:
        # Default to actual activity on this pile, not the MILP's own
        # (deliberately generous) horizon.
        t_hi = max(k_end for _, _, _, k_end in intervals) * delta
    else:
        t_hi = cm.K * delta
    if t_hi <= t_lo:
        raise ValueError("t_end must be greater than t_start")

    lbl_own = label_own_power if label_own_power is not None else DEFAULT_LABEL_POWER
    lbl_bms = label_bms if label_bms is not None else DEFAULT_LABEL_BMS
    lbl_brk = label_modules if label_modules is not None else DEFAULT_LABEL_MODULES
    lbl_other = label_other if label_other is not None else DEFAULT_LABEL_OTHER_POWER
    lbl_cap = label_pile_capacity if label_pile_capacity is not None else DEFAULT_LABEL_PILE_CAPACITY
    pile_capacity = cm.station.n_modules * cm.station.p_module
    fig_title = (
        title
        if title is not None
        else f"Pile {pile_id} (connector-lane OPT): power by connector (t in [{t_lo:.1f}, {t_hi:.1f}])"
    )

    by_connector: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for vid, cc, k_start, k_end in intervals:
        by_connector[cc].append((vid, k_start, k_end))
    color_of = _color_by_start(intervals)
    other_colors = viz_style.palette(n_connectors)

    k_lo = max(0, int(np.floor(t_lo / delta + 1e-9)))
    k_hi = min(cm.K, int(np.ceil(t_hi / delta - 1e-9)))
    ks_window = list(range(k_lo, k_hi))
    t_axis = np.array([k * delta for k in ks_window])

    def connector_arrays(connector: int):
        """
        Combined (p_act, p_req, module_power) on the shared t_axis, for the
        stacked "other connectors" bars and the pile-capacity reference
        line, plus per-EV segments -- (vid, power, request, module,
        active-mask) -- for everything that must be drawn as one line per
        vehicle rather than one line per connector (see below).
        """
        p_act_tot = np.zeros(t_axis.shape)
        p_req_tot = np.zeros(t_axis.shape)
        mod_tot = np.zeros(t_axis.shape)
        segments: list[tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
        for vid, k_start, k_end in by_connector.get(connector, []):
            by_time = {
                t: (p, req, mp) for t, p, req, mp in zip(*vehicle_slot_series(cm, vid))
            }
            seg_p = np.zeros(t_axis.shape)
            seg_req = np.zeros(t_axis.shape)
            seg_mod = np.zeros(t_axis.shape)
            seg_active = np.zeros(t_axis.shape, dtype=bool)
            for wi, k in enumerate(ks_window):
                if k_start <= k < k_end:
                    seg_active[wi] = True
                    kt = k * delta
                    if kt in by_time:
                        seg_p[wi], seg_req[wi], seg_mod[wi] = by_time[kt]
            p_act_tot += seg_p
            p_req_tot += seg_req
            mod_tot += seg_mod
            segments.append((vid, seg_p, seg_req, seg_mod, seg_active))
        return p_act_tot, p_req_tot, mod_tot, segments

    arrays_by_connector = {n: connector_arrays(n) for n in range(n_connectors)}

    if figsize is None:
        figsize = (
            viz_style.FIGSIZE_WIDE[0],
            max(viz_style.FIGSIZE_PANEL_HEIGHT * n_connectors * 0.85, 4.0),
        )
    fig, axes = plt.subplots(n_connectors, 1, sharex=True, figsize=figsize, squeeze=False)
    axes = axes[:, 0]

    any_bars = False
    any_modules = False

    for connector in range(n_connectors):
        ax = axes[connector]
        own_p, _own_req, _own_mod, own_segments = arrays_by_connector[connector]

        if show_pile_capacity:
            ax.axhline(
                pile_capacity,
                color=viz_style.COLORS["neutral"],
                linestyle=":",
                linewidth=viz_style.LINEWIDTH_SIM * 0.9,
                alpha=0.7,
                zorder=1,
            )

        if not own_segments:
            ax.set_ylabel(f"Connector {connector}\nP (kW)")
            ax.text(
                0.5,
                0.5,
                "no EVs in window",
                transform=ax.transAxes,
                ha="center",
                va="center",
                **viz_style.annotation_kwargs(color=viz_style.COLORS["muted"]),
            )
            ax.set_xlim(t_lo, t_hi)
            ax.set_ylim(0.0, cm.station.n_modules * cm.station.p_module)
            viz_style.style_axes(ax, title=None)
            ax.grid(True, axis="y", alpha=viz_style.GRID_ALPHA, linewidth=viz_style.GRID_LINEWIDTH)
            ax.grid(True, axis="x", alpha=viz_style.GRID_ALPHA * 0.7, linewidth=viz_style.GRID_LINEWIDTH)
            continue

        for vid, seg_p, _seg_req, _seg_mod, _seg_active in own_segments:
            if not np.any(seg_p > 0):
                continue
            any_bars = True
            color = color_of[vid]
            ax.bar(t_axis, seg_p, width=bar_width, align="edge", color=color, alpha=0.85, zorder=3)
            if show_ev_ids:
                active = np.flatnonzero(seg_p > 0)
                mid_idx = int(active[len(active) // 2])
                ax.text(
                    t_axis[mid_idx] + bar_width / 2,
                    seg_p[mid_idx],
                    f"EV {vid}",
                    ha="center",
                    va="bottom",
                    **viz_style.annotation_kwargs(fontsize=viz_style.FONT_SIZE_ANNOTATION, color=color),
                )

        running_bottom = own_p.copy()
        if stack_other_connectors:
            for other in range(n_connectors):
                if other == connector:
                    continue
                other_p, _other_req, _other_mod, _other_segs = arrays_by_connector[other]
                if not np.any(other_p > 0):
                    continue
                any_bars = True
                ax.bar(
                    t_axis,
                    other_p,
                    width=bar_width,
                    align="edge",
                    bottom=running_bottom,
                    color=other_colors[other % len(other_colors)],
                    alpha=other_alpha,
                    zorder=2,
                )
                running_bottom = running_bottom + other_p

        # BMS request and module capacity are drawn ONE LINE PER VEHICLE, not
        # one line for the whole connector: own_req/own_mod sum every
        # occupant's values onto the shared t_axis (fine for a bar, since
        # each bar is its own artist), but a single ax.plot/ax.step over that
        # combined array draws one continuous polyline through every point in
        # order -- including straight through the zero gap between two
        # vehicles, or, when one vehicle hands the connector to the next with
        # no gap at all, directly from one vehicle's last value to the next
        # vehicle's first, splicing two different vehicles' curves into one.
        # Slicing out each vehicle's own contiguous active window (occupancy
        # is always one contiguous stay, Proposition 2) and giving it its own
        # plot call keeps every vehicle visually and programmatically distinct.
        for vid, _seg_p, seg_req, seg_mod, seg_active in own_segments:
            idx = np.flatnonzero(seg_active)
            if idx.size == 0:
                continue
            t_v = t_axis[idx]

            if show_bms_request:
                ax.plot(
                    t_v + bar_width / 2,
                    seg_req[idx],
                    ls="--",
                    lw=viz_style.LINEWIDTH_SIM,
                    color=viz_style.series_color("sim_bms"),
                    marker="o",
                    markersize=2.5,
                    zorder=4,
                )

            if show_modules:
                any_modules = True
                # Same closing-point fix as _plot_one_vehicle -- a "post"
                # step otherwise never draws this vehicle's last slot at
                # all -- now closed off at THIS vehicle's own end, not the
                # connector's, so it never borrows the next occupant's value.
                step_t = np.append(t_v, t_v[-1] + delta)
                step_y = np.append(seg_mod[idx], seg_mod[idx][-1])
                ax.step(
                    step_t,
                    step_y,
                    where="post",
                    ls="--",
                    lw=viz_style.LINEWIDTH_SIM * 0.9,
                    color=viz_style.series_color("modules"),
                    zorder=2,
                )

        ax.set_ylabel(f"Connector {connector}\nP (kW)")
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylim(bottom=0.0)
        viz_style.style_axes(ax, title=None)
        ax.grid(True, axis="y", alpha=viz_style.GRID_ALPHA, linewidth=viz_style.GRID_LINEWIDTH)
        ax.grid(True, axis="x", alpha=viz_style.GRID_ALPHA * 0.7, linewidth=viz_style.GRID_LINEWIDTH)

    axes[-1].set_xlabel("Time (min)")
    axes[-1].xaxis.label.set_fontproperties(viz_style.body_fontproperties(size=viz_style.FONT_SIZE_AXIS_TITLE))
    viz_style.style_figure_title(fig, fig_title)

    if show_legend and (any_bars or any_modules or show_pile_capacity):
        handles: list = [Patch(facecolor=viz_style.COLORS["neutral"], alpha=0.85, label=lbl_own)]
        if show_pile_capacity:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle=":",
                    linewidth=viz_style.LINEWIDTH_SIM * 0.9,
                    alpha=0.7,
                    label=lbl_cap,
                )
            )
        if show_bms_request:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=viz_style.series_color("sim_bms"),
                    linestyle="--",
                    linewidth=viz_style.LINEWIDTH_SIM,
                    marker="o",
                    markersize=3,
                    label=lbl_bms,
                )
            )
        if show_modules:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=viz_style.series_color("modules"),
                    linestyle="--",
                    linewidth=viz_style.LINEWIDTH_SIM * 0.9,
                    label=lbl_brk,
                )
            )
        if stack_other_connectors:
            handles.append(Patch(facecolor=viz_style.COLORS["muted"], alpha=other_alpha, label=lbl_other))
        leg = fig.legend(
            handles=handles,
            loc="upper right",
            frameon=viz_style.LEGEND_FRAMEON,
            prop=viz_style.body_fontproperties(size=viz_style.FONT_SIZE_LEGEND),
        )
        for text in leg.get_texts():
            text.set_fontproperties(viz_style.body_fontproperties(size=viz_style.FONT_SIZE_LEGEND))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Per-pile figure (simulation-style step curves) -- directly comparable to
# visualization.pile_power.plot_pile_connector_power on the same PLOT_KW.
# ---------------------------------------------------------------------------

DEFAULT_LABEL_THEORY_V2 = "Unconstrained"
DEFAULT_LABEL_SIM_BMS_V2 = "BMS Request"
DEFAULT_LABEL_ACTUAL_V2 = "Actual"
DEFAULT_LABEL_CHARGE_STEPS_V2 = "Allotted Power"
DEFAULT_LABEL_CHARGE_CHANGE_V2 = "Charge-change event"
DEFAULT_TITLE_TEMPLATE_V2 = (
    "Pile {pile_id} (connector-lane OPT): theory vs actual power by connector "
    "(t in [{t_lo:.1f}, {t_hi:.1f}])"
)


def _theory_ev(v: VehicleData, plug_in: float) -> EV:
    """Stub EV so unconstrained P(t) uses the same helper as the DES plot.

    ``c_b`` needs the simulator's kW*min convention, not this package's own
    real-kWh ``Q`` (see ``instance.py``'s module docstring, "Unit
    convention") -- ``VehicleData.from_ev`` divides by ``HR2MIN`` to get
    ``Q``, so this multiplies back by it to undo that."""
    ev = EV(
        id=v.id,
        c_b=v.Q * HR2MIN,
        s_i=v.s_i,
        s_f=v.s_f,
        arrival_time=v.a,
        s_th=v.s_th,
        service_start_time=plug_in,
    )
    ev.p_req_max = float(v.p_max)
    ev.tan_B = ev.p_req_max / (1.0 - ev.s_th)
    return ev


def _energy_at_slot(cl_model: ConnectorLaneModel, vehicle_id: int, k_slot: int) -> float:
    """Energy delivered from release up to (not including) ``k_slot``, kWh."""
    cm = cl_model
    h = cm.delta / 60.0
    x = 0.0
    for k in range(cm.releases[vehicle_id], k_slot):
        x += h * cm.p[vehicle_id, k].X
    return x


def _slot_hold_curve(
    k_start: int,
    k_end: int,
    value_at: dict[int, float],
    delta: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Piecewise-constant curve: slot ``k`` holds its value on ``[kδ, (k+1)δ)``.

    Connecting bar *tips* would interpolate across slots; this instead draws
    the true hold of each MILP decision, matching DES ``p_act`` plateaus.
    """
    ts: list[float] = []
    ys: list[float] = []
    for k in range(k_start, k_end):
        t0 = k * delta
        t1 = (k + 1) * delta
        val = float(value_at.get(k, 0.0))
        ts.append(t0)
        ys.append(val)
        ts.append(t1)
        ys.append(val)
    return np.asarray(ts, dtype=float), np.asarray(ys, dtype=float)


def _slot_start_p_req(
    v: VehicleData,
    k_start: int,
    k_end: int,
    p_at: dict[int, float],
    x_at_start: float,
    delta: float,
) -> dict[int, float]:
    """
    BMS request at the start of each occupied slot.

    Matches constraint (18): ``p_jk`` is capped by the taper limit at
    ``x_jk`` (energy *before* the slot), not by the intra-slot SoC
    trajectory. Uses ``v.s_th`` directly (a per-vehicle field here, unlike
    ``offline_opt`` which takes a shared ``s_th`` parameter).
    """
    h = delta / 60.0
    req_at: dict[int, float] = {}
    x = x_at_start
    for k in range(k_start, k_end):
        s = v.s_i + x / v.Q
        req_at[k] = v.p_max if s <= v.s_th else v.p_max * (1.0 - s) / (1.0 - v.s_th)
        x += h * float(p_at.get(k, 0.0))
    return req_at


def _opt_slot_samples(
    v: VehicleData,
    k_start: int,
    k_end: int,
    p_at: dict[int, float],
    n_at: dict[int, float],
    x_at_start: float,
    delta: float,
    p_module: float,
) -> list[tuple[float, float, float, float, int]]:
    """``(t, s, p_act, p_allot, n_modules)`` at each occupied slot start."""
    h = delta / 60.0
    samples: list[tuple[float, float, float, float, int]] = []
    x = x_at_start
    for k in range(k_start, k_end):
        s = v.s_i + x / v.Q
        n = int(round(float(n_at.get(k, 0.0))))
        samples.append((k * delta, s, float(p_at.get(k, 0.0)), n * p_module, n))
        x += h * float(p_at.get(k, 0.0))
    return samples


def plot_pile_power_and_modules_v2(
    cl_model: ConnectorLaneModel,
    pile_id: int,
    t_start: float | None = None,
    t_end: float | None = None,
    densify_dt: float = 0.25,
    figsize: tuple[float, float] | None = None,
    *,
    show_theory: bool = True,
    show_sim_bms: bool = False,
    show_actual: bool = True,
    show_charge_change_lines: bool = False,
    show_charge_change_vlines: bool = False,
    show_trace_labels: bool = False,
    show_legend: bool = True,
    label_theory: str | None = None,
    label_sim_bms: str | None = None,
    label_actual: str | None = None,
    label_charge_steps: str | None = None,
    label_charge_change: str | None = None,
    title: str | None = None,
    show_event_lines: bool = True,
    event_line_alpha: float = 0.25,
    charge_step_alpha: float = 0.55,
    charge_change_vline_alpha: float = 0.45,
    show_ev_ids: bool = True,
    theory_n_points: int = 200,
) -> plt.Figure:
    """
    Simulation-style pile figure for a *solved* connector-lane MILP.

    Same visual contract as ``visualization.pile_power.plot_pile_connector_power``
    (stacked connectors, EV colors by plug-in, shared event / charge-change
    markers) -- same parameter names too (minus ``s_th``, since this
    package's ``VehicleData`` already carries its own), so the exact same
    ``PLOT_KW`` dict used for the DES plot can be reused here for a direct,
    side-by-side comparison. Slot decisions ``p_jk`` are drawn as
    piecewise-constant holds on ``[kδ, (k+1)δ)``, not as bar tips connected
    across slots.

    Unlike ``offline_opt`` (whose MILP tracks connector *occupancy* per pile
    but not *identity*, needing ``assign_display_connectors`` for display),
    connector identity is a real decision here (``y[j,m,c]``) -- occupants
    are grouped by the connector the model actually chose, straight off
    ``pile_vehicle_intervals``.

    There is also no per-vehicle module-count decision variable to read
    (unlike ``offline_opt``'s ``n[j,m,k]``): the "modules" curve here is
    reconstructed as ``ceil(p_jk/Delta)``, same convention as
    ``vehicle_slot_series``/``plot_pile_power_and_modules``'s ``module_power``
    and Section 8.1's Lemma.

    BMS request (``show_sim_bms``) is the start-of-slot taper cap from
    constraint (18), held constant on ``[kδ, (k+1)δ)`` like the per-slot
    sample -- not the intra-slot request the DES would see. ``densify_dt``
    is kept for signature parity with the DES plotter and is unused.
    """
    cm = cl_model
    if pile_id < 0 or pile_id >= cm.station.n_piles:
        raise ValueError(f"pile_id={pile_id} out of range [0, {cm.station.n_piles - 1}]")

    viz_style.apply_visualization_style()

    n_connectors = cm.station.n_connectors
    p_module = cm.station.p_module
    pile_capacity = cm.station.n_modules * p_module
    delta = cm.delta
    intervals = pile_vehicle_intervals(cm, pile_id)  # (vid, connector, k_start, k_end)

    t_lo = 0.0 if t_start is None else float(t_start)
    if t_end is not None:
        t_hi = float(t_end)
    elif intervals:
        t_hi = max(k_end for _, _, _, k_end in intervals) * delta
    else:
        t_hi = cm.K * delta
    if t_hi <= t_lo:
        raise ValueError("t_end must be greater than t_start")

    legend_theory = label_theory if label_theory is not None else DEFAULT_LABEL_THEORY_V2
    legend_bms = label_sim_bms if label_sim_bms is not None else DEFAULT_LABEL_SIM_BMS_V2
    legend_act = label_actual if label_actual is not None else DEFAULT_LABEL_ACTUAL_V2
    legend_steps = (
        label_charge_steps if label_charge_steps is not None else DEFAULT_LABEL_CHARGE_STEPS_V2
    )
    legend_cc = (
        label_charge_change if label_charge_change is not None else DEFAULT_LABEL_CHARGE_CHANGE_V2
    )
    fig_title = (
        title
        if title is not None
        else DEFAULT_TITLE_TEMPLATE_V2.format(pile_id=pile_id, t_lo=t_lo, t_hi=t_hi)
    )

    by_connector: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for vid, cc, k_start, k_end in intervals:
        by_connector[cc].append((vid, k_start, k_end))
    color_of = _color_by_start(intervals)

    event_times: list[tuple[float, str]] = []
    occupancy_times: list[float] = []
    for vid, _cc, k_start, k_end in intervals:
        color = color_of[vid]
        t0 = k_start * delta
        t1 = k_end * delta
        occupancy_times.extend([t0, t1])
        if t_lo <= t0 <= t_hi:
            event_times.append((t0, color))
        if t_lo <= t1 <= t_hi:
            event_times.append((t1, color))

    charge_change_times: list[float] = []
    if show_charge_change_vlines:
        seen_cc: set[float] = set()
        for vid, _cc, k_start, k_end in intervals:
            prev_n = None
            for k in range(k_start, k_end):
                p_val = cm.p[vid, k].X
                n = math.ceil(p_val / p_module - 1e-9) if p_val > 1e-9 else 0
                if prev_n is not None and n != prev_n:
                    t_cc = k * delta
                    if t_lo <= t_cc <= t_hi and not any(
                        abs(t_cc - t_ex) <= 1e-6 for t_ex in occupancy_times
                    ):
                        key = round(t_cc, 6)
                        if key not in seen_cc:
                            seen_cc.add(key)
                            charge_change_times.append(t_cc)
                prev_n = n
        charge_change_times.sort()

    if figsize is None:
        figsize = (
            viz_style.FIGSIZE_WIDE[0],
            max(viz_style.FIGSIZE_PANEL_HEIGHT * n_connectors * 0.75, 4.0),
        )

    fig, axes = plt.subplots(n_connectors, 1, sharex=True, figsize=figsize, squeeze=False)
    axes = axes[:, 0]

    any_curve = False
    any_charge_steps = False

    for connector in range(n_connectors):
        ax = axes[connector]
        occupants = [
            (vid, k0, k1)
            for vid, k0, k1 in by_connector.get(connector, [])
            if not (k1 * delta < t_lo or k0 * delta > t_hi)
        ]

        if not occupants:
            ax.set_ylabel(f"Connector {connector}\nP (kW)")
            ax.text(
                0.5,
                0.5,
                "no EVs in window",
                transform=ax.transAxes,
                ha="center",
                va="center",
                **viz_style.annotation_kwargs(color=viz_style.COLORS["muted"]),
            )
            ax.set_xlim(t_lo, t_hi)
            ax.set_ylim(0.0, pile_capacity)
            viz_style.style_axes(ax, title=None)
            ax.grid(True, axis="y", alpha=viz_style.GRID_ALPHA, linewidth=viz_style.GRID_LINEWIDTH)
            ax.grid(
                True, axis="x", alpha=viz_style.GRID_ALPHA * 0.7, linewidth=viz_style.GRID_LINEWIDTH
            )
            continue

        for vid, k_start, k_end in occupants:
            v = cm.vehicles[vid]
            color = color_of[vid]
            p_at = {k: cm.p[vid, k].X for k in range(k_start, k_end)}
            n_at = {
                k: (math.ceil(p_at[k] / p_module - 1e-9) if p_at[k] > 1e-9 else 0)
                for k in range(k_start, k_end)
            }
            allot_at = {k: float(n_at[k]) * p_module for k in range(k_start, k_end)}
            x0 = _energy_at_slot(cm, vid, k_start)

            if show_charge_change_lines:
                t_al, p_al = _slot_hold_curve(k_start, k_end, allot_at, delta)
                mask = (t_al >= t_lo) & (t_al <= t_hi)
                if np.any(mask):
                    keep = mask.copy()
                    keep[1:] |= mask[:-1]
                    keep[:-1] |= mask[1:]
                    any_charge_steps = True
                    ax.plot(
                        t_al[keep],
                        p_al[keep],
                        color=color,
                        linestyle="-",
                        linewidth=viz_style.LINEWIDTH_SIM * 1.15,
                        alpha=float(np.clip(charge_step_alpha, 0.0, 1.0)),
                        zorder=3,
                    )

            if show_trace_labels:
                for t_s, s_s, p_s, _allot, n_s in _opt_slot_samples(
                    v, k_start, k_end, p_at, n_at, x0, delta, p_module
                ):
                    if t_s < t_lo or t_s > t_hi:
                        continue
                    ax.scatter(
                        [t_s],
                        [p_s],
                        s=28,
                        color=color,
                        edgecolors=viz_style.COLORS["text"],
                        linewidths=0.45,
                        zorder=6,
                    )
                    ax.annotate(
                        f"P={p_s:.1f}\nt={t_s:.1f}\ns={s_s:.3f}\nn={n_s}",
                        (t_s, p_s),
                        xytext=(4, 4),
                        textcoords="offset points",
                        ha="left",
                        va="bottom",
                        **viz_style.annotation_kwargs(
                            fontsize=max(viz_style.FONT_SIZE_ANNOTATION - 1, 6),
                            color=color,
                        ),
                        bbox={
                            "boxstyle": "round,pad=0.25",
                            "facecolor": viz_style.COLORS["figure"],
                            "edgecolor": color,
                            "alpha": 0.85,
                            "linewidth": 0.6,
                        },
                        zorder=7,
                    )

            if show_theory:
                ev_th = _theory_ev(v, k_start * delta)
                t_th, p_th = theoretical_power_vs_global_time(ev_th, n_points=theory_n_points)
                if t_th.size:
                    mask_th = (t_th >= t_lo) & (t_th <= t_hi)
                    if np.any(mask_th):
                        any_curve = True
                        ax.plot(
                            t_th[mask_th],
                            p_th[mask_th],
                            color=color,
                            linestyle="--",
                            linewidth=viz_style.LINEWIDTH_THEORY * 0.75,
                            alpha=0.9,
                        )

            if show_sim_bms:
                req_at = _slot_start_p_req(v, k_start, k_end, p_at, x0, delta)
                t_bms, p_bms = _slot_hold_curve(k_start, k_end, req_at, delta)
                if t_bms.size:
                    mask = (t_bms >= t_lo) & (t_bms <= t_hi)
                    keep = mask.copy()
                    if keep.size:
                        keep[1:] |= mask[:-1]
                        keep[:-1] |= mask[1:]
                    if np.any(keep):
                        any_curve = True
                        ax.plot(
                            t_bms[keep],
                            p_bms[keep],
                            color=color,
                            linestyle=":",
                            linewidth=viz_style.LINEWIDTH_SIM * 0.85,
                            alpha=0.85,
                            zorder=5,
                        )

            if show_actual:
                t_act, p_act = _slot_hold_curve(k_start, k_end, p_at, delta)
                if t_act.size:
                    mask = (t_act >= t_lo) & (t_act <= t_hi)
                    keep = mask.copy()
                    if keep.size:
                        keep[1:] |= mask[:-1]
                        keep[:-1] |= mask[1:]
                    if np.any(keep):
                        any_curve = True
                        ax.plot(
                            t_act[keep],
                            p_act[keep],
                            color=color,
                            linestyle="-",
                            linewidth=viz_style.LINEWIDTH_SIM,
                            alpha=0.95,
                            zorder=4,
                        )
                        if show_ev_ids:
                            tm = t_act[keep]
                            pm = p_act[keep]
                            mid = len(tm) // 2
                            ax.text(
                                tm[mid],
                                pm[mid],
                                f"EV {vid}",
                                ha="center",
                                va="bottom",
                                **viz_style.annotation_kwargs(
                                    fontsize=viz_style.FONT_SIZE_ANNOTATION,
                                    color=color,
                                ),
                            )

        ax.set_ylabel(f"Connector {connector}\nP (kW)")
        ax.set_xlim(t_lo, t_hi)
        y_top = max(pile_capacity, ax.get_ylim()[1])
        ax.set_ylim(0.0, y_top)
        viz_style.style_axes(ax, title=None)
        ax.grid(True, axis="y", alpha=viz_style.GRID_ALPHA, linewidth=viz_style.GRID_LINEWIDTH)
        ax.grid(True, axis="x", alpha=viz_style.GRID_ALPHA * 0.7, linewidth=viz_style.GRID_LINEWIDTH)

    if show_event_lines and event_times:
        alpha = float(np.clip(event_line_alpha, 0.0, 1.0))
        for ax in axes:
            for t_ev, color in event_times:
                ax.axvline(t_ev, color=color, alpha=alpha, linewidth=1.0, zorder=0)

    if show_charge_change_vlines and charge_change_times:
        cc_alpha = float(np.clip(charge_change_vline_alpha, 0.0, 1.0))
        cc_color = viz_style.COLORS["alert"]
        for ax in axes:
            for t_cc in charge_change_times:
                ax.axvline(t_cc, color=cc_color, alpha=cc_alpha, linewidth=1.15, linestyle="-.", zorder=1)
        top = axes[0]
        y_max = top.get_ylim()[1]
        for t_cc in charge_change_times:
            top.text(
                t_cc,
                y_max,
                f"{t_cc:.1f}",
                ha="center",
                va="bottom",
                rotation=90,
                **viz_style.annotation_kwargs(
                    fontsize=max(viz_style.FONT_SIZE_ANNOTATION - 1, 6), color=cc_color
                ),
                clip_on=False,
                zorder=8,
            )

    axes[-1].set_xlabel("Time (min)")
    axes[-1].xaxis.label.set_fontproperties(viz_style.body_fontproperties(size=viz_style.FONT_SIZE_AXIS_TITLE))
    viz_style.style_figure_title(fig, fig_title)

    if show_legend and (any_curve or any_charge_steps or charge_change_times):
        handles: list[Line2D] = []
        if show_theory:
            handles.append(
                Line2D(
                    [0], [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle="--",
                    linewidth=viz_style.LINEWIDTH_THEORY * 0.75,
                    label=legend_theory,
                )
            )
        if show_sim_bms:
            handles.append(
                Line2D(
                    [0], [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle=":",
                    linewidth=viz_style.LINEWIDTH_SIM * 0.85,
                    label=legend_bms,
                )
            )
        if show_actual:
            handles.append(
                Line2D(
                    [0], [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle="-",
                    linewidth=viz_style.LINEWIDTH_SIM,
                    label=legend_act,
                )
            )
        if show_charge_change_lines and any_charge_steps:
            handles.append(
                Line2D(
                    [0], [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle="-",
                    linewidth=viz_style.LINEWIDTH_SIM * 1.15,
                    alpha=charge_step_alpha,
                    label=legend_steps,
                )
            )
        if show_charge_change_vlines and charge_change_times:
            handles.append(
                Line2D(
                    [0], [0],
                    color=viz_style.COLORS["alert"],
                    linestyle="-.",
                    linewidth=1.15,
                    alpha=charge_change_vline_alpha,
                    label=legend_cc,
                )
            )
        if handles:
            leg = fig.legend(
                handles=handles,
                loc="upper right",
                frameon=viz_style.LEGEND_FRAMEON,
                prop=viz_style.body_fontproperties(size=viz_style.FONT_SIZE_LEGEND),
            )
            for text in leg.get_texts():
                text.set_fontproperties(viz_style.body_fontproperties(size=viz_style.FONT_SIZE_LEGEND))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    plt.close(fig)
    return fig
