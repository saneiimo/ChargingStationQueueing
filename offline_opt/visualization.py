"""
Plots for a *solved* offline MILP (``offline_opt.model.OfflineModel``):
per-vehicle power/modules/BMS-request curves, and per-pile dispenser panels --
all read directly off the model's own decision variables, not the DES
simulator. Call ``solve_offline_model`` first (these read ``.X`` values).

Dispenser caveat
-----------------
The MILP tracks how many dispensers are occupied on a pile each slot
(constraint 12), not *which* one -- dispenser identity is not a decision
variable (``y_jm`` is pile-level only). For the per-pile figure,
``assign_display_dispensers`` tiles each vehicle's ``[k_start, k_end)``
occupancy interval onto the lowest-index free dispenser slot, purely for
display. This is always feasible: constraint (12) guarantees at most N
vehicles overlap on that pile at any slot, so an earliest-available-dispenser
assignment never gets stuck.

Styling reuses ``visualization.style`` so figures match the rest of the
project's plots (fonts, palette, grid, "sim_bms" / "sim_actual" series
colors -- reused here for the MILP's own BMS-request / power curves).
"""

from __future__ import annotations

from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from config import HR2MIN, S_THRESH
from visualization import style as viz_style

from .model import OfflineModel

DEFAULT_LABEL_POWER = "Power drawn (p_jk)"
DEFAULT_LABEL_BMS = "BMS max request"
DEFAULT_LABEL_MODULES = "Module power capacity"
DEFAULT_LABEL_OTHER_POWER = "Other dispenser (context)"
DEFAULT_LABEL_PILE_CAPACITY = "Pile power capacity"


# ---------------------------------------------------------------------------
# Data extraction (shared by both plots)
# ---------------------------------------------------------------------------


def vehicle_slot_series(
    offline_model: OfflineModel,
    vehicle_id: int,
    s_th: float = S_THRESH,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-slot ``(t, p_act, p_req, n_modules)`` for one vehicle, read directly
    off the solved MILP.

    ``t`` is each slot's start time (``k*delta``), for ``k = k_j`` through
    the slot the vehicle is marked finished in (``sigma[j,k]`` first hits 1;
    all released-but-unfinished-longer slots are dropped from the arrays --
    otherwise ``p_req`` would keep reporting a request at the frozen
    departure SoC all the way to the horizon end, which is meaningless once
    the vehicle has actually left). ``p_act`` is ``p[j,k].X`` (kW).
    ``n_modules`` is modules held across whichever pile the vehicle was
    assigned (``sum_m n[j,m,k].X``; only one ``m`` is ever nonzero).
    ``p_req`` reconstructs the BMS max-acceptance curve from the vehicle's
    own energy trajectory -- SoC(k) = s_i + x_jk/Q, recomputed here from
    ``p[j,k].X`` since the model only keeps ``x_jk`` (eq. 2) as a build-time
    linear expression, not a stored value.
    """
    om = offline_model
    if vehicle_id not in om.vehicles:
        raise KeyError(
            f"Vehicle {vehicle_id} not in this offline model "
            f"(known ids: {sorted(om.vehicles)})"
        )
    v = om.vehicles[vehicle_id]
    k0 = om.releases[vehicle_id]
    ks: list[int] = []
    for k in range(k0, om.K):
        ks.append(k)
        if om.sigma[vehicle_id, k].X > 0.5:
            break  # finished in this slot; later slots are all-zero and misleading to plot

    t = np.array([k * om.delta for k in ks])
    p_act = np.array([om.p[vehicle_id, k].X for k in ks])
    n_modules = np.array(
        [sum(om.n[vehicle_id, mm, k].X for mm in range(om.station.n_piles)) for k in ks]
    )

    p_req = np.zeros(len(ks))
    x = 0.0
    for i, p in enumerate(p_act):
        s = v.s_i + x / v.Q
        p_req[i] = v.p_max if s <= s_th else v.p_max * (1.0 - s) / (1.0 - s_th)
        x += om.delta * p

    return t, p_act, p_req, n_modules


def vehicle_pile(offline_model: OfflineModel, vehicle_id: int) -> int | None:
    """Which pile ``y[j,m].X > 0.5`` picked for this vehicle (None if unsolved/unassigned)."""
    for mm in range(offline_model.station.n_piles):
        if offline_model.y[vehicle_id, mm].X > 0.5:
            return mm
    return None


def _occupancy_interval(offline_model: OfflineModel, vehicle_id: int) -> tuple[int, int] | None:
    """[k_start, k_end): slots where vehicle_id is actually connected (alpha-sigma=1)."""
    om = offline_model
    k_start = None
    k_end = om.K
    for k in range(om.releases[vehicle_id], om.K):
        plugged_in = om.alpha[vehicle_id, k].X > 0.5
        finished = om.sigma[vehicle_id, k].X > 0.5
        if plugged_in and not finished and k_start is None:
            k_start = k
        if finished:
            k_end = k
            break
    if k_start is None:
        return None
    return k_start, k_end


def pile_vehicle_intervals(offline_model: OfflineModel, pile_id: int) -> list[tuple[int, int, int]]:
    """``(vehicle_id, k_start, k_end)`` for every vehicle assigned to ``pile_id``, sorted by k_start."""
    om = offline_model
    out: list[tuple[int, int, int]] = []
    for vid in om.vehicles:
        if vehicle_pile(om, vid) != pile_id:
            continue
        interval = _occupancy_interval(om, vid)
        if interval is None:
            continue
        out.append((vid, interval[0], interval[1]))
    out.sort(key=lambda item: item[1])
    return out


def assign_display_dispensers(
    intervals: list[tuple[int, int, int]], n_dispensers: int
) -> dict[int, list[tuple[int, int, int]]]:
    """
    Greedy lowest-index-free-dispenser assignment of occupancy intervals, for
    display only -- see the module docstring. Always feasible given a valid
    MILP solution (constraint 12 caps concurrent occupancy at n_dispensers).
    """
    free_at = [0] * n_dispensers  # slot index each dispenser becomes free again
    by_dispenser: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for vid, k_start, k_end in intervals:
        candidates = [n for n in range(n_dispensers) if free_at[n] <= k_start]
        if not candidates:
            raise RuntimeError(
                f"No free display dispenser for vehicle {vid} at slot {k_start} "
                "-- this would mean the solution violates constraint (12)."
            )
        dispenser = min(candidates)
        by_dispenser[dispenser].append((vid, k_start, k_end))
        free_at[dispenser] = k_end
    return by_dispenser


def _color_by_start(intervals: list[tuple[int, int, int]]) -> dict[int, str]:
    ordered = sorted(intervals, key=lambda item: (item[1], item[0]))
    colors = viz_style.palette()
    return {vid: colors[i % len(colors)] for i, (vid, _, _) in enumerate(ordered)}


# ---------------------------------------------------------------------------
# Per-vehicle figure
# ---------------------------------------------------------------------------


def _plot_one_vehicle(
    offline_model: OfflineModel,
    vehicle_id: int,
    s_th: float,
    *,
    show_bms_request: bool,
    show_modules: bool,
    bar_width: float,
    label_power: str,
    label_bms: str,
    label_modules: str,
) -> plt.Figure:
    v = offline_model.vehicles[vehicle_id]
    t, p_act, p_req, n_modules = vehicle_slot_series(offline_model, vehicle_id, s_th)
    module_power = n_modules * offline_model.station.p_module

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
        ax.step(
            t,
            module_power,
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

    pile = vehicle_pile(offline_model, vehicle_id)
    fig_title = (
        f"EV {vehicle_id} (offline OPT)  |  pile {pile}  |  "
        f"Q={v.Q / HR2MIN:.0f} kWh, s_i={v.s_i:.2f}, s_f={v.s_f:.2f}, "
        f"a={v.a:.2f}m  |  delta={offline_model.delta:g} min"
    )
    viz_style.style_figure_title(fig, fig_title)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    plt.close(fig)
    return fig


def plot_vehicle_power_and_modules(
    offline_model: OfflineModel,
    vehicle_ids: list[int],
    *,
    s_th: float = S_THRESH,
    show_bms_request: bool = True,
    show_modules: bool = True,
    bar_width: float = 0.9,
    label_power: str | None = None,
    label_bms: str | None = None,
    label_modules: str | None = None,
) -> list[plt.Figure]:
    """
    One figure per vehicle id, built from the offline MILP's own solution:
    power drawn each slot (bars, ``p[j,k].X``), the reconstructed BMS max
    request (dashed), and the power capacity of the modules assigned (dashed
    step, ``sum_m n[j,m,k].X * Delta``) -- all three share the same kW axis,
    so it's easy to see which constraint (BMS taper vs. module allotment) is
    actually binding ``p_act`` at any slot.

    Parameters
    ----------
    offline_model :
        A solved ``OfflineModel`` (call ``solve_offline_model`` first).
    vehicle_ids :
        Vehicle numbers to plot.
    s_th :
        BMS taper knee used to reconstruct ``p_req`` from each vehicle's own
        energy trajectory. Defaults to ``config.S_THRESH`` -- pass the same
        value used to build the model (via ``taper_time_constant``).
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
                offline_model,
                vid,
                s_th,
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
    offline_model: OfflineModel,
    pile_id: int,
    t_start: float | None = None,
    t_end: float | None = None,
    *,
    s_th: float = S_THRESH,
    stack_other_dispensers: bool = False,
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
    One figure for ``pile_id``, built from the offline MILP's own solution:
    vertical subplots = *display* dispensers (see module docstring), x = time.

    Each subplot bars its occupant(s)' power drawn each slot, with the
    reconstructed BMS max request and the power capacity of the modules
    assigned overlaid as dashed lines -- all sharing the same kW axis as the
    power bars. Optionally also a flat reference line at the pile's total
    power capacity (``show_pile_capacity``), the same on every subplot since
    it is a station-layout constant, not a per-dispenser quantity.

    Parameters
    ----------
    offline_model :
        A solved ``OfflineModel`` (call ``solve_offline_model`` first).
    pile_id :
        Which pile to plot (0 .. M-1).
    t_start, t_end :
        Optional time window (minutes). Defaults to ``[0, last occupancy end
        on this pile]`` -- not the MILP's full (deliberately padded) horizon,
        which is typically far longer than actual activity.
    s_th :
        BMS taper knee; see ``plot_vehicle_power_and_modules``.
    stack_other_dispensers :
        If True, stack every other display dispenser's power on top of this
        one's own bars (low alpha), so total bar height at any minute reads
        as the pile's total draw at that slot.
    other_alpha :
        Opacity of the stacked "other dispenser" layers, in ``[0, 1]``.
    show_bms_request, show_modules :
        Toggle the two dashed overlays, drawn for the OWN dispenser only.
    show_pile_capacity :
        If True, draw a flat horizontal line at the pile's total power
        capacity (``n_modules * Delta``, i.e. every module on the pile in use
        at once) on every subplot -- a ceiling to compare the (optionally
        stacked) bars against.
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
    om = offline_model
    if pile_id < 0 or pile_id >= om.station.n_piles:
        raise ValueError(f"pile_id={pile_id} out of range [0, {om.station.n_piles - 1}]")

    viz_style.apply_visualization_style()

    n_dispensers = om.station.n_dispensers
    delta = om.delta
    intervals = pile_vehicle_intervals(om, pile_id)

    t_lo = 0.0 if t_start is None else float(t_start)
    if t_end is not None:
        t_hi = float(t_end)
    elif intervals:
        # Default to actual activity on this pile, not the MILP's own
        # (deliberately generous) horizon -- the latter is typically padded
        # far past when every vehicle here has actually finished.
        t_hi = max(k_end for _, _, k_end in intervals) * delta
    else:
        t_hi = om.K * delta
    if t_hi <= t_lo:
        raise ValueError("t_end must be greater than t_start")

    lbl_own = label_own_power if label_own_power is not None else DEFAULT_LABEL_POWER
    lbl_bms = label_bms if label_bms is not None else DEFAULT_LABEL_BMS
    lbl_brk = label_modules if label_modules is not None else DEFAULT_LABEL_MODULES
    lbl_other = label_other if label_other is not None else DEFAULT_LABEL_OTHER_POWER
    lbl_cap = label_pile_capacity if label_pile_capacity is not None else DEFAULT_LABEL_PILE_CAPACITY
    pile_capacity = om.station.n_modules * om.station.p_module
    fig_title = (
        title
        if title is not None
        else f"Pile {pile_id} (offline OPT): power by display dispenser (t in [{t_lo:.1f}, {t_hi:.1f}])"
    )

    by_dispenser = assign_display_dispensers(intervals, n_dispensers)
    color_of = _color_by_start(intervals)
    other_colors = viz_style.palette(n_dispensers)

    k_lo = max(0, int(np.floor(t_lo / delta + 1e-9)))
    k_hi = min(om.K, int(np.ceil(t_hi / delta - 1e-9)))
    ks_window = list(range(k_lo, k_hi))
    t_axis = np.array([k * delta for k in ks_window])

    def dispenser_arrays(dispenser: int):
        """Combined (p_act, p_req, n_modules) on the shared t_axis, plus per-EV power segments."""
        p_act_tot = np.zeros(t_axis.shape)
        p_req_tot = np.zeros(t_axis.shape)
        n_brk_tot = np.zeros(t_axis.shape)
        segments: list[tuple[int, np.ndarray]] = []
        for vid, k_start, k_end in by_dispenser.get(dispenser, []):
            by_slot = {
                k: (p, req, nb)
                for k, p, req, nb in zip(*vehicle_slot_series(om, vid, s_th))
            }
            seg_p = np.zeros(t_axis.shape)
            seg_req = np.zeros(t_axis.shape)
            seg_brk = np.zeros(t_axis.shape)
            for wi, k in enumerate(ks_window):
                if k_start <= k < k_end:
                    kt = k * delta
                    if kt in by_slot:
                        seg_p[wi], seg_req[wi], seg_brk[wi] = by_slot[kt]
            p_act_tot += seg_p
            p_req_tot += seg_req
            n_brk_tot += seg_brk
            segments.append((vid, seg_p))
        return p_act_tot, p_req_tot, n_brk_tot, segments

    arrays_by_dispenser = {n: dispenser_arrays(n) for n in range(n_dispensers)}

    if figsize is None:
        figsize = (
            viz_style.FIGSIZE_WIDE[0],
            max(viz_style.FIGSIZE_PANEL_HEIGHT * n_dispensers * 0.85, 4.0),
        )
    fig, axes = plt.subplots(n_dispensers, 1, sharex=True, figsize=figsize, squeeze=False)
    axes = axes[:, 0]

    any_bars = False
    any_modules = False

    for dispenser in range(n_dispensers):
        ax = axes[dispenser]
        own_p, own_req, own_brk, own_segments = arrays_by_dispenser[dispenser]

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
            ax.set_ylabel(f"Dispenser {dispenser}\nP (kW)")
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
            ax.set_ylim(0.0, om.station.n_modules * om.station.p_module)
            viz_style.style_axes(ax, title=None)
            ax.grid(True, axis="y", alpha=viz_style.GRID_ALPHA, linewidth=viz_style.GRID_LINEWIDTH)
            ax.grid(True, axis="x", alpha=viz_style.GRID_ALPHA * 0.7, linewidth=viz_style.GRID_LINEWIDTH)
            continue

        for vid, seg_p in own_segments:
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
        if stack_other_dispensers:
            for other in range(n_dispensers):
                if other == dispenser:
                    continue
                other_p, _other_req, _other_brk, _other_segs = arrays_by_dispenser[other]
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

        if show_bms_request and t_axis.size:
            ax.plot(
                t_axis + bar_width / 2,
                own_req,
                ls="--",
                lw=viz_style.LINEWIDTH_SIM,
                color=viz_style.series_color("sim_bms"),
                marker="o",
                markersize=2.5,
                zorder=4,
            )

        if show_modules and t_axis.size:
            any_modules = True
            ax.step(
                t_axis,
                own_brk * om.station.p_module,
                where="post",
                ls="--",
                lw=viz_style.LINEWIDTH_SIM * 0.9,
                color=viz_style.series_color("modules"),
                zorder=2,
            )

        ax.set_ylabel(f"Dispenser {dispenser}\nP (kW)")
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
        if stack_other_dispensers:
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
