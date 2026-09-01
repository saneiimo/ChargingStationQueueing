"""
Plot Power vs time for every connector on one charging pile.

For a chosen pile, builds one figure with ``n_connectors`` vertically stacked axes.
Each axis shows every EV that used that connector:
  - dashed: unconstrained theory P(t) (same as ``plot_ev_theory_vs_sim`` P-T)
  - solid: actual drawn power densified from DES ``charge_trace``
  - optional horizontal segments: piecewise-constant ``p_act`` plateaus between
    ``charge_trace`` samples (charge redistributions)
  - optional vertical markers: sole module-allotment change times
    (shared across connectors; plug-in / departure times excluded)
  - optional labels at each trace sample: P, t, s, n_modules

Power-axis ticks and horizontal grid lines sit at multiples of one module power.

Styling comes from ``visualization.style`` / ``config`` ``VIZ_*`` constants.
EV colors are assigned by plug-in time on the *pile* (not restarted per connector).
Plug-in / departure markers are drawn as shared vertical lines across all
connector axes so events line up visually.

Example (after a FIFO run)::

    from visualization.pile_power import plot_pile_connector_power, run_fifo_episode
    import matplotlib.pyplot as plt

    env = run_fifo_episode(seed=42)
    fig = plot_pile_connector_power(env, pile_id=0, t_start=0, t_end=200)
    plt.show()

Terminal::

    python -m visualization.pile_power --pile 0 --t-start 0 --t-end 240 --seed 42
"""

from __future__ import annotations

import argparse
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from env.charging_env import ChargingStationEnv
from models.ev import EV
from visualization import style as viz_style
from visualization.charge_trace import (
    charge_change_event_times,
    densify_charge_trace,
    parsed_trace_samples,
    trace_label_text,
    trace_power_plateaus,
)
from visualization.charging_theory import theoretical_power_vs_global_time
from visualization.demo import run_fifo_episode
from visualization.ev_lookup import ev_window_times, evs_for_pile
from visualization.plot_common import color_by_plug_time

DEFAULT_LABEL_THEORY = "Ideal (full BMS)"
DEFAULT_LABEL_SIM_BMS = "BMS limit"
DEFAULT_LABEL_ACTUAL = "Actual power"
DEFAULT_LABEL_CHARGE_STEPS = "Allotted power"
DEFAULT_LABEL_CHARGE_CHANGE = "Charge-change event"
DEFAULT_TITLE_TEMPLATE = (
    "Pile {pile_id}: theory vs actual power by connector "
    "(t in [{t_lo:.1f}, {t_hi:.1f}])"
)


def _module_power_ticks(p_module: float, p_max: float) -> np.ndarray:
    """Tick locations at 0, p_module, 2*p_module, ... covering ``p_max``."""
    if p_module <= 0:
        return np.array([0.0])
    n = int(np.ceil(max(p_max, 0.0) / p_module - 1e-12))
    return np.arange(0, n + 1, dtype=float) * p_module

# Re-export for backward compatibility (tests / README import from here).
__all__ = [
    "plot_pile_connector_power",
    "run_fifo_episode",
    "densify_charge_trace",
]


def plot_pile_connector_power(
    env: ChargingStationEnv,
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
    One figure for ``pile_id``: vertical subplots = connectors; x = time; y = power.

    Parameters
    ----------
    env :
        Finished (or mid-run) ChargingStationEnv with charge traces filled in.
    pile_id :
        Which pile to plot (0 .. n_piles-1).
    t_start, t_end :
        Optional zoom window on the time axis (minutes). Defaults to full episode.
    densify_dt :
        Step (minutes) used when filling simulated actual-power curves.
    figsize :
        Optional ``(width, height)``. Defaults from ``visualization.style``.
    show_theory :
        Draw unconstrained theory P(t) (dashed), matching ``plot_ev_theory_vs_sim``.
    show_sim_bms :
        Optionally overlay densified simulated BMS request (dotted).
    show_actual :
        Draw densified simulated actual power (solid).
    show_charge_change_lines :
        Draw horizontal segments at ``p_act`` between consecutive ``charge_trace``
        samples (piecewise-constant power plateaus after each redistribution).
    show_charge_change_vlines :
        Draw vertical markers at module-allotment change times on **all** connector
        axes, with a time label on the top axis. Times that coincide with any
        EV plug-in or departure on this pile are omitted (sole charge changes).
    show_trace_labels :
        Annotate each ``charge_trace`` sample with P, t, s, and module count.
    show_legend :
        If True, show a single shared legend (not per EV).
    label_theory, label_sim_bms, label_actual, label_charge_steps,
    label_charge_change :
        Optional legend text overrides.
    title :
        Optional figure title. Default uses ``DEFAULT_TITLE_TEMPLATE``.
    show_event_lines :
        If True, draw vertical lines at every EV plug-in / departure that fall
        in the window, on **all** connector axes (aligned across the pile).
    event_line_alpha :
        Opacity of those shared event lines in ``[0, 1]``.
    charge_step_alpha :
        Opacity of charge-change horizontal segments in ``[0, 1]``.
    charge_change_vline_alpha :
        Opacity of shared charge-change vertical markers in ``[0, 1]``.
    show_ev_ids :
        If True, annotate each EV's id above its actual-power curve.
    theory_n_points :
        Samples along the unconstrained theory P(t) curve.

    Notes
    -----
    Power-axis (y) ticks and grid lines are placed at multiples of one module
    power (``p_module``), e.g. 0, 25, 50, ... kW.

    Returns
    -------
    matplotlib.figure.Figure
        In Jupyter, assign to a variable (``fig = plot_pile_connector_power(...)``)
        or end the cell with a semicolon. Otherwise the inline backend may show
        the figure twice (once from pyplot, once from the returned object).
    """
    viz_style.apply_visualization_style()

    station = env.engine.station
    if pile_id < 0 or pile_id >= station.n_piles:
        raise ValueError(f"pile_id={pile_id} out of range [0, {station.n_piles - 1}]")

    pile = station.piles[pile_id]
    n_connectors = pile.n_connectors
    t_lo = 0.0 if t_start is None else float(t_start)
    t_hi = float(env.engine.current_time if t_end is None else t_end)
    if t_hi <= t_lo:
        raise ValueError("t_end must be greater than t_start")

    legend_theory = label_theory if label_theory is not None else DEFAULT_LABEL_THEORY
    legend_bms = label_sim_bms if label_sim_bms is not None else DEFAULT_LABEL_SIM_BMS
    legend_act = label_actual if label_actual is not None else DEFAULT_LABEL_ACTUAL
    legend_steps = (
        label_charge_steps
        if label_charge_steps is not None
        else DEFAULT_LABEL_CHARGE_STEPS
    )
    legend_cc = (
        label_charge_change
        if label_charge_change is not None
        else DEFAULT_LABEL_CHARGE_CHANGE
    )
    fig_title = (
        title
        if title is not None
        else DEFAULT_TITLE_TEMPLATE.format(pile_id=pile_id, t_lo=t_lo, t_hi=t_hi)
    )

    episode_t = float(env.engine.current_time)
    by_connector: dict[int, list[EV]] = defaultdict(list)
    pile_evs_in_window: list[EV] = []

    for ev in evs_for_pile(env, pile_id):
        nid = ev.connector_id_tracker
        if nid is None:
            continue
        t0, t1 = ev_window_times(ev, episode_t)
        if t1 < t_lo or t0 > t_hi:
            continue
        by_connector[nid].append(ev)
        pile_evs_in_window.append(ev)

    color_of = color_by_plug_time(pile_evs_in_window)

    event_times: list[tuple[float, str]] = []
    for ev in pile_evs_in_window:
        color = color_of[ev.id]
        t0, t1 = ev_window_times(ev, episode_t)
        if t_lo <= t0 <= t_hi:
            event_times.append((t0, color))
        if t_lo <= t1 <= t_hi:
            event_times.append((t1, color))

    # Unique module-change times across all EVs on this pile (shared vlines).
    # Exclude plug-in / departure instants so only *sole* charge changes remain.
    charge_change_times: list[float] = []
    if show_charge_change_vlines:
        occupancy_times = [t for t, _ in event_times]
        seen_cc: set[float] = set()
        for ev in pile_evs_in_window:
            for t_cc in charge_change_event_times(
                ev,
                pile.p_module,
                t_lo=t_lo,
                t_hi=t_hi,
                exclude_times=occupancy_times,
            ):
                key = round(t_cc, 6)
                if key in seen_cc:
                    continue
                seen_cc.add(key)
                charge_change_times.append(t_cc)
        charge_change_times.sort()

    if figsize is None:
        figsize = (
            viz_style.FIGSIZE_WIDE[0],
            max(viz_style.FIGSIZE_PANEL_HEIGHT * n_connectors * 0.75, 4.0),
        )

    fig, axes = plt.subplots(
        n_connectors,
        1,
        sharex=True,
        figsize=figsize,
        squeeze=False,
    )
    axes = axes[:, 0]

    any_curve = False
    any_charge_steps = False
    p_module = pile.p_module

    for connector in range(n_connectors):
        ax = axes[connector]
        evs = sorted(
            by_connector.get(connector, []),
            key=lambda e: (e.service_start_time or 0.0, e.id),
        )

        if not evs:
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
            yticks = _module_power_ticks(p_module, pile.power_supp)
            ax.set_ylim(0.0, float(yticks[-1]) if yticks.size else pile.power_supp)
            ax.set_yticks(yticks)
            viz_style.style_axes(ax, title=None)
            ax.grid(
                True,
                axis="y",
                alpha=viz_style.GRID_ALPHA,
                linewidth=viz_style.GRID_LINEWIDTH,
            )
            ax.grid(
                True,
                axis="x",
                alpha=viz_style.GRID_ALPHA * 0.7,
                linewidth=viz_style.GRID_LINEWIDTH,
            )
            continue

        for ev in evs:
            color = color_of[ev.id]
            t_act = np.array([])
            p_req = np.array([])
            p_act = np.array([])
            _, t1 = ev_window_times(ev, episode_t)

            if show_actual or show_sim_bms:
                t_act, _s, p_req, p_act = densify_charge_trace(ev, dt=densify_dt)

            if show_charge_change_lines:
                plateaus = trace_power_plateaus(ev, p_module, t_end=t1)
                for t_seg_lo, t_seg_hi, p_seg, _sample in plateaus:
                    seg_lo = max(t_seg_lo, t_lo)
                    seg_hi = min(t_seg_hi, t_hi)
                    if seg_hi <= seg_lo:
                        continue
                    any_charge_steps = True
                    ax.hlines(
                        y=p_seg,
                        xmin=seg_lo,
                        xmax=seg_hi,
                        colors=color,
                        linewidth=viz_style.LINEWIDTH_SIM * 1.15,
                        alpha=float(np.clip(charge_step_alpha, 0.0, 1.0)),
                        zorder=3,
                    )

            if show_trace_labels:
                for sample in parsed_trace_samples(ev, p_module):
                    if sample.t < t_lo or sample.t > t_hi:
                        continue
                    ax.scatter(
                        [sample.t],
                        [sample.p_act],
                        s=28,
                        color=color,
                        edgecolors=viz_style.COLORS["text"],
                        linewidths=0.45,
                        zorder=6,
                    )
                    ax.annotate(
                        trace_label_text(sample),
                        (sample.t, sample.p_act),
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
                t_th, p_th = theoretical_power_vs_global_time(
                    ev, n_points=theory_n_points
                )
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

            if show_sim_bms and t_act.size:
                mask = (t_act >= t_lo) & (t_act <= t_hi)
                if np.any(mask):
                    any_curve = True
                    ax.plot(
                        t_act[mask],
                        p_req[mask],
                        color=color,
                        linestyle=":",
                        linewidth=viz_style.LINEWIDTH_SIM * 0.85,
                        alpha=0.85,
                    )

            if show_actual and t_act.size:
                mask = (t_act >= t_lo) & (t_act <= t_hi)
                if not np.any(mask):
                    continue

                any_curve = True
                ax.plot(
                    t_act[mask],
                    p_act[mask],
                    color=color,
                    linestyle="-",
                    linewidth=viz_style.LINEWIDTH_SIM,
                    alpha=0.95,
                )

                if show_ev_ids:
                    tm = t_act[mask]
                    pm = p_act[mask]
                    mid = len(tm) // 2
                    ax.text(
                        tm[mid],
                        pm[mid],
                        f"EV {ev.id}",
                        ha="center",
                        va="bottom",
                        **viz_style.annotation_kwargs(
                            fontsize=viz_style.FONT_SIZE_ANNOTATION,
                            color=color,
                        ),
                    )

        ax.set_ylabel(f"Connector {connector}\nP (kW)")
        ax.set_xlim(t_lo, t_hi)
        # Power ticks / grid at multiples of one module (kW).
        y_top = max(pile.power_supp, ax.get_ylim()[1])
        yticks = _module_power_ticks(p_module, y_top)
        ax.set_ylim(0.0, float(yticks[-1]) if yticks.size else pile.power_supp)
        ax.set_yticks(yticks)
        viz_style.style_axes(ax, title=None)
        ax.grid(
            True,
            axis="y",
            alpha=viz_style.GRID_ALPHA,
            linewidth=viz_style.GRID_LINEWIDTH,
        )
        ax.grid(
            True,
            axis="x",
            alpha=viz_style.GRID_ALPHA * 0.7,
            linewidth=viz_style.GRID_LINEWIDTH,
        )

    if show_event_lines and event_times:
        alpha = float(np.clip(event_line_alpha, 0.0, 1.0))
        for ax in axes:
            for t_ev, color in event_times:
                ax.axvline(
                    t_ev,
                    color=color,
                    alpha=alpha,
                    linewidth=1.0,
                    zorder=0,
                )

    if show_charge_change_vlines and charge_change_times:
        cc_alpha = float(np.clip(charge_change_vline_alpha, 0.0, 1.0))
        cc_color = viz_style.COLORS["alert"]
        for ax in axes:
            for t_cc in charge_change_times:
                ax.axvline(
                    t_cc,
                    color=cc_color,
                    alpha=cc_alpha,
                    linewidth=1.15,
                    linestyle="-.",
                    zorder=1,
                )
        # Time labels once on the top panel so they stay readable.
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
                    fontsize=max(viz_style.FONT_SIZE_ANNOTATION - 1, 6),
                    color=cc_color,
                ),
                clip_on=False,
                zorder=8,
            )

    axes[-1].set_xlabel("Time (min)")
    axes[-1].xaxis.label.set_fontproperties(
        viz_style.body_fontproperties(size=viz_style.FONT_SIZE_AXIS_TITLE)
    )

    viz_style.style_figure_title(fig, fig_title)

    if show_legend and (any_curve or any_charge_steps or charge_change_times):
        handles: list[Line2D] = []
        if show_theory:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle="--",
                    linewidth=viz_style.LINEWIDTH_THEORY * 0.75,
                    label=legend_theory,
                )
            )
        if show_sim_bms:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle=":",
                    linewidth=viz_style.LINEWIDTH_SIM * 0.85,
                    label=legend_bms,
                )
            )
        if show_actual:
            handles.append(
                Line2D(
                    [0],
                    [0],
                    color=viz_style.COLORS["neutral"],
                    linestyle="-",
                    linewidth=viz_style.LINEWIDTH_SIM,
                    label=legend_act,
                )
            )
        if show_charge_change_lines and any_charge_steps:
            handles.append(
                Line2D(
                    [0],
                    [0],
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
                    [0],
                    [0],
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
                text.set_fontproperties(
                    viz_style.body_fontproperties(size=viz_style.FONT_SIZE_LEGEND)
                )

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    # Drop from pyplot's active figure list so Jupyter inline does not also
    # render this figure at cell end (return value still displays once).
    plt.close(fig)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot theory vs actual charging power over time for one pile."
    )
    parser.add_argument("--pile", type=int, required=True, help="Pile index to plot")
    parser.add_argument("--t-start", type=float, default=None, help="Zoom start (min)")
    parser.add_argument("--t-end", type=float, default=None, help="Zoom end (min)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-piles", type=int, default=4)
    parser.add_argument("--n-connectors", type=int, default=2)
    parser.add_argument("--n-modules", type=int, default=5)
    parser.add_argument("--p-module", type=float, default=25.0)
    parser.add_argument(
        "--mean-interarrival",
        type=float,
        default=5.0,
        help="Mean inter-arrival time in minutes (arrival rate λ = 1 / this)",
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the figure (e.g. pile0.png)",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display the figure interactively",
    )
    parser.add_argument(
        "--no-legend",
        action="store_true",
        help="Hide the legend",
    )
    parser.add_argument(
        "--no-ev-ids",
        action="store_true",
        help="Hide EV id annotations on curves",
    )
    parser.add_argument(
        "--show-sim-bms",
        action="store_true",
        help="Also overlay densified simulated BMS request",
    )
    parser.add_argument(
        "--no-theory",
        action="store_true",
        help="Hide unconstrained theory P(t)",
    )
    parser.add_argument(
        "--show-charge-steps",
        action="store_true",
        help="Draw horizontal charge-change power plateaus from charge_trace",
    )
    parser.add_argument(
        "--show-charge-vlines",
        action="store_true",
        help="Draw shared vertical markers at module-allotment change times",
    )
    parser.add_argument(
        "--show-trace-labels",
        action="store_true",
        help="Label each charge_trace sample with P, t, s, n_modules",
    )
    parser.add_argument(
        "--event-line-alpha",
        type=float,
        default=0.25,
        help="Opacity of shared plug-in/departure vertical lines",
    )
    args = parser.parse_args(argv)

    print(f"Running FIFO episode (seed={args.seed}) then plotting pile {args.pile}...")
    env = run_fifo_episode(
        n_piles=args.n_piles,
        n_connectors=args.n_connectors,
        n_modules=args.n_modules,
        p_module=args.p_module,
        mean_interarrival=args.mean_interarrival,
        seed=args.seed,
    )
    print(
        f"Done: t={env.engine.current_time:.1f}, "
        f"finished={len(env.engine.metrics.finished_evs)}"
    )

    fig = plot_pile_connector_power(
        env,
        pile_id=args.pile,
        t_start=args.t_start,
        t_end=args.t_end,
        show_theory=not args.no_theory,
        show_sim_bms=args.show_sim_bms,
        show_charge_change_lines=args.show_charge_steps,
        show_charge_change_vlines=args.show_charge_vlines,
        show_trace_labels=args.show_trace_labels,
        show_legend=not args.no_legend,
        show_ev_ids=not args.no_ev_ids,
        event_line_alpha=args.event_line_alpha,
    )
    if args.save:
        fig.savefig(args.save, dpi=150, bbox_inches="tight")
        print(f"Saved {args.save}")
    if args.show or not args.save:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
