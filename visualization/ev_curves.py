"""
Theoretical vs simulated charging curves for selected EVs.

Theory (from the Charging Curve note; matches models/ev.py):
NOTE THAT THERE MIGHT BE A UNIT MISMATCH IF C_b is kWh and
T IS IN min.
  P(s) = P_m                         if s <= s_th
       = P_m (1-s)/(1-s_th)          if s >  s_th

  Charging time from s_i to s under full BMS power:
    T = (C_b / P_m) * {
          (s - s_i)                                           s_i < s <= s_th
          -(1-s_th) ln((1-s)/(1-s_i))                        s_th <= s_i < s
          (s_th - s_i) - (1-s_th) ln((1-s)/(1-s_th))         s_i < s_th < s
        }

  Power vs local time (unconstrained):
    P(t) = P_m                                           t <= t_th
         = P_m exp( -P_m / (C_b (1-s_th)) * (t - t_th) ) t >  t_th
    with t_th = (C_b / P_m) * max(s_th - s_i, 0).

These formulas are consistent with dE/dt = P and E = C_b * SoC. The simulator
uses the same algebraic time scaling as models/ev.py (clock labeled in minutes).

Chart codes (pass a subset):
  'P-S'  Power vs SoC
  'T-S'  global simulation time vs SoC
  'P-T'  Power vs global simulation time

Toggle series with show_theory / show_sim_bms / show_sim_actual /
show_recorded_samples. Optional label_* kwargs override legend text.

Styling (fonts, sizes, palette) lives in visualization.style.

Example (after you have a finished env):

    from visualization.ev_curves import plot_ev_theory_vs_sim
    import matplotlib.pyplot as plt

    figs = plot_ev_theory_vs_sim(
        env,
        ev_ids=[0, 3, 10],
        charts=['P-S', 'T-S', 'P-T'],
        show_theory=True,
        show_sim_bms=True,
        show_sim_actual=True,
        show_recorded_samples=True,
    )
    plt.show()

Or run a demo episode from the CLI:

    python -m visualization.ev_curves --evs 0,1,2 --charts P-S,T-S,P-T --seed 42
"""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from config import HR2MIN
from env.charging_env import ChargingStationEnv
from models.ev import EV
from visualization import style as viz_style
from visualization.charge_trace import densify_charge_trace, unique_charge_samples
from visualization.charging_theory import (
    theoretical_local_time_to_soc,
    theoretical_p_of_local_t,
    theoretical_p_of_s,
)
from visualization.demo import run_fifo_episode
from visualization.ev_lookup import ev_location_label, get_evs_from_env
from visualization.plot_common import scatter_recorded

VALID_CHARTS = ("P-S", "T-S", "P-T")

# Default legend labels (override via plot_ev_theory_vs_sim kwargs).
DEFAULT_LABEL_THEORY_PS = "Theory BMS P(s)"
DEFAULT_LABEL_THEORY_TS = "Theory t(s) [global]"
DEFAULT_LABEL_THEORY_PT = "Theory P(t) unconstrained"
DEFAULT_LABEL_SIM_BMS = "Sim BMS request"
DEFAULT_LABEL_SIM_ACTUAL = "Sim actual P"
DEFAULT_LABEL_SIM_TIME = "Sim t(s) [global]"
DEFAULT_LABEL_RECORDED = "Recorded samples"


def _resolve_labels(
    label_theory: str | None,
    label_sim_bms: str | None,
    label_sim_actual: str | None,
    label_recorded: str | None,
) -> dict[str, str]:
    """Fill legend strings from defaults when the caller passes None."""
    return {
        "theory_ps": label_theory if label_theory is not None else DEFAULT_LABEL_THEORY_PS,
        "theory_ts": label_theory if label_theory is not None else DEFAULT_LABEL_THEORY_TS,
        "theory_pt": label_theory if label_theory is not None else DEFAULT_LABEL_THEORY_PT,
        "sim_bms": label_sim_bms if label_sim_bms is not None else DEFAULT_LABEL_SIM_BMS,
        "sim_actual": (
            label_sim_actual if label_sim_actual is not None else DEFAULT_LABEL_SIM_ACTUAL
        ),
        "sim_time": (
            label_sim_actual if label_sim_actual is not None else DEFAULT_LABEL_SIM_TIME
        ),
        "recorded": (
            label_recorded if label_recorded is not None else DEFAULT_LABEL_RECORDED
        ),
    }


def _plot_one_ev(
    ev: EV,
    charts: list[str],
    *,
    show_theory: bool = True,
    show_sim_bms: bool = True,
    show_sim_actual: bool = True,
    show_recorded_samples: bool = True,
    label_theory: str | None = None,
    label_sim_bms: str | None = None,
    label_sim_actual: str | None = None,
    label_recorded: str | None = None,
) -> plt.Figure:
    charts = [c.strip().upper().replace("_", "-") for c in charts]
    for c in charts:
        if c not in VALID_CHARTS:
            raise ValueError(f"Unknown chart '{c}'. Choose from {VALID_CHARTS}.")

    labels = _resolve_labels(
        label_theory, label_sim_bms, label_sim_actual, label_recorded
    )
    c_theory = viz_style.series_color("theory")
    c_bms = viz_style.series_color("sim_bms")
    c_act = viz_style.series_color("sim_actual")

    t_sim, s_sim, preq_sim, pact_sim = densify_charge_trace(ev, dt=0.1)
    samples = unique_charge_samples(ev)
    t0 = ev.service_start_time if ev.service_start_time is not None else ev.arrival_time
    loc = ev_location_label(ev)

    n = len(charts)
    fig, axes = viz_style.new_figure(
        figsize=(
            viz_style.FIGSIZE_PANEL_WIDTH * n,
            viz_style.FIGSIZE_PANEL_HEIGHT,
        ),
        nrows=1,
        ncols=n,
        squeeze=False,
    )
    axes = axes[0]

    s_grid = np.linspace(ev.s_i, min(ev.s_f, 0.999), 200)
    p_theory_s = theoretical_p_of_s(ev, s_grid)
    t_global_theory = t0 + np.array(
        [theoretical_local_time_to_soc(ev, s) for s in s_grid]
    )
    t_end_local = theoretical_local_time_to_soc(ev, min(ev.s_f, 0.999))
    t_local_fine = np.linspace(0.0, max(t_end_local, 1e-6), 200)
    p_theory_t = theoretical_p_of_local_t(ev, t_local_fine)

    t_rec = np.array([t for t, _, _, _, _ in samples]) if samples else np.array([])
    s_rec = np.array([s for _, s, _, _, _ in samples]) if samples else np.array([])
    pa_rec = np.array([pa for _, _, _, pa, _ in samples]) if samples else np.array([])

    for ax, chart in zip(axes, charts):
        if chart == "P-S":
            if show_theory:
                ax.plot(
                    s_grid,
                    p_theory_s,
                    ls="--",
                    lw=viz_style.LINEWIDTH_THEORY,
                    color=c_theory,
                    label=labels["theory_ps"],
                )
            if s_sim.size:
                if show_sim_bms:
                    ax.plot(
                        s_sim,
                        preq_sim,
                        ls="-",
                        lw=viz_style.LINEWIDTH_SIM,
                        color=c_bms,
                        label=labels["sim_bms"],
                    )
                if show_sim_actual:
                    ax.plot(
                        s_sim,
                        pact_sim,
                        ls="-",
                        lw=viz_style.LINEWIDTH_SIM,
                        color=c_act,
                        label=labels["sim_actual"],
                    )
            if show_recorded_samples and samples:
                scatter_recorded(ax, s_rec, pa_rec, label=labels["recorded"])
            ax.set_xlabel("SoC")
            ax.set_ylabel("Power (kW)")
            viz_style.style_axes(ax, title="P-S")

        elif chart == "T-S":
            if show_theory:
                ax.plot(
                    s_grid,
                    t_global_theory,
                    ls="--",
                    lw=viz_style.LINEWIDTH_THEORY,
                    color=c_theory,
                    label=labels["theory_ts"],
                )
            if s_sim.size and show_sim_actual:
                ax.plot(
                    s_sim,
                    t_sim,
                    ls="-",
                    lw=viz_style.LINEWIDTH_SIM,
                    color=c_act,
                    label=labels["sim_time"],
                )
            if show_recorded_samples and samples:
                scatter_recorded(ax, s_rec, t_rec, label=labels["recorded"])
            ax.set_xlabel("SoC")
            ax.set_ylabel("Simulation time (min)")
            viz_style.style_axes(ax, title="T-S")

        elif chart == "P-T":
            if show_theory:
                ax.plot(
                    t0 + t_local_fine,
                    p_theory_t,
                    ls="--",
                    lw=viz_style.LINEWIDTH_THEORY,
                    color=c_theory,
                    label=labels["theory_pt"],
                )
            if t_sim.size:
                if show_sim_bms:
                    ax.plot(
                        t_sim,
                        preq_sim,
                        ls="-",
                        lw=viz_style.LINEWIDTH_SIM,
                        color=c_bms,
                        label=labels["sim_bms"],
                    )
                if show_sim_actual:
                    ax.plot(
                        t_sim,
                        pact_sim,
                        ls="-",
                        lw=viz_style.LINEWIDTH_SIM,
                        color=c_act,
                        label=labels["sim_actual"],
                    )
            if show_recorded_samples and samples:
                scatter_recorded(ax, t_rec, pa_rec, label=labels["recorded"])
            ax.set_xlabel("Simulation time (min)")
            ax.set_ylabel("Power (kW)")
            viz_style.style_axes(ax, title="P-T")

        ax.grid(True, alpha=viz_style.GRID_ALPHA, linewidth=viz_style.GRID_LINEWIDTH)
        if ax.get_legend_handles_labels()[0]:
            viz_style.style_legend(ax, loc="best")

    arr = ev.arrival_time
    srv = ev.service_start_time if ev.service_start_time is not None else float("nan")
    dpt = ev.departure_time if np.isfinite(ev.departure_time) else float("nan")
    viz_style.style_figure_title(
        fig,
        f"EV {ev.id}  |  {loc}  |  "
        f"C_b={ev.c_b / HR2MIN:.0f} kWh, s_i={ev.s_i:.2f}, s_f={ev.s_f:.2f}, "
        f"s_th={ev.s_th:.2f}, arr={arr:.2f}m, srv={srv:.2f}m, dpt={dpt:.2f}m",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


def plot_ev_theory_vs_sim(
    env: ChargingStationEnv,
    ev_ids: list[int],
    charts: list[str] | None = None,
    *,
    show_theory: bool = True,
    show_sim_bms: bool = True,
    show_sim_actual: bool = True,
    show_recorded_samples: bool = True,
    label_theory: str | None = None,
    label_sim_bms: str | None = None,
    label_sim_actual: str | None = None,
    label_recorded: str | None = None,
) -> list[plt.Figure]:
    """
    For each EV id, plot theoretical curves with simulated overlays.

    Series visibility is controlled by the ``show_*`` flags. Legend text can be
    overridden with ``label_*``; omitted labels fall back to the module defaults
    (chart-specific for theory / T-S sim time).

    Parameters
    ----------
    env :
        Finished (or mid-run) ChargingStationEnv with charge_trace filled in.
    ev_ids :
        EV numbers to plot.
    charts :
        Subset of 'P-S', 'T-S', 'P-T'. Default: all three.
    show_theory :
        Draw unconstrained theory curves (P(s), t(s), or P(t) by chart).
    show_sim_bms :
        Draw densified simulated BMS request (P-S and P-T only).
    show_sim_actual :
        Draw densified simulated actual power (P-S, P-T) or sim t(s) (T-S).
    show_recorded_samples :
        Scatter DES ``charge_trace`` samples.
    label_theory, label_sim_bms, label_sim_actual, label_recorded :
        Optional legend strings. If None, use the defaults defined in this
        module (``DEFAULT_LABEL_*``).

    Returns
    -------
    list of matplotlib Figures (one per EV).
    """
    if charts is None:
        charts = list(VALID_CHARTS)

    evs = get_evs_from_env(env, ev_ids)
    figs: list[plt.Figure] = []
    for ev in evs:
        print(
            f"Plotting EV {ev.id} ({ev_location_label(ev)}), "
            f"charts={charts}, trace_len={len(ev.charge_trace)}"
        )
        if not ev.charge_trace:
            print(
                f"  Warning: EV {ev.id} has empty charge_trace "
                f"(never started service?). Theory-only plot."
            )
        figs.append(
            _plot_one_ev(
                ev,
                charts,
                show_theory=show_theory,
                show_sim_bms=show_sim_bms,
                show_sim_actual=show_sim_actual,
                show_recorded_samples=show_recorded_samples,
                label_theory=label_theory,
                label_sim_bms=label_sim_bms,
                label_sim_actual=label_sim_actual,
                label_recorded=label_recorded,
            )
        )
    return figs


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Plot theoretical vs simulated P-S / T-S / P-T for selected EVs."
    )
    parser.add_argument(
        "--evs",
        type=str,
        required=True,
        help="Comma-separated EV ids, e.g. 0,1,5",
    )
    parser.add_argument(
        "--charts",
        type=str,
        default="P-S,T-S,P-T",
        help="Comma-separated subset of P-S,T-S,P-T",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-prefix", type=str, default=None)
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args(argv)

    ev_ids = [int(x) for x in args.evs.split(",") if x.strip() != ""]
    charts = [c.strip() for c in args.charts.split(",") if c.strip()]

    print(f"Running FIFO episode (seed={args.seed})...")
    env = run_fifo_episode(seed=args.seed)
    print(
        f"Done: finished={len(env.engine.metrics.finished_evs)}, "
        f"t={env.engine.current_time:.1f}"
    )

    figs = plot_ev_theory_vs_sim(env, ev_ids=ev_ids, charts=charts)
    for ev_id, fig in zip(ev_ids, figs):
        if args.save_prefix:
            path = f"{args.save_prefix}_ev{ev_id}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved {path}")
    if args.show or not args.save_prefix:
        plt.show()
    else:
        for fig in figs:
            plt.close(fig)


if __name__ == "__main__":
    main()
