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

Red markers are DES-recorded ``charge_trace`` samples (static).

Example (after you have a finished env):

    from visualization.ev_curves import plot_ev_theory_vs_sim
    import matplotlib.pyplot as plt

    figs = plot_ev_theory_vs_sim(env, ev_ids=[0, 3, 10], charts=['P-S', 'T-S', 'P-T'])
    plt.show()

Or run a demo episode from the CLI:

    python -m visualization.ev_curves --evs 0,1,2 --charts P-S,T-S,P-T --seed 42
"""

from __future__ import annotations

import argparse
from math import exp, log

import matplotlib.pyplot as plt
import numpy as np

from env.charging_env import ChargingStationEnv
from models.ev import EV
from visualization.pile_power import densify_charge_trace, run_fifo_episode

VALID_CHARTS = ("P-S", "T-S", "P-T")
_HR2MIN = 60

# ---------------------------------------------------------------------------
# Theory (isolated EV, always drawing full BMS request)
# ---------------------------------------------------------------------------


def theoretical_p_of_s(ev: EV, s: np.ndarray | float) -> np.ndarray | float:
    """BMS power acceptance P(s) [kW]."""
    s_arr = np.asarray(s, dtype=float)
    p = np.where(
        s_arr <= ev.s_th,
        ev.p_req_max,
        ev.p_req_max * (1.0 - s_arr) / (1.0 - ev.s_th),
    )
    return float(p) if np.ndim(s) == 0 else p


def theoretical_local_time_to_soc(ev: EV, s: float) -> float:
    """
    Minutes (sim clock) to go from s_i to SoC s under full BMS power.
    Returns 0 if s <= s_i; raises if s > 1.
    """
    if s <= ev.s_i + 1e-15:
        return 0.0
    if s >= 1.0 - 1e-15:
        s = 1.0 - 1e-12

    c_b, p_m, s_th, s_i = ev.c_b, ev.p_req_max, ev.s_th, ev.s_i

    scale = c_b / p_m

    if s <= s_th:
        # both in constant region (requires s_i < s <= s_th)
        return scale * (s - s_i)
    if s_i >= s_th:
        # both in taper
        return scale * (-(1.0 - s_th) * log((1.0 - s) / (1.0 - s_i)))
    # crosses threshold
    return scale * ((s_th - s_i) - (1.0 - s_th) * log((1.0 - s) / (1.0 - s_th)))


def theoretical_p_of_local_t(ev: EV, t_local: np.ndarray | float) -> np.ndarray | float:
    """Unconstrained P(t) for t measured from plug-in [sim minutes]."""
    t_arr = np.asarray(t_local, dtype=float)
    t_th = theoretical_local_time_to_soc(ev, min(ev.s_th, ev.s_f))
    # If already above threshold at plug-in, t_th = 0 and decay from s_i.
    if ev.s_i >= ev.s_th:
        t_th = 0.0
        k = ev.p_req_max / (ev.c_b * (1.0 - ev.s_th))
        # P(t) = P(s_i) * exp(-k t) with P(s_i) = p_req_max * (1-s_i)/(1-s_th)
        p0 = theoretical_p_of_s(ev, ev.s_i)
        p = p0 * np.exp(-k * t_arr)
    else:
        k = ev.p_req_max / (ev.c_b * (1.0 - ev.s_th))
        p = np.where(
            t_arr <= t_th,
            ev.p_req_max,
            ev.p_req_max * np.exp(-k * (t_arr - t_th)),
        )
    return float(p) if np.ndim(t_local) == 0 else p


def theoretical_soc_of_local_t(ev: EV, t_local: float) -> float:
    """SoC after t_local minutes of unconstrained charging from s_i."""
    if t_local <= 0:
        return ev.s_i
    if ev.s_i >= ev.s_th:
        k = ev.p_req_max / (ev.c_b * (1.0 - ev.s_th))
        return 1.0 - (1.0 - ev.s_i) * exp(-k * t_local)

    t_th = (ev.c_b / ev.p_req_max) * (ev.s_th - ev.s_i)
    if t_local <= t_th:
        return ev.s_i + (ev.p_req_max / ev.c_b) * t_local
    k = ev.p_req_max / (ev.c_b * (1.0 - ev.s_th))
    return 1.0 - (1.0 - ev.s_th) * exp(-k * (t_local - t_th))


# ---------------------------------------------------------------------------
# Env lookup
# ---------------------------------------------------------------------------


def _all_known_evs(env: ChargingStationEnv) -> dict[int, EV]:
    """Map EV id -> object from arrivals, finished, and still-plugged EVs."""
    by_id: dict[int, EV] = {}
    metrics = env.engine.metrics
    for ev in metrics.arrived_evs:
        by_id[ev.id] = ev
    for ev in metrics.finished_evs:
        by_id[ev.id] = ev
    for pile in env.engine.station.piles:
        for ev in pile.evs:
            by_id[ev.id] = ev
    return by_id


def get_evs_from_env(env: ChargingStationEnv, ev_ids: list[int]) -> list[EV]:
    """
    Resolve EV objects by id. Prints a clear error and raises if any id is missing.
    """
    known = _all_known_evs(env)
    if not known:
        raise ValueError("No EVs found in env (did the episode run?).")

    ids_sorted = sorted(known.keys())
    lo, hi = ids_sorted[0], ids_sorted[-1]
    missing = [i for i in ev_ids if i not in known]
    if missing:
        print(
            f"Error: EV id(s) {missing} not in env. "
            f"Valid EV numbers must be from {lo} to {hi} "
            f"(known ids: {ids_sorted[:20]}{'...' if len(ids_sorted) > 20 else ''})."
        )
        raise KeyError(f"Unknown EV ids: {missing}")

    return [known[i] for i in ev_ids]


def _ev_location_label(ev: EV) -> str:
    pile = ev.pile_tracker
    pile_id = pile.id if pile is not None else "?"
    nozzle = ev.nozzle_id_tracker if ev.nozzle_id_tracker is not None else "?"
    return f"pile {pile_id}, nozzle {nozzle}"


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def _unique_charge_samples(
    ev: EV,
) -> list[tuple[float, float, float, float, float]]:
    """Return charge_trace samples with duplicate times collapsed (keep last)."""
    cleaned: list[tuple[float, float, float, float, float]] = []
    for sample in ev.charge_trace:
        if cleaned and abs(cleaned[-1][0] - sample[0]) < 1e-12:
            cleaned[-1] = sample
        else:
            cleaned.append(sample)
    return cleaned


def _scatter_recorded(ax: plt.Axes, xs, ys) -> None:
    ax.scatter(
        xs,
        ys,
        s=36,
        c="C3",
        zorder=5,
        edgecolors="k",
        linewidths=0.6,
        label="Recorded samples",
    )


def _plot_one_ev(ev: EV, charts: list[str]) -> plt.Figure:
    charts = [c.strip().upper().replace("_", "-") for c in charts]
    for c in charts:
        if c not in VALID_CHARTS:
            raise ValueError(f"Unknown chart '{c}'. Choose from {VALID_CHARTS}.")

    t_sim, s_sim, preq_sim, pact_sim = densify_charge_trace(ev, dt=0.1)
    samples = _unique_charge_samples(ev)
    t0 = ev.service_start_time if ev.service_start_time is not None else ev.arrival_time
    loc = _ev_location_label(ev)

    n = len(charts)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2), squeeze=False)
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
            ax.plot(s_grid, p_theory_s, "k--", lw=2.0, label="Theory BMS P(s)")
            if s_sim.size:
                ax.plot(
                    s_sim, preq_sim, "-", lw=1.8, color="C0", label="Sim BMS request"
                )
                ax.plot(s_sim, pact_sim, "-", lw=2.0, color="C1", label="Sim actual P")
            if samples:
                _scatter_recorded(ax, s_rec, pa_rec)
            ax.set_xlabel("SoC")
            ax.set_ylabel("Power (kW)")
            ax.set_title("P-S")

        elif chart == "T-S":
            ax.plot(
                s_grid, t_global_theory, "k--", lw=2.0, label="Theory t(s) [global]"
            )
            if s_sim.size:
                ax.plot(
                    s_sim, t_sim, "-", lw=2.0, color="C1", label="Sim t(s) [global]"
                )
            if samples:
                _scatter_recorded(ax, s_rec, t_rec)
            ax.set_xlabel("SoC")
            ax.set_ylabel("Simulation time (min)")
            ax.set_title("T-S")

        elif chart == "P-T":
            ax.plot(
                t0 + t_local_fine,
                p_theory_t,
                "k--",
                lw=2.0,
                label="Theory P(t) unconstrained",
            )
            if t_sim.size:
                ax.plot(
                    t_sim, preq_sim, "-", lw=1.8, color="C0", label="Sim BMS request"
                )
                ax.plot(t_sim, pact_sim, "-", lw=2.0, color="C1", label="Sim actual P")
            if samples:
                _scatter_recorded(ax, t_rec, pa_rec)
            ax.set_xlabel("Simulation time (min)")
            ax.set_ylabel("Power (kW)")
            ax.set_title("P-T")

        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle(
        f"EV {ev.id}  |  {loc}  |  "
        f"C_b={ev.c_b:.0f} kWh, s_i={ev.s_i:.2f}, s_f={ev.s_f:.2f}, s_th={ev.s_th:.2f}",
        fontsize=11,
    )
    fig.tight_layout()
    return fig


def plot_ev_theory_vs_sim(
    env: ChargingStationEnv,
    ev_ids: list[int],
    charts: list[str] | None = None,
) -> list[plt.Figure]:
    """
    For each EV id, plot theoretical curves with simulated overlays.
    Red markers are DES-recorded ``charge_trace`` samples.

    Parameters
    ----------
    env :
        Finished (or mid-run) ChargingStationEnv with charge_trace filled in.
    ev_ids :
        EV numbers to plot.
    charts :
        Subset of 'P-S', 'T-S', 'P-T'. Default: all three.

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
            f"Plotting EV {ev.id} ({_ev_location_label(ev)}), "
            f"charts={charts}, trace_len={len(ev.charge_trace)}"
        )
        if not ev.charge_trace:
            print(
                f"  Warning: EV {ev.id} has empty charge_trace "
                f"(never started service?). Theory-only plot."
            )
        figs.append(_plot_one_ev(ev, charts))
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
