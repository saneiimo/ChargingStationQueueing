"""
Plots of what the search did. The schedule itself is plotted with
``offline_cl_opt``'s functions on ``PDSSolution.to_cl_model()``.

* ``plot_bound_history``: the certified bracket ``[lower bound, best
  schedule]`` over time, with the master's bound for reference.
* ``plot_slot_prices``: the rents the search's bound uses -- what holding a
  connector with ``q`` modules costs in each slot -- and the station term
  ``gamma[k]``. High rents mark the slots where connectors and modules are
  scarce in the master's relaxation.
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np

from visualization import style as viz_style

from .cost_to_go import round_prices
from .master import rent_and_gamma
from .solution import PDSSolution


def plot_bound_history(
    sol: PDSSolution,
    *,
    mean: bool = True,
    references: dict[str, float] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """
    Lower bound and best schedule over the solve (step lines), in mean
    sojourn per optimized vehicle (``mean=True``) or total sojourn (minutes).
    ``references``: extra horizontal lines, e.g. ``{"compact MILP": z}`` in
    the same units as the objective (total sojourn); they are scaled like the
    curves.
    """
    viz_style.apply_visualization_style()
    scale = 1.0 / sol.n_optimized if mean else 1.0
    h = [row for row in sol.bound_history if math.isfinite(row["upper_bound"]) or math.isfinite(row["lower_bound"])]
    t = [row["time_s"] for row in h]
    lb = [row["lower_bound"] * scale if math.isfinite(row["lower_bound"]) else np.nan for row in h]
    ub = [row["upper_bound"] * scale if math.isfinite(row["upper_bound"]) else np.nan for row in h]

    fig, ax = plt.subplots(figsize=viz_style.FIGSIZE_STANDARD)
    ax.step(t, ub, where="post", color=viz_style.color("secondary"), lw=2, label="best schedule (upper bound)")
    ax.step(t, lb, where="post", color=viz_style.color("primary"), lw=2, label="certified lower bound")
    ax.axhline(sol.master_bound * scale, color=viz_style.color("muted"), ls=":", lw=1.5,
               label=f"master bound ({sol.prices.kind})")
    for i, (name, value) in enumerate((references or {}).items()):
        ax.axhline(value * scale, color=viz_style.palette()[(i + 2) % len(viz_style.palette())], ls="--", lw=1.2,
                   label=name)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("mean sojourn (min)" if mean else "total sojourn (min)")
    ax.grid(True)
    viz_style.style_axes(ax, title=title or f"Price-directed search: {sol.status}")
    viz_style.style_legend(ax, loc="best")
    fig.tight_layout()
    plt.close(fig)
    return fig


def plot_slot_prices(sol: PDSSolution, *, title: str | None = None) -> plt.Figure:
    """
    Top: ``rent[k, q]`` (slots of total sojourn) as a heat map over slot ``k``
    and modules held ``q``. Bottom: the station term ``gamma[k]``, the most
    rent the whole station can pay in slot ``k``.
    """
    viz_style.apply_visualization_style()
    st = sol.inputs["station"]
    rent, gamma = rent_and_gamma(round_prices(sol.prices.mu), sol.prices.configs, st.n_piles)
    K = rent.shape[0]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=viz_style.FIGSIZE_WIDE, sharex=True, layout="constrained",
                                   gridspec_kw={"height_ratios": [2, 1]})
    im = ax1.imshow(rent.T, aspect="auto", origin="lower", cmap=viz_style.CMAP_SEQUENTIAL,
                    extent=(-0.5, K - 0.5, -0.5, rent.shape[1] - 0.5), interpolation="nearest")
    ax1.set_ylabel("modules held q")
    viz_style.style_axes(ax1, title=title or f"Slot prices ({sol.prices.kind} master)")
    ax2.bar(np.arange(K), gamma, width=0.9, color=viz_style.color("primary"))
    ax2.set_xlabel(f"slot k ({sol.delta:g} min)")
    ax2.set_ylabel("gamma[k]")
    ax2.grid(True, axis="y")
    viz_style.style_axes(ax2)
    fig.colorbar(im, ax=[ax1, ax2], label="rent[k, q] (slots)")  # on both panels, so their slot axes line up
    plt.close(fig)
    return fig
