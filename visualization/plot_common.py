"""Shared plotting helpers for EV / pile visualizations."""

from __future__ import annotations

import matplotlib.pyplot as plt

from models.ev import EV
from visualization import style as viz_style


def scatter_recorded(
    ax: plt.Axes,
    xs,
    ys,
    *,
    label: str,
) -> None:
    """Plot DES-recorded samples with the shared series style."""
    ax.scatter(
        xs,
        ys,
        s=viz_style.MARKER_SIZE_RECORDED,
        c=viz_style.series_color("recorded"),
        zorder=5,
        edgecolors=viz_style.series_color("recorded_edge"),
        linewidths=0.55,
        label=label,
    )


def color_by_plug_time(evs: list[EV]) -> dict[int, str]:
    """
    Map EV id -> palette color ordered by service_start_time on the pile.

    Colors are not restarted per nozzle.
    """
    ordered = sorted(evs, key=lambda e: (e.service_start_time or 0.0, e.id))
    colors = viz_style.palette()
    return {ev.id: colors[i % len(colors)] for i, ev in enumerate(ordered)}
