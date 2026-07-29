"""
Shared matplotlib styling for charging-station visualizations.

All tunable constants (fonts, sizes, colors, cmaps, figure sizes) live in
``config`` under the ``VIZ_*`` prefix. This module registers local font files,
applies ``rcParams``, and exposes helpers used by plot scripts.

Font groups
-----------
* **Title family** (``VIZ_TITLE_FONT_FAMILY``) — figure and subplot titles.
* **Body family** (``VIZ_BODY_FONT_FAMILY``) — axis titles, ticks, legend,
  and in-plot annotations.

Call ``apply_visualization_style()`` once before creating figures (helpers do
this automatically). After editing ``config`` in an interactive session, call
``apply_visualization_style(force=True)``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

from config import (
    VIZ_BODY_FONT_FAMILY,
    VIZ_CMAP_DIVERGING,
    VIZ_CMAP_SEQUENTIAL,
    VIZ_COLORS,
    VIZ_FIGSIZE_PANEL_HEIGHT,
    VIZ_FIGSIZE_PANEL_WIDTH,
    VIZ_FIGSIZE_STANDARD,
    VIZ_FIGSIZE_TALL,
    VIZ_FIGSIZE_WIDE,
    VIZ_FONT_SIZE_ANNOTATION,
    VIZ_FONT_SIZE_AXIS_TITLE,
    VIZ_FONT_SIZE_LEGEND,
    VIZ_FONT_SIZE_SUBTITLE,
    VIZ_FONT_SIZE_TICK,
    VIZ_FONT_SIZE_TITLE,
    VIZ_FONTS_DIR,
    VIZ_GRID_ALPHA,
    VIZ_GRID_LINEWIDTH,
    VIZ_LEGEND_FRAMEON,
    VIZ_LINEWIDTH_SIM,
    VIZ_LINEWIDTH_THEORY,
    VIZ_MARKER_SIZE_RECORDED,
    VIZ_PALETTE,
    VIZ_SERIES,
    VIZ_SPINE_WIDTH,
    VIZ_TITLE_FONT_FAMILY,
    VIZ_TITLE_PAD,
)

# ---------------------------------------------------------------------------
# Re-exports so plot modules can write ``viz_style.COLORS``, etc.
# ---------------------------------------------------------------------------

TITLE_FONT_FAMILY = VIZ_TITLE_FONT_FAMILY
BODY_FONT_FAMILY = VIZ_BODY_FONT_FAMILY
FONTS_DIR = VIZ_FONTS_DIR

FONT_SIZE_TITLE = VIZ_FONT_SIZE_TITLE
FONT_SIZE_SUBTITLE = VIZ_FONT_SIZE_SUBTITLE
FONT_SIZE_AXIS_TITLE = VIZ_FONT_SIZE_AXIS_TITLE
FONT_SIZE_TICK = VIZ_FONT_SIZE_TICK
FONT_SIZE_LEGEND = VIZ_FONT_SIZE_LEGEND
FONT_SIZE_ANNOTATION = VIZ_FONT_SIZE_ANNOTATION

FIGSIZE_STANDARD = VIZ_FIGSIZE_STANDARD
FIGSIZE_WIDE = VIZ_FIGSIZE_WIDE
FIGSIZE_TALL = VIZ_FIGSIZE_TALL
FIGSIZE_PANEL_WIDTH = VIZ_FIGSIZE_PANEL_WIDTH
FIGSIZE_PANEL_HEIGHT = VIZ_FIGSIZE_PANEL_HEIGHT

LINEWIDTH_THEORY = VIZ_LINEWIDTH_THEORY
LINEWIDTH_SIM = VIZ_LINEWIDTH_SIM
MARKER_SIZE_RECORDED = VIZ_MARKER_SIZE_RECORDED

GRID_ALPHA = VIZ_GRID_ALPHA
GRID_LINEWIDTH = VIZ_GRID_LINEWIDTH
SPINE_WIDTH = VIZ_SPINE_WIDTH
TITLE_PAD = VIZ_TITLE_PAD
LEGEND_FRAMEON = VIZ_LEGEND_FRAMEON

COLORS = VIZ_COLORS
PALETTE = list(VIZ_PALETTE)
SERIES = VIZ_SERIES
CMAP_SEQUENTIAL = VIZ_CMAP_SEQUENTIAL
CMAP_DIVERGING = VIZ_CMAP_DIVERGING

_STYLE_APPLIED = False
_RESOLVED_TITLE_FAMILY = VIZ_TITLE_FONT_FAMILY
_RESOLVED_BODY_FAMILY = VIZ_BODY_FONT_FAMILY


def color(name: str) -> str:
    """Return a named role color from ``VIZ_COLORS``."""
    return COLORS[name]


def series_color(name: str) -> str:
    """Return a semantic series color from ``VIZ_SERIES`` (e.g. ``\"theory\"``)."""
    return SERIES[name]


def palette(n: int | None = None) -> list[str]:
    """Return the categorical palette, optionally truncated or cycled."""
    base = list(PALETTE)
    if n is None:
        return base
    if n <= 0:
        return []
    return [base[i % len(base)] for i in range(n)]


def register_project_fonts(fonts_dir: Path | None = None) -> list[str]:
    """Register TTF/OTF faces from ``VIZ_FONTS_DIR`` (or ``fonts_dir``).

    Returns:
        List of font family names successfully registered from that directory.
    """
    directory = Path(fonts_dir) if fonts_dir is not None else Path(FONTS_DIR)
    registered: list[str] = []
    if not directory.is_dir():
        return registered

    for pattern in ("*.ttf", "*.otf"):
        for font_path in sorted(directory.glob(pattern)):
            try:
                font_manager.fontManager.addfont(str(font_path))
                props = font_manager.FontProperties(fname=str(font_path))
                name = props.get_name()
                if name and name not in registered:
                    registered.append(name)
            except (OSError, RuntimeError, ValueError):
                continue
    return registered


def _resolve_family(preferred: str, fallback: str) -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    return preferred if preferred in available else fallback


def apply_visualization_style(*, force: bool = False) -> None:
    """Apply project-wide matplotlib ``rcParams`` (idempotent).

    Registers fonts under ``assets/fonts``, then sets title/body families and
    sizes from ``config``. Use ``force=True`` after editing config values in an
    interactive session.
    """
    global _STYLE_APPLIED, _RESOLVED_TITLE_FAMILY, _RESOLVED_BODY_FAMILY
    if _STYLE_APPLIED and not force:
        return

    register_project_fonts()
    title_family = _resolve_family(TITLE_FONT_FAMILY, "DejaVu Serif")
    body_family = _resolve_family(BODY_FONT_FAMILY, "DejaVu Sans")
    _RESOLVED_TITLE_FAMILY = title_family
    _RESOLVED_BODY_FAMILY = body_family

    if title_family != TITLE_FONT_FAMILY or body_family != BODY_FONT_FAMILY:
        print(
            "Warning: requested fonts not fully available after registration; "
            f"using title={title_family!r}, body={body_family!r}. "
            f"Expected files in {FONTS_DIR}."
        )

    mpl.rcParams.update(
        {
            "font.family": body_family,
            "font.size": FONT_SIZE_TICK,
            "axes.titlesize": FONT_SIZE_SUBTITLE,
            "axes.titleweight": "semibold",
            "axes.titlelocation": "left",
            "axes.titlepad": TITLE_PAD,
            "axes.labelsize": FONT_SIZE_AXIS_TITLE,
            "axes.labelcolor": COLORS["text"],
            "axes.edgecolor": COLORS["spine"],
            "axes.facecolor": COLORS["background"],
            "axes.linewidth": SPINE_WIDTH,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": FONT_SIZE_TICK,
            "ytick.labelsize": FONT_SIZE_TICK,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "legend.fontsize": FONT_SIZE_LEGEND,
            "legend.frameon": LEGEND_FRAMEON,
            "figure.facecolor": COLORS["figure"],
            "figure.titlesize": FONT_SIZE_TITLE,
            "figure.titleweight": "semibold",
            "text.color": COLORS["text"],
            "grid.color": COLORS["grid"],
            "grid.linewidth": GRID_LINEWIDTH,
            "grid.alpha": GRID_ALPHA,
            "axes.prop_cycle": mpl.cycler(color=list(PALETTE)),
            "savefig.facecolor": COLORS["figure"],
            "savefig.bbox": "tight",
            "lines.solid_capstyle": "round",
        }
    )
    mpl.rcParams["font.sans-serif"] = [body_family, "DejaVu Sans"]
    mpl.rcParams["font.serif"] = [title_family, "DejaVu Serif"]
    _STYLE_APPLIED = True


def title_fontproperties(**overrides):
    """``FontProperties`` for figure / subplot titles (title family)."""
    apply_visualization_style()
    props = {
        "family": _RESOLVED_TITLE_FAMILY,
        "size": FONT_SIZE_TITLE,
        "weight": "semibold",
    }
    props.update(overrides)
    return font_manager.FontProperties(**props)


def body_fontproperties(**overrides):
    """``FontProperties`` for axis labels, ticks, legend, and annotations."""
    apply_visualization_style()
    props = {
        "family": _RESOLVED_BODY_FAMILY,
        "size": FONT_SIZE_TICK,
    }
    props.update(overrides)
    return font_manager.FontProperties(**props)


def style_axes(ax: plt.Axes, *, title: str | None = None, subtitle: bool = True) -> None:
    """Apply shared spine, face, tick, and optional title styling to ``ax``."""
    apply_visualization_style()
    ax.set_facecolor(COLORS["background"])
    ax.spines[["top", "right"]].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(COLORS["spine"])
        ax.spines[spine].set_linewidth(SPINE_WIDTH)

    ax.tick_params(colors=COLORS["text"], labelsize=FONT_SIZE_TICK, width=SPINE_WIDTH)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontproperties(body_fontproperties(size=FONT_SIZE_TICK))

    ax.xaxis.label.set_fontproperties(
        body_fontproperties(size=FONT_SIZE_AXIS_TITLE)
    )
    ax.yaxis.label.set_fontproperties(
        body_fontproperties(size=FONT_SIZE_AXIS_TITLE)
    )
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])

    if title is not None:
        size = FONT_SIZE_SUBTITLE if subtitle else FONT_SIZE_TITLE
        props = title_fontproperties(size=size)
        ax.set_title(
            title,
            loc="left",
            pad=TITLE_PAD,
            color=COLORS["text"],
        )
        # Apply after set_title so matplotlib does not fall back to rcParams body face.
        ax.title.set_fontproperties(props)


def style_figure_title(fig: plt.Figure, title: str, *, x: float = 0.01) -> None:
    """Left-aligned figure ``suptitle`` using the title font family."""
    apply_visualization_style()
    props = title_fontproperties(size=FONT_SIZE_TITLE)
    text = fig.suptitle(
        title,
        x=x,
        ha="left",
        color=COLORS["text"],
    )
    text.set_fontproperties(props)


def style_legend(ax: plt.Axes, **legend_kwargs):
    """Draw a frameless legend with the body font; returns the Legend or None."""
    apply_visualization_style()
    kwargs = {
        "frameon": LEGEND_FRAMEON,
        "prop": body_fontproperties(size=FONT_SIZE_LEGEND),
    }
    kwargs.update(legend_kwargs)
    legend = ax.legend(**kwargs)
    if legend is not None:
        for text in legend.get_texts():
            text.set_fontproperties(body_fontproperties(size=FONT_SIZE_LEGEND))
    return legend


def annotation_kwargs(**overrides) -> dict:
    """Default kwargs for matplotlib text / annotations on the axes."""
    apply_visualization_style()
    base = {
        "fontsize": FONT_SIZE_ANNOTATION,
        "fontfamily": _RESOLVED_BODY_FAMILY,
        "color": COLORS["text"],
    }
    base.update(overrides)
    return base


def new_figure(figsize=None, **subplots_kwargs):
    """Create a figure/axes pair after applying the shared style."""
    apply_visualization_style()
    if figsize is None:
        figsize = FIGSIZE_STANDARD
    return plt.subplots(figsize=figsize, **subplots_kwargs)
