"""Plotting helpers for charging-station simulation runs.

Shared matplotlib fonts, sizes, and colors live in ``visualization.style``.
Theory, trace densification, and demo episodes live in sibling modules
(``charging_theory``, ``charge_trace``, ``demo``, etc.).
"""

from visualization.style import apply_visualization_style

__all__ = ["apply_visualization_style"]
