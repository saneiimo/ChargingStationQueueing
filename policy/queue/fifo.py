"""
FIFO customer order; join the pile with the most free dispensers.

Leaves waiting order unchanged (head-of-line). Pile routing and optional
``max_wait`` override live on ``QueuePolicy``.
"""

from __future__ import annotations
from .base import QueuePolicy


class FIFOQueuePolicy(QueuePolicy):
    """HOL service; default free-dispenser pile choice."""

    # _select_ev defaults to queue[0]; select_pile defaults to most free dispensers.
    pass
