"""
Cartesian sweeps over any frozen-dataclass config.

Both pipelines describe one sweep point as a flat dataclass
(``objective_sweep.TrialConfig``, ``policy_sweep.PolicyConfig``), so the
grid builder is shared rather than written twice.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any, Sequence, TypeVar

TConfig = TypeVar("TConfig")


def config_grid(base: TConfig, **sweeps: Sequence[Any]) -> list[TConfig]:
    """
    Cartesian product of the given field values, on top of ``base``.

    >>> config_grid(BASE, mean_interarrival=[20.0, 30.0], delta=[1.0, 2.0])  # 4

    Fields not listed keep their ``base`` value. Order is row-major in the
    order the keyword arguments were given, so the FIRST keyword varies
    slowest -- put the expensive knob first and the cheap trials come out
    of the way early, so an early failure shows up on a cheap one.

    Raises ``KeyError`` on a field name the config does not have, rather
    than silently ignoring a typo'd sweep axis.
    """
    if not dataclasses.is_dataclass(base):
        raise TypeError(f"base must be a dataclass instance, got {type(base)!r}")

    unknown = set(sweeps) - {f.name for f in dataclasses.fields(base)}
    if unknown:
        raise KeyError(
            f"Not {type(base).__name__} fields: {sorted(unknown)}"
        )
    if not sweeps:
        return [base]

    names = list(sweeps)
    return [
        dataclasses.replace(base, **dict(zip(names, combo)))  # type: ignore[type-var]
        for combo in itertools.product(*(sweeps[n] for n in names))
    ]
