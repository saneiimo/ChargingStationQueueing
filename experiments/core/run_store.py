"""
Where a sweep's output goes, and how it gets there.

One ``RunStore`` owns one run directory under
``experiments/results/<run_name>/``. Both pipelines write through it, so
the on-disk conventions stay identical between them:

* every table is rewritten after each config, so an interrupted sweep
  still leaves usable output;
* ``run_meta.json`` records the sweep definition and timing;
* nothing is ever appended -- see ``RunStore``'s own docstring for what
  reusing a ``run_name`` does.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from config import PROJECT_ROOT

DEFAULT_OUT_DIR = PROJECT_ROOT / "experiments" / "results"


def jsonable(obj: Any) -> Any:
    """JSON fallback for numpy scalars, sets, and anything else exotic."""
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, (set, frozenset)):
        return sorted(str(x) for x in obj)
    return str(obj)


class RunStore:
    """
    One sweep run's output directory.

    Reusing a ``run_name`` **overwrites**: every table and ``run_meta.json``
    is rewritten from the new run alone, with no merge and no de-duplication
    against what was there. Per-config JSON sidecars are overwritten only at
    the indices the new run reaches, so a shorter re-run into the same name
    leaves the previous run's higher-numbered sidecars behind as stale
    orphans. Use a fresh ``run_name`` (the timestamp default does this) when
    that matters.
    """

    def __init__(
        self,
        run_name: str | None,
        out_dir: Path | str = DEFAULT_OUT_DIR,
        *,
        prefix: str = "sweep",
    ):
        self.run_name = run_name or f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
        self.dir = Path(out_dir) / self.run_name
        self.dir.mkdir(parents=True, exist_ok=True)
        self._meta: dict[str, Any] = {}
        self._t0 = time.perf_counter()

    # -- metadata ------------------------------------------------------- #
    def start_meta(self, **fields: Any) -> None:
        """Write ``run_meta.json`` up front, so a killed run still explains itself."""
        self._meta = {
            "run_name": self.run_name,
            "started": datetime.now().isoformat(timespec="seconds"),
            **fields,
        }
        self._flush_meta()

    def finish_meta(self, **fields: Any) -> dict[str, Any]:
        """Stamp completion time and total runtime; returns the final meta."""
        self._meta.update(fields)
        self._meta["finished"] = datetime.now().isoformat(timespec="seconds")
        self._meta["total_runtime_s"] = time.perf_counter() - self._t0
        self._flush_meta()
        return self._meta

    def _flush_meta(self) -> None:
        (self.dir / "run_meta.json").write_text(
            json.dumps(self._meta, indent=2, default=jsonable)
        )

    # -- tables --------------------------------------------------------- #
    def write_table(self, name: str, rows: Sequence[dict[str, Any]] | pd.DataFrame):
        """Write ``<name>.csv``. Called after every config, not just at the end."""
        frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(list(rows))
        frame.to_csv(self.dir / f"{name}.csv", index=False)
        return frame

    def write_json(self, relpath: str, payload: Any) -> None:
        """Write one JSON file, creating parent directories as needed."""
        path = self.dir / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=jsonable))


def load_table(
    run_name: str, name: str = "results", out_dir: Path | str = DEFAULT_OUT_DIR
) -> pd.DataFrame:
    """Read back one table from a finished (or in-progress) run."""
    path = Path(out_dir) / run_name / f"{name}.csv"
    if not path.exists():
        available = sorted(p.stem for p in path.parent.glob("*.csv"))
        raise FileNotFoundError(
            f"{path} does not exist. Tables in that run: {available or '(none)'}"
        )
    return pd.read_csv(path)


def load_meta(run_name: str, out_dir: Path | str = DEFAULT_OUT_DIR) -> dict[str, Any]:
    """Read back a run's ``run_meta.json``."""
    return json.loads((Path(out_dir) / run_name / "run_meta.json").read_text())


def list_runs(out_dir: Path | str = DEFAULT_OUT_DIR) -> list[str]:
    """Run names present under ``out_dir``, newest first."""
    root = Path(out_dir)
    if not root.is_dir():
        return []
    runs = [p for p in root.iterdir() if p.is_dir()]
    return [p.name for p in sorted(runs, key=lambda p: p.stat().st_mtime, reverse=True)]
