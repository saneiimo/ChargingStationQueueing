"""
Post-simulation validation and queueing-law summaries.

Call after an episode finishes (``SIM_OVER`` / Gym ``done``) to verify metric
invariants on finished and dropped EVs, and optionally print Little's law and
utilization.

Customer-level averages (W, W_q, S) and Little's-law algebra live on
``MetricsTracker``; this module only asserts and formats reports.

Example::

    from metrics.validate import validate_episode, report_queueing_laws

    validate_episode(env.engine, verbose=True)   # prints PASS lines; raises on fail
    report_queueing_laws(env.engine, verbose=True)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from config import HR2MIN

if TYPE_CHECKING:
    from env.charging_env import ChargingStationEnv
    from simulation.engine import SimulationEngine


class EpisodeValidationError(AssertionError):
    """Raised when one or more post-simulation checks fail."""


@dataclass
class CheckResult:
    """Outcome of a single named check."""

    name: str
    passed: bool
    detail: str = ""


@dataclass
class ValidationReport:
    """Aggregate result of ``validate_episode``."""

    checks: list[CheckResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed]


def _engine_from(
    env_or_engine: ChargingStationEnv | SimulationEngine,
) -> SimulationEngine:
    if hasattr(env_or_engine, "engine"):
        return env_or_engine.engine  # type: ignore[return-value]
    return env_or_engine  # type: ignore[return-value]


def _emit(verbose: bool, passed: bool, name: str, detail: str = "") -> None:
    """Print policy: always print failures; print passes only if ``verbose``."""
    if passed and not verbose:
        return
    status = "PASS" if passed else "FAIL"
    msg = f"  [{status}] {name}"
    if detail:
        msg = f"{msg}: {detail}"
    print(msg)


def _check(
    report: ValidationReport,
    name: str,
    passed: bool,
    detail: str,
    *,
    verbose: bool,
) -> None:
    report.checks.append(CheckResult(name=name, passed=passed, detail=detail))
    _emit(verbose, passed, name, detail)


def validate_episode(
    env_or_engine: ChargingStationEnv | SimulationEngine,
    *,
    verbose: bool = False,
    raise_on_fail: bool = True,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> ValidationReport:
    """
    Run post-simulation metric checks on a finished (or mid-run) episode.

    Parameters
    ----------
    env_or_engine :
        ``ChargingStationEnv`` or ``SimulationEngine``.
    verbose :
        If True, print a PASS line for every successful check. Failures are
        always printed. If False, print nothing unless a check fails.
    raise_on_fail :
        If True (default), raise ``EpisodeValidationError`` when any check fails.
    rtol, atol :
        Tolerances for floating SoC / energy comparisons.

    Returns
    -------
    ValidationReport
        Structured list of check outcomes (useful when ``raise_on_fail=False``).
    """
    engine = _engine_from(env_or_engine)
    metrics = engine.metrics
    station = engine.station
    report = ValidationReport()

    finished = list(metrics.finished_evs)
    dropped = list(metrics.dropped_evs)
    # Warm-up queue flush (SimulationEngine.flush_queue_at_warmup): EVs
    # discarded from the queue at the warm-up boundary. Empty unless a
    # warm-up flush actually happened -- see metrics_tracker.py's module
    # docstring, "Warm-up period". They remain in `arrived` too (they
    # really did arrive), so the conservation check below needs this as an
    # explicit fifth bucket, not folded into `dropped`.
    flushed = list(metrics.flushed_evs)
    arrived = list(metrics.arrived_evs)
    queued = list(station.queue)
    plugged = [ev for pile in station.piles for ev in pile.evs]

    if verbose:
        print(
            "\n=== Episode validation ===\n"
            f"  arrived={len(arrived)}, finished={len(finished)}, "
            f"dropped={len(dropped)}, flushed={len(flushed)}, "
            f"queued={len(queued)}, plugged={len(plugged)}"
        )

    # --- Finished EV checks -------------------------------------------------
    soc_bad: list[str] = []
    energy_target_bad: list[str] = []
    energy_agree_bad: list[str] = []
    time_order_bad: list[str] = []
    location_bad: list[str] = []
    power_cap_bad: list[str] = []

    for ev in finished:
        if not np.isclose(ev.s_current, ev.s_f, rtol=rtol, atol=atol):
            soc_bad.append(
                f"EV {ev.id}: s_current={ev.s_current:.6f} vs s_f={ev.s_f:.6f}"
            )

        target_kwh = (ev.s_f - ev.s_i) * ev.c_b / HR2MIN
        if not (
            np.isclose(ev.energy_received, target_kwh, rtol=rtol, atol=atol)
            and np.isclose(ev.energy_received, ev.energy_needed, rtol=rtol, atol=atol)
        ):
            energy_target_bad.append(
                f"EV {ev.id}: energy_received={ev.energy_received:.6e}, "
                f"energy_needed={ev.energy_needed:.6e}, "
                f"c_b*(s_f-s_i)/HR2MIN={target_kwh:.6e}"
            )

        if not np.isclose(
            ev.energy_received, ev.energy_received_2, rtol=rtol, atol=atol
        ):
            energy_agree_bad.append(
                f"EV {ev.id}: E_power={ev.energy_received:.6e} vs "
                f"E_SoC={ev.energy_received_2:.6e}"
            )

        t_arr = ev.arrival_time
        t_srv = ev.service_start_time
        t_dep = ev.departure_time
        if t_srv is None or not np.isfinite(t_dep) or not (t_dep >= t_srv >= t_arr):
            time_order_bad.append(
                f"EV {ev.id}: arrival={t_arr}, service_start={t_srv}, "
                f"departure={t_dep}"
            )

        if ev.pile_tracker is None or ev.connector_id_tracker is None:
            location_bad.append(
                f"EV {ev.id}: pile_tracker={ev.pile_tracker}, "
                f"connector_id_tracker={ev.connector_id_tracker}"
            )

        # Instantaneous samples logged during the DES (and at departure).
        for i, sample in enumerate(ev.charge_trace):
            _t, _s, p_req, p_act, _p_allot = sample
            if p_act > p_req + max(atol, rtol * max(abs(p_req), 1.0)):
                power_cap_bad.append(
                    f"EV {ev.id} sample[{i}]: p_act={p_act:.6f} > p_req={p_req:.6f}"
                )
                break

    _check(
        report,
        "Finished: s_current ~= s_f",
        not soc_bad,
        (
            f"ok for all finished EVs"
            if not soc_bad
            else f"{len(soc_bad)} failures; e.g. {soc_bad[0]}"
        ),
        verbose=verbose,
    )
    _check(
        report,
        "Finished: energy_received ~= c_b*(s_f-s_i)/HR2MIN",
        not energy_target_bad,
        (
            f"ok for all finished EVs"
            if not energy_target_bad
            else f"{len(energy_target_bad)} failures; e.g. {energy_target_bad[0]}"
        ),
        verbose=verbose,
    )
    _check(
        report,
        "Finished: energy_received ~= energy_received_2",
        not energy_agree_bad,
        (
            f"ok for all finished EVs"
            if not energy_agree_bad
            else f"{len(energy_agree_bad)} failures; e.g. {energy_agree_bad[0]}"
        ),
        verbose=verbose,
    )
    _check(
        report,
        "Finished: departure_time >= service_start_time >= arrival_time",
        not time_order_bad,
        (
            f"ok for all finished EVs"
            if not time_order_bad
            else f"{len(time_order_bad)} failures; e.g. {time_order_bad[0]}"
        ),
        verbose=verbose,
    )
    _check(
        report,
        "Finished: pile_tracker / connector_id_tracker set",
        not location_bad,
        (
            f"ok for all finished EVs"
            if not location_bad
            else f"{len(location_bad)} failures; e.g. {location_bad[0]}"
        ),
        verbose=verbose,
    )
    _check(
        report,
        "Finished: p_act <= p_req in charge_trace",
        not power_cap_bad,
        (
            f"ok for all finished EVs"
            if not power_cap_bad
            else f"{len(power_cap_bad)} failures; e.g. {power_cap_bad[0]}"
        ),
        verbose=verbose,
    )

    # --- Dropped EV checks --------------------------------------------------
    drop_bad: list[str] = []
    for ev in dropped:
        if ev.service_start_time is not None or ev.charge_trace:
            drop_bad.append(
                f"EV {ev.id}: service_start={ev.service_start_time}, "
                f"trace_len={len(ev.charge_trace)}"
            )
    _check(
        report,
        "Dropped: never got service_start_time / empty charge_trace",
        not drop_bad,
        (
            f"ok for all dropped EVs"
            if not drop_bad
            else f"{len(drop_bad)} failures; e.g. {drop_bad[0]}"
        ),
        verbose=verbose,
    )

    # --- Flushed EV checks (warm-up queue flush) -----------------------------
    # Same shape as the dropped-EV check above: a flushed EV was only ever
    # waiting in the queue, never plugged in -- see
    # SimulationEngine.flush_queue_at_warmup / metrics_tracker.py's module
    # docstring, "Warm-up period". Trivially passes (no flushed EVs to
    # check) whenever no warm-up flush happened.
    flush_bad: list[str] = []
    for ev in flushed:
        if ev.service_start_time is not None or ev.charge_trace:
            flush_bad.append(
                f"EV {ev.id}: service_start={ev.service_start_time}, "
                f"trace_len={len(ev.charge_trace)}"
            )
    _check(
        report,
        "Flushed: never got service_start_time / empty charge_trace",
        not flush_bad,
        (
            f"ok for all flushed EVs"
            if not flush_bad
            else f"{len(flush_bad)} failures; e.g. {flush_bad[0]}"
        ),
        verbose=verbose,
    )

    # --- Conservation of arrivals -------------------------------------------
    # `flushed` is a fifth bucket, not a subset of `dropped`: a flushed EV
    # remains in `arrived` (it really did arrive) but is no longer in
    # `queued` once removed, so it must appear explicitly on the right-hand
    # side or this invariant fails the moment a warm-up flush happens. With
    # no warm-up flush, flushed=[] and this is identical to the original
    # (pre-warm-up-feature) check.
    rhs = len(finished) + len(dropped) + len(flushed) + len(queued) + len(plugged)
    lhs = len(arrived)
    cons_ok = lhs == rhs
    _check(
        report,
        "arrived = finished + dropped + flushed + queued + plugged",
        cons_ok,
        (
            f"{lhs} vs {len(finished)}+{len(dropped)}+{len(flushed)}+"
            f"{len(queued)}+{len(plugged)}={rhs}"
        ),
        verbose=verbose,
    )

    if raise_on_fail and not report.ok:
        lines = [f"{c.name}: {c.detail}" for c in report.failures]
        raise EpisodeValidationError(
            "Episode validation failed:\n  - " + "\n  - ".join(lines)
        )
    return report


def report_queueing_laws(
    env_or_engine: ChargingStationEnv | SimulationEngine,
    *,
    verbose: bool = False,
    post_warmup_only: bool = False,
) -> dict[str, float]:
    """
    Print / return Little's law (system / queue) and connector utilization.

    Delegates all computation to ``MetricsTracker.queueing_summary`` so
    notebooks, tests, and this helper share one implementation.

    Parameters
    ----------
    env_or_engine :
        ``ChargingStationEnv`` or ``SimulationEngine``.
    verbose :
        If True, print the summary. If False, return the dict silently.
    post_warmup_only :
        If True, restrict the summary to the post-warm-up window
        (``since=metrics.warmup_period`` -- see ``MetricsTracker``'s module
        docstring, "Warm-up period"). ``0.0`` if no warm-up period was set,
        in which case this flag has no effect. If False (default), report
        over the whole run -- identical to this function's behaviour before
        the warm-up feature existed.

    Returns
    -------
    dict
        Same keys as ``MetricsTracker.queueing_summary`` (includes
        ``since``, so the caller can always see which window was used).

    Raises
    ------
    MetricsError
        If a finished EV is missing service / departure timestamps
        (propagated from the tracker).
    """
    from metrics.metrics_tracker import MetricsError

    engine = _engine_from(env_or_engine)
    metrics = engine.metrics
    station = engine.station
    T = float(engine.current_time) if engine.current_time > 0 else 1.0
    c = station.n_piles * station.n_connectors
    since = metrics.warmup_period if post_warmup_only else 0.0

    try:
        out = metrics.queueing_summary(T, n_servers=c, since=since)
    except MetricsError as exc:
        raise EpisodeValidationError(str(exc)) from exc

    if verbose:
        window_label = (
            f"post-warm-up, t>={since:g}" if since > 0 else "full run"
        )
        print(f"\n=== Queueing laws ({window_label}) ===")
        print(
            f"  T={out['T']:.1f} min, finished={int(out['n_finished'])}, "
            f"c={int(out['c'])} servers"
        )
        print(f"  lambda_eff={out['lambda_eff']:.6f} /min")
        print(
            f"  E[W]={out['W']:.4f} min, E[W_q]={out['W_q']:.4f} min, "
            f"E[S]={out['S']:.4f} min"
        )
        print(
            f"  Little L = lambda_eff * W :  theory={out['L_theory']:.4f}  "
            f"sim={out['L_sim']:.4f}  |diff|={abs(out['L_theory'] - out['L_sim']):.4f}"
        )
        print(
            f"  Little Q = lambda_eff * W_q:  theory={out['Q_theory']:.4f}  "
            f"sim={out['Q_sim']:.4f}  |diff|={abs(out['Q_theory'] - out['Q_sim']):.4f}"
        )
        print(
            f"  Utilization rho           :  theory={out['rho_theory']:.4f}  "
            f"sim={out['rho_sim']:.4f}  |diff|={abs(out['rho_theory'] - out['rho_sim']):.4f}"
        )

    return out
