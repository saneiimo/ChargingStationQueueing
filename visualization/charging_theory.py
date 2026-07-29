"""
Unconstrained BMS charging theory (matches models/ev.py and the Charging Curve note).

Times are in simulation minutes; power in kW; SoC in [0, 1].
"""

from __future__ import annotations

from math import exp, log

import numpy as np

from models.ev import EV


def theoretical_p_of_s(ev: EV, s: np.ndarray | float) -> np.ndarray | float:
    """BMS power acceptance P(s) [kW] under full request."""
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

    Returns 0 if s <= s_i; clamps s near 1.
    """
    if s <= ev.s_i + 1e-15:
        return 0.0
    if s >= 1.0 - 1e-15:
        s = 1.0 - 1e-12

    c_b, p_m, s_th, s_i = ev.c_b, ev.p_req_max, ev.s_th, ev.s_i
    scale = c_b / p_m

    if s <= s_th:
        return scale * (s - s_i)
    if s_i >= s_th:
        return scale * (-(1.0 - s_th) * log((1.0 - s) / (1.0 - s_i)))
    return scale * ((s_th - s_i) - (1.0 - s_th) * log((1.0 - s) / (1.0 - s_th)))


def theoretical_p_of_local_t(ev: EV, t_local: np.ndarray | float) -> np.ndarray | float:
    """Unconstrained P(t) for t measured from plug-in [sim minutes]."""
    t_arr = np.asarray(t_local, dtype=float)
    t_th = theoretical_local_time_to_soc(ev, min(ev.s_th, ev.s_f))
    if ev.s_i >= ev.s_th:
        t_th = 0.0
        k = ev.p_req_max / (ev.c_b * (1.0 - ev.s_th))
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


def theoretical_power_vs_global_time(
    ev: EV,
    *,
    n_points: int = 200,
    soc_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Unconstrained theory P(t) on the global simulation clock.

    Returns (t_global, p_theory). Empty arrays if the EV never started service.
    """
    t0 = ev.service_start_time
    if t0 is None:
        return np.array([]), np.array([])

    s_end = min(ev.s_f, 0.999) if soc_end is None else min(soc_end, 0.999)
    t_end_local = theoretical_local_time_to_soc(ev, s_end)
    t_local = np.linspace(0.0, max(t_end_local, 1e-6), n_points)
    p_theory = theoretical_p_of_local_t(ev, t_local)
    return t0 + t_local, p_theory
