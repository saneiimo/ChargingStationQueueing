"""
Slot dynamics on an exact energy grid (the paper's relaxed dynamics (46)).

The compact model's one-slot update (``offline_cl_opt`` rows (13), (14),
(17), (18) and ``x <= W``), with the most power the slot allows, is

    g_j(x, q) = min{ x + h*Delta*q,  x + h*P_bar_j,  phi_j*x + (1 - phi_j)*R_j,  W_j }    (5)

with ``phi_j = exp(-h/tau_j)``: the taper row ``tau_d*p + x <= R`` with
``tau_d = h / (1 - phi_j)`` (``VehicleData.tau_delta_hours``) is exactly the
third term. ``g`` is non-decreasing in ``x`` and in ``q`` (the paper's
Proposition 1), which is what labelling, the cost-to-go tables and the
dominance rule of the search all rely on.

Why a grid
----------
In floating point, (5) gains digits every slot (it multiplies by ``phi``),
so two equal energies reached by different module orders can differ in the
last bit, and a bound of exactly 100 can come out as ``100 + 1e-11`` and be
rounded up to 101. Both would make the certificate wrong. The search
therefore runs on a *relaxation* whose arithmetic is exact:

* energies are multiples of ``GRID = 2^-30`` kWh;
* every term of (5) is rounded **up** to the grid, with a ``MARGIN = 2^-40``
  kWh safety margin that covers the floating-point error of evaluating it:

      g_hat(x, q) = min{ x + q*kappa,  x + peak_j,  ceil(phi*x + (1-phi)*R + MARGIN),  ceil(W) }   (46)

  with ``kappa = ceil(h*Delta + MARGIN)`` (one module for one slot) and
  ``peak_j`` the peak-power term.

``g_hat >= g`` for the same decisions, so a schedule never holds less energy
under (46) than under (5): the relaxed optimum is at most the true optimum
and every lower bound computed on the grid is valid for the compact model
(Proposition 8). The best schedule is replayed with (5) afterwards
(``exact_plan``) and validated by ``offline_cl_PB.validate_schedule``.
Linear-phase energies add ``q*kappa`` exactly, so the same total of
module-slots in any order gives the same energy, which lets the dominance
rule merge such states.

Exactness needs energies below ``2^10`` kWh (``x + q*kappa`` then fits in 40
significant bits); ``SearchInstance`` checks it.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit

GRID = 2.0**-30  # energy grid, kWh
GRID_INV = 2.0**30
MARGIN = 2.0**-40  # kWh; covers the float error of one evaluation of a term of (5)

# Column layout of ``SearchInstance.params`` (one row per vehicle).
P_W, P_R, P_PHI, P_PHI_R, P_PEAK, P_WCAP = range(6)


def ceil_to_grid(y: float) -> float:
    """Smallest multiple of ``GRID`` that is ``>= y`` (exact: scaling by 2^30 is exact)."""
    return math.ceil(y * GRID_INV) * GRID


def module_energy(h: float, p_module: float) -> float:
    """``kappa``: energy of one module over one slot on the grid, ``>= h*Delta``."""
    return ceil_to_grid(h * p_module + MARGIN)


def peak_energy(p_bar: float, p_module: float, kappa: float) -> float:
    """
    The peak-power term of (46), ``>= h*P_bar``. When ``P_bar`` is a whole
    number of modules ``r`` it is exactly ``r*kappa``, so ``r`` modules and the
    peak cap give the same energy (as they do in (5)).
    """
    r = p_bar / p_module
    if r == math.floor(r) and r * p_module == p_bar:
        return r * kappa
    return ceil_to_grid(r * kappa + MARGIN)


def vehicle_params(W: float, R: float, phi: float, p_bar: float, p_module: float, kappa: float) -> np.ndarray:
    """One row of ``SearchInstance.params`` (columns ``P_W`` ... ``P_WCAP``)."""
    return np.array([W, R, phi, (1.0 - phi) * R, peak_energy(p_bar, p_module, kappa), ceil_to_grid(W)])


@njit(cache=True)
def slot_energy(x, q, kappa, prm):
    """``g_hat(x, q)``, eq. (46). ``prm`` is the vehicle's row of ``params``."""
    e = x + q * kappa
    t = x + prm[P_PEAK]
    if t < e:
        e = t
    t = np.ceil((prm[P_PHI] * x + prm[P_PHI_R] + MARGIN) * GRID_INV) * GRID
    if t < e:
        e = t
    if prm[P_WCAP] < e:
        e = prm[P_WCAP]
    return e


@njit(cache=True)
def useful_modules(x, N, kappa, prm):
    """
    ``q_hat(x)`` (6) on the grid: the fewest modules that give the same energy
    as all ``N``. More modules add nothing (Lemma 1(b)).
    """
    top = slot_energy(x, N, kappa, prm)
    for q in range(N + 1):
        if slot_energy(x, q, kappa, prm) >= top:
            return q
    return N


@njit(cache=True)
def fastest_trajectory(N, kappa, prm, max_slots):
    """
    Energies at boundaries 0, 1, ..., L of a charge from 0 with all ``N``
    modules, stopping at the first boundary with energy ``>= W`` (``L`` is the
    fewest slots the vehicle needs, (7)). By monotonicity, a vehicle holding
    ``x <= traj[i]`` needs at least ``L - i`` more slots.
    """
    traj = np.empty(max_slots + 2)
    traj[0] = 0.0
    n = 0
    while traj[n] < prm[P_W] and n <= max_slots:
        traj[n + 1] = slot_energy(traj[n], N, kappa, prm)
        n += 1
    return traj[: n + 1].copy()


@njit(cache=True)
def slots_still_needed(x, traj):
    """Lower bound on the slots needed from energy ``x``: ``L - min{i : traj[i] >= x}``."""
    L = traj.shape[0] - 1
    lo, hi = 0, L
    while lo < hi:
        mid = (lo + hi) // 2
        if traj[mid] >= x:
            hi = mid
        else:
            lo = mid + 1
    return L - lo
