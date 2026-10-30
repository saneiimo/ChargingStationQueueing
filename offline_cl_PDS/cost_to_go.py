"""
Cost-to-go tables (Section 5.4) and the station term of the bound.

For fixed rents ``rent[k, q]``, a vehicle that is connected during slot ``k``
holding ``x`` kWh can finish at a cost (departure plus rent) of at least

    V_j(k, x) = min_q { rent[k, q] + ( w_j*(k+1)                  if g(x, q) >= W_j
                                       V_j(k+1, g(x, q))           otherwise ) }     (32)

with ``V_j(K, x) = INF`` for ``x < W_j`` (it must depart by ``K``). A vehicle
not yet connected at boundary ``k`` can start at any later boundary:

    U_j(k) = min_{S >= max(k, k_j)} V_j(S, 0).                                       (33)

``V_j(k, .)`` is non-increasing and piecewise constant in ``x``
(Proposition 4), so each slot's table is a list of breakpoints ``bp`` and
values ``val``: ``V = val[i]`` on ``[bp[i], bp[i+1])``. Breakpoints of slot
``k`` are found by mapping those of slot ``k+1`` (and the departure
threshold) back through each module count: the smallest grid energy ``x``
with ``g(x, q) >= y`` (``_preimage``, guided by the inverse (34)).

Verification (the paper's (47))
-------------------------------
The search only looks up grid energies, and every table must be a *lower
bound* on the true cost to go for the bound to stay valid. After a slot's
table is built, each step is checked against ``H(x)`` -- the right-hand side
of (32) evaluated with the already-checked table of slot ``k+1``. ``H`` is
non-increasing, so if ``H`` at the step's last grid energy is still at least
the step's value, ``H`` equals it on the whole step. If not, a breakpoint was
missed: bisection finds it and a new step starts there with the value of
``H`` at the old step's end (the smallest ``H`` on that range), so the table
stays below ``H``. By induction from ``K`` every table is a lower bound at
every grid energy. ``TableStats.repairs`` counts these corrections (the
paper reports none; neither should these tests).

Steps closer than ``MERGE_KWH`` are merged into the earlier start with the
lower value -- the table can only go down, so it stays a lower bound.

Prices are rounded down to multiples of ``PRICE_GRID`` first (45): any
non-negative prices give a valid bound, and with these every rent, table
value and bound is a multiple of ``2^-20`` far below ``2^53``, so sums are
exact in double precision and ``ceil`` of a bound is exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit

from .dynamics import GRID, GRID_INV, P_PEAK, P_PHI, P_PHI_R, P_W, P_WCAP, slot_energy
from .instance import SearchInstance
from .master import rent_and_gamma

INF = 1e18
PRICE_GRID = 2.0**-20
MERGE_KWH = 1e-6


def round_prices(mu: np.ndarray) -> np.ndarray:
    """(45): every price rounded down to a multiple of ``PRICE_GRID`` (and clipped at 0)."""
    return np.floor(np.maximum(mu, 0.0) / PRICE_GRID) * PRICE_GRID


@dataclass
class Tables:
    rent: np.ndarray  # [K, N+1] grid rents
    gamma: np.ndarray  # [K] grid station terms
    tail: np.ndarray  # [K+2]: sum_{k'>=k} gamma[k'] - FIXED vehicles' rent in slots >= k
    off: np.ndarray  # int64[J, K+1], offset of V_j(k, .) in bp/val (-1 for k < k_j)
    cnt: np.ndarray  # int64[J, K+1], number of steps
    bp: np.ndarray  # float64, all breakpoints
    val: np.ndarray  # float64, all values
    U: np.ndarray  # [J, K+1]
    repairs: int  # steps corrected by the verification
    build_s: float


# --------------------------------------------------------------------------- #
# one vehicle
# --------------------------------------------------------------------------- #


@njit(cache=True)
def _lookup(bp, val, o, n, x):
    """Value of the step function stored at ``bp[o:o+n]`` / ``val[o:o+n]`` at ``x``."""
    lo, hi = 0, n - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if bp[o + mid] <= x:
            lo = mid
        else:
            hi = mid - 1
    return val[o + lo]


@njit(cache=True)
def _H(x, k, N, kappa, prm, weight, rent, nbp, nval, n_next):
    """Right-hand side of (32) at energy ``x``, using the next slot's table."""
    best = INF
    for q in range(N + 1):
        e = slot_energy(x, q, kappa, prm)
        if e >= prm[P_W]:
            c = rent[k, q] + weight * (k + 1)
        else:
            c = rent[k, q] + _lookup(nbp, nval, 0, n_next, e)
        if c < best:
            best = c
    return best if best < INF / 2 else INF


@njit(cache=True)
def _preimage(y, q, kappa, prm):
    """Smallest grid energy ``x >= 0`` with ``g(x, q) >= y``, starting from the inverse (34)."""
    if q == 0:
        est = y
    else:
        est = max(y - q * kappa, y - prm[P_PEAK], (y - prm[P_PHI_R]) / prm[P_PHI])
    i = max(np.int64(np.ceil(est * GRID_INV)) - 2, 0)
    steps = 0
    while slot_energy(i * GRID, q, kappa, prm) < y and steps < 64:
        i += 1
        steps += 1
    while i > 0 and slot_energy((i - 1) * GRID, q, kappa, prm) >= y and steps < 128:
        i -= 1
        steps += 1
    return i * GRID  # only a candidate: the verification below makes the table safe either way


@njit(cache=True)
def vehicle_table(release, K, N, kappa, prm, weight, rent):
    """
    ``V_j(k, .)`` for ``k = K, K-1, ..., release``. Returns
    ``(off[K+1], cnt[K+1], bp, val, repairs)`` with local offsets.
    """
    W, w_grid = prm[P_W], prm[P_WCAP]  # w_grid: smallest grid energy that departs
    off = np.full(K + 1, -1, np.int64)
    cnt = np.zeros(K + 1, np.int64)
    cap = 1024
    bp = np.empty(cap)
    val = np.empty(cap)
    used = 0
    repairs = 0
    # Slot K: no time left -- INF below W.
    nbp = np.zeros(1)
    nval = np.full(1, INF)
    bp[0], val[0] = 0.0, INF
    off[K], cnt[K], used = 0, 1, 1

    for k in range(K - 1, release - 1, -1):
        n_next = nbp.shape[0]
        # Candidate breakpoints: 0 and the preimages of the next slot's breakpoints and of W.
        cand = np.empty(1 + (N + 1) * n_next)
        nc = 1
        cand[0] = 0.0
        for q in range(N + 1):
            for i in range(n_next):
                y = w_grid if i == 0 else nbp[i]  # i == 0 stands for the departure threshold
                x = _preimage(y, q, kappa, prm)
                if x < W:
                    cand[nc] = x
                    nc += 1
        xs = np.unique(cand[:nc])

        # Steps where H drops (merging steps closer than MERGE_KWH).
        sbp = np.empty(xs.shape[0])
        sval = np.empty(xs.shape[0])
        ns = 0
        last = 2 * INF
        for x in xs:
            v = _H(x, k, N, kappa, prm, weight, rent, nbp, nval, n_next)
            if v < last:
                if ns > 1 and x - sbp[ns - 1] <= MERGE_KWH:
                    sval[ns - 1] = v
                else:
                    sbp[ns], sval[ns] = x, v
                    ns += 1
                last = v

        # Verification: each step [sbp[i], right) must satisfy table <= H at
        # every grid energy; split it where it does not.
        vbp = np.empty(2 * ns)
        vval = np.empty(2 * ns)
        nv = 0
        for i in range(ns):
            right = sbp[i + 1] if i + 1 < ns else w_grid
            il = np.int64(np.ceil(sbp[i] * GRID_INV))
            ir = np.int64(np.ceil(right * GRID_INV)) - 1  # last grid energy of the step
            if ir < il:
                continue
            v = min(sval[i], _H(il * GRID, k, N, kappa, prm, weight, rent, nbp, nval, n_next))
            vbp[nv], vval[nv] = sbp[i], v
            nv += 1
            h_right = _H(ir * GRID, k, N, kappa, prm, weight, rent, nbp, nval, n_next)
            if h_right >= v:
                continue
            repairs += 1
            lo, hi = il, ir  # H(lo) >= v > H(hi)
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if _H(mid * GRID, k, N, kappa, prm, weight, rent, nbp, nval, n_next) >= v:
                    lo = mid
                else:
                    hi = mid
            vbp[nv], vval[nv] = hi * GRID, h_right
            nv += 1

        while used + nv > cap:
            cap *= 2
            nb, nvl = np.empty(cap), np.empty(cap)
            nb[:used], nvl[:used] = bp[:used], val[:used]
            bp, val = nb, nvl
        off[k], cnt[k] = used, nv
        bp[used : used + nv] = vbp[:nv]
        val[used : used + nv] = vval[:nv]
        used += nv
        nbp, nval = vbp[:nv].copy(), vval[:nv].copy()
    return off, cnt, bp[:used].copy(), val[:used].copy(), repairs


@njit(cache=True)
def table_value(off, cnt, bp, val, j, k, x):
    """``V_j(k, x)`` from the assembled tables (``INF`` if ``k < k_j``)."""
    o = off[j, k]
    if o < 0:
        return INF
    return _lookup(bp, val, o, cnt[j, k], x)


# --------------------------------------------------------------------------- #
# all vehicles
# --------------------------------------------------------------------------- #


def station_tail(inst: SearchInstance, rent: np.ndarray, gamma: np.ndarray) -> np.ndarray:
    """
    ``tail[k] = sum_{k' >= k} gamma[k'] - (rent the FIXED vehicles pay in slots >= k)``,
    for ``k = 0..K+1`` (0 past the horizon). The FIXED vehicles' own rent is
    known exactly, so only the searched vehicles' rent needs the bound
    ``gamma`` (see ``search.py``).
    """
    K = inst.K
    per_slot = gamma[:K].copy()
    for plan in inst.fixed_plans.values():
        for k in plan.occupied_slots():
            per_slot[k] -= rent[k, plan.q(k)]
    tail = np.zeros(K + 2)
    tail[:K] = np.cumsum(per_slot[::-1])[::-1]
    return tail


def build_tables(inst: SearchInstance, mu: np.ndarray, configs: np.ndarray) -> Tables:
    """Round the prices (45), then build every searched vehicle's ``V`` and ``U``."""
    import time

    t0 = time.perf_counter()
    K, N, J = inst.K, inst.station.n_modules, inst.J
    rent, gamma = rent_and_gamma(round_prices(mu), configs, inst.station.n_piles)
    rent = np.ascontiguousarray(rent)
    off = np.full((J, K + 1), -1, dtype=np.int64)
    cnt = np.zeros((J, K + 1), dtype=np.int64)
    U = np.full((J, K + 1), INF)
    bps, vals = [], []
    base, repairs = 0, 0
    for i in range(J):
        o, c, bp, val, rep = vehicle_table(
            int(inst.release[i]), K, N, inst.kappa, inst.params[i], int(inst.weight[i]), rent
        )
        mask = o >= 0
        off[i, mask] = o[mask] + base
        cnt[i] = c
        bps.append(bp)
        vals.append(val)
        base += len(bp)
        repairs += rep
        # U_j(k) = min over S >= max(k, k_j) of V_j(S, 0) = val at the first step (bp = 0).
        for k in range(K - 1, -1, -1):
            v0 = val[o[k]] if k >= inst.release[i] else INF
            U[i, k] = min(v0, U[i, k + 1])
    return Tables(
        rent=rent, gamma=gamma, tail=station_tail(inst, rent, gamma), off=off, cnt=cnt,
        bp=np.concatenate(bps) if bps else np.zeros(0), val=np.concatenate(vals) if vals else np.zeros(0),
        U=U, repairs=repairs, build_s=time.perf_counter() - t0,
    )
