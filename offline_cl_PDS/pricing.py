"""
Exact pricing by forward labelling (the paper's pricing problem (27)).

For one vehicle and a rent table ``rent[k, q]`` (what holding a connector
with ``q`` modules in slot ``k`` costs, ``>= 0`` and non-decreasing in
``q``), find the plan -- start ``S >= k_j``, departure ``D <= K``, modules
``q_S .. q_{D-1}`` -- minimising

    w_j * D + sum_{k=S}^{D-1} rent[k, q_k]

where the vehicle departs at the first boundary its energy reaches ``W_j``
under the grid dynamics (``dynamics.slot_energy``).

A label is ``(energy at boundary k, rent paid so far)``. Because the
dynamics are non-decreasing in energy, a label with at least the energy and
at most the rent of another at the same boundary does at least as well in
every continuation, so only non-dominated labels are kept (energies are grid
values, so the comparison is exact). A label is also dropped when even the
fastest possible finish, ``w_j*(k + 1 + slots_still_needed)``, cannot beat
``cutoff`` or cannot finish by ``K``. The returned minimum is therefore exact
whenever it is below ``cutoff``; otherwise the true minimum is at least
``cutoff`` (``colgen`` uses ``min(best, cutoff)`` as the bound).

This is the same problem ``offline_cl_PB.labeling`` solves (in minutes and
with pile-indexed duals), compiled and on the grid.
"""

from __future__ import annotations

import numpy as np
from numba import njit

from .dynamics import P_W, slot_energy, slots_still_needed, useful_modules

INF = 1e18


@njit(cache=True)
def _grow(a, n):
    out = np.empty(2 * a.shape[0], a.dtype)
    out[:n] = a[:n]
    return out


@njit(cache=True)
def price_vehicle(release, K, forced_start, x_start, N, kappa, prm, weight, traj, rent, cutoff, n_columns, max_labels):
    """
    Returns ``(best, n_found, starts, departures, costs, modules, ok)``:
    ``best`` the minimum found (``INF`` if none below ``cutoff``), and the
    ``n_found <= n_columns`` cheapest plans with distinct departures
    (``modules[i, k]`` for ``k`` in ``[starts[i], departures[i])``). ``ok`` is
    False if more than ``max_labels`` labels were needed (the result is then
    unusable). ``forced_start``: the vehicle is plugged in at ``release``
    (an OPTIMIZE boundary vehicle) and starts with ``x_start``.
    """
    W = prm[P_W]
    # label pool: energy, rent, parent label, modules of the slot that created it, start slot
    cap = 1024
    lx = np.empty(cap)
    lc = np.empty(cap)
    lpar = np.empty(cap, np.int64)
    lq = np.empty(cap, np.int64)
    ls = np.empty(cap, np.int64)
    n_pool = 0
    cur = np.empty(0, np.int64)
    best_cost = np.full(K + 1, INF)
    best_label = np.full(K + 1, -1, np.int64)
    best_q = np.zeros(K + 1, np.int64)
    last_start = release if forced_start else K - 1

    for k in range(release, K):
        if k > last_start and cur.shape[0] == 0:
            break
        # Labels entering slot k: the survivors plus a fresh plug-in at boundary k.
        if k <= last_start:
            if n_pool == cap:
                lx, lc, lpar, lq, ls = _grow(lx, n_pool), _grow(lc, n_pool), _grow(lpar, n_pool), _grow(lq, n_pool), _grow(ls, n_pool)
                cap *= 2
            lx[n_pool], lc[n_pool], lpar[n_pool], lq[n_pool], ls[n_pool] = x_start, 0.0, -1, -1, k
            ext = np.empty(cur.shape[0] + 1, np.int64)
            ext[: cur.shape[0]] = cur
            ext[cur.shape[0]] = n_pool
            cur = ext
            n_pool += 1

        size = cur.shape[0] * (N + 1)
        cx = np.empty(size)
        cc = np.empty(size)
        cpar = np.empty(size, np.int64)
        cq = np.empty(size, np.int64)
        nc = 0
        for lab in cur:
            x, c = lx[lab], lc[lab]
            for q in range(useful_modules(x, N, kappa, prm) + 1):
                x2 = slot_energy(x, q, kappa, prm)
                c2 = c + rent[k, q]
                if x2 >= W:  # departs at boundary k+1
                    total = c2 + weight * (k + 1)
                    if total < best_cost[k + 1]:
                        best_cost[k + 1], best_label[k + 1], best_q[k + 1] = total, lab, q
                    continue
                rem = slots_still_needed(x2, traj)
                if k + 1 + rem > K or c2 + weight * (k + 1 + rem) >= cutoff:
                    continue
                cx[nc], cc[nc], cpar[nc], cq[nc] = x2, c2, lab, q
                nc += 1

        # Keep the non-dominated labels: scan by energy, highest first; within
        # one energy keep the cheapest; keep it only if cheaper than every
        # label with more energy.
        order = np.argsort(-cx[:nc], kind="mergesort")
        keep = np.empty(nc, np.int64)
        n_keep = 0
        cheapest = INF
        t = 0
        while t < nc:
            u, pick = t, order[t]
            while u < nc and cx[order[u]] == cx[order[t]]:
                if cc[order[u]] < cc[pick]:
                    pick = order[u]
                u += 1
            if cc[pick] < cheapest:
                cheapest = cc[pick]
                keep[n_keep] = pick
                n_keep += 1
            t = u
        if n_pool + n_keep > max_labels:
            return INF, 0, np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0), np.zeros((0, K), np.int64), False
        cur = np.empty(n_keep, np.int64)
        for t in range(n_keep):
            i = keep[t]
            if n_pool == cap:
                lx, lc, lpar, lq, ls = _grow(lx, n_pool), _grow(lc, n_pool), _grow(lpar, n_pool), _grow(lq, n_pool), _grow(ls, n_pool)
                cap *= 2
            lx[n_pool], lc[n_pool], lpar[n_pool], lq[n_pool], ls[n_pool] = cx[i], cc[i], cpar[i], cq[i], ls[cpar[i]]
            cur[t] = n_pool
            n_pool += 1

    # The n_columns cheapest departures, each with its module profile.
    order = np.argsort(best_cost)
    n_found = 0
    starts = np.zeros(n_columns, np.int64)
    departures = np.zeros(n_columns, np.int64)
    costs = np.zeros(n_columns)
    modules = np.zeros((n_columns, K), np.int64)
    for D in order:
        if n_found == n_columns or best_cost[D] >= INF:
            break
        departures[n_found], costs[n_found] = D, best_cost[D]
        modules[n_found, D - 1] = best_q[D]
        lab, k = best_label[D], D - 2
        while lq[lab] >= 0:
            modules[n_found, k] = lq[lab]
            lab, k = lpar[lab], k - 1
        starts[n_found] = ls[lab]
        n_found += 1
    return best_cost.min(), n_found, starts, departures, costs, modules, True
