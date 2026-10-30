"""
The layered state-space search (Sections 5.1 and 5.5-5.7): one *pass*.

State
-----
At boundary ``k`` a state is ``(departed set F, connected vehicles with their
pile and energy, g)``, ``g = sum_{j in F} w_j D_j`` (plus the FIXED vehicles'
constant). Each pass builds the states slot by slot; a transition over slot
``k`` (``_expand`` below)

1. connects a set of waiting vehicles (arrived, not connected) to free
   connectors -- each must get >= 1 module (Lemma 1(c));
2. gives every pile a *maximal* module vector: every vehicle at most its
   useful modules ``q_hat`` and, unless all get ``q_hat``, the pile's whole
   module budget used (Lemma 1(a)-(b));
3. moves energies by the grid dynamics; a vehicle reaching ``W_j`` departs at
   ``k+1`` and adds ``w_j*(k+1)`` to ``g``.

Bound and pruning
-----------------
Every state carries the bound (35)

    f(s) = g - tail[k] + sum_connected V_j(k, x_j) + sum_unconnected U_j(k),

a lower bound on every schedule through ``s`` (Proposition 5; ``tail`` is the
station term, ``cost_to_go.station_tail``). A pass has a threshold ``T`` and
looks only for schedules with value below it; ``v`` is the best value it has
found (starts at ``T``). A successor is kept only if ``f <= v - 1`` (values
are integers, ``f`` is exact on the price grid) and ``f < adm`` (admission
threshold after a memory cut). The enumeration of connection sets uses the
smallest bound increase ``delta_i`` of each waiting vehicle (Lemma 2(a), (40))
and the product of per-pile module vectors uses their sorted contributions
(Lemma 2(b)) to discard whole families of successors at once.

Vehicles that *must* connect now: if ``U_j(k+1) = INF`` (it could no longer
finish by ``K`` if it waited), the vehicle is part of every surviving
decision. Handling this explicitly keeps the arithmetic free of ``INF - INF``
cancellations, which near the horizon would otherwise make a bound inexact.

Dominance (Proposition 6): two states with the same departed set and the same
connected vehicles on the same piles -- one with ``g`` no larger and every
energy at least as large -- the first is kept. Identical piles are relabelled
canonically (by their smallest connected vehicle) once no FIXED vehicle
breaks their symmetry (``symmetric_from``), and a new vehicle goes onto an
empty pile only if every lower-numbered empty pile also gets one.

What a pass certifies (Theorem 1, eqs. (41)-(42))
-------------------------------------------------
``status``: 0 complete; 1 memory cut (or beam cut); 2 expansion budget;
3 time limit. ``beta`` is the smallest bound of anything
the pass gave up (memory/beam cuts: the admission threshold; budget/time
stop: the open states). The pass proves that no schedule has a value below
``min(T, v, ceil(beta))`` -- ``beta = INF`` for a complete pass.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from numba import njit, objmode

from .cost_to_go import INF, Tables, table_value
from .dynamics import P_W, slot_energy, useful_modules
from .instance import SearchInstance

BITS = 62  # vehicles per bitmask word
COMPLETE, MEMORY_CUT, BUDGET, TIME_LIMIT = range(4)
STATUS_NAMES = {COMPLETE: "complete", MEMORY_CUT: "memory cut", BUDGET: "expansion budget", TIME_LIMIT: "time limit"}


@njit(cache=True)
def _keep(f, v, adm):
    """Successor test: ``ceil(f) < v`` and ``f < adm`` (``f`` exact on the price grid)."""
    return f <= v - 1.0 and f < adm


@njit(cache=True)
def _hash(done, NW, veh, pile, n):
    h = np.int64(-3750763034362895579)
    for w in range(NW):
        h = (h ^ done[w]) * np.int64(1099511628211)
    for i in range(n):
        h = (h ^ (np.int64(veh[i]) * 131 + np.int64(pile[i]) + 7)) * np.int64(1099511628211)
    return h


@njit(cache=True)
def _pile_options(codes, nmem, cterm, cneed, wterm, wneed, budget, out_cost, out_q, out_order):
    """
    All module vectors Lemma 1 allows on one pile, with their contribution to
    the successor's bound (sum of the members' terms), sorted by it.
    ``codes[t] >= 0``: connected vehicle (lower bound 0 modules); ``< 0``:
    newly connected waiting vehicle ``-code-1`` (at least 1 module). Returns
    the number of vectors (0 = this pile cannot take the decision).
    """
    if nmem == 0:
        out_cost[0] = 0.0
        out_order[0] = 0
        return 1
    lo = np.empty(nmem, np.int64)
    hi = np.empty(nmem, np.int64)
    s_lo, s_hi = 0, 0
    for t in range(nmem):
        c = codes[t]
        lo[t] = 0 if c >= 0 else 1
        hi[t] = cneed[c] if c >= 0 else wneed[-c - 1]
        s_lo += lo[t]
        s_hi += hi[t]
    if s_lo > budget or s_lo > s_hi:
        return 0
    target = min(budget, s_hi)  # all modules used, or everyone at q_hat
    q = lo.copy()
    n = 0
    while True:
        rest = target
        for t in range(nmem - 1):
            rest -= q[t]
        if lo[nmem - 1] <= rest <= hi[nmem - 1]:
            q[nmem - 1] = rest
            cost = 0.0
            for t in range(nmem):
                c = codes[t]
                cost += cterm[c, q[t]] if c >= 0 else wterm[-c - 1, q[t]]
            out_cost[n] = cost
            out_q[n, :nmem] = q
            n += 1
        t = 0  # odometer over the first nmem-1 members
        while t < nmem - 1:
            q[t] += 1
            if q[t] <= hi[t]:
                break
            q[t] = lo[t]
            t += 1
        if t >= nmem - 1:
            break
    order = np.argsort(out_cost[:n], kind="mergesort")
    out_order[:n] = order
    return n


@njit(cache=True)
def _grow_history(hist, n, MC, M):
    hpar, hk, hv, hq, hp, hperm = hist
    cap = 2 * hpar.shape[0]
    nh = (np.empty(cap, np.int64), np.empty(cap, np.int32), np.empty((cap, MC), np.int16),
          np.empty((cap, MC), np.int8), np.empty((cap, MC), np.int8), np.empty((cap, M), np.int8))
    nh[0][:n], nh[1][:n], nh[2][:n], nh[3][:n], nh[4][:n], nh[5][:n] = hpar[:n], hk[:n], hv[:n], hq[:n], hp[:n], hperm[:n]
    return nh


@njit(cache=True)
def _compact(L, b, n):
    """Move the alive states of layer ``b`` to the front; returns their number."""
    done, veh, pile, ex, ncon, ndone, g, f, hid, alive, key, _nxt = L
    w = 0
    for s in range(n):
        if alive[b, s]:
            if w != s:
                done[b, w], veh[b, w], pile[b, w], ex[b, w] = done[b, s], veh[b, s], pile[b, s], ex[b, s]
                ncon[b, w], ndone[b, w], g[b, w], f[b, w] = ncon[b, s], ndone[b, s], g[b, s], f[b, s]
                hid[b, w], alive[b, w], key[b, w] = hid[b, s], True, key[b, s]
            w += 1
    return w


@njit(cache=True)
def _rehash(L, b, n, table, touched, ntouched):
    """Rebuild the hash chains of layer ``b``; returns the new ``ntouched``."""
    key, nxt = L[10], L[11]
    for t in range(ntouched):
        table[touched[t]] = -1
    ntouched = 0
    mask = table.shape[0] - 1
    for s in range(n):
        slot = key[b, s] & mask
        if table[slot] < 0:
            touched[ntouched] = slot
            ntouched += 1
        nxt[b, s] = table[slot]
        table[slot] = s
    return ntouched


@njit(cache=True)
def run_pass(J, M, C, N, K, kappa, release, weight, params, off, cnt, bp, val, U, tail,
             free_conn, free_mod, symmetric_from, init_veh, init_pile, init_x, g0,
             threshold, beam_width, layer_cap, keep_history, scan_cap, max_expand, deadline):
    """
    One pass with threshold ``threshold`` (``beam_width > 0``: beam search).
    Returns ``(v, best_record, status, beta, peak, expanded, stored, history, n_records)``.
    """
    NW = (J + BITS - 1) // BITS
    MC = M * C
    cap = layer_cap
    # Two layers (current, next) of states, struct-of-arrays:
    L = (np.zeros((2, cap, NW), np.int64),  # 0 departed bitmask
         np.empty((2, cap, MC), np.int16),  # 1 connected vehicles, sorted by (pile, vehicle)
         np.empty((2, cap, MC), np.int8),  # 2 their pile labels
         np.empty((2, cap, MC)),  # 3 their energies (grid)
         np.zeros((2, cap), np.int8),  # 4 number connected
         np.zeros((2, cap), np.int16),  # 5 number departed
         np.zeros((2, cap)),  # 6 g
         np.zeros((2, cap)),  # 7 f, the bound (35)
         np.full((2, cap), -1, np.int64),  # 8 history record that created the state
         np.zeros((2, cap), np.bool_),  # 9 alive
         np.zeros((2, cap), np.int64),  # 10 hash key
         np.full((2, cap), -1, np.int64))  # 11 next state in the same hash bucket
    S_done, S_veh, S_pile, S_x, S_n, S_ndone, S_g, S_f, S_hid, S_alive, S_key, S_next = L
    TS = 1
    while TS < 2 * cap:
        TS *= 2
    table = np.full(TS, -1, np.int64)
    touched = np.empty(cap, np.int64)
    ntouched = 0
    hcap = 1 << 16 if keep_history else 1
    hist = (np.empty(hcap, np.int64), np.empty(hcap, np.int32), np.empty((hcap, MC), np.int16),
            np.empty((hcap, MC), np.int8), np.empty((hcap, MC), np.int8), np.empty((hcap, M), np.int8))
    n_rec = 0

    # Option lists per pile: [0] reused for piles without a new vehicle, [1] per decision.
    OPTCAP = 1
    for i in range(1, C + 1):
        OPTCAP = OPTCAP * (N + i) // i
    OPTCAP += 1
    o_cost = np.zeros((2, M, OPTCAP))
    o_q = np.zeros((2, M, OPTCAP, C), np.int64)
    o_order = np.zeros((2, M, OPTCAP), np.int64)
    o_n = np.zeros((2, M), np.int64)
    o_src = np.zeros(M, np.int64)
    cached = np.zeros(M, np.bool_)
    codes = np.zeros((2, M, C), np.int64)
    ncodes = np.zeros((2, M), np.int64)

    # Per-state scratch.
    is_conn = np.zeros(J, np.bool_)
    cterm = np.zeros((MC, N + 1))  # connected i: bound term after q modules (37)
    cx2 = np.zeros((MC, N + 1))
    cdep = np.zeros((MC, N + 1), np.bool_)
    cneed = np.zeros(MC, np.int64)
    wterm = np.zeros((J, N + 1))  # waiting w: the same if connected now
    wx2 = np.zeros((J, N + 1))
    wdep = np.zeros((J, N + 1), np.bool_)
    wneed = np.zeros(J, np.int64)
    wveh = np.zeros(J, np.int64)
    wU = np.zeros(J)
    wdelta = np.zeros(J)
    optional = np.zeros(J, np.int64)
    pref = np.zeros(J + 1)
    mand = np.zeros(J, np.int64)
    on_pile = np.zeros(M, np.int64)
    members = np.zeros((M, C), np.int64)
    freec = np.zeros(M, np.int64)
    chosen = np.zeros(MC, np.int64)
    assign = np.zeros(MC, np.int64)
    used = np.zeros(M, np.int64)
    comb = np.zeros(MC + 1, np.int64)
    psum = np.zeros(MC + 1)
    sufmin = np.zeros(M + 1)
    partial = np.zeros(M + 1)
    pos = np.zeros(M, np.int64)
    pick = np.zeros(M, np.int64)
    # child scratch
    done_row = np.zeros(NW, np.int64)
    c_veh = np.zeros(MC, np.int64)
    c_pile = np.zeros(MC, np.int64)
    c_x = np.zeros(MC)
    r_veh = np.zeros(MC, np.int64)
    r_q = np.zeros(MC, np.int64)
    r_pile = np.zeros(MC, np.int64)
    pmin = np.zeros(M, np.int64)
    pile_map = np.zeros(M, np.int64)

    # Root: OPTIMIZE vehicles connected on their physical piles.
    n0 = init_veh.shape[0]
    f0 = g0 - tail[0]
    for i in range(n0):
        S_veh[0, 0, i], S_pile[0, 0, i], S_x[0, 0, i] = init_veh[i], init_pile[i], init_x[i]
        is_conn[init_veh[i]] = True
        f0 += table_value(off, cnt, bp, val, init_veh[i], 0, init_x[i])
    for j in range(J):
        if not is_conn[j]:
            f0 += U[j, 0]
        is_conn[j] = False
    S_n[0, 0], S_g[0, 0], S_f[0, 0], S_alive[0, 0] = n0, g0, f0, True
    nL = np.zeros(2, np.int64)
    nL[0] = 1

    v = float(threshold)
    best_rec = -1
    status = COMPLETE
    beta = INF
    adm = INF
    peak, expanded, stored = 1, 0, 0
    cur = 0
    for k in range(K):
        nx = 1 - cur
        nL[nx] = 0
        for t in range(ntouched):
            table[touched[t]] = -1
        ntouched = 0
        if beam_width > 0:
            adm = INF  # a beam cuts per layer; the cuts are recorded in beta
        sym = M > 1 and k >= symmetric_from  # identical piles at slot k
        sym_next = M > 1 and k + 1 >= symmetric_from
        any_alive = False
        for s in range(nL[cur]):
            if not S_alive[cur, s] or S_f[cur, s] >= adm:
                continue
            any_alive = True
            expanded += 1
            stop = max_expand > 0 and expanded > max_expand
            if not stop and deadline > 0.0 and (expanded & 1023) == 0:
                with objmode(now="float64"):
                    now = time.perf_counter()
                stop = now > deadline
            if stop:
                # Every unexpanded state is still open: their least bound is certified.
                open_min = adm
                for e in range(s, nL[cur]):
                    if S_alive[cur, e] and S_f[cur, e] < open_min:
                        open_min = S_f[cur, e]
                for e in range(nL[nx]):
                    if S_alive[nx, e] and S_f[nx, e] < open_min:
                        open_min = S_f[nx, e]
                beta = min(beta, open_min)
                status = BUDGET if max_expand > 0 and expanded > max_expand else TIME_LIMIT
                return v, best_rec, status, beta, peak, expanded, stored, hist, n_rec

            g = S_g[cur, s]
            nc = np.int64(S_n[cur, s])
            for i in range(nc):
                is_conn[S_veh[cur, s, i]] = True
            # --- unconnected vehicles. Waiting slot t = 0..n_wait-1 holds an
            # arrived vehicle: *mandatory* if it could not finish when connected
            # later (U = INF), else *optional* (its U goes into usum). A vehicle
            # not yet arrived only contributes its U.
            usum = 0.0
            n_wait, nw, nm = 0, 0, 0
            dead = False
            for j in range(J):
                if (S_done[cur, s, j // BITS] >> (j % BITS)) & 1 or is_conn[j]:
                    continue
                u = U[j, max(k + 1, release[j])]
                if release[j] <= k:
                    wveh[n_wait] = j
                    if u >= INF / 2:
                        wU[n_wait] = 0.0
                        mand[nm] = n_wait
                        nm += 1
                    else:
                        wU[n_wait] = u
                        optional[nw] = n_wait
                        nw += 1
                        usum += u
                    n_wait += 1
                elif u >= INF / 2:
                    dead = True  # cannot finish even if connected on arrival
                else:
                    usum += u
            for i in range(nc):
                is_conn[S_veh[cur, s, i]] = False
            for m in range(M):
                on_pile[m] = 0
                cached[m] = False
            for i in range(nc):
                m = np.int64(S_pile[cur, s, i])
                members[m, on_pile[m]] = i
                on_pile[m] += 1
            n_free = 0
            for m in range(M):
                freec[m] = free_conn[m, k] - on_pile[m]
                if freec[m] < 0:
                    dead = True
                n_free += freec[m]
            if dead or nm > n_free:
                continue

            # --- bound terms (37) of connected vehicles, for every useful q
            phi0 = g - tail[k + 1] + usum
            for i in range(nc):
                j = np.int64(S_veh[cur, s, i])
                x = S_x[cur, s, i]
                cneed[i] = useful_modules(x, N, kappa, params[j])
                best_t = INF
                for q in range(cneed[i] + 1):
                    x2 = slot_energy(x, q, kappa, params[j])
                    cx2[i, q] = x2
                    cdep[i, q] = x2 >= params[j, P_W]
                    cterm[i, q] = weight[j] * (k + 1) if cdep[i, q] else table_value(off, cnt, bp, val, j, k + 1, x2)
                    best_t = min(best_t, cterm[i, q])
                phi0 += best_t
            # --- the same for waiting vehicles if connected now (q >= 1); delta is
            # the least increase of the bound when connected (39). Mandatory
            # vehicles have wU = 0, so their delta is their whole term.
            for t in range(n_wait):
                j = wveh[t]
                wneed[t] = useful_modules(0.0, N, kappa, params[j])
                best_t = INF
                for q in range(1, wneed[t] + 1):
                    x2 = slot_energy(0.0, q, kappa, params[j])
                    wx2[t, q] = x2
                    wdep[t, q] = x2 >= params[j, P_W]
                    wterm[t, q] = weight[j] * (k + 1) if wdep[t, q] else table_value(off, cnt, bp, val, j, k + 1, x2)
                    best_t = min(best_t, wterm[t, q])
                wdelta[t] = best_t - wU[t]
            for t in range(nm):
                phi0 += wdelta[mand[t]]
            # optional vehicles sorted by their least bound increase delta (Lemma 2(a))
            for a in range(1, nw):
                b_ = optional[a]
                c_ = a - 1
                while c_ >= 0 and wdelta[optional[c_]] > wdelta[b_]:
                    optional[c_ + 1] = optional[c_]
                    c_ -= 1
                optional[c_ + 1] = b_
            pref[0] = 0.0
            for a in range(nw):
                pref[a + 1] = pref[a] + wdelta[optional[a]]
            # (No test of phi0 alone here: delta can be negative -- connecting
            # now can beat waiting -- so phi0 is not a bound on the successors.)

            # --- connection sets: mandatory vehicles + r optional ones
            r_max = min(n_free - nm, nw)
            for r in range(r_max + 1):
                d = 0
                comb[0] = 0
                psum[0] = 0.0
                first = True
                while True:
                    # Next combination comb[0..r-1] (indices into the sorted optional list)
                    # whose bound (40) passes; prefixes that fail are skipped whole.
                    found = False
                    if r == 0:
                        found = first
                        first = False
                    else:
                        while d >= 0:
                            if comb[d] > nw - r + d:
                                d -= 1
                                if d >= 0:
                                    comb[d] += 1
                                continue
                            part = psum[d] + wdelta[optional[comb[d]]]
                            a, b_ = comb[d] + 1, comb[d] + r - d - 1
                            if not _keep(phi0 + part + pref[b_ + 1] - pref[a], v, adm):
                                d -= 1
                                if d >= 0:
                                    comb[d] += 1
                                continue
                            if d == r - 1:
                                psum[r] = part
                                found = True
                                break
                            psum[d + 1] = part
                            d += 1
                            comb[d] = comb[d - 1] + 1
                    if not found:
                        break
                    nch = 0
                    base = g - tail[k + 1] + usum
                    for t in range(nm):
                        chosen[nch] = mand[t]
                        nch += 1
                    for t in range(r):
                        chosen[nch] = optional[comb[t]]
                        base -= wU[optional[comb[t]]]
                        nch += 1
                    if r > 0:
                        comb[r - 1] += 1
                        d = r - 1

                    # --- every assignment of the chosen vehicles to piles with a free connector
                    for t in range(nch):
                        assign[t] = 0
                    first_assign = True
                    while True:
                        if not first_assign:
                            t = 0
                            while t < nch:
                                assign[t] += 1
                                if assign[t] < M:
                                    break
                                assign[t] = 0
                                t += 1
                            if t >= nch:
                                break
                        first_assign = False
                        for m in range(M):
                            used[m] = 0
                        ok = True
                        for t in range(nch):
                            used[assign[t]] += 1
                            if used[assign[t]] > freec[assign[t]]:
                                ok = False
                        if ok and sym:
                            # identical empty piles: fill the lowest-numbered first
                            for m in range(M):
                                if on_pile[m] == 0 and used[m] > 0:
                                    for m2 in range(m):
                                        if on_pile[m2] == 0 and used[m2] == 0:
                                            ok = False
                        if not ok:
                            continue

                        # --- module vectors per pile (Lemma 1), sorted (Lemma 2(b))
                        for m in range(M):
                            if used[m] == 0:
                                if not cached[m]:
                                    for t in range(on_pile[m]):
                                        codes[0, m, t] = members[m, t]
                                    ncodes[0, m] = on_pile[m]
                                    o_n[0, m] = _pile_options(codes[0, m], on_pile[m], cterm, cneed, wterm, wneed,
                                                              free_mod[m, k], o_cost[0, m], o_q[0, m], o_order[0, m])
                                    cached[m] = True
                                o_src[m] = 0
                            else:
                                nmem = on_pile[m]
                                for t in range(nmem):
                                    codes[1, m, t] = members[m, t]
                                for t in range(nch):
                                    if assign[t] == m:
                                        codes[1, m, nmem] = -chosen[t] - 1
                                        nmem += 1
                                ncodes[1, m] = nmem
                                o_n[1, m] = _pile_options(codes[1, m], nmem, cterm, cneed, wterm, wneed,
                                                          free_mod[m, k], o_cost[1, m], o_q[1, m], o_order[1, m])
                                o_src[m] = 1
                        ok = True
                        sufmin[M] = 0.0
                        for m in range(M - 1, -1, -1):
                            if o_n[o_src[m], m] == 0:
                                ok = False
                                break
                            sufmin[m] = sufmin[m + 1] + o_cost[o_src[m], m, o_order[o_src[m], m, 0]]
                        if not ok or not _keep(base + sufmin[0], v, adm):
                            continue

                        # --- depth-first product of the piles' vectors
                        depth = 0
                        pos[0] = 0
                        partial[0] = base
                        while depth >= 0:
                            sm = o_src[depth]
                            if pos[depth] >= o_n[sm, depth]:
                                depth -= 1
                                if depth >= 0:
                                    pos[depth] += 1
                                continue
                            o = o_order[sm, depth, pos[depth]]
                            fval = partial[depth] + o_cost[sm, depth, o]
                            if not _keep(fval + sufmin[depth + 1], v, adm):
                                depth -= 1  # sorted: every later vector of this pile is worse
                                if depth >= 0:
                                    pos[depth] += 1
                                continue
                            pick[depth] = o
                            if depth < M - 1:
                                partial[depth + 1] = fval
                                depth += 1
                                pos[depth] = 0
                                continue
                            pos[depth] += 1

                            # ---------- build the successor (bound fval) ----------
                            g2 = g
                            for w in range(NW):
                                done_row[w] = S_done[cur, s, w]
                            nd2 = S_ndone[cur, s]
                            n3, nr = 0, 0
                            for m in range(M):
                                sm2 = o_src[m]
                                for t in range(ncodes[sm2, m]):
                                    c = codes[sm2, m, t]
                                    q = o_q[sm2, m, pick[m], t]
                                    if c >= 0:
                                        j, dep, x2 = np.int64(S_veh[cur, s, c]), cdep[c, q], cx2[c, q]
                                    else:
                                        j, dep, x2 = wveh[-c - 1], wdep[-c - 1, q], wx2[-c - 1, q]
                                    r_veh[nr], r_q[nr], r_pile[nr] = j, q, m
                                    nr += 1
                                    if dep:
                                        g2 += weight[j] * (k + 1)
                                        done_row[j // BITS] |= np.int64(1) << (j % BITS)
                                        nd2 += 1
                                    else:
                                        c_veh[n3], c_pile[n3], c_x[n3] = j, m, x2
                                        n3 += 1
                            if nd2 == J:  # every vehicle has departed: a complete schedule
                                if g2 < v:
                                    v = g2
                                    if keep_history:
                                        if n_rec == hist[0].shape[0]:
                                            hist = _grow_history(hist, n_rec, MC, M)
                                        best_rec = _record(hist, n_rec, S_hid[cur, s], k, r_veh, r_q, r_pile, nr, M, pile_map, False)
                                        n_rec += 1
                                continue
                            # canonical pile labels (identical piles only)
                            for m in range(M):
                                pile_map[m] = m
                            if sym_next:
                                for m in range(M):
                                    pmin[m] = 1 << 40
                                for t in range(n3):
                                    if c_veh[t] < pmin[c_pile[t]]:
                                        pmin[c_pile[t]] = c_veh[t]
                                order = np.argsort(pmin, kind="mergesort")
                                for t in range(M):
                                    pile_map[order[t]] = t
                                for t in range(n3):
                                    c_pile[t] = pile_map[c_pile[t]]
                            for a in range(1, n3):  # sort entries by (pile, vehicle)
                                kv, kp, kx = c_veh[a], c_pile[a], c_x[a]
                                b2 = a - 1
                                while b2 >= 0 and (c_pile[b2] > kp or (c_pile[b2] == kp and c_veh[b2] > kv)):
                                    c_veh[b2 + 1], c_pile[b2 + 1], c_x[b2 + 1] = c_veh[b2], c_pile[b2], c_x[b2]
                                    b2 -= 1
                                c_veh[b2 + 1], c_pile[b2 + 1], c_x[b2 + 1] = kv, kp, kx
                            key = _hash(done_row, NW, c_veh, c_pile, n3)

                            # dominance against stored states with the same signature (Proposition 6)
                            dominated = False
                            e = table[key & (TS - 1)]
                            scanned = 0
                            while e >= 0 and not dominated:
                                scanned += 1
                                if scan_cap > 0 and scanned > scan_cap:
                                    break
                                if S_key[nx, e] == key and S_alive[nx, e] and _same_signature(L, nx, e, done_row, NW, c_veh, c_pile, n3):
                                    e_better = S_g[nx, e] <= g2
                                    new_better = g2 <= S_g[nx, e]
                                    for i in range(n3):
                                        if S_x[nx, e, i] < c_x[i]:
                                            e_better = False
                                        if S_x[nx, e, i] > c_x[i]:
                                            new_better = False
                                    if e_better:
                                        dominated = True
                                    elif new_better:
                                        S_alive[nx, e] = False
                                e = S_next[nx, e]
                            if dominated:
                                continue

                            # memory cut: drop the worse half, remember the threshold
                            if nL[nx] >= cap:
                                n_alive = _compact(L, nx, nL[nx])
                                if n_alive > (3 * cap) // 4:
                                    fs = np.sort(S_f[nx, :n_alive])
                                    adm = min(adm, fs[cap // 2])
                                    beta = min(beta, adm)
                                    status = max(status, MEMORY_CUT)
                                    for e2 in range(n_alive):
                                        if S_f[nx, e2] >= adm:
                                            S_alive[nx, e2] = False
                                    n_alive = _compact(L, nx, n_alive)
                                nL[nx] = n_alive
                                ntouched = _rehash(L, nx, n_alive, table, touched, ntouched)
                                if fval >= adm:
                                    continue

                            rec = -1
                            if keep_history:
                                if n_rec == hist[0].shape[0]:
                                    hist = _grow_history(hist, n_rec, MC, M)
                                rec = _record(hist, n_rec, S_hid[cur, s], k, r_veh, r_q, r_pile, nr, M, pile_map, sym_next)
                                n_rec += 1
                            e = nL[nx]
                            nL[nx] += 1
                            for w in range(NW):
                                S_done[nx, e, w] = done_row[w]
                            for i in range(n3):
                                S_veh[nx, e, i], S_pile[nx, e, i], S_x[nx, e, i] = c_veh[i], c_pile[i], c_x[i]
                            S_n[nx, e], S_ndone[nx, e], S_g[nx, e], S_f[nx, e] = n3, nd2, g2, fval
                            S_hid[nx, e], S_alive[nx, e], S_key[nx, e] = rec, True, key
                            slot = key & (TS - 1)
                            if table[slot] < 0:
                                touched[ntouched] = slot
                                ntouched += 1
                            S_next[nx, e] = table[slot]
                            table[slot] = e
                            stored += 1

        # --- end of slot k
        n_alive = 0
        for e in range(nL[nx]):
            if S_alive[nx, e]:
                n_alive += 1
        if beam_width > 0 and n_alive > beam_width:
            fs = np.full(nL[nx], np.inf)
            for e in range(nL[nx]):
                if S_alive[nx, e]:
                    fs[e] = S_f[nx, e]
            order = np.argsort(fs, kind="mergesort")
            beta = min(beta, fs[order[beam_width]])
            status = max(status, MEMORY_CUT)
            for t in range(beam_width, nL[nx]):
                S_alive[nx, order[t]] = False
            n_alive = beam_width
        peak = max(peak, n_alive)
        cur = nx
        if not any_alive:
            break
    return v, best_rec, status, beta, peak, expanded, stored, hist, n_rec


@njit(cache=True)
def _same_signature(L, b, e, done_row, NW, c_veh, c_pile, n3):
    """Hash-collision guard: same departed set and same connected (vehicle, pile) list."""
    S_done, S_veh, S_pile, S_n = L[0], L[1], L[2], L[4]
    if S_n[b, e] != n3:
        return False
    for w in range(NW):
        if S_done[b, e, w] != done_row[w]:
            return False
    for i in range(n3):
        if S_veh[b, e, i] != c_veh[i] or S_pile[b, e, i] != c_pile[i]:
            return False
    return True


@njit(cache=True)
def _record(hist, n, parent, k, r_veh, r_q, r_pile, nr, M, pile_map, relabel):
    """
    History record ``n``: the transition over slot ``k`` from state record
    ``parent``. Vehicles carry their pile label in the *parent's* frame;
    ``perm[m]`` is the label pile ``m`` gets in the child (canonical relabelling).
    """
    hpar, hk, hv, hq, hp, hperm = hist
    hpar[n], hk[n] = parent, k
    MC = hv.shape[1]
    for i in range(MC):
        hv[n, i] = r_veh[i] if i < nr else -1
        hq[n, i] = r_q[i] if i < nr else 0
        hp[n, i] = r_pile[i] if i < nr else -1
    for m in range(M):
        hperm[n, m] = pile_map[m] if relabel else m
    return n


# --------------------------------------------------------------------------- #
# Python side
# --------------------------------------------------------------------------- #


@dataclass
class PassResult:
    threshold: int
    beam_width: int
    value: float  # best schedule value found (slots), == threshold if none
    status: int
    beta: float  # least bound given up (INF if nothing)
    certified: int  # min(T, v, ceil(beta)): no schedule is below this
    peak: int
    expanded: int
    stored: int
    runtime: float
    schedule: dict[int, tuple[int, int, list[int]]] | None  # search index -> (pile, start, modules)

    @property
    def found(self) -> bool:
        return self.value < self.threshold

    @property
    def status_name(self) -> str:
        return STATUS_NAMES[self.status]


def run_search_pass(
    inst: SearchInstance,
    tables: Tables,
    threshold: int,
    *,
    beam_width: int = 0,
    layer_cap: int = 2_000_000,
    keep_history: bool = True,
    scan_cap: int = 1024,
    max_expand: int = 0,
    deadline: float | None = None,
) -> PassResult:
    """Run one pass and decode its result (``schedule`` only with ``keep_history``)."""
    t0 = time.perf_counter()
    st = inst.station
    init = np.flatnonzero(inst.start_pile >= 0)
    order = init[np.lexsort((init, inst.start_pile[init]))]  # stored sorted by (pile, vehicle)
    v, rec, status, beta, peak, expanded, stored, hist, _n_rec = run_pass(
        inst.J, st.n_piles, st.n_connectors, st.n_modules, inst.K, inst.kappa,
        inst.release, inst.weight, inst.params, tables.off, tables.cnt, tables.bp, tables.val, tables.U, tables.tail,
        inst.free_connectors, inst.free_modules, inst.symmetric_from,
        order.astype(np.int64), inst.start_pile[order].astype(np.int64), inst.start_energy[order],
        float(inst.fixed_departure_total), int(threshold), int(beam_width), int(layer_cap), bool(keep_history),
        int(scan_cap), int(max_expand), float(deadline) if deadline is not None else -1.0,
    )
    certified = min(int(threshold), int(v), int(np.ceil(beta)) if beta < INF / 2 else int(threshold))
    schedule = _decode(hist, rec, st.n_piles) if (keep_history and rec >= 0) else None
    return PassResult(
        threshold=int(threshold), beam_width=int(beam_width), value=float(v), status=int(status), beta=float(beta),
        certified=certified, peak=int(peak), expanded=int(expanded), stored=int(stored),
        runtime=time.perf_counter() - t0, schedule=schedule,
    )


def _decode(hist, rec: int, M: int) -> dict[int, tuple[int, int, list[int]]]:
    """Follow the history records from the final transition back to the root."""
    hpar, hk, hv, hq, hp, hperm = hist
    chain = []
    while rec >= 0:
        chain.append(rec)
        rec = int(hpar[rec])
    phys = list(range(M))  # label in the current frame -> physical pile
    out: dict[int, list] = {}
    for r in reversed(chain):
        k = int(hk[r])
        for t in range(hv.shape[1]):
            j = int(hv[r, t])
            if j < 0:
                break
            if j not in out:
                out[j] = [phys[int(hp[r, t])], k, []]
            out[j][2].append(int(hq[r, t]))
        new = [0] * M
        for lab in range(M):
            new[int(hperm[r, lab])] = phys[lab]
        phys = new
    return {j: (p, S, q) for j, (p, S, q) in out.items()}
