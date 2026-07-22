"""
Compare simulation metrics to classic queueing approximations.

What this script checks (side by side):

1. Little's law (system):   L ~= lambda_eff * W
   - L from time-average occupancy in MetricsTracker
   - lambda_eff = (# finished EVs) / T
   - W = mean sojourn time of finished EVs (departure - arrival)

2. Little's law (queue):    Q ~= lambda_eff * W_q
   - Q from time-average queue length
   - W_q = mean wait before service starts (service_start - arrival)

3. Utilization:             rho ~= lambda_eff / (c * mu)
   - c = n_piles * n_nozzles (servers ~= nozzles)
   - mu = 1 / E[S], S = charge time of finished EVs
   - Simulated rho = mean fraction of nozzle-minutes busy

4. Kingman's G/G/c approx for mean queue wait:
   W_q ~= ((c_a^2 + c_s^2) / 2) * W_q(M/M/c)
   with c_a^2 = 1 (exponential arrivals), c_s^2 = Var(S)/E[S]^2 from finished EVs,
   and W_q(M/M/c) from the Erlang-C formula.

The single-queue multi-pile dispatch (HOL EV sent to the least-occupied pile)
is only approximately G/G/c, so Kingman is a rough benchmark, not an exact
identity like Little's law.

Run from the repo root (prints need -s):

    python -m pytest tests/test_queueing_laws.py -s -v

Or run this file directly:

    python tests/test_queueing_laws.py

In a notebook:

    !python -m pytest tests/test_queueing_laws.py -s -v
"""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from env.charging_env import ChargingStationEnv
from policy.queue.fifo import FIFOQueuePolicy
from config import MAX_TIME


# Station setup: few nozzles + fast arrivals so the queue is not always empty.
N_PILES = 2
N_NOZZLES = 1
N_BRICKS = 4
P_BRICK = 25.0
QUEUE_CAPACITY = 50
LAM = 1.0  # mean inter-arrival (minutes); arrival rate = 1/LAM
SEED = 42
POLICY_SEED = 1


def _run_fifo_episode(
    n_piles: int = N_PILES,
    n_nozzles: int = N_NOZZLES,
    n_bricks: int = N_BRICKS,
    p_brick: float = P_BRICK,
    queue_capacity: int = QUEUE_CAPACITY,
    lam: float = LAM,
    seed: int = SEED,
) -> ChargingStationEnv:
    """Roll out one full day under FIFO pile choice."""
    env = ChargingStationEnv(
        n_piles=n_piles,
        n_nozzles=n_nozzles,
        n_bricks=n_bricks,
        p_brick=p_brick,
        queue_capacity=queue_capacity,
        lam=lam,
    )
    obs, _ = env.reset(seed=seed)
    rng = np.random.default_rng(POLICY_SEED)
    policy = FIFOQueuePolicy()
    done = bool(env.engine.terminated)
    steps = 0
    while not done and steps < 50000:
        if not env.engine.needs_assignment_decision():
            break
        mask = env.action_masks()
        action = policy.select_pile(obs, mask, rng)
        obs, _, done, _, _ = env.step(action)
        steps += 1
    return env


def _finished_times(env: ChargingStationEnv):
    finished = env.engine.metrics.finished_evs
    waits = []
    services = []
    sojourns = []
    for ev in finished:
        if ev.service_start_time is None:
            continue
        if not np.isfinite(ev.departure_time):
            continue
        waits.append(ev.service_start_time - ev.arrival_time)
        services.append(ev.departure_time - ev.service_start_time)
        sojourns.append(ev.departure_time - ev.arrival_time)
    return (
        np.asarray(waits, dtype=float),
        np.asarray(services, dtype=float),
        np.asarray(sojourns, dtype=float),
    )


def erlang_c_wait(lam: float, mu: float, c: int) -> float:
    """
    Mean waiting time in queue for an M/M/c system (Erlang-C).

    Returns inf if the system is unstable (rho >= 1).
    """
    if lam <= 0 or mu <= 0 or c < 1:
        return float("nan")
    a = lam / mu  # offered load in Erlangs
    rho = a / c
    if rho >= 1.0 - 1e-12:
        return float("inf")

    sum_terms = sum(a**k / factorial(k) for k in range(c))
    last = (a**c / factorial(c)) * (1.0 / (1.0 - rho))
    p0 = 1.0 / (sum_terms + last)
    c_prob = last * p0  # probability of delay
    return c_prob / (c * mu - lam)


def kingman_ggc_wait(lam: float, mu: float, c: int, ca2: float, cs2: float) -> float:
    """Kingman / VUT approximation: ((ca^2 + cs^2) / 2) * W_q(M/M/c)."""
    wq_mm = erlang_c_wait(lam, mu, c)
    if not np.isfinite(wq_mm):
        return wq_mm
    return 0.5 * (ca2 + cs2) * wq_mm


def _side_by_side(name: str, theory: float, sim: float, unit: str = "") -> dict:
    """Print one comparison row and return a small result dict."""
    if theory is None or sim is None or not np.isfinite(theory) or not np.isfinite(sim):
        abs_diff = float("nan")
        rel = float("nan")
        print(f"  {name:32s}  theory={theory}  sim={sim}  (skipped)")
    else:
        abs_diff = abs(theory - sim)
        denom = max(abs(theory), abs(sim), 1e-12)
        rel = abs_diff / denom
        u = f" {unit}" if unit else ""
        print(
            f"  {name:32s}  theory={theory:10.4f}{u}  "
            f"sim={sim:10.4f}{u}  |diff|={abs_diff:8.4f}  rel={rel:6.2%}"
        )
    return {"name": name, "theory": theory, "sim": sim, "abs_diff": abs_diff, "rel": rel}


def test_queueing_laws_side_by_side():
    """Run one FIFO day and compare L, Q, utilization, and Kingman W_q."""
    print("\n=== test_queueing_laws_side_by_side ===")
    print(
        "Intent: after one FIFO simulated day, compare Little's law L and Q, "
        "nozzle utilization, and Kingman W_q against the same run's empirical averages."
    )
    c = N_PILES * N_NOZZLES
    lam_offered = 1.0 / LAM
    print(
        f"Setup: {N_PILES} piles x {N_NOZZLES} nozzles "
        f"(c={c} servers), mean interarrival={LAM} min "
        f"(lambda_offered={lam_offered:.4f}/min), T={MAX_TIME} min."
    )

    env = _run_fifo_episode()
    metrics = env.engine.metrics
    T = float(env.engine.current_time)
    waits, services, sojourns = _finished_times(env)

    n_arrived = len(metrics.arrived_evs)
    n_finished = len(metrics.finished_evs)
    n_dropped = len(metrics.dropped_evs)

    print(
        f"  Episode done: t={T:.1f}, arrived={n_arrived}, "
        f"finished={n_finished}, dropped={n_dropped}"
    )

    assert n_finished > 10, "Need enough finished EVs for meaningful averages"

    L_sim = float(metrics.average_L())
    Q_sim = float(metrics.average_Q())
    nozzle_util = metrics.nozzle_utilization(T)
    rho_sim = float(np.mean(nozzle_util))

    W = float(np.mean(sojourns))
    W_q = float(np.mean(waits))
    S = float(np.mean(services))
    var_S = float(np.var(services, ddof=1)) if len(services) > 1 else 0.0
    cs2 = var_S / (S**2) if S > 0 else float("nan")
    ca2 = 1.0  # exponential inter-arrivals

    lam_eff = n_finished / T
    mu = 1.0 / S if S > 0 else float("nan")
    rho_theory = lam_eff / (c * mu) if np.isfinite(mu) else float("nan")

    L_ll = lam_eff * W
    Q_ll = lam_eff * W_q
    W_q_kingman = kingman_ggc_wait(lam_eff, mu, c, ca2, cs2)

    print("\n  --- Inputs estimated from the run ---")
    print(f"  lambda_offered           = {lam_offered:.6f} /min")
    print(f"  lambda_eff (finished/T)  = {lam_eff:.6f} /min")
    print(f"  E[W]  sojourn            = {W:.4f} min")
    print(f"  E[W_q] queue wait        = {W_q:.4f} min")
    print(f"  E[S]  service            = {S:.4f} min  (mu={mu:.6f})")
    print(f"  c_a^2 (arrivals)         = {ca2:.4f}")
    print(f"  c_s^2 (service)          = {cs2:.4f}")
    print(f"  rho_theory=lambda_eff/(c*mu) = {rho_theory:.4f}")
    print(f"  rho_sim (nozzle busy)    = {rho_sim:.4f}")

    print("\n  --- Side-by-side: theory vs simulation ---")
    r_L = _side_by_side("Little L = lambda_eff * W", L_ll, L_sim, "cars")
    r_Q = _side_by_side("Little Q = lambda_eff * W_q", Q_ll, Q_sim, "cars")
    r_rho = _side_by_side("Utilization rho", rho_theory, rho_sim, "")
    r_k = _side_by_side("Kingman W_q", W_q_kingman, W_q, "min")

    assert r_L["rel"] < 0.15 or r_L["abs_diff"] < 0.25, (
        f"Little's law L mismatch: theory={L_ll}, sim={L_sim}"
    )
    assert r_Q["rel"] < 0.20 or r_Q["abs_diff"] < 0.25, (
        f"Little's law Q mismatch: theory={Q_ll}, sim={Q_sim}"
    )
    assert r_rho["rel"] < 0.35 or r_rho["abs_diff"] < 0.15, (
        f"Utilization mismatch: theory={rho_theory}, sim={rho_sim}"
    )

    if np.isfinite(W_q_kingman) and W_q_kingman != float("inf"):
        print(
            "  Note: Kingman is approximate here; a large gap is informative, "
            "not necessarily a bug."
        )
        if W_q < 1e-6 and W_q_kingman > 10:
            pytest.fail(
                f"Kingman W_q={W_q_kingman} but simulated wait is ~0; check load params"
            )
    else:
        print("  Kingman W_q is non-finite (system near/over capacity under M/M/c).")

    print("  PASS (Little's law + utilization checks)")


if __name__ == "__main__":
    print("Running test_queueing_laws.py (direct mode)")
    test_queueing_laws_side_by_side()
    print("\nAll tests in test_queueing_laws.py finished.")
