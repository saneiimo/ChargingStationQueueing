"""
Compare simulation metrics to classic queueing approximations.

Customer averages (W, W_q, S) and Little's-law algebra come from
``MetricsTracker.queueing_summary`` / ``finished_time_arrays``. This test
only adds the Kingman G/G/c benchmark and tolerance asserts.

What this script checks (side by side):

1. Little's law (system):   L ~= lambda_eff * W
2. Little's law (queue):    Q ~= lambda_eff * W_q
3. Utilization:             rho ~= lambda_eff / (c * mu)
4. Kingman's G/G/c approx for mean queue wait (rough benchmark only)

Run from the repo root (prints need -s):

    python -m pytest tests/test_queueing_laws.py -s -v
"""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from env.charging_env import ChargingStationEnv
from policy.queue.fifo import FIFOQueuePolicy
from config import MAX_TIME


# Station setup: few connectors + fast arrivals so the queue is not always empty.
N_PILES = 2
N_CONNECTORS = 1
N_MODULES = 4
P_MODULE = 25.0
QUEUE_CAPACITY = 50
LAM = 1.0  # mean inter-arrival (minutes); arrival rate λ = 1/LAM
SEED = 42
POLICY_SEED = 1


def _run_fifo_episode(
    n_piles: int = N_PILES,
    n_connectors: int = N_CONNECTORS,
    n_modules: int = N_MODULES,
    p_module: float = P_MODULE,
    queue_capacity: int = QUEUE_CAPACITY,
    mean_interarrival: float = LAM,
    seed: int = SEED,
) -> ChargingStationEnv:
    """Roll out one full day under FIFO pile choice."""
    env = ChargingStationEnv(
        n_piles=n_piles,
        n_connectors=n_connectors,
        n_modules=n_modules,
        p_module=p_module,
        queue_capacity=queue_capacity,
        mean_interarrival=mean_interarrival,
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
        ev, pile_id = policy.decide(obs, mask, rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile_id, ev=ev)
        steps += 1
    return env


def _finished_times(env: ChargingStationEnv):
    """Delegate to ``MetricsTracker.finished_time_arrays`` (single source of truth)."""
    return env.engine.metrics.finished_time_arrays()


def erlang_c_wait(lam: float, mu: float, c: int) -> float:
    """
    Mean waiting time in queue for an M/M/c system (Erlang-C).

    ``lam`` here is the arrival **rate** λ (1 / mean_interarrival), not the
    mean gap. Returns inf if the system is unstable (rho >= 1).
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
    return {
        "name": name,
        "theory": theory,
        "sim": sim,
        "abs_diff": abs_diff,
        "rel": rel,
    }


def test_queueing_laws_side_by_side():
    """Run one FIFO day and compare L, Q, utilization, and Kingman W_q."""
    print("\n=== test_queueing_laws_side_by_side ===")
    print(
        "Intent: after one FIFO simulated day, compare Little's law L and Q, "
        "connector utilization, and Kingman W_q against the same run's empirical averages."
    )
    c = N_PILES * N_CONNECTORS
    lam_offered = 1.0 / LAM
    print(
        f"Setup: {N_PILES} piles x {N_CONNECTORS} connectors "
        f"(c={c} servers), mean interarrival={LAM} min "
        f"(lambda_offered={lam_offered:.4f}/min), T={MAX_TIME} min."
    )

    env = _run_fifo_episode()
    metrics = env.engine.metrics
    T = float(env.engine.current_time)
    summary = metrics.queueing_summary(T, n_servers=N_PILES * N_CONNECTORS)
    _waits, services, _sojourns = metrics.finished_time_arrays()

    n_arrived = len(metrics.arrived_evs)
    n_finished = len(metrics.finished_evs)
    n_dropped = len(metrics.dropped_evs)

    print(
        f"  Episode done: t={T:.1f}, arrived={n_arrived}, "
        f"finished={n_finished}, dropped={n_dropped}"
    )

    assert n_finished > 10, "Need enough finished EVs for meaningful averages"

    L_sim = summary["L_sim"]
    Q_sim = summary["Q_sim"]
    rho_sim = summary["rho_sim"]

    W = summary["W"]
    W_q = summary["W_q"]
    S = summary["S"]
    cs2 = summary["c_s2"]
    ca2 = 1.0  # exponential inter-arrivals

    lam_eff = summary["lambda_eff"]
    mu = summary["mu"]
    rho_theory = summary["rho_theory"]

    L_ll = summary["L_theory"]
    Q_ll = summary["Q_theory"]
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
    print(f"  rho_sim (connector busy)    = {rho_sim:.4f}")

    print("\n  --- Side-by-side: theory vs simulation ---")
    r_L = _side_by_side("Little L = lambda_eff * W", L_ll, L_sim, "cars")
    r_Q = _side_by_side("Little Q = lambda_eff * W_q", Q_ll, Q_sim, "cars")
    r_rho = _side_by_side("Utilization rho", rho_theory, rho_sim, "")
    r_k = _side_by_side("Kingman W_q", W_q_kingman, W_q, "min")

    assert (
        r_L["rel"] < 0.15 or r_L["abs_diff"] < 0.25
    ), f"Little's law L mismatch: theory={L_ll}, sim={L_sim}"
    assert (
        r_Q["rel"] < 0.20 or r_Q["abs_diff"] < 0.25
    ), f"Little's law Q mismatch: theory={Q_ll}, sim={Q_sim}"
    assert (
        r_rho["rel"] < 0.35 or r_rho["abs_diff"] < 0.15
    ), f"Utilization mismatch: theory={rho_theory}, sim={rho_sim}"

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
