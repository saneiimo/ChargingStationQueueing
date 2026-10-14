"""
Station-load metrics (``metrics.load``; definitions in ``metrics/README.md``):
connector occupancy, power utilization, offered power load and offered
connector load, realized and theoretical.

  1. The solo charging time matches the simulator: a vehicle alone on an
     empty pile takes exactly ``solo_charging_minutes``, with and without
     the pile capping its power.
  2. Occupancy and power utilization match hand computations.
  3. Realized offered loads are the stated averages over the window's
     arrivals; theoretical ones match Monte Carlo draws from the arrival
     generator, and the realized ones converge to them.
  4. The metrics reach every report: ``queueing_summary``, the validation
     printout, the policy sweep's columns and the objective sweep's
     ``sim_*`` columns (no optimization model is solved).

    python -m pytest tests/test_station_load.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from config import BATTERY_CAP_OPTIONS, C_RATE, HR2MIN, S_THRESH, SOC_F_BOUNDS, SOC_I_BOUNDS
from env.charging_env import ChargingStationEnv
from metrics.load import (
    ArrivalModel,
    realized_offered_load,
    solo_charging_minutes,
    station_load_summary,
    theoretical_offered_load,
)
from metrics.metrics_tracker import MetricsTracker
from models.ev import EV
from policy.queue.fifo import FIFOQueuePolicy
from simulation.arrivals import generate_arrivals


def _ev(i, t, kwh, s_i=0.2, s_f=0.8):
    return EV(id=i, c_b=kwh * HR2MIN, s_i=s_i, s_f=s_f, arrival_time=t)


def _run(evs, *, n_connectors=2, n_modules=6, max_time=30.0, warmup=0.0):
    env = ChargingStationEnv(
        n_piles=1, n_connectors=n_connectors, n_modules=n_modules, p_module=25.0, queue_capacity=50,
        mean_interarrival=None, arrivals=evs, max_time=max_time, warmup_period=warmup, drain=True,
    )
    obs, _ = env.reset(seed=0)
    rng, policy, done = np.random.default_rng(1), FIFOQueuePolicy(), False
    while not done:
        ev, pile = policy.decide(obs, env.action_masks(), rng, env.engine.station)
        obs, _, done, _, _ = env.step(pile, ev=ev)
    return env


# ---------------------------------------------------------------------------
# 1. Solo charging time
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_modules", [8, 2])  # 200 kW (BMS binds) and 50 kW (pile binds)
def test_solo_time_matches_the_simulator(n_modules):
    env = _run([_ev(0, 0.5, 75.0, 0.2, 0.85)], n_connectors=1, n_modules=n_modules, max_time=1.0)
    ev = env.engine.metrics.finished_evs[0]
    simulated = ev.departure_time - ev.service_start_time
    predicted = solo_charging_minutes(
        ev.c_b, ev.s_i, ev.s_f, s_th=ev.s_th, c_rate=ev.c_rate, pile_power=n_modules * 25.0
    )
    assert simulated == pytest.approx(predicted, rel=1e-6)


def test_solo_time_reduces_to_the_unconstrained_curve():
    """With the pile never binding, the knee is s_th and the time is the
    unconstrained BMS curve's (visualization.charging_theory)."""
    from visualization.charging_theory import theoretical_local_time_to_soc

    ev = _ev(0, 0.0, 60.0, 0.25, 0.9)
    t = solo_charging_minutes(ev.c_b, ev.s_i, ev.s_f, s_th=ev.s_th, c_rate=ev.c_rate, pile_power=1e9)
    assert t == pytest.approx(theoretical_local_time_to_soc(ev, ev.s_f), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. Occupancy and power utilization
# ---------------------------------------------------------------------------


def test_occupancy_and_power_utilization_by_hand():
    env = _run([_ev(0, 0.0, 60.0), _ev(1, 2.0, 40.0), _ev(2, 4.0, 75.0)], max_time=5.0)
    m, st = env.engine.metrics, env.engine.station
    end = float(env.engine.current_time)
    out = station_load_summary(m, st, end, arrivals_until=env.engine.max_time)

    busy = np.zeros(2)
    for ev in m.finished_evs:
        busy[ev.connector_id_tracker] += ev.departure_time - ev.service_start_time
    for c in range(2):
        assert out[f"connector_occupancy_p0_c{c}"] == pytest.approx(busy[c] / end, rel=1e-9)
    assert out["connector_occupancy"] == pytest.approx(busy.mean() / end, rel=1e-9)

    delivered = sum(ev.energy_needed for ev in m.finished_evs)  # drained: everyone completed
    assert out["power_utilization_p0"] == pytest.approx(delivered / (150.0 * end / 60.0), rel=1e-6)
    assert out["power_utilization"] == out["power_utilization_p0"]


def test_occupancy_window_starts_at_the_boundary():
    """With since = warm-up end, both time averages divide by (end - since)."""
    evs = [_ev(i, t, 50.0) for i, t in enumerate([0.0, 3.0, 9.0, 14.0, 22.0])]
    env = _run(evs, max_time=20.0, warmup=10.0)
    m, st = env.engine.metrics, env.engine.station
    end, t0 = float(env.engine.current_time), m.warmup_period
    out = station_load_summary(m, st, end, since=t0, arrivals_until=env.engine.max_time)
    window = end - t0
    np.testing.assert_allclose(
        [out["connector_occupancy_p0_c0"], out["connector_occupancy_p0_c1"]],
        m.connector_active_minutes_since(t0)[0] / window,
    )
    assert out["power_utilization_p0"] == pytest.approx(m.pile_energy(t0)[0] / (150.0 * window / 60.0))


# ---------------------------------------------------------------------------
# 3. Offered loads
# ---------------------------------------------------------------------------


def test_realized_offered_load_is_the_stated_average():
    evs = [_ev(i, t, kwh, 0.2 + 0.02 * i, 0.8) for i, (t, kwh) in
           enumerate([(1.0, 50.0), (6.0, 75.0), (11.0, 100.0), (13.0, 50.0), (25.0, 75.0)])]
    env = _run(evs, max_time=15.0, warmup=5.0)
    m = env.engine.metrics
    # Window [5, 20]: vehicle 0 arrived during warm-up, vehicle 4 (t=25)
    # after arrivals closed, so it never entered; vehicles 1-3 count.
    since, until = m.warmup_period, env.engine.max_time
    out = realized_offered_load(m, since=since, arrivals_until=until, n_servers=2,
                                pile_power=150.0, rated_power=150.0)
    in_window = [ev for ev in m.arrived_evs if since <= ev.arrival_time <= until]
    assert out["n_arrivals_realized"] == len(in_window) == 3
    lam = 3 / (until - since)
    e_w = np.mean([ev.energy_needed for ev in in_window])
    e_s = np.mean([solo_charging_minutes(ev.c_b, ev.s_i, ev.s_f, s_th=ev.s_th, c_rate=ev.c_rate,
                                         pile_power=150.0) for ev in in_window])
    assert out["arrival_rate_realized"] == pytest.approx(lam)
    assert out["offered_power_load_realized"] == pytest.approx(60.0 * lam * e_w / 150.0)
    assert out["offered_connector_load_realized"] == pytest.approx(lam * e_s)
    assert out["offered_occupancy_realized"] == pytest.approx(lam * e_s / 2)


def test_theoretical_offered_load_matches_monte_carlo():
    model = ArrivalModel(mean_interarrival=12.0)
    th = theoretical_offered_load(model, n_servers=4, pile_power=150.0, rated_power=300.0)
    rng = np.random.default_rng(0)
    n = 400_000
    c_b = rng.choice(np.asarray(BATTERY_CAP_OPTIONS, dtype=float), n)
    s_i = rng.uniform(*SOC_I_BOUNDS, n)
    s_f = rng.uniform(*SOC_F_BOUNDS, n)
    e_w = np.mean((s_f - s_i) * c_b / HR2MIN)
    e_s = np.mean(solo_charging_minutes(c_b, s_i, s_f, s_th=S_THRESH, c_rate=C_RATE, pile_power=150.0))
    assert th["mean_energy_request_theory"] == pytest.approx(e_w, rel=3e-3)
    assert th["mean_solo_time_theory"] == pytest.approx(e_s, rel=3e-3)
    assert th["arrival_rate_theory"] == pytest.approx(1 / 12.0)
    assert th["offered_power_load_theory"] == pytest.approx(
        60.0 / 12.0 * th["mean_energy_request_theory"] / 300.0
    )
    assert th["offered_occupancy_theory"] == pytest.approx(th["offered_connector_load_theory"] / 4)


def test_realized_converges_to_theory_on_a_long_stream():
    """Thousands of draws from the generator itself: realized -> theory."""
    model = ArrivalModel(mean_interarrival=2.0)
    evs = generate_arrivals(mean_interarrival=2.0, max_time=10_000.0, rng=np.random.default_rng(3))
    m = MetricsTracker(n_piles=2, n_connectors=2)
    for ev in evs:
        m.update_arrived_evs(ev)
    kw = dict(n_servers=4, pile_power=150.0, rated_power=300.0)
    real = realized_offered_load(m, since=0.0, arrivals_until=10_000.0, **kw)
    th = theoretical_offered_load(model, **kw)
    for name in ("arrival_rate", "mean_energy_request", "mean_solo_time", "offered_power_load",
                 "offered_connector_load"):
        assert real[f"{name}_realized"] == pytest.approx(th[f"{name}_theory"], rel=0.05), name


def test_offered_occupancy_never_exceeds_measured_occupancy_much():
    """Sharing only stretches stays, so a drained run's measured occupancy
    is at least the offered occupancy of the same vehicles over the same
    window (here the window is the arrival window plus the drain, so the
    comparison uses busy minutes, not fractions)."""
    evs = [_ev(i, t, 75.0) for i, t in enumerate([0.0, 0.0, 1.0, 2.0, 3.0])]
    env = _run(evs, max_time=4.0)
    m, st = env.engine.metrics, env.engine.station
    busy = m.connector_active_minutes_since(0.0).sum()
    solo = sum(solo_charging_minutes(ev.c_b, ev.s_i, ev.s_f, s_th=ev.s_th, c_rate=ev.c_rate,
                                     pile_power=150.0) for ev in m.arrived_evs)
    assert busy >= solo - 1e-6


# ---------------------------------------------------------------------------
# 4. Reports
# ---------------------------------------------------------------------------


def test_queueing_summary_and_printout_carry_the_load_metrics(capsys):
    from metrics.validate import report_queueing_laws

    env = _run([_ev(0, 0.0, 60.0), _ev(1, 2.0, 40.0)], max_time=5.0)
    eng = env.engine
    out = eng.metrics.queueing_summary(eng.current_time, station=eng.station, arrivals_until=eng.max_time)
    assert out["connector_occupancy"] == pytest.approx(out["rho_sim"])
    assert "offered_power_load_realized" in out and "offered_power_load_theory" not in out
    printed = report_queueing_laws(env, verbose=True, mean_interarrival=10.0)
    assert printed["offered_power_load_theory"] > 0
    assert "connector occupancy" in capsys.readouterr().out


def test_policy_sweep_reports_the_load_columns():
    from experiments.policy_sweep.replications import run_replications

    scenario = dict(n_piles=1, n_connectors=2, n_modules=6, p_module=25.0, queue_capacity=50,
                    mean_interarrival=15.0, max_time=120.0, warmup_period=30.0)
    df = run_replications(scenario, n_reps=2, seed0=0)
    for col in ("connector occupancy", "power utilization", "offered power load (realized)",
                "offered power load (theory)", "offered connector load (theory)",
                "offered occupancy (realized) (measured)"):
        assert col in df.columns, col
        assert df[col].notna().all(), col
    assert df["offered power load (theory)"].nunique() == 1  # the generator's, not the sample's


def test_objective_sweep_sim_stage_has_the_load_columns():
    from experiments.objective_sweep import TrialConfig, flatten_trial, run_trial

    cfg = TrialConfig(n_piles=1, n_connectors=2, n_modules=6, max_time=60, warmup_period=120, delta=5.0,
                      seed=41, battery_cap_kwh=(75.0,), mean_interarrival=15.0)
    result = run_trial(cfg, run_exact_model=False, run_bp_model=False, run_dw_model=False, verbose=False)
    row = flatten_trial(result, 0)
    for col in ("sim_connector_occupancy_p0_c0", "sim_connector_occupancy", "sim_power_utilization_p0",
                "sim_offered_power_load_realized", "sim_offered_power_load_theory",
                "sim_offered_connector_load_theory", "sim_offered_occupancy_realized"):
        assert col in row and math.isfinite(row[col]), col
    # Theory for a 75 kWh fleet at one arrival per 15 min on a 150 kW pile.
    model = ArrivalModel(mean_interarrival=15.0, battery_cap_options=(75.0 * HR2MIN,))
    th = theoretical_offered_load(model, n_servers=2, pile_power=150.0, rated_power=150.0)
    assert row["sim_offered_power_load_theory"] == pytest.approx(th["offered_power_load_theory"])


def test_energy_identity_links_power_utilization_and_offered_power_load():
    """In a drained run with nobody turned away, the energy delivered after
    t0 is exactly the window's arrivals' requests plus what the vehicles
    already present at t0 still needed:

        U_P * R * (t_end - t0)/60 = rho_P * R * (T - t0)/60 + E_backlog(t0)

    with R the station's rated power (metrics/README.md, "How they relate")."""
    evs = [_ev(i, t, kwh) for i, (t, kwh) in
           enumerate([(0.0, 75.0), (1.0, 50.0), (2.0, 100.0), (4.0, 60.0), (12.0, 75.0), (15.0, 40.0)])]
    env = _run(evs, max_time=10.0, warmup=5.0)
    m, st, eng = env.engine.metrics, env.engine.station, env.engine
    assert not m.dropped_evs and m.queued_at_warmup_end and m.in_service_at_warmup_end
    t0, T, t_end = m.warmup_period, float(eng.max_time), float(eng.current_time)
    out = station_load_summary(m, st, t_end, since=t0, arrivals_until=T)
    R = 150.0
    delivered = out["power_utilization"] * R * (t_end - t0) / 60.0
    offered = out["offered_power_load_realized"] * R * (T - t0) / 60.0
    backlog = sum(ev.energy_needed for ev in m.queued_at_warmup_end) + sum(
        (s.s_f - s.s_current) * s.c_b / HR2MIN for s in m.in_service_at_warmup_end
    )
    assert delivered == pytest.approx(offered + backlog, rel=1e-6)


def test_trial_rate_from_a_target_offered_power_load():
    """mean_interarrival=None + offered_power_load=rho: the trial's arrival
    rate is the one whose theoretical offered power load is rho; a given
    mean_interarrival wins over it; replace() recomputes."""
    import dataclasses

    from experiments.objective_sweep import TrialConfig, flatten_trial, run_trial

    cfg = TrialConfig(n_piles=1, n_connectors=2, n_modules=6, max_time=60, warmup_period=60, delta=5.0,
                      seed=41, battery_cap_kwh=(75.0,), mean_interarrival=None, offered_power_load=0.8)
    # 75 kWh * (E[s_f] - E[s_i]) = 75 * (0.8 - 0.25) = 41.25 kWh; 60 * 41.25 / (0.8 * 150 kW)
    assert cfg.effective_mean_interarrival == pytest.approx(20.625)
    row = flatten_trial(run_trial(cfg, run_exact_model=False, run_bp_model=False, run_dw_model=False,
                                  verbose=False), 0)
    assert row["mean_interarrival"] == pytest.approx(20.625)
    assert row["sim_offered_power_load_theory"] == pytest.approx(0.8)
    assert dataclasses.replace(cfg, offered_power_load=0.5).effective_mean_interarrival == pytest.approx(33.0)
    assert dataclasses.replace(cfg, mean_interarrival=12.0).effective_mean_interarrival == 12.0
    with pytest.raises(ValueError, match="mean_interarrival or offered_power_load"):
        TrialConfig(mean_interarrival=None).to_row()


def test_policy_sweep_rate_from_a_target_offered_power_load(tmp_path):
    """PolicyConfig follows the same rule as TrialConfig: offered_power_load
    sets the rate when mean_interarrival is None; a given one wins."""
    import dataclasses

    from experiments.core import config_grid
    from experiments.policy_sweep import PolicyConfig, run_policy_sweep

    base = PolicyConfig(n_piles=1, n_connectors=2, n_modules=6, battery_cap_kwh=(75.0,), max_time=120.0,
                        warmup_period=30.0, mean_interarrival=None, n_reps=2, queue_policies=("FIFO",))
    configs = config_grid(base, offered_power_load=[0.4, 0.8])
    # 75 kWh * (0.8 - 0.25) = 41.25 kWh per vehicle; 60 * 41.25 / (rho * 150 kW)
    assert [c.effective_mean_interarrival for c in configs] == pytest.approx([41.25, 20.625])
    assert dataclasses.replace(configs[0], mean_interarrival=10.0).effective_mean_interarrival == 10.0
    df = run_policy_sweep(configs, "rho", out_dir=tmp_path, verbose=False)
    assert sorted(df["mean_interarrival"].unique()) == pytest.approx([20.625, 41.25])
    for rho, group in df.groupby("offered_power_load"):
        assert group["offered_power_load_theory_mean"].iloc[0] == pytest.approx(rho)
    with pytest.raises(ValueError, match="mean_interarrival or offered_power_load"):
        PolicyConfig(mean_interarrival=None).scenario_kwargs
