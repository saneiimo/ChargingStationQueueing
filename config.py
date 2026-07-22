"""
Shared knobs used across the station model, simulator, and Gym env.

Change values here rather than scattering magic numbers through the code.
Times are in minutes; power is in kW; energy is in kWh; SoC is in [0, 1].
"""

# When an EV is using less than this fraction of its last brick, we treat that
# brick as underutilized and may free it for another EV on the same pile.
BRICK_CHECK_THRESH = 0.2

# SoC where the BMS request starts tapering on the charging curve.
S_THRESH = 0.6

# C-rate: peak request p_req_max = battery_capacity_kWh * C_RATE.
C_RATE = 1

# How we sample arriving EVs in the engine.
BATTERY_CAP_OPTIONS = [50.0, 100.0, 150.0]
SOC_I_BOUNDS = [0.1, 0.3]
SOC_F_BOUNDS = [0.7, 0.9]

# Episode length (minutes).
MAX_TIME = 1440

# If True, piles assert nozzle/brick consistency after redistributions.
CHECK_INVARIANTS = True

# Gym reward pieces: cost of queue wait per (vehicle * minute), and per drop.
QUEUE_HOLDING_COST = 1.0
DROP_PENALTY = 50.0
