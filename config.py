# The threshold to check when an EV is utilizing less than
# this ratio of its last brick, considering assigning that brick to another EV where they might beter benefit from that
BRICK_CHECK_THRESH = 0.2
# The SoC where charging tapers in a charging curve
S_THRESH = 0.6
# A battery C-rate is a measure of the charge or discharge
# current relative to its maximum capacity, indicating how fast a battery is fully charged or discharged
C_RATE = 1

# Generate EVs battery capacity, initial and final SoC will be sample
# from the following options.
BATTERY_CAP_OPTIONS = [50.0, 100.0, 150.0]
SOC_I_BOUNDS = [0.1, 0.3]
SOC_F_BOUNDS = [0.7, 0.9]

# Duration for simulation, in minutes
MAX_TIME = 1440
