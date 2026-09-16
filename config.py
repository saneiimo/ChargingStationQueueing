"""
Shared knobs used across the station model, simulator, and Gym env.

Change values here rather than scattering magic numbers through the code.
Times are in minutes; power is in kW; energy is in kWh; SoC is in [0, 1].
"""

from pathlib import Path

# Repo root (directory containing this file).
PROJECT_ROOT = Path(__file__).resolve().parent

# When an EV is using less than this fraction of its last module, we treat that
# module as underutilized and may free it for another EV on the same pile.
MODULE_CHECK_THRESH = 0.2

# SoC where the BMS request starts tapering on the charging curve.
S_THRESH = 0.4

# How we sample arriving EVs in the engine.
BATTERY_CAP_OPTIONS = [50, 75, 100]
# BATTERY_CAP_OPTIONS = [75.0]
# Battery capacity are inputted as kWh, however for calculation
# we need a unit conversion, as the simulation time is in minutes
HR2MIN = 60
BATTERY_CAP_OPTIONS = [cap * HR2MIN for cap in BATTERY_CAP_OPTIONS]
SOC_I_BOUNDS = [0.15, 0.35]
SOC_F_BOUNDS = [0.7, 0.9]
# C-rate: peak request p_req_max = battery_capacity_kWh * C_RATE.
C_RATE = 2 / HR2MIN
# Episode length (minutes) -- the *measured* phase; see WARMUP_PERIOD below.
MAX_TIME = HR2MIN * 12

# Warm-up length (minutes) run *before* the measured phase, default used by
# SimulationEngine/ChargingStationEnv whenever warmup_period=None is passed
# (their own default). 0.0 means "no warm-up" -- fully backward compatible.
# See simulation/engine.py's own module docstring, "Warm-up period".
WARMUP_PERIOD = 0.0

# Default for SimulationEngine/ChargingStationEnv's flush_queue_at_warmup
# whenever flush_queue_at_warmup=None is passed (their own default) --
# same "None reads this constant" convention as WARMUP_PERIOD/MAX_TIME.
FLUSH_QUEUE_AT_WARMUP = False

# If True, piles assert connector/module consistency after redistributions.
CHECK_INVARIANTS = True

# Gym reward pieces: cost of queue wait per (vehicle * minute), and per drop.
QUEUE_HOLDING_COST = 1.0
DROP_PENALTY = 50.0

# ---------------------------------------------------------------------------
# Visualization style (fonts, sizes, palette, layout).
# Edit here to restyle all charts; visualization.style reads these values.
# ---------------------------------------------------------------------------

# Font families. Titles use the display serif; all other chart text uses the
# sans body face. TTF files under assets/fonts are registered at runtime.
VIZ_TITLE_FONT_FAMILY = "Cormorant Garamond"
VIZ_BODY_FONT_FAMILY = "Barlow"

# Font sizes in points.
VIZ_FONT_SIZE_TITLE = 14  # figure-level title (suptitle)
VIZ_FONT_SIZE_SUBTITLE = 12  # per-axes / subplot title
VIZ_FONT_SIZE_AXIS_TITLE = 11  # x / y axis labels
VIZ_FONT_SIZE_TICK = 9  # tick mark labels and numbers
VIZ_FONT_SIZE_LEGEND = 9  # legend text
VIZ_FONT_SIZE_ANNOTATION = 9  # free text / annotations on the axes

# Named roles in a shared, color-blind friendly palette (Okabe–Ito inspired).
VIZ_COLORS = {
    "background": "#FAFAFA",
    "figure": "#FFFFFF",
    "text": "#2B2B2B",
    "spine": "#B0B0B0",
    "grid": "#D8D8D8",
    "primary": "#0072B2",  # blue
    "secondary": "#E69F00",  # orange
    "tertiary": "#009E73",  # bluish green
    "quaternary": "#CC79A7",  # reddish purple
    "alert": "#D55E00",  # vermillion
    "neutral": "#4D4D4D",  # dark gray
    "sky": "#56B4E9",
    "muted": "#A8A29E",
}

# Ordered categorical palette (cycles for multi-series charts).
VIZ_PALETTE = (
    VIZ_COLORS["primary"],
    VIZ_COLORS["secondary"],
    VIZ_COLORS["tertiary"],
    VIZ_COLORS["quaternary"],
    VIZ_COLORS["alert"],
    VIZ_COLORS["sky"],
    VIZ_COLORS["neutral"],
)

# Semantic colors for EV theory-vs-sim overlays (also reused by
# offline_opt.visualization for the offline MILP's own solution plots).
VIZ_SERIES = {
    "theory": VIZ_COLORS["neutral"],
    "sim_bms": VIZ_COLORS["sky"],
    "sim_actual": VIZ_COLORS["secondary"],
    "recorded": VIZ_COLORS["alert"],
    "recorded_edge": VIZ_COLORS["text"],
    "modules": VIZ_COLORS["quaternary"],
}

# Colormaps for heatmaps / continuous fields (matplotlib names).
VIZ_CMAP_SEQUENTIAL = "Blues"
VIZ_CMAP_DIVERGING = "RdBu_r"

# Default figure sizes (width, height) in inches.
VIZ_FIGSIZE_STANDARD = (6.0, 4.2)
VIZ_FIGSIZE_WIDE = (13.0, 5.5)
VIZ_FIGSIZE_TALL = (10.0, 7.0)
VIZ_FIGSIZE_PANEL_WIDTH = 5.4  # width per subplot when tiling horizontally
VIZ_FIGSIZE_PANEL_HEIGHT = 4.0

# Line / marker defaults for EV curve overlays.
VIZ_LINEWIDTH_THEORY = 2.0
VIZ_LINEWIDTH_SIM = 1.9
VIZ_MARKER_SIZE_RECORDED = 36

# Minimal layout knobs.
VIZ_GRID_ALPHA = 0.28
VIZ_GRID_LINEWIDTH = 0.7
VIZ_SPINE_WIDTH = 0.8
VIZ_TITLE_PAD = 10
VIZ_LEGEND_FRAMEON = False

# Directory holding downloaded TTF/OTF faces for the configured families.
VIZ_FONTS_DIR = PROJECT_ROOT / "assets" / "fonts"
