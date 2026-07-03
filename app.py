from io import BytesIO
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Rectangle
from matplotlib.colors import LinearSegmentedColormap, Normalize
from mplsoccer import Pitch
from PIL import Image

from external_models import load_markov_model
from heuristic_scoring import POSITION_GROUPS_ORDER, GROUP_COLORS, is_outfield_position, position_group
from scipy.interpolate import RegularGridInterpolator

# ── PAGE CONFIG ────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="World Cup Stats — xT v4")

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .player-header {
        font-size: 1.15rem;
        font-weight: 700;
        color: #eef1f7;
        margin-bottom: 0.15rem;
    }
    .player-sub {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-bottom: 0.75rem;
    }
    .map-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #c7cdda;
        margin: 0.25rem 0 0.35rem 0;
        text-align: center;
    }
  </style>
    """,
    unsafe_allow_html=True,
)

# ── CONSTANTS ────────────────────────────────────────────────
FIELD_X, FIELD_Y = 120.0, 80.0
HALF_LINE_X = FIELD_X / 2
FINAL_THIRD_LINE_X = 80.0
GOAL_X, GOAL_Y = 120.0, 40.0
FIG_W, FIG_H = 7.2, 4.8
FIG_DPI = 220
PASS_START_MARKER_SIZE = 7
CARRY_START_MARKER_SIZE = 7
MAP_REF_WIDTH = 7.2
PASS_DEST_HEATMAP_COLS = 6
PASS_DEST_HEATMAP_ROWS = 4
ARROW_WIDTH = 0.75
ARROW_HEADWIDTH = 1.15
ARROW_HEADLENGTH = 1.15
ARROW_ALPHA = 0.68
ARROW_ALPHA_EMPH = 0.82
ALL_GAMES_LABEL = "todos os jogos"
DATA_CACHE_VERSION = 34
XT_ZONE_COLS = 3
XT_ZONE_ROWS = 2
NX_XT = 16
NY_XT = 12
XT_GRID_CMAP = LinearSegmentedColormap.from_list(
    "xt_grid", ["#1a1a2e", "#3b82f6", "#fbbf24", "#ef4444"]
)
WYSCOUT_PITCH_SIZE = 100.0
OPT_ATTACKING_TWO_THIRDS_X = 40.0
WYSCOUT_PROG_OWN_HALF = 30.0
WYSCOUT_PROG_CROSS_HALF = 15.0
WYSCOUT_PROG_OPP_HALF = 10.0
XT_MODEL_HEURISTIC_V3 = "heuristic_v3"
XT_MIN_PASS_DISTANCE = 9.5
XT_V3_FINE_NX = 96
XT_V3_FINE_NY = 64
XT_V3_DEF_MAX = 0.25
XT_V3_MID_MAX = 0.60
XT_V3_ATT_BYLINE = 0.94
XT_V3_SURFACE_MAX = 1.02
XT_V3_ZONE_BLEND_WIDTH = 22.0
XT_V3_LAT_DISC_MAX = 0.16
XT_V3_LAT_CURVE_POWER = 1.0
XT_V3_PROG_SCALE = 0.15
XT_V3_HIGH_SCALE = 0.35
XT_V3_PROG_FLOOR = 0.08
XT_V3_HIGH_FLOOR = 0.18
XT_V3_PROG_SCALE_CLASS = 0.17
XT_V3_HIGH_SCALE_CLASS = 0.40
XT_V3_PROG_FLOOR_CLASS = 0.10
XT_V3_HIGH_FLOOR_CLASS = 0.22
XT_V3_NEG_PENALTY_FACTOR = 0.55
XT_V3_PRESSURE_ESCAPE_BONUS = 0.02
XT_V3_PRESSURE_X_MAX = 50.0
XT_V3_WIDE_FRAC = 0.60
XT_V3_NEG_RECYCLE_X_MAX = 60.0
XT_V4_BOX_X_START = 90.0
XT_V4_BOX_X_FULL = 112.0
XT_V4_CORNER_LAT_ON = 0.58
XT_V4_CORNER_PENALTY = 0.10
XT_V4_CENTRAL_PREMIUM = 0.06
XT_V4_SHORT_PASS_DIST = 8.0
XT_V4_SHORT_PASS_FACTOR = 0.55
XT_V4_V1_WING_BASE = 0.80
XT_V4_V1_CENT_MULT = 1.00
XT_V5_MAX_DELTA_DEF = 0.28
XT_V5_MAX_DELTA_MID = 0.36
XT_V5_MAX_DELTA_ATT = 0.42
XT_V5_MAX_DELTA_BOX = 0.52

# xT Heurístico v3.1 — transições suaves e salto reduzido na linha de meio
XT_MODEL_HEURISTIC_V31 = "heuristic_v31"
XT_V31_ZONE_BLEND_WIDTH = 48.0
XT_V31_LAT_DISC_MAX = 0.06
XT_V31_LAT_GATE_X = HALF_LINE_X
XT_V31_GAUSS_SIGMA_X = 3.5
XT_V31_GAUSS_SIGMA_Y = 0.0
XT_V31_COL_SMOOTH_KERNEL = (0.22, 0.56, 0.22)
XT_V31_MAX_COL_STEP_DEF = 0.050
XT_V31_MAX_COL_STEP_ATT = 0.078
XT_V31_ATT_COL_START = 10

# xT Heurístico v3.2 — base uniforme entre quadrantes + bônus Markov mais forte
XT_MODEL_HEURISTIC_V32 = "heuristic_v32"
XT_V32_BASE_FLOOR = 0.055
XT_V32_BASE_SPREAD = 0.042
XT_V32_BASE_SHAPE_GAMMA = 0.85
XT_V32_QUADRANT_BONUS_MAX = 0.15
XT_V32_BONUS_POWER = 1.10
XT_V32_BONUS_SIGMA_X = 3.0
XT_V32_BONUS_SIGMA_Y = 1.5
XT_V32_SURFACE_MAX = 0.24
XT_V32_GAUSS_SIGMA_X = 0.0
XT_V32_GAUSS_SIGMA_Y = 0.0
XT_V32_COL_SMOOTH_KERNEL = (0.22, 0.56, 0.22)
XT_V32_MAX_COL_STEP_DEF = 0.012
XT_V32_MAX_COL_STEP_ATT = 0.018
XT_V32_ATT_COL_START = 10

# xT Heurístico v3.3 — bônus Markov escolhido por validação hold-out
XT_MODEL_HEURISTIC_V33 = "heuristic_v33"

# xT Heurístico v4 — v3.1 + bônus Top5 suave (2/3 defensivos) · forte no último terço
XT_MODEL_HEURISTIC_V4 = "heuristic_v4"
XT_V4_MARKOV_BONUS_MAX = 0.052
XT_V4_MARKOV_BONUS_POWER = 1.0
XT_V4_MARKOV_DEF_MID_FLOOR = 0.06
XT_V4_MARKOV_GATE_BLEND = 14.0
XT_V4_SURFACE_MAX = XT_V3_SURFACE_MAX

# xT Heurístico v4.1 — v3.1 + bônus Top5 uniforme e leve (comparação com v4)
XT_MODEL_HEURISTIC_V41 = "heuristic_v41"
XT_V41_MARKOV_BONUS_MAX = 0.038
XT_V41_MARKOV_BONUS_POWER = 1.0
XT_V41_SURFACE_MAX = XT_V3_SURFACE_MAX

# Modelo ativo para stats, impact plays e mapas de análise
XT_PRIMARY_VARIANT = "v4"

CARD_TITLE_TEXT = "11px"
CARD_LABEL_TEXT = "12px"
CARD_VALUE_TEXT = "18px"
CARD_INNER_BORDER = "rgba(107,114,128,0.45)"
TOP_DELTAXT_N = 10
IMPACT_PASS_MIN_GOAL_APPROACH_FINAL_THIRD = 5.0
IMPACT_PASS_MIN_GOAL_APPROACH_REST = 10.0
EXCLUDED_CSV = {"enzo.csv"}
CSV_X_FLIP_MATCHES = frozenset({"Uruguay"})

PLAYERS = [
    {"code": "BG", "name": "Bruno Guimarães", "position": "CM", "tone": "#5b9bd5", "glob": "BG-vs *.csv"},
    {"code": "CS", "name": "Casemiro", "position": "DM", "tone": "#e67e22", "glob": "CS-vs *.csv"},
    {"code": "LP", "name": "Lucas Paquetá", "position": "AM", "tone": "#22c55e", "glob": "LP-vs *.csv"},
    {"code": "PD", "name": "Pedri", "position": "CM", "tone": "#9333ea", "glob": "Pedri-vs *.csv"},
    {"code": "RD", "name": "Rodri", "position": "DM", "tone": "#dc2626", "glob": "Rodri-vs *.csv"},
]

CMAP_PASS = LinearSegmentedColormap.from_list(
    "pass_dxt", ["#bfdbfe", "#60a5fa", "#2563eb", "#1e3a8a"]
)
CMAP_CARRY = LinearSegmentedColormap.from_list(
    "carry_dxt", ["#fde68a", "#fbbf24", "#f59e0b", "#b45309"]
)
CMAP_PASS_DEST = LinearSegmentedColormap.from_list(
    "pass_dest", ["#1a1a2e", "#1e3a8a", "#3b82f6", "#fbbf24", "#ef4444"]
)

STAT_CARD_GENERAL_COLOR = "#3b82f6"
STAT_CARD_IMPACT_COLOR = "#22c55e"
STAT_CARD_XT_COLOR = "#a855f7"

COLOR_SUCCESS = "#6ee7b7"
COLOR_PROGRESSIVE = "#7dd3fc"
COLOR_HIGHLY_PROGRESSIVE = "#fcd34d"
COLOR_FAIL = "#fca5a5"
COLOR_CARRY = "#c4b5fd"
ALPHA_SUCCESS = 0.50
COLOR_CARRY_BASE_ALPHA = 0.50


# ── COORDINATE HELPERS ───────────────────────────────────────
def wyscout_to_statsbomb(x: float, y: float, *, flip_x: bool = False) -> tuple[float, float]:
    """Wyscout 0–100 → StatsBomb 120×80; espelha Y para corrigir corredores laterais."""
    x_sb = x * FIELD_X / WYSCOUT_PITCH_SIZE
    y_sb = FIELD_Y - (y * FIELD_Y / WYSCOUT_PITCH_SIZE)
    if flip_x:
        x_sb = FIELD_X - x_sb
    return x_sb, y_sb


def distance_to_goal(x: float, y: float) -> float:
    return float(np.sqrt((GOAL_X - x) ** 2 + (GOAL_Y - y) ** 2))


def pass_min_goal_approach(x_end: float) -> float:
    """Minimum goal approach (m) by pass end zone: 5 m in final third, 10 m elsewhere."""
    if x_end >= FINAL_THIRD_LINE_X:
        return IMPACT_PASS_MIN_GOAL_APPROACH_FINAL_THIRD
    return IMPACT_PASS_MIN_GOAL_APPROACH_REST


def pass_approaches_goal(
    x_start: float,
    y_start: float,
    x_end: float,
    y_end: float,
) -> bool:
    progress = distance_to_goal(x_start, y_start) - distance_to_goal(x_end, y_end)
    return progress >= pass_min_goal_approach(x_end)


def is_progressive_wyscout(x_start: float, y_start: float, x_end: float, y_end: float) -> bool:
    start_dist = distance_to_goal(x_start, y_start)
    end_dist = distance_to_goal(x_end, y_end)
    progress = start_dist - end_dist
    if progress <= 0:
        return False
    start_own = x_start < HALF_LINE_X
    end_own = x_end < HALF_LINE_X
    start_opp = x_start >= HALF_LINE_X
    end_opp = x_end >= HALF_LINE_X
    if start_own and end_own:
        return progress >= WYSCOUT_PROG_OWN_HALF
    if start_opp and end_opp:
        return progress >= WYSCOUT_PROG_OPP_HALF
    return progress >= WYSCOUT_PROG_CROSS_HALF


def _parse_bool(value) -> bool:
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"true", "1", "yes", "successful"}


def _has_coords(row, prefix: str) -> bool:
    x_col, y_col = f"{prefix}_x", f"{prefix}_y"
    return pd.notna(row.get(x_col)) and pd.notna(row.get(y_col))


# ── xT HEURÍSTICO v3 ─────────────────────────────────────────
def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _smootherstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def _centrality(y: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs((y / FIELD_Y) - 0.5) * 2.0


def _lateral_frac(y: float) -> float:
    return float(abs(y - GOAL_Y) / (FIELD_Y / 2.0))


def _lateral_relative_position(y: np.ndarray) -> np.ndarray:
    return np.abs(y - GOAL_Y) / (FIELD_Y / 2.0)


def _location_factor_v3(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lat = _lateral_relative_position(y)
    depth = np.clip(
        (x - OPT_ATTACKING_TWO_THIRDS_X) / (FIELD_X - OPT_ATTACKING_TWO_THIRDS_X),
        0.0,
        1.0,
    )
    zone_gate = _smoothstep(depth)
    max_discount = XT_V3_LAT_DISC_MAX * zone_gate
    lateral_curve = _smoothstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _v4_box_gate(x: np.ndarray) -> np.ndarray:
    span = max(XT_V4_BOX_X_FULL - XT_V4_BOX_X_START, 1.0)
    t = np.clip((x - XT_V4_BOX_X_START) / span, 0.0, 1.0)
    return _smoothstep(t)


def _v4_xg_finishing_factor(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    box_gate = _v4_box_gate(x)
    cent = _centrality(y)
    lat = _lateral_relative_position(y)
    central_bonus = XT_V4_CENTRAL_PREMIUM * box_gate * _smoothstep(cent)
    wide_in_box = box_gate * _smoothstep(np.clip((lat - XT_V4_CORNER_LAT_ON) / 0.42, 0.0, 1.0))
    wide_discount = XT_V4_CORNER_PENALTY * wide_in_box
    return np.clip(1.0 + central_bonus - wide_discount, 0.94, 1.06)


def _enforce_row_monotonic_x(grid: np.ndarray) -> np.ndarray:
    out = grid.copy()
    for iy in range(out.shape[0]):
        for ix in range(1, out.shape[1]):
            if out[iy, ix] < out[iy, ix - 1]:
                out[iy, ix] = out[iy, ix - 1]
    return out


def _map_zonal_threat_v3_smooth(x: np.ndarray) -> np.ndarray:
    blend = XT_V3_ZONE_BLEND_WIDTH
    x = np.clip(x, 0.0, FIELD_X)
    threat_def = XT_V3_DEF_MAX * np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0)
    mid_span = max(FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X, 1.0)
    mid_t = np.clip((x - OPT_ATTACKING_TWO_THIRDS_X) / mid_span, 0.0, 1.0)
    threat_mid = XT_V3_DEF_MAX + (XT_V3_MID_MAX - XT_V3_DEF_MAX) * _smootherstep(mid_t)
    att_span = max(FIELD_X - FINAL_THIRD_LINE_X, 1.0)
    att_t = np.clip((x - FINAL_THIRD_LINE_X) / att_span, 0.0, 1.0)
    threat_att = XT_V3_MID_MAX + (XT_V3_ATT_BYLINE - XT_V3_MID_MAX) * _smootherstep(att_t)
    w_def = 1.0 - _smootherstep(np.clip((x - (OPT_ATTACKING_TWO_THIRDS_X - blend)) / blend, 0.0, 1.0))
    w_att = _smootherstep(np.clip((x - (FINAL_THIRD_LINE_X - blend)) / blend, 0.0, 1.0))
    w_mid = np.clip(1.0 - w_def - w_att, 0.0, 1.0)
    w_sum = w_def + w_mid + w_att + 1e-12
    return (w_def * threat_def + w_mid * threat_mid + w_att * threat_att) / w_sum


def _build_heuristic_v3_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v3_smooth(Xc)
    surface = zonal * _location_factor_v3(Xc, Yc) * _v4_xg_finishing_factor(Xc, Yc)
    surface = np.clip(surface, 0.0, XT_V3_SURFACE_MAX)
    return _enforce_row_monotonic_x(surface)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3_xt_grid(
    n_x: int = NX_XT, n_y: int = NY_XT,
) -> np.ndarray:
    """16×12 display grid sampled from the v3 fine threat surface."""
    fine = compute_heuristic_v3_fine_grid()
    grid = zone_xt_means(fine, n_x=n_x, n_y=n_y)
    return _enforce_row_monotonic_x(grid)


@st.cache_data(show_spinner=False)
def compute_heuristic_v3_fine_grid(nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v3_threat_surface(Xc, Yc)


def xt_value_bilinear(x: float, y: float, fine_grid: np.ndarray) -> float:
    nx, ny = fine_grid.shape[1], fine_grid.shape[0]
    fx = float(np.clip(x / FIELD_X * (nx - 1), 0.0, nx - 1))
    fy = float(np.clip(y / FIELD_Y * (ny - 1), 0.0, ny - 1))
    x0, y0 = int(fx), int(fy)
    x1, y1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1)
    tx, ty = fx - x0, fy - y0
    v00, v10 = fine_grid[y0, x0], fine_grid[y0, x1]
    v01, v11 = fine_grid[y1, x0], fine_grid[y1, x1]
    return float(
        (1 - tx) * (1 - ty) * v00
        + tx * (1 - ty) * v10
        + (1 - tx) * ty * v01
        + tx * ty * v11
    )


def _v3_short_pass_multiplier(pass_distance: float) -> float:
    short_dist = XT_V4_SHORT_PASS_DIST
    short_factor = XT_V4_SHORT_PASS_FACTOR
    blend_span = 4.0
    if pass_distance < short_dist:
        return short_factor
    if pass_distance < short_dist + blend_span:
        blend = (pass_distance - short_dist) / blend_span
        return short_factor + (1.0 - short_factor) * blend
    return 1.0


def _v3_zone_max_pass_delta(x_start: float) -> float:
    x = float(np.clip(x_start, 0.0, FIELD_X))
    control_points = [
        (0.0, XT_V5_MAX_DELTA_DEF),
        (OPT_ATTACKING_TWO_THIRDS_X, XT_V5_MAX_DELTA_MID),
        (FINAL_THIRD_LINE_X, XT_V5_MAX_DELTA_ATT),
        (XT_V4_BOX_X_START, XT_V5_MAX_DELTA_BOX),
        (FIELD_X, XT_V5_MAX_DELTA_BOX),
    ]
    for idx in range(len(control_points) - 1):
        x0, cap0 = control_points[idx]
        x1, cap1 = control_points[idx + 1]
        if x <= x1:
            if x1 <= x0:
                return cap1
            t = float(_smoothstep(np.array([(x - x0) / (x1 - x0)]))[0])
            return cap0 + (cap1 - cap0) * t
    return control_points[-1][1]


def _adjust_heuristic_v3_pass_delta(row) -> float:
    if not row.is_won:
        return 0.0
    raw = float(row.xt_end - row.xt_start)
    if raw >= 0:
        adjusted = raw * _v3_short_pass_multiplier(row.pass_distance)
        return min(adjusted, _v3_zone_max_pass_delta(row.x_start))
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    if row.x_start < XT_V3_NEG_RECYCLE_X_MAX:
        adjusted = raw * (XT_V3_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    else:
        adjusted = raw
    if (
        row.x_start < XT_V3_PRESSURE_X_MAX
        and lat_start > XT_V3_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        adjusted += XT_V3_PRESSURE_ESCAPE_BONUS
    return adjusted


def apply_heuristic_v3_xt(df: pd.DataFrame) -> pd.DataFrame:
    fine = compute_heuristic_v3_fine_grid()
    out = df.copy()
    out["xt_start"] = out.apply(lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1)
    out["xt_end"] = out.apply(lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1)
    out["delta_xt"] = out.apply(_adjust_heuristic_v3_pass_delta, axis=1)
    return out


# ── xT HEURÍSTICO v3.1 (transições graduais) ─────────────────
def _gaussian_kernel_1d(sigma: float) -> np.ndarray:
    radius = max(1, int(np.ceil(3.0 * sigma)))
    xs = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (xs / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_smooth_2d(grid: np.ndarray, sigma_x: float, sigma_y: float) -> np.ndarray:
    out = grid
    if sigma_x > 0:
        kx = _gaussian_kernel_1d(sigma_x)
        out = np.apply_along_axis(lambda row: np.convolve(row, kx, mode="same"), axis=1, arr=out)
    if sigma_y > 0:
        ky = _gaussian_kernel_1d(sigma_y)
        out = np.apply_along_axis(lambda row: np.convolve(row, ky, mode="same"), axis=0, arr=out)
    return out


def _map_zonal_threat_v31(x: np.ndarray) -> np.ndarray:
    """Zonas com blend amplo e curvas smootherstep para transições homogêneas."""
    blend = XT_V31_ZONE_BLEND_WIDTH
    x = np.clip(x, 0.0, FIELD_X)
    threat_def = XT_V3_DEF_MAX * _smootherstep(np.clip(x / OPT_ATTACKING_TWO_THIRDS_X, 0.0, 1.0))
    mid_span = max(FINAL_THIRD_LINE_X - OPT_ATTACKING_TWO_THIRDS_X, 1.0)
    mid_t = np.clip((x - OPT_ATTACKING_TWO_THIRDS_X) / mid_span, 0.0, 1.0)
    threat_mid = XT_V3_DEF_MAX + (XT_V3_MID_MAX - XT_V3_DEF_MAX) * _smootherstep(mid_t)
    att_span = max(FIELD_X - FINAL_THIRD_LINE_X, 1.0)
    att_t = np.clip((x - FINAL_THIRD_LINE_X) / att_span, 0.0, 1.0)
    threat_att = XT_V3_MID_MAX + (XT_V3_ATT_BYLINE - XT_V3_MID_MAX) * _smootherstep(att_t)
    w_def = 1.0 - _smootherstep(np.clip((x - (OPT_ATTACKING_TWO_THIRDS_X - blend)) / blend, 0.0, 1.0))
    w_att = _smootherstep(np.clip((x - (FINAL_THIRD_LINE_X - blend)) / blend, 0.0, 1.0))
    w_mid = np.clip(1.0 - w_def - w_att, 0.0, 1.0)
    w_sum = w_def + w_mid + w_att + 1e-12
    return (w_def * threat_def + w_mid * threat_mid + w_att * threat_att) / w_sum


def _location_factor_v31(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    lat = _lateral_relative_position(y)
    depth = np.clip(
        (x - XT_V31_LAT_GATE_X) / (FIELD_X - XT_V31_LAT_GATE_X),
        0.0,
        1.0,
    )
    zone_gate = _smootherstep(depth)
    max_discount = XT_V31_LAT_DISC_MAX * zone_gate
    lateral_curve = _smootherstep(lat ** XT_V3_LAT_CURVE_POWER)
    return 1.0 - max_discount * lateral_curve


def _build_heuristic_v31_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    zonal = _map_zonal_threat_v31(Xc)
    surface = zonal * _location_factor_v31(Xc, Yc)
    surface = np.clip(surface, 0.0, XT_V3_SURFACE_MAX)
    smoothed = _gaussian_smooth_2d(surface, XT_V31_GAUSS_SIGMA_X, XT_V31_GAUSS_SIGMA_Y)
    return np.clip(smoothed, 0.0, XT_V3_SURFACE_MAX)


def _smooth_columns_1d(row: np.ndarray, kernel: tuple[float, ...]) -> np.ndarray:
    k = np.asarray(kernel, dtype=float)
    k = k / k.sum()
    pad = len(k) // 2
    padded = np.pad(row, (pad, pad), mode="edge")
    return np.convolve(padded, k, mode="valid")


def _limit_adjacent_column_step(
    grid: np.ndarray,
    max_step: float,
    *,
    att_col_start: int | None = None,
    max_step_att: float | None = None,
) -> np.ndarray:
    """Cap column-to-column jumps while preserving attack-direction growth."""
    out = grid.copy()
    att_start = att_col_start if att_col_start is not None else grid.shape[1]
    att_step = max_step_att if max_step_att is not None else max_step
    for iy in range(out.shape[0]):
        row = out[iy].copy()
        for ix in range(1, row.shape[0]):
            step = att_step if ix >= att_start else max_step
            lo = row[ix - 1]
            hi = lo + step
            if row[ix] > hi:
                row[ix] = hi
            elif row[ix] < lo:
                row[ix] = lo
        out[iy] = row
    return out


def _sample_display_grid(
    fine: np.ndarray,
    n_x: int = NX_XT,
    n_y: int = NY_XT,
    *,
    post_process=None,
) -> np.ndarray:
    grid = zone_xt_means(fine, n_x=n_x, n_y=n_y)
    if post_process is not None:
        grid = post_process(grid)
    return grid


@st.cache_data(show_spinner=False)
def compute_heuristic_v31_fine_grid(
    nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v31_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v31_xt_grid(n_x: int = NX_XT, n_y: int = NY_XT) -> np.ndarray:
    fine = compute_heuristic_v31_fine_grid()

    def _post(grid: np.ndarray) -> np.ndarray:
        smoothed = np.array([
            _smooth_columns_1d(grid[iy], XT_V31_COL_SMOOTH_KERNEL)
            for iy in range(grid.shape[0])
        ])
        return _limit_adjacent_column_step(
            smoothed,
            XT_V31_MAX_COL_STEP_DEF,
            att_col_start=XT_V31_ATT_COL_START,
            max_step_att=XT_V31_MAX_COL_STEP_ATT,
        )

    return _sample_display_grid(fine, n_x, n_y, post_process=_post)


@st.cache_data(show_spinner=False)
def compute_markov_fine_grid(
    nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY,
    model_key: str = "wsl",
) -> np.ndarray:
    """Upsample aligned Markov grid to the fine heuristic mesh."""
    grid = load_markov_model(model_key).xT
    y_coords = np.linspace(0.0, FIELD_Y, grid.shape[0])
    x_coords = np.linspace(0.0, FIELD_X, grid.shape[1])
    interp = RegularGridInterpolator(
        (y_coords, x_coords), grid, bounds_error=False, fill_value=0.0
    )
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    pts = np.column_stack([Yc.ravel(), Xc.ravel()])
    return interp(pts).reshape(ny, nx)


def _markov_bonus_from_fine(
    markov_fine: np.ndarray,
    *,
    bonus_max: float = XT_V32_QUADRANT_BONUS_MAX,
    bonus_power: float = XT_V32_BONUS_POWER,
    sigma_x: float = XT_V32_BONUS_SIGMA_X,
    sigma_y: float = XT_V32_BONUS_SIGMA_Y,
) -> np.ndarray:
    peak = max(float(markov_fine.max()), 1e-9)
    rel = (markov_fine / peak) ** bonus_power
    return _gaussian_smooth_2d(rel * bonus_max, sigma_x, sigma_y)


def _markov_quadrant_bonus_field(
    nx: int,
    ny: int,
    model_key: str = "top5",
    *,
    bonus_max: float = XT_V4_MARKOV_BONUS_MAX,
    bonus_power: float = XT_V4_MARKOV_BONUS_POWER,
) -> np.ndarray:
    """Small per-cell bonus from aligned Markov grid, upsampled to the fine mesh."""
    grid = load_markov_model(model_key).xT
    peak = max(float(grid.max()), 1e-9)
    rel = (grid / peak) ** bonus_power
    bonus_coarse = rel * bonus_max
    y_coords = np.linspace(0.0, FIELD_Y, grid.shape[0])
    x_coords = np.linspace(0.0, FIELD_X, grid.shape[1])
    interp = RegularGridInterpolator(
        (y_coords, x_coords), bonus_coarse, bounds_error=False, fill_value=0.0
    )
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    pts = np.column_stack([Yc.ravel(), Xc.ravel()])
    return interp(pts).reshape(ny, nx)


def _markov_final_third_envelope(
    Xc: np.ndarray,
    *,
    floor: float,
    blend: float = XT_V4_MARKOV_GATE_BLEND,
) -> np.ndarray:
    """Ramp Markov influence: low in the first two thirds, full in the final third."""
    t = _smootherstep(
        np.clip((Xc - (FINAL_THIRD_LINE_X - blend)) / max(blend, 1.0), 0.0, 1.0)
    )
    return floor + (1.0 - floor) * t


def _markov_top5_quadrant_bonus(
    nx: int,
    ny: int,
    Xc: np.ndarray,
    *,
    bonus_max: float,
    bonus_power: float,
    def_mid_floor: float | None = None,
    gate_blend: float = XT_V4_MARKOV_GATE_BLEND,
) -> np.ndarray:
    bonus = _markov_quadrant_bonus_field(
        nx, ny, model_key="top5", bonus_max=bonus_max, bonus_power=bonus_power
    )
    if def_mid_floor is not None:
        bonus *= _markov_final_third_envelope(Xc, floor=def_mid_floor, blend=gate_blend)
    return bonus


def _markov_bonus_field(
    nx: int,
    ny: int,
    model_key: str = "wsl",
    *,
    bonus_max: float = XT_V32_QUADRANT_BONUS_MAX,
    bonus_power: float = XT_V32_BONUS_POWER,
    sigma_x: float = XT_V32_BONUS_SIGMA_X,
    sigma_y: float = XT_V32_BONUS_SIGMA_Y,
) -> np.ndarray:
    """Per-zone bonus from Markov (emphasized peaks, less blur)."""
    markov = compute_markov_fine_grid(nx, ny, model_key=model_key)
    return _markov_bonus_from_fine(
        markov,
        bonus_max=bonus_max,
        bonus_power=bonus_power,
        sigma_x=sigma_x,
        sigma_y=sigma_y,
    )


def _build_heuristic_v32_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    """Flattened v3.1 base + stronger Markov quadrant bonus (WSL baseline)."""
    v31 = _build_heuristic_v31_threat_surface(Xc, Yc)
    peak = max(float(v31.max()), 1e-9)
    rel = (v31 / peak) ** XT_V32_BASE_SHAPE_GAMMA
    base = XT_V32_BASE_FLOOR + rel * XT_V32_BASE_SPREAD
    bonus = _markov_bonus_field(Xc.shape[1], Xc.shape[0], model_key="wsl")
    return np.clip(base + bonus, 0.0, XT_V32_SURFACE_MAX)


def _build_heuristic_v33_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    """Flattened v3.1 base + Markov bonus from validation winner."""
    v31 = _build_heuristic_v31_threat_surface(Xc, Yc)
    peak = max(float(v31.max()), 1e-9)
    rel = (v31 / peak) ** XT_V32_BASE_SHAPE_GAMMA
    base = XT_V32_BASE_FLOOR + rel * XT_V32_BASE_SPREAD
    bonus = _markov_bonus_field(Xc.shape[1], Xc.shape[0], model_key=get_v33_bonus_markov_key())
    return np.clip(base + bonus, 0.0, XT_V32_SURFACE_MAX)


@st.cache_data(show_spinner=False)
def compute_heuristic_v32_fine_grid(
    nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v32_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v32_xt_grid(n_x: int = NX_XT, n_y: int = NY_XT) -> np.ndarray:
    fine = compute_heuristic_v32_fine_grid()

    def _post(grid: np.ndarray) -> np.ndarray:
        smoothed = np.array([
            _smooth_columns_1d(grid[iy], XT_V32_COL_SMOOTH_KERNEL)
            for iy in range(grid.shape[0])
        ])
        return _limit_adjacent_column_step(
            smoothed,
            XT_V32_MAX_COL_STEP_DEF,
            att_col_start=XT_V32_ATT_COL_START,
            max_step_att=XT_V32_MAX_COL_STEP_ATT,
        )

    return _sample_display_grid(fine, n_x, n_y, post_process=_post)


@st.cache_data(show_spinner=False)
def compute_heuristic_v33_fine_grid(
    nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY,
    _bonus_key: str = "",
) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v33_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v33_xt_grid(n_x: int = NX_XT, n_y: int = NY_XT, _bonus_key: str = "") -> np.ndarray:
    fine = compute_heuristic_v33_fine_grid(_bonus_key=_bonus_key)

    def _post(grid: np.ndarray) -> np.ndarray:
        smoothed = np.array([
            _smooth_columns_1d(grid[iy], XT_V32_COL_SMOOTH_KERNEL)
            for iy in range(grid.shape[0])
        ])
        return _limit_adjacent_column_step(
            smoothed,
            XT_V32_MAX_COL_STEP_DEF,
            att_col_start=XT_V32_ATT_COL_START,
            max_step_att=XT_V32_MAX_COL_STEP_ATT,
        )

    return _sample_display_grid(fine, n_x, n_y, post_process=_post)


def _build_heuristic_v4_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    """v3.1 base + Top5 bonus quase nulo nos 2/3 defensivos, notável no último terço."""
    base = _build_heuristic_v31_threat_surface(Xc, Yc)
    bonus = _markov_top5_quadrant_bonus(
        Xc.shape[1],
        Xc.shape[0],
        Xc,
        bonus_max=XT_V4_MARKOV_BONUS_MAX,
        bonus_power=XT_V4_MARKOV_BONUS_POWER,
        def_mid_floor=XT_V4_MARKOV_DEF_MID_FLOOR,
    )
    return np.clip(base + bonus, 0.0, XT_V4_SURFACE_MAX)


def _build_heuristic_v41_threat_surface(Xc: np.ndarray, Yc: np.ndarray) -> np.ndarray:
    """v3.1 base + pequeno bônus Top5 uniforme em todo o campo (referência)."""
    base = _build_heuristic_v31_threat_surface(Xc, Yc)
    bonus = _markov_top5_quadrant_bonus(
        Xc.shape[1],
        Xc.shape[0],
        Xc,
        bonus_max=XT_V41_MARKOV_BONUS_MAX,
        bonus_power=XT_V41_MARKOV_BONUS_POWER,
    )
    return np.clip(base + bonus, 0.0, XT_V41_SURFACE_MAX)


def _heuristic_v4_post_process(grid: np.ndarray) -> np.ndarray:
    smoothed = np.array([
        _smooth_columns_1d(grid[iy], XT_V31_COL_SMOOTH_KERNEL)
        for iy in range(grid.shape[0])
    ])
    return _limit_adjacent_column_step(
        smoothed,
        XT_V31_MAX_COL_STEP_DEF,
        att_col_start=XT_V31_ATT_COL_START,
        max_step_att=XT_V31_MAX_COL_STEP_ATT,
    )


@st.cache_data(show_spinner=False)
def compute_heuristic_v4_fine_grid(
    nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v4_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v4_xt_grid(n_x: int = NX_XT, n_y: int = NY_XT) -> np.ndarray:
    fine = compute_heuristic_v4_fine_grid()
    return _sample_display_grid(fine, n_x, n_y, post_process=_heuristic_v4_post_process)


@st.cache_data(show_spinner=False)
def compute_heuristic_v41_fine_grid(
    nx: int = XT_V3_FINE_NX, ny: int = XT_V3_FINE_NY,
) -> np.ndarray:
    xe = np.linspace(0.0, FIELD_X, nx)
    ye = np.linspace(0.0, FIELD_Y, ny)
    Xc, Yc = np.meshgrid(xe, ye)
    return _build_heuristic_v41_threat_surface(Xc, Yc)


@st.cache_data(show_spinner=False)
def compute_heuristic_v41_xt_grid(n_x: int = NX_XT, n_y: int = NY_XT) -> np.ndarray:
    fine = compute_heuristic_v41_fine_grid()
    return _sample_display_grid(fine, n_x, n_y, post_process=_heuristic_v4_post_process)


def _variant_key_from_cols(cols: dict[str, str]) -> str:
    delta = cols.get("delta", "")
    if delta.endswith("_v41"):
        return "v41"
    if delta.endswith("_v4"):
        return "v4"
    if delta.endswith("_v33"):
        return "v33"
    if delta.endswith("_v32"):
        return "v32"
    if delta.endswith("_v31"):
        return "v31"
    return "v3"


def _primary_xt_cols() -> dict[str, str]:
    return _xt_column_set(XT_PRIMARY_VARIANT)


def _adjust_heuristic_v3_variant_pass_delta(row, start_col: str, end_col: str) -> float:
    if not row.is_won:
        return 0.0
    raw = float(getattr(row, end_col) - getattr(row, start_col))
    variant = (
        "v41" if start_col.endswith("_v41")
        else "v4" if start_col.endswith("_v4")
        else "v33" if start_col.endswith("_v33")
        else "v32" if start_col.endswith("_v32")
        else "v31" if start_col.endswith("_v31")
        else "v3"
    )
    scale = XT_V32_SCALE if variant in ("v32", "v33") else 1.0
    if raw >= 0:
        adjusted = raw * _v3_short_pass_multiplier(row.pass_distance)
        return min(adjusted, _v3_zone_max_pass_delta(row.x_start) * scale)
    lat_start = _lateral_frac(row.y_start)
    lat_end = _lateral_frac(row.y_end)
    if row.x_start < XT_V3_NEG_RECYCLE_X_MAX:
        adjusted = raw * (XT_V3_NEG_PENALTY_FACTOR if lat_end < lat_start else 1.0)
    else:
        adjusted = raw
    if (
        row.x_start < XT_V3_PRESSURE_X_MAX
        and lat_start > XT_V3_WIDE_FRAC
        and lat_end < lat_start - 0.12
    ):
        bonus = XT_V32_PRESSURE_ESCAPE_BONUS if variant in ("v32", "v33") else XT_V3_PRESSURE_ESCAPE_BONUS
        adjusted += bonus
    return adjusted


def _apply_heuristic_v3_variant_xt(
    df: pd.DataFrame, fine_fn, prefix: str,
) -> pd.DataFrame:
    fine = fine_fn()
    out = df.copy()
    start_col, end_col, delta_col = f"xt_start_{prefix}", f"xt_end_{prefix}", f"delta_xt_{prefix}"
    out[start_col] = out.apply(
        lambda r: xt_value_bilinear(r["x_start"], r["y_start"], fine), axis=1
    )
    out[end_col] = out.apply(
        lambda r: xt_value_bilinear(r["x_end"], r["y_end"], fine), axis=1
    )
    out[delta_col] = out.apply(
        lambda r: _adjust_heuristic_v3_variant_pass_delta(r, start_col, end_col), axis=1
    )
    return out


def apply_heuristic_v31_xt(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_heuristic_v3_variant_xt(df, compute_heuristic_v31_fine_grid, "v31")


def apply_heuristic_v32_xt(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_heuristic_v3_variant_xt(df, compute_heuristic_v32_fine_grid, "v32")


def apply_heuristic_v33_xt(df: pd.DataFrame) -> pd.DataFrame:
    bonus_key = get_v33_bonus_markov_key()

    def _fine_fn():
        return compute_heuristic_v33_fine_grid(_bonus_key=bonus_key)

    return _apply_heuristic_v3_variant_xt(df, _fine_fn, "v33")


def apply_heuristic_v4_xt(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_heuristic_v3_variant_xt(df, compute_heuristic_v4_fine_grid, "v4")


def apply_heuristic_v41_xt(df: pd.DataFrame) -> pd.DataFrame:
    return _apply_heuristic_v3_variant_xt(df, compute_heuristic_v41_fine_grid, "v41")


def _max_adjacent_col_jump_pct(grid: np.ndarray) -> float:
    if grid.shape[1] < 2:
        return 0.0
    jumps = np.abs(np.diff(grid, axis=1))
    return float(jumps.max() * 100.0)


def classify_xt_progressive_v3_adjusted(
    xt_start: float,
    delta_xt: float,
    x_end: float,
    pass_distance: float,
    *,
    variant: str = "v3",
) -> str:
    if delta_xt <= 0:
        return "none"
    if variant == "v32":
        prog_floor = XT_V32_PROG_FLOOR_CLASS
        high_floor = XT_V32_HIGH_FLOOR_CLASS
        prog_scale = XT_V32_PROG_SCALE_CLASS
        high_scale = XT_V32_HIGH_SCALE_CLASS
    elif variant == "v33":
        prog_floor = XT_V32_PROG_FLOOR_CLASS
        high_floor = XT_V32_HIGH_FLOOR_CLASS
        prog_scale = XT_V32_PROG_SCALE_CLASS
        high_scale = XT_V32_HIGH_SCALE_CLASS
    else:
        prog_floor = XT_V3_PROG_FLOOR_CLASS
        high_floor = XT_V3_HIGH_FLOOR_CLASS
        prog_scale = XT_V3_PROG_SCALE_CLASS
        high_scale = XT_V3_HIGH_SCALE_CLASS
    prog_thresh = max(prog_floor, prog_scale * (1.0 - xt_start))
    high_thresh = max(high_floor, high_scale * (1.0 - xt_start))
    if delta_xt <= prog_thresh:
        return "none"
    if delta_xt > high_thresh:
        return "highly"
    return "progressive"


def _row_if_won(row):
    if hasattr(row, "_replace"):
        return row._replace(is_won=True)
    if isinstance(row, pd.Series):
        won = row.copy()
        won["is_won"] = True
        return won
    return row


def _xt_column_set(variant: str = "v3") -> dict[str, str]:
    if variant == "v31":
        return {"start": "xt_start_v31", "end": "xt_end_v31", "delta": "delta_xt_v31"}
    if variant == "v32":
        return {"start": "xt_start_v32", "end": "xt_end_v32", "delta": "delta_xt_v32"}
    if variant == "v33":
        return {"start": "xt_start_v33", "end": "xt_end_v33", "delta": "delta_xt_v33"}
    if variant == "v4":
        return {"start": "xt_start_v4", "end": "xt_end_v4", "delta": "delta_xt_v4"}
    if variant == "v41":
        return {"start": "xt_start_v41", "end": "xt_end_v41", "delta": "delta_xt_v41"}
    return {"start": "xt_start", "end": "xt_end", "delta": "delta_xt"}


def hypothetical_delta_xt(row, cols: dict[str, str] | None = None) -> float:
    cols = cols or _xt_column_set("v3")
    return _adjust_heuristic_v3_variant_pass_delta(_row_if_won(row), cols["start"], cols["end"])


def progressive_delta_for_attempt(row, cols: dict[str, str] | None = None) -> float:
    cols = cols or _xt_column_set("v3")
    if row.is_won:
        return float(getattr(row, cols["delta"]))
    return hypothetical_delta_xt(row, cols)


def is_impact_attempt(row, cols: dict[str, str] | None = None) -> bool:
    cols = cols or _primary_xt_cols()
    variant = _variant_key_from_cols(cols)
    delta = progressive_delta_for_attempt(row, cols)
    return classify_xt_progressive_v3_adjusted(
        getattr(row, cols["start"]), delta, row.x_end, row.pass_distance, variant=variant
    ) in ("progressive", "highly")


def is_high_impact_attempt(row, cols: dict[str, str] | None = None) -> bool:
    cols = cols or _primary_xt_cols()
    variant = _variant_key_from_cols(cols)
    delta = progressive_delta_for_attempt(row, cols)
    return (
        classify_xt_progressive_v3_adjusted(
            getattr(row, cols["start"]), delta, row.x_end, row.pass_distance, variant=variant
        )
        == "highly"
    )


def is_impact_pass_attempt(row, cols: dict[str, str] | None = None) -> bool:
    cols = cols or _primary_xt_cols()
    if not row.has_end:
        return False
    if not pass_approaches_goal(row.x_start, row.y_start, row.x_end, row.y_end):
        return False
    return is_impact_attempt(row, cols)


def is_high_impact_pass_attempt(row, cols: dict[str, str] | None = None) -> bool:
    cols = cols or _primary_xt_cols()
    if not row.has_end:
        return False
    if not pass_approaches_goal(row.x_start, row.y_start, row.x_end, row.y_end):
        return False
    return is_high_impact_attempt(row, cols)


def classification_accuracy(df: pd.DataFrame, success_col: str, attempt_fn) -> dict:
    attempts = df.apply(attempt_fn, axis=1)
    successful = int(df[success_col].astype(bool).sum())
    attempted = int(attempts.sum())
    accuracy_pct = (successful / attempted * 100.0) if attempted else 0.0
    return {
        "successful": successful,
        "attempted": attempted,
        "accuracy_pct": round(accuracy_pct, 1),
    }


def classification_accuracy_fn(
    df: pd.DataFrame, attempt_fn, success_fn,
) -> dict:
    if df.empty:
        return {"successful": 0, "attempted": 0, "accuracy_pct": 0.0}
    attempts = df.apply(attempt_fn, axis=1)
    successful = int(df.apply(success_fn, axis=1).sum())
    attempted = int(attempts.sum())
    accuracy_pct = (successful / attempted * 100.0) if attempted else 0.0
    return {
        "successful": successful,
        "attempted": attempted,
        "accuracy_pct": round(accuracy_pct, 1),
    }


def enrich_with_xt_v3(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich pass/carry rows with heuristic xT v4 columns."""
    out = df.copy()
    out["pass_distance"] = np.where(
        out["has_end"],
        np.sqrt((out["x_end"] - out["x_start"]) ** 2 + (out["y_end"] - out["y_start"]) ** 2),
        0.0,
    )
    out["is_won"] = out["is_success"].astype(bool)
    carry_mask = out["category"] == "ball-carries"
    out.loc[carry_mask, "is_won"] = out.loc[carry_mask, "has_end"]

    for col in ("xt_start_v4", "xt_end_v4", "delta_xt_v4"):
        out[col] = 0.0
    out["progressive"] = False
    out["impact_pass"] = False
    out["high_impact_pass"] = False
    out["impact_carry"] = False
    out["high_impact_carry"] = False

    xt_mask = out["category"].isin(["passes", "ball-carries"]) & out["has_end"]
    if not xt_mask.any():
        return out

    xt_df_v4 = apply_heuristic_v4_xt(out.loc[xt_mask].copy())
    out.loc[xt_mask, ["xt_start_v4", "xt_end_v4", "delta_xt_v4"]] = xt_df_v4[
        ["xt_start_v4", "xt_end_v4", "delta_xt_v4"]
    ].values

    pass_mask = out["category"] == "passes"
    cols_primary = _primary_xt_cols()
    for idx, row in out.loc[pass_mask].iterrows():
        out.at[idx, "progressive"] = bool(
            row.is_won
            and is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)
        )
        out.at[idx, "impact_pass"] = bool(row.is_won and is_impact_pass_attempt(row, cols_primary))
        out.at[idx, "high_impact_pass"] = bool(row.is_won and is_high_impact_pass_attempt(row, cols_primary))

    carry_mask = out["category"] == "ball-carries"
    for idx, row in out.loc[carry_mask].iterrows():
        out.at[idx, "impact_carry"] = bool(row.is_won and is_impact_attempt(row, cols_primary))
        out.at[idx, "high_impact_carry"] = bool(row.is_won and is_high_impact_attempt(row, cols_primary))

    return out


def ensure_xt_model_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Backfill v4 xT columns when serving cached frames from older app versions."""
    if df.empty:
        return df

    out = df.copy()
    xt_mask = out["category"].isin(["passes", "ball-carries"]) & out["has_end"]
    delta_col = "delta_xt_v4"
    if delta_col not in out.columns:
        for col in ("xt_start_v4", "xt_end_v4", delta_col):
            out[col] = 0.0
        if xt_mask.any():
            xt_df = apply_heuristic_v4_xt(out.loc[xt_mask].copy())
            out.loc[xt_mask, ["xt_start_v4", "xt_end_v4", delta_col]] = xt_df[
                ["xt_start_v4", "xt_end_v4", delta_col]
            ].values
    return out


def _safe_col_sum(df: pd.DataFrame, col: str) -> float:
    if col not in df.columns or df.empty:
        return 0.0
    return float(pd.to_numeric(df[col], errors="coerce").fillna(0.0).sum())


# ── DATA LOADING ─────────────────────────────────────────────
def discover_csv_files(base_dir: Path | None = None) -> list[Path]:
    root = base_dir or Path(__file__).resolve().parent
    return sorted(
        p for p in root.glob("*.csv")
        if p.name not in EXCLUDED_CSV
    )


def _match_slug_from_csv(path: Path) -> str:
    stem = path.stem
    if "-vs " in stem:
        return stem.split("-vs ", 1)[1]
    return stem


def _csv_needs_x_flip(path: Path) -> bool:
    return _match_slug_from_csv(path) in CSV_X_FLIP_MATCHES


def load_player_csv(path: Path, *, flip_x: bool = False) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"category", "eventActionType", "start_x", "start_y"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Colunas ausentes em {path.name}: {', '.join(sorted(missing))}")

    rows = []

    for idx, row in frame.iterrows():
        sx, sy = wyscout_to_statsbomb(
            float(row["start_x"]), float(row["start_y"]), flip_x=flip_x
        )
        has_end = _has_coords(row, "end")
        ex = ey = np.nan
        if has_end:
            ex, ey = wyscout_to_statsbomb(
                float(row["end_x"]), float(row["end_y"]), flip_x=flip_x
            )

        rows.append(
            {
                "category": str(row["category"]).strip().lower(),
                "action_type": str(row["eventActionType"]).strip().lower(),
                "is_home": _parse_bool(row.get("isHome")),
                "is_success": _parse_bool(row.get("outcome")),
                "is_key_pass": _parse_bool(row.get("keypass")),
                "is_long_ball": _parse_bool(row.get("isLongBall")),
                "x_start": sx,
                "y_start": sy,
                "x_end": ex,
                "y_end": ey,
                "has_end": has_end,
                "player": path.stem.replace("_", " ").title(),
                "source_file": path.name,
                "row_id": idx + 1,
            }
        )

    return enrich_with_xt_v3(pd.DataFrame(rows))


def load_player_all_matches(player: dict, base_dir: Path | None = None) -> pd.DataFrame:
    """Aggregate all match CSVs for a player entry in PLAYERS."""
    root = base_dir or Path(__file__).resolve().parent
    pattern = player.get("glob", f"{player['code']}-vs *.csv")
    files = sorted(root.glob(pattern))
    if not files:
        return pd.DataFrame()

    frames = []
    for path in files:
        flip_x = _csv_needs_x_flip(path)
        match_df = load_player_csv(path, flip_x=flip_x)
        match_df["match"] = _match_slug_from_csv(path)
        frames.append(match_df)

    combined = pd.concat(frames, ignore_index=True)
    combined["player"] = player["name"]
    return combined


def top_deltaxt_actions(
    df: pd.DataFrame, n: int = TOP_DELTAXT_N, delta_col: str = "delta_xt"
) -> pd.DataFrame:
    """Top N passes and carries by positive delta xT."""
    actions = df[
        df["category"].isin(["passes", "ball-carries"]) & df["has_end"]
    ].copy()
    if delta_col not in actions.columns:
        return pd.DataFrame()
    actions = actions[actions[delta_col] > 0]
    if actions.empty:
        return actions
    return actions.nlargest(n, delta_col)


def _safe_ratio(numerator: float, denominator: int, *, decimals: int = 3) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / denominator, decimals)


def _fmt_count(value: int | float | None) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.0f}" if value.is_integer() else f"{value:.1f}"
    return f"{int(value)}"


def _fmt_decimal(value: float | None, *, decimals: int = 2) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}"


def _fmt_pct(value: float | None, *, decimals: int = 1) -> str:
    if value is None:
        return "—"
    return f"{value:.{decimals}f}%"


def _count_action(df: pd.DataFrame, action_type: str) -> int:
    if df.empty:
        return 0
    return int((df["action_type"] == action_type).sum())


def _is_prog_wyscout_row(row) -> bool:
    if not row.has_end:
        return False
    return is_progressive_wyscout(row.x_start, row.y_start, row.x_end, row.y_end)


# ── STATS ────────────────────────────────────────────────────
def compute_player_stats(df: pd.DataFrame, variant: str | None = None) -> dict:
    """Player stats for a heuristic xT variant (default: primary model)."""
    passes = df[df["category"] == "passes"]
    carries = df[df["category"] == "ball-carries"]
    empty_cls = {"successful": 0, "attempted": 0, "accuracy_pct": 0.0}
    cols = _xt_column_set(variant or XT_PRIMARY_VARIANT)
    delta_col, end_col = cols["delta"], cols["end"]

    total_passes = len(passes)
    completed_passes = passes[passes["is_success"]] if total_passes else passes.iloc[0:0]
    carries_total = len(carries)

    general = {
        "passes_total": total_passes,
        "passes_completed": int(len(completed_passes)),
        "passes_accuracy_pct": round(len(completed_passes) / total_passes * 100.0, 1) if total_passes else 0.0,
        "key_passes": int(passes["is_key_pass"].sum()) if total_passes else 0,
        "crosses": _count_action(passes, "cross"),
        "long_balls": int(passes["is_long_ball"].sum()) if total_passes else 0,
        "carries_total": carries_total,
        "dribbles": int((df["category"] == "dribbles").sum()),
        "tackles": _count_action(df, "tackle"),
        "interceptions": _count_action(df, "interception"),
        "clearances": _count_action(df, "clearance"),
        "ball_recoveries": _count_action(df, "ball-recovery"),
        "blocks": _count_action(df, "block"),
        "defensive_total": int((df["category"] == "defensive").sum()),
        "shots": None,
        "xg": None,
        "assists": None,
        "xa": None,
        "total_actions": len(df),
    }

    if total_passes == 0:
        return {
            **general,
            "accuracy_pct": 0.0,
            "progressive_wyscout": empty_cls.copy(),
            "impact_pass": empty_cls.copy(),
            "high_impact_pass": empty_cls.copy(),
            "impact_carry": empty_cls.copy(),
            "high_impact_carry": empty_cls.copy(),
            "sum_dxt_passes": 0.0,
            "sum_dxt_passes_offensive": 0.0,
            "sum_dxt_carries": 0.0,
            "sum_xt_end_passes": 0.0,
            "sum_xt_end_final_third": 0.0,
            "sum_xt_end_long_balls": 0.0,
            "sum_xt_end_top10_passes": 0.0,
            "sum_xt_end_key_passes": 0.0,
            "sum_xt_end_impact_passes": 0.0,
            "pos_pct": 0.0,
            "xt_per_pass": 0.0,
            "dxt_per_pass": 0.0,
            "xt_per_pass_final_third": 0.0,
            "xt_per_prog_pass": 0.0,
            "xt_per_impact_pass": 0.0,
            "xt_per_long_ball": 0.0,
            "by_action_type": df.groupby("action_type").size().to_dict() if not df.empty else {},
        }

    successful = int(passes["is_success"].sum())
    accuracy = successful / total_passes * 100.0
    progressive_wyscout = classification_accuracy(
        passes,
        "progressive",
        _is_prog_wyscout_row,
    )
    impact_pass = classification_accuracy_fn(
        passes,
        lambda r: is_impact_pass_attempt(r, cols),
        lambda r: bool(r.is_won and is_impact_pass_attempt(r, cols)),
    )
    high_impact_pass = classification_accuracy_fn(
        passes,
        lambda r: is_high_impact_pass_attempt(r, cols),
        lambda r: bool(r.is_won and is_high_impact_pass_attempt(r, cols)),
    )
    impact_carry = classification_accuracy_fn(
        carries,
        lambda r: is_impact_attempt(r, cols),
        lambda r: bool(r.is_won and is_impact_attempt(r, cols)),
    )
    high_impact_carry = classification_accuracy_fn(
        carries,
        lambda r: is_high_impact_attempt(r, cols),
        lambda r: bool(r.is_won and is_high_impact_attempt(r, cols)),
    )

    xt_actions = df[df["category"].isin(["passes", "ball-carries"]) & df["has_end"]]
    pos_count = int((xt_actions[delta_col] > 0).sum())
    pos_pct = (pos_count / len(xt_actions) * 100.0) if len(xt_actions) else 0.0

    sum_dxt_passes = float(passes[delta_col].sum())
    offensive_passes = passes[passes["x_start"] >= HALF_LINE_X]
    sum_dxt_passes_offensive = (
        float(offensive_passes[delta_col].sum()) if not offensive_passes.empty else 0.0
    )
    sum_dxt_carries = float(carries[delta_col].sum())
    sum_xt_end_passes = float(completed_passes[end_col].sum()) if not completed_passes.empty else 0.0
    completed_long_balls = completed_passes[completed_passes["is_long_ball"]]
    sum_xt_end_long_balls = (
        float(completed_long_balls[end_col].sum()) if not completed_long_balls.empty else 0.0
    )

    final_third_won = df[
        df["category"].isin(["passes", "ball-carries"])
        & df["has_end"]
        & df["is_won"]
        & (df["x_end"] >= FINAL_THIRD_LINE_X)
    ]
    sum_xt_end_final_third = float(final_third_won[end_col].sum()) if not final_third_won.empty else 0.0

    prog_success_mask = passes.apply(
        lambda r: bool(r.is_success and _is_prog_wyscout_row(r)), axis=1
    )
    prog_success = passes[prog_success_mask]
    impact_success_mask = passes.apply(
        lambda r: bool(r.is_won and is_impact_pass_attempt(r, cols)), axis=1
    )
    impact_success = passes[impact_success_mask]

    completed_ft = completed_passes[completed_passes["x_end"] >= FINAL_THIRD_LINE_X]
    sum_xt_end_passes_final_third = (
        float(completed_ft[end_col].sum()) if not completed_ft.empty else 0.0
    )

    positive_completed = completed_passes[
        completed_passes["has_end"] & (completed_passes[delta_col] > 0)
    ]
    top10_passes = (
        positive_completed.nlargest(TOP_DELTAXT_N, delta_col)
        if not positive_completed.empty
        else completed_passes.iloc[0:0]
    )
    sum_xt_end_top10_passes = (
        float(top10_passes[end_col].sum()) if not top10_passes.empty else 0.0
    )

    key_passes_completed = completed_passes[completed_passes["is_key_pass"]]
    sum_xt_end_key_passes = (
        float(key_passes_completed[end_col].sum()) if not key_passes_completed.empty else 0.0
    )
    sum_xt_end_impact_passes = (
        float(impact_success[end_col].sum()) if not impact_success.empty else 0.0
    )

    return {
        **general,
        "accuracy_pct": accuracy,
        "progressive_wyscout": progressive_wyscout,
        "impact_pass": impact_pass,
        "high_impact_pass": high_impact_pass,
        "impact_carry": impact_carry,
        "high_impact_carry": high_impact_carry,
        "sum_dxt_passes": sum_dxt_passes,
        "sum_dxt_passes_offensive": sum_dxt_passes_offensive,
        "sum_dxt_carries": sum_dxt_carries,
        "sum_xt_end_passes": sum_xt_end_passes,
        "sum_xt_end_final_third": sum_xt_end_final_third,
        "sum_xt_end_long_balls": sum_xt_end_long_balls,
        "sum_xt_end_top10_passes": sum_xt_end_top10_passes,
        "sum_xt_end_key_passes": sum_xt_end_key_passes,
        "sum_xt_end_impact_passes": sum_xt_end_impact_passes,
        "pos_pct": pos_pct,
        "xt_per_pass": _safe_ratio(sum_xt_end_passes, len(completed_passes)),
        "dxt_per_pass": _safe_ratio(sum_dxt_passes, len(completed_passes)),
        "xt_per_pass_final_third": _safe_ratio(sum_xt_end_passes_final_third, len(completed_ft)),
        "xt_per_prog_pass": _safe_ratio(float(prog_success[delta_col].sum()), len(prog_success)),
        "xt_per_impact_pass": _safe_ratio(sum_xt_end_impact_passes, len(impact_success)),
        "xt_per_long_ball": _safe_ratio(sum_xt_end_long_balls, len(completed_long_balls)),
        "by_action_type": df.groupby("action_type").size().to_dict(),
    }


def _wc_v4_player_summary(df: pd.DataFrame, player: dict) -> dict:
    """Compact v4 benchmark row for one WC player."""
    stats = compute_player_stats(df, variant="v4")
    xt_actions = df[_xt_action_mask(df)]
    delta = pd.to_numeric(xt_actions.get("delta_xt_v4"), errors="coerce").fillna(0.0)
    pos_pct = float((delta > 0).mean() * 100.0) if len(delta) else 0.0
    return {
        "Jogador": player["name"],
        "Posição": player.get("position", "—"),
        "Σ ΔxT": round(stats["sum_dxt_passes"] + stats["sum_dxt_carries"], 3),
        "Σ xT passe": round(stats["sum_xt_end_passes"], 3),
        "ΔxT/passe": round(stats["dxt_per_pass"], 4),
        "xT/passe": round(stats["xt_per_pass"], 4),
        "Passes": stats["passes_completed"],
        "% ΔxT+": round(pos_pct, 1),
    }


def _laliga_player_label(player: dict, *, show_season_ref: bool) -> str:
    name = player.get("player_name", "—")
    if show_season_ref:
        season = player.get("season_label") or player.get("season_key", "")
        if season:
            return f"{name} · {season}"
    return name


def _laliga_enrich_player_metrics(player: dict) -> dict:
    """Attach derived metrics and flag legacy rows missing pass/carry split."""
    enriched = dict(player)
    passes = int(enriched.get("passes") or 0)
    carries = int(enriched.get("carries") or 0)
    total_actions = passes + carries
    enriched["total_actions"] = enriched.get("total_actions") or total_actions

    if "sum_delta_xt_passes" not in enriched:
        enriched["_legacy_metrics"] = True
        return enriched

    enriched["_legacy_metrics"] = False
    sum_delta = float(enriched.get("sum_delta_xt") or 0.0)
    sum_delta_passes = float(enriched.get("sum_delta_xt_passes") or 0.0)
    sum_delta_carries = float(enriched.get("sum_delta_xt_carries") or 0.0)

    if enriched.get("pass_share_of_actions") is None and total_actions:
        enriched["pass_share_of_actions"] = round(passes / total_actions, 4)
    if enriched.get("carry_share_of_actions") is None and total_actions:
        enriched["carry_share_of_actions"] = round(carries / total_actions, 4)
    if enriched.get("pass_share_of_dxt") is None and sum_delta:
        enriched["pass_share_of_dxt"] = round(sum_delta_passes / sum_delta, 4)
    if enriched.get("carry_share_of_dxt") is None and sum_delta:
        enriched["carry_share_of_dxt"] = round(sum_delta_carries / sum_delta, 4)
    if enriched.get("dxt_per_action") is None and total_actions:
        enriched["dxt_per_action"] = round(sum_delta / total_actions, 4)
    return enriched


def _laliga_prepare_players(players: list[dict]) -> list[dict]:
    """Exclude goalkeepers and attach aggregated position group + derived metrics."""
    prepared: list[dict] = []
    for player in players:
        pos = player.get("position", "—")
        if not is_outfield_position(pos):
            continue
        enriched = _laliga_enrich_player_metrics(player)
        enriched["position_group"] = player.get("position_group") or position_group(pos)
        prepared.append(enriched)
    return prepared


def _laliga_has_extended_metrics(players: list[dict]) -> bool:
    return bool(players) and not any(p.get("_legacy_metrics") for p in players)


def _laliga_stats_to_dataframe(players: list[dict], *, show_season_ref: bool) -> pd.DataFrame:
    rows: list[dict] = []
    extended = _laliga_has_extended_metrics(players)
    for rank, player in enumerate(players, start=1):
        row = {
            "#": rank,
            "Jogador": _laliga_player_label(player, show_season_ref=show_season_ref),
            "Grupo": player.get("position_group", "—"),
            "Σ ΔxT": player.get("sum_delta_xt"),
            "Passes": player.get("passes"),
            "Conduções": player.get("carries"),
            "ΔxT/passe": player.get("dxt_per_pass"),
            "xT/passe": player.get("xt_per_pass"),
        }
        if extended:
            row.update(
                {
                    "Σ ΔxT passe": player.get("sum_delta_xt_passes"),
                    "Σ ΔxT condução": player.get("sum_delta_xt_carries"),
                    "ΔxT/condução": player.get("dxt_per_carry"),
                    "% ΔxT passe": player.get("pass_share_of_dxt"),
                }
            )
        else:
            row["Σ xT passe"] = player.get("sum_xt_end_passes")
        rows.append(row)
    return pd.DataFrame(rows)


def _laliga_detailed_metrics_dataframe(players: list[dict], *, show_season_ref: bool) -> pd.DataFrame:
    rows: list[dict] = []
    for player in players:
        rows.append(
            {
                "Jogador": _laliga_player_label(player, show_season_ref=show_season_ref),
                "Grupo": player.get("position_group", "—"),
                "Σ ΔxT": player.get("sum_delta_xt"),
                "Σ ΔxT passe": player.get("sum_delta_xt_passes"),
                "Σ ΔxT condução": player.get("sum_delta_xt_carries"),
                "Σ xT passe": player.get("sum_xt_end_passes"),
                "Σ xT condução": player.get("sum_xt_end_carries"),
                "Passes": player.get("passes"),
                "Conduções": player.get("carries"),
                "Ações": player.get("total_actions"),
                "ΔxT/passe": player.get("dxt_per_pass"),
                "ΔxT/condução": player.get("dxt_per_carry"),
                "ΔxT/ação": player.get("dxt_per_action"),
                "xT/passe": player.get("xt_per_pass"),
                "xT/condução": player.get("xt_per_carry"),
                "Gap passe": player.get("progression_gap_pass"),
                "% ações passe": player.get("pass_share_of_actions"),
                "% ΔxT passe": player.get("pass_share_of_dxt"),
                "% ΔxT condução": player.get("carry_share_of_dxt"),
                "% ações ΔxT>0": player.get("positive_action_rate"),
                "% passes ΔxT>0": player.get("positive_pass_rate"),
                "% conduções ΔxT>0": player.get("positive_carry_rate"),
            }
        )
    return pd.DataFrame(rows)


def _laliga_rankings_by_position(players: list[dict], *, show_season_ref: bool) -> pd.DataFrame:
    if not players:
        return pd.DataFrame()
    by_pos: dict[str, list[dict]] = {g: [] for g in POSITION_GROUPS_ORDER}
    for player in players:
        group = player.get("position_group") or position_group(player.get("position"))
        if group and group in by_pos:
            by_pos[group].append(player)

    ranked_rows: list[dict] = []
    for group in POSITION_GROUPS_ORDER:
        group_players = sorted(by_pos[group], key=lambda p: p.get("sum_delta_xt", 0.0), reverse=True)
        for pos_rank, player in enumerate(group_players, start=1):
            ranked_rows.append(
                {
                    "Grupo": group,
                    "# no grupo": pos_rank,
                    "Jogador": _laliga_player_label(player, show_season_ref=show_season_ref),
                    "Σ ΔxT": player.get("sum_delta_xt"),
                    "Σ ΔxT passe": player.get("sum_delta_xt_passes"),
                    "Σ ΔxT condução": player.get("sum_delta_xt_carries"),
                    "ΔxT/passe": player.get("dxt_per_pass"),
                    "ΔxT/condução": player.get("dxt_per_carry"),
                    "Passes": player.get("passes"),
                    "Conduções": player.get("carries"),
                }
            )
    return pd.DataFrame(ranked_rows)


def _laliga_plotly_scatter_layout(fig, *, title: str, x_label: str, y_label: str) -> None:
    """Apply shared dark-theme layout to Ligas scatter charts."""
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left", font=dict(size=16, color="#f3f4f6")),
        paper_bgcolor="#1a1a2e",
        plot_bgcolor="#1a1a2e",
        font=dict(family="Inter, system-ui, sans-serif", color="#d1d5db", size=12),
        margin=dict(l=56, r=28, t=72, b=56),
        height=520,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(26,26,46,0)",
            bordercolor="rgba(255,255,255,0)",
            font=dict(size=11, color="#e5e7eb"),
        ),
        hovermode="closest",
        hoverlabel=dict(
            bgcolor="#111827",
            bordercolor="#374151",
            font=dict(size=13, color="#f9fafb"),
            align="left",
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=13, color="#9ca3af")),
            tickfont=dict(color="#9ca3af"),
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False,
            showline=True,
            linecolor="rgba(255,255,255,0.18)",
            ticks="outside",
            tickcolor="rgba(255,255,255,0.18)",
        ),
        yaxis=dict(
            title=dict(text=y_label, font=dict(size=13, color="#9ca3af")),
            tickfont=dict(color="#9ca3af"),
            gridcolor="rgba(255,255,255,0.07)",
            zeroline=False,
            showline=True,
            linecolor="rgba(255,255,255,0.18)",
            ticks="outside",
            tickcolor="rgba(255,255,255,0.18)",
        ),
    )


def _laliga_plotly_scatter_chart(
    players: list[dict],
    *,
    title: str,
    show_season_ref: bool,
    selected_groups: list[str],
    x_key: str,
    y_key: str,
    x_label: str,
    y_label: str,
    x_hover_fmt: str,
    y_hover_fmt: str,
):
    """Interactive scatter by position group with hover player names."""
    import plotly.graph_objects as go

    filtered = [p for p in players if p.get("position_group") in selected_groups]
    if not filtered:
        return None

    fig = go.Figure()
    for group in POSITION_GROUPS_ORDER:
        if group not in selected_groups:
            continue
        subset = [p for p in filtered if p.get("position_group") == group]
        if not subset:
            continue
        names = [_laliga_player_label(p, show_season_ref=show_season_ref) for p in subset]
        fig.add_trace(
            go.Scatter(
                x=[p[x_key] for p in subset],
                y=[p[y_key] for p in subset],
                mode="markers",
                name=group,
                text=names,
                customdata=[
                    [
                        p.get("position", "—"),
                        p.get("passes", 0),
                        p.get("carries", 0),
                        p.get("sum_delta_xt", 0.0),
                        p.get("sum_delta_xt_passes") or 0.0,
                        p.get("sum_delta_xt_carries") or 0.0,
                        p.get("dxt_per_pass", 0.0),
                        p.get("dxt_per_carry") or 0.0,
                        p.get("xt_per_pass", 0.0),
                        p.get("sum_xt_end_passes", 0.0),
                        p.get("pass_share_of_dxt") or 0.0,
                    ]
                    for p in subset
                ],
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    f"<span style='color:#9ca3af'>Grupo</span>: {group}<br>"
                    "<span style='color:#9ca3af'>Posição</span>: %{customdata[0]}<br>"
                    f"<span style='color:#9ca3af'>{x_label}</span>: {x_hover_fmt}<br>"
                    f"<span style='color:#9ca3af'>{y_label}</span>: {y_hover_fmt}<br>"
                    "<span style='color:#9ca3af'>Passes</span>: %{customdata[1]:,}<br>"
                    "<span style='color:#9ca3af'>Conduções</span>: %{customdata[2]:,}<br>"
                    "<span style='color:#9ca3af'>Σ ΔxT</span>: %{customdata[3]:.3f}<br>"
                    "<span style='color:#9ca3af'>Σ ΔxT passe</span>: %{customdata[4]:.3f}<br>"
                    "<span style='color:#9ca3af'>Σ ΔxT condução</span>: %{customdata[5]:.3f}<br>"
                    "<span style='color:#9ca3af'>ΔxT/passe</span>: %{customdata[6]:.4f}<br>"
                    "<span style='color:#9ca3af'>ΔxT/condução</span>: %{customdata[7]:.4f}<br>"
                    "<span style='color:#9ca3af'>xT/passe</span>: %{customdata[8]:.4f}<br>"
                    "<span style='color:#9ca3af'>Σ xT passe</span>: %{customdata[9]:.2f}<br>"
                    "<span style='color:#9ca3af'>% ΔxT via passe</span>: %{customdata[10]:.1%}"
                    "<extra></extra>"
                ),
                marker=dict(
                    size=11,
                    color=GROUP_COLORS.get(group, "#aaaaaa"),
                    opacity=0.88,
                    line=dict(width=1.2, color="rgba(255,255,255,0.82)"),
                ),
            )
        )

    _laliga_plotly_scatter_layout(fig, title=title, x_label=x_label, y_label=y_label)
    return fig


def _laliga_scatter_chart(
    players: list[dict],
    *,
    title: str,
    show_season_ref: bool,
    selected_groups: list[str],
):
    """Interactive scatter: Σ ΔxT vs passes with hover player names."""
    return _laliga_plotly_scatter_chart(
        players,
        title=title,
        show_season_ref=show_season_ref,
        selected_groups=selected_groups,
        x_key="passes",
        y_key="sum_delta_xt",
        x_label="Passes na temporada",
        y_label="Σ ΔxT",
        x_hover_fmt="%{x:,}",
        y_hover_fmt="%{y:.3f}",
    )


def _laliga_rate_scatter_chart(
    players: list[dict],
    *,
    title: str,
    show_season_ref: bool,
    selected_groups: list[str],
):
    """Interactive scatter: ΔxT/passe vs xT/passe with hover player names."""
    return _laliga_plotly_scatter_chart(
        players,
        title=title,
        show_season_ref=show_season_ref,
        selected_groups=selected_groups,
        x_key="dxt_per_pass",
        y_key="xt_per_pass",
        x_label="ΔxT/passe",
        y_label="xT/passe",
        x_hover_fmt="%{x:.4f}",
        y_hover_fmt="%{y:.4f}",
    )


def _laliga_threat_progression_scatter_chart(
    players: list[dict],
    *,
    title: str,
    show_season_ref: bool,
    selected_groups: list[str],
):
    """Σ xT no destino dos passes vs Σ ΔxT total — ameaça entregue vs progressão."""
    return _laliga_plotly_scatter_chart(
        players,
        title=title,
        show_season_ref=show_season_ref,
        selected_groups=selected_groups,
        x_key="sum_xt_end_passes",
        y_key="sum_delta_xt",
        x_label="Σ xT passe (destino)",
        y_label="Σ ΔxT (total)",
        x_hover_fmt="%{x:.2f}",
        y_hover_fmt="%{y:.3f}",
    )


def _laliga_style_scatter_chart(
    players: list[dict],
    *,
    title: str,
    show_season_ref: bool,
    selected_groups: list[str],
):
    """Passes vs conduções — perfil de estilo com a bola."""
    return _laliga_plotly_scatter_chart(
        players,
        title=title,
        show_season_ref=show_season_ref,
        selected_groups=selected_groups,
        x_key="passes",
        y_key="carries",
        x_label="Passes",
        y_label="Conduções",
        x_hover_fmt="%{x:,}",
        y_hover_fmt="%{y:,}",
    )


def _render_laliga_plotly_chart(chart_fn, **kwargs) -> None:
    """Render a Ligas Plotly chart with import-error handling."""
    scatter_fig = None
    scatter_error = None
    try:
        scatter_fig = chart_fn(**kwargs)
    except ImportError:
        scatter_error = (
            "O pacote **plotly** não está instalado neste ambiente. "
            "Confirme que `plotly` consta em `requirements.txt` e reinicie o app "
            "(no Streamlit Cloud: *Manage app* → *Reboot app*)."
        )
    if scatter_error:
        st.error(scatter_error)
    elif scatter_fig is None:
        st.info("Selecione ao menos um grupo de posição para exibir o gráfico.")
    else:
        st.plotly_chart(
            scatter_fig,
            use_container_width=True,
            config={"displayModeBar": True, "displaylogo": False},
        )


def _wc_player_metrics(player_data: dict[str, pd.DataFrame], player: dict) -> dict | None:
    """Build scatter/table metrics for one WC player under heuristic v4."""
    df = player_data.get(player["code"], pd.DataFrame())
    if df.empty:
        return None
    stats = compute_player_stats(df, variant="v4")
    carries = int(stats["carries_total"])
    passes = int(stats["passes_completed"])
    total_actions = passes + carries
    sum_delta = stats["sum_dxt_passes"] + stats["sum_dxt_carries"]
    return {
        "player_name": player["name"],
        "position": player.get("position", "—"),
        "position_group": position_group(player.get("position")),
        "sum_delta_xt": round(sum_delta, 3),
        "sum_delta_xt_passes": round(stats["sum_dxt_passes"], 3),
        "sum_delta_xt_carries": round(stats["sum_dxt_carries"], 3),
        "sum_xt_end_passes": round(stats["sum_xt_end_passes"], 3),
        "passes": passes,
        "carries": carries,
        "total_actions": total_actions,
        "dxt_per_pass": round(stats["dxt_per_pass"], 4),
        "dxt_per_carry": _safe_ratio(stats["sum_dxt_carries"], carries, decimals=4),
        "xt_per_pass": round(stats["xt_per_pass"], 4),
        "dxt_per_action": _safe_ratio(sum_delta, total_actions, decimals=4),
        "pass_share_of_actions": round(passes / total_actions, 4) if total_actions else 0.0,
        "carry_share_of_actions": round(carries / total_actions, 4) if total_actions else 0.0,
        "pass_share_of_dxt": round(stats["sum_dxt_passes"] / sum_delta, 4) if sum_delta else 0.0,
        "carry_share_of_dxt": round(stats["sum_dxt_carries"] / sum_delta, 4) if sum_delta else 0.0,
        "positive_action_rate": round(stats["pos_pct"] / 100.0, 4),
        "_legacy_metrics": False,
    }


def _build_wc_players(player_data: dict[str, pd.DataFrame]) -> list[dict]:
    players: list[dict] = []
    for player in PLAYERS:
        metrics = _wc_player_metrics(player_data, player)
        if metrics is not None:
            players.append(metrics)
    players.sort(key=lambda p: p.get("sum_delta_xt", 0.0), reverse=True)
    return players


def render_world_cup_tab(player_data: dict[str, pd.DataFrame]) -> None:
    """World Cup squad — scatter charts and rankings under heuristic v4."""
    st.markdown("### Copa do Mundo · Heurístico v4")
    st.caption(
        "Comparação da seleção brasileira nos jogos exportados (CSVs Wyscout), "
        "com métricas agregadas pelo modelo **heurístico v4** (v3.1 + bônus Top5 no último terço)."
    )

    players = _laliga_prepare_players(_build_wc_players(player_data))
    if not players:
        st.warning("Nenhum jogador com dados disponíveis.")
        return

    st.caption(f"{len(players)} jogadores · todos os jogos agregados")

    with st.expander("Guia de métricas · passes vs conduções", expanded=False):
        st.markdown(
            """
**Totais**
| Métrica | Significado |
|---|---|
| **Σ ΔxT** | Soma do ganho bruto de xT (destino − origem) em passes **e** conduções |
| **Σ ΔxT passe** | Parte do Σ ΔxT gerada só por passes |
| **Σ ΔxT condução** | Parte do Σ ΔxT gerada só por conduções |
| **Σ xT passe** | Soma do xT no **destino** dos passes (ameaça entregue, não o ganho) |

**Eficiência**
| Métrica | Significado |
|---|---|
| **ΔxT/passe** | Σ ΔxT passe ÷ passes — progressão média por passe |
| **ΔxT/condução** | Σ ΔxT condução ÷ conduções — progressão média conduzindo |
| **xT/passe** | xT médio no destino do passe |
            """
        )

    scatter_title = "Seleção Brasileira · Copa do Mundo"
    st.markdown("#### Gráficos de dispersão")
    selected_groups = st.multiselect(
        "Filtrar grupos de posição nos gráficos",
        options=list(POSITION_GROUPS_ORDER),
        default=list(POSITION_GROUPS_ORDER),
        help="Selecione quais grupos aparecem nos gráficos de dispersão.",
        key="wc_scatter_groups",
    )
    chart_kwargs = dict(
        players=players,
        title=scatter_title,
        show_season_ref=False,
        selected_groups=selected_groups,
    )

    st.markdown("##### Σ ΔxT × Passes")
    st.caption("Volume de passes vs impacto total na Copa.")
    _render_laliga_plotly_chart(_laliga_scatter_chart, **chart_kwargs)

    st.markdown("##### ΔxT/passe × xT/passe")
    st.caption("Eficiência de progressão vs ameaça no destino do passe.")
    _render_laliga_plotly_chart(_laliga_rate_scatter_chart, **chart_kwargs)

    st.markdown("##### Σ xT passe × Σ ΔxT")
    st.caption("Ameaça entregue no destino dos passes vs progressão total.")
    _render_laliga_plotly_chart(_laliga_threat_progression_scatter_chart, **chart_kwargs)

    st.markdown("##### Passes × Conduções")
    st.caption("Perfil de estilo: mais passe vs mais condução.")
    _render_laliga_plotly_chart(_laliga_style_scatter_chart, **chart_kwargs)

    st.markdown("#### Classificação geral")
    overall_df = _laliga_stats_to_dataframe(players, show_season_ref=False)
    st.dataframe(overall_df, use_container_width=True, hide_index=True)

    st.markdown("#### Classificação por grupo de posição")
    by_pos_df = _laliga_rankings_by_position(players, show_season_ref=False)
    st.dataframe(by_pos_df, use_container_width=True, hide_index=True)

    with st.expander("Tabela completa de métricas"):
        detail_df = _laliga_detailed_metrics_dataframe(players, show_season_ref=False)
        st.dataframe(detail_df, use_container_width=True, hide_index=True)

    with st.expander("Top por xT/passe"):
        xt_df = overall_df.sort_values("xT/passe", ascending=False).reset_index(drop=True)
        xt_df["#"] = range(1, len(xt_df) + 1)
        st.dataframe(xt_df, use_container_width=True, hide_index=True)



def _item_sep(idx: int, total: int) -> str:
    return "" if idx == total - 1 else f"margin-bottom:8px;padding-bottom:8px;border-bottom:1px solid {CARD_INNER_BORDER};"


def _accent_rgb(border_color: str) -> tuple[int, int, int]:
    h = border_color.lstrip("#")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _stats_card_shell_html(title: str, border_color: str, body_html: str) -> str:
    r, g, b = _accent_rgb(border_color)
    grad = (
        f"linear-gradient(150deg, rgba({r},{g},{b},0.18) 0%, "
        f"rgba(24,24,38,0.55) 55%, rgba(16,16,26,0.82) 100%)"
    )
    html = (
        f'<div style="background:{grad};border:1px solid rgba({r},{g},{b},0.55);'
        f'border-radius:10px;padding:12px 14px 10px 14px;margin-bottom:8px;">'
    )
    html += (
        f'<div style="border-bottom:2px solid rgb({r},{g},{b});padding-bottom:6px;margin-bottom:8px;">'
        f'<span style="font-size:{CARD_TITLE_TEXT};color:#eef1f7;font-weight:700;letter-spacing:0.04em;">'
        f"{title.upper()}</span></div>"
    )
    html += body_html
    html += "</div>"
    return html


def _simple_body_scoreboard(items: list[tuple[str, str]]) -> str:
    body = ""
    for idx, (label, disp_val) in enumerate(items):
        body += f'<div style="{_item_sep(idx, len(items))}">'
        body += (
            '<div style="display:flex;justify-content:space-between;align-items:baseline;gap:12px;">'
            f'<span style="font-size:{CARD_LABEL_TEXT};color:#c7cdda;font-weight:600;">{label}</span>'
            f'<span style="font-size:{CARD_VALUE_TEXT};color:#ffffff;font-weight:700;line-height:1;">{disp_val}</span>'
            "</div>"
        )
        body += "</div>"
    return body


def stats_section_card(title: str, border_color: str, items: list[tuple[str, str]]) -> None:
    inner = _simple_body_scoreboard(items)
    st.markdown(_stats_card_shell_html(title, border_color, inner), unsafe_allow_html=True)


def render_general_stats_card(stats: dict, tone: str) -> None:
    stats_section_card(
        "Geral",
        tone,
        [
            ("Passes", _fmt_count(stats["passes_total"])),
            ("Passes completados", _fmt_count(stats["passes_completed"])),
            ("% acerto passes", _fmt_pct(stats["passes_accuracy_pct"])),
            ("Key passes", _fmt_count(stats["key_passes"])),
            ("Crosses", _fmt_count(stats["crosses"])),
            ("Bolas longas", _fmt_count(stats["long_balls"])),
            ("Conduções", _fmt_count(stats["carries_total"])),
            ("Dribles", _fmt_count(stats["dribbles"])),
            ("Finalizações", _fmt_count(stats["shots"])),
            ("xG", _fmt_decimal(stats["xg"], decimals=2)),
            ("Assistências", _fmt_count(stats["assists"])),
            ("xA", _fmt_decimal(stats["xa"], decimals=2)),
            ("Desarmes", _fmt_count(stats["tackles"])),
            ("Interceptações", _fmt_count(stats["interceptions"])),
            ("Cortes", _fmt_count(stats["clearances"])),
            ("Recuperações", _fmt_count(stats["ball_recoveries"])),
            ("Bloqueios", _fmt_count(stats["blocks"])),
            ("Ações defensivas", _fmt_count(stats["defensive_total"])),
        ],
    )


def render_impact_card(stats: dict, tone: str) -> None:
    impact = stats["impact_pass"]
    high_impact = stats["high_impact_pass"]
    carry_impact = stats["impact_carry"]
    carry_high = stats["high_impact_carry"]
    prog = stats["progressive_wyscout"]
    stats_section_card(
        "Impact (xT v4)",
        tone,
        [
            ("Pass Impact (xT v4)", _fmt_decimal(stats["sum_dxt_passes"])),
            ("ΔxT passes campo ofensivo", _fmt_decimal(stats["sum_dxt_passes_offensive"])),
            ("Σ xT final passes", _fmt_decimal(stats["sum_xt_end_passes"])),
            ("Σ xT final bolas longas", _fmt_decimal(stats["sum_xt_end_long_balls"])),
            ("Carry Impact (xT v4)", _fmt_decimal(stats["sum_dxt_carries"])),
            (
                "Total Impact (xT v4)",
                _fmt_decimal(stats["sum_dxt_passes"] + stats["sum_dxt_carries"]),
            ),
            ("Σ xT terço final", _fmt_decimal(stats["sum_xt_end_final_third"])),
            ("Impact Passes", _fmt_count(impact["successful"])),
            ("High Impact Passes", _fmt_count(high_impact["successful"])),
            ("Impact Carries", _fmt_count(carry_impact["successful"])),
            ("High Impact Carries", _fmt_count(carry_high["successful"])),
            ("Passes prog. Wyscout", _fmt_count(prog["successful"])),
            ("% Positive Impact", _fmt_pct(stats["pos_pct"])),
        ],
    )


def render_xt_efficiency_card(stats: dict, tone: str) -> None:
    stats_section_card(
        "Eficiência xT (v4)",
        tone,
        [
            ("Σ xT / passe completado", _fmt_decimal(stats["xt_per_pass"], decimals=3)),
            ("Σ ΔxT / passe completado", _fmt_decimal(stats["dxt_per_pass"], decimals=3)),
            (
                "Σ xT terço final / passe compl. terço final",
                _fmt_decimal(stats["xt_per_pass_final_third"], decimals=3),
            ),
            ("Σ xT top 10 passes", _fmt_decimal(stats["sum_xt_end_top10_passes"])),
            ("Σ xT key passes", _fmt_decimal(stats["sum_xt_end_key_passes"])),
            ("Σ xT / impact passe", _fmt_decimal(stats["xt_per_impact_pass"], decimals=3)),
        ],
    )


# ── PITCH DRAWING ────────────────────────────────────────────
def _base_pitch(bg="#1a1a2e"):
    pitch = Pitch(pitch_type="statsbomb", pitch_color=bg, line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor(bg)
    fig.set_dpi(FIG_DPI)
    return fig, ax, pitch


def _map_scale() -> float:
    return FIG_W / MAP_REF_WIDTH


def _add_map_legend(ax, handles: list) -> None:
    scale = _map_scale()
    leg = ax.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        frameon=True,
        facecolor="#1a1a2e",
        edgecolor="#444466",
        fontsize=6.2 * scale,
        labelspacing=0.35 * scale,
        borderpad=0.45 * scale,
        handlelength=1.9 * scale,
    )
    for text in leg.get_texts():
        text.set_color("white")
    leg.get_frame().set_alpha(0.90)


def _attack_arrow(fig, has_cbar: bool = False):
    scale = _map_scale()
    ox = -0.04 if has_cbar else 0.0
    fig.patches.append(
        FancyArrowPatch(
            (0.44 + ox, 0.045),
            (0.56 + ox, 0.045),
            transform=fig.transFigure,
            arrowstyle="-|>",
            mutation_scale=10 * scale,
            linewidth=1.4 * scale,
            color="#aaaaaa",
        )
    )
    fig.text(
        0.50 + ox,
        0.012,
        "Attacking Direction",
        ha="center",
        va="bottom",
        transform=fig.transFigure,
        fontsize=7.0 * scale,
        color="#aaaaaa",
    )


def _save_fig(fig):
    fig.canvas.draw()
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=FIG_DPI, facecolor=fig.get_facecolor(), bbox_inches="tight")
    buf.seek(0)
    return Image.open(buf)


def _delicate_arrows(
    pitch, ax, x1, y1, x2, y2, color, scale: float, *, alpha: float | None = None, width_mult: float = 1.0
) -> None:
    pitch.arrows(
        x1, y1, x2, y2,
        color=color,
        width=ARROW_WIDTH * scale * width_mult,
        headwidth=ARROW_HEADWIDTH * scale * width_mult,
        headlength=ARROW_HEADLENGTH * scale * width_mult,
        ax=ax,
        zorder=3,
        alpha=alpha if alpha is not None else ARROW_ALPHA,
    )


def filter_impact_plays(df: pd.DataFrame) -> pd.DataFrame:
    """Keep successful impact passes and carries only."""
    if df.empty:
        return df
    mask = df["has_end"] & (
        ((df["category"] == "passes") & df["impact_pass"])
        | ((df["category"] == "ball-carries") & df["impact_carry"])
    )
    return df[mask].copy()


def draw_impact_plays_map(df: pd.DataFrame, player_name: str, match_label: str):
    """Successful impact passes (goal approach + xT v4) and impact carries."""
    actions = filter_impact_plays(df)

    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    if actions.empty:
        ax.text(
            60, 40, "Sem impact plays no recorte",
            ha="center", va="center", color="white", fontsize=9,
        )
    else:
        for _, row in actions.iterrows():
            is_pass = row["category"] == "passes"
            is_high = bool(
                row.get("high_impact_pass", False) if is_pass else row.get("high_impact_carry", False)
            )
            if is_high:
                color, alpha = COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH
            else:
                color, alpha = COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH

            _delicate_arrows(
                pitch, ax,
                row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                color, scale, alpha=alpha,
            )
            if is_pass:
                pitch.scatter(
                    row["x_start"], row["y_start"],
                    s=PASS_START_MARKER_SIZE + 2, marker="o", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
                )
            else:
                pitch.scatter(
                    row["x_end"], row["y_end"],
                    s=CARRY_START_MARKER_SIZE + 4, marker="s", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
                )

    legend_handles = [
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Impact", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_PROGRESSIVE, markersize=4, linestyle="None", label="Passe certo"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=COLOR_PROGRESSIVE, markersize=4, linestyle="None", label="Condução"),
    ]
    _add_map_legend(ax, legend_handles)
    ax.set_title(
        f"{player_name}\nImpact Plays · xT v4 · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_pass_map(df: pd.DataFrame, player_name: str, match_label: str, *, impact_only: bool = False):
    passes = df[df["category"] == "passes"].copy()
    if impact_only:
        passes = passes[passes["impact_pass"].astype(bool)]
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    for _, row in passes.iterrows():
        if not row["has_end"]:
            continue
        is_lost = not row["is_success"]
        is_high_impact = bool(row.get("high_impact_pass", False))
        is_prog = bool(row.get("progressive", False))
        if is_lost:
            color, alpha = COLOR_FAIL, ARROW_ALPHA_EMPH
        elif is_high_impact:
            color, alpha = COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH
        elif is_prog:
            color, alpha = COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH
        else:
            color, alpha = COLOR_SUCCESS, ARROW_ALPHA

        _delicate_arrows(
            pitch, ax,
            row["x_start"], row["y_start"], row["x_end"], row["y_end"],
            color, scale, alpha=alpha,
        )
        pitch.scatter(
            row["x_start"], row["y_start"],
            s=PASS_START_MARKER_SIZE, marker="o", color=color,
            edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_SUCCESS, lw=1.4 * scale, label="Completado", alpha=0.65),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Progressivo", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
        Line2D([0], [0], color=COLOR_FAIL, lw=1.4 * scale, label="Incompleto", alpha=0.80),
    ]
    _add_map_legend(ax, legend_handles)
    title_suffix = " · Impact" if impact_only else ""
    ax.set_title(
        f"{player_name}\nPasses{title_suffix} · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_carry_map(df: pd.DataFrame, player_name: str, match_label: str, *, impact_only: bool = False):
    carries = df[df["category"] == "ball-carries"].copy()
    if impact_only:
        carries = carries[carries["impact_carry"].astype(bool)]
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    for _, row in carries.iterrows():
        if not row["has_end"]:
            continue
        is_high_impact = bool(row.get("high_impact_carry", False))
        is_impact = bool(row.get("impact_carry", False))
        if is_high_impact:
            color, alpha = COLOR_HIGHLY_PROGRESSIVE, ARROW_ALPHA_EMPH
        elif is_impact:
            color, alpha = COLOR_PROGRESSIVE, ARROW_ALPHA_EMPH
        else:
            color, alpha = COLOR_CARRY, COLOR_CARRY_BASE_ALPHA

        _delicate_arrows(
            pitch, ax,
            row["x_start"], row["y_start"], row["x_end"], row["y_end"],
            color, scale, alpha=alpha,
        )
        pitch.scatter(
            row["x_start"], row["y_start"],
            s=CARRY_START_MARKER_SIZE, marker="o", color=color,
            edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=alpha,
        )

    legend_handles = [
        Line2D([0], [0], color=COLOR_CARRY, lw=1.4 * scale, label="Condução", alpha=0.60),
        Line2D([0], [0], color=COLOR_PROGRESSIVE, lw=1.4 * scale, label="Impact", alpha=0.80),
        Line2D([0], [0], color=COLOR_HIGHLY_PROGRESSIVE, lw=1.4 * scale, label="High Impact", alpha=0.85),
    ]
    _add_map_legend(ax, legend_handles)
    title_suffix = " · Impact" if impact_only else ""
    ax.set_title(
        f"{player_name}\nConduções{title_suffix} · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_pass_destination_heatmap(
    df: pd.DataFrame, player_name: str, match_label: str, *, impact_only: bool = False,
):
    """6×4 heatmap of pass end locations on the pitch."""
    passes = df[(df["category"] == "passes") & df["has_end"]].copy()
    if impact_only:
        passes = passes[passes["impact_pass"].astype(bool)]
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    x_bins = np.linspace(0.0, FIELD_X, PASS_DEST_HEATMAP_COLS + 1)
    y_bins = np.linspace(0.0, FIELD_Y, PASS_DEST_HEATMAP_ROWS + 1)
    grid = np.zeros((PASS_DEST_HEATMAP_ROWS, PASS_DEST_HEATMAP_COLS), dtype=float)

    if not passes.empty:
        x_idx = np.clip(
            np.digitize(passes["x_end"].to_numpy(), x_bins, right=True) - 1,
            0,
            PASS_DEST_HEATMAP_COLS - 1,
        )
        y_idx = np.clip(
            np.digitize(passes["y_end"].to_numpy(), y_bins, right=True) - 1,
            0,
            PASS_DEST_HEATMAP_ROWS - 1,
        )
        for ix, iy in zip(x_idx, y_idx):
            grid[iy, ix] += 1.0

    vmax = max(float(grid.max()), 1.0)
    norm = Normalize(vmin=0.0, vmax=vmax)
    threshold = vmax * 0.45

    for iy in range(PASS_DEST_HEATMAP_ROWS):
        for ix in range(PASS_DEST_HEATMAP_COLS):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=CMAP_PASS_DEST(norm(value)),
                    edgecolor=(1, 1, 1, 0.22),
                    linewidth=0.5,
                    alpha=0.94,
                    zorder=2,
                )
            )
            if value > 0:
                label = f"{int(value)}"
                ax.text(
                    (x0 + x1) / 2, (y0 + y1) / 2, label,
                    ha="center", va="center",
                    color="#000000" if value <= threshold else "#ffffff",
                    fontsize=7.2 * scale, fontweight="600", zorder=4,
                )

    pitch.draw(ax=ax)
    sm = plt.cm.ScalarMappable(cmap=CMAP_PASS_DEST, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#ffffff")
    cbar.set_label("Passes", color="#c7cdda", fontsize=7 * scale)
    title_suffix = " · Impact" if impact_only else ""
    ax.set_title(
        f"{player_name}\nDestino dos passes · 6×4{title_suffix} · {match_label}",
        color="white", fontsize=9.2 * scale, pad=5,
    )
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


def draw_top_deltaxt_map(
    df: pd.DataFrame,
    player_name: str,
    match_label: str,
    *,
    delta_col: str = "delta_xt",
    model_label: str = "v3",
):
    """Top delta-xT actions with distinct colormaps for passes vs carries."""
    top = top_deltaxt_actions(df, TOP_DELTAXT_N, delta_col=delta_col)
    fig, ax, pitch = _base_pitch()
    scale = _map_scale()

    if top.empty:
        ax.text(
            60, 40, "Sem ações com ΔxT positivo",
            ha="center", va="center", color="white", fontsize=9,
        )
    else:
        passes = top[top["category"] == "passes"]
        carries = top[top["category"] == "ball-carries"]
        pass_vmax = max(float(passes[delta_col].max()), 0.01) if not passes.empty else 0.01
        carry_vmax = max(float(carries[delta_col].max()), 0.01) if not carries.empty else 0.01

        if not passes.empty:
            norm_pass = Normalize(vmin=0, vmax=pass_vmax)
            for _, row in passes.iterrows():
                color = CMAP_PASS(norm_pass(row[delta_col]))
                _delicate_arrows(
                    pitch, ax,
                    row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                    color, scale, alpha=ARROW_ALPHA_EMPH,
                )
                pitch.scatter(
                    row["x_start"], row["y_start"],
                    s=PASS_START_MARKER_SIZE + 4, marker="o", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=0.88,
                )

        if not carries.empty:
            norm_carry = Normalize(vmin=0, vmax=carry_vmax)
            for _, row in carries.iterrows():
                color = CMAP_CARRY(norm_carry(row[delta_col]))
                _delicate_arrows(
                    pitch, ax,
                    row["x_start"], row["y_start"], row["x_end"], row["y_end"],
                    color, scale, alpha=ARROW_ALPHA_EMPH, width_mult=1.05,
                )
                pitch.scatter(
                    row["x_end"], row["y_end"],
                    s=CARRY_START_MARKER_SIZE + 6, marker="s", color=color,
                    edgecolors="white", linewidths=0.3, ax=ax, zorder=6, alpha=0.88,
                )

        legend_handles = [
            Line2D([0], [0], color=CMAP_PASS(0.85), lw=1.4 * scale, label="Passe (ΔxT)"),
            Line2D([0], [0], color=CMAP_CARRY(0.85), lw=1.4 * scale, label="Condução (ΔxT)"),
        ]
        _add_map_legend(ax, legend_handles)

    ax.set_title(
        f"{player_name}\nTop {TOP_DELTAXT_N} ΔxT · {model_label} · {match_label}",
        color="white", fontsize=8.8 * scale, pad=5,
    )
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_xt_threat_surface(grid: np.ndarray, title: str, vmax: float):
    """Heatmap of the heuristic xT threat surface on the pitch."""
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(FIG_W, FIG_H))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    scale = _map_scale()

    ny, nx = grid.shape
    x_edges = np.linspace(0, FIELD_X, nx + 1)
    y_edges = np.linspace(0, FIELD_Y, ny + 1)
    ax.pcolormesh(
        x_edges, y_edges, grid,
        cmap="magma", vmin=0, vmax=vmax, shading="auto", alpha=0.88, zorder=1,
    )
    pitch.draw(ax=ax)

    ax.set_title(title, color="white", fontsize=8.8 * scale, pad=5)
    _attack_arrow(fig)
    return _save_fig(fig), fig


def draw_xt_grid_map(
    grid: np.ndarray,
    title: str,
    *,
    as_percent: bool = True,
    color_percentile: tuple[float, float] | None = (5, 95),
    value_fmt: str = ".2f",
    n_x: int | None = None,
    n_y: int | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
):
    """Pitch grid with xT value labeled in each cell (Hudson-style)."""
    grid_rows, grid_cols = grid.shape
    cols = n_x if n_x is not None else grid_cols
    rows = n_y if n_y is not None else grid_rows
    if cols != grid_cols or rows != grid_rows:
        cols, rows = grid_cols, grid_rows
    pitch = Pitch(pitch_type="statsbomb", pitch_color="#1a1a2e", line_color="#ffffff", line_alpha=0.95)
    fig, ax = pitch.draw(figsize=(7.8, 5.2))
    fig.set_facecolor("#1a1a2e")
    fig.set_dpi(FIG_DPI)
    scale = 7.8 / MAP_REF_WIDTH

    x_bins = np.linspace(0, FIELD_X, cols + 1)
    y_bins = np.linspace(0, FIELD_Y, rows + 1)
    if vmin is not None and vmax is not None:
        vmin_f, vmax_f = float(vmin), float(vmax)
    elif color_percentile is not None:
        vmin_f = float(np.percentile(grid, color_percentile[0]))
        vmax_f = float(np.percentile(grid, color_percentile[1]))
    else:
        vmin_f = 0.0
        vmax_f = max(float(grid.max()), 1e-6)
    if vmax_f <= vmin_f:
        vmax_f = vmin_f + 1e-6

    norm = Normalize(vmin=vmin_f, vmax=vmax_f)
    threshold = vmin_f + (vmax_f - vmin_f) * 0.45

    for iy in range(rows):
        for ix in range(cols):
            value = float(grid[iy, ix])
            x0, x1 = x_bins[ix], x_bins[ix + 1]
            y0, y1 = y_bins[iy], y_bins[iy + 1]
            ax.add_patch(
                Rectangle(
                    (x0, y0), x1 - x0, y1 - y0,
                    facecolor=XT_GRID_CMAP(norm(value)),
                    edgecolor=(1, 1, 1, 0.15),
                    linewidth=0.4,
                    alpha=0.95,
                    zorder=2,
                )
            )
            label = f"{value * 100:.1f}%" if as_percent else f"{value:{value_fmt}}"
            ax.text(
                (x0 + x1) / 2, (y0 + y1) / 2, label,
                ha="center", va="center",
                color="#000000" if value <= threshold else "#ffffff",
                fontsize=5.2 * scale, fontweight="600", zorder=4,
            )

    pitch.draw(ax=ax)
    ax.set_title(title, color="#eef1f7", fontsize=10 * scale, pad=8)
    sm = plt.cm.ScalarMappable(cmap=XT_GRID_CMAP, norm=norm)
    cbar = fig.colorbar(sm, ax=ax, fraction=0.022, pad=0.02, shrink=0.55)
    cbar.ax.yaxis.set_tick_params(color="#ffffff", labelsize=6)
    plt.setp(cbar.ax.axes.get_yticklabels(), color="#ffffff")
    _attack_arrow(fig, has_cbar=True)
    return _save_fig(fig), fig


def zone_xt_means(grid: np.ndarray, n_x: int = XT_ZONE_COLS, n_y: int = XT_ZONE_ROWS) -> np.ndarray:
    """Mean xT per pitch zone from the fine threat grid."""
    ny, nx = grid.shape
    zones = np.zeros((n_y, n_x), dtype=float)
    for iy in range(n_y):
        y_start = int(iy * ny / n_y)
        y_end = int((iy + 1) * ny / n_y)
        for ix in range(n_x):
            x_start = int(ix * nx / n_x)
            x_end = int((ix + 1) * nx / n_x)
            zones[iy, ix] = float(grid[y_start:y_end, x_start:x_end].mean())
    return zones


@st.cache_data(show_spinner=False)
def load_all_players(_cache_version: int = DATA_CACHE_VERSION) -> dict[str, pd.DataFrame]:
    return {
        player["code"]: load_player_all_matches(player)
        for player in PLAYERS
    }


def _show_map(
    draw_fn,
    df: pd.DataFrame,
    player_name: str,
    match_label: str,
    empty_msg: str,
    **draw_kwargs,
) -> None:
    if df.empty:
        st.info(empty_msg)
        return
    img, fig = draw_fn(df, player_name, match_label, **draw_kwargs)
    plt.close(fig)
    st.image(img, use_container_width=True)


def _player_selector(key: str) -> dict:
    """Sidebar-style player picker shared by Análise and Stats tabs."""
    options = {p["name"]: p for p in PLAYERS}
    name = st.selectbox("Jogador", list(options.keys()), key=key)
    return options[name]


def render_player_stats_cards(stats: dict) -> None:
    """Render the three stat cards with distinct accent colors."""
    stat_cols = st.columns(3)
    with stat_cols[0]:
        render_general_stats_card(stats, STAT_CARD_GENERAL_COLOR)
    with stat_cols[1]:
        render_impact_card(stats, STAT_CARD_IMPACT_COLOR)
    with stat_cols[2]:
        render_xt_efficiency_card(stats, STAT_CARD_XT_COLOR)


def render_analysis_tab(
    player_data: dict[str, pd.DataFrame],
    *,
    impact_plays_only: bool = False,
) -> None:
    player = _player_selector("analysis_player")
    match_label = ALL_GAMES_LABEL
    df = player_data[player["code"]]

    st.markdown(
        f'<div class="player-header">{player["name"]}</div>'
        f'<div class="player-sub">{player["position"]} · {match_label}</div>',
        unsafe_allow_html=True,
    )

    if df.empty:
        st.warning(f"Sem dados para {player['name']}.")
        return

    map_kwargs = {"impact_only": impact_plays_only}
    label_suffix = " · Impact" if impact_plays_only else ""
    map_cols = st.columns(3)
    with map_cols[0]:
        st.markdown(f'<div class="map-label">Passes{label_suffix}</div>', unsafe_allow_html=True)
        _show_map(
            draw_pass_map, df, player["name"], match_label,
            "Sem passes no recorte.", **map_kwargs,
        )
    with map_cols[1]:
        st.markdown(f'<div class="map-label">Conduções{label_suffix}</div>', unsafe_allow_html=True)
        _show_map(
            draw_carry_map, df, player["name"], match_label,
            "Sem conduções no recorte.", **map_kwargs,
        )
    with map_cols[2]:
        st.markdown(f'<div class="map-label">Destino dos passes{label_suffix}</div>', unsafe_allow_html=True)
        _show_map(
            draw_pass_destination_heatmap, df, player["name"], match_label,
            "Sem passes com destino no recorte.", **map_kwargs,
        )

    st.markdown("---")
    st.markdown("#### Estatísticas")
    st.caption(
        f"**{ALL_GAMES_LABEL.capitalize()}** · xT heurístico **v4** · "
        "Finalizações, xG, assistências e xA não constam nos CSVs Wyscout."
    )
    render_player_stats_cards(compute_player_stats(df))


def render_xt_model_comparison(
    player_data: dict[str, pd.DataFrame],
) -> None:
    """Compare xT v3 original vs v3.1 (transições suaves)."""
    match_label = ALL_GAMES_LABEL

    compare_models = [
        {
            "key": "v3",
            "label": "Heurístico v3",
            "grid_fn": compute_heuristic_v3_xt_grid,
            "fine_fn": compute_heuristic_v3_fine_grid,
            "vmax": XT_V3_SURFACE_MAX,
            "delta_col": "delta_xt",
            "xt_end_col": "xt_end",
            "desc": "Original — zonas com blend 22 m e monotonicidade por linha.",
        },
        {
            "key": "v3.1",
            "label": "Heurístico v3.1",
            "grid_fn": compute_heuristic_v31_xt_grid,
            "fine_fn": compute_heuristic_v31_fine_grid,
            "vmax": XT_V3_SURFACE_MAX,
            "delta_col": "delta_xt_v31",
            "xt_end_col": "xt_end_v31",
            "desc": (
                "Blend amplo (48 m) + gaussiana só em X (σx=3.5) + rampa 5.0/7.8 pp por coluna. "
                "Penalização lateral (6%) apenas no campo ofensivo (x≥60)."
            ),
        },
    ]

    st.markdown("### Mapa xT por quadrante")
    st.caption(
        "Comparação do **v3 original** com o **v3.1** — transições mais suaves entre colunas "
        "e menor salto defensivo→ofensivo, preservando valorização central no ataque."
    )

    grids = {m["key"]: m["grid_fn"]() for m in compare_models}
    jump_rows = [
        {
            "Modelo": m["key"],
            "Máx salto col. adj. (%)": round(_max_adjacent_col_jump_pct(grids[m["key"]]), 1),
            "Salto col. 8→9 (%)": round(
                abs(grids[m["key"]][:, 8].mean() - grids[m["key"]][:, 7].mean()) * 100.0, 1
            ),
        }
        for m in compare_models
    ]
    st.dataframe(pd.DataFrame(jump_rows), use_container_width=True, hide_index=True)

    grid_cols = st.columns(2)
    for col, model in zip(grid_cols, compare_models):
        grid = grids[model["key"]]
        with col:
            st.markdown(f'<div class="map-label">{model["label"]}</div>', unsafe_allow_html=True)
            img, fig = draw_xt_grid_map(grid, model["label"], as_percent=True)
            plt.close(fig)
            st.image(img, use_container_width=True)
            st.caption(
                f"16×12 · Máx: {grid.max():.3f} · Média: {grid.mean():.3f} · "
                f"{model['desc']}"
            )

    with st.expander("Superfície contínua xT"):
        surf_cols = st.columns(2)
        for col, model in zip(surf_cols, compare_models):
            with col:
                fine = model["fine_fn"]()
                img, fig = draw_xt_threat_surface(
                    fine, f"Superfície {model['key']}", model["vmax"]
                )
                plt.close(fig)
                st.image(img, use_container_width=True)

    st.markdown("---")
    st.markdown("### Top 10 ΔxT — v3 / v3.1")

    summary_rows = []
    for player in PLAYERS:
        df = player_data[player["code"]]
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)

        if df.empty:
            st.warning(f"Sem dados para {player['name']}.")
            continue

        xt_actions = df[df["category"].isin(["passes", "ball-carries"]) & df["has_end"]]
        passes = df[df["category"] == "passes"]
        row = {"Jogador": player["name"]}
        for model in compare_models:
            key = model["key"]
            row[f"Σ ΔxT {key}"] = round(_safe_col_sum(xt_actions, model["delta_col"]), 3)
            row[f"Σ xT final {key}"] = round(_safe_col_sum(passes, model["xt_end_col"]), 3)
        summary_rows.append(row)

        cmp_cols = st.columns(2)
        for col, model in zip(cmp_cols, compare_models):
            with col:
                st.markdown(f'<div class="map-label">Top ΔxT · {model["key"]}</div>', unsafe_allow_html=True)
                _show_map(
                    lambda d, n, m, mc=model["delta_col"], ml=model["key"]: draw_top_deltaxt_map(
                        d, n, m, delta_col=mc, model_label=ml
                    ),
                    df, player["name"], match_label,
                    f"Sem ações com ΔxT positivo ({model['key']}).",
                )

    if summary_rows:
        st.markdown("---")
        st.markdown("### Resumo comparativo")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    _render_external_model_comparison(player_data)


def _xt_action_mask(df: pd.DataFrame) -> pd.Series:
    return df["category"].isin(["passes", "ball-carries"]) & df["has_end"]


def _model_correlation(df: pd.DataFrame, col_a: str, col_b: str) -> float | None:
    subset = df[[col_a, col_b]].dropna()
    if len(subset) < 3:
        return None
    return float(subset[col_a].corr(subset[col_b]))


def _render_external_model_comparison(
    player_data: dict[str, pd.DataFrame],
) -> None:
    """Compare heuristic v3.1, v3.2 and xT Markov."""
    match_label = ALL_GAMES_LABEL

    markov_ok = any(
        "delta_xt_markov" in df.columns and df["delta_xt_markov"].notna().any()
        for df in player_data.values()
    )

    compare_models = [
        {
            "key": "v3.1",
            "label": "Heurístico v3.1",
            "delta_col": "delta_xt_v31",
            "grid_fn": compute_heuristic_v31_xt_grid,
            "desc": "Transições suaves · magnitude alta · penalização lateral no ataque.",
        },
        {
            "key": "v3.2",
            "label": "Heurístico v3.2",
            "delta_col": "delta_xt_v32",
            "grid_fn": compute_heuristic_v32_xt_grid,
            "desc": (
                "Base uniforme entre quadrantes + bônus Markov reforçado por zona (WSL 2018/19)."
            ),
        },
    ]
    if markov_ok:
        compare_models.append(
            {
                "key": "Markov",
                "label": "xT Markov",
                "delta_col": "delta_xt_markov",
                "grid_fn": lambda: markov_grid_for_display("wsl"),
                "desc": "Grid 16×12 treinado em FA WSL 2018/19 (StatsBomb Open Data).",
            }
        )

    st.markdown("---")
    st.markdown("### v3.1 · v3.2 · xT Markov")
    st.caption(
        "O **v3.2** usa uma base mais uniforme entre quadrantes (perfil v3.1 comprimido) "
        "e soma um **bônus Markov reforçado** por zona (WSL 2018/19)."
    )

    if not markov_ok:
        st.info("Grid Markov indisponível — inclua `models/xt_markov_wsl_16x12.json`.")

    try:
        grid_cols = st.columns(len(compare_models))
        for col, model in zip(grid_cols, compare_models):
            grid = model["grid_fn"]()
            with col:
                st.markdown(f'<div class="map-label">{model["label"]}</div>', unsafe_allow_html=True)
                img, fig = draw_xt_grid_map(grid, model["key"], as_percent=True)
                plt.close(fig)
                st.image(img, use_container_width=True)
                st.caption(
                    f"16×12 · Máx: {grid.max():.3f} · Média: {grid.mean():.3f}"
                )
    except FileNotFoundError as exc:
        st.warning(str(exc))

    summary_rows: list[dict] = []
    corr_rows: list[dict] = []

    for player in PLAYERS:
        df = player_data[player["code"]]
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)

        if df.empty:
            st.warning(f"Sem dados para {player['name']}.")
            continue

        xt_actions = df[_xt_action_mask(df)]
        row: dict = {"Jogador": player["name"]}
        for model in compare_models:
            row[f"Σ {model['key']}"] = round(_safe_col_sum(xt_actions, model["delta_col"]), 3)
            rated = xt_actions[model["delta_col"]].dropna()
            row[f"Média {model['key']}"] = round(float(rated.mean()), 4) if not rated.empty else None
        summary_rows.append(row)

        corr_row = {"Jogador": player["name"]}
        corr_pairs = [
            ("delta_xt_v31", "delta_xt_v32", "v3.1 × v3.2"),
            ("delta_xt_v32", "delta_xt_markov", "v3.2 × Markov"),
            ("delta_xt_v31", "delta_xt_markov", "v3.1 × Markov"),
        ]
        for col_a, col_b, label in corr_pairs:
            if col_a not in xt_actions.columns or col_b not in xt_actions.columns:
                continue
            corr = _model_correlation(xt_actions, col_a, col_b)
            corr_row[label] = round(corr, 3) if corr is not None else None
        corr_rows.append(corr_row)

        cmp_cols = st.columns(len(compare_models))
        for col, model in zip(cmp_cols, compare_models):
            with col:
                st.markdown(
                    f'<div class="map-label">Top · {model["key"]}</div>',
                    unsafe_allow_html=True,
                )
                _show_map(
                    lambda d, n, m, mc=model["delta_col"], ml=model["key"]: draw_top_deltaxt_map(
                        d, n, m, delta_col=mc, model_label=ml
                    ),
                    df,
                    player["name"],
                    match_label,
                    f"Sem ações com valor positivo ({model['key']}).",
                )
                st.caption(model["desc"])

    if summary_rows:
        st.markdown("---")
        st.markdown("### Resumo v3.1 / v3.2 / Markov")
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    if corr_rows:
        st.markdown("### Correlação (passes + conduções)")
        st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)


def _heuristic_test_models() -> list[dict]:
    bonus_key = get_v33_bonus_markov_key()
    report = load_validation_report()
    return [
        {
            "key": "v3.2",
            "kind": "heuristic",
            "label": "Heurístico v3.2",
            "grid_fn": compute_heuristic_v32_xt_grid,
            "delta_col": "delta_xt_v32",
            "desc": "Base uniforme + bônus Markov WSL (baseline atual).",
        },
        {
            "key": "v3.3",
            "kind": "heuristic",
            "label": f"Heurístico v3.3 · Markov {bonus_key}",
            "grid_fn": lambda: compute_heuristic_v33_xt_grid(_bonus_key=bonus_key),
            "delta_col": "delta_xt_v33",
            "desc": (
                f"Bônus Markov validado em hold-out "
                f"({report.get('winner_reason', '—')})."
            ),
        },
        {
            "key": "v4",
            "kind": "heuristic",
            "label": "Heurístico v4 · Top5 (último terço)",
            "grid_fn": compute_heuristic_v4_xt_grid,
            "delta_col": "delta_xt_v4",
            "desc": "v3.1 + bônus Top5 quase nulo nos 2/3 defensivos · notável no último terço.",
        },
        {
            "key": "v41",
            "kind": "heuristic",
            "label": "Heurístico v4.1 · Top5 uniforme",
            "grid_fn": compute_heuristic_v41_xt_grid,
            "delta_col": "delta_xt_v41",
            "desc": "v3.1 + bônus Top5 leve e uniforme em todo o campo (referência para v4).",
        },
    ]


MARKOV_FIELD_ORDER = ("wsl", "womens", "top5", "bayesian")


def _markov_test_models() -> list[dict]:
    models = []
    for key in MARKOV_FIELD_ORDER:
        if key not in list_available_markov_models():
            continue
        spec = MARKOV_MODEL_SPECS[key]
        models.append(
            {
                "key": spec["label"],
                "kind": "markov",
                "model_key": key,
                "label": spec["label"],
                "grid_fn": lambda k=key: markov_grid_for_display(k),
                "delta_col": spec["delta_col"],
                "desc": spec["description"],
            }
        )
    return models


def _shared_markov_color_scale(grids: dict[str, np.ndarray]) -> tuple[float, float]:
    """Common vmin/vmax (p5–p95) across Markov grids for fair visual comparison."""
    if not grids:
        return 0.0, 1.0
    stacked = np.concatenate([g.ravel() for g in grids.values()])
    vmin = float(np.percentile(stacked, 5))
    vmax = float(np.percentile(stacked, 95))
    if vmax <= vmin:
        vmax = vmin + 1e-6
    return vmin, vmax


def render_markov_fields_comparison() -> None:
    """Side-by-side pitch grids for all four Markov xT models."""
    status_rows = markov_models_status()
    available = [row["key"] for row in status_rows if row["present"]]
    missing = [row for row in status_rows if not row["present"]]

    st.markdown("### Campos Markov — comparação 16×12")
    st.caption(
        "Quatro variantes treinadas em StatsBomb Open Data · escala de cor compartilhada "
        "(p5–p95) quando todos os arquivos estão em `models/`."
    )

    ref_table = []
    for row in status_rows:
        ref_table.append(
            {
                "Chave": row["key"],
                "Modelo": row["label"],
                "Base de dados": row["description"],
                "Coluna ΔxT": row["delta_col"],
                "Arquivo": row["filename"],
                "Status": "OK" if row["present"] else "Ausente",
                "Máx xT": round(row["max_xt"], 4) if row.get("max_xt") is not None else None,
                "Média xT": round(row["mean_xt"], 4) if row.get("mean_xt") is not None else None,
                "Jogos treino": row.get("n_games_train"),
            }
        )
    st.dataframe(pd.DataFrame(ref_table), use_container_width=True, hide_index=True)

    if missing:
        st.warning(
            f"**{len(missing)} de {len(status_rows)} grids ausentes** em `models/`. "
            "Só o WSL aparece se os outros JSONs não foram commitados ou o deploy não os incluiu. "
            "Para gerar todos: `pip install -r requirements-train.txt && "
            "python scripts/train_external_models.py`"
        )
        for row in missing:
            st.error(f"`{row['filename']}` não encontrado · esperado em `{row['path']}`")

    if not available:
        st.info("Nenhum grid Markov carregável.")
        return

    grids: dict[str, np.ndarray] = {}
    for key in available:
        grids[key] = markov_grid_for_display(key)

    vmin, vmax = _shared_markov_color_scale(grids)
    st.caption(f"Escala compartilhada: {vmin * 100:.2f}% – {vmax * 100:.2f}% (valores xT no grid)")

    for row_start in range(0, len(MARKOV_FIELD_ORDER), 2):
        row_keys = MARKOV_FIELD_ORDER[row_start : row_start + 2]
        cols = st.columns(2)
        for col, key in zip(cols, row_keys):
            spec = MARKOV_MODEL_SPECS[key]
            with col:
                st.markdown(f'<div class="map-label">{spec["label"]}</div>', unsafe_allow_html=True)
                if key not in grids:
                    st.warning(
                        f"Grid ausente — inclua `{spec['filename']}` em `models/` "
                        f"ou execute o script de treino."
                    )
                    continue
                grid = grids[key]
                img, fig = draw_xt_grid_map(
                    grid,
                    spec["label"],
                    as_percent=True,
                    color_percentile=None,
                    vmin=vmin,
                    vmax=vmax,
                )
                plt.close(fig)
                st.image(img, use_container_width=True)
                st.caption(f"`{key}` · {spec['description']}")

    with st.expander("Superfície contínua (interpolação no campo)"):
        st.caption("Upsample 96×64 · escala 0 – máx global · colormap magma.")
        if not grids:
            st.info("Nenhuma superfície disponível.")
        else:
            surf_vmax = max(float(g.max()) for g in grids.values())
            for row_start in range(0, len(MARKOV_FIELD_ORDER), 2):
                row_keys = MARKOV_FIELD_ORDER[row_start : row_start + 2]
                surf_cols = st.columns(2)
                for col, key in zip(surf_cols, row_keys):
                    spec = MARKOV_MODEL_SPECS[key]
                    with col:
                        if key not in grids:
                            st.warning(f"`{spec['filename']}` ausente.")
                            continue
                        fine = compute_markov_fine_grid(model_key=key)
                        img, fig = draw_xt_threat_surface(fine, spec["label"], surf_vmax)
                        plt.close(fig)
                        st.image(img, use_container_width=True)


def _all_xt_test_models() -> list[dict]:
    return _heuristic_test_models() + _markov_test_models()


def render_xt_validation_section() -> None:
    report = load_validation_report()
    st.markdown("### Validação hold-out (StatsBomb)")
    st.caption(
        "Métrica: **AUC** de ΔxT para prever chute da mesma equipe nas próximas "
        f"{report.get('lookahead_actions', 8)} ações · hold-out "
        f"{int(float(report.get('holdout_fraction', 0.3)) * 100)}%."
    )

    metrics = report.get("metrics", {})
    if not metrics:
        st.info(
            "Relatório de validação ausente. Execute "
            "`python scripts/train_external_models.py` para gerar "
            "`models/xt_validation_report.json`."
        )
        return

    rows = []
    for key, vals in metrics.items():
        spec = MARKOV_MODEL_SPECS.get(key, {})
        rows.append(
            {
                "Modelo": spec.get("label", key),
                "Chave": key,
                "N ações": vals.get("n_moves"),
                "AUC chute+8": round(float(vals["auc_shot_8"]), 4)
                if vals.get("auc_shot_8") == vals.get("auc_shot_8")
                else None,
                "Correlação": round(float(vals["corr_shot_8"]), 4)
                if vals.get("corr_shot_8") == vals.get("corr_shot_8")
                else None,
                "ΔxT médio (perigoso)": round(float(vals["mean_delta_dangerous"]), 4)
                if vals.get("mean_delta_dangerous") == vals.get("mean_delta_dangerous")
                else None,
                "ΔxT médio (seguro)": round(float(vals["mean_delta_safe"]), 4)
                if vals.get("mean_delta_safe") == vals.get("mean_delta_safe")
                else None,
            }
        )
    df_metrics = pd.DataFrame(rows)
    winner = report.get("winner", "—")
    st.dataframe(df_metrics, use_container_width=True, hide_index=True)
    st.success(
        f"**Vencedor validação:** `{winner}` · {report.get('winner_reason', '—')} · "
        f"v3.3 usa bônus `{report.get('v33_bonus_source', winner)}`."
    )


def render_xt_tests_tab(player_data: dict[str, pd.DataFrame]) -> None:
    """Laboratório de comparação entre heurísticos e variantes Markov."""
    match_label = ALL_GAMES_LABEL
    test_models = _all_xt_test_models()

    st.markdown("### Testes xT — laboratório de modelos")
    st.caption(
        "Compare o heurístico **v3.2** (baseline), **v3.3** (bônus Markov validado) "
        "e as variantes Markov treinadas com mais dados, orientação LTR e suavização bayesiana."
    )

    render_xt_validation_section()
    st.markdown("---")
    render_markov_fields_comparison()
    st.markdown("---")

    render_v4_player_benchmarks(player_data)
    st.markdown("---")
    st.markdown("### Grids heurísticos (v3.2 · v3.3 · v4 · v4.1)")
    heuristic_models = _heuristic_test_models()
    hcols = st.columns(len(heuristic_models))
    for col, model in zip(hcols, heuristic_models):
        with col:
            try:
                grid = model["grid_fn"]()
                img, fig = draw_xt_grid_map(grid, model["label"], as_percent=True)
                plt.close(fig)
                st.image(img, use_container_width=True)
                st.caption(f"{model['desc']} · máx {grid.max():.3f}")
            except FileNotFoundError as exc:
                st.warning(str(exc))

    st.markdown("---")
    st.markdown("### Σ ΔxT por jogador")
    summary_rows: list[dict] = []
    corr_base = "delta_xt_v32"

    for player in PLAYERS:
        df = player_data[player["code"]]
        if df.empty:
            continue
        xt_actions = df[_xt_action_mask(df)]
        row: dict = {"Jogador": player["name"]}
        for model in test_models:
            col = model["delta_col"]
            row[model["key"]] = round(_safe_col_sum(xt_actions, col), 3)
        summary_rows.append(row)

    if summary_rows:
        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    st.markdown("### Correlação com v3.2 (passes + conduções)")
    corr_rows: list[dict] = []
    for player in PLAYERS:
        df = player_data[player["code"]]
        if df.empty or corr_base not in df.columns:
            continue
        xt_actions = df[_xt_action_mask(df)]
        corr_row: dict = {"Jogador": player["name"]}
        for model in test_models:
            col = model["delta_col"]
            if col == corr_base or col not in xt_actions.columns:
                continue
            corr = _model_correlation(xt_actions, corr_base, col)
            corr_row[model["key"]] = round(corr, 3) if corr is not None else None
        corr_rows.append(corr_row)
    if corr_rows:
        st.dataframe(pd.DataFrame(corr_rows), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### Top 10 ΔxT por modelo")
    for player in PLAYERS:
        df = player_data[player["code"]]
        st.markdown(f'<div class="player-header">{player["name"]}</div>', unsafe_allow_html=True)
        if df.empty:
            st.warning(f"Sem dados para {player['name']}.")
            continue

        cmp_cols = st.columns(min(3, len(test_models)))
        for col, model in zip(cmp_cols, test_models):
            with col:
                st.markdown(
                    f'<div class="map-label">{model["label"]}</div>',
                    unsafe_allow_html=True,
                )
                _show_map(
                    lambda d, n, m, mc=model["delta_col"], ml=model["key"]: draw_top_deltaxt_map(
                        d, n, m, delta_col=mc, model_label=ml
                    ),
                    df,
                    player["name"],
                    match_label,
                    f"Sem ΔxT positivo ({model['key']}).",
                )


# ── MAIN ─────────────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center;margin-bottom:1rem;">
      <h1 style="margin:0;color:#eef1f7;">World Cup Stats</h1>
      <p style="color:#94a3b8;font-size:0.95rem;margin-top:0.35rem;">
        Bruno Guimarães · Casemiro · Lucas Paquetá · Pedri · Rodri — xT Heurístico v4
      </p>
    </div>
    """,
    unsafe_allow_html=True,
)

player_data = {
    code: ensure_xt_model_columns(df)
    for code, df in load_all_players().items()
}
if not any(not df.empty for df in player_data.values()):
    st.error(
        "Nenhum CSV de jogador encontrado. "
        "Esperado: `BG-vs *.csv`, `CS-vs *.csv`, `LP-vs *.csv`, "
        "`Pedri-vs *.csv`, `Rodri-vs *.csv`."
    )
    st.stop()

with st.sidebar:
    st.markdown(
        """
        <div style="text-align:center;">
          <h3 style="margin:0;color:#eef1f7;">Opções</h3>
          <p style="color:#94a3b8;font-size:0.85rem;">Mapas · todos os jogos</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")
    impact_plays_only = st.checkbox(
        "Apenas impact plays nos mapas",
        value=False,
        help=(
            "Mostra um mapa unificado só com passes certos de impacto "
            "(aproximação ao gol + xT v4) e conduções de impacto."
        ),
    )
    st.caption("xT heurístico v4 · 5 jogadores · Stats agregadas")

tab_analysis, tab_world_cup = st.tabs(
    ["Análise", "World Cup"]
)

with tab_analysis:
    render_analysis_tab(player_data, impact_plays_only=impact_plays_only)

with tab_world_cup:
    render_world_cup_tab(player_data)
