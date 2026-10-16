"""Project-wide constants: CRS, grid, thresholds, weights and paths.

Every notebook imports from here, so changing a threshold or weight in one
place changes the whole pipeline. Thresholds are illustrative starting
values, not mission specifications; justify your final choices in the README.
"""
from pathlib import Path

# --- Paths ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"

# --- Mars geodesy -------------------------------------------------------------
MARS_RADIUS_M = 3_396_190.0                 # IAU 2015 Mars sphere
M_PER_DEG = MARS_RADIUS_M * 3.141592653589793 / 180.0   # ~59,274.7 m per degree

# Geographic (lat/lon) and projected CRS. Both planetocentric, east-positive.
GEOG_CRS = "IAU_2015:49900"   # Mars (2015) - Sphere / Ocentric
PROJ_CRS = "IAU_2015:49910"   # Mars (2015) - Sphere / Ocentric / Equirectangular, clon = 0

# --- Reference grid -----------------------------------------------------------
LAT_LIMIT = 30.0              # analysis band is +/- LAT_LIMIT degrees
PIXEL_M = 1000.0              # working resolution
FINE_PIXEL_M = 200.0          # native-ish DEM resolution for roughness / zoom-ins

# --- Normalization: (best, worst) for linear fuzzy membership ----------------
# None means "use a data-driven value" (min or 95th percentile, see criteria).
SCORE_RANGES = {
    "elev":   (-4000.0, 1000.0),   # m relative to the areoid
    "slope":  (0.0, 5.0),          # degrees at 1 km scale
    "rough":  (None, None),        # min -> 95th percentile
    "ti":     (300.0, 100.0),      # J m^-2 K^-1 s^-1/2
    "crater": (0.0, None),         # 0 -> 95th percentile
    "lat":    (0.0, LAT_LIMIT),    # |latitude| in degrees
}

# --- Hard constraints (cells failing any of these score 0) -------------------
MAX_ELEV_M = 1000.0
MAX_SLOPE_DEG = 10.0
MIN_TI = 100.0

# --- AHP ----------------------------------------------------------------------
CRITERIA = ["slope", "rough", "elev", "ti", "crater", "lat"]

# Upper triangle of the pairwise comparison matrix, row vs column, in the
# order of CRITERIA. The lower triangle is filled with reciprocals.
AHP_UPPER = {
    ("slope", "rough"): 1,   ("slope", "elev"): 2,   ("slope", "ti"): 2,
    ("slope", "crater"): 3,  ("slope", "lat"): 4,
    ("rough", "elev"): 2,    ("rough", "ti"): 2,     ("rough", "crater"): 3,
    ("rough", "lat"): 4,
    ("elev", "ti"): 1,       ("elev", "crater"): 2,  ("elev", "lat"): 3,
    ("ti", "crater"): 2,     ("ti", "lat"): 3,
    ("crater", "lat"): 2,
}

# --- Candidate zones and validation -------------------------------------------
CRATER_RADIUS_KM = 50.0       # neighbourhood for crater density
MIN_ZONE_KM2 = 50.0           # about one landing ellipse (Perseverance ~7.7 x 6.6 km)
TOP_N = 10
SITE_RADIUS_KM = 10.0         # sample mean score within this radius of each site
N_RANDOM = 1000
SEED = 42
