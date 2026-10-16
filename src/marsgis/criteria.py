"""Criterion layers: slope, roughness, crater density, latitude, science value.

Every function takes and returns plain NumPy arrays on the reference grid,
with NaN for nodata, so they can be tested without any files.
"""
from __future__ import annotations

import numpy as np

from . import config as C
from .grid import Grid, lon360_to_180


def _row_cos(grid: Grid, shape) -> np.ndarray:
    if shape != grid.shape:
        raise ValueError(f"array shape {shape} does not match grid {grid.shape}")
    return np.cos(np.radians(grid.row_lat()))[:, None]


# --- Slope -----------------------------------------------------------------------
def slope_deg(dem: np.ndarray, grid: Grid) -> np.ndarray:
    """Slope in degrees, with east-west pixel spacing corrected for latitude.

    ``np.gradient(dem, px)`` alone treats every pixel as px x px metres; in
    equirectangular the east-west spacing is ``px * cos(lat)``, so without the
    correction slopes at 30 deg would be underestimated by up to ~13 %.
    """
    px = grid.res
    dzdy, dzdx = np.gradient(dem.astype("float64"), px)
    dzdx = dzdx / _row_cos(grid, dem.shape)
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy))).astype("float32")


# --- Roughness -------------------------------------------------------------------
def tri(dem: np.ndarray) -> np.ndarray:
    """Terrain Ruggedness Index: mean absolute difference to the 8 neighbours (m).

    On a 1 km grid this mostly measures slope; prefer ``block_std`` from the
    200 m DEM when it is available.
    """
    p = np.pad(dem.astype("float32"), 1, mode="edge")
    h, w = dem.shape
    diffs = [np.abs(p[i:i + h, j:j + w] - dem)
             for i in range(3) for j in range(3) if (i, j) != (1, 1)]
    with np.errstate(invalid="ignore"):
        return np.nanmean(diffs, axis=0).astype("float32")


def block_std(fine_dem: np.ndarray, factor: int = 5, detrend: bool = True) -> np.ndarray:
    """Roughness as the std of fine elevations inside each coarse cell.

    ``fine_dem`` must be on a grid that nests exactly into the coarse grid
    (e.g. 200 m pixels, ``factor=5`` for 1 km). With ``detrend=True`` a best-fit
    plane is removed from each block first, so a smooth tilted plain is not
    reported as rough (slope is already its own criterion).
    """
    h, w = fine_dem.shape
    H, W = h // factor, w // factor
    blocks = (fine_dem[:H * factor, :W * factor]
              .reshape(H, factor, W, factor).swapaxes(1, 2)
              .reshape(H, W, factor * factor).astype("float64"))
    if detrend:
        yy, xx = np.mgrid[0:factor, 0:factor]
        X = np.column_stack([np.ones(factor * factor), xx.ravel(), yy.ravel()])
        P = X @ np.linalg.pinv(X)              # projection onto the plane
        filled = np.where(np.isnan(blocks), np.nanmean(blocks, axis=2, keepdims=True), blocks)
        blocks = np.where(np.isnan(blocks), np.nan, filled - filled @ P.T)
    with np.errstate(invalid="ignore"):
        return np.nanstd(blocks, axis=2).astype("float32")


# --- Neighbourhood counts (crater density, mineral proximity) ----------------------
def _ellipse_kernel(ry: float, rx: float, half_h: int) -> np.ndarray:
    half_w = int(np.ceil(rx))
    yy, xx = np.mgrid[-half_h:half_h + 1, -half_w:half_w + 1]
    return ((yy / ry) ** 2 + (xx / rx) ** 2 <= 1.0).astype("float64")


def neighbourhood_count(lon, lat, grid: Grid, radius_km: float, band_deg: float = 2.0) -> np.ndarray:
    """Number of points within ``radius_km`` (true ground distance) of each cell.

    Points outside the grid's latitude band still count for cells near its
    edge, so pass a catalogue clipped to a slightly wider band (e.g. +/-31 deg).
    Longitude wraps around at +/-180 deg.
    """
    from scipy.signal import oaconvolve

    res_km = grid.res / 1000.0
    ry = radius_km / res_km
    m = int(np.ceil(ry))
    H, W = grid.shape

    # Count grid extended m rows north and south of the band.
    row, col = grid.lonlat_to_rowcol(lon360_to_180(lon), lat)
    row = row + m
    keep = (row >= 0) & (row < H + 2 * m) & (col >= 0) & (col < W)
    ext = np.zeros((H + 2 * m, W), dtype="float64")
    np.add.at(ext, (row[keep], col[keep]), 1.0)

    out = np.empty((H, W), dtype="float32")
    lats = grid.row_lat()
    band_rows = max(1, int(round(band_deg * C.M_PER_DEG / grid.res)))
    for r0 in range(0, H, band_rows):
        r1 = min(H, r0 + band_rows)
        lat_c = np.mean(lats[r0:r1])
        rx = ry / np.cos(np.radians(lat_c))
        k = _ellipse_kernel(ry, rx, m)
        kx = k.shape[1] // 2
        sub = ext[r0:r1 + 2 * m]
        sub = np.concatenate([sub[:, -kx:], sub, sub[:, :kx]], axis=1)   # wrap longitude
        out[r0:r1] = oaconvolve(sub, k, mode="valid")
    return np.clip(np.round(out, 6), 0, None)


def crater_density(lon, lat, grid: Grid, radius_km: float = C.CRATER_RADIUS_KM) -> np.ndarray:
    """Craters per 10,000 km^2, counted within ``radius_km`` of each cell."""
    counts = neighbourhood_count(lon, lat, grid, radius_km)
    return (counts / (np.pi * radius_km ** 2) * 10_000.0).astype("float32")


# --- Latitude ----------------------------------------------------------------------
def abs_latitude(grid: Grid) -> np.ndarray:
    return np.abs(grid.lat_grid())


# --- Science value (optional seventh criterion) -----------------------------------
def geologic_unit_score(unit: str) -> float:
    """Score a Tanaka et al. (2014) unit label by age.

    Unit labels start with lowercase epoch modifiers (e/m/l = early/mid/late)
    then period letters: N = Noachian, H = Hesperian, A = Amazonian, e.g.
    ``eNh``, ``lHl``, ``AHv``, ``HNt``. Noachian/Hesperian -> 1.0,
    Amazonian -> 0.5, mixed Amazonian-Hesperian -> 0.75, unknown -> NaN.
    """
    if not isinstance(unit, str):
        return float("nan")
    s = unit.strip().lstrip("eml")
    periods = ""
    for ch in s:
        if ch in "NHA":
            periods += ch
        else:
            break
    if not periods:
        return float("nan")
    if "A" in periods and ("H" in periods or "N" in periods):
        return 0.75
    return 0.5 if periods == "A" else 1.0


def rasterize_polygons(gdf, values, grid: Grid, fill: float = np.nan) -> np.ndarray:
    """Burn polygon ``values`` onto the grid. ``gdf`` must already be in ``grid.crs``."""
    from rasterio.features import rasterize
    shapes = ((geom, float(v)) for geom, v in zip(gdf.geometry, values) if np.isfinite(v))
    return rasterize(shapes, out_shape=grid.shape, transform=grid.transform,
                     fill=fill, dtype="float32", all_touched=False)


def proximity_bonus(lon, lat, grid: Grid, radius_km: float = 50.0) -> np.ndarray:
    """1 where any point (e.g. hydrated-mineral detection) lies within ``radius_km``, else 0."""
    return (neighbourhood_count(lon, lat, grid, radius_km) > 0).astype("float32")
