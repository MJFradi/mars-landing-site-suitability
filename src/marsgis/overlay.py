"""Normalization, hard constraints, weighted overlay and candidate zones."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import config as C
from .grid import Grid


# --- Normalization -----------------------------------------------------------------
def linear_score(x, best, worst):
    """Linear fuzzy membership: 1 at ``best``, 0 at ``worst``, clipped. NaN stays NaN."""
    if best == worst:
        raise ValueError("best and worst must differ")
    with np.errstate(invalid="ignore"):
        s = (np.asarray(x, dtype="float32") - worst) / (best - worst)
    return np.clip(s, 0, 1)


def normalize(layers: dict[str, np.ndarray], ranges: dict = C.SCORE_RANGES,
              valid: np.ndarray | None = None) -> tuple[dict[str, np.ndarray], dict]:
    """Score every criterion 0-1 (1 = best).

    ``None`` in ``ranges`` means: best = minimum, worst = 95th percentile,
    computed over ``valid`` cells only so nodata and excluded ground don't
    shift the percentiles. Returns the scores and the ranges actually used
    (report these in the README).
    """
    scores, used = {}, {}
    for name, arr in layers.items():
        best, worst = ranges[name]
        sample = arr[valid & np.isfinite(arr)] if valid is not None else arr[np.isfinite(arr)]
        if best is None:
            best = float(np.min(sample))
        if worst is None:
            worst = float(np.percentile(sample, 95))
        scores[name] = linear_score(arr, best, worst)
        used[name] = (best, worst)
    return scores, used


def constraint_mask(dem, slope, ti, max_elev=C.MAX_ELEV_M, max_slope=C.MAX_SLOPE_DEG,
                    min_ti=C.MIN_TI) -> np.ndarray:
    """True where landing is not ruled out. NaN in any input -> False."""
    with np.errstate(invalid="ignore"):
        return (dem <= max_elev) & (slope <= max_slope) & (ti >= min_ti)


def valid_data(*arrays) -> np.ndarray:
    """True where every input is finite."""
    out = np.ones(arrays[0].shape, dtype=bool)
    for a in arrays:
        out &= np.isfinite(a)
    return out


# --- Overlay -------------------------------------------------------------------------
def weighted_overlay(scores: dict[str, np.ndarray], weights: dict[str, float],
                     mask: np.ndarray | None = None) -> np.ndarray:
    """S = sum(w_i * s_i), 0 where ``mask`` is False, NaN where any score is NaN."""
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        raise ValueError(f"weights sum to {total:.4f}, expected 1")
    S = np.zeros(next(iter(scores.values())).shape, dtype="float32")
    for k, w in weights.items():
        S += np.float32(w) * scores[k]
    if mask is not None:
        S = np.where(mask, S, np.where(np.isfinite(S), 0.0, np.nan)).astype("float32")
    return S


def classify(S: np.ndarray, n_classes: int = 5, method: str = "quantile") -> tuple[np.ndarray, np.ndarray]:
    """Classes 1..n (very low .. very high) over cells with S > 0; 0 = excluded, -1 = nodata.

    ``method`` is "quantile" or "equal". Returns (classes, breaks).
    """
    vals = S[np.isfinite(S) & (S > 0)]
    if method == "quantile":
        breaks = np.quantile(vals, np.linspace(0, 1, n_classes + 1))
    elif method == "equal":
        breaks = np.linspace(vals.min(), vals.max(), n_classes + 1)
    else:
        raise ValueError(method)
    cls = np.digitize(S, breaks[1:-1], right=False) + 1
    cls = np.where(np.isfinite(S) & (S > 0), cls, 0)
    cls = np.where(np.isfinite(S), cls, -1)
    return cls.astype("int8"), breaks


# --- Candidate zones -----------------------------------------------------------------
def _label_wrapped(binary: np.ndarray):
    """8-connected labels, merging zones that touch across the +/-180 deg seam."""
    from scipy import ndimage
    lab, n = ndimage.label(binary, structure=np.ones((3, 3), dtype=int))
    if n == 0:
        return lab, n

    # Label pairs that touch across the seam (last column vs first column, 8-connected).
    east, west = lab[:, -1], lab[:, 0]
    pairs = set()
    for shift in (-1, 0, 1):
        w = np.roll(west, shift)
        if shift == 1:
            w[0] = 0
        elif shift == -1:
            w[-1] = 0
        ok = (east > 0) & (w > 0)
        pairs.update(zip(east[ok].tolist(), w[ok].tolist()))
    if not pairs:
        return lab, n

    parent = {}

    def find(a):
        parent.setdefault(a, a)
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    lut = np.arange(n + 1)
    for a in list(parent):
        lut[a] = find(a)
    lab = lut[lab]
    return lab, int(np.unique(lab[lab > 0]).size)


def candidate_zones(S: np.ndarray, grid: Grid, threshold: float, min_km2: float = C.MIN_ZONE_KM2,
                    top_n: int = C.TOP_N) -> tuple[pd.DataFrame, np.ndarray]:
    """Group cells with S >= threshold into zones, drop small ones, rank by area-weighted mean score.

    Returns (table of the top_n zones, label raster where 1..top_n is the rank, 0 elsewhere).
    Area uses the true cell area at each latitude.
    """
    binary = np.isfinite(S) & (S >= threshold)
    lab, n = _label_wrapped(binary)
    cols_out = ["rank", "area_km2", "mean_score", "max_score", "lon", "lat"]
    if n == 0:
        return pd.DataFrame(columns=cols_out), np.zeros(S.shape, dtype="int16")

    # Statistics over labelled cells only (bincount is far faster than ndimage.maximum).
    rows, cols = np.nonzero(lab)
    z = lab[rows, cols]
    s = S[rows, cols].astype("float64")
    a = grid.cell_area_km2()[rows, 0]
    m = int(z.max()) + 1
    area_k = np.bincount(z, a, m)
    mean_k = np.bincount(z, a * s, m) / np.where(area_k > 0, area_k, 1)
    max_k = np.zeros(m)
    np.maximum.at(max_k, z, s)

    idx = np.nonzero(area_k)[0]            # seam merges leave gaps in the label numbers
    df = pd.DataFrame({"zone": idx, "area_km2": area_k[idx], "mean_score": mean_k[idx],
                       "max_score": max_k[idx]})
    df = df[df.area_km2 >= min_km2].sort_values("mean_score", ascending=False).head(top_n)

    W = S.shape[1]
    lons, lats = [], []
    for zone in df.zone:
        sel = z == zone
        r_z, c_z, a_z = rows[sel], cols[sel].astype("float64"), a[sel]
        if c_z.min() == 0 and c_z.max() == W - 1:            # straddles the seam
            c_z = np.where(c_z < W / 2, c_z + W, c_z)
        lon, lat = grid.rowcol_to_lonlat(np.average(r_z, weights=a_z),
                                         np.average(c_z, weights=a_z) % W)
        lons.append(float(lon)); lats.append(float(lat))
    df["lon"], df["lat"] = lons, lats
    df.insert(0, "rank", np.arange(1, len(df) + 1))

    lut = np.zeros(m, dtype="int16")
    lut[df.zone.to_numpy()] = df["rank"].to_numpy()
    ranked = np.zeros(S.shape, dtype="int16")
    ranked[rows, cols] = lut[z]
    return df.drop(columns="zone").reset_index(drop=True)[cols_out], ranked


def zones_to_geodataframe(ranked: np.ndarray, table: pd.DataFrame, grid: Grid):
    """Polygonize the ranked zone raster and join the attribute table."""
    import geopandas as gpd
    from rasterio.features import shapes
    from shapely.geometry import shape
    from shapely.ops import unary_union

    geoms = {}
    for geom, val in shapes(ranked, mask=ranked > 0, transform=grid.transform):
        geoms.setdefault(int(val), []).append(shape(geom))
    t = table.copy()
    t["geometry"] = [unary_union(geoms[r]) for r in t["rank"]]
    return gpd.GeoDataFrame(t, geometry="geometry", crs=grid.crs)
