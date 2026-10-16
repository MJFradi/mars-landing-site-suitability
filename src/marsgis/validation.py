"""Validation against real landing sites."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from .grid import Grid, lon360_to_180

SITES_CSV = C.ROOT / "data" / "landing_sites.csv"


def load_sites(path: str | Path = SITES_CSV) -> pd.DataFrame:
    """Landing sites with planetocentric lat and east-positive lon (-180..180)."""
    df = pd.read_csv(path, comment="#")
    df["lon_e180"] = lon360_to_180(df["lon_e"])
    return df


def disk_offsets(grid: Grid, lat: float, radius_km: float):
    """Row/col offsets of cells whose centres lie within ``radius_km`` (true distance)."""
    res_km = grid.res / 1000.0
    ry = radius_km / res_km
    rx = ry / np.cos(np.radians(lat))
    hy, hx = int(np.ceil(ry)), int(np.ceil(rx))
    dy, dx = np.mgrid[-hy:hy + 1, -hx:hx + 1]
    inside = (dy / ry) ** 2 + (dx / rx) ** 2 <= 1.0
    return dy[inside], dx[inside]


def disk_mean(S: np.ndarray, grid: Grid, lon: float, lat: float,
              radius_km: float = C.SITE_RADIUS_KM) -> float:
    """Mean of finite S within ``radius_km`` of (lon, lat); longitude wraps."""
    r, c = grid.lonlat_to_rowcol(lon, lat)
    dy, dx = disk_offsets(grid, lat, radius_km)
    rr, cc = int(r) + dy, (int(c) + dx) % S.shape[1]
    ok = (rr >= 0) & (rr < S.shape[0])
    vals = S[rr[ok], cc[ok]]
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if vals.size else float("nan")


def percentile_of(value: float, reference: np.ndarray) -> float:
    """Share of ``reference`` strictly below ``value`` plus half the ties, in %."""
    ref = reference[np.isfinite(reference)]
    below = np.count_nonzero(ref < value)
    ties = np.count_nonzero(ref == value)
    return 100.0 * (below + 0.5 * ties) / ref.size


def random_points(valid: np.ndarray, grid: Grid, n: int = C.N_RANDOM, seed: int = C.SEED,
                  min_sep_cells: int = 0):
    """``n`` random valid cells, uniform by area (rows weighted by cos(lat)).

    Returns (lon, lat) arrays of cell centres.
    """
    rng = np.random.default_rng(seed)
    rows, cols = np.nonzero(valid)
    w = np.cos(np.radians(grid.row_lat()))[rows]
    pick = rng.choice(rows.size, size=n, replace=False, p=w / w.sum())
    return grid.rowcol_to_lonlat(rows[pick], cols[pick])


@dataclass
class ValidationResult:
    sites: pd.DataFrame
    random_scores: np.ndarray
    u_statistic: float
    p_value: float
    top_share: float           # share of sites in the top ``top_pct`` %
    top_pct: float
    p_value_passing: float = float("nan")      # vs. constraint-passing ground only
    top_share_passing: float = float("nan")

    def sentence(self) -> str:
        k = int(round(self.top_share * len(self.sites)))
        return (f"{k} of {len(self.sites)} sites fall in the top {self.top_pct:.0f}% "
                f"(Mann-Whitney U = {self.u_statistic:.0f}, one-sided p = {self.p_value:.2g}).")

    def strict_sentence(self) -> str:
        if np.isnan(self.p_value_passing):
            return "strict test not run (pass ``passing`` to validate)"
        k = int(round(self.top_share_passing * len(self.sites)))
        return (f"Against constraint-passing ground only: {k} of {len(self.sites)} sites in the top "
                f"{self.top_pct:.0f}% (one-sided p = {self.p_value_passing:.2g}).")


def validate(S: np.ndarray, grid: Grid, sites: pd.DataFrame, valid: np.ndarray,
             radius_km: float = C.SITE_RADIUS_KM, n_random: int = C.N_RANDOM,
             seed: int = C.SEED, top_pct: float = 20.0,
             passing: np.ndarray | None = None) -> ValidationResult:
    """Score each site, rank it against the map, and test it against random points.

    ``valid`` marks cells with data (excluded cells, S = 0, are still valid:
    landing sites are compared with the whole band, not only with ground that
    already passed the constraints, which would flatter the result).
    Percentiles are against the distribution of disk means at random points,
    so sites and the reference are measured the same way. If ``passing`` (the
    hard-constraint mask) is given, a stricter second reference is drawn only
    from ground that passes the constraints (column ``percentile_passing``).
    """
    from scipy.stats import mannwhitneyu

    rlon, rlat = random_points(valid, grid, n_random, seed)
    rand = np.array([disk_mean(S, grid, lo, la, radius_km) for lo, la in zip(rlon, rlat)])

    out = sites.copy()
    out["score"] = [disk_mean(S, grid, lo, la, radius_km) for lo, la in zip(out.lon_e180, out.lat)]
    r, c = grid.lonlat_to_rowcol(out.lon_e180.values, out.lat.values)
    out["point_score"] = S[r, c]
    out["percentile"] = [percentile_of(v, rand) for v in out.score]
    u, p = mannwhitneyu(out.score.dropna(), rand[np.isfinite(rand)], alternative="greater")
    share = float(np.mean(out.percentile >= 100 - top_pct))
    res = ValidationResult(out, rand, float(u), float(p), share, top_pct)
    if passing is not None:
        plon, plat = random_points(passing & valid, grid, n_random, seed + 1)
        rp = np.array([disk_mean(S, grid, lo, la, radius_km) for lo, la in zip(plon, plat)])
        out["percentile_passing"] = [percentile_of(v, rp) for v in out.score]
        res.p_value_passing = float(mannwhitneyu(out.score.dropna(), rp[np.isfinite(rp)],
                                                 alternative="greater").pvalue)
        res.top_share_passing = float(np.mean(out.percentile_passing >= 100 - top_pct))
    return res
