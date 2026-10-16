"""Reference grid, coordinate conventions and raster I/O.

All rasters in the project share one grid: Mars equirectangular
(IAU_2015:49910), +/-30 deg latitude, full longitude, 1 km pixels.

Equirectangular keeps north-south distances true everywhere, but an
east-west pixel covers only ``PIXEL_M * cos(lat)`` of real ground. At 30 deg
that is 866 m instead of 1000 m, so slope, kernels and areas below take the
latitude into account instead of assuming square 1 km cells.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import config as C


# --- Longitude / latitude conventions -----------------------------------------
def lon360_to_180(lon):
    """0-360 E -> -180..180 E (both east-positive)."""
    return ((np.asarray(lon, dtype="float64") + 180.0) % 360.0) - 180.0


def lon180_to_360(lon):
    """-180..180 E -> 0-360 E."""
    return np.asarray(lon, dtype="float64") % 360.0


def west_to_east(lon_west):
    """West-positive longitude (old Viking-era products) -> east-positive 0-360."""
    return (360.0 - np.asarray(lon_west, dtype="float64")) % 360.0


def planetographic_to_planetocentric(lat_pg, a=3_396_190.0, b=3_376_200.0):
    """Convert planetographic latitude (deg) to planetocentric (deg).

    Older mission documents sometimes give planetographic latitudes; the
    difference reaches ~0.34 deg (~20 km) at mid-latitudes.
    """
    lat = np.radians(np.asarray(lat_pg, dtype="float64"))
    return np.degrees(np.arctan((b / a) ** 2 * np.tan(lat)))


# --- Reference grid -------------------------------------------------------------
@dataclass(frozen=True)
class Grid:
    xmin: float
    ymin: float
    xmax: float
    ymax: float
    res: float
    crs: str = C.PROJ_CRS

    @property
    def width(self) -> int:
        return int(round((self.xmax - self.xmin) / self.res))

    @property
    def height(self) -> int:
        return int(round((self.ymax - self.ymin) / self.res))

    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def transform(self):
        from rasterio.transform import from_origin
        return from_origin(self.xmin, self.ymax, self.res, self.res)

    def gdalwarp_te(self) -> str:
        return f"{self.xmin:.0f} {self.ymin:.0f} {self.xmax:.0f} {self.ymax:.0f}"

    def row_lat(self) -> np.ndarray:
        """Planetocentric latitude (deg) of each row centre, north to south."""
        y = self.ymax - (np.arange(self.height) + 0.5) * self.res
        return y / C.M_PER_DEG

    def col_lon(self) -> np.ndarray:
        """East-positive longitude (deg, -180..180) of each column centre."""
        x = self.xmin + (np.arange(self.width) + 0.5) * self.res
        return x / C.M_PER_DEG

    def lat_grid(self) -> np.ndarray:
        return np.repeat(self.row_lat()[:, None], self.width, axis=1).astype("float32")

    def lonlat_to_rowcol(self, lon, lat):
        """East-positive lon (either convention) and planetocentric lat -> (row, col)."""
        lon = lon360_to_180(lon)
        x = np.asarray(lon) * C.M_PER_DEG
        y = np.asarray(lat, dtype="float64") * C.M_PER_DEG
        col = np.floor((x - self.xmin) / self.res).astype(int)
        row = np.floor((self.ymax - y) / self.res).astype(int)
        # The grid stops just short of +/-180 deg; snap that sliver to the edge column.
        col = np.clip(col, 0, self.width - 1)
        return row, col

    def rowcol_to_lonlat(self, row, col):
        x = self.xmin + (np.asarray(col, dtype="float64") + 0.5) * self.res
        y = self.ymax - (np.asarray(row, dtype="float64") + 0.5) * self.res
        return x / C.M_PER_DEG, y / C.M_PER_DEG

    def cell_area_km2(self) -> np.ndarray:
        """True ground area of each row's cells (km^2), shape (height, 1)."""
        return ((self.res / 1000.0) ** 2 * np.cos(np.radians(self.row_lat())))[:, None]


def reference_grid(res: float = C.PIXEL_M, lat_limit: float = C.LAT_LIMIT) -> Grid:
    """The project grid, snapped inward to whole pixels so it never crosses +/-180."""
    half_x = np.floor(np.pi * C.MARS_RADIUS_M / res) * res
    half_y = np.floor(lat_limit * C.M_PER_DEG / res) * res
    return Grid(-half_x, -half_y, half_x, half_y, res)


def gdalwarp_cmd(src: str | Path, dst: str | Path, grid: Grid, resampling: str = "average",
                 src_nodata: float | None = None, dst_nodata: float = -9999) -> str:
    """Build the gdalwarp command that puts ``src`` onto ``grid``.

    Use ``average`` when downsampling (DEM 200 m -> 1 km) and ``bilinear``
    when upsampling (thermal inertia ~3 km -> 1 km).
    """
    parts = [
        "gdalwarp", "-overwrite",
        f'-t_srs "{grid.crs}"',
        f"-te {grid.gdalwarp_te()}",
        f"-tr {grid.res:.0f} {grid.res:.0f}",
        f"-r {resampling}",
        f"-dstnodata {dst_nodata}",
        "-ot Float32",
        "-co COMPRESS=DEFLATE -co TILED=YES -co BIGTIFF=IF_SAFER",
        "-multi -wo NUM_THREADS=ALL_CPUS",
    ]
    if src_nodata is not None:
        parts.append(f"-srcnodata {src_nodata}")
    parts += [f'"{src}"', f'"{dst}"']
    return " ".join(parts)


# --- Raster I/O -------------------------------------------------------------------
def read_raster(path: str | Path):
    """Read band 1 as float32 with nodata replaced by NaN. Returns (array, profile)."""
    import rasterio
    with rasterio.open(path) as src:
        arr = src.read(1).astype("float32")
        nodata = src.nodata
        profile = src.profile.copy()
    if nodata is not None:
        arr[arr == nodata] = np.nan
    return arr, profile


def write_raster(path: str | Path, arr: np.ndarray, grid: Grid | None = None,
                 profile: dict | None = None, nodata: float = -9999.0):
    """Write a float32 GeoTIFF; NaN is stored as ``nodata``."""
    import rasterio
    from rasterio.crs import CRS
    if profile is None:
        if grid is None:
            raise ValueError("pass a grid or a profile")
        profile = {"driver": "GTiff", "height": grid.height, "width": grid.width,
                   "count": 1, "crs": CRS.from_user_input(grid.crs),
                   "transform": grid.transform}
    profile = dict(profile)
    profile.update(dtype="float32", nodata=nodata, count=1, compress="deflate",
                   tiled=True, blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
    out = np.where(np.isfinite(arr), arr, nodata).astype("float32")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(out, 1)


def check_aligned(paths) -> None:
    """Raise if any raster differs from the first in shape, transform or CRS."""
    import rasterio
    ref = None
    for p in paths:
        with rasterio.open(p) as src:
            key = (src.shape, tuple(round(v, 6) for v in src.transform[:6]), src.crs.to_wkt())
        if ref is None:
            ref, ref_path = key, p
        elif key != ref:
            raise ValueError(f"{p} is not aligned with {ref_path}")
