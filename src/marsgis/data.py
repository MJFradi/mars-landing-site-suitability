"""Data acquisition and warping onto the reference grid.

Sources (verified 2026-10-15, see data/SOURCES.md):
- DEM: USGS MOLA-HRSC blended DEM v2, 200 m, int16, nodata -32768, geographic
  Mars 2000 sphere, -180..180 E. 11 GB, but it has internal overviews and can be
  read over HTTP, so the +/-30 deg band can be warped without downloading it all.
- Thermal inertia: Putzig & Mellon (2007) TES map, PDS MGS-M-TES-5-TIMAP-V1.0.
  7200 x 3600 big-endian int16, 20 px/deg, -180..180 E, 0 = null.
- Craters: Robbins & Hynek (2012) database, 384,343 craters >= 1 km, from the
  USGS Astrogeology OGC API (fields lon_e [-180..180], lat, diamkm, craterid).
- Geology (optional): Tanaka et al. (2014) SIM 3292, shapefiles in degrees.
"""
from __future__ import annotations

import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from . import config as C
from .grid import Grid

DEM_URL = ("https://asc-pds-services.s3.us-west-2.amazonaws.com/mosaic/Mars/HRSC_MOLA_Blend/"
           "Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif")
TI_URL = "https://pds-geosciences.wustl.edu/mgs/mgs-m-tes-5-timap-v1/mgst_9001/data/"
CRATER_API = "https://astrogeology.usgs.gov/pygeoapi/collections/mars/robbinsv1/items"
GEOLOGY_ZIP = "https://pubs.usgs.gov/sim/3292/downloads/sim3292_database.zip"

# Geographic CRS of the source products (Mars 2000 / IAU 2015 sphere, same radius).
SRC_GEOG_WKT = ('GEOGCS["Mars 2000 Sphere",DATUM["Mars_2000_Sphere",'
                'SPHEROID["Mars_2000_Sphere",3396190,0]],PRIMEM["Reference_Meridian",0],'
                'UNIT["degree",0.0174532925199433]]')


USER_AGENT = "mars-landing-site-suitability/0.1 (research; python-urllib)"


def _open(url: str, timeout: int = 120):
    """urlopen with a User-Agent (the USGS API rejects Python's default one with 403)."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    return urllib.request.urlopen(req, timeout=timeout)


def _download(url: str, dst: Path, chunk: int = 1 << 20) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        print(f"exists, skipping: {dst.name}")
        return dst
    tmp = dst.with_suffix(dst.suffix + ".part")
    print(f"downloading {url}")
    with _open(url) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        while block := r.read(chunk):
            f.write(block)
            done += len(block)
            if total:
                print(f"\r  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB", end="")
    print()
    tmp.replace(dst)
    return dst


# --- Warping ------------------------------------------------------------------------
def warp_to_grid(src, dst: str | Path, grid: Grid, resampling: str = "average",
                 src_nodata: float | None = None, overview_level: int | None = None,
                 src_transform=None, rows_per_chunk: int = 256) -> Path:
    """Warp any GDAL-readable raster (local path or /vsicurl/ URL) onto ``grid``.

    Reads in row chunks through a WarpedVRT, so only the source pixels under
    the band are fetched and memory stays small. ``overview_level`` reads an
    internal source overview (0 = first, 2x coarser). ``src_transform``
    georeferences a source that has none (e.g. a page of an external .ovr;
    see ``dem_overview``).
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    kw = {} if overview_level is None else {"overview_level": overview_level}
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    profile = {"driver": "GTiff", "height": grid.height, "width": grid.width, "count": 1,
               "dtype": "float32", "crs": CRS.from_user_input(grid.crs), "transform": grid.transform,
               "nodata": -9999.0, "compress": "deflate", "tiled": True,
               "blockxsize": 512, "blockysize": 512, "BIGTIFF": "IF_SAFER"}
    with rasterio.Env(GDAL_HTTP_MULTIRANGE="YES", GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
                      GDAL_HTTP_MAX_RETRY="10", GDAL_HTTP_RETRY_DELAY="5", GDAL_HTTP_TIMEOUT="120",
                      VSI_CACHE="TRUE"):   # keep readdir on: the DEM overviews are an external .ovr
        with rasterio.open(src, **kw) as s:
            vrt_kw = dict(crs=CRS.from_user_input(grid.crs), transform=grid.transform,
                          width=grid.width, height=grid.height,
                          resampling=getattr(Resampling, resampling), dtype="float32",
                          nodata=-9999.0)
            if src_nodata is not None:
                vrt_kw["src_nodata"] = src_nodata
            if s.crs is None:
                vrt_kw["src_crs"] = CRS.from_wkt(SRC_GEOG_WKT)
            if src_transform is not None:
                vrt_kw["src_transform"] = src_transform
            with WarpedVRT(s, **vrt_kw) as vrt, rasterio.open(dst, "w", **profile) as out:
                t0 = time.time()
                for r0 in range(0, grid.height, rows_per_chunk):
                    h = min(rows_per_chunk, grid.height - r0)
                    win = Window(0, r0, grid.width, h)
                    out.write(vrt.read(1, window=win), 1, window=win)
                    q = 4 * (r0 + h) // grid.height          # report at 25 % steps
                    if q > 4 * r0 // grid.height:
                        print(f"  {dst.name}: {25 * q}%  ({time.time() - t0:,.0f} s)")
    return dst


# --- DEM ------------------------------------------------------------------------------
def dem_source(local: str | Path | None = None) -> str:
    """Local copy if given and present, else the remote file via /vsicurl/."""
    if local and Path(local).exists():
        return str(local)
    return "/vsicurl/" + DEM_URL


def dem_overview(factor: int = 4, base: str | None = None):
    """(path, transform) of a DEM overview page, for a quick first pass.

    The DEM's overviews live in an external 3.8 GB ``.ovr`` (pages 2x, 4x,
    8x, ...), tiled 128 x 128, without georeferencing of their own. Page 4x
    (~800 m) covers the +/-30 deg band in ~240 MB of reads.
    """
    import rasterio
    from rasterio.transform import from_bounds
    base = base or dem_source()
    page = {2: 1, 4: 2, 8: 3, 16: 4}[factor]
    ovr = f"GTIFF_DIR:{page}:{base}.ovr"
    with rasterio.Env(VSI_CACHE="TRUE"):
        with rasterio.open(base) as b, rasterio.open(ovr) as o:
            t = from_bounds(*b.bounds, width=o.width, height=o.height)
    return ovr, t


def fine_dem_window(lon: float, lat: float, half_km: float, dst: str | Path,
                    src: str | None = None) -> Path:
    """Cut a ~2*half_km square of the 200 m DEM around (lon, lat), in the project CRS."""
    half = int(np.ceil(half_km * 1000 / C.FINE_PIXEL_M)) * C.FINE_PIXEL_M
    x = lon * C.M_PER_DEG
    y = lat * C.M_PER_DEG
    x0 = np.floor((x - half) / C.FINE_PIXEL_M) * C.FINE_PIXEL_M
    y0 = np.floor((y - half) / C.FINE_PIXEL_M) * C.FINE_PIXEL_M
    g = Grid(x0, y0, x0 + 2 * half, y0 + 2 * half, C.FINE_PIXEL_M)
    return warp_to_grid(src or dem_source(), dst, g, resampling="bilinear", src_nodata=-32768)


# --- Thermal inertia -------------------------------------------------------------------
def thermal_inertia_geotiff(dst: str | Path, which: str = "night") -> Path:
    """Download the TES thermal inertia map and save it as a georeferenced GeoTIFF.

    ``which`` = "night" (default; nightside TI is less affected by slope and
    albedo heating) or "day". 0 (null) becomes nodata.
    """
    import rasterio
    from rasterio.crs import CRS
    from rasterio.transform import from_origin

    name = f"global_ti_{which}_2007.img"
    raw = _download(TI_URL + name, C.RAW / name)
    a = np.fromfile(raw, dtype=">i2")
    if a.size != 7200 * 3600:
        raise ValueError(f"unexpected size {a.size} for {raw}; check the PDS label")
    a = a.reshape(3600, 7200).astype("float32")
    a[a <= 0] = np.nan
    dst = Path(dst)
    with rasterio.open(dst, "w", driver="GTiff", height=3600, width=7200, count=1, dtype="float32",
                       crs=CRS.from_wkt(SRC_GEOG_WKT), transform=from_origin(-180, 90, 0.05, 0.05),
                       nodata=-9999.0, compress="deflate") as out:
        out.write(np.where(np.isfinite(a), a, -9999.0).astype("float32"), 1)
    return dst


# --- Craters --------------------------------------------------------------------------------
def download_craters(dst: str | Path, lat_limit: float = C.LAT_LIMIT + 1.0, min_diam_km: float = 1.0,
                     page: int = 10_000) -> Path:
    """Page through the USGS OGC API for craters within +/-lat_limit; save CSV.

    The band is 1 deg wider than the analysis band so crater density near
    the edges still counts craters just outside it.
    """
    import pandas as pd
    dst = Path(dst)
    if dst.exists():
        print(f"exists, skipping: {dst.name}")
        return dst
    rows, offset = [], 0
    bbox = f"-180,{-lat_limit},180,{lat_limit}"
    while True:
        url = f"{CRATER_API}?f=json&limit={page}&offset={offset}&bbox={bbox}"
        for attempt in range(5):
            try:
                with _open(url) as r:
                    feats = json.load(r)["features"]
                break
            except Exception as e:                      # flaky network: retry with backoff
                if attempt == 4:
                    raise
                print(f"  retry {attempt + 1}: {e}")
                time.sleep(2 ** attempt)
        rows += [f["properties"] for f in feats]
        offset += len(feats)
        print(f"\r  {offset:,} craters", end="")
        if len(feats) < page:
            break
    print()
    df = pd.DataFrame(rows)
    df = df[df.diamkm >= min_diam_km]
    dst.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst, index=False)
    return dst


def download_geology(dst_dir: str | Path = C.RAW / "sim3292") -> Path:
    """Download and unzip the Tanaka et al. (2014) database (~810 MB)."""
    import zipfile
    z = _download(GEOLOGY_ZIP, C.RAW / "sim3292_database.zip")
    dst_dir = Path(dst_dir)
    if not dst_dir.exists():
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dst_dir)
    return dst_dir
