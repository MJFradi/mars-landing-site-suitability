"""Fetch every input and put it on the 1 km reference grid.

    python scripts/download_data.py              # DEM read remotely at full 200 m (slow, ~4 GB streamed)
    python scripts/download_data.py --quick      # DEM from the 4x overview (~800 m, ~240 MB, minutes)
    python scripts/download_data.py --dem-file data/raw/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
    python scripts/download_data.py --geology    # also fetch the 810 MB geologic map

Outputs in data/processed/: dem_1km.tif, ti_1km.tif, and data/raw/robbins_craters_lat31.csv.
"""
import argparse

from marsgis import config as C, data, grid as G


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="read the DEM's 4x overview instead of full resolution")
    ap.add_argument("--dem-file", help="local copy of the 11 GB DEM (otherwise read over HTTP)")
    ap.add_argument("--geology", action="store_true", help="also download the Tanaka et al. (2014) map")
    ap.add_argument("--skip-dem", action="store_true")
    a = ap.parse_args()

    grid = G.reference_grid()
    print(f"reference grid: {grid.width} x {grid.height} px, {grid.res:.0f} m, {grid.crs}")
    C.RAW.mkdir(parents=True, exist_ok=True)
    C.PROCESSED.mkdir(parents=True, exist_ok=True)

    print("\n[1/3] craters (Robbins & Hynek 2012, USGS API)")
    data.download_craters(C.RAW / "robbins_craters_lat31.csv")

    print("\n[2/3] thermal inertia (Putzig & Mellon 2007, PDS)")
    ti_geo = data.thermal_inertia_geotiff(C.RAW / "ti_night_2007_geo.tif")
    data.warp_to_grid(ti_geo, C.PROCESSED / "ti_1km.tif", grid, resampling="bilinear")

    if not a.skip_dem:
        print("\n[3/3] DEM (MOLA-HRSC blend v2, USGS)")
        src, t = data.dem_source(a.dem_file), None
        if a.quick:
            src, t = data.dem_overview(4, src)
        data.warp_to_grid(src, C.PROCESSED / "dem_1km.tif", grid, resampling="average",
                          src_nodata=-32768, src_transform=t)

    if a.geology:
        print("\n[extra] geologic map (Tanaka et al. 2014)")
        print(data.download_geology())

    G.check_aligned([p for p in (C.PROCESSED / "dem_1km.tif", C.PROCESSED / "ti_1km.tif") if p.exists()])
    print("\nAll rasters aligned. Next: notebooks/01_preprocessing.ipynb")


if __name__ == "__main__":
    main()
