# Data sources

All links and product details checked on 2026-10-15. Everything is free and public.
`scripts/download_data.py` fetches all of it.

## Topography — MOLA–HRSC blended DEM v2 (200 m)

- **Product page:** https://astrogeology.usgs.gov/search/map/mars_mgs_mola_mex_hrsc_blended_dem_global_200m
- **File:** https://planetarymaps.usgs.gov/mosaic/Mars/HRSC_MOLA_Blend/Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif
  (redirects to an AWS S3 copy; 11 GB)
- **Format:** GeoTIFF, int16, nodata −32768, 106,694 × 53,347 px, ~200 m/px (0.003374°),
  simple cylindrical (geographic) on the Mars 2000 sphere (R = 3,396,190 m),
  planetocentric latitude, longitude −180…180° E. Elevations in metres relative to the areoid.
  Internal overviews at 2×–510×; row-striped, uncompressed, so remote reads of a latitude band are efficient.
- **Citation:** Fergason, R. L., Hare, T. M., & Laura, J. (2018). *HRSC and MOLA Blended Digital
  Elevation Model at 200m v2.* Astrogeology PDS Annex, U.S. Geological Survey.
- **Credit line for maps:** "MOLA/HRSC DEM: NASA/ESA/DLR/USGS"

## Surface hardness — TES thermal inertia (Putzig & Mellon 2007)

- **PDS volume:** https://pds-geosciences.wustl.edu/missions/mgs/tes-timap.html
  (data set MGS-M-TES-5-TIMAP-V1.0, volume MGST_9001)
- **File used:** `data/global_ti_night_2007.img` (+ `.lbl`), nightside apparent thermal inertia.
  Daytime map and interpolation masks (`global_ti_msk_*`) are in the same folder.
- **Format:** 7,200 × 3,600, 16-bit big-endian signed integer, 20 px/deg (~2.96 km/px),
  simple cylindrical, planetocentric, longitude −180…180° E, 0 = null.
  Units J m⁻² K⁻¹ s⁻½ (tiu); valid range 5–5,000. Gaps between ±87° are infilled by interpolation (<8 % of pixels).
- **Citation:** Putzig, N. E., & Mellon, M. T. (2007). Apparent thermal inertia and the surface
  heterogeneity of Mars. *Icarus*, 191, 68–94. https://doi.org/10.1016/j.icarus.2007.05.013
- Also mirrored at https://se.psi.edu/inertia/2007 and http://www.mars.asu.edu/data/tes_putzigti/

## Hazards — Robbins & Hynek crater database (D ≥ 1 km)

- **Access:** USGS Astrogeology OGC API — https://astrogeology.usgs.gov/pygeoapi/collections/mars/robbinsv1
  (`/items?f=json&bbox=-180,-31,180,31&limit=10000&offset=…`)
- **Content:** 384,343 craters globally (231,918 within ±31°), fields `lon_e` (−180…180° E despite the name),
  `lat` (planetocentric), `diamkm`, `craterid`. Statistically complete to D ≈ 1 km.
- **Citation:** Robbins, S. J., & Hynek, B. M. (2012). A new global database of Mars impact craters ≥1 km:
  1. Database creation, properties, and parameters. *JGR Planets*, 117, E05004. https://doi.org/10.1029/2011JE003966
- Note: press coverage quotes "635,000 craters"; that count includes craters below 1 km. The ≥1 km catalogue is 384,343.

## Science value (optional) — Geologic map of Mars (Tanaka et al. 2014)

- **Publication:** https://pubs.usgs.gov/sim/3292/ (https://doi.org/10.3133/sim3292)
- **File:** https://pubs.usgs.gov/sim/3292/downloads/sim3292_database.zip (~810 MB)
- **Format:** ESRI file geodatabase (Robinson projection, Mars 2000 sphere) **and** shapefiles in decimal degrees.
  Polygon layer `SIM3292_Global_Geology`, unit label in field `Unit`.
- **Citation:** Tanaka, K. L., Skinner, J. A., Jr., Dohm, J. M., Irwin, R. P., III, Kolb, E. J., Fortezzo, C. M.,
  Platz, T., Michael, G. G., & Hare, T. M. (2014). *Geologic map of Mars.* U.S. Geological Survey
  Scientific Investigations Map 3292, scale 1:20,000,000. Public domain.

## Landing sites

`data/landing_sites.csv` — planetocentric, east-positive. Primary sources:

- PDS Orbital Data Explorer landing-site list (Viking 1, Pathfinder: Folkner et al. 1997; Spirit, Opportunity:
  Arvidson et al. 2004): https://ode.rsl.wustl.edu/mars/pagehelp/Content/Missions_Instruments/Landing_Sites.htm
- NASA GISS Mars24 lander list (Curiosity, InSight, Perseverance, Zhurong):
  https://www.giss.nasa.gov/tools/mars24/help/landers.html
- Lakdawalla / The Planetary Society landing-site map: https://www.planetary.org/space-images/mars_landing_site_map_lakdawalla

Known discrepancy: Mars24 lists Pathfinder at 19.47° N, 326.75° E, while Folkner et al. (1997, via PDS ODE) give
19.095° N, 326.747° E and NASA's mission page 19.13° N, 33.22° W (= 326.78° E). This project uses 19.095° N;
the ~0.4° (≈ 22 km) difference is larger than the 10 km sampling radius, so the choice is noted in the README.

## Coordinate reference systems

- Geographic: `IAU_2015:49900` — Mars (2015) Sphere / Ocentric
- Projected (all analysis rasters): `IAU_2015:49910` — Mars (2015) Sphere / Ocentric / Equirectangular, clon = 0
  (`+proj=eqc +lat_ts=0 +lat_0=0 +lon_0=0 +R=3396190`). Requires PROJ ≥ 8.2.
