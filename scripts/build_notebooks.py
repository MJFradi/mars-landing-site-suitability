"""Generate notebooks/01-04 from the cell lists below (keeps them diff-friendly).

    python scripts/build_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

NB = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = """\
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from marsgis import config as C, grid as G, criteria, overlay, ahp, validation, sensitivity, data, plots
grid = G.reference_grid()
C.FIGURES.mkdir(parents=True, exist_ok=True)
print(grid.shape, grid.crs)"""


def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)


notebooks = {
# ---------------------------------------------------------------------------------------------
"01_preprocessing.ipynb": [
md("""# 01 · Data and reference grid

Downloads the three core datasets, puts them on one Mars equirectangular grid
(IAU_2015:49910, ±30°, 1 km) and checks that everything lines up.

Conventions used everywhere: **planetocentric latitude, east-positive longitude (−180…180)**."""),
code(SETUP),
md("## 1. Check that PROJ knows Mars"),
code("""from pyproj import CRS
for code_ in (C.GEOG_CRS, C.PROJ_CRS):
    crs = CRS.from_user_input(code_)
    print(code_, '→', crs.name)
print(CRS.from_user_input(C.PROJ_CRS).to_proj4())"""),
md("""## 2. Download and warp

Equivalent to `python scripts/download_data.py`. Set `QUICK = True` for a first pass
(DEM from its 4× overview, ~800 m, a few minutes); set it to `False` for the version you
publish (full 200 m read and averaged to 1 km; streams ~4 GB, or pass a local copy)."""),
code("""QUICK = False   # True = DEM from its 4x overview (~800 m) for a fast first pass
DEM_FILE = None   # e.g. C.RAW / 'Mars_HRSC_MOLA_BlendDEM_Global_200mp_v2.tif'

craters_csv = data.download_craters(C.RAW / 'robbins_craters_lat31.csv')
ti_geo = data.thermal_inertia_geotiff(C.RAW / 'ti_night_2007_geo.tif')
data.warp_to_grid(ti_geo, C.PROCESSED / 'ti_1km.tif', grid, resampling='bilinear')
if not (C.PROCESSED / 'dem_1km.tif').exists():
    data.warp_to_grid(data.dem_source(DEM_FILE), C.PROCESSED / 'dem_1km.tif', grid,
                      resampling='average', src_nodata=-32768, overview_level=1 if QUICK else None)"""),
md("## 3. Inspect each layer: CRS, range, nodata, units"),
code("""import rasterio
for name in ('dem_1km.tif', 'ti_1km.tif'):
    with rasterio.open(C.PROCESSED / name) as s:
        a = s.read(1, masked=True)
        print(f"{name}: {s.width}x{s.height}, {s.crs.to_string()[:40]}, nodata={s.nodata}, "
              f"min={a.min():.1f} median={np.ma.median(a):.1f} max={a.max():.1f}, nodata share={a.mask.mean():.4f}")
cr = pd.read_csv(craters_csv)
print(f"craters: {len(cr):,}, lon {cr.lon_e.min():.2f}…{cr.lon_e.max():.2f} (−180…180 ✓), "
      f"lat {cr.lat.min():.2f}…{cr.lat.max():.2f}, D ≥ {cr.diamkm.min()} km")"""),
md("## 4. Alignment checks"),
code("""G.check_aligned([C.PROCESSED / 'dem_1km.tif', C.PROCESSED / 'ti_1km.tif'])
print('rasters share shape, transform and CRS ✓')

# Visual check: big craters should sit on DEM depressions (here around Gale).
dem, _ = G.read_raster(C.PROCESSED / 'dem_1km.tif')
r0, c0 = grid.lonlat_to_rowcol(137.4, -5.4)
win = dem[r0-300:r0+300, c0-300:c0+300]
lon0, lat0 = grid.rowcol_to_lonlat(r0-300, c0-300); lon1, lat1 = grid.rowcol_to_lonlat(r0+300, c0+300)
sub = cr[(cr.lon_e.between(lon0, lon1)) & (cr.lat.between(lat1, lat0)) & (cr.diamkm > 20)]
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(plots.hillshade(win, grid.res), cmap='gray', extent=[lon0, lon1, lat1, lat0])
ax.scatter(sub.lon_e, sub.lat, s=sub.diamkm, facecolors='none', edgecolors='r', lw=0.8)
ax.set_title('Craters > 20 km on the DEM near Gale (circles should sit on rims)', fontsize=9);"""),
md("""**Pitfalls checked here:** longitude convention (all three sources are −180…180 E, so no
shift is needed; `G.lon360_to_180` is there if you add a 0–360 dataset), DEM nodata
−32768 converted to NaN before any statistics, TES null (0) converted to NaN."""),
],

# ---------------------------------------------------------------------------------------------
"02_criteria.ipynb": [
md("""# 02 · Criteria layers

Six criteria on the 1 km grid. Slope, crater kernels and areas account for the
east–west pixel shrinking with cos(latitude) in equirectangular (−13 % at 30°)."""),
code(SETUP),
code("""dem, profile = G.read_raster(C.PROCESSED / 'dem_1km.tif')
ti, _ = G.read_raster(C.PROCESSED / 'ti_1km.tif')
cr = pd.read_csv(C.RAW / 'robbins_craters_lat31.csv')"""),
md("## Slope (degrees, latitude-corrected)"),
code("""slope = criteria.slope_deg(dem, grid)
G.write_raster(C.PROCESSED / 'slope_1km.tif', slope, grid=grid)
print('slope percentiles 50/95/99:', np.nanpercentile(slope, [50, 95, 99]).round(2))"""),
md("""## Roughness

TRI on the 1 km DEM (fast, but mostly measures slope at this scale). For the zoom-in
regions, `criteria.block_std` on the 200 m DEM gives a better roughness — see notebook 04."""),
code("""rough = criteria.tri(dem)
G.write_raster(C.PROCESSED / 'rough_1km.tif', rough, grid=grid)
print('TRI percentiles 50/95/99 (m):', np.nanpercentile(rough, [50, 95, 99]).round(1))"""),
md("## Crater density (craters ≥ 1 km per 10,000 km², within 50 km)"),
code("""crater = criteria.crater_density(cr.lon_e.values, cr.lat.values, grid, C.CRATER_RADIUS_KM)
G.write_raster(C.PROCESSED / 'crater_1km.tif', crater, grid=grid)
print('crater density percentiles 50/95:', np.percentile(crater, [50, 95]).round(1))"""),
md("## Latitude"),
code("""lat = criteria.abs_latitude(grid)
G.write_raster(C.PROCESSED / 'lat_1km.tif', lat, grid=grid)"""),
md("""## Science value (optional)

Needs `data.download_geology()` (810 MB). Noachian/Hesperian units → 1, Amazonian → 0.5,
mixed Amazonian–Hesperian → 0.75. Skipped automatically if the map isn't downloaded."""),
code("""geo_dir = C.RAW / 'sim3292'
shp = next(geo_dir.rglob('SIM3292_Global_Geology.shp'), None) if geo_dir.exists() else None
if shp:
    import geopandas as gpd
    units = gpd.read_file(shp).to_crs(grid.crs)
    science = criteria.rasterize_polygons(units, units['Unit'].map(criteria.geologic_unit_score), grid)
    G.write_raster(C.PROCESSED / 'science_1km.tif', science, grid=grid)
    print(units['Unit'].map(criteria.geologic_unit_score).value_counts(dropna=False))
else:
    print('geologic map not downloaded; science criterion skipped')"""),
md("## Quick look"),
code("""fig, axes = plt.subplots(4, 1, figsize=(14, 9), constrained_layout=True)
for ax, (name, a, cmap) in zip(axes, [('Elevation (m)', dem, 'terrain'), ('Slope (°)', slope, 'magma'),
                                      ('Thermal inertia (tiu)', ti, 'cividis'), ('Craters / 10⁴ km²', crater, 'viridis')]):
    im = ax.imshow(a[::8, ::8], cmap=cmap, vmax=np.nanpercentile(a, 99), extent=[-180, 180, -30, 30])
    ax.set_title(name, fontsize=9, loc='left'); fig.colorbar(im, ax=ax, fraction=0.02)"""),
],

# ---------------------------------------------------------------------------------------------
"03_ahp_overlay.ipynb": [
md("""# 03 · AHP weights, weighted overlay and candidate zones"""),
code(SETUP),
md("""## AHP

Judgements (Saaty 1–9, row vs column) live in `config.AHP_UPPER`. Justify each one in the
README: slope and roughness dominate because a tip-over or a rock strike ends the mission;
elevation and thermal inertia matter for entry and for bearing strength; crater density is
partly captured already by slope/roughness; latitude is a soft preference inside ±30°."""),
code("""res = ahp.ahp(ahp.matrix_from_upper(C.CRITERIA, C.AHP_UPPER), C.CRITERIA)
print(res)
res.table().round(3)"""),
md("## Load criteria, normalize, constrain"),
code("""L = {k: G.read_raster(C.PROCESSED / f'{f}_1km.tif')[0] for k, f in
     [('elev', 'dem'), ('slope', 'slope'), ('rough', 'rough'), ('ti', 'ti'), ('crater', 'crater'), ('lat', 'lat')]}
valid = overlay.valid_data(*L.values())
mask = overlay.constraint_mask(L['elev'], L['slope'], L['ti'])
print(f"valid cells: {valid.mean():.1%}; pass hard constraints: {mask[valid].mean():.1%} of valid")
excl = {'elev > +1 km': (L['elev'] > C.MAX_ELEV_M) & valid, 'slope > 10°': (L['slope'] > C.MAX_SLOPE_DEG) & valid,
        'TI < 100 (dust)': (L['ti'] < C.MIN_TI) & valid}
for k, m in excl.items():
    print(f"  excluded by {k}: {m.sum() / valid.sum():.1%}")

# Percentile-based ranges are computed on ground that passes the constraints.
scores, used = overlay.normalize(L, valid=mask)
pd.DataFrame(used, index=['score = 1 at', 'score = 0 at']).T.round(2)"""),
md("## Weighted overlay"),
code("""S = overlay.weighted_overlay(scores, res.weights, mask)
S = np.where(valid, S, np.nan).astype('float32')
G.write_raster(C.OUTPUTS / 'suitability.tif', S, grid=grid)
cls, breaks = overlay.classify(S, 5, 'quantile')
print('class breaks:', breaks.round(3))"""),
md("## Candidate zones (very-high class, ≥ 50 km², top 10)"),
code("""top, ranked = overlay.candidate_zones(S, grid, threshold=breaks[-2])
gdf = overlay.zones_to_geodataframe(ranked, top, grid)
gdf.to_file(C.OUTPUTS / 'top10_sites.gpkg', layer='top10', driver='GPKG')
top.assign(lon_e360=lambda d: d.lon % 360).to_csv(C.OUTPUTS / 'top10_sites.csv', index=False)
np.save(C.PROCESSED / 'ranked_zones.npy', ranked)
top.round(3)"""),
md("## Figures 1 and 2"),
code("""sites = validation.load_sites()
ax = plots.hero_map(S, L['elev'], grid, sites=sites, zones=ranked)
ax.figure.savefig(C.FIGURES / 'hero_map.png', dpi=300, bbox_inches='tight')
ax.figure.savefig(C.FIGURES / 'hero_map.pdf', bbox_inches='tight')
fig = plots.criteria_panel(scores, grid)
fig.savefig(C.FIGURES / 'criteria_panel.png', dpi=300, bbox_inches='tight')"""),
],

# ---------------------------------------------------------------------------------------------
"04_validation_sensitivity.ipynb": [
md("""# 04 · Validation and sensitivity

Are the 8 real landing sites inside ±30° scored clearly above random ground?
And how much does the answer depend on the weights?"""),
code(SETUP),
code("""import json
S, _ = G.read_raster(C.OUTPUTS / 'suitability.tif')
L = {k: G.read_raster(C.PROCESSED / f'{f}_1km.tif')[0] for k, f in
     [('elev', 'dem'), ('slope', 'slope'), ('rough', 'rough'), ('ti', 'ti'), ('crater', 'crater'), ('lat', 'lat')]}
valid = np.isfinite(S)
mask = overlay.constraint_mask(L['elev'], L['slope'], L['ti'])
scores, _ = overlay.normalize(L, valid=mask)
weights = ahp.ahp(ahp.matrix_from_upper(C.CRITERIA, C.AHP_UPPER), C.CRITERIA).weights
base_ranked = np.load(C.PROCESSED / 'ranked_zones.npy')
sites = validation.load_sites()"""),
md("""## Validation

Each site's score is the mean within 10 km. Percentiles are against 1,000 random points
drawn uniformly **by area** over all valid ground (including excluded cells, which score 0),
and measured the same way. One-sided Mann–Whitney U: are sites higher than random points?"""),
code("""val = validation.validate(S, grid, sites, valid, passing=mask)
print(val.sentence())
print(val.strict_sentence())
val.sites[['site', 'year', 'chosen_for', 'lat', 'lon_e', 'score', 'point_score', 'percentile',
           'percentile_passing']].round(3)"""),
md("""`percentile` compares each site with random spots anywhere in the band (61 % of which
are excluded and score 0). `percentile_passing` is the stricter test: random spots drawn
only from ground that already passes the hard constraints."""),
code("""ax = plots.validation_chart(val)
ax.figure.savefig(C.FIGURES / 'validation.png', dpi=300, bbox_inches='tight')"""),
md("""**Reading misses.** Curiosity (Gale) and Perseverance (Jezero) were chosen for science and
relied on guided entry / Terrain-Relative Navigation to land near hazards. A lower score
there shows the model encodes engineering safety only — that trade-off is a finding."""),
md("## One-at-a-time weight sensitivity (±20 %, ±40 %)"),
code("""def mean_site_pct(S_):
    return float(validation.validate(S_, grid, sites, valid, n_random=500).sites.percentile.mean())

oat = sensitivity.one_at_a_time(scores, weights, mask, grid, base_ranked, validate_fn=mean_site_pct)
oat.to_csv(C.OUTPUTS / 'sensitivity_oat.csv', index=False)
ax = plots.sensitivity_plot(oat)
ax.figure.savefig(C.FIGURES / 'sensitivity.png', dpi=300, bbox_inches='tight')
oat.pivot(index='change', columns='criterion', values='top_overlap')"""),
md("## Named scenarios"),
code("""have_science = (C.PROCESSED / 'science_1km.tif').exists()
sc_scores = dict(scores)
if have_science:
    sc_scores['science'] = np.nan_to_num(G.read_raster(C.PROCESSED / 'science_1km.tif')[0], nan=0.5)
rows = []
for name, w in sensitivity.named_scenarios(weights, science=have_science).items():
    S_w = overlay.weighted_overlay({k: sc_scores[k] for k in w}, w, mask)
    S_w = np.where(valid, S_w, np.nan)
    _, rk = overlay.candidate_zones(S_w, grid, sensitivity.quantile_threshold(S_w))
    v = validation.validate(S_w, grid, sites, valid)
    rows.append({'scenario': name, 'top10_overlap': sensitivity.top_overlap(base_ranked, rk),
                 'mean_site_percentile': v.sites.percentile.mean(), 'summary': v.sentence()})
scen = pd.DataFrame(rows); scen.to_csv(C.OUTPUTS / 'scenarios.csv', index=False); scen"""),
md("## Monte Carlo: probability of being in the top 10 % (500 Dirichlet weight draws, ~7 min)"),
code("""prob, draws = sensitivity.monte_carlo(scores, weights, mask, n_runs=500)
G.write_raster(C.OUTPUTS / 'prob_top10pct.tif', prob, grid=grid)
ax = plots.prob_map(prob, grid, sites=sites)
ax.figure.savefig(C.FIGURES / 'prob_top10pct.png', dpi=300, bbox_inches='tight')"""),
md("## Figure 5: zoom-in on the best zone at 200 m"),
code("""top = pd.read_csv(C.OUTPUTS / 'top10_sites.csv')
best = top.iloc[0]
fine = data.fine_dem_window(best.lon, best.lat, half_km=40, dst=C.PROCESSED / 'dem200_best.tif')
import rasterio
with rasterio.open(fine) as s_:
    dem_f = s_.read(1).astype('float64'); t_f = s_.transform
dem_f[dem_f == -9999] = np.nan
# Zone #1 on the 200 m grid (same CRS, so map fine cell centres to 1 km rows/cols)
xs = t_f.c + (np.arange(dem_f.shape[1]) + 0.5) * t_f.a
ys = t_f.f + (np.arange(dem_f.shape[0]) + 0.5) * t_f.e
zone_f = base_ranked[np.ix_(np.floor((grid.ymax - ys) / grid.res).astype(int),
                            np.floor((xs - grid.xmin) / grid.res).astype(int))] == 1
# 200 m slope (window is small, so cos(lat) is treated as constant)
dzdy, dzdx = np.gradient(dem_f, C.FINE_PIXEL_M)
slope_f = np.degrees(np.arctan(np.hypot(dzdx / np.cos(np.radians(best.lat)), dzdy)))
for name, m in [('80 km window', np.isfinite(slope_f)), ('inside zone #1', zone_f)]:
    v = slope_f[m]
    print(f"{name:>15}: 200 m slope median {np.nanmedian(v):.2f}°, 95th pct {np.nanpercentile(v, 95):.2f}°, "
          f"> 5°: {np.mean(v > 5):.1%}, > 10°: {np.mean(v > 10):.1%}")
ax = plots.zoom_in(dem_f, slope_fine=slope_f, title=f'Zone #1 · {best.lat:.2f}°, {best.lon % 360:.2f}°E (200 m)')
ax.contour(zone_f.astype(float), levels=[0.5], colors='#ff3b30', linewidths=1.0)
ax.figure.savefig(C.FIGURES / 'zoom_best.png', dpi=300, bbox_inches='tight')"""),
md("""The 1 km model can't see channel walls or streamlined islands next to a zone, so the
200 m check matters: report the in-zone numbers, not the whole window."""),
md("## Numbers for the README and LinkedIn post"),
code("""k = int(round(val.top_share * len(val.sites)))
results = {'sites_in_top20': k, 'n_sites': len(val.sites), 'mann_whitney_p': val.p_value,
           'site_percentiles': dict(zip(val.sites.site, val.sites.percentile.round(1))),
           'strict': val.strict_sentence(),
           'site_percentiles_passing': dict(zip(val.sites.site, val.sites.percentile_passing.round(1))),
           'ahp_cr': ahp.ahp(ahp.matrix_from_upper(C.CRITERIA, C.AHP_UPPER), C.CRITERIA).cr,
           'top10': top.round(4).to_dict('records')}
json.dump(results, open(C.OUTPUTS / 'results.json', 'w'), indent=2)
print(json.dumps({k_: v for k_, v in results.items() if k_ != 'top10'}, indent=2))"""),
],
}


def main():
    NB.mkdir(exist_ok=True)
    for name, cells in notebooks.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3 (mars-gis)", "language": "python"}
        nbf.write(nb, NB / name)
        print("wrote", NB / name)


if __name__ == "__main__":
    main()
