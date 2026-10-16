"""Tests on synthetic data: every step has a case whose answer is known."""
import numpy as np
import pandas as pd
import pytest

from marsgis import ahp, config as C, criteria, grid as G, overlay, sensitivity, validation


@pytest.fixture(scope="module")
def g():
    return G.reference_grid(res=20_000)            # coarse: ~1067 x 177 cells


# --- grid ---------------------------------------------------------------------------
def test_lon_conversions():
    assert G.lon360_to_180(354.47) == pytest.approx(-5.53)
    assert G.lon360_to_180(180) == pytest.approx(-180)
    assert G.lon180_to_360(-5.53) == pytest.approx(354.47)
    assert G.west_to_east(5.53) == pytest.approx(354.47)


def test_planetographic_smaller_than_planetocentric_input():
    pc = G.planetographic_to_planetocentric(45.0)
    assert 44.6 < pc < 45.0


def test_reference_grid_inside_band(g):
    full = G.reference_grid()
    assert full.width == 21338 and full.height == 3556
    assert abs(full.row_lat()).max() < 30
    assert abs(full.col_lon()).max() < 180


def test_rowcol_roundtrip(g):
    r, c = g.lonlat_to_rowcol(137.44, -4.59)
    lon, lat = g.rowcol_to_lonlat(r, c)
    assert abs(lon - 137.44) < 0.2 and abs(lat + 4.59) < 0.2
    r2, c2 = g.lonlat_to_rowcol(137.44 - 360, -4.59)
    assert (r, c) == (r2, c2)


def test_raster_roundtrip(tmp_path, g):
    a = np.random.default_rng(0).random(g.shape).astype("float32")
    a[0, 0] = np.nan
    G.write_raster(tmp_path / "a.tif", a, grid=g)
    b, prof = G.read_raster(tmp_path / "a.tif")
    assert np.isnan(b[0, 0]) and np.allclose(a[1:], b[1:])
    G.write_raster(tmp_path / "b.tif", b, profile=prof)
    G.check_aligned([tmp_path / "a.tif", tmp_path / "b.tif"])


def test_gdalwarp_cmd(g):
    cmd = G.gdalwarp_cmd("in.tif", "out.tif", g, "average")
    assert "IAU_2015:49910" in cmd and "-tr 20000 20000" in cmd and "-r average" in cmd


# --- criteria ---------------------------------------------------------------------------
def test_slope_north_south_plane(g):
    rows = np.arange(g.height)[:, None] * g.res
    dem = np.broadcast_to(rows * np.tan(np.radians(3.0)), g.shape).astype("float32")
    s = criteria.slope_deg(dem, g)
    assert np.allclose(s, 3.0, atol=1e-3)


def test_slope_east_west_corrected_for_latitude(g):
    # True ground gradient of 2 deg everywhere east-west: dz per pixel shrinks with cos(lat).
    cos = np.cos(np.radians(g.row_lat()))[:, None]
    cols = np.arange(g.width)[None, :]
    dem = (cols * g.res * cos * np.tan(np.radians(2.0))).astype("float64")
    s = criteria.slope_deg(dem, g)
    # Near column 0 the north-south gradient of this surface vanishes, so slope is pure east-west.
    assert np.allclose(s[:, 1:4], 2.0, atol=1e-3)
    assert s[0, 1] > np.degrees(np.arctan(np.tan(np.radians(2.0)) * np.cos(np.radians(29))))


def test_tri_flat_and_step():
    assert np.all(criteria.tri(np.zeros((5, 5))) == 0)
    dem = np.zeros((5, 5)); dem[2, 2] = 8
    assert criteria.tri(dem)[2, 2] == pytest.approx(8)
    assert criteria.tri(dem)[1, 1] == pytest.approx(1)


def test_block_std_detrended_plane_is_smooth():
    yy, xx = np.mgrid[0:50, 0:50]
    plane = 3.0 * xx + 2.0 * yy
    assert np.allclose(criteria.block_std(plane, 5, detrend=True), 0, atol=1e-9)
    assert np.all(criteria.block_std(plane, 5, detrend=False) > 0)
    noisy = plane + np.random.default_rng(1).normal(0, 2, plane.shape)
    assert np.nanmean(criteria.block_std(noisy, 5)) == pytest.approx(2 * np.sqrt(22 / 25), rel=0.15)


def test_neighbourhood_count_true_distance(g):
    # One point at 25 N: cells within 100 km should count it, beyond should not.
    c = criteria.neighbourhood_count([10.0], [25.0], g, radius_km=100)
    assert c.sum() > 0
    r0, c0 = g.lonlat_to_rowcol(10.0, 25.0)
    assert c[r0, c0] == 1
    lon, lat = g.rowcol_to_lonlat(*np.nonzero(c))
    d_km = np.hypot((lon - 10) * np.cos(np.radians(25)), lat - 25) * C.M_PER_DEG / 1000
    assert d_km.max() <= 100 + g.res / 1000      # within radius (+ one pixel of discretisation)


def test_neighbourhood_count_wraps_and_edge(g):
    c = criteria.neighbourhood_count([179.9], [0.0], g, radius_km=100)
    assert c[:, 0].sum() > 0 and c[:, -1].sum() > 0      # wraps across 180
    c = criteria.neighbourhood_count([0.0], [30.5], g, radius_km=100)
    assert c[0].sum() > 0                               # crater just outside band still counts


def test_crater_density_uniform_field(g):
    # 1 crater per 400 km^2 (one per 20 km cell at the equator) -> 25 per 10,000 km^2 near 0 deg.
    lon, lat = g.rowcol_to_lonlat(*np.indices(g.shape).reshape(2, -1))
    keep = np.abs(lat) < 8      # wider than the 200 km kernel
    d = criteria.crater_density(lon[keep], lat[keep], g, radius_km=200)
    r, _ = g.lonlat_to_rowcol(0, 0)
    assert np.median(d[r]) == pytest.approx(25, rel=0.1)


def test_geologic_unit_score():
    assert criteria.geologic_unit_score("eNh") == 1.0
    assert criteria.geologic_unit_score("lHl") == 1.0
    assert criteria.geologic_unit_score("HNt") == 1.0
    assert criteria.geologic_unit_score("Av") == 0.5
    assert criteria.geologic_unit_score("lAv") == 0.5
    assert criteria.geologic_unit_score("AHv") == 0.75
    assert np.isnan(criteria.geologic_unit_score("xyz"))


# --- normalization / AHP / overlay ------------------------------------------------------
def test_linear_score_both_directions():
    assert np.allclose(overlay.linear_score([-4000, -1500, 1000, 2000], -4000, 1000), [1, .5, 0, 0])
    assert np.allclose(overlay.linear_score([50, 200, 400], 300, 100), [0, .5, 1])
    assert np.isnan(overlay.linear_score([np.nan], 0, 1)[0])


def test_ahp_reproduces_guide():
    A = ahp.matrix_from_upper(C.CRITERIA, C.AHP_UPPER)
    r = ahp.ahp(A, C.CRITERIA)
    expected = dict(slope=0.274, rough=0.274, elev=0.154, ti=0.154, crater=0.089, lat=0.056)
    for k, v in expected.items():
        assert r.weights[k] == pytest.approx(v, abs=5e-4)
    assert r.lambda_max == pytest.approx(6.041, abs=1e-3)
    assert r.cr == pytest.approx(0.0067, abs=1e-4) and r.consistent


def test_ahp_rejects_non_reciprocal():
    with pytest.raises(ValueError):
        ahp.ahp(np.array([[1, 2], [2, 1]]), ["a", "b"])


def _synthetic(g, seed=0):
    rng = np.random.default_rng(seed)
    shape = g.shape
    base = rng.random(shape).astype("float32")
    layers = {k: rng.random(shape).astype("float32") for k in C.CRITERIA}
    scores = {k: np.clip(0.5 * base + 0.5 * v, 0, 1) for k, v in layers.items()}
    mask = rng.random(shape) > 0.1
    return scores, mask


def test_overlay_and_mask(g):
    scores, mask = _synthetic(g)
    w = ahp.ahp(ahp.matrix_from_upper(C.CRITERIA, C.AHP_UPPER), C.CRITERIA).weights
    S = overlay.weighted_overlay(scores, w, mask)
    assert np.all(S[~mask] == 0) and np.all((S >= 0) & (S <= 1))
    with pytest.raises(ValueError):
        overlay.weighted_overlay(scores, {k: 0.5 for k in w})


def test_constraint_mask_nan_excluded():
    m = overlay.constraint_mask(np.array([0, 2000, np.nan, 0]), np.array([1, 1, 1, 12]),
                                np.array([200, 200, 200, 200]))
    assert m.tolist() == [True, False, False, False]


def test_normalize_percentiles_ignore_invalid():
    x = np.array([0, 1, 2, 3, 1e6], dtype="float32")
    valid = np.array([True, True, True, True, False])
    s, used = overlay.normalize({"rough": x}, {"rough": (None, None)}, valid)
    assert used["rough"][0] == 0 and used["rough"][1] < 4


def test_classify():
    S = np.array([np.nan, 0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0])
    cls, br = overlay.classify(S)
    assert cls[0] == -1 and cls[1] == 0 and cls[2] == 1 and cls[-1] == 5


def test_candidate_zones_ranking_area_and_wrap(g):
    S = np.zeros(g.shape, dtype="float32")
    r, c = g.lonlat_to_rowcol(0, 0)
    S[r - 2:r + 3, c - 2:c + 3] = 0.9                    # big block, high score
    S[r - 20:r - 18, c:c + 2] = 0.95                     # small block (4 cells)
    S[r + 10:r + 14, :2] = 0.85; S[r + 10:r + 14, -2:] = 0.85   # zone across 180 deg
    tab, ranked = overlay.candidate_zones(S, g, threshold=0.8, min_km2=5 * 400)
    assert len(tab) == 2                                  # small block dropped
    assert tab.loc[0, "mean_score"] == pytest.approx(0.9)
    assert abs(tab.loc[1, "lon"]) > 179                  # seam zone centred on 180
    assert tab.loc[0, "area_km2"] == pytest.approx(25 * 400, rel=0.01)
    gdf = overlay.zones_to_geodataframe(ranked, tab, g)
    assert len(gdf) == 2 and gdf.crs is not None


# --- validation -----------------------------------------------------------------------
def test_validation_detects_planted_sites(g):
    S = np.full(g.shape, 0.2, dtype="float32")
    S += np.random.default_rng(3).normal(0, 0.05, g.shape).astype("float32")
    sites = pd.DataFrame({"site": list("abcd"), "lat": [0, 10, -10, 20], "lon_e": [10, 100, 200, 300]})
    sites["lon_e180"] = G.lon360_to_180(sites.lon_e)
    for lo, la in zip(sites.lon_e180, sites.lat):
        r, c = g.lonlat_to_rowcol(lo, la)
        S[r - 2:r + 3, c - 2:c + 3] = 0.9
    passing = np.random.default_rng(4).random(g.shape) > 0.5
    res = validation.validate(S, g, sites, np.isfinite(S), radius_km=10, n_random=300, passing=passing)
    assert np.all(res.sites.percentile > 99) and res.p_value < 0.01
    assert "4 of 4" in res.sentence()
    assert np.all(res.sites.percentile_passing > 99) and "4 of 4" in res.strict_sentence()


def test_random_points_area_weighted(g):
    lon, lat = validation.random_points(np.ones(g.shape, bool), g, n=20000)
    # area-uniform over +/-30 deg: E|lat| = 14.65 deg (15 deg if rows were unweighted)
    assert np.mean(np.abs(lat)) == pytest.approx(14.65, abs=0.25)


# --- sensitivity ---------------------------------------------------------------------------
def test_perturb_keeps_sum_and_ratios():
    w = dict(a=0.5, b=0.3, c=0.2)
    p = sensitivity.perturb(w, "a", 0.2)
    assert sum(p.values()) == pytest.approx(1) and p["a"] == pytest.approx(0.6)
    assert p["b"] / p["c"] == pytest.approx(1.5)


def test_scenarios():
    w = dict(slope=.274, rough=.274, elev=.154, ti=.154, crater=.089, lat=.055)
    s = sensitivity.named_scenarios(w, science=True)
    for v in s.values():
        assert sum(v.values()) == pytest.approx(1)
    assert s["Safety first"]["slope"] + s["Safety first"]["rough"] == pytest.approx(0.7)
    assert s["Science first"]["science"] == pytest.approx(0.4)


def test_oat_and_monte_carlo(g):
    scores, mask = _synthetic(g)
    w = ahp.ahp(ahp.matrix_from_upper(C.CRITERIA, C.AHP_UPPER), C.CRITERIA).weights
    S = overlay.weighted_overlay(scores, w, mask)
    th = sensitivity.quantile_threshold(S)
    _, base = overlay.candidate_zones(S, g, th, min_km2=400)
    df = sensitivity.one_at_a_time(scores, w, mask, g, base, changes=(-0.2, 0.2))
    assert set(df.criterion) == set(w) and df.top_overlap.between(0, 10).all()
    prob, draws = sensitivity.monte_carlo(scores, w, mask, n_runs=50)
    assert np.allclose(draws.sum(1), 1) and np.nanmax(prob) <= 1 and np.all(prob[~mask] == 0)
