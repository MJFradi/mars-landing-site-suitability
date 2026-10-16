"""The five figures: hero map, criteria panel, validation, sensitivity, zoom-in.

Matplotlib only, so figures are reproducible from the notebooks; use QGIS
Print Layout for a polished version of the hero map if you like.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import config as C
from .grid import Grid

CREDIT = "MOLA/HRSC DEM: NASA/ESA/USGS · Thermal inertia: Putzig & Mellon (TES) · Craters: Robbins & Hynek (2012)"
CAPTION_CRS = "Planetocentric latitude, east-positive longitude · Mars 2015 equirectangular (IAU_2015:49910)"


def hillshade(dem: np.ndarray, res: float, azimuth: float = 315, altitude: float = 45,
              z_factor: float = 1.0) -> np.ndarray:
    az, alt = np.radians(360 - azimuth + 90), np.radians(altitude)
    dy, dx = np.gradient(np.nan_to_num(dem) * z_factor, res)
    slope = np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dy, dx)
    hs = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(hs, 0, 1)


def _extent_deg(grid: Grid):
    return [grid.xmin / C.M_PER_DEG, grid.xmax / C.M_PER_DEG,
            grid.ymin / C.M_PER_DEG, grid.ymax / C.M_PER_DEG]


def _graticule(ax, step_lon=30, step_lat=10):
    ax.set_xticks(np.arange(-180, 181, step_lon))
    ax.set_yticks(np.arange(-30, 31, step_lat))
    ax.set_xticklabels([f"{int(v % 360)}°E" for v in np.arange(-180, 181, step_lon)], fontsize=7)
    ax.set_yticklabels([f"{v}°" for v in np.arange(-30, 31, step_lat)], fontsize=7)
    ax.grid(color="white", alpha=0.25, lw=0.4)


def _coarsen(a: np.ndarray, step: int) -> np.ndarray:
    return a[::step, ::step] if step > 1 else a


def hero_map(S, dem, grid: Grid, sites=None, zones=None, step: int = 4, ax=None,
             title="Landing suitability, ±30° band"):
    """Suitability over hillshade, sites as labelled stars, top zones outlined.

    ``sites``: DataFrame with site, lat, lon_e180. ``zones``: ranked zone raster.
    ``step`` subsamples for plotting speed (4 -> ~4 km display pixels).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 3.6), dpi=150)
    ext = _extent_deg(grid)
    ax.imshow(_coarsen(hillshade(dem, grid.res), step), cmap="gray", extent=ext, interpolation="nearest")
    Sm = np.ma.masked_where(~np.isfinite(S) | (S <= 0), S)
    im = ax.imshow(_coarsen(Sm, step), cmap="viridis", vmin=0, vmax=1, alpha=0.75, extent=ext,
                   interpolation="nearest")
    if zones is not None:
        z = _coarsen(zones, step)
        lon = np.linspace(ext[0], ext[1], z.shape[1])
        lat = np.linspace(ext[3], ext[2], z.shape[0])
        ax.contour(lon, lat, (z > 0).astype(float), levels=[0.5], colors="#ff3b30", linewidths=0.8)
    if sites is not None:
        ax.scatter(sites.lon_e180, sites.lat, marker="*", s=90, c="white", edgecolors="black", lw=0.6, zorder=5)
        for _, r in sites.iterrows():
            east_edge = r.lon_e180 > 150                     # keep labels inside the frame
            ax.annotate(r["site"], (r.lon_e180, r.lat), xytext=(-4 if east_edge else 4, 4),
                        textcoords="offset points", ha="right" if east_edge else "left",
                        fontsize=6.5, color="white", weight="bold")
    _graticule(ax)
    ax.set_title(title, fontsize=10, loc="left")
    cb = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01)
    cb.set_label("Suitability (0–1)", fontsize=7)
    ax.figure.text(0.01, 0.005, f"{CAPTION_CRS}\n{CREDIT}", fontsize=5.5, color="0.35", va="bottom")
    return ax


def criteria_panel(scores: dict, grid: Grid, step: int = 8, names=None):
    names = names or {"slope": "Slope", "rough": "Roughness", "elev": "Elevation",
                      "ti": "Thermal inertia", "crater": "Crater density", "lat": "Latitude"}
    keys = [k for k in names if k in scores]
    fig, axes = plt.subplots(3, 2, figsize=(14, 6.5), dpi=150, constrained_layout=True)
    ext = _extent_deg(grid)
    for ax, k in zip(axes.ravel(), keys):
        im = ax.imshow(_coarsen(scores[k], step), cmap="viridis", vmin=0, vmax=1, extent=ext)
        ax.set_title(f"{names[k]} score", fontsize=8, loc="left")
        _graticule(ax, 60, 15)
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.colorbar(im, ax=axes, fraction=0.02, label="Score (1 = best)")
    return fig


def validation_chart(result, ax=None):
    """Site percentiles against the distribution of random-point scores."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), dpi=150)
    rand = result.random_scores[np.isfinite(result.random_scores)]
    ax.hist(rand, bins=40, color="0.8", edgecolor="white", label=f"{rand.size} random points")
    ymax = ax.get_ylim()[1]
    sites = result.sites.sort_values("score")
    for i, (_, r) in enumerate(sites.iterrows()):
        y = ymax * (0.15 + 0.8 * i / max(1, len(sites) - 1))
        ax.plot([r.score], [y], "o", color="#d1495b", ms=6, zorder=5)
        ax.annotate(f"{r['site']} ({r.percentile:.0f}th)", (r.score, y), xytext=(6, -3),
                    textcoords="offset points", fontsize=7)
    ax.set_xlim(-0.03, 1.25)                                 # room for labels on the right
    ax.set_xlabel(f"Mean suitability within {C.SITE_RADIUS_KM:.0f} km")
    ax.set_ylabel("Random points")
    ax.set_title(result.sentence(), fontsize=8, loc="left")
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    return ax


def sensitivity_plot(df, ax=None, n_top: int = C.TOP_N):
    """Top-N overlap against weight change, one line per criterion."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4), dpi=150)
    for name, g in df.groupby("criterion"):
        ax.plot(g.change * 100, g.top_overlap, marker="o", ms=4, label=name)
    ax.set_xlabel("Change in criterion weight (%)")
    ax.set_ylabel(f"Baseline top-{n_top} zones still in top {n_top}")
    ax.set_ylim(0, n_top + 0.5)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.legend(fontsize=7, frameon=False, ncol=3)
    return ax


def zoom_in(dem_fine, S_fine=None, res: float = C.FINE_PIXEL_M, slope_fine=None, ax=None,
            title="Best candidate zone (200 m)"):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6), dpi=150)
    ax.imshow(hillshade(dem_fine, res), cmap="gray")
    if S_fine is not None:
        ax.imshow(np.ma.masked_invalid(S_fine), cmap="viridis", vmin=0, vmax=1, alpha=0.5)
    if slope_fine is not None:
        cs = ax.contour(slope_fine, levels=[2, 5, 10], colors=["#ffd166", "#ef476f", "#7b2cbf"], linewidths=0.7)
        ax.clabel(cs, fmt="%d°", fontsize=6)
    ax.set_title(title, fontsize=9, loc="left")
    ax.set_xticks([]); ax.set_yticks([])
    n = dem_fine.shape[1] * res / 1000
    bar = 10 if n > 30 else 5
    ax.plot([5, 5 + bar * 1000 / res], [dem_fine.shape[0] - 8] * 2, color="white", lw=3)
    ax.text(5, dem_fine.shape[0] - 12, f"{bar} km", color="white", fontsize=7)
    return ax


def prob_map(prob, grid: Grid, sites=None, step: int = 4, ax=None):
    """Monte Carlo probability of ranking in the top 10 %, with the same graticule as the hero map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(16, 3.6), dpi=150)
    im = ax.imshow(_coarsen(prob, step), cmap="magma", vmin=0, vmax=1, extent=_extent_deg(grid),
                   interpolation="nearest")
    if sites is not None:
        ax.scatter(sites.lon_e180, sites.lat, marker="*", c="cyan", s=70, edgecolors="black", lw=0.5, zorder=5)
    _graticule(ax)
    ax.set_title("Probability of ranking in the top 10 % under 500 random AHP-centred weightings",
                 fontsize=9, loc="left")
    plt.colorbar(im, ax=ax, fraction=0.015, pad=0.01).set_label("P(top 10 %)", fontsize=7)
    return ax


def region_map(S, dem, grid: Grid, lon_range, lat_range, sites=None, zones=None,
               figsize=(12, 6.28), title=None, subtitle=None):
    """Full-resolution crop of the hero map, sized for social media (default 1.91:1).

    ``lon_range`` in east-positive degrees, either convention (e.g. (285, 355)).
    Pick ranges with lon span / lat span ~ 2.2 so the fixed axes don't stretch the map.
    """
    from .grid import lon360_to_180
    lo0, lo1 = (float(v) for v in lon360_to_180(np.asarray(lon_range)))
    r0, c0 = grid.lonlat_to_rowcol(lo0, lat_range[1])
    r1, c1 = grid.lonlat_to_rowcol(lo1, lat_range[0])
    r0, r1 = max(int(r0), 0), min(int(r1), grid.height - 1)
    win = (slice(r0, r1 + 1), slice(int(c0), int(c1) + 1))
    lon_a, lat_a = grid.rowcol_to_lonlat(r0, c0)
    lon_b, lat_b = grid.rowcol_to_lonlat(r1, c1)
    ext = [lon_a, lon_b, lat_b, lat_a]

    fig = plt.figure(figsize=figsize, dpi=200)
    fig.patch.set_facecolor("#0e1117")
    ax = fig.add_axes([0.045, 0.1, 0.855, 0.75])
    cax = fig.add_axes([0.912, 0.1, 0.014, 0.75])
    ax.imshow(hillshade(dem[win], grid.res), cmap="gray", extent=ext, interpolation="bilinear", aspect="auto")
    Sm = np.ma.masked_where(~np.isfinite(S[win]) | (S[win] <= 0), S[win])
    im = ax.imshow(Sm, cmap="viridis", vmin=0, vmax=1, alpha=0.72, extent=ext, interpolation="nearest",
                   aspect="auto")
    if zones is not None:
        z = zones[win] > 0
        ax.contour(np.linspace(lon_a, lon_b, z.shape[1]), np.linspace(lat_a, lat_b, z.shape[0]),
                   z.astype(float), levels=[0.5], colors="#ff3b30", linewidths=1.2)
    if sites is not None:
        inside = sites[(sites.lon_e180.between(min(lon_a, lon_b), max(lon_a, lon_b)))
                       & (sites.lat.between(lat_b, lat_a))]
        ax.scatter(inside.lon_e180, inside.lat, marker="*", s=260, c="white", edgecolors="black", lw=0.8, zorder=5)
        for _, r in inside.iterrows():
            near_east = r.lon_e180 > max(lon_a, lon_b) - 0.15 * abs(lon_b - lon_a)
            ax.annotate(f"{r['site']} ({int(r['year'])})", (r.lon_e180, r.lat),
                        xytext=(-10 if near_east else 8, 6), textcoords="offset points",
                        ha="right" if near_east else "left", fontsize=10, color="white", weight="bold")
    ticks = np.arange(np.ceil(lon_a / 10) * 10, lon_b + 0.1, 10)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{int(v % 360)}°E" for v in ticks], fontsize=8, color="#c9d1d9")
    yt = np.arange(np.ceil(lat_b / 10) * 10, lat_a + 0.1, 10)
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{int(v)}°" for v in yt], fontsize=8, color="#c9d1d9")
    for s_ in ax.spines.values():
        s_.set_color("#30363d")
    if title:
        fig.text(0.02, 0.965, title, fontsize=17, weight="bold", color="white", va="top")
    if subtitle:
        fig.text(0.02, 0.905, subtitle, fontsize=10.5, color="#c9d1d9", va="top")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Landing suitability (0–1)", color="#c9d1d9", fontsize=8)
    cb.ax.tick_params(colors="#c9d1d9", labelsize=7)
    fig.text(0.02, 0.012, f"{CREDIT}\n{CAPTION_CRS}", fontsize=6.5, color="#8b949e", va="bottom")
    return fig
