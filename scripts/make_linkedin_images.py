"""LinkedIn assets: a step-by-step explainer carousel anyone can follow.

    python scripts/make_linkedin_images.py          (run after notebook 04)

Each slide has three layers, so it works for every reader:
  1. a title that states the point,
  2. "In plain words" - no jargon,
  3. a picture, and a "Technical detail" box for GIS / remote-sensing readers.

Writes to outputs/linkedin/:
  carousel.pdf      15 slides, 1080 x 1350 (upload as a LinkedIn "document" post)
  slide_01..15.png  the same slides as images
  post_image.png    1.91:1 map with a "how to read" legend, for a single-image post
"""
import json
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import ListedColormap
from matplotlib.patches import Ellipse, FancyArrowPatch, FancyBboxPatch

from marsgis import ahp, config as C, grid as G, overlay, plots, validation

# --- style ------------------------------------------------------------------------------
BG, PANEL, FG, MUTED, LINE = "#0e1117", "#161b22", "#e6edf3", "#8b949e", "#30363d"
ACCENT, GREEN, BLUE, RED, YELLOW = "#ff7b54", "#3fb950", "#58a6ff", "#ff5c5c", "#e3b341"
W_IN, H_IN, DPI = 5.4, 6.75, 200                   # 1080 x 1350 px
REPO = "github.com/MJFradi/mars-landing-site-suitability"
AUTHOR = "Mohamed Jihed Fradi · Geomatics Engineering student, MSE Manouba"
N_SLIDES = 15
OUT = C.OUTPUTS / "linkedin"
CHOSEN_COLOR = {"safety": GREEN, "science": ACCENT}
CHOSEN_LABEL = {"safety": "site chosen mainly for safety", "science": "site chosen for its science"}


def _lh(fs, spacing=1.35):
    """Height of one text line in figure fraction."""
    return fs * spacing / 72 / H_IN


def _wrap(text, fs, width_frac=0.86):
    chars = int(width_frac * W_IN * 72 / (0.55 * fs))
    return "\n".join(textwrap.fill(p, chars) if p else "" for p in text.split("\n"))


class Slide:
    """One 1080 x 1350 slide; text blocks stack downward from the top."""

    def __init__(self, n, tag, title):
        self.n = n
        self.fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI)
        self.fig.patch.set_facecolor(BG)
        self.y = 0.955
        self.text(tag.upper(), 7.5, ACCENT, weight="bold", gap=0.006)
        self.text(title, 16, FG, width=0.78, weight="bold", gap=0.018, spacing=1.2)

    def text(self, s, fs, color, x=0.07, width=0.86, weight="normal", gap=0.012, spacing=1.35):
        s = _wrap(s, fs, width)
        self.fig.text(x, self.y, s, fontsize=fs, color=color, weight=weight, va="top",
                      linespacing=spacing)
        self.y -= _lh(fs, spacing) * (s.count("\n") + 1) + gap
        return self

    def plain(self, s, fs=10):
        self.text("IN PLAIN WORDS", 6.5, MUTED, weight="bold", gap=0.004)
        return self.text(s, fs, FG, gap=0.02)

    def axes(self, height, left=0.07, width=0.86, gap=0.015):
        ax = self.fig.add_axes([left, self.y - height, width, height])
        self.y -= height + gap
        return ax

    def tech(self, s, fs=7.2, bottom=0.055):
        s = _wrap(s, fs, 0.8)
        n_lines = s.count("\n") + 2
        h = _lh(fs) * n_lines + 0.022
        self.fig.patches.append(FancyBboxPatch((0.06, bottom), 0.88, h, boxstyle="round,pad=0,rounding_size=0.012",
                                               transform=self.fig.transFigure, facecolor=PANEL,
                                               edgecolor=LINE, lw=0.8))
        self.fig.text(0.09, bottom + h - 0.012, "TECHNICAL DETAIL", fontsize=6.5, color=BLUE,
                      weight="bold", va="top")
        self.fig.text(0.09, bottom + h - 0.012 - _lh(6.5) - 0.004, s, fontsize=fs, color=MUTED,
                      va="top", linespacing=1.35)
        return bottom + h                                   # top of the box

    def finish(self):
        self.fig.text(0.07, 0.022, AUTHOR, fontsize=6, color=MUTED)
        self.fig.text(0.93, 0.022, f"{self.n}/{N_SLIDES}", fontsize=6.5, color=MUTED, ha="right")
        if self.n < N_SLIDES:
            self.fig.text(0.93, 0.037, "swipe →", fontsize=6.5, color=ACCENT, ha="right")
        return self.fig


def dark(ax, grid_lines=False):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color(LINE)
    ax.tick_params(colors=MUTED, labelsize=6.5)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if grid_lines:
        ax.grid(color=LINE, lw=0.5)
        ax.set_axisbelow(True)


# --- map helpers -------------------------------------------------------------------------------
def crop(grid, lon_range, lat_range):
    """Row/col slices and degree extent for a lon/lat box (east-positive, either convention)."""
    lo0, lo1 = (float(v) for v in G.lon360_to_180(np.asarray(lon_range)))
    r0, c0 = grid.lonlat_to_rowcol(lo0, lat_range[1])
    r1, c1 = grid.lonlat_to_rowcol(lo1, lat_range[0])
    r0, r1 = max(int(r0), 0), min(int(r1), grid.height - 1)
    lon_a, lat_a = grid.rowcol_to_lonlat(r0, c0)
    lon_b, lat_b = grid.rowcol_to_lonlat(r1, c1)
    return (slice(r0, r1 + 1), slice(int(c0), int(c1) + 1)), [lon_a, lon_b, lat_b, lat_a]


def lon_ticks(ax, ext, step=10, fs=6):
    t = np.arange(np.ceil(ext[0] / step) * step, ext[1] + 0.01, step)
    ax.set_xticks(t)
    ax.set_xticklabels([f"{int(v % 360)}°E" for v in t], fontsize=fs, color=MUTED)
    t = np.arange(np.ceil(ext[2] / step) * step, ext[3] + 0.01, step)
    ax.set_yticks(t)
    ax.set_yticklabels([f"{int(v)}°" for v in t], fontsize=fs, color=MUTED)


def suit_map(ax, S, dem, grid, win, ext, step=1, alpha=0.75):
    d = dem[win][::step, ::step]
    s = S[win][::step, ::step]
    ax.imshow(plots.hillshade(d, grid.res * step), cmap="gray", extent=ext, aspect="auto")
    ax.imshow(np.ma.masked_where(~np.isfinite(s) | (s <= 0), s), cmap="viridis", vmin=0, vmax=1,
              alpha=alpha, extent=ext, aspect="auto", interpolation="nearest")


def band_strips(sl, S, dem, grid, sites, height=0.17, layer=None, cmap=None, vmin=0, vmax=1, step=6):
    """The whole +/-30 deg band as two undistorted strips (west and east hemispheres)."""
    half = grid.width // 2
    for cols, lon0, label in [(slice(0, half), -180, "180°E → 360°E"), (slice(half, None), 0, "0°E → 180°E")]:
        ax = sl.axes(height, left=0.04, width=0.92, gap=0.008)
        ext = [lon0, lon0 + 180, -30, 30]
        d = dem[:, cols][::step, ::step]
        ax.imshow(plots.hillshade(d, grid.res * step), cmap="gray", extent=ext, aspect="auto")
        if layer is None:
            s = S[:, cols][::step, ::step]
            ax.imshow(np.ma.masked_where(~np.isfinite(s) | (s <= 0), s), cmap="viridis", vmin=0,
                      vmax=1, alpha=0.75, extent=ext, aspect="auto")
        else:
            l_ = layer[:, cols][::step, ::step]
            ax.imshow(np.ma.masked_invalid(l_), cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.85,
                      extent=ext, aspect="auto", interpolation="nearest")
        if sites is not None:
            sel = sites[(sites.lon_e180 >= lon0) & (sites.lon_e180 < lon0 + 180)]
            ax.scatter(sel.lon_e180, sel.lat, marker="*", s=55, c="white", edgecolors="black", lw=0.5, zorder=5)
            for _, r in sel.iterrows():
                east = r.lon_e180 > lon0 + 160
                ax.annotate(r["site"], (r.lon_e180, r.lat), xytext=(-4 if east else 4, 3),
                            textcoords="offset points", ha="right" if east else "left",
                            fontsize=5.2, color="white", weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(LINE)
        ax.text(0.006, 0.04, label, transform=ax.transAxes, fontsize=5.5, color="white")


def colorbar_row(sl, cmap, left_label, right_label, title=None, vmin=0, vmax=1):
    ax = sl.axes(0.012, left=0.2, width=0.6, gap=0.004)
    ax.imshow(np.linspace(vmin, vmax, 256)[None, :], cmap=cmap, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    y = sl.y
    sl.fig.text(0.2, y, left_label, fontsize=6.5, color=MUTED, va="top")
    sl.fig.text(0.8, y, right_label, fontsize=6.5, color=MUTED, va="top", ha="right")
    if title:
        sl.fig.text(0.5, y, title, fontsize=6.5, color=FG, va="top", ha="center")
    sl.y -= _lh(6.5) + 0.01


# --- data ----------------------------------------------------------------------------------------
def load():
    grid = G.reference_grid()
    rd = lambda name: G.read_raster(C.PROCESSED / name)[0]                      # noqa: E731
    L = {"elev": rd("dem_1km.tif"), "slope": rd("slope_1km.tif"), "rough": rd("rough_1km.tif"),
         "ti": rd("ti_1km.tif"), "crater": rd("crater_1km.tif"), "lat": rd("lat_1km.tif")}
    mask = overlay.constraint_mask(L["elev"], L["slope"], L["ti"])
    scores, used = overlay.normalize(L, valid=mask)
    return dict(
        grid=grid, L=L, mask=mask, scores=scores, used=used,
        S=G.read_raster(C.OUTPUTS / "suitability.tif")[0],
        ranked=np.load(C.PROCESSED / "ranked_zones.npy"),
        sites=validation.load_sites(),
        res=json.load(open(C.OUTPUTS / "results.json", encoding="utf-8")),
        top=pd.read_csv(C.OUTPUTS / "top10_sites.csv"),
        oat=pd.read_csv(C.OUTPUTS / "sensitivity_oat.csv"),
        scen=pd.read_csv(C.OUTPUTS / "scenarios.csv"),
        prob=G.read_raster(C.OUTPUTS / "prob_top10pct.tif")[0],
        ahp=ahp.ahp(ahp.matrix_from_upper(C.CRITERIA, C.AHP_UPPER), C.CRITERIA),
        craters=pd.read_csv(C.RAW / "robbins_craters_lat31.csv"),
    )


# --- slides ----------------------------------------------------------------------------------------
def s01_cover(d):
    r = d["res"]
    sl = Slide(1, "Planetary GIS project", "Can an Earth GIS method choose where to land on Mars?")
    sl.text("I took the same multi-criteria analysis we use on Earth to pick a site for a road, a farm "
            "or a building, and applied it to the whole equatorial belt of Mars using free NASA/ESA "
            "orbital data. Then I checked it against the 8 places where real landers actually touched down.",
            9.5, MUTED, gap=0.03)
    band_strips(sl, d["S"], d["L"]["elev"], d["grid"], d["sites"], height=0.16)
    colorbar_row(sl, "viridis", "riskier", "safer", "landing suitability (grey = ruled out)")
    sl.y -= 0.02
    sl.text(f"Result: all {r['sites_in_top20']} of {r['n_sites']} real landing sites scored "
            f"in the top 20 % of the map.", 15, ACCENT, weight="bold", gap=0.015, spacing=1.2)
    sl.text("15 slides · method, results, and the part that surprised me (slide 12).", 8, MUTED)
    return sl.finish()


def s02_problem(d):
    sl = Slide(2, "The problem", "Why choosing a landing site is hard")
    sl.plain("A Mars lander has about 7 minutes to go from 20,000 km/h to zero, on its own, with no "
             "pilot. It can't choose where it lands precisely: it comes down somewhere inside an "
             "oval several kilometres wide. So the whole oval has to be safe.")
    items = [("Steep slopes", "the lander can tip over, and its radar gets confused"),
             ("Rocks and rough ground", "a single big rock can break a leg or puncture an airbag"),
             ("High ground", "less air above it to slow the parachute down"),
             ("Deep dust", "soft ground that can't support the lander or a rover"),
             ("Craters", "rims, boulders and steep walls everywhere"),
             ("Far from the equator", "colder, and less sunlight for solar panels")]
    for name, why in items:
        sl.fig.text(0.07, sl.y, "▲", fontsize=9, color=RED, va="top")
        sl.fig.text(0.12, sl.y, name, fontsize=9.5, color=FG, weight="bold", va="top")
        sl.y -= _lh(9.5) + 0.002
        sl.text(why, 8.5, MUTED, x=0.12, width=0.8, gap=0.012)
    sl.y -= 0.01
    sl.text("NASA and other agencies spend years on this, with teams of scientists and engineers. "
            "My question: how close can a transparent, reproducible GIS model get, using only public data?",
            9.5, ACCENT, gap=0.01)
    sl.tech("Perseverance's landing ellipse was ~7.7 × 6.6 km (with Terrain-Relative Navigation); "
            "Viking's in 1976 was ~280 × 100 km. Entry, descent and landing (EDL) takes ~7 min. "
            "This project uses a minimum candidate-zone size of 50 km², about one modern landing ellipse.")
    return sl.finish()


def s03_method(d):
    sl = Slide(3, "The idea", "Multi-criteria analysis: stack maps, score, combine")
    sl.plain("On Earth, to find the best site for something, GIS analysts stack several maps "
             "(slope, soil, distance to roads...), give every location a score on each one, decide "
             "how much each map matters, and add the scores up. I did exactly that for Mars:")
    steps = [("1", "Collect the data", "3 free datasets measured from orbit"),
             ("2", "Put everything on one grid", "76 million cells of 1 km × 1 km"),
             ("3", "Build 6 criteria", "slope, roughness, height, dust, craters, latitude"),
             ("4", "Score 0 → 1 and rule out the impossible", "too high, too dusty or too steep = out"),
             ("5", "Weigh the criteria (AHP)", "structured pairwise comparisons"),
             ("6", "Combine, then test against real landers", "8 real sites vs 1,000 random spots")]
    ax = sl.axes(0.44)
    ax.set_xlim(0, 1); ax.set_ylim(0, len(steps)); ax.axis("off")
    for i, (num, head, sub) in enumerate(steps):
        y = len(steps) - i - 0.5
        ax.add_patch(FancyBboxPatch((0.02, y - 0.36), 0.96, 0.72, boxstyle="round,pad=0,rounding_size=0.06",
                                    facecolor=PANEL, edgecolor=LINE, lw=0.8))
        ax.text(0.09, y, num, ha="center", va="center", fontsize=9, color=BG, weight="bold",
                bbox=dict(boxstyle="circle,pad=0.35", facecolor=ACCENT, lw=0))
        ax.text(0.17, y + 0.12, head, fontsize=9.5, color=FG, weight="bold", va="center")
        ax.text(0.17, y - 0.18, sub, fontsize=7.5, color=MUTED, va="center")
        if i < len(steps) - 1:
            ax.add_patch(FancyArrowPatch((0.09, y - 0.38), (0.09, y - 0.62), arrowstyle="-|>",
                                         mutation_scale=7, color=MUTED, lw=0.8))
    sl.tech("Method: Analytic Hierarchy Process (Saaty, 1980) + weighted linear combination, a standard "
            "GIS multi-criteria decision analysis (MCDA). Everything is code: Python, GDAL/rasterio, "
            "NumPy/SciPy, GeoPandas; 4 Jupyter notebooks and 27 automated tests, all on GitHub.")
    return sl.finish()


def s04_data(d):
    sl = Slide(4, "Step 1 · Data", "Three free datasets, all measured from orbit")
    sl.plain("Every map below covers the same piece of Mars (Chryse Planitia, where Viking 1 and "
             "Pathfinder landed). Each one tells us something different about the ground.")
    grid, L = d["grid"], d["L"]
    win, ext = crop(grid, (295, 335), (0, 30))
    panels = [("Height (elevation)", "laser altimeter + stereo camera", L["elev"], "terrain", None,
               "blue/green = low ground  ·  brown/white = high ground"),
              ("Dust vs. solid ground", "heat sensor: how fast the ground cools at night", L["ti"],
               "cividis", (50, 450), "dark blue = soft dust  ·  yellow = solid, rocky ground"),
              ("Crater density", "384,343 catalogued craters ≥ 1 km", L["crater"], "magma", None,
               "dark = few craters  ·  bright = many craters")]
    for title, how, arr, cmap, lims, legend in panels:
        sl.fig.text(0.07, sl.y, title, fontsize=9, color=FG, weight="bold", va="top")
        sl.fig.text(0.93, sl.y, how, fontsize=6.5, color=MUTED, va="top", ha="right")
        sl.y -= _lh(9) + 0.004
        ax = sl.axes(0.106, gap=0.004)
        a = arr[win][::2, ::2]
        vmin, vmax = lims or (np.nanpercentile(a, 2), np.nanpercentile(a, 98))
        ax.imshow(plots.hillshade(L["elev"][win][::2, ::2], grid.res * 2), cmap="gray", extent=ext, aspect="auto")
        im = ax.imshow(a, cmap=cmap, vmin=vmin, vmax=vmax, alpha=0.8, extent=ext, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(LINE)
        sl.text(legend, 6.8, MUTED, gap=0.014)
    sl.tech("MOLA–HRSC blended DEM v2 (USGS, 200 m/px; Fergason et al. 2018) · TES nightside thermal "
            "inertia (Putzig & Mellon 2007, ~3 km/px; < 100 J m⁻² K⁻¹ s⁻½ ≈ deep dust) · Robbins & Hynek "
            "(2012) crater database. The 11 GB DEM is streamed over HTTP; only the ±30° band is read.")
    return sl.finish()


def s05_grid(d):
    sl = Slide(5, "Step 2 · One common grid", "Making every map line up, pixel for pixel")
    sl.plain("The datasets come from different instruments, with different pixel sizes and even "
             "different longitude conventions. Before comparing them, I re-projected everything onto "
             "one grid of 1 km cells covering the band 30° north to 30° south of the equator. "
             "Check: the red circles are craters from the catalogue, drawn on top of the elevation "
             "map. They sit exactly on the crater rims, so the layers are aligned.")
    grid, L = d["grid"], d["L"]
    win, ext = crop(grid, (131, 143), (-10, -1))
    ax = sl.axes(0.36)
    ax.imshow(plots.hillshade(L["elev"][win], grid.res), cmap="gray", extent=ext, aspect="auto")
    ax.imshow(L["elev"][win], cmap="terrain", alpha=0.35, extent=ext, aspect="auto")
    cr = d["craters"]
    sub = cr[cr.lon_e.between(ext[0], ext[1]) & cr.lat.between(ext[2], ext[3]) & (cr.diamkm >= 8)]
    for _, c in sub.iterrows():
        rdeg = c.diamkm / 2 / (C.M_PER_DEG / 1000)
        ax.add_patch(Ellipse((c.lon_e, c.lat), 2 * rdeg / np.cos(np.radians(c.lat)),
                                                     2 * rdeg, fill=False, ec=RED, lw=0.7))
    s = d["sites"].set_index("site").loc["Curiosity"]
    ax.scatter([s.lon_e180], [s.lat], marker="*", s=90, c="white", edgecolors="black", zorder=5)
    ax.annotate("Curiosity, inside Gale crater", (s.lon_e180, s.lat), xytext=(6, 6), textcoords="offset points",
                fontsize=7, color="white", weight="bold")
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    lon_ticks(ax, ext, step=4, fs=6)
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    sl.tech("CRS: Mars 2015 equirectangular, IAU_2015:49910 (sphere R = 3,396,190 m), planetocentric "
            "latitude, east-positive longitude. Grid 21,338 × 3,556 px. In this projection an east–west "
            "pixel covers only cos(latitude) km on the ground (866 m at 30°), so slope, crater kernels, "
            "zone areas and random sampling are all corrected for it (otherwise slope is off by up to 13 %).")
    return sl.finish()


def s06_criteria(d):
    sl = Slide(6, "Step 3 · Six criteria", "Six questions asked of every 1 km cell")
    sl.plain("From the data I computed six maps. Each one answers a simple question. Below, each map "
             "is already converted to a score: yellow = good for landing, purple = bad.")
    rows = [("slope", "How steep is it?", "flatter is better"),
            ("rough", "How bumpy is it?", "smoother is better"),
            ("elev", "How high is it?", "lower is better"),
            ("ti", "Dust or solid ground?", "solid is better"),
            ("crater", "How many craters nearby?", "fewer is better"),
            ("lat", "How far from the equator?", "closer is better")]
    grid = d["grid"]
    win, ext = crop(grid, (295, 335), (0, 30))
    top = sl.y
    h, wcol = 0.1, 0.415
    for i, (k, q, better) in enumerate(rows):
        col, row = i % 2, i // 2
        x = 0.07 + col * (wcol + 0.03)
        y = top - row * (h + 0.058)
        sl.fig.text(x, y, q, fontsize=8, color=FG, weight="bold", va="top")
        sl.fig.text(x, y - _lh(8), better, fontsize=6.5, color=ACCENT, va="top")
        ax = sl.fig.add_axes([x, y - _lh(8) - _lh(6.5) - h - 0.004, wcol, h])
        ax.imshow(d["scores"][k][win][::2, ::2], cmap="viridis", vmin=0, vmax=1, extent=ext, aspect="auto")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color(LINE)
    sl.y = top - 3 * (h + 0.058) - 0.005
    sl.tech("Slope from the latitude-corrected DEM gradient; roughness = Terrain Ruggedness Index "
            "(mean absolute height difference to the 8 neighbours); crater density = craters ≥ 1 km per "
            "10,000 km² within 50 km (elliptical kernel in pixel space, catalogue read to ±31° to avoid "
            "edge effects); |latitude| from the grid. Same region as slide 4. At 1 km, roughness closely "
            "follows slope (visible above); a 200 m roughness layer is the planned improvement.")
    return sl.finish()


def s07_constraints(d):
    L, mask, grid = d["L"], d["mask"], d["grid"]
    valid = overlay.valid_data(*L.values())
    cat = np.full(mask.shape, np.nan, dtype="float32")
    cat[valid] = 0
    cat[valid & (L["slope"] > C.MAX_SLOPE_DEG)] = 3
    cat[valid & (L["ti"] < C.MIN_TI)] = 2
    cat[valid & (L["elev"] > C.MAX_ELEV_M)] = 1
    share = {k: float(np.mean(cat[valid] == k)) for k in (0, 1, 2, 3)}
    used = d["used"]
    sl = Slide(7, "Step 4 · Scores and hard limits", "Grade everything 0 → 1, and rule out the impossible")
    sl.plain("Each criterion is turned into a grade from 0 (worst) to 1 (best), like marking an exam. "
             "Some ground is simply impossible, whatever its other grades, so it's removed first. "
             f"Only {share[0]:.0%} of the band survives:")
    cmap = ListedColormap(["#2ea043", "#8957e5", YELLOW, RED])
    band_strips(sl, None, L["elev"], grid, None, height=0.125, layer=cat, cmap=cmap, vmin=-0.5, vmax=3.5)
    for col, lab in [(0, f"possible ({share[0]:.0%})"), (1, f"too high: above +1 km ({share[1]:.0%})"),
                     (2, f"too dusty: soft, deep dust ({share[2]:.0%})"), (3, f"too steep: over 10° ({share[3]:.0%})")]:
        sl.fig.patches.append(FancyBboxPatch((0.07, sl.y - 0.012), 0.025, 0.012, boxstyle="square,pad=0",
                                             transform=sl.fig.transFigure, facecolor=cmap(col), lw=0))
        sl.fig.text(0.11, sl.y, lab, fontsize=7.5, color=FG, va="top")
        sl.y -= _lh(7.5) + 0.004
    sl.y -= 0.01
    sl.text("Big volcanoes (Tharsis, Elysium) are out because they're too high; the bright, dusty "
            "plains of Arabia and Tharsis are out because the surface is too soft.", 8.5, MUTED)
    sl.tech(f"Linear fuzzy membership: elevation 1 at ≤ −4 km → 0 at ≥ +1 km; slope 1 at 0° → 0 at ≥ 5°; "
            f"thermal inertia 0 at ≤ 100 → 1 at ≥ 300; roughness and craters 1 at the minimum → 0 at the 95th "
            f"percentile of possible ground (TRI {used['rough'][1]:.0f} m, {used['crater'][1]:.0f} craters/10⁴ km²); "
            f"latitude 1 at 0° → 0 at 30°. Hard constraints: elevation > +1 km, slope > 10°, TI < 100.")
    return sl.finish()


def s08_ahp(d):
    a = d["ahp"]
    sl = Slide(8, "Step 5 · Weights (AHP)", "Which criteria matter most?")
    sl.plain("Instead of guessing percentages, I used a method called AHP: compare the criteria two at "
             "a time and say how much more important one is than the other. For example:")
    ex = [("Slope", "Elevation", "2×", "a tip-over is immediate; high ground only reduces margins"),
          ("Slope", "Craters", "3×", "craters mostly matter through the slopes they create"),
          ("Slope", "Latitude", "4×", "within ±30°, latitude is a preference, not a danger")]
    for a_, b_, k, why in ex:
        sl.fig.text(0.07, sl.y, f"{a_}  vs  {b_}", fontsize=9, color=FG, weight="bold", va="top")
        sl.fig.text(0.93, sl.y, f"{k} more important", fontsize=9, color=ACCENT, weight="bold", va="top", ha="right")
        sl.y -= _lh(9)
        sl.text(why, 7.5, MUTED, gap=0.01)
    sl.text("15 comparisons like these are turned into weights by matrix maths, and the method checks "
            "that my answers don't contradict each other.", 9, FG, gap=0.015)
    names = {"slope": "Slope", "rough": "Roughness", "elev": "Elevation", "ti": "Dust (thermal inertia)",
             "crater": "Craters", "lat": "Latitude"}
    ax = sl.axes(0.25, left=0.3, width=0.6)
    ks = list(a.weights)[::-1]
    vals = [a.weights[k] * 100 for k in ks]
    ax.barh(range(len(ks)), vals, color=[ACCENT if v > 20 else BLUE for v in vals], height=0.6)
    for i, v in enumerate(vals):
        ax.text(v + 0.8, i, f"{v:.0f} %", va="center", fontsize=8, color=FG, weight="bold")
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels([names[k] for k in ks], fontsize=8, color=FG)
    ax.set_xlim(0, 34); ax.set_xticks([])
    dark(ax)
    for s_ in ("top", "right", "bottom"):
        ax.spines[s_].set_visible(False)
    sl.tech(f"Saaty 1–9 scale, 6 × 6 reciprocal matrix; weights = principal eigenvector. λmax = {a.lambda_max:.3f}, "
            f"consistency index CI = {a.ci:.4f}, consistency ratio CR = CI / RI = {a.cr:.4f} (RI = 1.24 for n = 6). "
            f"CR < 0.10 means the judgements are consistent. Safety criteria dominate: a failed landing "
            f"ends the mission.")
    return sl.finish()


def s09_result(d):
    sl = Slide(9, "Step 6 · The result", "The landing-suitability map of equatorial Mars")
    sl.plain("Each cell's final score is the weighted sum of its six grades: 1 would be perfect on "
             "everything. Yellow areas are the safest places to land; grey areas were ruled out. "
             "Stars are the 8 real landers.")
    band_strips(sl, d["S"], d["L"]["elev"], d["grid"], d["sites"], height=0.155)
    colorbar_row(sl, "viridis", "0 · riskier", "1 · safer", "suitability score")
    sl.y -= 0.01
    sl.text("Two huge low plains stand out: Chryse Planitia (top strip, right) and the "
            "Isidis–Elysium–Utopia lowlands (bottom strip). That's exactly where most landers went.",
            9, FG)
    sl.tech("S = Σ wᵢ·sᵢ over the 6 criteria, set to 0 where a hard constraint fails. Output: a 1 km "
            "GeoTIFF (21,338 × 3,556). Classified into 5 quantile classes; the \"very high\" class "
            "(S ≥ 0.80) feeds the candidate-zone search on the next slide.")
    return sl.finish()


def s10_zones(d):
    sl = Slide(10, "Candidate sites", "Zooming in: the best landing zones")
    top = d["top"]
    sl.plain("I grouped the best-scoring cells into zones and kept only those at least as big as a "
             "modern landing oval (50 km²). The 10 best zones are outlined in red. Most sit on the "
             "flat floors of ancient flood channels (Ares and Tiu Valles), just south of where "
             "Pathfinder landed in 1997.")
    grid = d["grid"]
    win, ext = crop(grid, (295, 345), (-6, 26))
    ax = sl.axes(0.42)
    suit_map(ax, d["S"], d["L"]["elev"], grid, win, ext)
    z = d["ranked"][win] > 0
    ax.contour(np.linspace(ext[0], ext[1], z.shape[1]), np.linspace(ext[3], ext[2], z.shape[0]),
               z.astype(float), levels=[0.5], colors=RED, linewidths=1.1)
    s = d["sites"]
    inside = s[s.lon_e180.between(ext[0], ext[1]) & s.lat.between(ext[2], ext[3])]
    ax.scatter(inside.lon_e180, inside.lat, marker="*", s=110, c="white", edgecolors="black", zorder=5)
    for _, r in inside.iterrows():
        ax.annotate(f"{r['site']} ({r['year']})", (r.lon_e180, r.lat), xytext=(6, 4),
                    textcoords="offset points", fontsize=7.5, color="white", weight="bold")
    b = top.iloc[0]
    ax.annotate("zone #1", (b.lon, b.lat), xytext=(-70, -6), textcoords="offset points", fontsize=7.5,
                color="white", weight="bold", bbox=dict(facecolor=RED, lw=0, boxstyle="round,pad=0.25"),
                arrowprops=dict(arrowstyle="-", color=RED, lw=1))
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3])
    lon_ticks(ax, ext, fs=6)
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    sl.tech(f"Zones = 8-connected cells with S ≥ 0.80 (merged across the 180° seam), area from true cell "
            f"areas, minimum 50 km², ranked by area-weighted mean score. Zone #1: {b.lat:.2f}° N, "
            f"{b.lon % 360:.2f}° E, {b.area_km2:.0f} km², mean score {b.mean_score:.2f}. "
            f"Exported as a GeoPackage.")
    return sl.finish()


def s11_validation(d):
    r = d["res"]
    pct = r["site_percentiles"]
    chosen = dict(zip(d["sites"].site, d["sites"].chosen_for))
    year = dict(zip(d["sites"].site, d["sites"].year))
    sl = Slide(11, "The test", "Does the map agree with where real landers went?")
    sl.plain("If the model makes sense, the 8 places where landers really touched down should score "
             "much higher than random spots on Mars. So I compared each landing site with 1,000 "
             "random locations. A score of 90 means \"better than 90 % of random places\".")
    ax = sl.axes(0.3, left=0.33, width=0.58, gap=0.062)
    order = sorted(pct, key=pct.get)
    ax.barh(range(len(order)), [pct[s] for s in order], color=[CHOSEN_COLOR[chosen[s]] for s in order], height=0.6)
    for i, s in enumerate(order):
        ax.text(pct[s] - 1.5, i, f"{pct[s]:.0f}", va="center", ha="right", fontsize=8, color=BG, weight="bold")
    ax.axvline(80, color=FG, lw=0.8, ls="--")
    ax.text(80, len(order) - 0.35, " top 20 %", fontsize=7, color=FG, va="bottom")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([f"{s} ({year[s]})" for s in order], fontsize=7.5, color=FG)
    ax.set_xlim(0, 100)
    ax.set_xlabel("percentile vs. 1,000 random places", fontsize=7)
    dark(ax)
    for k, c in CHOSEN_COLOR.items():
        sl.fig.text(0.07 + list(CHOSEN_COLOR).index(k) * 0.45, sl.y, "■ " + CHOSEN_LABEL[k], color=c,
                    fontsize=7.5, va="top")
    sl.y -= _lh(7.5) + 0.015
    sl.text(f"All {r['sites_in_top20']} landing sites are in the top 20 %. If the map were random, "
            f"the chance of this would be about 1 in 100,000.", 10, ACCENT, weight="bold")
    sl.tech(f"Each site = mean score within 10 km. Random points drawn uniformly by area (weighted by "
            f"cos latitude) over the whole band, measured the same way. One-sided Mann–Whitney U test: "
            f"p = {r['mann_whitney_p']:.1e}. Coordinates are planetocentric, from PDS/NASA sources.")
    return sl.finish()


def s12_insight(d):
    r = d["res"]
    pct, strict = r["site_percentiles"], r["site_percentiles_passing"]
    chosen = dict(zip(d["sites"].site, d["sites"].chosen_for))
    sl = Slide(12, "The surprise", "The \"misses\" show where science beat safety")
    sl.plain("A tougher test: compare each site only with ground that was already possible (no "
             "volcanoes, no dust fields). Now the 8 sites split cleanly in two. The four chosen mainly "
             "for safety (Viking 1, Pathfinder, InSight, Zhurong) stay at the very top. The four rover "
             "sites chosen for their geology (Gusev, Meridiani, Gale, Jezero) drop to the middle.")
    ax = sl.axes(0.27, left=0.3, width=0.62, gap=0.062)
    order = sorted(strict, key=strict.get)
    for i, s in enumerate(order):
        ax.plot([strict[s], pct[s]], [i, i], color=LINE, lw=2.5, zorder=1)
        ax.scatter(pct[s], i, s=28, color=MUTED, zorder=2)
        ax.scatter(strict[s], i, s=48, color=CHOSEN_COLOR[chosen[s]], zorder=3)
        ax.text(strict[s] - 2.5, i, f"{strict[s]:.0f}", ha="right", va="center", fontsize=7.5,
                color=CHOSEN_COLOR[chosen[s]], weight="bold")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order, fontsize=8, color=FG)
    ax.set_xlim(35, 102)
    ax.set_xlabel("percentile   (grey = easy test, colour = tough test)", fontsize=7)
    dark(ax)
    sl.text("Scientists accepted less safety margin to reach interesting rocks, helped by better landing "
            "technology (airbags, then guided entry, then Terrain-Relative Navigation). The model only "
            "knows about safety, so the gap measures the margin traded for science.", 8.8, FG)
    sl.tech(f"Reference = 1,000 random points drawn only from constraint-passing ground. 4 of 8 sites stay in "
            f"the top 20 % (one-sided Mann–Whitney p = {float(r['strict'].split('p = ')[1].rstrip(').')):.4f}). "
            f"Safety-first sites: {min(strict[k] for k in strict if chosen[k] == 'safety'):.0f}–"
            f"{max(strict[k] for k in strict if chosen[k] == 'safety'):.0f}th percentile; science-first: "
            f"{min(strict[k] for k in strict if chosen[k] == 'science'):.0f}–"
            f"{max(strict[k] for k in strict if chosen[k] == 'science'):.0f}th. Grouping = primary driver of "
            f"the final site choice (simplified).")
    return sl.finish()


def s13_zoom(d):
    grid, top = d["grid"], d["top"]
    b = top.iloc[0]
    import rasterio
    with rasterio.open(C.PROCESSED / "dem200_best.tif") as src:
        dem_f = src.read(1).astype("float64")
        t = src.transform
    dem_f[dem_f == -9999] = np.nan
    xs = t.c + (np.arange(dem_f.shape[1]) + 0.5) * t.a
    ys = t.f + (np.arange(dem_f.shape[0]) + 0.5) * t.e
    zone = d["ranked"][np.ix_(np.floor((grid.ymax - ys) / grid.res).astype(int),
                              np.floor((xs - grid.xmin) / grid.res).astype(int))] == 1
    dzdy, dzdx = np.gradient(dem_f, C.FINE_PIXEL_M)
    slope = np.degrees(np.arctan(np.hypot(dzdx / np.cos(np.radians(b.lat)), dzdy)))
    v = slope[zone]
    sl = Slide(13, "Reality check", "Looking at the #1 zone 5× more closely")
    sl.plain("The model works with 1 km cells, which can hide cliffs. So I re-downloaded the #1 zone at "
             "200 m resolution and measured its slopes. Inside the red outline, the ground is "
             f"almost perfectly flat (typical slope {np.median(v):.1f}°, no spot above 5°), while the "
             "channel walls around it are steep. The zone passes.")
    ax = sl.axes(0.38, left=0.13, width=0.74)
    ax.imshow(plots.hillshade(dem_f, C.FINE_PIXEL_M), cmap="gray")
    cs = ax.contourf(slope, levels=[5, 10, 90], colors=[YELLOW, RED], alpha=0.45)
    ax.contour(zone.astype(float), levels=[0.5], colors=[RED], linewidths=1.6)
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    ax.plot([8, 8 + 10_000 / C.FINE_PIXEL_M], [dem_f.shape[0] - 10] * 2, color="white", lw=3)
    ax.text(8, dem_f.shape[0] - 16, "10 km", color="white", fontsize=7)
    for i, (c, lab) in enumerate([(YELLOW, "slope 5–10°"), (RED, "slope > 10°"), (RED, "zone #1 outline")]):
        sl.fig.text(0.16 + i * 0.25, sl.y, ("■ " if i < 2 else "▢ ") + lab, color=c, fontsize=7.5, va="top")
    sl.y -= _lh(7.5)
    sl.tech(f"200 m DEM window (80 × 80 km) cut from the remote MOLA–HRSC file. Inside zone #1: slope median "
            f"{np.median(v):.2f}°, 95th percentile {np.percentile(v, 95):.2f}°, 0 % of cells > 5°. Whole window: "
            f"median {np.nanmedian(slope):.1f}°, 95th percentile {np.nanpercentile(slope, 95):.1f}° (channel walls). "
            f"Location: {b.lat:.2f}° N, {b.lon % 360:.2f}° E, Ares Vallis floor.")
    return sl.finish()


def s14_robust(d):
    oat, scen = d["oat"], d["scen"]
    kept20 = oat[oat.change.abs().round(2) == 0.2].top_overlap
    sl = Slide(14, "Robustness", "What if my weights were wrong?")
    sl.plain("The weights from slide 8 are my judgement, so I tested how much the answer depends on "
             "them. I changed each weight up and down by 20 % and 40 % and re-ran everything, and "
             "then ran 500 versions with random weights. The best zones mostly stay the same, and "
             "the real landing sites score high every time.")
    ax = sl.axes(0.2, left=0.15, width=0.77, gap=0.06)
    names = {"slope": "slope", "rough": "roughness", "elev": "elevation", "ti": "dust", "crater": "craters",
             "lat": "latitude"}
    cols = [ACCENT, BLUE, GREEN, YELLOW, "#d2a8ff", "#79c0ff"]
    for (k, g), c in zip(oat.groupby("criterion"), cols):
        ax.plot(g.change * 100, g.top_overlap, marker="o", ms=3, lw=1.3, color=c, label=names[k])
    ax.set_ylim(0, 10.5)
    ax.set_xlabel("change in one criterion's weight (%)", fontsize=7)
    ax.set_ylabel("best-10 zones kept", fontsize=7)
    ax.legend(fontsize=6, ncol=3, frameon=False, labelcolor=FG, loc="lower center")
    dark(ax, grid_lines=True)
    sl.y -= 0.01
    sl.text("500 random weightings: how often is each place in the top 10 %?", 8.5, FG, weight="bold", gap=0.006)
    grid = d["grid"]
    win, ext = crop(grid, (295, 345), (-6, 26))
    ax = sl.axes(0.11, gap=0.004)
    ax.imshow(plots.hillshade(d["L"]["elev"][win][::2, ::2], grid.res * 2), cmap="gray", extent=ext, aspect="auto")
    p = d["prob"][win][::2, ::2]
    ax.imshow(np.ma.masked_where(~(p > 0.02), p), cmap="magma", vmin=0, vmax=1, alpha=0.9, extent=ext, aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])
    for s_ in ax.spines.values():
        s_.set_color(LINE)
    colorbar_row(sl, "magma", "never", "every time", None)
    base = scen[scen.scenario.str.startswith("AHP")].mean_site_percentile.iloc[0]
    sl.tech(f"One-at-a-time: ±20 % keeps {kept20.min()}–{kept20.max()} of the 10 best zones; ±40 % keeps 6–10. "
            f"Scenarios: \"Safety first\" (slope + roughness 70 %) and \"Equal weights\" keep only "
            f"{', '.join(str(v) for v in scen.top10_overlap.iloc[1:])} of the 10 zones, but the mean landing-site "
            f"percentile stays at {' and '.join(f'{v:.0f}' for v in scen.mean_site_percentile.iloc[1:])} (baseline {base:.0f}). "
            f"Monte Carlo: 500 Dirichlet draws centred on the AHP weights. Ranking is sensitive; the map is stable.")
    return sl.finish()


def s15_end(d):
    sl = Slide(15, "Takeaways", "What I learned, and what's next")
    pts = [("Earth GIS methods transfer to other planets.",
            "Projection, resampling, raster algebra and MCDA work on Mars exactly as on Earth, once you "
            "respect its own coordinate system."),
           ("Validation is where the insight is.",
            "The simple test said \"8/8, great\". The tougher test revealed the safety-vs-science "
            "trade-off, which is the real story."),
           ("Honest limits.",
            "1 km cells can't see individual rocks; the thresholds are illustrative, not mission "
            "specifications; real site selection also uses rock counts from HiRISE images, radar "
            "and wind data."),
           ("Next.",
            "200 m roughness for the whole band, a science-value layer from the geological map, and the "
            "same analysis for the Moon's south pole (Artemis).")]
    for head, body in pts:
        sl.text(head, 10.5, ACCENT, weight="bold", gap=0.004)
        sl.text(body, 8.8, FG, gap=0.022)
    sl.y -= 0.01
    sl.text("Code, data sources, notebooks and all figures:", 9, MUTED, gap=0.004)
    sl.text(REPO, 10.5, BLUE, weight="bold", gap=0.02)
    sl.text("Questions and feedback welcome, especially from planetary scientists and GIS people!", 9, FG)
    sl.tech("Data: NASA/ESA/DLR/USGS (MOLA–HRSC DEM v2), Putzig & Mellon 2007 (TES thermal inertia), Robbins & "
            "Hynek 2012 (craters). Tools: Python, GDAL/rasterio, NumPy, SciPy, pandas, GeoPandas, Matplotlib, "
            "Jupyter. MIT licence. Fully reproducible: one download script, 4 notebooks, 27 tests.")
    return sl.finish()


def post_image(d):
    fig = plots.region_map(d["S"], d["L"]["elev"], d["grid"], (285, 355), (-5, 27.5), sites=d["sites"],
                           zones=d["ranked"],
                           title="Where is it safest to land on Mars? (an Earth GIS method, tested on real landers)",
                           subtitle="Yellow = safest ground · grey = ruled out (too high, too dusty or too steep) · "
                                    "red outline = 10 best candidate zones · ★ = real landing sites")
    fig.savefig(OUT / "post_image.png", dpi=200, facecolor=BG)
    plt.close(fig)


SLIDES = [s01_cover, s02_problem, s03_method, s04_data, s05_grid, s06_criteria, s07_constraints, s08_ahp,
          s09_result, s10_zones, s11_validation, s12_insight, s13_zoom, s14_robust, s15_end]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in OUT.glob("slide_*.png"):
        old.unlink()
    d = load()
    post_image(d)
    with PdfPages(OUT / "carousel.pdf") as pdf:
        for i, make in enumerate(SLIDES, 1):
            fig = make(d)
            pdf.savefig(fig, facecolor=BG)
            fig.savefig(OUT / f"slide_{i:02d}.png", dpi=DPI, facecolor=BG)
            plt.close(fig)
            print(f"  slide {i:2d}: {make.__name__[4:]}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
