"""LinkedIn assets from the finished analysis (run after notebook 04).

    python scripts/make_linkedin_images.py

Writes to outputs/linkedin/:
- post_image.png      1.91:1 regional map (Chryse Planitia / Ares Vallis), for a single-image post
- carousel.pdf        5 slides, 1080 x 1350, for a LinkedIn document (carousel) post
- slide_1..5.png      the same slides as images
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from marsgis import config as C, grid as G, plots, validation

BG, FG, MUTED, ACCENT = "#0e1117", "#e6edf3", "#8b949e", "#ff6b4a"
OUT = C.OUTPUTS / "linkedin"
SLIDE = (5.4, 6.75)          # inches; x 200 dpi = 1080 x 1350


def slide(title, subtitle=None, sub_y=0.885):
    fig = plt.figure(figsize=SLIDE, dpi=200)
    fig.patch.set_facecolor(BG)
    fig.text(0.07, 0.95, title, fontsize=17, weight="bold", color=FG, va="top", linespacing=1.15)
    if subtitle:
        fig.text(0.07, sub_y, subtitle, fontsize=9.5, color=MUTED, va="top")
    return fig


def footer(fig, n, total=5):
    fig.text(0.07, 0.025, "Mohamed Jihed Fradi · Geomatics Engineering, MSE Manouba", fontsize=7, color=MUTED)
    fig.text(0.93, 0.025, f"{n}/{total}", fontsize=7, color=MUTED, ha="right")


def dark_axes(ax):
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_color("#30363d")
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    grid = G.reference_grid()
    S, _ = G.read_raster(C.OUTPUTS / "suitability.tif")
    dem, _ = G.read_raster(C.PROCESSED / "dem_1km.tif")
    ranked = np.load(C.PROCESSED / "ranked_zones.npy")
    sites = validation.load_sites()
    res = json.load(open(C.OUTPUTS / "results.json", encoding="utf-8"))
    top = pd.read_csv(C.OUTPUTS / "top10_sites.csv")
    pct = res["site_percentiles"]
    pct_pass = res["site_percentiles_passing"]
    k, n = res["sites_in_top20"], res["n_sites"]

    # --- single post image ---------------------------------------------------------
    fig = plots.region_map(S, dem, grid, (285, 355), (-5, 27.5), sites=sites, zones=ranked,
                           title="Where would you land on Mars?",
                           subtitle=f"GIS multi-criteria suitability (AHP + weighted overlay) · {k} of {n} real "
                                    f"landing sites score in the top 20 % · top-10 zones in red")
    fig.savefig(OUT / "post_image.png", dpi=200, facecolor=BG)
    plt.close(fig)

    slides = []

    # 1 — hook: the whole band as two undistorted strips (west / east hemisphere)
    f = slide("Can Earth GIS methods pick\na landing site on Mars?",
              "Same multi-criteria workflow used for site selection on Earth,\n"
              "applied to public NASA/ESA orbital data for Mars' ±30° band.", sub_y=0.825)
    step = 6
    half = grid.width // 2
    for i, (cols, lon0) in enumerate([(slice(0, half), -180), (slice(half, None), 0)]):
        ax = f.add_axes([0.03, 0.47 - i * 0.215, 0.94, 0.2])
        d, s_ = dem[:, cols][::step, ::step], S[:, cols][::step, ::step]
        ext = [lon0, lon0 + 180, -30, 30]
        ax.imshow(plots.hillshade(d, grid.res * step), cmap="gray", extent=ext, aspect="auto")
        ax.imshow(np.ma.masked_where(~np.isfinite(s_) | (s_ <= 0), s_), cmap="viridis", vmin=0, vmax=1,
                  alpha=0.75, extent=ext, aspect="auto")
        sel = sites[(sites.lon_e180 >= lon0) & (sites.lon_e180 < lon0 + 180)]
        ax.scatter(sel.lon_e180, sel.lat, marker="*", s=70, c="white", edgecolors="black", lw=0.5, zorder=5)
        for _, r in sel.iterrows():
            east = r.lon_e180 > lon0 + 160                    # keep labels inside the strip
            ax.annotate(r["site"], (r.lon_e180, r.lat), xytext=(-5 if east else 4, 3),
                        textcoords="offset points", ha="right" if east else "left",
                        fontsize=5.5, color="white", weight="bold")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#30363d")
        ax.text(0.005, 0.03, "180°–360°E" if i == 0 else "0°–180°E", transform=ax.transAxes,
                fontsize=6, color="white")
    f.text(0.07, 0.19, f"Result: {k} of {n} real landing sites\nscored in the top 20 %.", fontsize=15,
           weight="bold", color=ACCENT, va="top")
    footer(f, 1)
    slides.append(f)

    # 2 — data & criteria
    f = slide("Six criteria, three free datasets",
              "Everything on one Mars equirectangular grid\n(IAU_2015:49910), 1 km pixels, ±30° latitude.")
    rows = [("Slope", "MOLA–HRSC DEM", "27 %"), ("Roughness", "MOLA–HRSC DEM", "27 %"),
            ("Elevation", "MOLA–HRSC DEM", "15 %"), ("Thermal inertia", "TES (Putzig & Mellon)", "15 %"),
            ("Crater density", "Robbins & Hynek, 384k craters", "9 %"), ("Latitude", "grid", "6 %")]
    y = 0.74
    f.text(0.07, y, "Criterion", fontsize=8, color=MUTED); f.text(0.42, y, "Source", fontsize=8, color=MUTED)
    f.text(0.93, y, "AHP weight", fontsize=8, color=MUTED, ha="right")
    for name, src, w in rows:
        y -= 0.065
        f.text(0.07, y, name, fontsize=11, color=FG, weight="bold")
        f.text(0.42, y, src, fontsize=8.5, color=MUTED)
        f.text(0.93, y, w, fontsize=11, color=ACCENT, ha="right", weight="bold")
    f.text(0.07, 0.29, "Hard constraints remove ground above +1 km,\nslopes over 10° and dust "
                       "(thermal inertia < 100).\nAHP consistency ratio = 0.007 (limit 0.10).",
           fontsize=9.5, color=FG, va="top", linespacing=1.5)
    footer(f, 2)
    slides.append(f)

    # 3 — validation
    f = slide("Validation: 8 real landing sites",
              "Mean score within 10 km, ranked against 1,000 random points.\n"
              "Grey: vs. the whole band · colour: vs. ground that is already safe.")
    ax = f.add_axes([0.32, 0.1, 0.6, 0.64])
    order = sorted(pct_pass, key=pct_pass.get)
    yy = np.arange(len(order))
    chosen = dict(zip(sites.site, sites.chosen_for))
    ax.barh(yy, [pct[s] for s in order], color="#30363d", height=0.6)
    ax.barh(yy, [pct_pass[s] for s in order], height=0.35,
            color=[ACCENT if chosen[s] == "science" else "#58a6ff" if "science" in chosen[s] else "#3fb950"
                   for s in order])
    ax.set_yticks(yy)
    ax.set_yticklabels(order, fontsize=9, color=FG)
    ax.axvline(80, color=MUTED, lw=0.8, ls="--")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Percentile")
    dark_axes(ax)
    ax.tick_params(axis="y", colors=FG, labelsize=9)
    f.text(0.07, 0.78, "■ safety-first", color="#3fb950", fontsize=8)
    f.text(0.33, 0.78, "■ safety + science", color="#58a6ff", fontsize=8)
    f.text(0.63, 0.78, "■ science-driven", color=ACCENT, fontsize=8)
    footer(f, 3)
    slides.append(f)

    # 4 — the insight
    safe = [s for s in order[::-1] if chosen[s] == "safety" or s == "Zhurong"]
    f = slide("The misses are the interesting part",
              None)
    f.text(0.07, 0.84,
           f"Even against ground that is already safe,\n"
           f"{', '.join(safe[:-1])} and {safe[-1]}\nstay in the top {100 - min(pct_pass[s] for s in safe):.0f} %.\n\n"
           f"Jezero (Perseverance) and Gale (Curiosity)\ndrop to the {pct_pass['Perseverance']:.0f}th and "
           f"{pct_pass['Curiosity']:.0f}th percentile.\n\n"
           f"They were chosen for science, and needed\nprecision landing (guided entry, Terrain-\n"
           f"Relative Navigation) to get there.\n\n"
           f"The model encodes engineering safety only;\nthe gap measures what science bought\nat the cost of margin.",
           fontsize=11, color=FG, va="top", linespacing=1.45)
    lo_s, hi_s = min(pct_pass[s] for s in safe), max(pct_pass[s] for s in safe)
    sci = ["Perseverance", "Curiosity"]
    lo_c, hi_c = min(pct_pass[s] for s in sci), max(pct_pass[s] for s in sci)
    for x, big, label, col in [(0.07, f"{lo_s:.0f}–{hi_s:.0f}", "percentile\nViking 1, Pathfinder,\nInSight, Zhurong", "#3fb950"),
                               (0.52, f"{lo_c:.0f}–{hi_c:.0f}", "percentile\nJezero & Gale", ACCENT)]:
        f.text(x, 0.3, big, fontsize=34, weight="bold", color=col, va="top")
        f.text(x, 0.19, label, fontsize=9, color=MUTED, va="top")
    footer(f, 4)
    slides.append(f)

    # 5 — best zone + robustness + link
    b = top.iloc[0]
    f = slide("Best candidate & robustness",
              f"Zone #1: {b.lat:.2f}° N, {b.lon % 360:.2f}° E · {b.area_km2:.0f} km² · Ares Vallis floor")
    zoom = plt.imread(C.FIGURES / "zoom_best.png")
    ax = f.add_axes([0.07, 0.36, 0.86, 0.5])
    ax.imshow(zoom)
    ax.axis("off")
    oat = pd.read_csv(C.OUTPUTS / "sensitivity_oat.csv")
    kept = oat[oat.change.abs().round(2) == 0.2].top_overlap
    f.text(0.07, 0.31,
           "• Checked at 200 m: flat inside the zone, steep walls outside\n"
           f"• ±20 % on any weight keeps {kept.min()}–{kept.max()} of the top-10 zones\n"
           "• 500 Monte Carlo weightings map where the result is stable\n\n"
           "Code, data sources & notebooks:\ngithub.com/MJFradi/mars-landing-site-suitability",
           fontsize=8.5, color=FG, va="top", linespacing=1.5)
    footer(f, 5)
    slides.append(f)

    with PdfPages(OUT / "carousel.pdf") as pdf:
        for i, f in enumerate(slides, 1):
            pdf.savefig(f, facecolor=BG)
            f.savefig(OUT / f"slide_{i}.png", dpi=200, facecolor=BG)
            plt.close(f)
    print("wrote", *sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
