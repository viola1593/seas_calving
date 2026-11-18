"""Plot Svalbard glacier train/val/test splits over a grayscale basemap.

Runs inside the Docker container (cartopy / geopandas / matplotlib). Needs
internet once for the Natural Earth land download (cached afterwards).

    python svalbard_splits_map.py            # light, for the paper
    python svalbard_splits_map.py --dark     # dark, for slides
"""

import argparse
import csv
import os

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

# --- Config ------------------------------------------------------------------
# Full suffixed keys (e.g. "RGI60-07.00030_A"), one per FrontChange CSV.
# Fill these in with the real split assignments.
train = [  "RGI60-07.00006",
      "RGI60-07.00025",
      "RGI60-07.00028_B",
      "RGI60-07.00029",
      "RGI60-07.00030_A",
      "RGI60-07.00030_B",
      "RGI60-07.00035_A",
      "RGI60-07.00036",
      "RGI60-07.00042",
      "RGI60-07.00067_C",
      "RGI60-07.00238",
      "RGI60-07.00239",
      "RGI60-07.00242",
      "RGI60-07.00244",
      "RGI60-07.00251",
      "RGI60-07.00254",
      "RGI60-07.00258_A",
      "RGI60-07.00259",
      "RGI60-07.00265",
      "RGI60-07.00275",
      "RGI60-07.00277",
      "RGI60-07.00299_A",
      "RGI60-07.00321_B",
      "RGI60-07.00324",
      "RGI60-07.00422",
      "RGI60-07.00424_B",
      "RGI60-07.00425",
      "RGI60-07.00440",
      "RGI60-07.00448",
      "RGI60-07.00464",
      "RGI60-07.00474",
      "RGI60-07.00501_B",
      "RGI60-07.00529_B",
      "RGI60-07.00552",
      "RGI60-07.00570_A",
      "RGI60-07.00570_B",
      "RGI60-07.00645",
      "RGI60-07.00661_B",
      "RGI60-07.00703_C",
      "RGI60-07.00703_D",
      "RGI60-07.00710_A",
      "RGI60-07.00710_B",
      "RGI60-07.00712",
      "RGI60-07.00714",
      "RGI60-07.00721",
      "RGI60-07.00733",
      "RGI60-07.00775_A",
      "RGI60-07.00789",
      "RGI60-07.00790",
      "RGI60-07.00791",
      "RGI60-07.00794",
      "RGI60-07.00832",
      "RGI60-07.00849",
      "RGI60-07.00854",
      "RGI60-07.00854_2",
      "RGI60-07.00859",
      "RGI60-07.00874",
      "RGI60-07.00883",
      "RGI60-07.00887_B",
      "RGI60-07.00896",
      "RGI60-07.00903",
      "RGI60-07.00909",
      "RGI60-07.00910",
      "RGI60-07.00945",
      "RGI60-07.00984_B",
      "RGI60-07.00986",
      "RGI60-07.01100",
      "RGI60-07.01354_A",
      "RGI60-07.01357_B",
      "RGI60-07.01425",
      "RGI60-07.01449",
      "RGI60-07.01461",
      "RGI60-07.01464",
      "RGI60-07.01468",
      "RGI60-07.01487",
      "RGI60-07.01488",
      "RGI60-07.01492",
      "RGI60-07.01563_B"]
val = [  "RGI60-07.00224",
      "RGI60-07.00285_B",
      "RGI60-07.00285_C",
      "RGI60-07.00301",
      "RGI60-07.00429",
      "RGI60-07.00558",
      "RGI60-07.00659",
      "RGI60-07.00682",
      "RGI60-07.00715",
      "RGI60-07.00887_A",
      "RGI60-07.00984_A",
      "RGI60-07.01328"]
test = ["RGI60-07.00030_D",
      "RGI60-07.00035_B",
      "RGI60-07.00060",
      "RGI60-07.00061",
      "RGI60-07.00062_C",
      "RGI60-07.00064",
      "RGI60-07.00240",
      "RGI60-07.00250",
      "RGI60-07.00315",
      "RGI60-07.00318",
      "RGI60-07.00424_A",
      "RGI60-07.00470",
      "RGI60-07.00502",
      "RGI60-07.00524",
      "RGI60-07.00681",
      "RGI60-07.00727",
      "RGI60-07.00898",
      "RGI60-07.00993",
      "RGI60-07.01326",
      "RGI60-07.01353",
      "RGI60-07.01354_2",
      "RGI60-07.01493"]

FRONT_DIR = "../glacier_data/FrontChange"
OUT = "svalbard_splits_map.png"
TITLE = ""
EXTENT = [9.5, 31.5, 76.3, 80.9]  # lon/lat; excludes Bjørnøya
DPI = 300
COLORS = {"train": "C0", "val": "C1", "test": "C2"}

LON_TICKS = [10, 15, 20, 25, 30]
LAT_TICKS = [77, 78, 79, 80]
GRID_LABEL_COLOR = "0.6"

# (lon, lat, name, is_icecap) — placements are hand-tuned for EXTENT above.
REGIONS = [
    (15.6, 78.85, "Spitsbergen", False),
    #(22.8, 80.45, "Nordaustlandet", False),
    (22.6, 77.70, "Edgeøya", False),
    (21.7, 78.55, "Barentsøya", False),
    #(10.3, 78.55, "Prins Karls\nForland", False),
    #(28.9, 78.90, "Kong Karls\nLand", False),
    (24.3, 79.55, "Austfonna", True),
    (20.3, 79.95, "Vestfonna", True),
]

# --- Point loading -----------------------------------------------------------
missing = []


def load_point(key):
    """Return (lon, lat) for a glacier key, or None if its CSV is absent."""
    path = os.path.join(FRONT_DIR, f"{key}.csv")
    if not os.path.exists(path):
        missing.append(key)
        return None
    row = pd.read_csv(path, nrows=1).iloc[0]
    return (row["glacier_lon"], row["glacier_lat"])


splits = {"train": train, "val": val, "test": test}
points = {
    name: [p for p in (load_point(k) for k in keys) if p is not None]
    for name, keys in splits.items()
}

if missing:
    print(f"WARNING: {len(missing)} listed key(s) have no matching CSV:")
    for k in missing:
        print(f"  - {k}")

# --- Figure / axes -----------------------------------------------------------
proj = ccrs.epsg(32633)  # UTM 33N, matches the Li et al. gpkg native CRS
ax = plt.axes(projection=proj)
ax.set_extent(EXTENT, crs=ccrs.PlateCarree())

land = cfeature.NaturalEarthFeature("physical", "land", "10m")
ax.add_feature(land, facecolor="0.85", edgecolor="0.6", linewidth=0.4)

# --- Plot points -------------------------------------------------------------
for name, pts in points.items():
    if not pts:
        continue
    lons, lats = zip(*pts)
    ax.scatter(
        lons,
        lats,
        s=40,
        c=COLORS[name],
        edgecolor="white",
        linewidth=0.5,
        zorder=5,
        transform=ccrs.PlateCarree(),
        label=f"{name.capitalize()} (n={len(pts)})",
    )

# --- Region names ------------------------------------------------------------
for lon, lat, name, is_icecap in REGIONS:
    ax.text(
        lon,
        lat,
        name,
        transform=ccrs.PlateCarree(),
        ha="center",
        va="center",
        fontsize=8,
        style="italic", #if is_icecap else "normal",
        color="0.35", #if is_icecap else "0.2",
        zorder=2,
       # path_effects=[pe.withStroke(linewidth=2, foreground="white")],
    )

# --- Graticule ---------------------------------------------------------------
ax.gridlines(
    draw_labels=False,
    xlocs=LON_TICKS,
    ylocs=LAT_TICKS,
    color="gray",
    alpha=0.3,
    linestyle="--",
)

# Inline tick labels: meridians labelled along the top edge, parallels along the
# right edge, both just inside the frame and sitting on their dashed line.
x0, x1 = ax.get_xlim()
y0, y1 = ax.get_ylim()


def _to_axes_crs(lons, lats):
    pts = proj.transform_points(ccrs.PlateCarree(), np.asarray(lons), np.asarray(lats))
    return pts[:, 0], pts[:, 1]


_lat_scan = np.linspace(74.0, 83.0, 4001)
y_top = y1 - 0.015 * (y1 - y0)
for lon in LON_TICKS:
    xs, ys = _to_axes_crs(np.full_like(_lat_scan, lon), _lat_scan)
    i = np.argmin(np.abs(ys - y_top))
    if x0 < xs[i] < x1:
        ax.text(
            xs[i],
            ys[i],
            f"{lon}°E",
            ha="center",
            va="top",
            fontsize=7,
            color=GRID_LABEL_COLOR,
            zorder=6,
        )

_lon_scan = np.linspace(0.0, 45.0, 4001)
x_right = x1 - 0.01 * (x1 - x0)
for lat in LAT_TICKS:
    xs, ys = _to_axes_crs(_lon_scan, np.full_like(_lon_scan, lat))
    i = np.argmin(np.abs(xs - x_right))
    if y0 < ys[i] < y1:
        ax.text(
            xs[i],
            ys[i],
            f"{lat}°N",
            ha="right",
            va="bottom",
            fontsize=7,
            color=GRID_LABEL_COLOR,
            zorder=6,
        )

# --- Legend + title + save ---------------------------------------------------
ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
if TITLE:
    ax.set_title(TITLE)

plt.savefig(OUT, dpi=DPI, bbox_inches="tight")
print(f"Saved {OUT}")
