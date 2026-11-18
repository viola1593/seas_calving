"""Map each Svalbard glacier to its subbasin and dominant current (ESC/WSC).

Assigns a current to every FrontChange glacier by point-in-polygon containment
of its terminus coordinate against the 6 Noel et al. (2020) subbasin polygons.
Overwrites glacier_data/glacier_current_lookup.json and saves a verification PNG.
"""

# --- Step 1: paths & config ---------------------------------------------------
import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]          # CalvingFronts/
GLACIER_DATA = ROOT / "glacier_data"
FRONT_DIR    = GLACIER_DATA / "FrontChange"
BASIN_DIR    = GLACIER_DATA / "SubBasins" / "SubBasins"
OUT_JSON     = GLACIER_DATA / "glacier_current_lookup.json"
OUT_PNG      = GLACIER_DATA / "glacier_current_subbasin_map.png"

BASINS = ["AF", "BE", "NE", "NW", "SS", "VF"]
ESC_BASINS = {"AF", "BE"}                            # else -> WSC
EPSG_PTS, EPSG_PROJ = 4326, 3995


# --- Step 2: load 6 subbasin polygons into one tagged GeoDataFrame ------------
def load_basins():
    """Return a GeoDataFrame (one row per basin) tagged with `basin`, in EPSG:3995."""
    parts = []
    for b in BASINS:
        g = gpd.read_file(BASIN_DIR / f"Svalbard_Basin_{b}.shp")[["geometry"]].copy()
        g["basin"] = b
        parts.append(g.to_crs(epsg=EPSG_PROJ))       # normalise CRS defensively
    basins = pd.concat(parts, ignore_index=True)
    return gpd.GeoDataFrame(basins, crs=f"EPSG:{EPSG_PROJ}")


# --- Step 3: build glacier points (one row per CSV) ---------------------------
def load_points():
    """Return a GeoDataFrame of glacier terminus points (one per FrontChange CSV)."""
    rows = []
    for csv in sorted(FRONT_DIR.glob("*.csv")):
        r = pd.read_csv(csv, nrows=1).iloc[0]        # one unique coord per glacier
        rows.append({"gid": csv.stem, "lon": r["glacier_lon"], "lat": r["glacier_lat"]})
    pts = gpd.GeoDataFrame(
        rows,
        geometry=gpd.points_from_xy([x["lon"] for x in rows],
                                    [x["lat"] for x in rows]),
        crs=f"EPSG:{EPSG_PTS}",
    )
    return pts.to_crs(epsg=EPSG_PROJ)


# --- Step 4: containment join, then nearest-polygon fallback ------------------
def assign_basins(pts, basins):
    """Assign each point a `basin` by containment; fall back to nearest polygon.

    Returns (joined, fallback) where `joined` has one row per gid with a non-null
    `basin`, and `fallback` holds the rows that fell outside every polygon.
    """
    joined = gpd.sjoin(pts, basins, how="left", predicate="within").drop(
        columns="index_right")
    # dedupe in case a point lands on a shared edge / overlap -> keep first
    joined = joined[~joined["gid"].duplicated()].reset_index(drop=True)

    fallback = joined[joined["basin"].isna()].copy()
    for _, row in fallback.iterrows():
        dists = basins.distance(row.geometry)        # metres (projected CRS)
        nb = basins.loc[dists.idxmin(), "basin"]
        joined.loc[joined["gid"] == row["gid"], "basin"] = nb

    if len(fallback):
        print(f"WARNING: {len(fallback)} glacier(s) not inside any polygon "
              "-- assigned nearest:")
        for _, r in fallback.iterrows():
            nb = joined.loc[joined.gid == r.gid, "basin"].iloc[0]
            print(f"  - {r['gid']} -> {nb}")
    return joined, fallback


# --- Step 5: map basin -> current and write JSON ------------------------------
def write_lookup(joined, fallback):
    """Map each basin to ESC/WSC, then write the lookup JSON. Returns the dict."""
    joined["current"] = joined["basin"].apply(
        lambda b: "ESC" if b in ESC_BASINS else "WSC")
    subbasin = dict(zip(joined.gid, joined.basin))
    current  = dict(zip(joined.gid, joined.current))

    out = {
        "_comment": "Spitsbergen current per glacier, assigned by containment in "
                    "Noel et al. (2020) subbasin polygons (glacier_data/SubBasins). "
                    "ESC = {AF,BE}; WSC = {NW,NE,VF,SS}. Termini in open fjord water "
                    "fall back to the nearest polygon (see _fallback).",
        "_method": "point-in-polygon (EPSG:3995) of FrontChange terminus coords "
                   "against 6 subbasin shapefiles",
        "_fallback": sorted(fallback["gid"].tolist()),
        "subbasin": dict(sorted(subbasin.items())),
        "current":  dict(sorted(current.items())),
    }
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"Wrote {OUT_JSON}: {sum(v == 'ESC' for v in current.values())} ESC / "
          f"{sum(v == 'WSC' for v in current.values())} WSC over {len(current)} glaciers")
    return out


# --- Step 6: verification PNG -------------------------------------------------
def save_map(joined, fallback, basins):
    """Save a PNG: 6 labeled polygons + points colored by ESC/WSC, fallbacks ringed."""
    fig, ax = plt.subplots(figsize=(9, 9))
    basins.plot(ax=ax, facecolor="0.92", edgecolor="0.5", linewidth=0.6)
    for c, color in [("ESC", "C3"), ("WSC", "C0")]:
        sub = joined[joined.current == c]
        ax.scatter(sub.geometry.x, sub.geometry.y, s=28, c=color, edgecolor="w",
                   linewidth=0.4, label=f"{c} (n={len(sub)})", zorder=5)
    fb = joined[joined.gid.isin(fallback.gid)]
    ax.scatter(fb.geometry.x, fb.geometry.y, s=110, facecolors="none",
               edgecolors="k", linewidth=1.2, zorder=6, label="fallback (nearest)")
    for _, r in basins.iterrows():                   # basin labels at centroid
        ax.annotate(r.basin, (r.geometry.centroid.x, r.geometry.centroid.y),
                    ha="center", fontsize=9, color="0.3")
    ax.legend(loc="upper right")
    ax.set_axis_off()
    plt.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {OUT_PNG}")


if __name__ == "__main__":
    # Step 1 sanity check: every expected input path resolves.
    assert GLACIER_DATA.is_dir(), GLACIER_DATA
    assert FRONT_DIR.is_dir(), FRONT_DIR
    assert BASIN_DIR.is_dir(), BASIN_DIR
    missing = [b for b in BASINS if not (BASIN_DIR / f"Svalbard_Basin_{b}.shp").is_file()]
    assert not missing, f"missing shapefiles: {missing}"
    n_csv = len(list(FRONT_DIR.glob("*.csv")))
    print(f"ROOT          = {ROOT}")
    print(f"GLACIER_DATA  = {GLACIER_DATA}")
    print(f"basins found  = {len(BASINS) - len(missing)}/{len(BASINS)}")
    print(f"FrontChange   = {n_csv} CSVs")
    print("Step 1 OK: paths & config resolve.")

    # Step 2 sanity check: 6 polygons load, tagged, in the projected CRS.
    basins = load_basins()
    assert len(basins) == len(BASINS), basins
    assert sorted(basins["basin"]) == sorted(BASINS), basins["basin"].tolist()
    assert basins.crs.to_epsg() == EPSG_PROJ, basins.crs
    assert basins.geometry.notna().all() and basins.is_valid.all(), "invalid geometry"
    print(f"Step 2 OK: {len(basins)} basins loaded in EPSG:{EPSG_PROJ} "
          f"({', '.join(basins['basin'])}).")

    # Step 3 sanity check: one point per CSV, unique gids, in the projected CRS.
    pts = load_points()
    assert len(pts) == n_csv, (len(pts), n_csv)
    assert not pts["gid"].duplicated().any(), "duplicate gids"
    assert pts.crs.to_epsg() == EPSG_PROJ, pts.crs
    assert pts.geometry.notna().all(), "missing point geometry"
    print(f"Step 3 OK: {len(pts)} glacier points in EPSG:{EPSG_PROJ}.")

    # Step 4 sanity check: every glacier assigned exactly one basin.
    joined, fallback = assign_basins(pts, basins)
    assert len(joined) == len(pts), (len(joined), len(pts))
    assert not joined["gid"].duplicated().any(), "duplicate gids after join"
    assert joined["basin"].notna().all(), "unassigned basin after fallback"
    assert set(joined["basin"]) <= set(BASINS), set(joined["basin"])
    print(f"Step 4 OK: {len(joined)} glaciers assigned a basin "
          f"({len(fallback)} via nearest-polygon fallback).")

    # Step 5 sanity check: lookup written, contract intact (current in {ESC,WSC}).
    out = write_lookup(joined, fallback)
    assert set(out["current"].values()) <= {"ESC", "WSC"}, set(out["current"].values())
    assert len(out["current"]) == len(joined), (len(out["current"]), len(joined))
    assert set(out["current"]) == set(out["subbasin"]), "current/subbasin gids differ"
    assert OUT_JSON.is_file(), OUT_JSON
    print(f"Step 5 OK: wrote {OUT_JSON.name} for {len(out['current'])} glaciers.")

    # Step 6 sanity check: verification PNG written.
    save_map(joined, fallback, basins)
    assert OUT_PNG.is_file(), OUT_PNG
    print(f"Step 6 OK: wrote {OUT_PNG.name}.")