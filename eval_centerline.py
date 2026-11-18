"""
Per-glacier calving front displacement along the centerline — end-to-end.

Runs inference from one or more experiment checkpoints and plots true vs.
predicted calving front displacement (metres along the centerline) for each
glacier. One PNG per glacier.

The y-axis shows displacement relative to t0 (the earliest target date for
that glacier in the evaluated split):
    positive = advance  (front moved ocean-ward)
    negative = retreat  (front moved inland)

For each sample the calving front position is measured by:
  1. Densifying the centerline geometry to one point per pixel (~30 m).
  2. Walking the ordered pixel list (inland → ocean) over the SDT map.
  3. Finding where the SDT sign changes from + to − (inside → outside glacier).
  4. Computing arc length from the inland anchor to that crossing using
     Shapely's LineString.project(), in metres.
  Fallback: if no sign change is found, use the pixel of minimum |SDT|.

This is applied to three SDT maps per sample:
  true        z["sdt"][target_zarr_index]
  predicted   reconstructed from model output: (pred/disp_scale + last_norm)*sdt_std + sdt_mean
  persistence z["sdt"][last_input_zarr_index]   (last observed front = 0-displacement baseline)

Plots are written to <experiment>/centerlines/ (the first experiment when
comparing several).

Usage:
    python scripts/eval_centerline.py \\
        --experiment experiments/seasonal/convlstm_filmpervar_v1_20260529_073112 \\
        --split      test

Compare two models:
    python scripts/eval_centerline.py \\
        --experiment experiments/seasonal/exp_A  experiments/seasonal/exp_B \\
        --persistence

Single glacier preview:
    python scripts/eval_centerline.py \\
        --experiment experiments/seasonal/exp_A \\
        --glacier    RGI60-07.00470
"""
from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
import torch
import zarr
from shapely.geometry import Point
from shapely.ops import linemerge, unary_union
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CALVING_GPKG = "../glacier_data/Svalbard_ESSD_Calving_Product/Li_et_al_2023_Svalbard_v3/Svalbard_Calving_Front_Product.gpkg"
TRAINING_DIR = "../glacier_data/TrainingDataset_V2/TrainingDataset"

_MODEL_COLORS = ["lightcoral", "mediumseagreen", "orchid", "firebrick", "gold"]
FS = 14  # font size for plots
plt.rcParams.update({
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
})
# Pixel spacing used for densifying the centerline (metres). Should match the
# raster resolution so every pixel along the centerline gets one sample.
_DENSIFY_STEP_M = 30.0

# Minimum |true − persistence| displacement (m) for migration direction to be
# considered defined. Below this the front is effectively stationary and no
# direction marker is drawn. ~1 pixel; bump to 60 if 30 m is within noise.
_DIRECTION_DEADBAND_M = 30.0

_cache_transform: dict = {}
_cache_hw:        dict = {}
_cache_centerline: dict = {}   # glacier_id → shapely LineString in CRS coords
_cache_cl_pixels:  dict = {}   # glacier_id → list[(row, col, arc_m)]
_zarr_stores:     dict = {}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _open_zarr(path: str) -> zarr.Group:
    if path not in _zarr_stores:
        try:
            _zarr_stores[path] = zarr.open_group(path, mode="r")
        except Exception:
            _zarr_stores[path] = zarr.open_group("../" + path, mode="r")
    return _zarr_stores[path]


def _get_raster_meta(glacier_id: str) -> tuple[rasterio.Affine, int, int] | None:
    """Return (transform, H, W) for the glacier raster, cached."""
    if glacier_id not in _cache_transform:
        masks = sorted(Path(TRAINING_DIR).glob(f"{glacier_id}/*/Glacier_MASK.tif"))
        if not masks:
            warnings.warn(f"No Glacier_MASK.tif for {glacier_id}")
            _cache_transform[glacier_id] = None
            return None
        with rasterio.open(masks[0]) as src:
            _cache_transform[glacier_id] = src.transform
            _cache_hw[glacier_id] = (src.height, src.width)
    if _cache_transform[glacier_id] is None:
        return None
    return _cache_transform[glacier_id], *_cache_hw[glacier_id]


def _get_centerline(glacier_id: str, centerlines: gpd.GeoDataFrame):
    """Return a Shapely LineString for the glacier centerline, cached.

    coords[0] is always the inland anchor (glacier-interior end).
    """
    if glacier_id in _cache_centerline:
        return _cache_centerline[glacier_id]

    cl = centerlines[centerlines["Glacier"] == glacier_id]
    if cl.empty:
        warnings.warn(f"No centerline for {glacier_id}")
        _cache_centerline[glacier_id] = None
        return None

    geom = unary_union(cl.geometry)
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    if geom.geom_type == "MultiLineString":
        geom = list(geom.geoms)[0]

    _cache_centerline[glacier_id] = geom
    return geom


def _get_cl_pixels(
    glacier_id: str,
    centerlines: gpd.GeoDataFrame,
) -> list[tuple[int, int, float]] | None:
    """Return cached list of (row, col, arc_m) for the densified centerline."""
    if glacier_id in _cache_cl_pixels:
        return _cache_cl_pixels[glacier_id]

    meta = _get_raster_meta(glacier_id)
    if meta is None:
        _cache_cl_pixels[glacier_id] = None
        return None
    transform, H, W = meta

    line = _get_centerline(glacier_id, centerlines)
    if line is None:
        _cache_cl_pixels[glacier_id] = None
        return None

    distances = np.arange(0.0, line.length, _DENSIFY_STEP_M)
    if len(distances) == 0:
        _cache_cl_pixels[glacier_id] = None
        return None

    pixels: list[tuple[int, int, float]] = []
    prev_rc = None
    for d in distances:
        pt = line.interpolate(d)
        row, col = rasterio.transform.rowcol(transform, pt.x, pt.y)
        row = int(max(0, min(row, H - 1)))
        col = int(max(0, min(col, W - 1)))
        rc = (row, col)
        if rc == prev_rc:
            continue
        pixels.append((row, col, d))
        prev_rc = rc

    result = pixels if pixels else None
    _cache_cl_pixels[glacier_id] = result
    return result


def _centerline_crossing(
    sdt_map: np.ndarray,
    glacier_id: str,
    centerlines: gpd.GeoDataFrame,
) -> tuple[float | None, tuple[int, int] | None, bool]:
    """Return (arc_m, (row, col), fallback_used) for the SDT zero-crossing."""
    pixels = _get_cl_pixels(glacier_id, centerlines)
    if not pixels:
        return None, None, False

    sdt_vals = [sdt_map[r, c] for r, c, _ in pixels]

    # Search backward: last + → − crossing is the calving front (ocean-most).
    # Searching forward would find a spurious crossing at the glacier-land
    # boundary first for centerlines that wander outside the glacier upstream.
    for i in range(len(sdt_vals) - 2, -1, -1):
        v0, v1 = sdt_vals[i], sdt_vals[i + 1]
        if v0 >= 0 and v1 < 0:
            frac = v0 / (v0 - v1)
            arc = pixels[i][2] + frac * (pixels[i + 1][2] - pixels[i][2])
            r = round(pixels[i][0] + frac * (pixels[i + 1][0] - pixels[i][0]))
            c = round(pixels[i][1] + frac * (pixels[i + 1][1] - pixels[i][1]))
            return arc, (r, c), False

    abs_vals = np.abs(sdt_vals)
    idx = int(np.argmin(abs_vals))
    return pixels[idx][2], (pixels[idx][0], pixels[idx][1]), True


def _centerline_arc_length(
    sdt_map: np.ndarray,
    glacier_id: str,
    centerlines: gpd.GeoDataFrame,
) -> float | None:
    """
    Return the arc length (metres) from the inland anchor to where the
    centerline crosses the calving front (SDT = 0) in sdt_map.

    sdt_map: 2-D float array in metres (H, W), already cropped to original_hw.
    """
    arc, _, _ = _centerline_crossing(sdt_map, glacier_id, centerlines)
    return arc


# ── Inference ─────────────────────────────────────────────────────────────────

def _experiment_label(exp_dir: Path) -> str:
    return re.sub(r"_\d{8}_\d{6}$", "", exp_dir.name)


def run_inference(
    exp_dir: Path,
    splits: list[str],
    centerlines: gpd.GeoDataFrame,
    device: torch.device,
    glacier_filter: set[str] | None,
    debug: bool = False,
) -> tuple[list[dict], dict[str, dict]]:
    from evaluate_film import find_best_checkpoint, load_module_and_config
    from data.datamodule import CalvingFrontDataModule

    ckpt = find_best_checkpoint(exp_dir)
    module, cfg = load_module_and_config(ckpt, exp_dir, device)

    disp_scale = module.model.disp_scale.item()
    disp_std   = float(module.hparams.disp_std)
    sdt_std    = disp_scale * disp_std
    sdt_mean   = float(module.hparams.sdt_mean)

    records = []
    debug_best: dict[str, dict] = {}
    for split in splits:
        dm = CalvingFrontDataModule(
            **{**cfg["data"], "batch_size": 4, "num_workers": 4, "augment_train": False}
        )
        dm.setup("test" if split == "test" else "fit")
        if split == "train":
            loader = dm.train_dataloader()
        else:
            loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"  {split}", unit="batch"):
                batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                pred = module.model(batch)  # (B, 1, H, W) displacement norm

                # Reconstruct predicted SDT map (normalised → metres)
                last_norm  = batch["inputs"][:, -1, 0:1]          # (B, 1, H, W)
                pred_sdt_n = pred / disp_scale + last_norm         # (B, 1, H, W)
                pred_sdt_m = (pred_sdt_n[:, 0] * sdt_std + sdt_mean).cpu().numpy()

                for b in range(pred.shape[0]):
                    gid = batch["glacier_id"][b]
                    if glacier_filter is not None and gid not in glacier_filter:
                        continue

                    orig_h, orig_w = batch["original_hw"][b]

                    try:
                        z = _open_zarr(batch["zarr_path"][b])
                        true_sdt_map    = np.array(z["sdt"][int(batch["target_zarr_index"][b])])
                        persist_sdt_map = np.array(z["sdt"][int(batch["last_input_zarr_index"][b])])
                    except Exception as e:
                        warnings.warn(f"zarr read error for {gid}: {e}")
                        continue

                    true_sdt_map    = true_sdt_map[:orig_h, :orig_w]
                    persist_sdt_map = persist_sdt_map[:orig_h, :orig_w]
                    pred_sdt_crop   = pred_sdt_m[b, :orig_h, :orig_w]

                    tdate = batch["target_date"][b]

                    true_arc,    true_rc,    true_fb    = _centerline_crossing(true_sdt_map,    gid, centerlines)
                    pred_arc,    pred_rc,    pred_fb    = _centerline_crossing(pred_sdt_crop,   gid, centerlines)
                    persist_arc, _,          persist_fb = _centerline_crossing(persist_sdt_map, gid, centerlines)

                    if true_fb:    print(f"[centerline fallback] {gid}  true     target={tdate}")
                    if pred_fb:    print(f"[centerline fallback] {gid}  pred     target={tdate}")
                    if persist_fb: print(f"[centerline fallback] {gid}  persist  target={tdate}")

                    if true_arc is None:
                        continue

                    if debug:
                        if gid not in debug_best or tdate < debug_best[gid]["target_date"]:
                            debug_best[gid] = {
                                "target_date":   tdate,
                                "true_sdt":      true_sdt_map.copy(),
                                "pred_sdt":      pred_sdt_crop.copy(),
                                "cl_pixels":     _get_cl_pixels(gid, centerlines),
                                "true_rc":       true_rc,
                                "pred_rc":       pred_rc,
                                "true_fallback": true_fb,
                                "pred_fallback": pred_fb,
                            }

                    records.append({
                        "glacier_id":         gid,
                        "split":              split,
                        "target_date":        batch["target_date"][b],
                        "true_arc_m":         true_arc,
                        "pred_arc_m":         pred_arc,
                        "persist_arc_m":      persist_arc,
                        "true_fallback":      true_fb,
                        "pred_fallback":      pred_fb,
                        "persist_fallback":   persist_fb,
                    })

    return records, debug_best


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_direction_band(ax, df, y_frac):
    """Mark migration-direction errors. Where the predicted direction (vs
    persistence) disagrees with the true direction, underlay a semi-transparent
    red vertical line behind the datapoint. Where the true migration is below
    the deadband (effectively stationary, direction undefined), draw a black
    circle at the bottom of the axis. df is one model's series.

    Returns (n_correct, n_defined) over the samples with a defined direction."""
    trans = ax.get_xaxis_transform()
    n_wrong = 0
    n_defined = 0
    for _, r in df.iterrows():
        if pd.isna(r.pred_arc_m) or pd.isna(r.persist_arc_m) or pd.isna(r.true_arc_m):
            continue
        true_move = r.true_arc_m - r.persist_arc_m
        pred_move = r.pred_arc_m - r.persist_arc_m

        if abs(true_move) < _DIRECTION_DEADBAND_M:
            # True front is stationary — direction undefined. Mark with a black
            # circle at the bottom of the axis.
            ax.scatter(r.target_date, y_frac, transform=trans, marker="o",
                       s=25, facecolors="none", edgecolors="black",
                       linewidths=0.8, zorder=6, clip_on=False)
            continue

        n_defined += 1
        if np.sign(pred_move) != np.sign(true_move):
            # Wrong predicted direction — underlay a semi-transparent red line.
            ax.axvline(r.target_date, color="red", alpha=0.20,
                       linewidth=4, zorder=0)
            n_wrong += 1

    print(f"[wrong direction] {df.glacier_id.iloc[0]}  "
          f"{n_wrong}/{len(df)} = {n_wrong / len(df):.2f}")
    return n_defined - n_wrong, n_defined


def plot_glacier(
    gid: str,
    model_series: list[tuple[str, pd.DataFrame]],
    out_path: Path,
    show_persistence: bool,
) -> None:
    # t0 reference: minimum target_date across all series (true arc)
    first_df  = model_series[0][1].sort_values("target_date")
    t0_arc    = first_df["true_arc_m"].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 4))

    # True front (same across all model series — from the first one)
    true_disp = first_df["true_arc_m"] - t0_arc
    ax.plot(first_df["target_date"], true_disp,
            color="darkmagenta", linewidth=1.4, marker="o", markersize=3, label="True")

    if show_persistence:
        persist_disp = first_df["persist_arc_m"] - t0_arc
        valid = first_df["true_arc_m"].notna() & first_df["persist_arc_m"].notna()
        persist_mae = np.abs(first_df.loc[valid, "true_arc_m"] - first_df.loc[valid, "persist_arc_m"]).mean()
        persist_label = f"Last input (MAD: {persist_mae:.0f} m)" if not np.isnan(persist_mae) else "Persistence"
        ax.plot(first_df["target_date"], persist_disp,
                color="grey", linewidth=1.0, linestyle="--", marker="", label=persist_label)

    dir_stats: list[tuple[str, int, int]] = []
    for i, (label, df) in enumerate(model_series):
        df = df.sort_values("target_date")
        pred_disp = df["pred_arc_m"] - t0_arc
        color = _MODEL_COLORS[i % len(_MODEL_COLORS)]
        merged = first_df[["target_date", "true_arc_m"]].merge(
            df[["target_date", "pred_arc_m"]], on="target_date"
        ).sort_values("target_date")
        mae = np.abs(merged["true_arc_m"] - merged["pred_arc_m"]).mean() if not merged.empty else float("nan")
        plot_label = f"Prediction (MAD: {mae:.0f} m)" if not np.isnan(mae) else label
        ax.plot(df["target_date"], pred_disp,
                color=color, linewidth=1.2, marker="s", markersize=3, label=plot_label)

        n_correct, n_defined = _draw_direction_band(ax, df, y_frac=0.015 + 0.035 * i)
        dir_stats.append((label, n_correct, n_defined))

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.4)
    dir_parts = []
    for label, n_correct, n_defined in dir_stats:
        pct = f"{100 * n_correct / n_defined:.0f}%" if n_defined else "n/a"
        prefix = f"{label} " if len(dir_stats) > 1 else ""
        dir_parts.append(f"{prefix}correct direction: {pct} ({n_correct}/{n_defined})")
    title = gid + ("   —   " + ",  ".join(dir_parts) if dir_parts else "")
    ax.set_title(title, fontsize=FS)
    ax.set_xlabel("Target date", fontsize=FS)
    ax.set_ylabel("Front displacement from t₀ (m)\n← retreat   advance →", fontsize=FS)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.margins(y=0.12)  # room at bottom for the direction band
    fig.autofmt_xdate(rotation=30)
    model_legend = ax.legend(fontsize=FS, loc="upper right")
    ax.add_artist(model_legend)
    dir_handles = [
        Line2D([0], [0], color="red", alpha=0.20, linewidth=4,
               label="wrong direction"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="none",
               markeredgecolor="black", markersize=6,
               label="no motion (<30 m)"),
    ]
    ax.legend(handles=dir_handles, fontsize=FS-2, loc="lower left",
              framealpha=0.7, ncol=2, handletextpad=0.3, columnspacing=1.0)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_debug_crossings(
    gid: str,
    info: dict,
    out_path: Path,
) -> None:
    """Side-by-side true / predicted SDT with centerline + crossing markers."""
    true_sdt  = info["true_sdt"]
    pred_sdt  = info["pred_sdt"]
    cl_pixels = info["cl_pixels"]   # list[(row, col, arc_m)] or None
    true_rc   = info["true_rc"]
    pred_rc   = info["pred_rc"]
    true_fb   = info["true_fallback"]
    pred_fb   = info["pred_fallback"]

    vlim = float(np.percentile(np.abs(true_sdt), 98))
    vlim = max(vlim, 100.0)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, sdt, title in [
        (axes[0], true_sdt,  f"True SDT  [{info['target_date']}]"),
        (axes[1], pred_sdt,  "Predicted SDT"),
    ]:
        im = ax.imshow(sdt, cmap="RdBu", vmin=-vlim, vmax=vlim, origin="upper")
        plt.colorbar(im, ax=ax, label="SDT (m)", fraction=0.046, pad=0.04)
        ax.set_title(title, fontsize=FS)
        ax.set_xlabel("col", fontsize=FS)
        ax.set_ylabel("row", fontsize=FS)

        if cl_pixels:
            rows = [p[0] for p in cl_pixels]
            cols = [p[1] for p in cl_pixels]
            ax.plot(cols, rows, color="black", linewidth=0.8, alpha=0.6, label="Centerline")

        if true_rc is not None:
            marker = "x" if true_fb else "o"
            label  = "True (fallback)" if true_fb else "True crossing"
            ax.scatter(true_rc[1], true_rc[0], marker=marker, c="steelblue",
                       s=100, zorder=5, linewidths=2, label=label)

        if pred_rc is not None:
            marker = "x" if pred_fb else "s"
            label  = "Pred (fallback)" if pred_fb else "Pred crossing"
            ax.scatter(pred_rc[1], pred_rc[0], marker=marker, c="darkorange",
                       s=100, zorder=5, linewidths=2, label=label)

        ax.legend(fontsize=FS, loc="upper right")

    fig.suptitle(gid, fontsize=FS)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", nargs="+", type=Path, required=True,
                        help="One or more experiment directories")
    parser.add_argument("--split", choices=["val", "test", "train"], default="test")
    parser.add_argument("--persistence", action="store_true",
                        help="Also plot the persistence baseline")
    parser.add_argument("--glacier", nargs="*", default=None,
                        help="Optional: restrict to these glacier IDs")
    parser.add_argument("--device", type=str, default="cuda",
                        help="'cuda', 'cpu', or 'cuda:N' (default: cuda)")
    parser.add_argument("--debug-crossings", action="store_true",
                        help="Plot SDT + centerline + crossing points for the first test sample per glacier")
    args = parser.parse_args()

    splits = ["val", "test", "train"] if args.split == "both" else [args.split]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    glacier_filter = set(args.glacier) if args.glacier else None

    print("Loading centerlines …")
    centerlines = gpd.read_file(CALVING_GPKG, layer="centerlines")

    all_results: dict[str, pd.DataFrame] = {}
    debug_info: dict[str, dict] = {}
    for exp_dir in args.experiment:
        label = _experiment_label(exp_dir)
        print(f"\nRunning inference: {label}")
        records, exp_debug = run_inference(
            exp_dir, splits, centerlines, device, glacier_filter,
            debug=args.debug_crossings,
        )
        if not debug_info:
            debug_info = exp_debug
        df = pd.DataFrame(records)
        df["target_date"] = pd.to_datetime(df["target_date"])
        df = df.sort_values("target_date")
        all_results[label] = df
        print(f"  {len(df)} samples, {df['glacier_id'].nunique()} glaciers")

    all_glaciers = sorted(
        set().union(*[set(df["glacier_id"]) for df in all_results.values()])
    )

    out_dir = args.experiment[0] / "centerlines"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlotting {len(all_glaciers)} glaciers → {out_dir}")

    for gid in all_glaciers:
        model_series = [
            (label, df[df["glacier_id"] == gid].copy())
            for label, df in all_results.items()
            if (df["glacier_id"] == gid).any()
        ]
        plot_glacier(gid, model_series, out_dir / f"{gid}.png", args.persistence)

    if args.debug_crossings and debug_info:
        print(f"\nPlotting crossing debug for {len(debug_info)} glaciers → {out_dir}")
        for gid in sorted(debug_info):
            plot_debug_crossings(gid, debug_info[gid], out_dir / f"{gid}_crossings.png")

    print("Done.")


if __name__ == "__main__":
    main()
