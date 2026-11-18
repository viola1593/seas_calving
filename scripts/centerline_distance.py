"""
Measure calving front distance from the centerline inland anchor per sample.

The reference point is coords[0] of the centerline geometry (inland anchor,
always inside the glacier).  The SDT value there gives how far the calving
front currently is from that fixed inland point (positive = front is farther
ocean-ward; decreasing = front is retreating toward the anchor).

true_distance_m  = sdt[target_zi,  row, col]                     (raw metres)
pred_distance_m  = sdt[anchor_zi, row, col] + pred[…] * disp_std (model forecast)

Usage (true-only):
    python scripts/centerline_distance.py \\
        --index  ../glacier_data/TrainingDataset_processedV2/splits_seasonal/val.json \\
                 ../glacier_data/TrainingDataset_processedV2/splits_seasonal/test.json \\
        --out    centerline_distance.parquet

With model predictions:
    python scripts/centerline_distance.py \\
        --index  … \\
        --out    centerline_distance.parquet \\
        --experiment experiments/seasonal/convlstm_filmpervar_v1_20260529_073112
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import date, timedelta
from pathlib import Path

import geopandas as gpd
import pandas as pd
import rasterio
import zarr
from shapely.ops import linemerge, unary_union

# Allow imports from the package root (evaluate_film, data.*) when the script
# is invoked as  python scripts/centerline_distance.py  from svalcalvingdb/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

CALVING_GPKG = "../glacier_data/Svalbard_ESSD_Calving_Product/Li_et_al_2023_Svalbard_v3/Svalbard_Calving_Front_Product.gpkg"
TRAINING_DIR = "../glacier_data/TrainingDataset_V2/TrainingDataset"


# ── Per-glacier caches ────────────────────────────────────────────────────────

_cache_transform: dict = {}
_cache_hw:        dict = {}
_cache_anchor:    dict = {}


def _open_zarr(path: str) -> zarr.Group:
    try:
        return zarr.open_group(path, mode="r")
    except Exception:
        return zarr.open_group("../" + path, mode="r")


def _get_anchor_pixel(
    glacier_id: str,
    centerlines: gpd.GeoDataFrame,
) -> tuple[int, int] | None:
    """Return (row, col) of the centerline inland anchor (coords[0])."""
    if glacier_id in _cache_anchor:
        return _cache_anchor[glacier_id]

    if glacier_id not in _cache_transform:
        masks = sorted(Path(TRAINING_DIR).glob(f"{glacier_id}/*/Glacier_MASK.tif"))
        if not masks:
            warnings.warn(f"No Glacier_MASK.tif for {glacier_id}")
            _cache_anchor[glacier_id] = None
            return None
        with rasterio.open(masks[0]) as src:
            _cache_transform[glacier_id] = src.transform
            _cache_hw[glacier_id] = (src.height, src.width)

    cl = centerlines[centerlines["Glacier"] == glacier_id]
    if cl.empty:
        warnings.warn(f"No centerline for {glacier_id}")
        _cache_anchor[glacier_id] = None
        return None

    geom = unary_union(cl.geometry)
    if geom.geom_type == "MultiLineString":
        geom = linemerge(geom)
    if geom.geom_type == "MultiLineString":
        geom = list(geom.geoms)[0]

    x, y = list(geom.coords)[0]  # inland anchor (coords[0] = glacier-interior end)
    H, W = _cache_hw[glacier_id]
    row, col = rasterio.transform.rowcol(_cache_transform[glacier_id], x, y)
    result = (int(max(0, min(row, H - 1))), int(max(0, min(col, W - 1))))
    _cache_anchor[glacier_id] = result
    return result


def _target_date_str(s: dict) -> str:
    if "target_date" in s:
        return s["target_date"]
    t_ref = date.fromisoformat(s["reference_date"])
    return (t_ref + timedelta(days=int(s["target_day"]))).isoformat()


# ── True-distance path ────────────────────────────────────────────────────────

def build_true_records(
    index_files: list[Path],
    centerlines: gpd.GeoDataFrame,
) -> list[dict]:
    records = []
    for idx_path in index_files:
        split = idx_path.stem
        with open(idx_path) as f:
            samples = json.load(f)
        print(f"  {split}: {len(samples)} samples")

        for i, s in enumerate(samples):
            if i % 500 == 0:
                print(f"    {i}/{len(samples)}")

            gid = s["glacier_id"]
            rec = {
                "glacier_id":      gid,
                "split":           split,
                "target_date":     _target_date_str(s),
                "horizon_days":    int(s.get("horizon_days", s.get("target_day", 0))),
                "true_distance_m": float("nan"),
                "pred_distance_m": float("nan"),
                "seasonal":        bool(s.get("seasonal", False)),
            }

            anchor_px = _get_anchor_pixel(gid, centerlines)
            if anchor_px is not None:
                row, col = anchor_px
                try:
                    z = _open_zarr(s["zarr_path"])
                    rec["true_distance_m"] = float(z["sdt"][s["target_zarr_index"], row, col])
                except Exception as e:
                    warnings.warn(f"zarr read error for {gid}: {e}")

            records.append(rec)
    return records


# ── Inference path ────────────────────────────────────────────────────────────

def _run_inference_split(
    split: str,
    cfg: dict,
    module,
    device,
    disp_std: float,
    centerlines: gpd.GeoDataFrame,
    pred_map: dict,
) -> None:
    import torch
    from tqdm import tqdm
    from data.datamodule import CalvingFrontDataModule

    dm = CalvingFrontDataModule(
        **{**cfg["data"], "batch_size": 4, "num_workers": 4, "augment_train": False}
    )
    dm.setup("test" if split == "test" else "fit")
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()
    print(f"    {split}: {len(loader.dataset)} samples")

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"  {split}", unit="batch"):
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            pred = module.model(batch)

            for b in range(pred.shape[0]):
                gid = batch["glacier_id"][b]
                key = (gid, batch["target_date"][b])
                if key not in pred_map:
                    continue

                anchor_px = _get_anchor_pixel(gid, centerlines)
                if anchor_px is None:
                    continue

                r, c = anchor_px
                orig_h, orig_w = batch["original_hw"][b]
                if r >= orig_h or c >= orig_w:
                    continue

                try:
                    z = _open_zarr(batch["zarr_path"][b])
                    anchor_sdt_m = float(z["sdt"][int(batch["last_input_zarr_index"][b]), r, c])
                except Exception:
                    continue

                pred_map[key]["pred_distance_m"] = (
                    anchor_sdt_m + float(pred[b, 0, r, c].item()) * disp_std
                )


def run_inference(
    index_files: list[Path],
    experiment_dir: Path,
    records: list[dict],
    centerlines: gpd.GeoDataFrame,
) -> None:
    import torch
    from evaluate_film import find_best_checkpoint, load_module_and_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = find_best_checkpoint(experiment_dir)
    module, cfg = load_module_and_config(ckpt, experiment_dir, device)
    disp_std = float(module.hparams.disp_std)

    pred_map = {(r["glacier_id"], r["target_date"]): r for r in records}

    for idx_path in index_files:
        split = idx_path.stem
        if split not in ("val", "test"):
            warnings.warn(f"Skipping inference for split '{split}' (val/test only)")
            continue
        _run_inference_split(split, cfg, module, device, disp_std, centerlines, pred_map)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",      nargs="+", type=Path, required=True,
                        help="Index JSON files (val.json, test.json, …)")
    parser.add_argument("--out",        type=Path,            required=True,
                        help="Output parquet path")
    parser.add_argument("--experiment", type=Path,            default=None,
                        help="Experiment directory for model predictions")
    args = parser.parse_args()

    print("Loading centerlines …")
    centerlines = gpd.read_file(CALVING_GPKG, layer="centerlines")

    print("Computing true distances …")
    records = build_true_records(args.index, centerlines)

    if args.experiment is not None:
        print("Running model inference …")
        run_inference(args.index, args.experiment, records, centerlines)

    df = pd.DataFrame(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(args.out, index=False)
    print(f"\nSaved {len(df)} rows → {args.out}")
    print(f"  Glaciers : {df['glacier_id'].nunique()}")
    print(f"  true NaN : {df['true_distance_m'].isna().sum()}")
    print(f"  pred NaN : {df['pred_distance_m'].isna().sum()}")


if __name__ == "__main__":
    main()
