# preprocessing/02_seasonal_observations.py

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import zarr
import rasterio
from scipy.ndimage import distance_transform_edt
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

INVENTORY_PATH = Path(
    "../glacier_data/TrainingDataset_processedV2"
    "/inventory.parquet"
)
SEASONAL_LABELS_PATH = Path("../glacier_data/Merged_Seasonality_DecadalRates.csv")
OUT_DIR = Path(
    "../glacier_data/TrainingDataset_processedV2"
    "/zarr"
 
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

SDT_CLIP_M  = 2000.0
D_GEOPHYS   = 5     # bed, surface elev, thickness, velocity, strain rate
D_CLIMATE   = 2     # MAR runoff, ocean temperature

SENSOR_CODES = {"S2": 0, "LE07": 1, "LC08": 2, "LC09": 3, "S1A": 4, "S1B": 5}

STATIC_FILES = {
    "bed_elevation":     "Bed_Elevation.tif",
    "strain_rate":       "Eff_Strain_Rate.tif",
    "ice_fjord_mask":    "Ice_Fjord_Mask.tif",
    "mar_runoff":        "MAR_Runoff.tif",
    "ocean_temp":        "Ocean_Temp.tif",
    "surface_elevation": "Surface_Elevation.tif",
    "thickness":         "Thickness.tif",
    "trace_mask":        "Trace_MASK.tif",
    "glacier_mask":      "Glacier_MASK.tif",
    "velocity_vx":       "Velocity_VX.tif",
    "velocity_vy":       "Velocity_VY.tif",
    "velocity":          "Velocity.tif",
}

# Fixed channel order for geophys array — index position = channel number
GEOPHYS_CHANNEL_KEYS = [
    "bed_elevation",      # ch0
    "surface_elevation",  # ch1
    "thickness",          # ch2
    "velocity",           # ch3
    "strain_rate",        # ch4
]


# ── Filename helpers ──────────────────────────────────────────────────────────

def get_static_path(image_dir: Path, key: str) -> Path:
    return image_dir / STATIC_FILES[key]


def load_grid_meta(image_dir: Path) -> dict | None:
    """
    Read raster geometry from Glacier_MASK.tif.

    Returns a dict with height, width, and resolution_m, or None if the
    mask is missing or unreadable.
    """
    mask_path = get_static_path(image_dir, "glacier_mask")
    if not mask_path.exists():
        return None

    try:
        with rasterio.open(mask_path) as src:
            return {
                "height": src.height,
                "width": src.width,
                "resolution_m": float(abs(src.res[0])),
            }
    except Exception as e:
        print(f"    grid metadata error in {image_dir}: {e}")
        return None

def load_trace_mask(image_dir: Path, shape: tuple[int, int]) -> np.ndarray | None:
    """
    Load Trace_MASK.tif as a boolean array.
    Returns (H, W) bool with True on calving front pixels, or None if missing.
    The trace contains only the calving front — no fjord boundary pixels.
    """
    path = get_static_path(image_dir, "trace_mask")
    if not path.exists():
        return None

    try:
        with rasterio.open(path) as src:
            raw = src.read(1)
        return (raw > 0).astype(bool)
    except Exception as e:
        print(f"    trace_mask error in {image_dir}: {e}")
        return None


# ── SDT from glacier polygon mask ─────────────────────────────────────────────

def compute_sdt(image_dir: Path, resolution_m: float) -> np.ndarray | None:
    """
    Compute the signed distance transform from Glacier_MASK.tif.

    Glacier_MASK.tif is a binary raster:
      1 = glacier ice body
      0 = ocean / non-glacier

    SDT convention:
      SDT > 0   inside the glacier
      SDT = 0   on the calving front boundary (zero-level set)
      SDT < 0   in the ocean / proglacial area

    Method:
      positive_dist = distance_transform_edt(glacier_mask)
      negative_dist = distance_transform_edt(~glacier_mask)
      SDT = (positive_dist − negative_dist) × resolution_m

    No flood fill or seed points required — the mask sign is already encoded.
    Returns float32 (H, W) clipped to ±SDT_CLIP_M, or None on failure.
    """
    mask_path = get_static_path(image_dir, "glacier_mask")
    if not mask_path.exists():
        return None

    try:
        with rasterio.open(mask_path) as src:
            raw = src.read(1)

        glacier_mask = (raw > 0).astype(np.uint8)

        if glacier_mask.all() or not glacier_mask.any():
            return None   # no boundary — entirely ice or entirely ocean

        dist_inside  = distance_transform_edt(glacier_mask).astype(np.float32)
        dist_outside = distance_transform_edt(
            ~glacier_mask.astype(bool)
        ).astype(np.float32)

        sdt = (dist_inside - dist_outside) * resolution_m
        sdt = np.clip(sdt, -SDT_CLIP_M, SDT_CLIP_M).astype(np.float32)
        return sdt

    except Exception as e:
        print(f"    SDT error in {image_dir}: {e}")
        return None


# ── Geophysical channels ──────────────────────────────────────────────────────

def load_static_raster(path: Path, shape: tuple[int, int]) -> np.ndarray:
    """
    Load a single-band GeoTIFF as float32.
    Returns a zero array of the given shape when the file is absent.
    """
    H, W = shape
    if not path.exists():
        return np.zeros((H, W), dtype=np.float32)

    with rasterio.open(path) as src:
        data   = src.read(1).astype(np.float32)
        nodata = src.nodata

    if nodata is not None:
        data[data == nodata] = 0.0

    return data


def compute_geophys(image_dir: Path, shape: tuple[int, int]) -> np.ndarray:
    """
    Load five geophysical channels in the fixed order defined by
    GEOPHYS_CHANNEL_KEYS:

      ch0 bed elevation       (m a.s.l.)   Bed_Elevation.tif
      ch1 surface elevation   (m a.s.l.)   Surface_Elevation.tif
      ch2 ice thickness       (m)           Thickness.tif
      ch3 velocity magnitude  (m yr⁻¹)     Velocity.tif
      ch4 effective strain rate (yr⁻¹)     Eff_Strain_Rate.tif

    Missing files produce zero-filled channels.
    Returns (5, H, W) float32. Never returns None.
    """
    channels = [
        load_static_raster(get_static_path(image_dir, key), shape)
        for key in GEOPHYS_CHANNEL_KEYS
    ]
    return np.stack(channels, axis=0).astype(np.float32)


# ── Climate scalars ───────────────────────────────────────────────────────────

def extract_climate_scalars(image_dir: Path) -> np.ndarray | None:
    """
    Spatially average MAR_Runoff.tif and Ocean_Temp.tif to scalars.

      scalar 0: MAR runoff      (mm w.e.)
      scalar 1: ocean temperature (°C)

    Returns (2,) float32, or None when both files are absent.
    """
    values    = []
    any_found = False

    for key in ["mar_runoff", "ocean_temp"]:
        path = get_static_path(image_dir, key)
        if path.exists():
            with rasterio.open(path) as src:
                data   = src.read(1).astype(np.float32)
                nodata = src.nodata
            if nodata is not None:
                data[data == nodata] = np.nan
            valid = data[np.isfinite(data)]
            values.append(float(valid.mean()) if len(valid) > 0 else 0.0)
            any_found = True
        else:
            values.append(0.0)

    return np.array(values, dtype=np.float32) if any_found else None


# ── Phase A: pre-allocate zarr stores (serial) ────────────────────────────────

def preallocate_zarr_stores(inventory: pd.DataFrame, seasonal_labels: dict) -> None:
    """
    Create one zarr store per glacier with all arrays pre-allocated.
    Must complete before any parallel processing begins.

    Array layout:
      trace      (T, H, W)         bool      calving front pixels (evaluation only)
      sdt        (T, H, W)         float32   signed distance maps, metres
      geophys    (T, 5, H, W)      float32   see GEOPHYS_CHANNEL_KEYS
      climate    (T, 2)            float32   MAR runoff, ocean temperature
      dates      (T,)              str       ISO date strings "YYYY-MM-DD"
      sensors    (T,)              uint8     sensor code
    valid_trace (T,)             bool      trace raster loaded successfully
      valid_sdt  (T,)              bool      SDT computed successfully
      valid_clm  (T,)              bool      at least one climate file found
    """
    for glacier_id, gdf in inventory.groupby("glacier_id"):
        store_path = OUT_DIR / f"{glacier_id}.zarr"

        required_arrays = {
            "sdt", "geophys", "climate",
            "dates", "sensors", "valid_sdt", "valid_clm", "trace", "valid_trace"
        }

        if store_path.exists():
            try:
                existing_store = zarr.open_group(str(store_path), mode="r")
                existing_arrays = set(existing_store.array_keys())
                missing = sorted(required_arrays - existing_arrays)

                chunking_ok = False
                if not missing:
                    chunking_ok = (
                        existing_store["trace"].chunks[0] == 1
                        and existing_store["sdt"].chunks[0] == 1
                        and existing_store["geophys"].chunks[0] == 1
                        and existing_store["climate"].chunks[0] == 1
                        and existing_store["dates"].chunks[0] == 1
                        and existing_store["sensors"].chunks[0] == 1
                        and existing_store["valid_trace"].chunks[0] == 1
                        and existing_store["valid_sdt"].chunks[0] == 1
                        and existing_store["valid_clm"].chunks[0] == 1
                    )

                if not missing and chunking_ok:
                    print(f"  {glacier_id}: zarr exists, schema ok, skipping pre-allocation")
                    continue
                print(
                    f"  {glacier_id}: zarr schema/chunk mismatch (missing {missing}) "
                    "— rebuilding"
                )
                shutil.rmtree(store_path)
            except Exception as e:
                print(f"  {glacier_id}: failed to validate existing zarr ({e}) — rebuilding")
                shutil.rmtree(store_path, ignore_errors=True)

        first_row = gdf.sort_values("date").iloc[0]
        grid = load_grid_meta(Path(first_row["image_dir"]))
        if grid is None:
            print(f"  {glacier_id}: no grid metadata — skipping")
            continue

        T = len(gdf)
        H = grid["height"]
        W = grid["width"]

        store = zarr.open_group(str(store_path), mode="w")

        store.create_dataset(
            "trace",
            shape=(T, H, W),
            chunks=(1, H, W),
            dtype=bool,
            fill_value=False,
        )

        store.create_dataset(
            "sdt",
            shape=(T, H, W),
            chunks=(1, H, W),
            dtype="f4",
            fill_value=0.0,
        )
        store.create_dataset(
            "geophys",
            shape=(T, D_GEOPHYS, H, W),
            chunks=(1, D_GEOPHYS, H, W),
            dtype="f4",
            fill_value=0.0,
        )
        store.create_dataset(
            "climate",
            shape=(T, D_CLIMATE),
            chunks=(1, D_CLIMATE),
            dtype="f4",
            fill_value=0.0,
        )
        store.create_dataset(
            "dates",
            shape=(T,),
            chunks=(1,),
            dtype="U10",            # fixed-length 10-char unicode for "YYYY-MM-DD"
            fill_value="",
        )
        store.create_dataset(
            "sensors",
            shape=(T,),
            chunks=(1,),
            dtype="u1",
            fill_value=255,
        )
        store.create_dataset(
            "valid_trace",
            shape=(T,),
            chunks=(1,),
            dtype=bool,
            fill_value=False,
        )
        store.create_dataset(
            "valid_sdt",
            shape=(T,),
            chunks=(1,),
            dtype=bool,
            fill_value=False,
        )
        store.create_dataset(
            "valid_clm",
            shape=(T,),
            chunks=(1,),
            dtype=bool,
            fill_value=False,
        )

        store.attrs.update({
            "glacier_id":       glacier_id,
            "T":                T,
            "H":                H,
            "W":                W,
            "resolution_m":     grid["resolution_m"],
            "crs":              "EPSG:3995",
            "sdt_clip_m":       SDT_CLIP_M,
            "dates_format":     "YYYY-MM-DD",          # documents the string format
            "geophys_channels": GEOPHYS_CHANNEL_KEYS,
            "climate_channels": ["mar_runoff", "ocean_temp"],
            "seasonal":         bool(seasonal_labels.get(glacier_id, False)),
            "trace_description": (
                "Rasterised calving front line from Trace_MASK.tif. "
                "True pixels lie on the calving front only — no fjord boundary. "
                "Used for evaluation metrics only, not as a model input."
            ),
        })

        print(f"  {glacier_id}: pre-allocated  T={T}  H={H}  W={W}")


# ── Phase B: process one observation (parallel worker) ────────────────────────

def process_observation(args: tuple) -> dict:
    """
    Process a single image_id folder and write all arrays directly into
    the pre-allocated zarr store at the correct time index.

    Chunk shape is (1, …) on the time axis so each time index maps to its
    own on-disk chunk — concurrent workers writing to different indices
    cannot conflict.

    If valid_sdt is already True for this time index, the observation was
    already processed — returns cached status without rewriting.
    """
    (glacier_id, image_id, image_dir_str,
     time_index, date_val, sensor,
     grid_meta, zarr_path_str) = args

    image_dir    = Path(image_dir_str)
    resolution_m = grid_meta["resolution_m"]
    H            = grid_meta["height"]
    W            = grid_meta["width"]
    shape        = (H, W)

    status = {
        "glacier_id": glacier_id,
        "image_id":   image_id,
        "time_index": time_index,
        "trace_ok":   False,
        "sdt_ok":     False,
        "clm_ok":     False,
        "error":      None,
    }

    try:
        store = zarr.open_group(zarr_path_str, mode="r+")

        if bool(store["valid_sdt"][time_index]):
            status["trace_ok"] = bool(store["valid_trace"][time_index])
            status["sdt_ok"]   = True
            status["clm_ok"]   = bool(store["valid_clm"][time_index])
            return status

        # ── Date (ISO string) and sensor code ─────────────────────────────
        # date_val is a datetime.date object from the inventory parquet.
        # isoformat() produces "YYYY-MM-DD" — exact, timezone-naive, unambiguous.
        
        store["dates"][time_index]   = date_val.isoformat()
        store["sensors"][time_index] = SENSOR_CODES.get(sensor, 255)

        # ── Calving front trace (evaluation ground truth) ─────────────────────
        # Stored separately from SDT — used only for metric computation,
        # never as a model input or output.
        trace = load_trace_mask(image_dir, shape)
        if trace is not None:
            store["trace"][time_index]       = trace
            store["valid_trace"][time_index] = True
            status["trace_ok"]               = True

        # ── SDT from Glacier_MASK.tif ─────────────────────────────────────
        sdt = compute_sdt(image_dir, resolution_m)
        if sdt is not None:
            store["sdt"][time_index]       = sdt
            store["valid_sdt"][time_index] = True
            status["sdt_ok"]               = True

        # ── Geophysical channels (always written, missing → zeros) ────────
        store["geophys"][time_index] = compute_geophys(image_dir, shape)

        # ── Climate scalars ───────────────────────────────────────────────
        climate = extract_climate_scalars(image_dir)
        if climate is not None:
            store["climate"][time_index]   = climate
            store["valid_clm"][time_index] = True
            status["clm_ok"]               = True

    except Exception as e:
        status["error"] = str(e)

    return status


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # inventory = pq.read_table(INVENTORY_PATH).to_pandas(timestamp_as_object=True)
    # inventory["date"] = [v.date() for v in inventory["date"]]
    inventory = pd.read_parquet(INVENTORY_PATH)
    usable = (
        inventory[inventory["has_glacier_mask"]]
        .sort_values(["glacier_id", "date"])
        .reset_index(drop=True)
    )

    # # Smoke test mode: only process the first glacier.
    # if len(usable):
    #     first_glacier_id = usable.iloc[0]["glacier_id"]
    #     usable = usable[usable["glacier_id"] == first_glacier_id].reset_index(drop=True)

    print(f"Observations with Glacier_MASK.tif : {len(usable)}")
    print(f"Glaciers                            : {usable.glacier_id.nunique()}\n")
    seasonal_csv = pd.read_csv(SEASONAL_LABELS_PATH, sep=";", index_col=0)
    seasonal_labels = dict(zip(
        seasonal_csv["Glacier"],
        seasonal_csv["seasonal?"].str.upper() == "TRUE",
    ))

    # ── Phase A ───────────────────────────────────────────────────────────
    print("Phase A — pre-allocating zarr stores...")
    preallocate_zarr_stores(usable, seasonal_labels)
    print()

    # ── Build task list ───────────────────────────────────────────────────
    tasks = []
    for glacier_id, gdf in usable.groupby("glacier_id"):
        first_row = gdf.sort_values("date").iloc[0]
        grid = load_grid_meta(Path(first_row["image_dir"]))
        if grid is None:
            continue
        zarr_path = str(OUT_DIR / f"{glacier_id}.zarr")
        for time_index, (_, row) in enumerate(
            gdf.sort_values("date").iterrows()
        ):
            tasks.append((
                glacier_id,
                row["image_id"],
                row["image_dir"],
                time_index,
                row["date"],      # datetime.date object — .isoformat() called in worker
                row["sensor"],
                grid,
                zarr_path,
            ))

    # ── Phase B ───────────────────────────────────────────────────────────
    print(f"Phase B — processing {len(tasks)} observations...")

    results  = []
    n_workers = 16

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_observation, t): t for t in tasks
        }
        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            results.append(result)
            if result["error"]:
                print(
                    f"  ERROR  {result['glacier_id']}  "
                    f"t={result['time_index']:4d}  {result['error']}"
                )
            if i % 500 == 0 or i == len(tasks):
                print(f"  {i:5d} / {len(tasks)}")

    # ── Summary ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results)
    n          = len(results_df)
    print(f"\nDone")
    print(f"  Trace ok  : {results_df.trace_ok.sum():5d} / {n}")
    print(f"  SDT ok    : {results_df.sdt_ok.sum():5d} / {n}")
    print(f"  Climate ok: {results_df.clm_ok.sum():5d} / {n}")
    print(f"  Errors    : {results_df.error.notna().sum():5d}")

    log_path = Path(str(OUT_DIR).replace("/zarr", "")) / "processing_log_seasonal.parquet"
    results_df.to_parquet(log_path, index=False)

    errors = results_df[results_df.error.notna()]
    if len(errors):
        err_path = OUT_DIR / "processing_errors.csv"
        errors.to_csv(err_path, index=False)
        print(f"  Error log : {err_path}")


if __name__ == "__main__":
    main()