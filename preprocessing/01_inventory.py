# preprocessing/01_inventory.py
"""
Docstring for preprocessing.01_inventory

Walk the directory tree, parse every `image_id` folder name, extract glacier ID, sensor, and acquisition date. Write a flat inventory table. This is the only script that touches the raw directory structure.
"""

import re
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

RAW_DIR  = Path("../glacier_data/TrainingDataset_V2/TrainingDataset")
OUT_DIR  = Path("../glacier_data/TrainingDataset_processedV2")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "inventory.parquet"

# ── Band definitions per sensor ───────────────────────────────────────────────
# Exactly as they appear in the filename: {image_id}_{band}.tif

SENSOR_BANDS = {
    "S2": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8A", "B9", "B11", "B12"],
    "LE07": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11"],
    "LC08": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11"],
    "LC09": ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B10", "B11"],
    "S1A": [],
    "S1B": [],
}

# ── Semantic band role mapping ────────────────────────────────────────────────
# Maps logical role → actual band name for each sensor.
# Common training bands: Blue, Green, Red, NIR.
# SWIR1 included for NDSI computation (snow/ice index) — not a training band
# but needed for preprocessing indices.

BAND_ROLES = {
    #          Blue   Green  Red    NIR    SWIR1
    "S2": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B8A", "swir1": "B11"},
    "LE07": {"blue": "B1", "green": "B2", "red": "B3", "nir": "B4", "swir1": "B5"},
    "LC08": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6"},
    "LC09": {"blue": "B2", "green": "B3", "red": "B4", "nir": "B5", "swir1": "B6"},
}

# Common training bands — the four that exist across all three sensors.
# Any observation missing one of these cannot be used for model training.
COMMON_TRAINING_ROLES = ["blue", "green", "red", "nir", "swir1"]

# ── Static geophysical files — fixed names, not prefixed with image_id ────────

STATIC_FILES = {
    "bed_elevation": "Bed_Elevation.tif",
    "strain_rate": "Eff_Strain_Rate.tif",
    "ice_fjord_mask": "Ice_Fjord_Mask.tif",
    "mar_runoff": "MAR_Runoff.tif",
    "ocean_temp": "Ocean_Temp.tif",
    "surface_elevation": "Surface_Elevation.tif",
    "thickness": "Thickness.tif",
    "trace_mask": "Trace_MASK.tif",
    "glacier_mask": "Glacier_MASK.tif",  # calving front trace → SDT input
    "velocity_vx": "Velocity_VX.tif",
    "velocity_vy": "Velocity_VY.tif",
    "velocity": "Velocity.tif",
}

# ── Image ID parsing ──────────────────────────────────────────────────────────

# Sentinel-2: 20160401T122652_20160401T122743_T35XMJ
#   sensing date = first 8 chars of segment 1
#   tile ID      = segment 3 (T + 5 alphanumeric chars)
S2_PATTERN = re.compile(r"^(\d{8})T\d{6}_\d{8}T\d{6}_(T[A-Z0-9]{5})$")

# Landsat: LC08_010242_20220529  /  LE07_010242_20220529
#   sensor    = first 4 chars (LC08 / LC09 / LE07)
#   path_row  = 6-digit segment
#   date      = 8-digit segment
LS_PATTERN = re.compile(r"^(LC08|LC09|LE07)_(\d{6})_(\d{8})$")

# Sentinel-1: S1A_EW_GRDM_1SDH_20150909T054247_20150909T054339_007634_00A950_D932
#   platform  = S1A or S1B
#   date      = first 8 chars of the 5th underscore-segment (start datetime)
#   orbit     = 6-digit absolute orbit number (segment 7)
S1_PATTERN = re.compile(
    r"^(S1[AB])_[A-Z0-9]+_[A-Z0-9]+_[A-Z0-9]+_(\d{8})T\d{6}_\d{8}T\d{6}_(\d{6})_[0-9A-F]{6}_[0-9A-F]{4}$"
)


def parse_image_id(image_id: str) -> dict | None:
    """
    Parse a Sentinel-2 or Landsat image folder name into structured metadata.
    Returns None for any unrecognised format so the caller can log and skip.

    S2 example : 20160401T122652_20160401T122743_T35XMJ
    LS example : LC08_010242_20220529
                 LE07_010242_20190712
    """
    
    # ── Sentinel-2 ───────────────────────────────────────────────────────
    m = S2_PATTERN.match(image_id)
    if m:
        try:
            acq_date = datetime.strptime(m.group(1), "%Y%m%d").date()
        except ValueError:
            return None
        return {
            "sensor": "S2",
            "date": acq_date,
            "tile_or_pathrow": m.group(2),  # e.g. "T35XMJ"
        }

    # ── Landsat ───────────────────────────────────────────────────────────
    m = LS_PATTERN.match(image_id)
    if m:
        sensor = m.group(1)  # "LC08" / "LC09" / "LE07"
        path_row = m.group(2)  # e.g. "010242"
        try:
            acq_date = datetime.strptime(m.group(3), "%Y%m%d").date()
        except ValueError:
            return None
        return {
            "sensor": sensor,
            "date": acq_date,
            "tile_or_pathrow": path_row,
        }

    # ── Sentinel-1 ────────────────────────────────────────────────────────
    m = S1_PATTERN.match(image_id)
    if m:
        sensor = m.group(1)   # "S1A" or "S1B"
        try:
            acq_date = datetime.strptime(m.group(2), "%Y%m%d").date()
        except ValueError:
            return None
        return {
            "sensor": sensor,
            "date": acq_date,
            "tile_or_pathrow": m.group(3),  # absolute orbit number
        }

    return None


# ── Band presence checking ────────────────────────────────────────────────────

# ── Band file naming patterns ─────────────────────────────────────────────────

# Sentinel-2 : {image_id}_{band}.tif   e.g. 20160401T122652_20160401T122743_T35XMJ_B4.tif
# Landsat     : {image_id}.{band}.tif   e.g. LC08_010242_20220529.B4.tif

BAND_FILENAME_PATTERNS = {
    "S2": "{image_id}_{band}.tif",
    "LE07": "{image_id}.{band}.tif",
    "LC08": "{image_id}.{band}.tif",
    "LC09": "{image_id}.{band}.tif",
    "S1A": "",
    "S1B": "",
}


def get_band_filename(image_id: str, band: str, sensor: str) -> str:
    """
    Construct the expected filename for a given band, respecting the
    sensor-specific naming convention.

      Sentinel-2 : {image_id}_{band}.tif
      Landsat    : {image_id}.{band}.tif
    """
    pattern = BAND_FILENAME_PATTERNS.get(sensor)
    if pattern is None:
        raise ValueError(f"Unknown sensor: {sensor}")
    return pattern.format(image_id=image_id, band=band)


def check_bands(image_dir: Path, image_id: str, sensor: str) -> dict:
    """
    Check presence of every expected band file for this sensor.
    Filename convention differs between sensors:

      Sentinel-2 : {image_id}_{band}.tif
      Landsat    : {image_id}.{band}.tif

    Returns:
      - one boolean column per band   e.g. band_B3 = True / False
      - has_common_bands              all required training bands present
      - has_swir1                     SWIR1 present (needed for NDSI)
      - n_bands_present / n_bands_expected
    """
    expected_bands = SENSOR_BANDS.get(sensor, [])
    roles = BAND_ROLES.get(sensor, {})

    # ── Per-band presence ─────────────────────────────────────────────────
    per_band = {}
    for band in expected_bands:
        fname = get_band_filename(image_id, band, sensor)
        per_band[f"band_{band}"] = (image_dir / fname).is_file()

    # ── Required training bands ───────────────────────────────────────────
    # Resolve logical roles to actual band names for this sensor
    common_band_names = [roles[role] for role in COMMON_TRAINING_ROLES if role in roles]
    has_common = bool(common_band_names) and all(
        per_band.get(f"band_{b}", False) for b in common_band_names
    )

    # ── SWIR1 — needed for NDSI in preprocessing, not a training channel ──
    swir1_name = roles.get("swir1")
    has_swir1 = per_band.get(f"band_{swir1_name}", False) if swir1_name else False

    return {
        **per_band,
        "has_common_bands": has_common,
        "has_swir1": has_swir1,
        "n_bands_present": sum(per_band.values()),
        "n_bands_expected": len(expected_bands),
    }


def check_static_files(image_dir: Path) -> dict:
    """
    Check which static geophysical files are present.
    These use fixed names and are not prefixed with the image_id.
    """
    result = {}
    for key, fname in STATIC_FILES.items():
        result[f"has_{key}"] = (image_dir / fname).is_file()

    # Derived summary flags used as usability filters downstream
    result["has_trace"] = result["has_glacier_mask"]  # glacier mask includes the trace
    result["has_geophys"] = all(
        [
            result["has_bed_elevation"],
            result["has_surface_elevation"],
            result["has_thickness"],
            result["has_ice_fjord_mask"],
        ]
    )
    result["has_velocity"] = all(
        [
            result["has_velocity_vx"],
            result["has_velocity_vy"],
            result["has_velocity"],
            result["has_strain_rate"],
        ]
    )
    result["has_climate"] = all(
        [
            result["has_mar_runoff"],
            result["has_ocean_temp"],
        ]
    )
    return result


# ── Main inventory walk ───────────────────────────────────────────────────────


def main():
    records = []
    skipped = []

    glacier_dirs = sorted(d for d in RAW_DIR.iterdir() if d.is_dir())
    print(f"Found {len(glacier_dirs)} glacier directories under {RAW_DIR}\n")

    for glacier_dir in glacier_dirs:
        glacier_id = glacier_dir.name
        image_dirs = sorted(d for d in glacier_dir.iterdir() if d.is_dir())

        for image_dir in image_dirs:
            image_id = image_dir.name

            # ── Parse image ID ────────────────────────────────────────────
            parsed = parse_image_id(image_id)
            if parsed is None:
                skipped.append(
                    {
                        "glacier_id": glacier_id,
                        "image_id": image_id,
                        "reason": "unrecognised image_id format",
                    }
                )
                continue

            sensor = parsed["sensor"]

            # ── Check files ───────────────────────────────────────────────
            band_info = check_bands(image_dir, image_id, sensor)
            static_info = check_static_files(image_dir)

            records.append(
                {
                    # ── Identity ──────────────────────────────────────────────
                    "glacier_id": glacier_id,
                    "image_id": image_id,
                    "image_dir": str(image_dir),
                    "date": parsed["date"],
                    "sensor": sensor,
                    "tile_or_pathrow": parsed["tile_or_pathrow"],
                    # ── Band presence ─────────────────────────────────────────
                    **band_info,
                    # ── Static file presence ──────────────────────────────────
                    **static_info,
                }
            )
    # ── DataFrame ─────────────────────────────────────────────────────────────
    df = pd.DataFrame(records)

    if df.empty:
        print("WARNING: no valid observations found — check RAW_DIR")
        return

    # df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["glacier_id", "date"]).reset_index(drop=True)

    # An observation is "fully usable" when the SDT can be computed
    # AND the four common training bands are all present
    df["fully_usable"] = df["has_trace"] & df["has_common_bands"]

    # ── Console summary ───────────────────────────────────────────────────────
    print("=" * 60)
    print("INVENTORY SUMMARY")
    print("=" * 60)
    print(f"  Total observations      : {len(df)}")
    print(f"  Glaciers                : {df.glacier_id.nunique()}")
    print(f"  Date range              : {df.date.min()} → {df.date.max()}")
    print()
    print(f"  Sensor breakdown:")
    for sensor, count in df.sensor.value_counts().items():
        print(f"    {sensor:6s}  {count:5d}")
    print()
    print(f"  has calving front trace : {df.has_trace.sum():5d}")
    print(f"  has common bands (BGRN) : {df.has_common_bands.sum():5d}")
    print(f"  has SWIR1 (for NDSI)    : {df.has_swir1.sum():5d}")
    print(f"  has geophys channels    : {df.has_geophys.sum():5d}")
    print(f"  has climate variables   : {df.has_climate.sum():5d}")
    print(f"  has velocity fields     : {df.has_velocity.sum():5d}")
    print(f"  fully usable (trace+BGRN): {df.fully_usable.sum():5d}")
    print()
    print(f"  Skipped (bad image_id)  : {len(skipped)}")

    # ── Per-glacier summary ───────────────────────────────────────────────────
    per_glacier = (
        df.groupby("glacier_id")
        .agg(
            n_obs=("image_id", "count"),
            n_with_trace=("has_trace", "sum"),
            n_with_bands=("has_common_bands", "sum"),
            n_fully_usable=("fully_usable", "sum"),
            n_with_geophys=("has_geophys", "sum"),
            date_min=("date", "min"),
            date_max=("date", "max"),
            sensors=("sensor", lambda x: ",".join(sorted(x.unique()))),
        )
        .reset_index()
    )

    # A glacier needs ≥ K+1 = 6 fully usable observations to form any sample
    # (5 inputs + 1 target, all with traces and common bands)
    per_glacier["sufficient_obs"] = per_glacier["n_fully_usable"] >= 6
    print(
        f"  Glaciers with ≥6 usable obs : "
        f"{per_glacier.sufficient_obs.sum()} / {len(per_glacier)}"
    )
    print("=" * 60)

    # ── Write outputs ─────────────────────────────────────────────────────────
    df.to_parquet(OUT_PATH, index=False)
    per_glacier.to_parquet(OUT_DIR / "glacier_summary.parquet", index=False)

    if skipped:
        pd.DataFrame(skipped).to_csv(OUT_DIR / "skipped_image_ids.csv", index=False)

    # ── File schema for downstream scripts ───────────────────────────────────
    # Scripts 02–04 read this so they never need to re-implement naming logic
    schema = {
        "sensor_bands": SENSOR_BANDS,
        "band_roles": BAND_ROLES,
        "common_training_roles": COMMON_TRAINING_ROLES,
        "static_files": STATIC_FILES,
        "band_filename_patterns": BAND_FILENAME_PATTERNS,  # ← add this
        "sensors_supported": ["S2", "LE07", "LC08", "LC09", "S1A", "S1B"],
    }

    json.dump(schema, open(OUT_DIR / "file_schema.json", "w"), indent=2)

    print(f"\nOutputs written to {OUT_DIR}")
    print(f"  inventory.parquet")
    print(f"  glacier_summary.parquet")
    print(f"  file_schema.json")
    if skipped:
        print(f"  skipped_image_ids.csv")


if __name__ == "__main__":
    main()
