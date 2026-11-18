"""
Shared helpers for the model_analysis scripts (sample selection, channel names,
season / ocean-current grouping, results cache I/O).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def select_samples(loader, n: int | None) -> list[dict]:
    if n is None:
        return list(loader)
    half = n // 2
    seasonal, non_seasonal = [], []
    for batch in loader:
        is_seas = bool(batch["glacier_seasonality"][0].item())
        bucket = seasonal if is_seas else non_seasonal
        if len(bucket) < half:
            bucket.append(batch)
        if len(seasonal) >= half and len(non_seasonal) >= half:
            break
    return seasonal + non_seasonal


_GEOPHYS_NAMES = ("bed", "surf", "thick", "vel", "strain")

# Headline driver channels: physical fields only. Excludes sdt, sdt_diff
# (autoregressive state). bed_slope included — it is a physical field, not
# autoregressive/imagery. `if name in channel_names` keeps this robust to
# configs trained without bed_slope.
_DRIVER_CHANNELS = ("bed", "surf", "thick", "vel", "strain", "bed_slope")


def _spatial_channel_names(cfg) -> list[str]:
    """Channel names in dataset_v2 assembly order, driven by config (NOT count).
    Order matches data/dataset_v2.py.__getitem__:
    sdt, sdt_diff, [sdt_trend], [geophys×5], [bed_slope]."""
    sp = cfg["data"]["config"]["inputs"]["spatial"]
    names = ["sdt", "sdt_diff"]
    if "sdt_trend" in sp:
        names.append("sdt_trend")
    if "geophys" in sp:
        names += list(_GEOPHYS_NAMES)
    if "bed_slope" in sp:
        names.append("bed_slope")
    return names


_CURRENT_LOOKUP_PATH = Path(__file__).parents[2] / "glacier_data" / "glacier_current_lookup.json"


def _load_current_lookup() -> dict:
    with open(_CURRENT_LOOKUP_PATH) as f:
        return json.load(f)["current"]


_CURRENTS = ("ESC", "WSC")
_CURRENT_COLORS = {"ESC": "#4C72B0", "WSC": "#C44E52"}


_SEASONS = {
    12: "DJF", 1: "DJF", 2: "DJF",
    3:  "MAM", 4: "MAM", 5: "MAM",
    6:  "JJA", 7: "JJA", 8: "JJA",
    9:  "SON", 10: "SON", 11: "SON",
}
_SEASON_ORDER = ["DJF", "MAM", "JJA", "SON"]


def _season(date_str: str) -> str:
    month = int(date_str[5:7])
    return _SEASONS[month]


def save_results(path: Path, results: dict) -> None:
    arrays = {}
    for var, by_glacier in results.items():
        for gid, by_season in by_glacier.items():
            for season, vals in by_season.items():
                if vals:
                    arrays[f"{var}__{gid}__{season}"] = np.array(vals, dtype=np.float32)
    np.savez(path, **arrays)
    print(f"Saved cache: {path}")
