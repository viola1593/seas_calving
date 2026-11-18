# preprocessing/03_seasonal_index.py
"""
Build the training sample index from the seasonal zarr stores produced by
02_process_observations.py (seasonal variant).

Identical algorithm to 03_build_index.py but with quarterly anchor spacing
and a 90-day prediction horizon, targeting one-season-ahead forecasting.
Every valid (glacier, t, Δ) triplet is written as a flat dict of pre-resolved
zarr integer indices so that Dataset.__getitem__ performs pure array reads
with no temporal logic.

Outputs
-------
  data/splits_seasonal/train.json
  data/splits_seasonal/val.json
  data/splits_seasonal/test.json
  data/splits_seasonal/sampling_params.json
"""

import json
import zarr
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import date, timedelta
from typing import Optional
from collections import deque

# ── Paths ─────────────────────────────────────────────────────────────────────

ZARR_DIR  = Path("../glacier_data/TrainingDataset_processedV2/zarr")
OUT_DIR   = Path("../glacier_data/TrainingDataset_processedV2/splits_seasonal")
# OUT_DIR   = Path("../glacier_data/TrainingDataset_processedV2/splits_seasonal_irr")

OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── SEASONAL Anchor-proximity sampling parameters ──────────────────────────────────────

ANCHOR_OFFSETS    = [-360, -270, -180, -90, 0]  # quarterly spacing within past year
HORIZONS          = [90]                           # one-season-ahead horizon
TOLERANCE         = 30                             # ±days (narrower: 90d spacing → 60 causes 67% overlap)
MIN_GAP           = 30                             # minimum days between any two selected inputs
REFERENCE_MIN_GAP = 30                             # minimum days between kept reference times


# ── ANNUAL Anchor-proximity sampling parameters ──────────────────────────────────────

# ANCHOR_OFFSETS    = [-365*4, -365*3, -365*2, -365, 0]  # quarterly spacing within past year
# HORIZONS          = [365]                           # one-season-ahead horizon
# TOLERANCE         = 60                             # ±days (narrower: 90d spacing → 60 causes 67% overlap)
# MIN_GAP           = 60                             # minimum days between any two selected inputs
# REFERENCE_MIN_GAP = 30                             # minimum days between kept reference times

# ── Glacier-level split ratios (from all glaciers present in ZARR_DIR) ──────
TRAIN_RATIO = 0.70
VAL_RATIO   = 0.10
TEST_RATIO  = 0.20
SPLIT_SEED  = 42


# ── Anchor-proximity selection algorithm ──────────────────────────────────────

def _backtrack_assign(
    obs: np.ndarray,
    candidates: list[list[int]],
    min_gap: int,
    slot: int,
    chosen: list[int],
) -> Optional[list[int]]:
    """
    Recursive backtracking assignment.
    For each slot, try candidates in order of proximity to the anchor.
    Accept the first assignment where every pair of chosen observations
    satisfies the minimum gap constraint.
    Returns a list of obs-array indices (one per slot), or None if no
    valid assignment exists.
    """
    if slot == len(candidates):
        return chosen

    for idx in candidates[slot]:
        day = obs[idx]
        if all(abs(day - obs[c]) >= min_gap for c in chosen):
            result = _backtrack_assign(
                obs, candidates, min_gap, slot + 1, chosen + [idx]
            )
            if result is not None:
                return result

    return None


def select_input_observations(
    obs_days: list[float],
    anchor_offsets: list[int],
    horizon: int,
    tolerance: int,
    min_gap: int,
) -> Optional[dict]:
    """
    Select one observation per anchor slot plus one target observation,
    subject to anchor-proximity and minimum-spacing constraints.

    Parameters
    ----------
    obs_days       : all observation timestamps in days relative to t=0
    anchor_offsets : nominal anchor positions, e.g. [-90,-70,-50,-30,-10]
    horizon        : target offset in days, e.g. 10
    tolerance      : ±days around each anchor that qualify as a candidate
    min_gap        : minimum days between any two selected input observations

    Returns
    -------
    dict with keys:
      input_indices   list[int]  indices into obs_days (one per anchor)
      input_days      list[float] actual selected timestamps
      target_index    int        index into obs_days for the target
      target_day      float      actual target timestamp
      day_gaps        list[float] selected_day − anchor for each slot
    or None if any slot cannot be filled.
    """
    obs     = np.array(obs_days, dtype=float)
    anchors = sorted(anchor_offsets)

    # ── Build candidate list per slot ─────────────────────────────────────
    candidates = []
    for anchor in anchors:
        lo   = anchor - tolerance
        hi   = min(anchor + tolerance, 0)    # no future observations as inputs
        mask = (obs >= lo) & (obs <= hi)
        idxs = np.where(mask)[0]
        if len(idxs) == 0:
            return None   # slot unfillable
        # Sort by proximity to anchor so backtracking finds best-fit first
        idxs_sorted = idxs[np.argsort(np.abs(obs[idxs] - anchor))]
        candidates.append(idxs_sorted.tolist())

    # ── Assign observations to slots with min_gap constraint ──────────────
    assignment = _backtrack_assign(obs, candidates, min_gap, slot=0, chosen=[])
    if assignment is None:
        return None

    # ── Find target observation ───────────────────────────────────────────
    target_lo   = horizon - tolerance
    target_hi   = horizon + tolerance
    target_mask = (obs >= target_lo) & (obs <= target_hi)
    target_idxs = np.where(target_mask)[0]
    if len(target_idxs) == 0:
        return None

    target_idx = target_idxs[int(np.argmin(np.abs(obs[target_idxs] - horizon)))]

    return {
        "input_indices": assignment,
        "input_days":    obs[assignment].tolist(),
        "target_index":  int(target_idx),
        "target_day":    float(obs[target_idx]),
        "day_gaps":      [float(obs[i] - a)
                          for i, a in zip(assignment, anchors)],
    }


# ── Per-glacier sample generation ─────────────────────────────────────────────

def generate_samples_for_glacier(zarr_path: Path) -> list[dict]:
    """
    Open one glacier's zarr store and enumerate all valid training samples.

    Only observations where valid_sdt=True are eligible as inputs or targets.
    Every such observation is tried as a reference time t.
    For each t, the anchor-proximity algorithm is run for every horizon in
    HORIZONS.  Successful assignments are recorded as dicts of zarr integer
    indices — the only information Dataset.__getitem__ needs.

    Returns a (possibly empty) list of sample dicts.
    """
    try:
        store = zarr.open_group(str(zarr_path), mode="r")
    except Exception as e:
        print(f"  WARNING: could not open {zarr_path}: {e}")
        return []

    glacier_id = store.attrs.get("glacier_id", zarr_path.stem)
    seasonal   = bool(store.attrs.get("seasonal", False))

    # Read only the small metadata arrays — spatial data is not touched here
    dates_raw = store["dates"][:]       # (T,)  str  "YYYY-MM-DD"
    valid_sdt = store["valid_sdt"][:]   # (T,)  bool

    # Convert ISO strings to datetime.date objects
    dates_parsed = []
    for d in dates_raw:
        
        try:
            dates_parsed.append(date.fromisoformat(str(d)[:10]))  # fromisoformat only accepts "YYYY-MM-DD"; some dates have a trailing time part
        except (ValueError, TypeError):
            dates_parsed.append(None)

    # Zarr integer indices and calendar dates for observations with valid SDT
    valid_zarr_indices = [
        i for i, (v, d) in enumerate(zip(valid_sdt, dates_parsed))
        if v and d is not None
    ]

    valid_dates = [dates_parsed[i] for i in valid_zarr_indices]

    n_valid = len(valid_zarr_indices)
    if n_valid < len(ANCHOR_OFFSETS) + 1:
        # Not enough valid observations to form even one complete sample
        return []
    
    # Thin reference candidates so adjacent reference times are at least
    # REFERENCE_MIN_GAP days apart.
    reference_candidates = []
    last_reference_date = None
    for ref_zarr_idx, ref_date in zip(valid_zarr_indices, valid_dates):
        if last_reference_date is not None:
            if (ref_date - last_reference_date).days < REFERENCE_MIN_GAP:
                continue
        reference_candidates.append((ref_zarr_idx, ref_date))
        last_reference_date = ref_date

    samples   = []
    seen      = set()   # fingerprint-based deduplication

    # ── Outer loop: every valid observation as candidate reference time ────
    for ref_pos, (ref_zarr_idx, ref_date) in enumerate(reference_candidates):
        # Express all valid observation dates as integer day offsets
        # relative to this reference time (ref_date = t = 0)
        rel_days = [
            (d - ref_date).days for d in valid_dates
        ]

        # ── Inner loop: each prediction horizon ───────────────────────────
        for horizon in HORIZONS:
            result = select_input_observations(
                obs_days=rel_days,
                anchor_offsets=ANCHOR_OFFSETS,
                horizon=horizon,
                tolerance=TOLERANCE,
                min_gap=MIN_GAP,
            )
            if result is None:
                continue

            # Map positions in valid_zarr_indices back to absolute zarr indices
            input_zarr_indices = [
                valid_zarr_indices[i] for i in result["input_indices"]
            ]
            target_zarr_index  = valid_zarr_indices[result["target_index"]]

            # ── Deduplication ─────────────────────────────────────────────
            # Two reference times that select identical observations produce
            # identical training signals.  Keep only the first occurrence.
            fingerprint = (
                glacier_id,
                tuple(input_zarr_indices),
                target_zarr_index,
            )
            if fingerprint in seen:
                continue
            seen.add(fingerprint)

            samples.append({
                # Identity
                "glacier_id":          glacier_id,
                "zarr_path":           str(zarr_path),
                "reference_date":      ref_date.isoformat(),
                "horizon_days":        horizon,
                "seasonal":            seasonal,

                # Pre-resolved zarr integer positions
                # Dataset.__getitem__ uses these directly — no date arithmetic
                "input_zarr_indices":  input_zarr_indices,   # list[int], len=K
                "target_zarr_index":   target_zarr_index,    # int

                # Temporal metadata → τ feature vector in the model
                "input_days":          result["input_days"],  # actual offsets
                "target_day":          result["target_day"],
                "day_gaps":            result["day_gaps"],    # obs − anchor
                "anchor_offsets":      ANCHOR_OFFSETS,
            })

    return samples


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    zarr_paths  = sorted(ZARR_DIR.glob("*.zarr"))

    if not zarr_paths:
        print(f"ERROR: no zarr stores found under {ZARR_DIR}")
        return

    print(f"Found {len(zarr_paths)} zarr stores\n")
    print(f"Sampling parameters")
    print(f"  anchor_offsets : {ANCHOR_OFFSETS}")
    print(f"  horizons       : {HORIZONS} days")
    print(f"  tolerance      : ±{TOLERANCE} days")
    print(f"  min_gap        : {MIN_GAP} days\n")
    print(f"  ref_min_gap    : {REFERENCE_MIN_GAP} days\n")

    # ── Build deterministic glacier split from available zarr stores ─────
    glacier_ids = [p.stem for p in zarr_paths]
    rng = np.random.default_rng(SPLIT_SEED)
    shuffled_ids = glacier_ids.copy()
    rng.shuffle(shuffled_ids)

    n_glaciers = len(shuffled_ids)
    n_train = int(np.floor(n_glaciers * TRAIN_RATIO))
    n_val = int(np.floor(n_glaciers * VAL_RATIO))
    n_test = n_glaciers - n_train - n_val

    train_glacier_ids = set(shuffled_ids[:n_train])
    val_glacier_ids = set(shuffled_ids[n_train:n_train + n_val])
    test_glacier_ids = set(shuffled_ids[n_train + n_val:])

    print("Glacier split")
    print(f"  total glaciers : {n_glaciers}")
    print(f"  train glaciers : {len(train_glacier_ids)} ({TRAIN_RATIO:.0%})")
    print(f"  val glaciers   : {len(val_glacier_ids)} ({VAL_RATIO:.0%})")
    print(f"  test glaciers  : {len(test_glacier_ids)} ({TEST_RATIO:.0%})")
    print(f"  split seed     : {SPLIT_SEED}\n")

    # ── Generate samples ──────────────────────────────────────────────────
    all_samples   = []
    per_glacier   = []

    for zarr_path in zarr_paths:
        glacier_id = zarr_path.stem
        samples    = generate_samples_for_glacier(zarr_path)
        all_samples.extend(samples)
        per_glacier.append({
            "glacier_id": glacier_id,
            "n_samples":  len(samples),
        })
        print(f"  {glacier_id:40s}  {len(samples):6d} samples")

    print(f"\nTotal samples before split : {len(all_samples)}")

    # ── Glacier-level split ───────────────────────────────────────────────
    train = [s for s in all_samples if s["glacier_id"] in train_glacier_ids]
    val   = [s for s in all_samples if s["glacier_id"] in val_glacier_ids]
    test  = [s for s in all_samples if s["glacier_id"] in test_glacier_ids]

    # ── Verify no glacier appears in more than one split ──────────────────
    train_ids = {s["glacier_id"] for s in train}
    val_ids   = {s["glacier_id"] for s in val}
    test_ids  = {s["glacier_id"] for s in test}

    assert train_ids.isdisjoint(val_ids),  "Glacier leakage: train ∩ val"
    assert train_ids.isdisjoint(test_ids), "Glacier leakage: train ∩ test"
    assert val_ids.isdisjoint(test_ids),   "Glacier leakage: val ∩ test"

    # ── Write split indices ───────────────────────────────────────────────
    json.dump(train, open(OUT_DIR / "train.json", "w"), indent=2)
    json.dump(val,   open(OUT_DIR / "val.json",   "w"), indent=2)
    json.dump(test,  open(OUT_DIR / "test.json",  "w"), indent=2)

    # ── Write per-glacier sample counts ──────────────────────────────────
    pd.DataFrame(per_glacier).to_parquet(
        OUT_DIR / "per_glacier_counts.parquet", index=False
    )

    # ── Write sampling parameters alongside the index for reproducibility -
    params = {
        "model_variant":   "seasonal",
        "anchor_offsets":  ANCHOR_OFFSETS,
        "horizons":        HORIZONS,
        "tolerance_days":  TOLERANCE,
        "min_gap_days":    MIN_GAP,
        "reference_min_gap_days": REFERENCE_MIN_GAP,
        "split_ratios": {
            "train": TRAIN_RATIO,
            "val": VAL_RATIO,
            "test": TEST_RATIO,
            "seed": SPLIT_SEED,
        },
        "split": {
            "test_glaciers":  sorted(test_ids),
            "val_glaciers":   sorted(val_ids),
            "train_glaciers": sorted(train_ids),
        },
        "counts": {
            "train": len(train),
            "val":   len(val),
            "test":  len(test),
            "total": len(all_samples),
        },
    }
    json.dump(params, open(OUT_DIR / "sampling_params.json", "w"), indent=2)

    # ── Final summary ─────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"SPLIT SUMMARY")
    print(f"{'='*50}")
    print(f"  Train : {len(train):7d} samples  ({len(train_ids)} glaciers)")
    print(f"  Val   : {len(val):7d} samples  ({len(val_ids)} glaciers)")
    print(f"  Test  : {len(test):7d} samples  ({len(test_ids)} glaciers)")
    print(f"  Total : {len(all_samples):7d} samples")
    print(f"{'='*50}")
    print(f"\nOutputs written to {OUT_DIR}")
    print(f"  train.json  val.json  test.json")
    print(f"  sampling_params.json")
    print(f"  per_glacier_counts.parquet")


if __name__ == "__main__":
    main()
