# metrics.py
"""
Evaluation metrics for calving front SDT prediction.

All metrics operate in metres (denormalised SDT) and are computed within a
spatially restricted evaluation zone around the calving front.

Dataset alignment
-----------------
- Spatial resolution : 30 m (EPSG:3995)
- SDT source         : computed from Glacier_MASK.tif via distance_transform_edt
                       clipped to ±2000 m, stored as float32 in zarr["sdt"]
- target_trace       : zarr["trace"] — from Trace_MASK.tif (calving front line only,
                       no fjord boundary pixels)
- spatial_valid_mask : batch["spatial_valid_mask"] from calving_collate_fn
                       (B, 1, H_pad, W_pad) bool — marks real vs padded pixels
- original_hw        : batch["original_hw"] — list of (H, W) before padding

Usage in training loop
----------------------
    from metrics import evaluate_batch, aggregate_metrics, denormalise_sdt

    # NOTE: displacement models do NOT output normalised SDT directly.
    # Reconstruct absolute normalised SDT first (see lightning_module.py):
    #     pred_sdt_norm = pred_disp / disp_scale + last_input_sdt_norm
    # and only then denormalise:
    pred_m   = denormalise_sdt(pred_sdt_norm.cpu().numpy(),   stats)  # (B,1,H,W) → m
    target_m = denormalise_sdt(batch["target"].cpu().numpy(), stats)

    records = evaluate_batch(
        pred_m, target_m,
        target_traces        = traces_np,                    # (B, H, W) bool from zarr
        spatial_valid_masks  = batch["spatial_valid_mask"].cpu().numpy(),
        original_hw          = batch["original_hw"],
    )
    agg = aggregate_metrics(records)

Public API
----------
  compute_metrics(pred_sdt, target_sdt, target_trace, ...) → dict
  compute_skill_score(model_mde, persistence_mde)           → float
  aggregate_metrics(records)                                → dict
  evaluate_batch(preds, targets, traces, ...)               → list[dict]
  denormalise_sdt(normalised_sdt, stats)                    → ndarray

Metric definitions
------------------
  mae_front             Mean absolute SDT error within the front band (m)
  rmse_front            Root mean square SDT error within the front band (m)
  active_mde            Mean Distance Error between predicted and true front (m),
                        computed within the active mask (not the front band).
                        Symmetric: average of min-distances in both directions.
                        NaN when active_mask is not provided.
  active_hausdorff      Hausdorff distance between predicted and true front (m),
                        within the active mask. Worst-case minimum distance
                        in either direction. NaN when active_mask is not provided.
  advance_retreat_bias  Signed mean true SDT sampled at the predicted front (m),
                        negated so sign matches the ice convention (sdt > 0 = ice)
                        > 0 → predicted front too advanced (advance bias)
                        < 0 → predicted front too retreated (retreat bias)
  area_error_km2        Area of symmetric difference between predicted and
                        true glacier extents within the fjord ROI (km²)
  front_iou             Intersection-over-Union of dilated front bands [0, 1]
  active_skill_score    1 − (model_active_MDE / persistence_active_MDE)
                        > 0 → beats persistence baseline
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial.distance import cdist
from skimage.measure import find_contours


# ── Constants ─────────────────────────────────────────────────────────────────

# Spatial resolution of the zarr arrays (metres per pixel) — EPSG:3995, 30 m
DEFAULT_RESOLUTION_M: float = 30.0

# Evaluation band half-width around the calving front (metres).
# At 30 m resolution, 300 m = 10 pixels.
DEFAULT_FRONT_BAND_M: float = 300.0

# Dilation radius for front IoU metric (metres).
# At 30 m resolution, 150 m = 5 pixels.
DEFAULT_IOU_DILATION_M: float = 150.0

# Temporal variance threshold (m²) for excluding static fjord boundary pixels.
# Pixels where SDT variance across K input slots < threshold are considered
# static (fjord rock wall, zero-level-set fixed in place) and excluded from
# the evaluation zone.  100 m² ≈ 10 m std, comfortably above noise but well
# below the motion at an actively calving front (typ. 50–500 m/yr).
DEFAULT_SDT_VAR_THRESHOLD_M2: float = 100.0

# Sentinel value returned when a metric cannot be computed
NAN = float("nan")


# ── Evaluation zone construction ──────────────────────────────────────────────

def _dilate_euclidean(mask: np.ndarray, radius_px: float) -> np.ndarray:
    """
    Dilate a boolean mask by a Euclidean radius (pixels).

    Unlike binary_dilation(iterations=n), which grows an L1 diamond and
    under-reaches diagonally by ~30%, this produces an isotropic disk, so
    band widths are orientation-independent.
    """
    mask = mask.astype(bool)
    if not mask.any():
        return mask.copy()
    return distance_transform_edt(~mask) <= radius_px


def build_evaluation_zone(
    target_sdt:   np.ndarray,
    persist_sdt:  Optional[np.ndarray] = None,
    active_mask:  Optional[np.ndarray] = None,
    front_band_m: float = DEFAULT_FRONT_BAND_M,
) -> np.ndarray:
    """
    Field-metric evaluation zone (MAE/RMSE/bias/area/IoU): the adaptive union
    band of the target and persistence fronts, intersected with the temporally
    active pixels.

        zone = (|target_sdt| ≤ front_band_m  ∪  |persist_sdt| ≤ front_band_m)
               ∩ active_mask

    The band is obtained by thresholding the (Euclidean, metre-scale) SDTs
    directly — no morphological dilation, so it is isotropic by construction.
    Taking the union with the persistence front's band covers the whole
    old-front → new-front corridor, so IoU / area error do not saturate when
    the calving front moves more than front_band_m between anchor and target.

    The ∩ active_mask is essential: the SDT zero-level-set is the *entire*
    glacier outline (fjord rock wall included), so |target_sdt| ≤ band alone
    would band the static walls too; active_mask (sdt_var > threshold over the
    K input slots) keeps only the migrating calving front.

    Parameters
    ----------
    target_sdt : (H, W) float32
        Ground-truth SDT at t+horizon (metres, denormalised) — zarr["sdt"].
    persist_sdt : (H, W) float32 or None
        Persistence SDT = last input slot / anchor (metres).  If None, the zone
        is the target band alone (union degenerates to one front).
    active_mask : (H, W) bool or None
        True where SDT has varied across K input slots (temporally active).
        Computed as sdt_var > DEFAULT_SDT_VAR_THRESHOLD_M2.  If None, no static-
        pixel exclusion is applied — NOTE: without it the band follows the whole
        glacier outline, not just the front; real evaluation always passes it.
    front_band_m : float
        Half-width of each front band (metres).

    Returns
    -------
    (H, W) bool — True where field metrics should be computed.
    """
    band = np.abs(target_sdt) <= front_band_m

    # if persist_sdt is not None:
    #     band = band | (np.abs(persist_sdt) <= front_band_m)

    if active_mask is not None:
        band = band & active_mask.astype(bool)

    return band


# ── Predicted front extraction ────────────────────────────────────────────────

def extract_predicted_front(
    pred_sdt:     np.ndarray,
    eval_zone:    np.ndarray,
    resolution_m: float = DEFAULT_RESOLUTION_M,
) -> Optional[np.ndarray]:
    """
    Extract the predicted calving front as pixel-coordinate points from the
    zero-level-set of the predicted SDT, restricted to eval_zone.

    The eval_zone restriction prevents picking up the static Ice_Fjord_Mask
    boundary zero-crossing, which is also present in the SDT.

    Parameters
    ----------
    pred_sdt : (H, W) float32
        Predicted signed distance transform (metres, denormalised).
    eval_zone : (H, W) bool
        Evaluation zone from build_evaluation_zone.
    resolution_m : float

    Returns
    -------
    (N, 2) float array of (row, col) coordinates in metres, or None if no
    valid contour found within the evaluation zone.
    """
    masked = pred_sdt.copy()
    masked[~eval_zone] =  1.0   # force non-zero outside zone

    try:
        contours = find_contours(masked, level=0.0)
    except Exception:
        return None

    if not contours:
        return None

    front_px = np.concatenate(contours, axis=0)   # (N, 2) pixel coords

    # Filter to eval_zone: masking non-zone pixels with 1.0 can create spurious
    # zero-crossings at the zone boundary; keep only points inside the zone.
    H, W = pred_sdt.shape
    row_idx = np.clip(np.round(front_px[:, 0]).astype(int), 0, H - 1)
    col_idx = np.clip(np.round(front_px[:, 1]).astype(int), 0, W - 1)
    in_zone = eval_zone[row_idx, col_idx]
    front_px = front_px[in_zone]
    if len(front_px) == 0:
        return None

    return front_px * resolution_m                 # (N, 2) metres


def trace_to_points(
    target_trace: np.ndarray,
    eval_zone:    np.ndarray,
    resolution_m: float = DEFAULT_RESOLUTION_M,
) -> Optional[np.ndarray]:
    """
    Convert the binary Trace_MASK raster (zarr["trace"]) to metre-coordinate
    points, restricted to the evaluation zone.

    Parameters
    ----------
    target_trace : (H, W) bool
        True pixels on the calving front (from zarr["trace"]).
    eval_zone : (H, W) bool
    resolution_m : float

    Returns
    -------
    (N, 2) float array of (row, col) coordinates in metres, or None.
    """
    active = target_trace & eval_zone
    rows, cols = np.where(active)
    if len(rows) == 0:
        return None
    pts = np.stack([rows.astype(float), cols.astype(float)], axis=1)
    return pts * resolution_m   # (N, 2) metres


# ── Individual metric functions ───────────────────────────────────────────────

def _mae_rmse_in_zone(
    pred_sdt:   np.ndarray,
    target_sdt: np.ndarray,
    zone:       np.ndarray,
) -> tuple[float, float]:
    """MAE and RMSE of the SDT prediction within the evaluation zone (metres)."""
    err     = pred_sdt[zone] - target_sdt[zone]
    abs_err = np.abs(err)
    return float(abs_err.mean()), float(np.sqrt((err ** 2).mean()))


def _mde_hausdorff(
    pred_pts:   np.ndarray,
    target_pts: np.ndarray,
) -> tuple[float, float]:
    """
    Symmetric Mean Distance Error and Hausdorff distance (metres).

    MDE = 0.5 × (mean min-dist pred→target + mean min-dist target→pred)
    H   = max(max min-dist pred→target, max min-dist target→pred)
    """
    D       = cdist(pred_pts, target_pts, metric="euclidean")
    min_p2t = D.min(axis=1)
    min_t2p = D.min(axis=0)
    mde       = float((min_p2t.mean() + min_t2p.mean()) / 2.0)
    hausdorff = float(max(min_p2t.max(), min_t2p.max()))
    return mde, hausdorff


def _advance_retreat_bias(
    pred_sdt:   np.ndarray,
    target_sdt: np.ndarray,
    eval_zone:  np.ndarray,
) -> float:
    """
    Signed mean true SDT at the predicted calving front (metres).

    The predicted front is the zero-level-set of pred_sdt (restricted to
    eval_zone). The true SDT is sampled there: because target_sdt is a valid
    unit-gradient distance field, its value at each predicted-front pixel is
    the signed metric distance from the predicted front to the true front —
    honest metres. (Sampling pred_sdt at the true front instead would measure
    displacement in predicted-SDT units, and pred_sdt is not guaranteed to be
    a unit-gradient field.) The sign is negated so that, with sdt > 0 = ice:

    > 0 → predicted front too advanced (predicted front beyond true front)
    < 0 → predicted front too retreated

    This is the signed companion to the pred→target term of active_mde.
    Returns NaN if no predicted front is found within eval_zone.
    """
    masked = pred_sdt.copy()
    masked[~eval_zone] = 1.0   # force non-zero outside zone
    try:
        contours = find_contours(masked, level=0.0)
    except Exception:
        return NAN
    if not contours:
        return NAN

    front_px = np.concatenate(contours, axis=0)
    H, W = pred_sdt.shape
    row_idx = np.clip(np.round(front_px[:, 0]).astype(int), 0, H - 1)
    col_idx = np.clip(np.round(front_px[:, 1]).astype(int), 0, W - 1)
    in_zone = eval_zone[row_idx, col_idx]
    if not in_zone.any():
        return NAN

    return float(-target_sdt[row_idx[in_zone], col_idx[in_zone]].mean())


def _area_error_km2(
    pred_sdt:    np.ndarray,
    target_sdt:  np.ndarray,
    eval_zone:   np.ndarray,
    resolution_m: float,
) -> float:
    """
    Area of the symmetric difference between predicted and true glacier
    extent, restricted to the evaluation zone around the calving front (km²).

    Glacier extent is SDT > 0 within the eval_zone (trace-based front band).
    """
    pred_ice   = (pred_sdt   > 0) & eval_zone
    target_ice = (target_sdt > 0) & eval_zone
    sym_diff  = pred_ice ^ target_ice
    pixel_km2 = (resolution_m / 1000.0) ** 2
    return float(sym_diff.sum() * pixel_km2)


def _front_iou(
    pred_sdt:       np.ndarray,
    target_trace:   np.ndarray,
    eval_zone:      np.ndarray,
    iou_dilation_m: float,
    resolution_m:   float,
) -> float:
    """
    Intersection-over-Union of dilated front bands [0, 1].

    Both the predicted front (zero-level-set of pred_sdt) and the true front
    (zarr["trace"] = Trace_MASK.tif) are dilated by iou_dilation_m metres.
    At 30 m resolution, default 150 m = 5 pixels.
    """
    dilation_px = max(1.0, iou_dilation_m / resolution_m)

    masked_pred = pred_sdt.copy()
    masked_pred[~eval_zone] = 1.0
    try:
        pred_contours = find_contours(masked_pred, level=0.0)
    except Exception:
        return NAN

    if not pred_contours:
        return NAN

    front_px = np.concatenate(pred_contours, axis=0)
    H, W = pred_sdt.shape
    row_idx = np.clip(np.round(front_px[:, 0]).astype(int), 0, H - 1)
    col_idx = np.clip(np.round(front_px[:, 1]).astype(int), 0, W - 1)
    in_zone = eval_zone[row_idx, col_idx]
    if not in_zone.any():
        return NAN

    pred_front_mask = np.zeros(pred_sdt.shape, dtype=bool)
    pred_front_mask[row_idx[in_zone], col_idx[in_zone]] = True

    pred_band   = _dilate_euclidean(pred_front_mask, dilation_px) & eval_zone
    target_band = _dilate_euclidean(target_trace & eval_zone, dilation_px) & eval_zone

    intersection = (pred_band & target_band).sum()
    union        = (pred_band | target_band).sum()

    if union == 0:
        return NAN
    return float(intersection / union)


# ── Master metric function ────────────────────────────────────────────────────

def compute_metrics(
    pred_sdt:      np.ndarray,
    target_sdt:    np.ndarray,
    target_trace:  Optional[np.ndarray],
    active_mask:   Optional[np.ndarray] = None,
    persist_sdt:   Optional[np.ndarray] = None,
    resolution_m:  float = DEFAULT_RESOLUTION_M,
    front_band_m:  float = DEFAULT_FRONT_BAND_M,
    iou_dilation_m: float = DEFAULT_IOU_DILATION_M,
) -> dict[str, float]:
    """
    Compute the full evaluation metric suite for one prediction.

    Call this on single (H, W) arrays that have already been cropped to
    their original spatial extent (i.e. padding from calving_collate_fn
    has been removed). See evaluate_batch for batch-level handling.

    Two evaluation zones are used deliberately:
      * Field metrics (MAE/RMSE/bias/area/IoU) use the adaptive union band
        (|target_sdt| ≤ front_band_m ∪ |persist_sdt| ≤ front_band_m) ∩ active
        (see build_evaluation_zone) — covers the old-front → new-front corridor.
      * Contour metrics (MDE/Hausdorff) use the raw active_mask alone, so
        front-position error is measured wherever the predicted front lands
        rather than being capped by a truth-anchored corridor.
    active_mask should be derived from temporal variance of the K input SDT
    frames: active_mask = np.var(sdt_stack, axis=0) > threshold.  Pixels where
    variance ≈ 0 are the static fjord rock wall.

    Parameters
    ----------
    pred_sdt : (H, W) float32
        Predicted SDT in metres — must be denormalised before calling.
    target_sdt : (H, W) float32
        Ground truth SDT in metres — from zarr["sdt"].
    target_trace : (H, W) bool or None
        Calving front trace — zarr["trace"].  If None, all metrics are NaN.
    active_mask : (H, W) bool or None
        True where SDT has varied temporally (sdt_var > threshold).
        If None, no static-pixel exclusion is applied (original behaviour).
    persist_sdt : (H, W) float32 or None
        Persistence SDT = last input slot / anchor (metres).  Used only to
        build the field-metric union band; does not affect MDE/Hausdorff.
    resolution_m : float
        Spatial resolution in metres per pixel (default 30 m, EPSG:3995).
    front_band_m : float
        Half-width of the evaluation band around the front (metres).
    iou_dilation_m : float
        Dilation radius for front IoU computation (metres).

    Returns
    -------
    dict with keys:
      mae_front             float  metres
      rmse_front            float  metres
      active_mde            float  metres  (NaN if trace or active_mask unavailable)
      active_hausdorff      float  metres  (NaN if trace or active_mask unavailable)
      advance_retreat_bias  float  metres  (NaN if no predicted front in zone)
      area_error_km2        float  km²     (within front band)
      front_iou             float  [0,1]   (NaN if trace unavailable)
    """
    pred_sdt   = pred_sdt.astype(np.float32)
    target_sdt = target_sdt.astype(np.float32)

    # ── Fallback when trace is unavailable ───────────────────────────────
    if target_trace is None:
        warnings.warn(
            "target_trace is None — all metrics will be NaN. "
            "Ensure zarr['valid_trace'] is True before calling.",
            stacklevel=2,
        )
        return {k: NAN for k in [
            "mae_front", "rmse_front", "active_mde", "active_hausdorff",
            "advance_retreat_bias", "area_error_km2", "front_iou",
        ]}

    target_trace = target_trace.astype(bool)

    # ── Field-metric zone: adaptive union band ∩ active_mask ─────────────
    zone = build_evaluation_zone(
        target_sdt,
        persist_sdt  = persist_sdt,
        active_mask  = active_mask,
        front_band_m = front_band_m,
    )

    if not zone.any():
        warnings.warn(
            "Evaluation zone is empty — all metrics will be NaN. "
            "Check that zarr['trace'] (Trace_MASK.tif) is non-empty.",
            stacklevel=2,
        )
        return {k: NAN for k in [
            "mae_front", "rmse_front", "active_mde", "active_hausdorff",
            "advance_retreat_bias", "area_error_km2", "front_iou",
        ]}

    # ── Metrics 1–2: MAE and RMSE within front band ───────────────────────
    mae, rmse = _mae_rmse_in_zone(pred_sdt, target_sdt, zone)

    # ── Metrics 3–4: MDE and Hausdorff in active zone ───────────────────────────────────
    # evaluation zone could be too restrictive if we have large displacements of the calving front (>300m)
    # leads to NaNs in the MDE and Hausdorff metrics, so MDE/Hausdorff use the
    # raw active_mask alone.
    #
    # The predicted front is the subpixel zero-level-set polyline of pred_sdt
    # (find_contours). The target front is the actual rasterised trace mask
    # (zarr["trace"]) via trace_to_points — the real annotated front pixels,
    # not a zero-crossing derived from target_sdt.
    if active_mask is not None:
        active_zone = active_mask.astype(bool)
        pred_pts   = extract_predicted_front(pred_sdt,   active_zone, resolution_m)
        target_pts = trace_to_points(target_trace, active_zone, resolution_m)
        if pred_pts is not None and target_pts is not None:
            active_mde, active_hausdorff = _mde_hausdorff(pred_pts, target_pts)
        else:
            active_mde = active_hausdorff = NAN
    else:
        active_mde = active_hausdorff = NAN

    # ── Metric 5: Advance / retreat bias ─────────────────────────────────
    bias = _advance_retreat_bias(pred_sdt, target_sdt, zone)

    # ── Metric 6: Area error within front band ────────────────────────────
    area_err = _area_error_km2(pred_sdt, target_sdt, zone, resolution_m)

    # ── Metric 7: Front IoU ───────────────────────────────────────────────
    iou = _front_iou(pred_sdt, target_trace, zone, iou_dilation_m, resolution_m)

    return {
        "mae_front":            mae,
        "rmse_front":           rmse,
        # "mde":                  mde,
        # "hausdorff":            hausdorff,
        "active_mde":           active_mde,
        "active_hausdorff":     active_hausdorff,
        "advance_retreat_bias": bias,
        "area_error_km2":       area_err,
        "front_iou":            iou,
    }


# ── Skill score ───────────────────────────────────────────────────────────────

def compute_skill_score(
    model_mde:       float,
    persistence_mde: float,
) -> float:
    """
    Skill score of the model relative to the persistence baseline.

    Skill = 1 − (model_MDE / persistence_MDE)

    Interpretation
    --------------
      skill = 1.0   perfect prediction
      skill > 0.0   model beats persistence
      skill = 0.0   model equals persistence (no skill)
      skill < 0.0   model worse than persistence

    The persistence baseline predicts the front stays at its position in
    the most recent input time step (the last anchor slot, a_K = -10 days,
    as defined in 03_build_index.py).

    Returns NaN if either input is NaN or persistence_mde ≤ 0.
    """
    if (
        np.isnan(model_mde)
        or np.isnan(persistence_mde)
        or persistence_mde <= 0.0
    ):
        return NAN
    return float(1.0 - model_mde / persistence_mde)


# ── Batch evaluation ──────────────────────────────────────────────────────────

def evaluate_batch(
    pred_sdts:           np.ndarray,
    target_sdts:         np.ndarray,
    target_traces:       Optional[np.ndarray],
    persist_sdts:        Optional[np.ndarray] = None,
    active_masks:        Optional[np.ndarray] = None,
    spatial_valid_masks: Optional[np.ndarray] = None,
    original_hw:         Optional[list[tuple[int, int]]] = None,
    resolution_m:        float = DEFAULT_RESOLUTION_M,
    front_band_m:        float = DEFAULT_FRONT_BAND_M,
    **metric_kwargs,
) -> list[dict]:
    """
    Compute metrics for a batch of predictions, with support for padded
    variable-size patches from calving_collate_fn.

    Parameters
    ----------
    pred_sdts : (B, H, W) or (B, 1, H, W)
        Model predictions, denormalised to metres via denormalise_sdt.
    target_sdts : (B, H, W) or (B, 1, H, W)
        Ground truth SDT maps, denormalised to metres.
    target_traces : (B, H, W) bool or None
        Calving front trace masks — from zarr["trace"] (Trace_MASK.tif).
    persist_sdts : (B, H, W) or None
        Persistence baseline SDT maps (last input time step, denormalised).
        If provided, skill score is computed for each sample.
    active_masks : (B, H, W) bool or None
        Per-sample temporal activity masks (sdt_var > threshold).
        Computed from np.var(sdt_input_stack, axis=0) in evaluate.py.
        Static fjord boundary pixels (var ≈ 0) are excluded when provided.
    spatial_valid_masks : (B, 1, H_pad, W_pad) bool or None
        From batch["spatial_valid_mask"] (calving_collate_fn).
    original_hw : list of (H, W) tuples or None
        From batch["original_hw"] (calving_collate_fn).
    resolution_m : float
        Spatial resolution (default 30 m, EPSG:3995).
    front_band_m : float
    **metric_kwargs : passed to compute_metrics

    Returns
    -------
    List of metric dicts, one per sample. Each dict contains all metrics
    from compute_metrics, plus (when persist_sdts is provided):
      active_persist_mde, active_persist_hausdorff, active_skill_score,
      mean_true_displacement_m, signed_true_displacement_m,
      signed_pred_displacement_m
    """
    # Squeeze channel dimension if present
    def _squeeze(arr):
        return arr[:, 0] if arr is not None and arr.ndim == 4 else arr

    pred_sdts     = _squeeze(pred_sdts)
    target_sdts   = _squeeze(target_sdts)
    persist_sdts  = _squeeze(persist_sdts)
    target_traces = _squeeze(target_traces)
    active_masks  = _squeeze(active_masks)

    B = pred_sdts.shape[0]
    records = []

    for b in range(B):
        pred  = pred_sdts[b]
        target = target_sdts[b]
        trace  = target_traces[b]  if target_traces is not None else None
        pers   = persist_sdts[b]   if persist_sdts  is not None else None
        active = active_masks[b]   if active_masks  is not None else None

        # ── Crop padded pixels (padding is top-left-anchored, so cropping
        # to original_hw is sufficient; spatial_valid_mask not needed) ────
        if original_hw is not None:
            h, w = original_hw[b]
            pred   = pred[:h, :w]
            target = target[:h, :w]
            if trace  is not None: trace  = trace[:h, :w]
            if pers   is not None: pers   = pers[:h, :w]
            if active is not None: active = active[:h, :w]

        m = compute_metrics(
            pred, target, trace,
            active_mask  = active,
            persist_sdt  = pers,
            resolution_m = resolution_m,
            front_band_m = front_band_m,
            **metric_kwargs,
        )

        if pers is not None:
            # Same zone (built from target ∪ persist) as the model call above,
            # so model and persistence are scored on an identical region.
            p = compute_metrics(
                pers, target, trace,
                active_mask  = active,
                persist_sdt  = pers,
                resolution_m = resolution_m,
                front_band_m = front_band_m,
                **metric_kwargs,
            )
            # m["persist_mde"]       = p["mde"]
            # m["persist_hausdorff"] = p["hausdorff"]
            m["active_persist_mde"]       = p["active_mde"]
            m["active_persist_hausdorff"] = p["active_hausdorff"]
            # m["skill_score"]       = compute_skill_score(m["mde"], p["mde"])
            m["active_skill_score"] = compute_skill_score(m["active_mde"], p["active_mde"])


            # Mean absolute displacement at the calving front trace pixels.
            # Used downstream to stratify skill scores by displacement magnitude.
            #
            # The two signed variants pair the true and predicted front motion at
            # the same pixels, so they can be plotted against each other to expose
            # regression-to-the-mean (predicted magnitude systematically below
            # true magnitude).  Sign follows the ice convention (sdt > 0 = ice):
            # > 0 → advance, < 0 → retreat.
            if trace is not None and trace.astype(bool).any():
                front_px = trace.astype(bool)
                m["mean_true_displacement_m"] = float(
                    np.abs((target - pers)[front_px]).mean()
                )
                m["signed_true_displacement_m"] = float((target - pers)[front_px].mean())
                m["signed_pred_displacement_m"] = float((pred   - pers)[front_px].mean())
            else:
                m["mean_true_displacement_m"]   = float("nan")
                m["signed_true_displacement_m"] = float("nan")
                m["signed_pred_displacement_m"] = float("nan")

        records.append(m)

    return records


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_metrics(
    records:   list[dict],
    quantiles: list[float] = [0.5, 0.9, 0.95],
) -> dict:
    """
    Aggregate per-sample metric dicts into population statistics.

    Parameters
    ----------
    records : list of dicts from compute_metrics / evaluate_batch
    quantiles : quantiles to compute for each metric

    Returns
    -------
    dict with keys:
      {metric}_mean     float
      {metric}_std      float
      {metric}_q{q}     float   for each q in quantiles
      n_valid_{metric}  int     number of non-NaN samples per metric
    """
    if not records:
        return {}

    out = {}
    all_keys = list(dict.fromkeys(k for r in records for k in r.keys()))
    for key in all_keys:
        sample = next((r[key] for r in records if key in r), None)
        if not isinstance(sample, (int, float, np.floating, np.integer)):
            continue
        vals  = np.array([r[key] for r in records if key in r], dtype=np.float64)
        valid = vals[np.isfinite(vals)]

        out[f"{key}_mean"]    = float(np.mean(valid))   if len(valid) else NAN
        out[f"{key}_std"]     = float(np.std(valid))    if len(valid) else NAN
        # out[f"{key}_median"]  = float(np.median(valid)) if len(valid) else NAN
        out[f"n_valid_{key}"] = len(valid)

        for q in quantiles:
            qname = str(int(q * 100))
            out[f"{key}_q{qname}"] = (
                float(np.quantile(valid, q)) if len(valid) else NAN
            )

    # Skill score stratified by displacement magnitude (<100 m vs ≥100 m).
    # Slow-moving samples (most of the dataset) drag aggregate skill toward 0;
    # stratifying reveals whether the model has genuine skill on fast-movers.
    # if all("mean_true_displacement_m" in r and "skill_score" in r for r in records):
    #     for label, lo, hi in [("slow", 0.0, 100.0), ("fast", 100.0, float("inf"))]:
    #         subset = np.array(
    #             [r["skill_score"] for r in records
    #              if np.isfinite(r.get("mean_true_displacement_m", NAN))
    #              and lo <= r["mean_true_displacement_m"] < hi
    #              and np.isfinite(r.get("skill_score", NAN))],
    #             dtype=np.float64,
    #         )
    #         out[f"skill_score_{label}_mean"]   = float(subset.mean())   if len(subset) else NAN
    #         out[f"skill_score_{label}_median"] = float(np.median(subset)) if len(subset) else NAN
    #         out[f"n_{label}"]                  = len(subset)


    # Active Skill score stratified by displacement magnitude (<100 m vs ≥100 m).
    # Slow-moving samples (most of the dataset) drag aggregate skill toward 0;
    # stratifying reveals whether the model has genuine skill on fast-movers.
    if all("mean_true_displacement_m" in r and "active_skill_score" in r for r in records):
        for label, lo, hi in [("slow", 0.0, 100.0), ("fast", 100.0, float("inf"))]:
            subset = np.array(
                [r["active_skill_score"] for r in records
                 if np.isfinite(r.get("mean_true_displacement_m", NAN))
                 and lo <= r["mean_true_displacement_m"] < hi
                 and np.isfinite(r.get("active_skill_score", NAN))],
                dtype=np.float64,
            )
            out[f"active_skill_score_{label}_mean"]   = float(subset.mean())   if len(subset) else NAN
            # out[f"active_skill_score_{label}_median"] = float(np.median(subset)) if len(subset) else NAN
            out[f"n_{label}"]                        = len(subset)

    return out


# ── Denormalisation helper ────────────────────────────────────────────────────

def denormalise_sdt(
    normalised_sdt: np.ndarray,
    stats:          dict,
) -> np.ndarray:
    """
    Reverse z-score normalisation for SDT arrays before metric computation.

    All metrics expect values in metres — call this on model outputs and
    zarr["sdt"] targets before passing to compute_metrics / evaluate_batch.

    Parameters
    ----------
    normalised_sdt : array of any shape
        Normalised SDT values (model output or normalised zarr["sdt"]).
    stats : dict
        From normalisation_stats.json — must have stats["sdt"]["mean"] and
        stats["sdt"]["std"], computed by compute_and_save_stats in dataset.py.

    Returns
    -------
    Array in metres (float32), same shape as input.
    """
    mean = float(stats["sdt"]["mean"])
    std  = float(stats["sdt"]["std"])
    return (normalised_sdt * std + mean).astype(np.float32)


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running metrics self-test with synthetic 30 m data...\n")

    rng = np.random.default_rng(0)
    H, W = 256, 256
    res  = 30.0   # 30 m EPSG:3995

    # Synthetic ground truth: glacier occupies upper half, SDT clipped ±2000 m
    target_sdt = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        target_sdt[r, :] = np.clip((r - H // 2) * res, -2000, 2000)

    # True front: pixels at row H//2 (Trace_MASK.tif equivalent)
    target_trace = np.zeros((H, W), dtype=bool)
    target_trace[H // 2, :] = True

    # Active mask: band around the front — mirrors the real pipeline (variance
    # is high near the front) and keeps np.roll wrap artifacts at the image
    # edges out of the front extraction.
    active_band = np.zeros((H, W), dtype=bool)
    active_band[H // 2 - 20 : H // 2 + 20, :] = True

    # ── Case 1: perfect prediction ────────────────────────────────────────
    m_perfect = compute_metrics(
        target_sdt, target_sdt, target_trace,
        active_mask=active_band, resolution_m=res,
    )
    print("Perfect prediction (pred = target):")
    for k, v in m_perfect.items():
        print(f"  {k:30s} = {v:.4f}")
    assert m_perfect["active_mde"] < res, "Perfect prediction MDE should be ≈ 0"

    # ── Case 2: 210 m advance shift (7 pixels at 30 m) ───────────────────
    shift_px   = int(210 / res)   # 7 pixels
    shifted    = np.roll(target_sdt, -shift_px, axis=0).astype(np.float32)
    m_shifted  = compute_metrics(
        shifted, target_sdt, target_trace,
        active_mask=active_band, resolution_m=res,
    )
    print(f"\nPrediction with {shift_px * res:.0f} m advance shift:")
    for k, v in m_shifted.items():
        print(f"  {k:30s} = {v:.4f}")
    assert m_shifted["advance_retreat_bias"] > 0, "Advance bias should be positive"
    assert abs(m_shifted["active_mde"] - shift_px * res) < res, \
        f"MDE should be ≈ {shift_px * res:.0f} m, got {m_shifted['active_mde']:.1f}"

    # ── Case 3: skill score ───────────────────────────────────────────────
    # Model off by 210 m, persistence off by 90 m → model worse, skill < 0.
    retreat_px = int(90 / res)   # 3 pixels at 30 m
    retreated  = np.roll(target_sdt, retreat_px, axis=0).astype(np.float32)
    m_persist  = compute_metrics(
        retreated, target_sdt, target_trace,
        active_mask=active_band, resolution_m=res,
    )
    skill = compute_skill_score(m_shifted["active_mde"], m_persist["active_mde"])
    print(f"\nSkill score (model vs persistence): {skill:.4f}")
    assert skill < 0, "210 m model vs 90 m persistence error should give negative skill"

    # ── Case 4: batch with spatial_valid_mask cropping ───────────────────
    # Simulate two padded samples: second sample is smaller (200×200) padded to 256×256
    H2, W2 = 200, 200
    tgt2   = target_sdt[:H2, :W2]
    trc2   = target_trace[:H2, :W2]
    pred2  = shifted[:H2, :W2]

    pad_pred   = np.zeros((2, H, W), dtype=np.float32)
    pad_target = np.zeros((2, H, W), dtype=np.float32)
    pad_trace  = np.zeros((2, H, W), dtype=bool)

    pad_pred[0],  pad_target[0] = target_sdt, target_sdt
    pad_trace[0]                = target_trace
    pad_pred[1,:H2,:W2]   = pred2
    pad_target[1,:H2,:W2] = tgt2
    pad_trace[1,:H2,:W2]  = trc2

    svm = np.zeros((2, 1, H, W), dtype=bool)
    svm[0, 0] = True
    svm[1, 0, :H2, :W2] = True

    records = evaluate_batch(
        pad_pred, pad_target, pad_trace,
        spatial_valid_masks=svm,
        original_hw=[(H, W), (H2, W2)],
        resolution_m=res,
    )
    agg = aggregate_metrics(records)
    print(f"\nBatch aggregation (2 samples, one padded):")
    for k, v in agg.items():
        if "mean" in k:
            print(f"  {k:35s} = {v:.4f}")

    # ── Case 5: adaptive union zone (target ∪ persist bands) ∩ active ─────
    # Persist front 450 m (15 px) below the target front → the union band must
    # span the whole corridor between them; ∩ active must only remove pixels.
    active_mask_test = np.zeros((H, W), dtype=bool)
    active_mask_test[:, W // 2:] = True   # right half active

    persist_sdt = np.roll(target_sdt, 15, axis=0).astype(np.float32)  # 450 m corridor

    zone_target_only  = build_evaluation_zone(target_sdt, active_mask=None)
    zone_union        = build_evaluation_zone(target_sdt, persist_sdt=persist_sdt,
                                              active_mask=None)
    zone_union_active = build_evaluation_zone(target_sdt, persist_sdt=persist_sdt,
                                              active_mask=active_mask_test)

    # Union must contain the target-only band and be strictly wider (persist band)
    assert (zone_target_only & ~zone_union).sum() == 0, \
        "union band must contain the target band"
    assert zone_union.sum() > zone_target_only.sum(), \
        "persist band must widen the zone (fronts are 450 m apart)"

    # ∩ active_mask must only remove pixels, never add them
    assert (zone_union_active & ~zone_union).sum() == 0, \
        "active_mask intersection should only remove pixels, never add them"
    left_before = zone_union[:, :W // 2]
    left_after  = zone_union_active[:, :W // 2]
    if left_before.any():
        assert (~left_after[left_before]).all(), \
            "Inactive (left-half) pixels should be excluded"

    print("\nAdaptive union zone (target ∪ persist) ∩ active OK")

    # Verify compute_metrics accepts the new persist_sdt path without error
    m_with_active = compute_metrics(
        target_sdt, target_sdt, target_trace,
        active_mask=active_mask_test,
        persist_sdt=persist_sdt,
        resolution_m=res,
    )
    assert not np.isnan(m_with_active["mae_front"]), \
        "compute_metrics with active_mask should not return NaN on perfect prediction"
    print("compute_metrics with active_mask OK")

    # ── Denormalise roundtrip ─────────────────────────────────────────────
    stats = {"sdt": {"mean": -50.0, "std": 600.0}}
    orig  = target_sdt
    norm  = (orig - (-50.0)) / 600.0
    back  = denormalise_sdt(norm, stats)
    assert np.allclose(back, orig, atol=1e-3), "Denormalise roundtrip failed"
    print("\nDenormalise roundtrip OK")

    print("\nSelf-test passed.")
