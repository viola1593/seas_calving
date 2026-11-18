"""
Long-term autoregressive rollout evaluation for the seasonal FiLM model.

This module is structured as:

  * PURE functions (importable, GPU/zarr/torch-free) — unit-tested in
    tests/test_rollout_*.py.
  * INTEGRATION engine + CLI (GPU + zarr + checkpoint) — smoke-tested only.

Torch/zarr are imported lazily inside the integration functions only, so that
`import rollout_film` stays cheap and side-effect-free for the pure unit tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label


# ══════════════════════════════════════════════════════════════════════════
# 2a. Buffer & date matching
# ══════════════════════════════════════════════════════════════════════════

def frame_dates(sample: dict) -> list[date]:
    """
    Calendar date of every input frame of a sample.

    t_ref = date(sample["target_date"]) - sample["horizon_days"]
    return [t_ref + timedelta(days=input_days[k]) for k in range(K)]

    where input_days = sample["input_days"] (signed day offsets from t_ref,
    K = len(input_days)). Mirrors data/dataset_v2.py exactly.
    """
    target_date = date.fromisoformat(sample["target_date"])
    t_ref = target_date - timedelta(days=int(sample["horizon_days"]))
    return [t_ref + timedelta(days=int(off)) for off in sample["input_days"]]


@dataclass
class FrameProvenance:
    """Where a single input frame's SDT came from after substitution."""
    synthetic: bool
    borrowed_date: Optional[date]   # producing sample's target_date, or None if real
    day_offset: Optional[int]       # frame_date - borrowed_date, signed days; None if real


class PredictionBuffer:
    """
    Store of past predicted SDT frames (in NORMALISED space), keyed by the
    target_date of the sample that produced them. Entries are REUSED, never
    consumed: one prediction may serve many frames and many steps.
    """

    def __init__(self) -> None:
        self._entries: dict[date, np.ndarray] = {}

    def add(self, target_date: date, pred_sdt_norm: np.ndarray) -> None:
        """Store predicted normalised SDT keyed by producing sample's target_date."""
        self._entries[target_date] = pred_sdt_norm

    def match(self, frame_date: date, delta_days: int) -> Optional[tuple[date, np.ndarray]]:
        """
        Nearest buffered entry with |frame_date - key| <= delta_days.

        Returns (key_date, pred_sdt_norm) or None if the buffer is empty or all
        entries are out of tolerance. Tie-break: smaller |Δdays| first, then the
        LATER key date (most detached prediction). Entries are never removed.
        """
        best_key: Optional[date] = None
        best_absdelta: Optional[int] = None
        for key in self._entries:
            absdelta = abs((frame_date - key).days)
            if absdelta >= delta_days:
                continue
            # smaller |Δdays| first, then later key date
            if (best_key is None
                    or absdelta < best_absdelta
                    or (absdelta == best_absdelta and key > best_key)):
                best_key = key
                best_absdelta = absdelta
        if best_key is None:
            return None
        return best_key, self._entries[best_key]


def substitute_sdt(
    inputs_sdt_norm: np.ndarray,        # (K, H, W) real normalised channel-0 SDT
    frame_dates: list[date],
    buffer: PredictionBuffer,
    delta_days: int,
) -> tuple[np.ndarray, list[FrameProvenance]]:
    """
    For each frame, overwrite the SDT with the matched buffered prediction if
    one exists within delta_days, else keep the real SDT. Returns a COPY of
    inputs_sdt_norm (the input is never mutated) plus a per-frame provenance
    list (length K).
    """
    out = inputs_sdt_norm.copy()
    prov: list[FrameProvenance] = []
    for k, fd in enumerate(frame_dates):
        matched = buffer.match(fd, delta_days)
        if matched is None:
            prov.append(FrameProvenance(synthetic=False, borrowed_date=None, day_offset=None))
        else:
            key_date, pred_sdt_norm = matched
            out[k] = pred_sdt_norm
            prov.append(FrameProvenance(
                synthetic=True,
                borrowed_date=key_date,
                day_offset=(fd - key_date).days,
            ))
    return out, prov


def recompute_sdt_diff(sdt_norm: np.ndarray) -> np.ndarray:   # (K,H,W) -> (K,H,W)
    """Recompute channel-1 sdt_diff, mirroring data/dataset_v2.py exactly."""
    return np.diff(sdt_norm, axis=0, prepend=sdt_norm[[0]])


def autoreg_depth(prov: list[FrameProvenance]) -> int:
    """Number of synthetic frames in a provenance list (0..K)."""
    return sum(1 for p in prov if p.synthetic)


# ══════════════════════════════════════════════════════════════════════════
# 2b. Reconstruction math
# ══════════════════════════════════════════════════════════════════════════

def reconstruct_pred_sdt_norm(
    pred_disp_norm: np.ndarray,
    last_sdt_norm: np.ndarray,
    disp_scale: float,
) -> np.ndarray:
    """pred_disp_norm / disp_scale + last_sdt_norm   (evaluate_film.py:209)."""
    return pred_disp_norm / disp_scale + last_sdt_norm


def denorm_sdt(sdt_norm: np.ndarray, sdt_mean: float, sdt_std: float) -> np.ndarray:
    """Normalised SDT -> metres:  sdt_norm * sdt_std + sdt_mean."""
    return sdt_norm * sdt_std + sdt_mean


def norm_sdt(sdt_m: np.ndarray, sdt_mean: float, sdt_std: float) -> np.ndarray:
    """Metres -> normalised SDT:  (sdt_m - sdt_mean) / sdt_std."""
    return (sdt_m - sdt_mean) / sdt_std


# ══════════════════════════════════════════════════════════════════════════
# 2c. Eval zone
# ══════════════════════════════════════════════════════════════════════════

def compute_active_mask(real_sdt_stack_m: np.ndarray, threshold_m2: float = 100.0) -> np.ndarray:
    """np.var(real_sdt_stack_m, axis=0) > threshold_m2  (matches metrics threshold)."""
    return np.var(real_sdt_stack_m, axis=0) > threshold_m2


def persistence_sdt_m(t0_real_sdt_m: np.ndarray) -> np.ndarray:
    """Frozen front at t0 (identity)."""
    return np.array(t0_real_sdt_m)


def reaches_full_depth(max_depth: int, K: int) -> bool:
    """True iff a glacier reached a fully-synthetic window (max_depth == K)."""
    return bool(max_depth == K)


# ══════════════════════════════════════════════════════════════════════════
# 2d. Records & output
# ══════════════════════════════════════════════════════════════════════════

def build_record(
    glacier_id,
    step: int,
    autoreg_depth: int,
    lead_time_days: int,
    target_date: date,
    model_m: dict,          # metrics dict
    teacher_m: dict,        # metrics dict
    persist_m: dict,        # metrics dict
    n_frames_synthetic: int,
    borrowed_dates,
    max_date_offset,
) -> dict:
    """
    Flat record for one rollout step. Metric dicts are flattened with the
    prefixes model_ / teacher_ / persist_ (e.g. model_m["active_mde"] ->
    "model_active_mde").

    Column order: leading identity cols, then the model_ / teacher_ / persist_
    metric blocks (each metric dict flattened in its own key order), then the
    trailing bookkeeping cols. Python dicts preserve insertion order, so this
    fixes the DataFrame column order downstream.
    """
    rec: dict = {
        "glacier_id":     glacier_id,
        "step":           step,
        "autoreg_depth":  autoreg_depth,
        "lead_time_days": lead_time_days,
        "target_date":    target_date,
    }
    for prefix, metrics in (("model", model_m), ("teacher", teacher_m), ("persist", persist_m)):
        for key, val in metrics.items():
            rec[f"{prefix}_{key}"] = val
    rec["n_frames_synthetic"] = n_frames_synthetic
    rec["borrowed_dates"]     = borrowed_dates
    rec["max_date_offset"]    = max_date_offset
    return rec


def records_to_dataframe(records: list[dict]) -> pd.DataFrame:
    """Records -> DataFrame with a stable, deterministic column order.

    Column order follows the insertion order of the first record (all records
    share the same schema via build_record), giving a deterministic layout with
    leading identity cols first and trailing bookkeeping cols last.
    """
    if not records:
        return pd.DataFrame()
    columns = list(records[0].keys())
    return pd.DataFrame(records, columns=columns)


def plot_divergence(df: pd.DataFrame, out_path) -> None:
    """Divergence plot (smoke only, not unit-tested).

    Model / teacher-forced / persistence active_mde vs lead time, averaged
    across glaciers at each lead-time bin. Illustrates how autoregressive error
    diverges from the teacher-forced upper bound as the rollout detaches.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lead = sorted(df["lead_time_days"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    series = [
        ("model_active_mde",   "model (autoregressive)", "C0"),
        ("teacher_active_mde", "teacher-forced",         "C1"),
        ("persist_active_mde", "persistence",            "C2"),
    ]
    for col, label, color in series:
        if col not in df.columns:
            continue
        means = [df.loc[df["lead_time_days"] == lt, col].mean() for lt in lead]
        ax.plot(lead, means, marker="o", label=label, color=color)

    ax.set_xlabel("lead time (days)")
    ax.set_ylabel("active MDE (m)")
    ax.set_title("Autoregressive rollout divergence")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════
# 2e. Engine glue (integration, smoke-tested only). torch/zarr imported lazily.
# ══════════════════════════════════════════════════════════════════════════

def _sample_target_date(s: dict) -> date:
    """Target calendar date, supporting both index schemas (mirrors dataset_v2)."""
    if "target_date" in s:
        return date.fromisoformat(s["target_date"])
    return date.fromisoformat(s["reference_date"]) + timedelta(days=int(s["target_day"]))


def _normalized_sample(s: dict) -> dict:
    """Minimal sample view with the keys frame_dates expects (target_date/horizon_days).

    For the old reference_date/target_day schema, dataset_v2 uses target_day as
    the horizon so t_ref == reference_date; we reproduce that exactly so the
    input frame dates match the real observation dates.
    """
    if "target_date" in s:
        return {"target_date": s["target_date"], "horizon_days": int(s["horizon_days"]),
                "input_days": s["input_days"]}
    horizon = int(s["target_day"])
    return {"target_date": _sample_target_date(s).isoformat(), "horizon_days": horizon,
            "input_days": s["input_days"]}


def load_glacier_sample_indices(dataset, glacier_id) -> list[int]:
    """Dataset indices for one glacier, sorted by target_date (ties by target_zarr_index)."""
    idxs = [i for i, s in enumerate(dataset.samples) if s["glacier_id"] == glacier_id]
    idxs.sort(key=lambda i: (_sample_target_date(dataset.samples[i]),
                             int(dataset.samples[i]["target_zarr_index"])))
    return idxs


def glacier_zarr_indices(dataset, indices) -> list[int]:
    """Sorted union of all real SDT zarr slots (input + target) across a glacier's samples."""
    zi: set[int] = set()
    for i in indices:
        s = dataset.samples[i]
        zi.update(int(z) for z in s["input_zarr_indices"])
        zi.add(int(s["target_zarr_index"]))
    return sorted(zi)


def gather_real_sdt_stack(store, zarr_indices) -> np.ndarray:
    """Real SDT slots in metres, shape (N, H, W), for the frozen active_mask.

    ``zarr_indices`` is the union of input+target zarr slots for one glacier
    (see glacier_zarr_indices). zarr["sdt"] is stored in metres already, so no
    denormalisation is needed — NaNs (out-of-domain) are zeroed to match the
    dataset's np.nan_to_num handling.
    """
    zi = sorted(int(z) for z in zarr_indices)
    return np.nan_to_num(store["sdt"][zi].astype(np.float32))   # (N, H, W)


def field_grad_invalid_frac(pred_sdt_m: np.ndarray, mask: np.ndarray,
                            resolution_m: float = 30.0, tol: float = 0.5) -> float:
    """Action Item 9 diagnostic: fraction of near-front pixels where ||∇pred_sdt| − 1| > tol.

    A valid signed-distance field has unit gradient (|∇|=1). The raw fed-back
    reconstruction (last_sdt + pred_disp) is not guaranteed to be one; this
    measures how far it drifts, over the frozen active (near-front) mask.
    """
    sel = mask.astype(bool)
    if not sel.any():
        return float("nan")
    gy, gx = np.gradient(pred_sdt_m.astype(np.float32), resolution_m, resolution_m)
    grad_mag = np.sqrt(gy ** 2 + gx ** 2)
    return float((np.abs(grad_mag[sel] - 1.0) > tol).mean())


# Mirrors preprocessing/02_seasonal_observations.py compute_sdt.
SDT_CLIP_M = 2000.0


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the biggest connected True-component of a boolean mask."""
    lab, n = label(mask)
    if n <= 1:
        return mask
    counts = np.bincount(lab.ravel())
    counts[0] = 0                                   # ignore background
    return lab == counts.argmax()


def redistance_sdt_m(pred_sdt_m: np.ndarray, resolution_m: float = 30.0,
                     clip_m: float = SDT_CLIP_M) -> np.ndarray:
    """Reinitialise a drifted field to a true SDF (|∇|=1) with a single front.

    Threshold at the zero level set, keep only the largest ice body and fill
    enclosed ocean pockets (glaciers are guaranteed single-front), then recompute
    the signed EDT exactly as preprocessing.compute_sdt does. Preserves the
    predicted front to ±half a pixel; only repairs the field away from it and
    removes spurious disconnected fronts.
    """
    ice = pred_sdt_m >= 0.0
    if ice.all() or not ice.any():
        return pred_sdt_m
    ice = binary_fill_holes(_largest_component(ice))
    sdt = (distance_transform_edt(ice) - distance_transform_edt(~ice)) * resolution_m
    return np.clip(sdt, -clip_m, clip_m).astype(np.float32)


def front_component_count(pred_sdt_m: np.ndarray) -> int:
    """Number of distinct closed fronts in a prediction (ice blobs + enclosed
    ocean pockets). A clean single-terminus prediction is 1; fragmentation
    raises it. Fragmentation diagnostic for the divergence analysis.
    """
    ice = pred_sdt_m >= 0.0
    if ice.all() or not ice.any():
        return 0
    _, n_ice = label(ice)
    _, n_holes = label(binary_fill_holes(ice) & ~ice)   # enclosed ocean pockets
    return int(n_ice + n_holes)


def _model_predict_sdt_norm(module, sample: dict, inputs_tensor, device) -> np.ndarray:
    """Run one forward pass and reconstruct absolute normalised SDT at the target.

    Returns pred_sdt_norm (H, W): pred_disp_norm / disp_scale + last_input_sdt_norm,
    where last_input_sdt_norm is the (possibly substituted) final input frame.
    """
    import torch
    from data.dataset_v2 import collate_fn

    disp_scale = module.model.disp_scale.item()

    s = dict(sample)
    s["inputs"] = inputs_tensor
    batch = collate_fn([s])
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    with torch.no_grad():
        out = module.model(batch)
    pred = out[0] if isinstance(out, (tuple, list)) else out   # (1, 1, H, W)
    pred_disp_norm = pred.detach().cpu().float().numpy()[0, 0]  # (H, W)

    last_sdt_norm = inputs_tensor[-1, 0].detach().cpu().float().numpy()  # (H, W)
    return reconstruct_pred_sdt_norm(pred_disp_norm, last_sdt_norm, disp_scale)


def rollout_glacier(module, dataset, glacier_id, frozen_mask, delta_days: int = 30,
                    indices=None, plot_out: Optional[list] = None,
                    redistance: bool = True, freeze: str = "none", **kwargs) -> list[dict]:
    """Autoregressive rollout over one glacier's samples. Returns records.

    Per sample, chronological: substitute real input SDT frames with buffered
    predictions where dates match (autoregressive feedback), run the model on
    both the substituted inputs (model / autoregressive) and the real inputs
    (teacher-forced), and score both plus a t0-frozen persistence baseline on
    the shared frozen active_mask. The autoregressive prediction is buffered to
    feed later steps. τ / climate keep each frame's true date and real values;
    only channel-0 SDT (and the recomputed channel-1 sdt_diff) are substituted.
    """
    import torch
    from metrics import compute_metrics

    if indices is None:
        indices = load_glacier_sample_indices(dataset, glacier_id)
        print(f"Rollout glacier {glacier_id}: {len(indices)} samples, delta_days={delta_days}")
    if not indices:
        return []

    device = next(module.parameters()).device

    disp_scale = module.model.disp_scale.item()
    sdt_std    = disp_scale * float(module.hparams.disp_std)
    sdt_mean   = float(module.hparams.sdt_mean)

    first_s = dataset.samples[indices[0]]
    store   = dataset._get_store(first_s["zarr_path"])

    # Rollout persistence: front frozen at the first sample's real anchor (t0).
    t0_zi         = int(first_s["input_zarr_indices"][-1])
    persist_sdt_m = np.nan_to_num(store["sdt"][t0_zi].astype(np.float32))
    t0_date       = frame_dates(_normalized_sample(first_s))[-1]

    buffer  = PredictionBuffer()
    records: list[dict] = []
    from tqdm import tqdm

    for step, idx in tqdm(enumerate(indices), total=len(indices)):
        s  = dataset.samples[idx]
        ti = int(s["target_zarr_index"])
        target_date = _sample_target_date(s)
        fdates      = frame_dates(_normalized_sample(s))

        sample = dataset[idx]                              # normalised tensors
        real_inputs = sample["inputs"]                     # (K, C, H, W)
        sdt_norm_real = real_inputs[:, 0].numpy().copy()   # (K, H, W)
        K = sdt_norm_real.shape[0]

        # Snapshot the first sample's full input state for the freeze ablations.
        if step == 0:
            frozen_sample = {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in sample.items()}
            frozen_inputs = real_inputs.clone()

        new_sdt_norm, prov = substitute_sdt(sdt_norm_real, fdates, buffer, delta_days)

        # Autoregressive inputs: substituted SDT + recomputed sdt_diff.
        auto_inputs = real_inputs.clone()
        auto_inputs[:, 0] = torch.from_numpy(new_sdt_norm)
        auto_inputs[:, 1] = torch.from_numpy(recompute_sdt_diff(new_sdt_norm))

        # Model-reliance ablations. All hold channels 0-1 (SDF + sdt_diff)
        # substituted with the fed-back prediction, so the forecast is always a
        # real autoregressive rollout, and all leave τ (observation timing) live
        # so every mode sees the true calendar cadence. They differ in what else
        # is frozen at the first sample: --freeze features holds the historical
        # input state (geophys features, historical climate) AND forcing_climate
        # fixed; --freeze history is the same but lets the future forcing_climate
        # stay live; --freeze geophys freezes ONLY the geophysical spatial channels
        # (2+) and lets all climate (history + forcing) evolve live; --freeze
        # climate is the mirror image: it freezes all climate (history + forcing)
        # at the first sample and lets the geophysical channels evolve live.
        # Teacher-forced / persistence stay live as references.
        auto_sample, auto_in = sample, auto_inputs
        if freeze in ("features", "history"):
            auto_in = frozen_inputs.clone()
            auto_in[:, 0] = torch.from_numpy(new_sdt_norm)
            auto_in[:, 1] = torch.from_numpy(recompute_sdt_diff(new_sdt_norm))
            auto_sample = {**frozen_sample, "tau": sample["tau"]}
            if freeze == "history":
                auto_sample = {**auto_sample, "forcing_climate": sample["forcing_climate"]}
        elif freeze == "geophys":
            auto_in = auto_inputs.clone()
            auto_in[:, 2:] = frozen_inputs[:, 2:]
        elif freeze == "climate":
            auto_sample = {**sample,
                           "climate": frozen_sample["climate"],
                           "forcing_climate": frozen_sample["forcing_climate"]}

        auto_sdt_norm    = _model_predict_sdt_norm(module, auto_sample, auto_in, device)
        teacher_sdt_norm = _model_predict_sdt_norm(module, sample, real_inputs, device)

        # Feed the autoregressive prediction forward (never the teacher one).
        # With --redistance the *buffered* frame is snapped back to a valid SDF
        # so the next input stays in-distribution; the scored prediction below is
        # left raw so the divergence analysis stays honest.
        if redistance:
            buf_sdt_m = redistance_sdt_m(denorm_sdt(auto_sdt_norm, sdt_mean, sdt_std))
            buffer.add(target_date, norm_sdt(buf_sdt_m, sdt_mean, sdt_std))
        else:
            buffer.add(target_date, auto_sdt_norm)

        model_sdt_m   = denorm_sdt(auto_sdt_norm,    sdt_mean, sdt_std)
        teacher_sdt_m = denorm_sdt(teacher_sdt_norm, sdt_mean, sdt_std)

        target_sdt_m = np.nan_to_num(store["sdt"][ti].astype(np.float32))
        if "trace" in store and bool(store["valid_trace"][ti]):
            target_trace = store["trace"][ti].astype(bool)
        else:
            target_trace = None

        def _score(pred_m):
            return compute_metrics(pred_m, target_sdt_m, target_trace,
                                   active_mask=frozen_mask, persist_sdt=persist_sdt_m)

        model_m   = _score(model_sdt_m)
        teacher_m = _score(teacher_sdt_m)
        persist_metrics = _score(persist_sdt_m)

        # Field-validity diagnostics live in the model block (Action Item 9):
        # eikonal drift and front fragmentation of the raw scored prediction.
        model_m["field_grad_invalid_frac"] = field_grad_invalid_frac(model_sdt_m, frozen_mask)
        model_m["front_components"]        = front_component_count(model_sdt_m)

        depth   = autoreg_depth(prov)
        syn     = [p for p in prov if p.synthetic]
        rec = build_record(
            glacier_id=glacier_id,
            step=step,
            autoreg_depth=depth,
            lead_time_days=(target_date - t0_date).days,
            target_date=target_date,
            model_m=model_m,
            teacher_m=teacher_m,
            persist_m=persist_metrics,
            n_frames_synthetic=depth,
            borrowed_dates=[p.borrowed_date.isoformat() for p in syn],
            max_date_offset=max((abs(p.day_offset) for p in syn), default=0),
        )
        records.append(rec)

        # Stash spatial arrays for fully-synthetic steps so a downstream plotter
        # can render them with plot_samples.plot_glacier_predictions. The input
        # fronts drawn are the fed-back synthetic predictions (all K substituted).
        if plot_out is not None and depth == K:
            h, w        = int(sample["inputs"].shape[-2]), int(sample["inputs"].shape[-1])
            all_input_m = denorm_sdt(new_sdt_norm, sdt_mean, sdt_std)   # (K, H, W) synthetic inputs
            last_m      = all_input_m[-1]
            ttr         = (target_trace.astype(np.float32)
                           if target_trace is not None else np.zeros_like(last_m))
            skill       = (1.0 - model_m["active_mde"] / persist_metrics["active_mde"]
                           if np.isfinite(persist_metrics["active_mde"])
                           and persist_metrics["active_mde"] > 0 else float("nan"))
            plot_out.append({
                "last_sdt_m":         last_m,
                "pred_disp_m":        model_sdt_m - last_m,
                "true_disp_m":        target_sdt_m - last_m,
                "pred_sdt_m":         model_sdt_m,
                "input_trace":        sample["input_trace"][0].numpy(),
                "target_trace":       ttr,
                "all_input_sdts_m":   all_input_m,
                "input_dates":        [d.isoformat() for d in fdates],
                "original_hw":        (h, w),
                "glacier_id":         glacier_id,
                "target_date":        target_date.isoformat(),
                "active_skill_score": skill,
                "active_mde":         model_m["active_mde"],
                "autoreg_depth":      depth,
            })

    return records


# ── CLI ────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    """CLI: --experiment <dir>, --gpu, --delta-days 30, --split, --chkpt."""
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="Long-term autoregressive rollout evaluation (seasonal FiLM)."
    )
    p.add_argument("--experiment", type=Path, required=True,
                   help="Experiment dir (e.g. experiments/seasonal/paper_M3_seed3).")
    p.add_argument("--split", type=str, default="test", choices=["val", "test"])
    p.add_argument("--chkpt", type=str, default="best")
    p.add_argument("--gpu", type=str, default="0", help="CUDA device index (sets CUDA_VISIBLE_DEVICES).")
    p.add_argument("--delta-days", type=int, default=30,
                   help="Date tolerance for matching a buffered prediction to an input frame.")
    p.add_argument("--redistance", action=argparse.BooleanOptionalAction, default=True,
                   help="Reinitialise each fed-back frame to a valid single-front SDF "
                        "(keeps autoregressive inputs in-distribution; scored preds stay raw). "
                        "On by default; disable with --no-redistance.")
    p.add_argument("--freeze", choices=["none", "history", "features", "geophys", "climate"],
                   default="none",
                   help="Model-reliance ablation on the autoregressive prediction; the "
                        "SDF/sdt_diff always stay autoregressive and tau (observation timing) "
                        "always stays live. 'features' freezes the historical inputs (geophys, "
                        "historical climate) AND forcing_climate at the first sample; 'history' "
                        "is the same but leaves the future forcing_climate live; 'geophys' "
                        "freezes only the geophysical spatial channels and lets all climate "
                        "(history + forcing) evolve live; 'climate' is the mirror image, "
                        "freezing all climate (history + forcing) and letting geophys evolve. "
                        "Teacher-forced / persistence baselines stay live in every mode.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output dir (default: <experiment>/rollout/).")
    return p.parse_args(argv)


def main(argv=None) -> None:
    import os

    args = parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    import torch
    from evaluate_film import find_best_checkpoint, load_module_and_config
    from data.dataset_v2 import CalvingFrontDatasetV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = find_best_checkpoint(args.experiment, chkpt=args.chkpt)
    module, cfg = load_module_and_config(ckpt, args.experiment, device)
    module.eval()

    from pathlib import Path
    splits_dir = Path(cfg["data"]["splits_dir"])
    dataset = CalvingFrontDatasetV2(
        index_path=splits_dir / f"{args.split}.json",
        config=cfg["data"]["config"],
        stats_path=cfg["data"]["stats_path"],
        augment=False,
    )

    out_dir = args.out or (args.experiment / "rollout")
    out_dir.mkdir(parents=True, exist_ok=True)

    glacier_ids = sorted({s["glacier_id"] for s in dataset.samples})
    all_records: list[dict] = []
    for gid in glacier_ids:
        indices = load_glacier_sample_indices(dataset, gid)
        if len(indices) < 2:
            continue
        store       = dataset._get_store(dataset.samples[indices[0]]["zarr_path"])
        real_stack  = gather_real_sdt_stack(store, glacier_zarr_indices(dataset, indices))
        frozen_mask = compute_active_mask(real_stack)
        recs = rollout_glacier(module, dataset, gid, frozen_mask,
                               delta_days=args.delta_days, indices=indices,
                               redistance=args.redistance, freeze=args.freeze)
        all_records.extend(recs)
        max_depth = max((r["autoreg_depth"] for r in recs), default=0)
        print(f"{gid}: {len(recs)} steps, max autoreg_depth={max_depth}")

    tag = "" if args.freeze == "none" else f"_{args.freeze}"
    df = records_to_dataframe(all_records)
    csv_path = out_dir / f"{args.split}_rollout{tag}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nWrote {len(df)} rows to {csv_path}")

    # Feasibility filter (Action Item 8): headline curves use only glaciers that
    # reach a fully-synthetic window (max autoreg_depth == K).
    if not df.empty:
        K = len(dataset.samples[0]["input_zarr_indices"])
        max_depth = df.groupby("glacier_id")["autoreg_depth"].max()
        headline = [g for g, d in max_depth.items() if reaches_full_depth(int(d), K)]
        print(f"Feasible (depth==K={K}) glaciers: {len(headline)}/{df['glacier_id'].nunique()}")
        plot_df = df[df["glacier_id"].isin(headline)] if headline else df
        plot_path = out_dir / f"{args.split}_divergence{tag}.png"
        plot_divergence(plot_df, plot_path)
        print(f"Wrote divergence plot to {plot_path}")


if __name__ == "__main__":
    main()
