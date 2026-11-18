# data/dataset_v2.py
"""
CalvingFrontDatasetV2 — calving front prediction with configurable variables
and future forcings.

Variable selection is configured via a YAML file (or plain dict):

    inputs:
      spatial: [sdt, geophys]    # → inputs  (K, C_in, H, W)
      scalar:  [climate]         # → climate (K, D_c)
    forcings:                    # single t+horizon snapshot at target_zarr_index
      spatial: [geophys]         # → forcing_spatial (C_f, H, W); C_f=0 if disabled
      scalar:  [climate]         # → forcing_climate (D_c,); zeros if disabled

Normalisation is applied here, not in the Lightning module: training-split
statistics belong in the data layer.  Pass stats_path=None to bypass
normalisation for raw-value inspection.
"""

from __future__ import annotations

import json
import yaml
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

# Geophys channel layout in zarr: (T, 6, H, W).
# ch5 = ice_fjord_mask is excluded — unreliable across glaciers.
GEOPHYS_IDX = [0, 1, 2, 3, 4]  # bed elev, surf elev, thickness, velocity, strain rate
N_CLIMATE   = 2                 # mar_runoff, ocean_temp

# Individual geophys channel names → zarr index (subset of GEOPHYS_IDX).
# ch5 (ice_fjord_mask) is intentionally NOT exposed.
GEOPHYS_VARS = {
    "bed_elev":    0,
    "surf_elev":   1,
    "thickness":   2,
    "velocity":    3,
    "strain_rate": 4,
}

# Everything inputs.spatial may legally contain. Note: 'sdt_diff' is NOT here —
# it lives in inputs.scalar (FiLM) and is always added unconditionally as ch1.
SPATIAL_VOCAB = {"sdt", "sdt_trend", "bed_slope", "geophys", *GEOPHYS_VARS}


def resolve_geophys_idx(spatial_list) -> list[int]:
    """Resolve inputs.spatial into the sorted list of geophys zarr indices to load.

      - 'geophys' alias           -> [0, 1, 2, 3, 4]
      - individual names          -> sorted zarr indices of those names
      - no geophys tokens         -> []

    Validates the full spatial vocabulary and forbids mixing the alias with
    individual names. Returned indices are in zarr-index space, so the same
    list slices both store['geophys'][:, idx] and the length-5 stats array.
    """
    unknown = [t for t in spatial_list if t not in SPATIAL_VOCAB]
    if unknown:
        raise ValueError(
            f"Unrecognised inputs.spatial token(s): {unknown}. "
            f"Allowed: {sorted(SPATIAL_VOCAB)}"
        )
    individual = [t for t in spatial_list if t in GEOPHYS_VARS]
    if "geophys" in spatial_list:
        if individual:
            raise ValueError(
                f"inputs.spatial mixes the 'geophys' alias with individual geophys "
                f"name(s) {individual}. Use one form or the other, not both."
            )
        return [0, 1, 2, 3, 4]
    return sorted(GEOPHYS_VARS[t] for t in individual)


class CalvingFrontDatasetV2(Dataset):
    """
    Each sample contains three components:

      inputs              (K, C_in, H, W)   SDT + sdt_diff [+ geophys]
      climate             (K, D_c)          input climate scalars
      forcing_spatial     (C_f, H, W)       future geophys at t+horizon
      forcing_climate     (D_c,)            future climate at t+horizon
      target_displacement (1, H, W)         normalised SDT displacement target
    """

    def __init__(
        self,
        index_path: str | Path,
        config:     str | Path | dict,   # YAML path or pre-parsed dict
        stats_path: str | Path | None,
        augment:    bool = False,
    ):
        self.augment = augment

        with open(index_path) as f:
            self.samples: list[dict] = json.load(f)
        if not self.samples:
            raise RuntimeError(f"{index_path} is empty.")
        
        # read tolerance so _build_tau normalises correctly
        params_path = Path(index_path).parent / "sampling_params.json"
        if params_path.exists():
            with open(params_path) as f:
                self.tolerance_days = int(json.load(f)["tolerance_days"])
        else:
            self.tolerance_days = 60  # matches 03_build_index.py default


        # ── Parse variable config ─────────────────────────────────────────
        cfg   = yaml.safe_load(open(config)) if not isinstance(config, dict) else config
        in_sp = cfg.get("inputs",   {}).get("spatial", [])
        in_sc = cfg.get("inputs",   {}).get("scalar",  [])
        fo_sp = cfg.get("forcings", {}).get("spatial", [])
        fo_sc = cfg.get("forcings", {}).get("scalar",  [])

        if "sdt" not in in_sp:
            raise ValueError("'sdt' must be in inputs.spatial — it is the input and target basis.")

        self.geophys_idx      = resolve_geophys_idx(in_sp)   # zarr indices; [] = don't load
        self.use_sdt_trend    = "sdt_trend" in in_sp
        self.use_climate      = "climate" in in_sc
        self.use_sdt_diff_film = "sdt_diff" in in_sc
        self.use_bed_slope    = "bed_slope" in in_sp
        self.forcing_geophys  = "geophys" in fo_sp
        self.forcing_climate  = "climate" in fo_sc

        if self.forcing_geophys and self.geophys_idx != [0, 1, 2, 3, 4]:
            raise ValueError(
                "forcings.spatial 'geophys' requires the full geophys bundle in "
                "inputs.spatial; per-variable geophys inputs with geophys forcing "
                "is not supported."
            )

        # ── Normalisation stats ───────────────────────────────────────────
        # Loaded once at init; applied per sample in __getitem__.
        self.stats:      dict | None = None
        self._np:        dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.disp_scale = 1.0

        if stats_path is not None:
            p = Path(stats_path)
            if p.exists():
                with open(p) as f:
                    self.stats = json.load(f)
                self._np = self._parse_stats(self.stats)
                if "sdt" in self.stats:
                    sdt_std  = float(self.stats["sdt"]["std"])
                    disp_std = float(self.stats.get("displacement", {}).get("std", sdt_std))
                    self.disp_scale = sdt_std / max(disp_std, 1e-6)
            else:
                print(f"WARNING: normalisation_stats not found at {p} — skipping normalisation.")

        if self.use_bed_slope and self.stats is not None and "bed_slope" not in self.stats:
            raise KeyError(
                "inputs.spatial includes 'bed_slope' but normalisation_stats.json has no "
                "'bed_slope' entry — re-run 04_compute_stats.py. (Refusing to feed an "
                "unnormalised channel via _normalise's silent passthrough.)"
            )

        # Zarr stores are opened lazily and cached per worker process.
        self._stores: dict[str, zarr.Group] = {}
        # (H, W) per zarr_path, filled on first sample_sizes() call.
        self._hw: dict[str, tuple[int, int]] = {}

    # ── Normalisation ─────────────────────────────────────────────────────────
    # disp_scale is a public attribute so the DataModule can read it once
    # and pass it to the Lightning module, which needs it to convert
    # normalised displacement predictions to metres.

    def _parse_stats(self, stats: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Convert JSON stats to (mean, std) pairs; geophys sliced to self.geophys_idx."""
        out = {}
        for key in ("sdt", "geophys", "climate", "bed_slope"):
            if key not in stats:
                continue
            m = np.asarray(stats[key]["mean"], dtype=np.float32)
            s = np.maximum(np.asarray(stats[key]["std"],  dtype=np.float32), 1e-6)
            if key == "geophys":
                m, s = m[self.geophys_idx], s[self.geophys_idx]   # zarr-index slice
            out[key] = (m, s)
        return out

    def _normalise(self, arr: np.ndarray, key: str) -> np.ndarray:
        if key not in self._np:
            return arr
        mean, std = self._np[key]

        def _bc(v: np.ndarray) -> np.ndarray:
            """Reshape 1-D stats to broadcast over (K,C,H,W), (C,H,W), or (K,C)."""
            if v.ndim != 1:
                return v
            if arr.ndim == 4 and arr.shape[1] == v.shape[0]: return v.reshape(1, -1, 1, 1)
            if arr.ndim == 3 and arr.shape[0] == v.shape[0]: return v.reshape(-1, 1, 1)
            if arr.ndim == 2 and arr.shape[1] == v.shape[0]: return v.reshape(1, -1)
            return v

        return (arr - _bc(mean)) / _bc(std)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _get_store(self, path: str) -> zarr.Group:
        if path not in self._stores:
            try:
                self._stores[path] = zarr.open_group(path, mode="r")
            except Exception as e:
                # print(f"Error opening zarr group at {path}: {e}")
                self._stores[path] = zarr.open_group("../"+path, mode="r")
        return self._stores[path]

    @staticmethod
    def _build_tau(input_days: list[float], anchor_offsets: list[int],
                   t_ref: date, tolerance: int) -> np.ndarray:
        """
        Temporal feature matrix τ ∈ ℝ^(K, 4) — all values bounded, no further
        normalisation needed.
          col 0: day offset / max_anchor_offset        ∈ [-1, 0]
          col 1: fractional day-of-year (seasonality)  ∈ [0, 1]
          col 2: gap to next slot / max_spacing         ∈ [0, 1]
          col 3: anchor deviation / tolerance            ∈ [-1, 1]
        """
        K         = len(input_days)
        max_off   = float(max(abs(a) for a in anchor_offsets)) + tolerance
        max_space = float(max(abs(anchor_offsets[i] - anchor_offsets[i-1]) for i in range(1, K)))

        tau = np.zeros((K, 4), dtype=np.float32)
        for k in range(K):
            d   = float(input_days[k])
            doy = (t_ref + timedelta(days=int(d))).timetuple().tm_yday
            gap = float(input_days[k + 1]) - d if k < K - 1 else 0.0
            tau[k] = [d / max_off, doy / 365.0, gap / max(max_space + 2 * tolerance, 1),
                      (d - anchor_offsets[k]) / float(tolerance)]
        return tau

    # ── Dataset interface ─────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def sample_sizes(self) -> list[tuple[int, int]]:
        """(H, W) per sample, from zarr metadata. Constant per glacier."""
        for s in self.samples:
            p = s["zarr_path"]
            if p not in self._hw:
                h, w = self._get_store(p)["sdt"].shape[-2:]
                self._hw[p] = (int(h), int(w))
        return [self._hw[s["zarr_path"]] for s in self.samples]

    def __getitem__(self, idx: int) -> dict:
        s     = self.samples[idx]
        store = self._get_store(s["zarr_path"])
        ii    = s["input_zarr_indices"]   # list[int], len=K
        ti    = s["target_zarr_index"]    # int
        K     = len(ii)

        # t_ref is the virtual reference 365 days before the target
        # Support both old format (reference_date + target_day) and new (target_date + horizon_days)
        if "target_date" in s:
            target_date  = date.fromisoformat(s["target_date"])
            horizon_days = int(s["horizon_days"])
        else:
            t_ref        = date.fromisoformat(s["reference_date"])
            horizon_days = int(s["target_day"])
            target_date  = t_ref + timedelta(days=horizon_days)
        t_ref = target_date - timedelta(days=horizon_days)

        valid_clm         = np.array([bool(store["valid_clm"][i]) for i in ii], dtype=bool)
        valid_forcing_clm = bool(store["valid_clm"][ti])

        # ── Historical spatial inputs ──────────────────────────────────────
        sdt      = np.nan_to_num(store["sdt"][ii].astype(np.float32))   # (K, H, W)
        sdt      = self._normalise(sdt, "sdt")
        sdt_diff = np.diff(sdt, axis=0, prepend=sdt[[0]])               # (K, H, W)

        ch = [sdt[:, None], sdt_diff[:, None]]                          # (K, 1, H, W) each

        if self.use_sdt_trend:
            k_vals = np.arange(K, dtype=np.float32)
            k_dev  = k_vals - k_vals.mean()
            denom  = (k_dev ** 2).sum() + 1e-8
            trend  = (k_dev[:, None, None] * sdt).sum(axis=0) / denom   # (H, W)
            ch.append(np.broadcast_to(trend[None, None], (K, 1) + trend.shape).copy())

        if self.geophys_idx:
            geo = np.nan_to_num(store["geophys"][ii].astype(np.float32)[:, self.geophys_idx])
            ch.append(self._normalise(geo, "geophys"))                   # (K, len(idx), H, W)

        if self.use_bed_slope:
            bed_raw  = store["geophys"][ii].astype(np.float32)[:, 0]    # (K, H, W) raw, NaN out-of-domain
            valid    = np.isfinite(bed_raw)
            bed_fill = np.nan_to_num(bed_raw)
            gy, gx   = np.gradient(bed_fill, axis=(1, 2))
            mag      = np.sqrt(gx**2 + gy**2)
            mag      = np.where(valid, mag, 0.0)[:, None]               # (K, 1, H, W), zero invalid
            ch.append(self._normalise(mag, "bed_slope"))

        inputs = np.concatenate(ch, axis=1)                             # (K, C_in, H, W)
        H, W   = inputs.shape[-2], inputs.shape[-1]

        # ── Input climate ─────────────────────────────────────────────────
        clm = store["climate"][ii].astype(np.float32)                   # (K, D_c)
        if self.use_climate:
            clm = self._normalise(clm, "climate")
            clm[~valid_clm] = 0.0
        else:
            clm[:] = 0.0

        if self.use_sdt_diff_film:
            mean_sdt_diff = sdt_diff.mean(axis=(-2, -1))[:, None]      # (K, 1)
            clm = np.concatenate([clm, mean_sdt_diff], axis=1)         # (K, D_c+1)

        # ── Future forcings at target_zarr_index ─────────────────────────
        f_ch = []
        if self.forcing_geophys:
            geo_f = np.nan_to_num(store["geophys"][ti].astype(np.float32)[GEOPHYS_IDX])
            f_ch.append(self._normalise(geo_f, "geophys"))              # (5, H, W)

        # (C_f, H, W) — C_f=0 if no spatial forcings configured
        forcing_spatial = (
            np.concatenate(f_ch, axis=0) if f_ch
            else np.zeros((0, H, W), dtype=np.float32)
        )

        clm_f = store["climate"][ti].astype(np.float32)                 # (D_c,)
        if self.forcing_climate:
            clm_f = self._normalise(clm_f, "climate")
            if not valid_forcing_clm:
                clm_f[:] = 0.0
        else:
            clm_f[:] = 0.0

        # ── Target displacement ────────────────────────────────────────────
        # displacement needs to be scaled to be on a similar numerical scale to the inputs.
        # taking the difference of normalised SDTs gives a normalised displacement, that is tiny.
        # displacement_metres = disp_norm / _disp_scale * sdt_std
        tgt_sdt   = self._normalise(
            np.nan_to_num(store["sdt"][ti].astype(np.float32))[None], "sdt"
        )[0]                                                             # (H, W)
        last_sdt  = sdt[-1].copy()                                      # (H, W)
        disp_norm = (tgt_sdt - last_sdt) * self.disp_scale               # (H, W)

        # ── Calving front traces ───────────────────────────────────────────
        li = ii[-1]
        if "trace" in store and bool(store["valid_trace"][li]):
            input_trace = store["trace"][li].astype(np.float32)[None]
        else:
            # Fallback: threshold last SDT around the zero level (±45 m = 1.5 px)
            sdt_m = sdt[-1] * float(self.stats["sdt"]["std"]) + float(self.stats["sdt"]["mean"]) \
                    if self.stats else sdt[-1]
            input_trace = (np.abs(sdt_m) < 45.0).astype(np.float32)[None]

        if "trace" in store and bool(store["valid_trace"][ti]):
            target_trace = store["trace"][ti].astype(np.float32)[None]
        else:
            target_trace = np.zeros((1, H, W), dtype=np.float32)

        # ── Temporal features τ ────────────────────────────────────────────
        tau = self._build_tau(s["input_days"], s["anchor_offsets"], t_ref, self.tolerance_days)  # (K, 4)

        # ── Augmentation ──────────────────────────────────────────────────
        if self.augment:
            flip_w = torch.rand(1).item() > 0.5
            flip_h = torch.rand(1).item() > 0.5

            def _flip(a: np.ndarray) -> np.ndarray:
                if flip_w: a = np.flip(a, axis=-1)
                if flip_h: a = np.flip(a, axis=-2)
                return a.copy()

            inputs, input_trace, target_trace = _flip(inputs), _flip(input_trace), _flip(target_trace)
            disp_norm = _flip(disp_norm[None])[0]
            if forcing_spatial.shape[0] > 0:
                forcing_spatial = _flip(forcing_spatial)

            # Slot dropout: zero a random non-final slot (last slot anchors reconstruction)
            if K > 1 and torch.rand(1).item() > 0.5:
                inputs[np.random.randint(0, K - 1)] = 0.0

            # Noise on sdt (ch 0) only; sdt_diff (ch 1) recomputed to stay consistent
            inputs[:, 0] += (np.random.randn(*inputs[:, 0].shape) * 0.06).astype(np.float32)
            inputs[:, 1] = np.diff(inputs[:, 0], axis=0, prepend=inputs[[0], 0])

        return {
            # ── Inputs ────────────────────────────────────────────────────
            "inputs":             torch.from_numpy(inputs),             # (K, C_in, H, W)
            "climate":            torch.from_numpy(clm),                # (K, D_c)
            "tau":                torch.from_numpy(tau),                # (K, 4)
            "valid_clm":          torch.from_numpy(valid_clm),          # (K,) bool
            # ── Future forcings ───────────────────────────────────────────
            "forcing_spatial":    torch.from_numpy(forcing_spatial),    # (C_f, H, W)
            "forcing_climate":    torch.from_numpy(clm_f),              # (D_c,)
            "valid_forcing_clm":  valid_forcing_clm,                    # bool
            # ── Targets ───────────────────────────────────────────────────
            # last_input_sdt is not returned — recover it as batch["inputs"][:, -1, 0:1]
            # target_sdt is not returned  — recover it as disp / disp_scale + last_input_sdt
            "target_displacement": torch.from_numpy(disp_norm[None]),   # (1, H, W)
            "input_trace":         torch.from_numpy(input_trace),       # (1, H, W) — drives loss weighting
            "target_trace":        torch.from_numpy(target_trace),      # (1, H, W) — for evaluation
            # ── Metadata ──────────────────────────────────────────────────
            "glacier_id":            s["glacier_id"],
            "target_date":           target_date.isoformat(),
            "horizon_days":          horizon_days,
            "zarr_path":             s["zarr_path"],
            "input_zarr_indices":    ii,
            "target_zarr_index":     ti,
            "last_input_zarr_index": int(ii[-1]),
            "glacier_seasonality":         float(s["seasonal"]),   # 1.0=seasonal, 0.0=non-seasonal
        }


def collate_fn(batch: list[dict]) -> dict:
    """
    Collate variable-size samples by padding spatial dims to the batch maximum.
    forcing_spatial with C_f=0 channels is handled correctly by F.pad.
    """
    if not batch:
        raise ValueError("Cannot collate an empty batch.")

    max_h = max(s["inputs"].shape[-2] for s in batch)
    max_w = max(s["inputs"].shape[-1] for s in batch)

    def pad(t: torch.Tensor) -> torch.Tensor:
        return F.pad(t, (0, max_w - t.shape[-1], 0, max_h - t.shape[-2]))

    spatial_keys = [
        "inputs", "forcing_spatial", "target_displacement",
        "input_trace", "target_trace",
    ]
    result = {k: torch.stack([pad(s[k]) for s in batch]) for k in spatial_keys}

    # True where pixels belong to the actual glacier (not batch padding)
    masks = []
    for s in batch:
        m = torch.zeros(1, max_h, max_w, dtype=torch.bool)
        m[:, :s["inputs"].shape[-2], :s["inputs"].shape[-1]] = True
        masks.append(m)
    result["spatial_valid_mask"] = torch.stack(masks)

    for key in ["climate", "tau", "valid_clm", "forcing_climate"]:
        result[key] = torch.stack([s[key] for s in batch])
    result["valid_forcing_clm"] = torch.tensor(
        [s["valid_forcing_clm"] for s in batch], dtype=torch.bool
    )
    result["glacier_seasonality"] = torch.tensor(
        [s["glacier_seasonality"] for s in batch], dtype=torch.float32
    )  # (B,)
    result["horizon_days"] = torch.tensor([s["horizon_days"] for s in batch], dtype=torch.long)

    for key in ["glacier_id", "target_date", "zarr_path",
                "input_zarr_indices", "target_zarr_index", "last_input_zarr_index"]:
        result[key] = [s[key] for s in batch]

    result["original_hw"] = [(s["inputs"].shape[-2], s["inputs"].shape[-1]) for s in batch]
    return result
