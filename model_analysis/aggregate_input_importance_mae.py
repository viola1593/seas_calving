"""
Aggregate input (spatial-driver + climate) importance as ΔMAE.

For every test sample, perturb each input and measure the change in the front-band
field metric MAE:  ΔMAE = MAE_perturbed − MAE_full (signed, metres).
+ΔMAE = perturbation hurts the forecast, −ΔMAE = perturbation helps. Two methods
(--method):

  ablate (default) — set one input to its global mean (z=0), everything else fixed;
      measures that input's *contribution* to the forecast. Like IG, the effect
      scales with how far the input sits from the mean, so it conflates genuine
      importance with input-magnitude differences.

  sensitivity — nudge one input by +sigma (z units, default 0.25) and take the
      forward finite difference ΔMAE = MAE(x+σ) − MAE(x); measures the model's
      magnitude-independent *accuracy sensitivity* per +σ. The perturbation size is
      identical across seasons/currents, so this separates genuine sensitivity from
      input-magnitude differences.
      NOTE: for the level-like drivers bed/surf/bed_slope a uniform +σ shift is a
      numerical accuracy-gradient, not a physically realizable perturbation.

Spatial drivers (bed, surf, thick, vel, strain, bed_slope — whichever are in the
config) and the 4 climate variables (hist/forc × runoff/ocean) share one metres axis,
all recomputed here (no stale climate cache dependency). sdt / sdt_diff / tau are
perturbed only in --groups mode (see below); imagery is never perturbed.

--groups perturbs whole groups of inputs jointly instead of one at a time, giving
each group's combined contribution (individual ablations can all look small when
inputs are redundant):
    sdt        inputs ch0 + ch1 jointly — sdt_diff is np.diff(sdt) by construction,
               so a zeroed (constant) SDT field implies a zeroed sdt_diff; ablating
               ch0 alone would feed the model an impossible sample. ablate-only.
    tau        tau[:, :, 0..3] — all four temporal features. ablate-only: τ is
               bounded and unnormalised, so +σ is not a σ. NOTE z=0 is *not* the τ
               mean; it states all K slots were observed simultaneously, on Jan 1,
               with zero spacing, which is off-distribution.
    geometry   bed, surf, thick, bed_slope
    dynamics   vel, strain
    climate    climate[:, :, 0..1] + forcing_climate[:, 0..1] — historical and
               forcing runoff + ocean, perturbed jointly.
Outputs get a group_ prefix, so they never overwrite the per-input caches/figures.

--reference climatology (ablate only) sets the climate inputs to the glacier's own
calendar-month climatology instead of the global mean (z=0): every historical slot and
the forcing get the mean over all years of that glacier's store for the slot's month
(valid_clm dates only). The perturbation is then exactly the climate *anomaly*, so a
normal month is barely perturbed and ΔMAE measures what the model takes from the
anomalous part of the climate. Slots with invalid climate keep their (zeroed) value.
Spatial drivers / sdt / tau are ablated as usual. Outputs get a _climref suffix.

Perturbations never touch the metrics' frame of reference: the model is run on the
perturbed tensors, but _batch_to_records is called with the *unperturbed* batch, so
the reconstruction anchor (inputs[:, -1, 0]) and the active mask (temporal variance
of inputs[:, :, 0]) both stay pinned to the true SDT.

Outputs under <experiment>/explain/input_importance_mae/ (prefix mae_importance for
ablate, sens_importance for sensitivity):
    <prefix>_results.npz          signed ΔMAE per sample, {var__gid__season: [m]}
    <prefix>_nan_counts.npz       per-driver NaN-metric counts (NaN MAE_perturbed)
    <prefix>_per_sample.csv       one row per (sample, input): sample_idx, glacier_id,
                                  target_date, season, mae_full_m, variable, delta_mae_m;
                                  NaN kept (incl. NaN-MAE_full samples). Written on compute
                                  only — rerun with --recompute for existing caches
    <prefix>_by_glacier.csv       variable, glacier_id, season, n, [mean_abs_mae_m (sens)],
                                  mean_signed_mae_m, n_nanMAE
    <prefix>_glacier_mean.csv     same columns minus season: all seasons pooled per glacier
    <prefix>_glacier_spread.png       one panel per input: within-glacier sd(ΔMAE)
                                  vs that glacier's baseline test MAE_front
    <prefix>_glacier_spread_rel.png   same, sd as % of the glacier's own MAE_front
    <prefix>_overall.png          headline: one violin per input (|ΔMAE| or |ΔMAE|/σ)
    ablate: _overall_combined.png signed violins + % hurt / ⟨|ΔMAE|⟩ text columns
    ablate: _by_current.png       ESC vs WSC per input
    _by_season.png                DJF/MAM/JJA/SON per input
    sensitivity: _overall_signed.png  signed ΔMAE/σ companion to the magnitude figure

Usage (module, from project root):
  python -m model_analysis.aggregate_input_importance_mae --experiment experiments/<name>
  python -m model_analysis.aggregate_input_importance_mae --experiment experiments/<name> --n_samples 4
  python -m model_analysis.aggregate_input_importance_mae --experiment experiments/<name> --method sensitivity --all
  python -m model_analysis.aggregate_input_importance_mae --experiment experiments/<name> --groups --all
  python -m model_analysis.aggregate_input_importance_mae --experiment experiments/<name> --groups --all --reference climatology
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from functools import partial
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import torch
import zarr
from tqdm import tqdm

from data.datamodule import CalvingFrontDataModule
from evaluate_film import find_best_checkpoint, load_module_and_config
from model_analysis.utils import (
    _CURRENTS, _CURRENT_COLORS, _DRIVER_CHANNELS, _SEASON_ORDER, _load_current_lookup,
    _season, _spatial_channel_names, save_results, select_samples,
)

# (name, batch_key, index). climate is (B,K,2) → zero [:, :, idx] over all K slots;
# forcing_climate is (B,2) → zero [:, idx]. Index 0 = runoff, 1 = ocean.
_CLIMATE_TARGETS = [
    ("hist_runoff", "climate",         0),
    ("hist_ocean",  "climate",         1),
    ("forc_runoff", "forcing_climate", 0),
    ("forc_ocean",  "forcing_climate", 1),
]
_SEASON_COLORS = {"DJF": "#4c72b0", "MAM": "#55a868", "JJA": "#c44e52", "SON": "#8172b3"}

# --groups: perturb whole groups of inputs jointly instead of one at a time.
# (name, spatial channel names, [(batch_key, index), ...]). Channels absent from the
# config are skipped; a group with nothing present is dropped. tau is (B, K, 4), so
# the (B, K, D) scalar path sets tau[:, :, idx] to its per-slot mean (tau is
# unnormalised — see _tau_mean); listing all 4 ablates the whole tensor.
_GROUPS = [
    ("sdt",          ("sdt", "sdt_diff"),                   ()),
    ("tau",          (),  (("tau", 0), ("tau", 1), ("tau", 2), ("tau", 3))),
    ("geometry",     ("bed", "surf", "thick", "bed_slope"), ()),
    ("dynamics",     ("vel", "strain"),                     ()),
    ("climate",      (),  (("climate",         0), ("climate",         1),
                           ("forcing_climate", 0), ("forcing_climate", 1))),
]

# Groups excluded from --method sensitivity. sdt: a uniform +σ on ch0 leaves the true
# sdt_diff invariant, which the shared +σ loop cannot express. tau: bounded and
# unnormalised, so there is no σ to step in.
_ABLATE_ONLY_GROUPS = {"sdt", "tau"}

# Groups that exist for every config (geometry/dynamics need the geophys channels;
# sdt/sdt_diff/tau are always in the batch). Used to spot caches predating _GROUPS.
_CONFIG_INDEPENDENT_GROUPS = {"climate", "sdt", "tau"}


# ── caches (flat {var__gid__season} for later seed pooling) ───────────────────

def _empty() -> dict:
    return defaultdict(lambda: defaultdict(lambda: defaultdict(list)))


def _load_results(path: Path) -> dict:
    data = np.load(path)
    results = _empty()
    for key in data.files:
        var, gid, season = key.split("__")
        results[var][gid][season] = data[key].tolist()
    return results


def _save_counts(path: Path, counts: dict) -> None:
    arrays = {f"{v}__{g}__{s}": np.array([c], np.int32)
              for v, byg in counts.items()
              for g, bys in byg.items()
              for s, c in bys.items() if c}
    np.savez(path, **arrays)


def _load_counts(path: Path) -> dict:
    data = np.load(path)
    counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for key in data.files:
        var, gid, season = key.split("__")
        counts[var][gid][season] = int(data[key][0])
    return counts


def _pool(matrix) -> list[float]:
    """All per-sample ΔMAE for one variable, pooled across glaciers and seasons."""
    return [v for by_season in matrix.values() for vals in by_season.values() for v in vals]


# ── ablate-and-score inner loop ──────────────────────────────────────────────

class ClimatologyRef:
    """Per-sample climate ablation reference: glacier × calendar-month climatology in z
    units, for every input slot (climate (1, K, 2)) and the target (forcing_climate (1, 2)).
    Built from each zarr store's full climate series (valid_clm dates, one value per
    year-month, then mean over years); cached per store."""

    def __init__(self, climate_stats: dict):
        self.mu = np.asarray(climate_stats["mean"], np.float32)
        self.sd = np.asarray(climate_stats["std"], np.float32)
        self._cache: dict[str, tuple] = {}
        self.n_missing = 0                               # slot-months with no climatology

    def _store(self, path: str):
        if path not in self._cache:
            g = zarr.open_group(path, mode="r")
            x, valid = g["climate"][:], g["valid_clm"][:].astype(bool)
            dates = pd.to_datetime(g["dates"][:])
            d = pd.DataFrame(x, columns=range(x.shape[1])).assign(date=dates)[valid]
            d = d.drop_duplicates("date")
            ym = d.groupby([d.date.dt.year, d.date.dt.month]).mean(numeric_only=True)
            clim = ym.groupby(level=1).mean().reindex(range(1, 13)).values   # (12, D_c), NaN = no data
            self._cache[path] = ((clim - self.mu) / self.sd, dates.month.values - 1)
        return self._cache[path]

    def __call__(self, batch) -> dict[str, torch.Tensor]:
        assert len(batch["glacier_id"]) == 1, "ClimatologyRef expects batch_size=1"
        clim_z, month = self._store(batch["zarr_path"][0])
        ii, ti = batch["input_zarr_indices"][0], batch["target_zarr_index"][0]
        hist, forc = clim_z[month[ii]], clim_z[month[ti]]                    # (K, D_c), (D_c,)
        self.n_missing += int(np.isnan(hist).any(1).sum() + np.isnan(forc).any())
        hist = torch.as_tensor(np.nan_to_num(hist), dtype=batch["climate"].dtype)  # missing → global mean
        forc = torch.as_tensor(np.nan_to_num(forc), dtype=batch["forcing_climate"].dtype)
        dev = batch["climate"].device
        valid = batch["valid_clm"][0].to(dev)[:, None]                       # (K, 1)
        hist = torch.where(valid, hist.to(dev), batch["climate"][0])         # invalid slots keep 0
        return {"climate": hist[None], "forcing_climate": forc.to(dev)[None]}


@torch.no_grad()
def score_sample(module, batch, driver_idx: dict, sigma: float | None = None,
                 clim_ref: dict | None = None) -> tuple[float, dict]:
    """Returns (mae_full, {input_name: signed ΔMAE_or_nan}).
    driver_idx: {spatial_driver_name -> channel index in batch['inputs']}.
    sigma=None → ablate to global mean (z=0); sigma set → +σ sensitivity.
    clim_ref (ablate only): {batch_key: tensor} climate ablation reference replacing z=0
    (see ClimatologyRef); spatial drivers are unaffected.
    NOTE: for bed/surf/bed_slope a uniform +σ shift is a numerical accuracy-gradient,
    not a physically realizable perturbation."""
    model = module.model
    valid = batch["spatial_valid_mask"][0, 0].bool()          # (H, W)

    pred_full = model(batch)                                   # (1, 1, H, W)
    mae_full = module._batch_to_records(pred_full, batch)[0]["mae_front"]
    if not np.isfinite(mae_full):
        return mae_full, {}                                   # caller drops the whole sample

    deltas: dict[str, float] = {}

    # ── spatial drivers: ablate-to-mean (z=0) or +σ shift over valid pixels, all K slots ──
    for name, idx in driver_idx.items():
        t = batch["inputs"].clone()                           # (1, K, C, H, W)
        chan = t[:, :, idx]                                   # (1, K, H, W)
        new = chan + sigma if sigma is not None else torch.zeros_like(chan)
        t[:, :, idx] = torch.where(valid, new, chan)
        pred = model({**batch, "inputs": t})
        mae_abl = module._batch_to_records(pred, batch)[0]["mae_front"]
        deltas[name] = mae_abl - mae_full                     # NaN propagates if eval zone empty

    # ── climate: ablate to 0 or +σ shift (climate over all K slots, forcing_climate flat) ──
    for name, key, idx in _CLIMATE_TARGETS:
        t = batch[key].clone()
        sl = (slice(None), slice(None), idx) if t.dim() == 3 else (slice(None), idx)
        t[sl] = (t[sl] + sigma if sigma is not None else
                 clim_ref[key][sl] if clim_ref is not None else 0.0)
        pred = model({**batch, key: t})
        mae_abl = module._batch_to_records(pred, batch)[0]["mae_front"]
        deltas[name] = mae_abl - mae_full

    return mae_full, deltas


@torch.no_grad()
def score_sample_groups(module, batch, driver_idx: dict, sigma: float | None = None,
                        clim_ref: dict | None = None,
                        tau_mean: torch.Tensor | None = None) -> tuple[float, dict]:
    """Same as score_sample but perturbs every member of a _GROUPS entry jointly,
    so the returned ΔMAE is the group's combined contribution.
    driver_idx here also carries sdt/sdt_diff (see _group_channel_idx).
    tau_mean (1, K, 4): ablation reference for tau, which is unnormalised (see
    _tau_mean). Required whenever the tau group is scored.
    clim_ref: as in score_sample — replaces z=0 for the climate group."""
    model = module.model
    valid = batch["spatial_valid_mask"][0, 0].bool()          # (H, W)

    pred_full = model(batch)
    mae_full = module._batch_to_records(pred_full, batch)[0]["mae_front"]
    if not np.isfinite(mae_full):
        return mae_full, {}

    deltas: dict[str, float] = {}
    for name, drivers, climate in _GROUPS:
        if sigma is not None and name in _ABLATE_ONLY_GROUPS:
            continue
        idxs = [driver_idx[d] for d in drivers if d in driver_idx]
        if not idxs and not climate:
            continue
        patch: dict[str, torch.Tensor] = {}
        if idxs:
            t = batch["inputs"].clone()                       # (1, K, C, H, W)
            for idx in idxs:
                chan = t[:, :, idx]                           # (1, K, H, W)
                new = chan + sigma if sigma is not None else torch.zeros_like(chan)
                t[:, :, idx] = torch.where(valid, new, chan)
            patch["inputs"] = t
        for key, idx in climate:
            t = patch.get(key) if key in patch else batch[key].clone()
            sl = (slice(None), slice(None), idx) if t.dim() == 3 else (slice(None), idx)
            t[sl] = (t[sl] + sigma if sigma is not None else
                     tau_mean[sl] if key == "tau" else
                     clim_ref[key][sl] if clim_ref is not None else 0.0)
            patch[key] = t
        pred = model({**batch, **patch})
        mae_abl = module._batch_to_records(pred, batch)[0]["mae_front"]
        deltas[name] = mae_abl - mae_full

    return mae_full, deltas


# ── group helpers ────────────────────────────────────────────────────────────

def _group_channel_idx(driver_idx: dict, channel_names: list[str]) -> dict:
    """driver_idx plus sdt/sdt_diff, which are always inputs ch0/ch1 but are not
    _DRIVER_CHANNELS (they must stay out of the per-input path)."""
    return {**driver_idx,
            "sdt":      channel_names.index("sdt"),
            "sdt_diff": channel_names.index("sdt_diff")}


def _tau_mean(samples) -> torch.Tensor:
    """Per-(slot, column) mean tau over the scored samples — the ablation reference.
    tau is bounded but unnormalised, so 0 is not its mean: zeroing cols 0/2 asserts
    that all K slots are coincident and col 1 forces every sample to 1 January,
    which is off-manifold rather than neutral. col 3 is ~0-mean either way."""
    return torch.stack([b["tau"] for b in samples]).mean(0)          # (1, K, 4)


def _group_order(group_idx: dict, sens: bool) -> list[str]:
    """Groups present in this config, ablate-only ones dropped under sensitivity."""
    return [name for name, drivers, climate in _GROUPS
            if (climate or any(d in group_idx for d in drivers))
            and not (sens and name in _ABLATE_ONLY_GROUPS)]


# ── sample sweep ─────────────────────────────────────────────────────────────

def _compute(args, results_cache: Path, counts_cache: Path, samples_csv: Path):
    device = torch.device(
        args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    )
    print(f"Device  : {device}")

    ckpt = find_best_checkpoint(args.experiment)
    module, cfg = load_module_and_config(ckpt, args.experiment, device)
    module.model.eval()                                   # module is already .eval() from loader

    dm = CalvingFrontDataModule(
        **{**cfg["data"], "batch_size": 1, "num_workers": 4, "augment_train": False}
    )
    dm.setup("test")
    n = None if (args.all or args.n_samples is None) else args.n_samples
    samples = select_samples(dm.test_dataloader(shuffle=False), n)
    print(f"Samples : {len(samples)}")

    channel_names = _spatial_channel_names(cfg)
    driver_idx = {name: channel_names.index(name)
                  for name in _DRIVER_CHANNELS if name in channel_names}
    if args.groups:
        driver_idx = _group_channel_idx(driver_idx, channel_names)
        var_order  = _group_order(driver_idx, args.method == "sensitivity")
        score = partial(score_sample_groups, tau_mean=_tau_mean(samples).to(device))
    else:
        var_order = list(driver_idx) + [t[0] for t in _CLIMATE_TARGETS]
        score = score_sample
    sigma = args.sigma if args.method == "sensitivity" else None
    clim_ref = (ClimatologyRef(dm._test_ds.stats["climate"])
                if args.reference == "climatology" else None)

    results    = _empty()
    nan_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    rows: list[list] = []                                # per-sample long table, NaN kept
    n_dropped_full = 0

    for i, batch in enumerate(tqdm(samples, desc="Computing ΔMAE input importance")):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                 for k, v in batch.items()}
        mae_full, deltas = score(module, batch, driver_idx, sigma,
                                 clim_ref=clim_ref(batch) if clim_ref else None)
        gid    = batch["glacier_id"][0]
        date   = batch["target_date"][0]
        season = _season(date)
        rows += [[i, gid, date, season, float(mae_full), name,
                  float(deltas.get(name, float("nan")))] for name in var_order]
        if not deltas:                                   # MAE_full NaN → global-drop
            n_dropped_full += 1
            continue
        for name in var_order:
            d = deltas[name]
            if np.isfinite(d):
                results[name][gid][season].append(float(d))
            else:
                nan_counts[name][gid][season] += 1

    print(f"Dropped (MAE_full NaN): {n_dropped_full}/{len(samples)}")
    if clim_ref:
        print(f"Climatology reference: {clim_ref.n_missing} slot-months without data "
              f"→ global mean used")
    results_cache.parent.mkdir(parents=True, exist_ok=True)
    save_results(results_cache, results)
    _save_counts(counts_cache, nan_counts)
    with open(samples_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_idx", "glacier_id", "target_date", "season",
                    "mae_full_m", "variable", "delta_mae_m"])
        w.writerows(rows)
    print(f"Saved: {samples_csv}")
    return results, nan_counts, var_order


# ── CSV ──────────────────────────────────────────────────────────────────────

def save_csv(out_path: Path, results: dict, nan_counts: dict, var_order: list[str],
             sens: bool = False, by_season: bool = True) -> None:
    """by_season=False pools all seasons into one row per (variable, glacier)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        cols = ["variable", "glacier_id"] + (["season"] if by_season else []) + ["n"]
        cols += ["mean_abs_mae_m", "mean_signed_mae_m"] if sens else ["mean_signed_mae_m"]
        w.writerow(cols + ["n_nanMAE"])
        season_sets = [[s] for s in _SEASON_ORDER] if by_season else [_SEASON_ORDER]
        for var in var_order:
            gids = sorted(set(results[var]) | set(nan_counts.get(var, {})))
            for gid in gids:
                for ss in season_sets:
                    vals = [v for s in ss for v in results[var][gid][s]]
                    n_nan = sum(nan_counts.get(var, {}).get(gid, {}).get(s, 0) for s in ss)
                    if vals or n_nan:
                        signed = float(np.mean(vals)) if vals else float("nan")
                        row = [var, gid] + ([ss[0]] if by_season else []) + [len(vals)]
                        if sens:
                            row.append(float(np.mean(np.abs(vals))) if vals else float("nan"))
                        w.writerow(row + [signed, n_nan])
    print(f"Saved: {out_path}")


# ── figures (signed ΔMAE, metres axis) ───────────────────────────────────────
FS=14
# ── display: labels + climate/driver colouring for the headline figures ──────
# Keys not listed fall through to the raw variable name, so --groups keeps working.
_VAR_LABELS = {
    "bed":         "Bed elevation",
    "surf":        "Surface elevation",
    "thick":       "Ice thickness",
    "vel":         "Ice flow speed",
    "strain":      "Strain rate",
    "bed_slope":   "Bed slope",
    "hist_runoff": "Runoff (hist.)",
    "hist_ocean":  "Ocean temp. (hist.)",
    "forc_runoff": "Runoff (forecast)",
    "forc_ocean":  "Ocean temp. (forecast)",
    # --groups
    "sdt":         "Calving fronts",
    "tau":         "Time encoding",
    "geometry":    "Topography",
    "dynamics":    "Ice-dynamics",
    "climate":     "Climate",
}
_CLIMATE_VARS = {n for n, _, _ in _CLIMATE_TARGETS} | {"climate"}
# violin body (light) + scatter dots (same hue, darker so they stay readable on it)
_BODY_CLIMATE, _DOT_CLIMATE = "#e8a0a0", "#b5484a"
_BODY_DRIVER,  _DOT_DRIVER  = "#a8c4e0", "#3d6fa5"


def _label(var: str) -> str:
    return _VAR_LABELS.get(var, var)


def _var_colors(var: str) -> tuple[str, str]:
    """(violin body, scatter dot) for one variable."""
    return ((_BODY_CLIMATE, _DOT_CLIMATE) if var in _CLIMATE_VARS
            else (_BODY_DRIVER, _DOT_DRIVER))


def make_overall_figure(out_path: Path, results: dict, var_order: list[str],
                        noise_floor_m: float, signed: bool,
                        groups: bool = False) -> None:
    """Horizontal violins, one row per input: climate light red, spatial drivers
    light blue. groups=True (--groups): wider fixed axis (±20 m signed, 0–50 m
    absolute), since the sdt group sits well above the per-input ±7.5 m scale."""
    present = [v for v in var_order if _pool(results[v])]
    cols = [_pool(results[v]) for v in present]
    if not signed:
        cols = [np.abs(c) for c in cols]
    body_c, dot_c = zip(*(_var_colors(v) for v in present))

    fig, ax = plt.subplots(figsize=(7.5, max(4, len(present) * 0.55)))
    # ax.axvspan(-noise_floor_m, noise_floor_m, color="gray", alpha=0.15,
    #            label=f"±{noise_floor_m:g} m seed noise floor")
    ax.axvline(0.0, color="gray", lw=0.8, ls="--")

    parts = ax.violinplot(cols, vert=False, showmeans=True, showextrema=False)
    for b, c in zip(parts["bodies"], body_c):
        b.set_facecolor(c)
        b.set_edgecolor("none")
        b.set_alpha(0.6)
    for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")

    for i, (col, c) in enumerate(zip(cols, dot_c), start=1):
        jitter = (np.random.rand(len(col)) - 0.5) * 0.3
        ax.scatter(col, i + jitter, s=2, color=c, alpha=0.3, linewidth=0, zorder=1)
        ax.text(0.99, i, f"{np.mean(col):+.2f}", transform=ax.get_yaxis_transform(),
                ha="right", va="center", fontsize=FS - 3, zorder=4,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.5))

    ax.set_yticks(range(1, len(present) + 1))
    ax.set_yticklabels([_label(v) for v in present], fontsize=FS)
    ax.invert_yaxis()                                  # first variable at the top
    ax.tick_params(axis="x", labelsize=FS - 2)
    if groups:
        ax.set_xlim(-20 if signed else 0, 20 if signed else 50)
    else:
        ax.set_xlim(-max(noise_floor_m, 7.5) if signed else 0, max(noise_floor_m, 7.5))
    ax.set_xlabel("ΔMAE = MAE_ablated − MAE_full  (m)" if signed
                  else "|ΔMAE| = |MAE_ablated − MAE_full|  (m)", fontsize=FS)
    # ax.set_title("Feature influence (ablate-to-mean, ΔMAE) with means")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_combined_figure(out_path: Path, results: dict, var_order: list[str]) -> None:
    """Signed ΔMAE violins with the magnitude story as two text columns to the right.

    Direction and spread come from the violin (unclipped linear axis, black diamond =
    signed mean); magnitude enters as numbers: % hurt = 100·mean(Δ>0) (50 % = no
    preferred direction) and ⟨|ΔMAE|⟩ = mean of per-sample |Δ|, which stays large for
    inputs whose effect is big but sign-flipping. Rows are two blocks — spatial
    drivers, then climate — each sorted by signed mean, descending.
    """
    present = [v for v in var_order if _pool(results[v])]
    drivers = sorted((v for v in present if v not in _CLIMATE_VARS),
                     key=lambda v: np.mean(_pool(results[v])), reverse=True)
    climate = sorted((v for v in present if v in _CLIMATE_VARS),
                     key=lambda v: np.mean(_pool(results[v])), reverse=True)
    rows = drivers + climate
    # half-row gap between the two blocks, so positions are not 1..N
    # positions = ([1.0 + i for i in range(len(drivers))]
    #              + [len(drivers) + 1.5 + i for i in range(len(climate))])
    positions = list(range(1, len(rows) + 1))
    cols = [np.asarray(_pool(results[v])) for v in rows]

    fig, ax = plt.subplots(figsize=(6, 5.8))
    ax.axvline(0.0, color="lightgray", lw=0.8, ls="--", zorder=0)

    parts = ax.violinplot(cols, positions=positions, vert=False,
                          showmeans=True, showextrema=False, 
                        #   quantiles=[[0.1, 0.9] for _ in cols]
                          )
    for b, v in zip(parts["bodies"], rows):
        b.set_facecolor(_var_colors(v)[0])
        b.set_edgecolor("none")
        b.set_alpha(0.9)
    parts["cmeans"].set_color("black")

    # for col, pos in zip(cols, positions):
    #     ax.scatter(col.mean(), pos, marker="D", s=18, color="black", zorder=4)

    # text columns outside the axes: the violin tails now reach the axis edge
    # ax.text(1.06, 1.01, "% hurt", transform=ax.transAxes, ha="center", va="bottom",
    #         fontsize=FS - 2, clip_on=False)
    
    ax.text(1.06, 1.01, "mean", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS - 2, clip_on=False)
    ax.text(1.22, 1.01, "std", transform=ax.transAxes, ha="center", va="bottom",
            fontsize=FS - 2, clip_on=False)
    for col, pos in zip(cols, positions):
        # ax.text(1.06, pos, f"{100 * np.mean(col > 0):.1f}",
        #         transform=ax.get_yaxis_transform(), ha="center", va="center",
        #         fontsize=FS - 2, clip_on=False)
        ax.text(1.06, pos, f"{np.mean(col):.1f}",
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=FS - 2, clip_on=False)
        ax.text(1.22, pos, f"{np.std(col):.1f}",
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=FS - 2, clip_on=False)
    ax.text(0.0, 1.01, f"n = {max(len(c) for c in cols)}", transform=ax.transAxes,
            ha="left", va="bottom", fontsize=FS - 2, color="gray")
    ax.set_xlim(-max(40, 0.5* np.max([np.abs(c).max() for c in cols])), max(40,0.5*np.max([np.abs(c).max() for c in cols])))
    ax.set_yticks(positions)
    ax.set_yticklabels([_label(v) for v in rows], fontsize=FS)
    ax.invert_yaxis()                                  # first row at the top
    ax.tick_params(axis="x", labelsize=FS - 2)
    ax.set_xlabel("ΔMAE", fontsize=FS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_sens_overall_figure(out_path: Path, results: dict, var_order: list[str],
                             sigma: float, *,
                             signed: bool, noise_floor_m: float | None) -> None:
    """+σ accuracy sensitivity headline. signed=False → |ΔMAE|/σ ([0,floor] band);
    signed=True → ΔMAE/σ (±floor band). Autoscaled y (no ablate-scale ±20 clamp)."""
    present = [v for v in var_order if _pool(results[v])]
    cols = [np.asarray(_pool(results[v])) / sigma for v in present]
    if not signed:
        cols = [np.abs(c) for c in cols]

    fig, ax = plt.subplots(figsize=(max(7, len(present) * 0.9), 5))
    if noise_floor_m is not None:
        ax.axhspan(-noise_floor_m if signed else 0.0, noise_floor_m,
                   color="gray", alpha=0.15,
                   label=f"{'±' if signed else ''}{noise_floor_m:g} m/σ seed noise floor")
    if signed:
        ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    parts = ax.violinplot(cols, showmeans=False, showmedians=True, showextrema=False)
    for b in parts["bodies"]: 
        b.set_facecolor('white')
        # b.set_edgecolor("black")
        b.set_alpha(0.0)
    for key in ("cmeans", "cmedians", "cmins", "cmaxes", "cbars"):
        if key in parts:
            parts[key].set_color("black")

    for i, col in enumerate(cols, start=1):
        jitter = (np.random.rand(len(col)) - 0.5) * 0.3
        ax.scatter(i + jitter, col, s=2, color="tab:blue", alpha=0.3,
                   linewidth=0, zorder=1)
        ax.text(i, np.median(col)+1, f"{np.median(col):+.2f}", ha="center", va="bottom",
                fontsize=11,)
        # ax.annotate(f"{np.median(col):+.2f}", xy=(i, np.median(col)), xytext=(0, 4),
        #             textcoords="offset points", ha="center", va="bottom", fontsize=11)
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present, rotation=45, ha="right")
    metric = "ΔMAE/σ" if signed else "|ΔMAE|/σ"
    ax.set_ylim(-10 if signed else 0.0, 20 if signed else max(0, 40))
    ax.set_ylabel(f"{metric}")
    ax.set_title(f"Input accuracy sensitivity (+{sigma:g}σ) with medians")
    if noise_floor_m is not None:
        ax.legend(loc="best", fontsize=10)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def make_season_figure(out_path: Path, results: dict, var_order: list[str], *,
                       transform=lambda v: v,
                       ylabel="ΔMAE = MAE_ablated − MAE_full  (m)",
                       title="Input importance by season (ΔMAE) — signed  (+ = driver helps)") -> None:
    """Grouped violins: one group per input, one violin per season."""
    n_seasons = len(_SEASON_ORDER)
    group_w = 0.8
    edges = np.linspace(-group_w / 2, group_w / 2, n_seasons + 1)
    offsets = (edges[:-1] + edges[1:]) / 2  # slot centers within each group
    vw = group_w / n_seasons * 0.9

    present = [v for v in var_order if _pool(results[v])]
    fig, ax = plt.subplots(figsize=(max(6, len(present) * 1.2), 5))
    for si, s in enumerate(_SEASON_ORDER):
        for j, var in enumerate(present):
            vals = [transform(v) for gid in results[var] for v in results[var][gid][s]]
            if not vals:
                continue
            parts = ax.violinplot(vals, positions=[j + 1 + offsets[si]],
                                  widths=vw, showmeans=True)
            for b in parts["bodies"]:
                b.set_facecolor(_SEASON_COLORS[s])
                # b.set_edgecolor("black")
                b.set_alpha(0.7)
            for key in ("cmeans", "cmins", "cmaxes", "cbars"):
                if key in parts:
                    parts[key].set_color("black")
    ax.axhline(0.0, color="gray", lw=0.8, ls="--")
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(present, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    handles = [Patch(facecolor=_SEASON_COLORS[s], alpha=0.7, edgecolor="black")
               for s in _SEASON_ORDER]
    ax.legend(handles, _SEASON_ORDER, title="season", loc="best", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def _violin_mae(ax, matrix, groups, title):
    """Signed-only ESC/WSC × season violins for one input (no fixed ylim)."""
    width = 0.35
    for ci, cur in enumerate(_CURRENTS):
        offset = (ci - 0.5) * width
        color = _CURRENT_COLORS[cur]
        for si, s in enumerate(_SEASON_ORDER):
            vals = [v for gid in groups[cur] for v in matrix[gid][s]]
            if not vals:
                continue
            pos = si + offset
            if len(vals) > 1:
                parts = ax.violinplot(vals, positions=[pos], widths=width * 0.9,
                                      showmeans=True, showextrema=False)
                for body in parts["bodies"]:
                    body.set_facecolor(color)
                    body.set_alpha(0.4)
                parts["cmeans"].set_color(color)
            jitter = (np.random.rand(len(vals)) - 0.5) * width * 0.5
            ax.scatter(pos + jitter, vals, s=4, color=color, alpha=0.3, linewidth=0, zorder=3)
    ax.axhline(0.0, color="gray", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(_SEASON_ORDER)))
    ax.set_xticklabels(_SEASON_ORDER)
    ax.set_ylabel("ΔMAE (m)")
    ax.set_title(title, fontsize=9)


def make_current_figure(out_path, results, var_order, lookup, ncols=3):
    groups = {c: [] for c in _CURRENTS}
    all_gids = {gid for v in var_order for gid in results[v]}
    for gid in all_gids:
        groups[lookup[gid]].append(gid)
    present = [v for v in var_order if _pool(results[v])]
    nrows = ceil(len(present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3.2), squeeze=False)
    for idx, var in enumerate(present):
        _violin_mae(axes[idx // ncols][idx % ncols], results[var], groups, var)
    for idx in range(len(present), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")
    handles = [Patch(facecolor=_CURRENT_COLORS[c], alpha=0.4, edgecolor="k",
                     label=f"{c} ({len(groups[c])} glaciers)") for c in _CURRENTS]
    fig.legend(handles=handles, loc="upper right", fontsize=9)
    fig.suptitle("Input importance by current (ΔMAE) — ESC vs WSC  (+ = driver helps)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── glacier-wise spread (ΔMAE variability vs baseline error) ─────────────────
# Where evaluate_film writes its per-glacier test metrics, newest convention first.
_BASELINE_SUBDIRS = ("eval_paper/best", "eval/best")


def _load_baseline_mae(experiment: Path, override: Path | None) -> dict[str, float]:
    """{glacier_id -> mean test MAE_front (m)} from the standard evaluation CSV.

    Same checkpoint and test set as this sweep, so the two are comparable, with one
    caveat: samples the sweep dropped for a NaN MAE_full still count here, so the
    baseline is over a marginally larger sample set than the ΔMAE it normalises.
    Returns {} if no CSV is found — the caller then skips the glacier figures.
    """
    paths = [override] if override else [experiment / d / "test_per_glacier.csv"
                                         for d in _BASELINE_SUBDIRS]
    for p in paths:
        if p.exists():
            print(f"Baseline MAE_front from: {p}")
            with open(p) as f:
                return {r["glacier_id"]: float(r["mae_front_mean"])
                        for r in csv.DictReader(f)}
    print(f"No test_per_glacier.csv under {experiment} → skipping glacier figures")
    return {}


def make_glacier_spread_figure(out_path: Path, results: dict, var_order: list[str],
                               baseline: dict, lookup: dict, *, relative: bool,
                               min_n: int = 20, scale: float = 1.0, unit: str = "m",
                               ncols: int = 4) -> None:
    """One panel per input: within-glacier sd(ΔMAE) vs that glacier's baseline
    MAE_front, one point per glacier, coloured by current.

    The per-glacier *mean* ΔMAE is small next to its within-glacier spread, so the
    spread is what actually separates glaciers. Two regimes show up here:
    absolute sd tracks baseline error (high-MAE glaciers swing most in metres),
    while relative=True (sd as % of the glacier's own MAE_front) inverts it for the
    spatial drivers — the well-predicted glaciers are the ones leaning on them.

    Glaciers with fewer than min_n samples are dropped: sd has ~sd/sqrt(2(n-1))
    standard error, so short records would dominate the range with sampling noise.
    scale divides the raw ΔMAE (pass sigma for --method sensitivity).
    y is shared across panels so inputs stay comparable to each other.
    """
    present = [v for v in var_order if _pool(results[v])]
    nrows = ceil(len(present) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 3.0),
                             squeeze=False, sharex=True, sharey=True)
    n_kept = 0
    for idx, var in enumerate(present):
        ax = axes[idx // ncols][idx % ncols]
        gids = [g for g in sorted(results[var])
                if g in baseline and sum(len(v) for v in results[var][g].values()) >= min_n]
        n_kept = len(gids)
        x, y = [], []
        for g in gids:
            vals = np.array([v for s in results[var][g].values() for v in s]) / scale
            x.append(baseline[g])
            sd = float(np.std(vals, ddof=1))
            y.append(100.0 * sd / baseline[g] if relative else sd)
        x, y = np.asarray(x), np.asarray(y)
        for g, xi, yi in zip(gids, x, y):
            ax.scatter(xi, yi, s=26, color=_CURRENT_COLORS[lookup[g]], alpha=0.8,
                       edgecolor="none", zorder=3)
        title = _label(var)
        if len(x) > 2:
            r = np.corrcoef(x, y)[0, 1]
            m, b = np.polyfit(x, y, 1)
            xs = np.array([x.min(), x.max()])
            ax.plot(xs, m * xs + b, color="gray", lw=1.0, ls="--", zorder=2)
            title += f"   r = {r:+.2f}"          # in the title, so it never collides
        # name the glacier with the largest spread — it carries the panel's story
        if len(y):
            top = int(np.argmax(y))
            ax.annotate(gids[top].split(".")[-1], xy=(x[top], y[top]), xytext=(4, -9),
                        textcoords="offset points", fontsize=FS - 6, color="gray")
        ax.set_title(title, fontsize=FS - 4)
        ax.tick_params(labelsize=FS - 5)
    for idx in range(len(present), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    # one shared label per axis; x goes on the lowest *visible* panel of each column
    fig.supylabel(("sd(ΔMAE) / MAE_front  (% of glacier baseline)" if relative
                   else f"sd(ΔMAE)  ({unit})"), fontsize=FS - 3)
    for c_i in range(ncols):
        rows = [r_i for r_i in range(nrows) if r_i * ncols + c_i < len(present)]
        if rows:
            axes[rows[-1]][c_i].set_xlabel("baseline MAE_front (m)", fontsize=FS - 3)
            # sharex hides ticks above the last row; restore them where the grid ends
            axes[rows[-1]][c_i].tick_params(labelbottom=True, labelsize=FS - 5)
    handles = [Patch(facecolor=_CURRENT_COLORS[c], alpha=0.8, edgecolor="none", label=c)
               for c in _CURRENTS]
    fig.legend(handles=handles, loc="upper right", fontsize=FS - 4, ncol=len(_CURRENTS))
    fig.suptitle(f"Per-glacier ΔMAE spread vs baseline error "
                 f"({n_kept} glaciers, n ≥ {min_n} samples)", fontsize=FS - 1)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


# ── CLI + main ───────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Input importance as ΔMAE (ablate-to-mean or +σ sensitivity).")
    p.add_argument("--experiment", type=Path, required=True)
    p.add_argument("--method", choices=["ablate", "sensitivity"], default="ablate",
                   help="ablate: set each input to global mean (z=0), effect scales with "
                        "distance-from-mean. sensitivity: +σ nudge, magnitude-independent "
                        "accuracy sensitivity ΔMAE=MAE(x+σ)−MAE(x).")
    p.add_argument("--groups", action="store_true",
                   help="Perturb whole input groups jointly (geometry, dynamics, "
                        "climate, and — ablate only — sdt, tau) "
                        "instead of one input at a time.")
    p.add_argument("--reference", choices=["mean", "climatology"], default="mean",
                   help="Climate ablation reference (ablate only): global mean (z=0) or "
                        "the glacier's calendar-month climatology (ΔMAE of the anomaly). "
                        "climatology → _climref output suffix.")
    p.add_argument("--sigma", type=float, default=0.25,
                   help="Perturbation size (z units) for --method sensitivity.")
    p.add_argument("--n_samples", type=int, default=None,
                   help="Number of test samples. Default: full test set.")
    p.add_argument("--all", action="store_true", help="Sweep full test set.")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--recompute", action="store_true",
                   help="Force recomputation even if the cached NPZs exist.")
    p.add_argument("--baseline_csv", type=Path, default=None,
                   help="test_per_glacier.csv supplying each glacier's baseline "
                        f"MAE_front for the glacier-spread figures. Default: the "
                        f"first of {'/'.join(_BASELINE_SUBDIRS)} that exists.")
    p.add_argument("--min_glacier_n", type=int, default=20,
                   help="Drop glaciers with fewer samples from the spread figures, "
                        "where their sd would be mostly sampling noise.")
    p.add_argument("--noise_floor_m", type=float, default=None,
                   help="Half-width of the ±reference band on the overall figure. "
                        "Default: 0.4 for ablate (seasonal seed-to-seed MAE_front std), "
                        "no band for sensitivity (per-σ floor uncalibrated).")
    args = p.parse_args()
    if args.reference == "climatology" and args.method != "ablate":
        p.error("--reference climatology only applies to --method ablate")
    if args.noise_floor_m is None and args.method == "ablate":   # restore ablate's calibrated default
        args.noise_floor_m = 0.4
    return args


def main(args: argparse.Namespace) -> None:
    out_dir = (args.output_dir or (args.experiment / "explain")) / "input_importance_mae"
    sens = args.method == "sensitivity"
    prefix = "sens_importance" if sens else "mae_importance"
    if args.groups:
        prefix = f"group_{prefix}"
    if args.reference == "climatology":
        prefix = f"{prefix}_climref"
    results_cache = out_dir / f"{prefix}_results.npz"
    counts_cache  = out_dir / f"{prefix}_nan_counts.npz"
    samples_csv   = out_dir / f"{prefix}_per_sample.csv"

    results = None
    if not args.recompute and results_cache.exists() and counts_cache.exists():
        results    = _load_results(results_cache)
        nan_counts = _load_counts(counts_cache)
        known = ([g[0] for g in _GROUPS] if args.groups else
                 [*_DRIVER_CHANNELS, *[t[0] for t in _CLIMATE_TARGETS]])
        var_order = [v for v in known if v in results]
        if args.groups:   # cache predating a _GROUPS change would silently lose bars
            missing = {g for g in _CONFIG_INDEPENDENT_GROUPS
                       if not (sens and g in _ABLATE_ONLY_GROUPS)} - set(results)
            if missing:
                print(f"Cache {results_cache.name} is missing {sorted(missing)} "
                      f"→ recomputing")
                results = None
    if results is None:
        results, nan_counts, var_order = _compute(args, results_cache, counts_cache,
                                                  samples_csv)

    if sens:
        make_sens_overall_figure(out_dir / f"{prefix}_overall.png", results, var_order,
                                 args.sigma, signed=False, noise_floor_m=args.noise_floor_m)
        make_sens_overall_figure(out_dir / f"{prefix}_overall_signed.png", results, var_order,
                                 args.sigma, signed=True, noise_floor_m=args.noise_floor_m)
        make_season_figure(out_dir / f"{prefix}_by_season.png", results, var_order,
                           transform=lambda v: abs(v) / args.sigma,
                           ylabel=f"|ΔMAE|/σ  (m per +{args.sigma:g}σ)",
                           title=f"Input accuracy sensitivity by season (+{args.sigma:g}σ, |ΔMAE|)")
        save_csv(out_dir / f"{prefix}_by_glacier.csv", results, nan_counts, var_order, sens=True)
    else:                                                          # ablate — unchanged
        make_overall_figure(out_dir / f"{prefix}_overall.png", results, var_order,
                            args.noise_floor_m, signed=False, groups=args.groups)
        make_overall_figure(out_dir / f"{prefix}_overall_signed.png", results, var_order,
                            args.noise_floor_m, signed=True, groups=args.groups)
        make_combined_figure(out_dir / f"{prefix}_overall_combined.png", results, var_order)
        make_current_figure(out_dir / f"{prefix}_by_current.png", results, var_order,
                            _load_current_lookup())
        make_season_figure(out_dir / f"{prefix}_by_season.png", results, var_order)
        save_csv(out_dir / f"{prefix}_by_glacier.csv", results, nan_counts, var_order)
    save_csv(out_dir / f"{prefix}_glacier_mean.csv", results, nan_counts, var_order,
             sens=sens, by_season=False)

    baseline = _load_baseline_mae(args.experiment, args.baseline_csv)
    if baseline:
        for relative in (False, True):
            name = f"{prefix}_glacier_spread{'_rel' if relative else ''}.png"
            make_glacier_spread_figure(
                out_dir / name, results, var_order, baseline, _load_current_lookup(),
                relative=relative, min_n=args.min_glacier_n,
                scale=args.sigma if sens else 1.0, unit="m/σ" if sens else "m")


if __name__ == "__main__":
    main(parse_args())
