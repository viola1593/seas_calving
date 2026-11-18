# evaluate_film.py
"""
Post-training evaluation for CalvingConvLSTMFiLM trained via train_film.py.

Loads a Lightning checkpoint, auto-discovers config.yaml from the experiment
directory, and runs the full geometric metric suite on val or test split.

The summary JSON adds fast/slow stratified skill scores on top of the
standard evaluate.py outputs.  The 100 m threshold is applied per sample:
a sample is "fast" if the true calving front displacement for that prediction
exceeds 100 m (≈ 100 m/yr at the nominal 365-day horizon).

Checkpoint discovery
--------------------
Given --experiment experiments/<name>, the script searches
experiments/<name>/checkpoints/ for a file matching best*.ckpt (falling back
to any single .ckpt).  config.yaml is read from the experiment root.

Outputs
-------
  <experiment>/eval/{split}_results.csv           per-sample metrics
  <experiment>/eval/{split}_per_glacier.csv       per-glacier aggregates
  <experiment>/eval/{split}_stratified.csv        stratified by sensor / season / horizon
  <experiment>/eval/{split}_summary.json          headline numbers + fast/slow skill scores
  <experiment>/eval/{split}_error_vs_displacement.png   hedging diagnostic (png + pdf)

Usage
-----
  python evaluate_film.py --experiment experiments/convlstm_film_v1_20250101_120000
  python evaluate_film.py --experiment ... --split val --batch_size 4
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import date
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
import zarr
from tqdm import tqdm

from data.datamodule import CalvingFrontDataModule
from models.lightning_module import CalvingLightningModule
from metrics import aggregate_metrics, DEFAULT_RESOLUTION_M, DEFAULT_FRONT_BAND_M
from plotting.plot_predictions import plot_glacier_predictions, plot_single_prediction, plot_glacier_skill


# Glacier IDs to plot after evaluation. Set to [] to skip all plotting.
PLOT_GLACIERS: list[str] = []
['RGI60-07.00470', # in test dataset
                            'RGI60-07.00042', 
                            # 'RGI60-07.00224', does not seem like very much movement but lots of samples
                            'RGI60-07.00251',
                            'RGI60-07.00285_C',
                            'RGI60-07.00424_A', # test set
                            'RGI60-07.00424_B',
                            'RGI60-07.00551',
                            'RGI60-07.00552',
                            'RGI60-07.00727', # test set
                            'RGI60-07.00735',
                            'RGI60-07.00887_B',
                            'RGI60-07.00984_A',
                            'RGI60-07.01354_2', # test set
                            'RGI60-07.01449',
                            'RGI60-07.01563_B',
                            'RGI60-07.00240',
                            'RGI60-07.00250',
                            'RGI60-07.01493', #test set

                            ]  # e.g. ['RGI60-11.00897', 'RGI60-11.00901']

LOG_SUBDIR = "eval_paper"
SENSOR_CODE_MAP = {0: "S2", 1: "LE07", 2: "LC08", 3: "LC09", 255: "unknown"}
HEADLINE_METRICS = [
    "mae_front", "rmse_front",
     "active_mde", "active_persist_mde",
     "active_hausdorff", "active_persist_hausdorff",
    "advance_retreat_bias", "area_error_km2", "front_iou",
]

_zarr_stores: dict[str, zarr.Group] = {}


def _get_zarr(path: str) -> zarr.Group:
    if path not in _zarr_stores:
        try:
            _zarr_stores[path] = zarr.open_group(path, mode="r")
        except Exception as e:
            _zarr_stores[path] = zarr.open_group("../"+path, mode="r")


    return _zarr_stores[path]


def get_sensor_name(zarr_path: str, input_zarr_indices: list[int]) -> str:
    store = _get_zarr(zarr_path)
    if "sensors" not in store:
        return "unknown"
    return SENSOR_CODE_MAP.get(int(store["sensors"][input_zarr_indices[-1]]), "unknown")


def date_to_season(iso_date: str) -> str:
    try:
        m = date.fromisoformat(iso_date).month
    except (ValueError, TypeError):
        return "unknown"
    if m in (12, 1, 2): return "DJF"
    if m in (3,  4, 5): return "MAM"
    if m in (6,  7, 8): return "JJA"
    return "SON"


# ── Checkpoint discovery ──────────────────────────────────────────────────────

def find_best_checkpoint(experiment_dir: Path, chkpt: str='best') -> Path:
    ckpt_dir = experiment_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        raise FileNotFoundError(f"No checkpoints/ directory in {experiment_dir}")
    best = sorted(ckpt_dir.rglob(chkpt + "*.ckpt"))
    if len(best) == 1:
        return best[0]
    if len(best) > 1:
        raise RuntimeError(
            f"Multiple best checkpoints found in {ckpt_dir}; delete all but one:\n"
            + "\n".join(f"  {p}" for p in best)
        )
    all_ckpts = sorted(ckpt_dir.rglob("*.ckpt"))
    if len(all_ckpts) == 1:
        return all_ckpts[0]
    if len(all_ckpts) > 1:
        # make warning instead of error since some users may have multiple checkpoints and just want to evaluate one of them without deleting the others; we pick the first one in sorted order but the user should really be more explicit
        
        warnings.warn(
            f"Multiple checkpoints found but none named 'best*'; using checkpoint\n"
            + "\n"+f": {all_ckpts[0]}\n" 
                        )
        
        return all_ckpts[0]
    raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")


# ── Model loading ─────────────────────────────────────────────────────────────

def load_module_and_config(
    checkpoint_path: Path,
    experiment_dir:  Path,
    device:          torch.device,
) -> tuple[CalvingLightningModule, dict]:
    """
    Load CalvingLightningModule from a Lightning checkpoint and return the
    config.yaml that train_film.py saved alongside it.
    """
    config_path = experiment_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {config_path}.\n"
            "Expected layout:\n"
            "  experiments/<name>/config.yaml\n"
            "  experiments/<name>/checkpoints/<ckpt>.ckpt"
        )
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    module = CalvingLightningModule.load_from_checkpoint(
        str(checkpoint_path), map_location=device
    )
    module.to(device)
    module.eval()
    print(f"Loaded  : {checkpoint_path.name}")
    print(f"Config  : {config_path}")
    return module, cfg


# ── Plot sample collection ────────────────────────────────────────────────────

_STORE_CACHE: dict[str, zarr.Group] = {}


def _open_store(path: str) -> zarr.Group:
    """Open a zarr store, mirroring the dataset's ``../`` path fallback."""
    if path not in _STORE_CACHE:
        try:
            _STORE_CACHE[path] = zarr.open_group(path, mode="r")
        except Exception:
            _STORE_CACHE[path] = zarr.open_group("../" + path, mode="r")
    return _STORE_CACHE[path]


def _collect_plot_samples(
    module:         CalvingLightningModule,
    pred:           torch.Tensor,
    batch:          dict,
    sample_records: list[dict],
    plot_samples:   dict[str, list[dict]],
    collect_all:    bool = False,
) -> None:
    """Extract spatial arrays for glaciers in PLOT_GLACIERS and accumulate them."""
    disp_scale = module.model.disp_scale.item()
    disp_std   = float(module.hparams.disp_std)
    sdt_std    = disp_scale * disp_std
    sdt_mean   = float(module.hparams.sdt_mean)

    last_norm    = batch["inputs"][:, -1, 0:1]           # (B, 1, H, W)
    pred_sdt_n   = pred / disp_scale + last_norm

    last_sdt_m   = (last_norm[:, 0]  * sdt_std + sdt_mean).cpu().numpy()
    pred_disp_m  = (pred[:, 0]       * disp_std           ).cpu().numpy()
    true_disp_m  = (batch["target_displacement"][:, 0] * disp_std).cpu().numpy()
    pred_sdt_m   = (pred_sdt_n[:, 0] * sdt_std + sdt_mean).cpu().numpy()
    input_trace  = batch["input_trace"][:, 0].cpu().numpy()
    target_trace = batch["target_trace"][:, 0].cpu().numpy()
    all_input_sdts_m = batch["inputs"][:, :, 0].cpu().numpy() * sdt_std + sdt_mean  # (B, K, H, W)

    for b in range(pred.shape[0]):
        gid = batch["glacier_id"][b]
        if not collect_all and gid not in PLOT_GLACIERS:
            continue
        store = _open_store(batch["zarr_path"][b])
        input_dates = [str(d) for d in store["dates"][list(batch["input_zarr_indices"][b])]]
        plot_samples.setdefault(gid, []).append({
            "last_sdt_m":        last_sdt_m[b],
            "pred_disp_m":       pred_disp_m[b],
            "true_disp_m":       true_disp_m[b],
            "pred_sdt_m":        pred_sdt_m[b],
            "input_trace":       input_trace[b],
            "target_trace":      target_trace[b],
            "all_input_sdts_m":  all_input_sdts_m[b],
            "input_dates":         input_dates,
            "original_hw":         batch["original_hw"][b],
            "glacier_id":          gid,
            "target_date":         batch["target_date"][b],
            "active_skill_score":  sample_records[b].get("active_skill_score", float("nan")),
            "active_mde":          sample_records[b].get("active_mde", float("nan")),
            "active_persist_mde":  sample_records[b].get("active_persist_mde", float("nan")),
            "glacier_seasonality": sample_records[b]["glacier_seasonality"],
        })


# ── Evaluation loop ───────────────────────────────────────────────────────────

@torch.no_grad()
def run_evaluation(
    module:     CalvingLightningModule,
    dataloader,
    device:     torch.device,
    plot_skill: bool = False,
    plot_mde:   bool = False,
) -> tuple[list[dict], dict[str, list[dict]]]:
    """
    Run inference over the dataloader and return per-sample metric dicts and
    spatial plot samples for any glaciers listed in PLOT_GLACIERS.
    """
    records:      list[dict]            = []
    plot_samples: dict[str, list[dict]] = {}

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        batch = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        pred           = module.model(batch)
        sample_records = module._batch_to_records(pred, batch)

        for b, rec in enumerate(sample_records):
            zarr_path = batch["zarr_path"][b]
            ii        = batch["input_zarr_indices"][b]
            seas_val  = float(batch["glacier_seasonality"][b])
            rec.update({
                "glacier_id":          batch["glacier_id"][b],
                "target_date":         batch["target_date"][b],
                "horizon_days":        int(batch["horizon_days"][b]),
                "season":              date_to_season(batch["target_date"][b]),
                "sensor":              get_sensor_name(zarr_path, ii),
                "glacier_seasonality": "seasonal" if seas_val > 0.5 else "non-seasonal",
            })

        records.extend(sample_records)

        if PLOT_GLACIERS or plot_skill or plot_mde:
            _collect_plot_samples(
                module, pred, batch, sample_records, plot_samples,
                collect_all=(plot_skill or plot_mde),
            )

    return records, plot_samples


# ── Aggregation helpers ───────────────────────────────────────────────────────

def compute_per_glacier(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c in HEADLINE_METRICS or c == "active_skill_score"]
    rows = []
    for glacier_id, gdf in df.groupby("glacier_id"):
        row = {
            "glacier_id": glacier_id,
            "n_samples":  len(gdf),
            "date_min":   gdf["target_date"].min(),
            "date_max":   gdf["target_date"].max(),
            "sensors":    ",".join(sorted(gdf["sensor"].unique())),
        }
        for col in metric_cols:
            v = gdf[col].dropna()
            row[f"{col}_mean"]   = float(v.mean())   if len(v) else float("nan")
            row[f"{col}_median"] = float(v.median()) if len(v) else float("nan")
            row[f"{col}_std"]    = float(v.std())    if len(v) else float("nan")
        rows.append(row)
    return pd.DataFrame(rows).sort_values("glacier_id")


def compute_stratified_breakdown(df: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in df.columns if c in HEADLINE_METRICS or c == "active_skill_score"]
    strata = []
    for groupby_col in ["sensor", "season", "horizon_days", "glacier_seasonality"]:
        if groupby_col not in df.columns:
            continue
        for group_val, group_df in df.groupby(groupby_col):
            row = {
                "stratum_col": groupby_col,
                "stratum_val": str(group_val),
                "n_samples":   len(group_df),
            }
            for col in metric_cols:
                v = group_df[col].dropna()
                row[f"{col}_mean"]   = float(v.mean())   if len(v) else float("nan")
                row[f"{col}_median"] = float(v.median()) if len(v) else float("nan")
            strata.append(row)
    return pd.DataFrame(strata)


# ── Error-vs-displacement diagnostic ──────────────────────────────────────────

# Kept clear of the blue/orange used for the model/persistence binned medians.
SEASONALITY_COLORS = {"seasonal": "tab:green", "non-seasonal": "tab:purple"}


def _scatter_by_seasonality(ax, d: pd.DataFrame, x, y, label_suffix: str) -> None:
    """Scatter x vs y, split by glacier seasonality if the column is present."""
    if "glacier_seasonality" not in d.columns:
        ax.scatter(x, y, s=9, alpha=0.25, color="tab:blue", edgecolors="none",
                   label=f"model{label_suffix}")
        return
    seas = d["glacier_seasonality"].to_numpy()
    for label, colour in SEASONALITY_COLORS.items():
        m = seas == label
        if m.any():
            ax.scatter(x[m], y[m], s=9, alpha=0.25, color=colour, edgecolors="none",
                       label=f"{label}{label_suffix} (n = {int(m.sum())})")


def _opaque_legend(ax, **kwargs) -> None:
    """Legend whose markers ignore the scatter's alpha, so they stay readable."""
    leg = ax.legend(**kwargs)
    for h in leg.legend_handles:
        h.set_alpha(1.0)


def plot_error_vs_displacement(df: pd.DataFrame, out_path: Path) -> float:
    """
    Two-panel diagnostic for regression-to-the-mean ("hedging").

    Left  : predicted vs true signed front displacement, one point per sample.
            A model that hedges toward zero sits on a line flatter than y = x;
            the fitted slope quantifies how much of the true motion it reproduces.
    Right : error vs true displacement magnitude, model and persistence, with
            binned medians. If the two curves converge as displacement grows,
            the model adds nothing over persistence on the large movers.

    Returns the fitted slope of the left panel (NaN if it cannot be fitted).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    # ── Left: predicted vs true signed displacement ───────────────────────
    d = df.dropna(subset=["signed_true_displacement_m", "signed_pred_displacement_m"])
    t = d["signed_true_displacement_m"].to_numpy()
    p = d["signed_pred_displacement_m"].to_numpy()

    slope = float("nan")
    if len(t) >= 2:
        lim = float(np.percentile(np.abs(np.concatenate([t, p])), 99.5))
        slope, intercept = np.polyfit(t, p, 1)
        corr = float(np.corrcoef(t, p)[0, 1])

        _scatter_by_seasonality(ax1, d, t, p, "")
        ax1.plot([-lim, lim], [-lim, lim], "k--", lw=1.2, label="perfect (slope 1.00)")
        ax1.plot([-lim, lim], [-lim * slope + intercept, lim * slope + intercept],
                 color="tab:red", lw=1.8, label=f"fit (slope {slope:.2f}, r = {corr:.2f})")
        ax1.axhline(0, color="grey", lw=0.6)
        ax1.axvline(0, color="grey", lw=0.6)
        ax1.set_xlim(-lim, lim)
        ax1.set_ylim(-lim, lim)
        ax1.set_aspect("equal")
        _opaque_legend(ax1, loc="upper left", fontsize=9)

    ax1.set_xlabel("true displacement at front (m)   ← retreat | advance →")
    ax1.set_ylabel("predicted displacement at front (m)")
    ax1.set_title("Does the model match the size of the motion?")

    # ── Right: error vs displacement magnitude ────────────────────────────
    d2 = df.dropna(subset=["mean_true_displacement_m", "active_mde"])
    x  = d2["mean_true_displacement_m"].to_numpy()
    ym = d2["active_mde"].to_numpy()
    yp = d2["active_persist_mde"].to_numpy()

    _scatter_by_seasonality(ax2, d2, x, ym, " (per sample)")

    if len(x) >= 10:
        # Equal-count bins so every marker rests on the same number of samples.
        edges = np.unique(np.percentile(x, np.linspace(0, 100, 9)))
        idx   = np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)
        centres = np.array([np.median(x[idx == i])  for i in range(len(edges) - 1)])
        med_m   = np.array([np.median(ym[idx == i]) for i in range(len(edges) - 1)])
        med_p   = np.array([np.median(yp[idx == i]) for i in range(len(edges) - 1)])
        ax2.plot(centres, med_m, "o-", color="tab:blue", lw=2, label="model (binned median)")
        ax2.plot(centres, med_p, "s--", color="tab:orange", lw=2, label="persistence (binned median)")

    ax2.set_xlabel("true displacement magnitude at front (m)")
    ax2.set_ylabel("MDE (m)")
    ax2.set_title("Where does the error come from?")
    _opaque_legend(ax2, loc="upper left", fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    fig.savefig(out_path.with_suffix(".pdf"))
    plt.close(fig)
    return slope


def build_summary(
    df:        pd.DataFrame,
    agg:       dict,
    split:     str,
    ckpt_path: Path,
) -> dict:
    def _stats(col: str) -> dict:
        v = df[col].dropna()
        if len(v) == 0:
            return {"mean": float("nan"), "std": float("nan"), "median": float("nan")}
        return {"mean": float(v.mean()), "std": float(v.std()), "median": float(v.median())}

    # valid_skill = df["skill_score"].dropna()
    active_skill = df["active_skill_score"].dropna() if "active_skill_score" in df.columns else pd.Series(dtype=float)
    return {
        "split":      split,
        "checkpoint": str(ckpt_path),
        "n_samples":  len(df),
        "n_glaciers": int(df["glacier_id"].nunique()),
        "model":      {m: _stats(m) for m in HEADLINE_METRICS},
        # "skill": {
        #     **_stats("skill_score"),
        #     "frac_positive": float((valid_skill > 0).mean()) if len(valid_skill) else float("nan"),
        #     "slow_front_movers": {
        #         "description": "true calving front displacement < 100 m",
        #         "mean":        float(agg.get("skill_score_slow_mean",   float("nan"))),
        #         "median":      float(agg.get("skill_score_slow_median", float("nan"))),
        #         "n":           int(agg.get("n_slow", 0)),
        #     },
        #     "fast_front_movers": {
        #         "description": "true calving front displacement >= 100 m",
        #         "mean":        float(agg.get("skill_score_fast_mean",   float("nan"))),
        #         "median":      float(agg.get("skill_score_fast_median", float("nan"))),
        #         "n":           int(agg.get("n_fast", 0)),
        #     },
        # },
        "active_skill": {
            **_stats("active_skill_score"),
            "frac_positive": float((active_skill > 0).mean()) if len(active_skill) else float("nan"),
            "slow_front_movers": {
                "description": "true calving front displacement < 100 m",
                "mean":        float(agg.get("active_skill_score_slow_mean",   float("nan"))),
                "median":      float(agg.get("active_skill_score_slow_median", float("nan"))),
                "n":           int(agg.get("n_slow", 0)),
            },
            "fast_front_movers": {
                "description": "true calving front displacement >= 100 m",
                "mean":        float(agg.get("active_skill_score_fast_mean",   float("nan"))),
                "median":      float(agg.get("active_skill_score_fast_median", float("nan"))),
                "n":           int(agg.get("n_fast", 0)),
            },
        },
        "seasonality": {
            label: {
                "n":                  int(len(sub)),
                "active_skill_score": {"mean": float(sub["active_skill_score"].mean()) if len(sub) else float("nan"),
                                       "median": float(sub["active_skill_score"].median()) if len(sub) else float("nan")},
                "active_mde":         {"mean": float(sub["active_mde"].mean()) if len(sub) else float("nan"),
                                       "median": float(sub["active_mde"].median()) if len(sub) else float("nan")},
                "mae_front":          {"mean": float(sub["mae_front"].mean()) if len(sub) else float("nan"),
                                       "median": float(sub["mae_front"].median()) if len(sub) else float("nan")},
            }
            for label, sub in df.groupby("glacier_seasonality")
        },
    }


# ── Pretty-print ──────────────────────────────────────────────────────────────

def print_summary(summary: dict) -> None:
    print(f"\n{'='*65}")
    print(f"EVALUATION RESULTS — {summary['split'].upper()} SPLIT")
    print(f"{'='*65}")
    print(f"  Checkpoint : {summary['checkpoint']}")
    print(f"  Samples    : {summary['n_samples']}")
    print(f"  Glaciers   : {summary['n_glaciers']}")

    metric_units = {
        "mae_front":            "m",
        "rmse_front":           "m",
        # "mde":                  "m",
        "active_mde":           "m",
        "active_persist_mde":          "m",
        # "hausdorff":            "m",
        "active_hausdorff":     "m",
        "active_persist_hausdorff":    "m",
        "advance_retreat_bias": "m",
        "area_error_km2":       "km²",
        "front_iou":            "",
    }
    print(f"\n  {'Metric':<28} {'Mean':>10} {'Std':>8} {'Median':>8}")
    print(f"  {'-'*56}")
    for m, unit in metric_units.items():
        s     = summary["model"].get(m, {})
        label = f"{m} ({unit})" if unit else m
        mean, std, med = s.get("mean", float("nan")), s.get("std", float("nan")), s.get("median", float("nan"))
        print(f"  {label:<28} {mean:>10.2f} {std:>8.2f} {med:>8.2f}")

    sk = summary["active_skill"]
    print(f"\n  Overall active skill score  : {sk['mean']:.3f} ± {sk['std']:.3f}  (median {sk['median']:.3f})")
    print(f"  Beats persistence    : {sk.get('frac_positive', float('nan')):.1%} of samples")

    slow = sk.get("slow_front_movers", {})
    fast = sk.get("fast_front_movers", {})
    print(f"\n Active skill by calving front displacement (threshold 100 m):")
    print(f"    Slow (<100 m)  n={slow.get('n', 0):5d}   mean={slow.get('mean', float('nan')):+.3f}   median={slow.get('median', float('nan')):+.3f}")
    print(f"    Fast (≥100 m)  n={fast.get('n', 0):5d}   mean={fast.get('mean', float('nan')):+.3f}   median={fast.get('median', float('nan')):+.3f}")

    seas_block = summary.get("seasonality", {})
    print(f"\n Active skill by glacier seasonality:")
    for label in ("seasonal", "non-seasonal"):
        g = seas_block.get(label, {})
        n    = g.get("n", 0)
        sk_  = g.get("active_skill_score", {})
        mde_ = g.get("active_mde", {})
        mae_ = g.get("mae_front", {})
        print(f"    {label:<13}  n={n:5d}   skill={sk_.get('mean', float('nan')):+.3f}   "
              f"mde={mde_.get('mean', float('nan')):6.1f} m   mae={mae_.get('mean', float('nan')):6.1f} m")
    print(f"{'='*65}\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def evaluate(
    experiment_dir: Path,
    split:          str = "test",
    chkpt:          str = "best",
    batch_size:     int = 4,
    num_workers:    int = 4,
    device_str:     str = "cuda",
    plot_skill:     bool = False,
    plot_mde:       bool = False,
) -> pd.DataFrame:
    checkpoint_path = find_best_checkpoint(experiment_dir, chkpt=chkpt)
    log_dir = experiment_dir / LOG_SUBDIR / chkpt
    log_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(
        device_str if torch.cuda.is_available() or device_str == "cpu" else "cpu"
    )
    print(f"Device  : {device}")
    print(f"Split   : {split}")

    module, cfg = load_module_and_config(checkpoint_path, experiment_dir, device)
    n_params = sum(p.numel() for p in module.parameters())
    print(f"Params  : {n_params:,}")

    dm = CalvingFrontDataModule(
        **{**cfg["data"], "batch_size": batch_size, "num_workers": num_workers, "augment_train": False}
    )
    dm.setup("test" if split == "test" else "fit")
    loader = dm.test_dataloader() if split == "test" else dm.val_dataloader()
    print(f"Samples : {len(loader.dataset)} ({len(loader)} batches)\n")

    records, plot_samples = run_evaluation(module, loader, device, plot_skill=plot_skill, plot_mde=plot_mde)
    if not records:
        warnings.warn("No records produced — check that the DataLoader is non-empty.")
        return pd.DataFrame()

    df          = pd.DataFrame(records)
    agg         = aggregate_metrics(records)
    per_glacier = compute_per_glacier(df)
    stratified  = compute_stratified_breakdown(df)
    summary     = build_summary(df, agg, split, checkpoint_path)

    csv_path     = log_dir / f"{split}_results.csv"
    glacier_path = log_dir / f"{split}_per_glacier.csv"
    strat_path   = log_dir / f"{split}_stratified.csv"
    summary_path = log_dir / f"{split}_summary.json"

    df.to_csv(csv_path,              index=False)
    per_glacier.to_csv(glacier_path, index=False)
    stratified.to_csv(strat_path,    index=False)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    disp_plot = log_dir / f"{split}_error_vs_displacement.png"
    slope     = plot_error_vs_displacement(df, disp_plot)
    print(f"\n  Displacement response slope: {slope:.3f}  "
          f"(1.0 = full motion reproduced, 0.0 = predicts persistence)")

    if plot_samples:
        if not plot_skill:
            plots_dir = log_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            for gid, samples in plot_samples.items():
                if gid not in PLOT_GLACIERS:
                    continue
                samples_sorted = sorted(samples, key=lambda s: s["target_date"])
                plot_glacier_predictions(samples_sorted, plots_dir / f"{gid}.png")
            print(f"Glacier plots written to {plots_dir}")

    if plot_skill:
        skill_dir = log_dir / "plots_skill"
        skill_dir.mkdir(exist_ok=True)
        skipped = []
        for gid, samples in plot_samples.items():
            if any(np.isfinite(s.get("active_skill_score", float("nan"))) for s in samples):
                seas_tag = "S" if samples[0]["glacier_seasonality"] == "seasonal" else "NS"
                plot_glacier_skill(samples, skill_dir / f"{gid}_{seas_tag}.png")
            else:
                skipped.append(gid)
        if skipped:
            warnings.warn(f"plot-skill: skipped all-NaN glaciers: {', '.join(skipped)}")
        print(f"Per-glacier skill plots written to {skill_dir}")
    
    if plot_mde:
        skill_dir = log_dir / "plots_mde"
        skill_dir.mkdir(exist_ok=True)
        skipped = []
        for gid, samples in plot_samples.items():
            if any(np.isfinite(s.get("active_mde", float("nan"))) for s in samples):
                seas_tag = "S" if samples[0]["glacier_seasonality"] == "seasonal" else "NS"
                plot_glacier_skill(samples, skill_dir / f"{gid}_{seas_tag}.png", metric="active_mde")
            else:
                skipped.append(gid)
        if skipped:
            warnings.warn(f"plot-mde: skipped all-NaN glaciers: {', '.join(skipped)}")
        print(f"Per-glacier mde plots written to {skill_dir}")

    print_summary(summary)
    print(f"Outputs written to {log_dir}")
    print(f"  {csv_path.name:<40}  per-sample results")
    print(f"  {glacier_path.name:<40}  per-glacier aggregates")
    print(f"  {strat_path.name:<40}  stratified breakdown")
    print(f"  {summary_path.name:<40}  headline numbers (JSON)")
    print(f"  {disp_plot.name:<40}  error vs true displacement")

    return df


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a CalvingConvLSTMFiLM Lightning checkpoint."
    )
    parser.add_argument(
        "--experiment", type=Path, required=True,
        help="Path to experiment directory (e.g. experiments/convlstm_film_v1_20250101_120000). "
             "The best checkpoint is discovered automatically from checkpoints/. "
             f"Outputs are written to <experiment>/{LOG_SUBDIR}/.",
    )
    parser.add_argument(
        "--split", type=str, default="test", choices=["val", "test"],
        help="Split to evaluate (default: test)",
    )
    parser.add_argument(
        "--chkpt", type=str, default="best",
        help="Checkpoint to evaluate (best, last, default: best)",
    )
    parser.add_argument("--batch_size",  type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="'cuda', 'cpu', or 'cuda:N' (default: cuda)",
    )
    parser.add_argument(
        "--plot-skill", action="store_true",
        help="Emit a per-glacier best/median/worst skill figure to plots_skill/.",
    )
    parser.add_argument(
        "--plot-mde", action="store_true",
        help="Emit a per-glacier best/median/worst MDE figure to plots_mde/.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.split == "test":
        val_summary = args.experiment / LOG_SUBDIR / "val_summary.json"
        # if not val_summary.exists():
        #     raise RuntimeError(
        #         f"val_summary.json not found at {val_summary}. "
        #         "Run on val split first:\n"
        #         "  python evaluate_film.py --experiment ... --split val\n"
        #         "Only evaluate on test when all modelling decisions are final."
        #     )

    evaluate(
        experiment_dir = args.experiment,
        split          = args.split,
        chkpt           = args.chkpt, 
        batch_size     = args.batch_size,
        num_workers    = args.num_workers,
        device_str     = args.device,
        plot_skill     = args.plot_skill,
        plot_mde       = args.plot_mde
    )
