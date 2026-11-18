"""
Per-glacier calving front displacement along the centerline — under rollout.

Like eval_centerline.py, but the "model" series is the AUTOREGRESSIVE rollout
(rollout_film.py): each prediction is fed back as the SDT input of later steps,
so error compounds as the rollout detaches from real observations. For each
glacier we plot front displacement (metres along the centerline, relative to t0)
for four series:

    true        target SDT at each step
    model       autoregressive rollout prediction (fed back into itself)
    teacher     teacher-forced prediction (real inputs every step; upper bound)
    persistence front frozen at t0 (first sample's real anchor) — flat baseline

Front position is measured exactly as in eval_centerline.py: densify the
centerline to one point per pixel, walk it inland→ocean over the SDT map, and
take the arc length to the last +→− zero-crossing (fallback: min |SDT|).

The model markers are coloured by autoregressive depth (number of synthetic
input frames), so you can see the front position drift as the rollout detaches.

Plots are written to <experiment>/rollout/centerlines/ (one PNG per glacier),
together with {split}_centerline_arcs.csv (per-step arcs, elapsed_days and the
rollout / single-step MAD) and {split}_centerline_mad_by_lead.csv (MAD binned by
months since each glacier's step-0 target, averaged per glacier, then across glaciers).

Usage:
    python rollout_centerline.py \\
        --experiment experiments/seasonal/paper_M3_seed3 \\
        --split      test

    python rollout_centerline.py \\
        --experiment experiments/seasonal/paper_M3_seed3 \\
        --glacier    RGI60-07.00470  --delta-days 30  --gpu 0

    python rollout_centerline.py \\
        --experiment experiments/seasonal/paper_M3_seed3 \\
        --glacier    RGI60-07.00470  --start-date 2021-06-01
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_centerline import CALVING_GPKG, _centerline_arc_length
from rollout_film import (
    PredictionBuffer,
    _model_predict_sdt_norm,
    _normalized_sample,
    _sample_target_date,
    autoreg_depth,
    denorm_sdt,
    frame_dates,
    load_glacier_sample_indices,
    norm_sdt,
    recompute_sdt_diff,
    redistance_sdt_m,
    substitute_sdt,
)


def _arc(sdt_m, gid, centerlines):
    """_centerline_arc_length but None → np.nan for tidy DataFrames/plots."""
    a = _centerline_arc_length(sdt_m, gid, centerlines)
    return np.nan if a is None else a


# ── Rollout with per-step centerline arcs ────────────────────────────────────

def rollout_glacier_arcs(
    module, dataset, glacier_id, centerlines, delta_days, indices,
    redistance: bool = True, freeze: str = "none",
) -> list[dict]:
    """Autoregressive rollout over one glacier, recording centerline arc lengths.

    Mirrors rollout_film.rollout_glacier's step loop, but instead of scoring
    metrics it measures the front's centerline arc length (metres) for the
    model / teacher / true / persistence SDT maps at each step.
    """
    import torch

    device     = next(module.parameters()).device
    disp_scale = module.model.disp_scale.item()
    sdt_std    = disp_scale * float(module.hparams.disp_std)
    sdt_mean   = float(module.hparams.sdt_mean)

    first_s = dataset.samples[indices[0]]
    store   = dataset._get_store(first_s["zarr_path"])

    # Rollout persistence: front frozen at the first sample's real anchor (t0).
    t0_zi         = int(first_s["input_zarr_indices"][-1])
    persist_sdt_m = np.nan_to_num(store["sdt"][t0_zi].astype(np.float32))
    persist_arc   = _arc(persist_sdt_m, glacier_id, centerlines)

    buffer  = PredictionBuffer()
    records: list[dict] = []
    for step, idx in enumerate(indices):
        s           = dataset.samples[idx]
        ti          = int(s["target_zarr_index"])
        target_date = _sample_target_date(s)
        fdates      = frame_dates(_normalized_sample(s))

        sample        = dataset[idx]                     # normalised tensors
        real_inputs   = sample["inputs"]                 # (K, C, H, W)
        sdt_norm_real = real_inputs[:, 0].numpy().copy()

        # Snapshot the first sample's full input state for the freeze ablations.
        if step == 0:
            frozen_sample = {k: (v.clone() if torch.is_tensor(v) else v)
                             for k, v in sample.items()}
            frozen_inputs = real_inputs.clone()

        new_sdt_norm, prov = substitute_sdt(sdt_norm_real, fdates, buffer, delta_days)

        auto_inputs = real_inputs.clone()
        auto_inputs[:, 0] = torch.from_numpy(new_sdt_norm)
        auto_inputs[:, 1] = torch.from_numpy(recompute_sdt_diff(new_sdt_norm))

        # Model-reliance ablation on the autoregressive prediction (see rollout_film).
        # All modes KEEP channels 0-1 (SDF + sdt_diff) substituted with the fed-back
        # prediction (autoregressive) and leave τ (observation timing) live. --freeze
        # features freezes the historical inputs (geophys, historical climate) AND
        # forcing_climate at the first sample; --freeze history lets forcing_climate stay
        # live; --freeze geophys freezes only the geophysical spatial channels and lets all
        # climate evolve; --freeze climate is the mirror image, freezing all climate and
        # letting geophys evolve.
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
        # With redistance the buffered frame is snapped back to a valid single-
        # front SDF so the next input stays in-distribution; the measured
        # model_sdt_m below is left raw.
        if redistance:
            buf_sdt_m = redistance_sdt_m(denorm_sdt(auto_sdt_norm, sdt_mean, sdt_std))
            buffer.add(target_date, norm_sdt(buf_sdt_m, sdt_mean, sdt_std))
        else:
            buffer.add(target_date, auto_sdt_norm)

        model_sdt_m   = denorm_sdt(auto_sdt_norm,    sdt_mean, sdt_std)
        teacher_sdt_m = denorm_sdt(teacher_sdt_norm, sdt_mean, sdt_std)
        target_sdt_m  = np.nan_to_num(store["sdt"][ti].astype(np.float32))

        records.append({
            "glacier_id":    glacier_id,
            "step":          step,
            "autoreg_depth": autoreg_depth(prov),
            "target_date":   target_date,
            "true_arc_m":    _arc(target_sdt_m,  glacier_id, centerlines),
            "model_arc_m":   _arc(model_sdt_m,   glacier_id, centerlines),
            "teacher_arc_m": _arc(teacher_sdt_m, glacier_id, centerlines),
            "persist_arc_m": persist_arc,
        })

    return records


# ── Plotting ─────────────────────────────────────────────────────────────────

def _mae(df, col):
    v = df["true_arc_m"].notna() & df[col].notna()
    return np.abs(df.loc[v, "true_arc_m"] - df.loc[v, col]).mean()


def add_mad_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Per-row elapsed time since each glacier's step-0 target and |arc − true| errors."""
    t0 = df.groupby("glacier_id")["target_date"].transform("min")
    df["elapsed_days"]      = (df["target_date"] - t0).dt.days
    df["rollout_mad_m"]     = (df["model_arc_m"]   - df["true_arc_m"]).abs()
    df["single_step_mad_m"] = (df["teacher_arc_m"] - df["true_arc_m"]).abs()
    return df


def mad_by_lead(df: pd.DataFrame) -> pd.DataFrame:
    """Glacier-weighted rollout vs single-step MAD, binned by elapsed months."""
    # Paired: only rows where true, rollout and single-step arcs all exist.
    df = df.dropna(subset=["true_arc_m", "model_arc_m", "teacher_arc_m"])
    months = df["elapsed_days"] / 30.44
    edges = [0, 3, 6, 9, 12, 15, 24]
    while edges[-1] <= months.max():
        edges.append(edges[-1] + 12)
    labels = [f"{a}-{b}" for a, b in zip(edges[:-1], edges[1:])]
    df = df.assign(bin=pd.cut(months, edges, right=False, labels=labels))

    cols = ["rollout_mad_m", "single_step_mad_m", "autoreg_depth"]
    per_glacier = df.groupby(["bin", "glacier_id"], observed=True)[cols].mean()
    g = per_glacier.groupby(level="bin", observed=True)
    out = pd.DataFrame({
        "n_glaciers":            g.size(),
        "mean_autoreg_depth":    g["autoreg_depth"].mean(),
        "rollout_mad_mean":      g["rollout_mad_m"].mean(),
        "rollout_mad_median":    g["rollout_mad_m"].median(),
        "single_step_mad_mean":  g["single_step_mad_m"].mean(),
        "single_step_mad_median": g["single_step_mad_m"].median(),
    }).reset_index()
    out["low_n"] = out["n_glaciers"] < 5
    return out


def plot_glacier_rollout(gid: str, df: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    FS=14

    df = df.sort_values("target_date")
    t0 = df["true_arc_m"].iloc[0]

    fig, ax = plt.subplots(figsize=(11, 4))

    ax.plot(df["target_date"], df["true_arc_m"] - t0,
            color="darkmagenta", linewidth=1.4, marker="o", markersize=3, label="True")

    # ax.plot(df["target_date"], df["persist_arc_m"] - t0,
    #         color="grey", linewidth=1.0, linestyle="--", marker="",
    #         label=f"Persistence (MAD: {_mae(df, 'persist_arc_m'):.0f} m)")

    ax.plot(df["target_date"], df["teacher_arc_m"] - t0,
            color="lightcoral", linewidth=1.0, linestyle="--", marker="^", markersize=3,
            label=f"Single step (MAD: {_mae(df, 'teacher_arc_m'):.0f} m)")

    # Autoregressive model: line + markers coloured by autoregressive depth.
    model_disp = df["model_arc_m"] - t0
    ax.plot(df["target_date"], model_disp,
            color="darkorange", linewidth=1.2, zorder=3,
            label=f"Roll-out (MAD: {_mae(df, 'model_arc_m'):.0f} m)")
    sc = ax.scatter(df["target_date"], model_disp, c=df["autoreg_depth"],
                    cmap="viridis", s=28, zorder=4, edgecolors="black", linewidths=0.3)
    cb = fig.colorbar(sc, ax=ax, pad=0.01, fraction=0.04)
    cb.set_label("autoreg depth\n(predicted frames)", fontsize=FS)

    ax.axhline(0, color="black", linewidth=0.7, linestyle=":", alpha=0.4)
    ax.set_title(gid, fontsize=FS)
    ax.set_xlabel("Target date", fontsize=FS)
    ax.set_ylabel("Front displacement from t₀ [m]\n← retreat   advance →", fontsize=FS)
    # January = major (labelled, thick); Feb–Dec = minor (thin, unlabelled)
    ax.xaxis.set_major_locator(mdates.YearLocator(month=1, day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator(bymonth=range(2, 13)))
    ax.set_ylim(min(df[["true_arc_m", "model_arc_m", "teacher_arc_m"]].min().min() - t0 - 50, -200),
                df[["true_arc_m", "model_arc_m", "teacher_arc_m"]].max().max() - t0 + 50)
    fig.autofmt_xdate(rotation=30)
    ax.legend(fontsize=FS, loc="best")
    ax.grid(True, which="major", axis="x", alpha=0.45, linewidth=1.0)
    ax.grid(True, which="minor", axis="x", alpha=0.18, linewidth=0.5)
    ax.grid(True, which="major", axis="y", alpha=0.3)

    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ── Entry point ──────────────────────────────────────────────────────────────

def _experiment_label(exp_dir: Path) -> str:
    return re.sub(r"_\d{8}_\d{6}$", "", exp_dir.name)


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--chkpt", type=str, default="best")
    parser.add_argument("--glacier", nargs="*", default=None,
                        help="Optional: restrict to these glacier IDs")
    parser.add_argument("--delta-days", type=int, default=30)
    parser.add_argument("--start-date", type=date.fromisoformat, default=None,
                        help="Optional YYYY-MM-DD: drop samples with an earlier target date so "
                             "the rollout starts there (that sample's real inputs become t0).")
    parser.add_argument("--redistance", action=argparse.BooleanOptionalAction, default=True,
                        help="Reinitialise each fed-back frame to a valid single-front SDF "
                             "(matches rollout_film; on by default, disable with --no-redistance).")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--freeze", choices=["none", "history", "features", "geophys", "climate"],
                        default="none",
                        help="Model-reliance ablation (matches rollout_film); the SDF/sdt_diff "
                             "always stay autoregressive and tau always stays live. 'features' "
                             "freezes the historical inputs (geophys, historical climate) and "
                             "forcing_climate at the first sample; 'history' leaves forcing_climate "
                             "live; 'geophys' freezes only the geophysical spatial channels and lets "
                             "all climate evolve; 'climate' freezes all climate and lets geophys evolve.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu))

    import geopandas as gpd
    import torch
    from evaluate_film import find_best_checkpoint, load_module_and_config
    from data.dataset_v2 import CalvingFrontDatasetV2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = find_best_checkpoint(args.experiment, chkpt=args.chkpt)
    module, cfg = load_module_and_config(ckpt, args.experiment, device)
    module.eval()

    splits_dir = Path(cfg["data"]["splits_dir"])
    dataset = CalvingFrontDatasetV2(
        index_path=splits_dir / f"{args.split}.json",
        config=cfg["data"]["config"],
        stats_path=cfg["data"]["stats_path"],
        augment=False,
    )

    print("Loading centerlines …")
    centerlines = gpd.read_file(CALVING_GPKG, layer="centerlines")

    glacier_filter = set(args.glacier) if args.glacier else None
    glacier_ids = sorted({s["glacier_id"] for s in dataset.samples})
    if glacier_filter is not None:
        glacier_ids = [g for g in glacier_ids if g in glacier_filter]

    tag = "" if args.freeze == "none" else f"_{args.freeze}"
    out_dir = args.out or (args.experiment / "rollout" / f"centerlines{tag}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nPlotting → {out_dir}")

    all_dfs = []
    for gid in glacier_ids:
        indices = load_glacier_sample_indices(dataset, gid)
        if args.start_date is not None:
            indices = [i for i in indices
                       if _sample_target_date(dataset.samples[i]) >= args.start_date]
        if len(indices) < 2:
            continue
        recs = rollout_glacier_arcs(module, dataset, gid, centerlines,
                                    args.delta_days, indices, redistance=args.redistance,
                                    freeze=args.freeze)
        df = pd.DataFrame(recs)
        df["target_date"] = pd.to_datetime(df["target_date"])
        plot_glacier_rollout(gid, df, out_dir / f"{gid}.png")
        all_dfs.append(df)
        max_depth = df["autoreg_depth"].max()
        print(f"  {gid}: {len(df)} steps, max autoreg_depth={max_depth}")

    # The arc series itself, so downstream analysis (e.g. when the front stops
    # moving) does not have to re-run the rollout to recover it.
    if all_dfs:
        csv_path = out_dir / f"{args.split}_centerline_arcs.csv"
        arcs = add_mad_columns(pd.concat(all_dfs, ignore_index=True))
        arcs.to_csv(csv_path, index=False)
        print(f"\nWrote arcs → {csv_path}")

        table = mad_by_lead(arcs)
        table_path = out_dir / f"{args.split}_centerline_mad_by_lead.csv"
        table.to_csv(table_path, index=False)
        print(f"\nCenterline MAD by elapsed months [m] → {table_path}")
        print(table.to_string(index=False, float_format="%.0f"))

    print("Done.")


if __name__ == "__main__":
    main()
