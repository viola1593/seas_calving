"""
Calving front length and domain ice area per split, and per-glacier score vs geometry.

Two products:

1. Distributions (always). Sample-level violins of front length and domain ice
   area for train / val / test of the seasonal split, plus a descriptive summary
   table (n, min, Q1, median, Q3, max, mean). No significance test: samples from
   one glacier are near-copies (the domain polygon is fixed, so its ice area
   barely moves over time), which makes a sample-level p-value meaningless.
   Written to ``figures/``, together with a per-test-glacier table of front
   length and domain ice area (mean/std over that glacier's test samples).

2. Score vs geometry (only with ``--experiment``). One point per test glacier,
   MDE and two skill variants against both geometry measures (3x2), coloured by
   the seasonal flag and annotated with Spearman rho. Written to
   ``<experiment>/eval_paper/``.

Definitions, both read from the sample's TARGET frame (``target_zarr_index``):

  front length  = n_pixels(trace) * 30 m. A proxy: the trace is a rasterised
                  line, so the count carries an orientation-dependent bias of
                  roughly +-20% (a 45-degree front is undercounted relative to
                  an axis-aligned one). Monotone in true length, which is all
                  the rank correlation needs.
  ice area      = n_pixels(sdt > 0) * 900 m^2. SDT > 0 is inside the glacier
                  (02_seasonal_observations.py). Glacier_MASK is the fjord-mask
                  polygon on the glacier side of the trace, clipped to the
                  domain polygon, so this is ice area WITHIN THE ANALYSIS
                  DOMAIN, not RGI glacier area. Labelled accordingly.

Samples whose target frame has ``valid_trace = False`` contribute no front
length (NaN) but keep their ice area — the split index only requires
``valid_sdt``, so such frames exist and metrics.py already returns NaN for them.

The zarr scan is cached per sample in ``figures/glacier_geometry_per_sample.csv``
and reused unless ``--refresh`` is given.

Usage:
    # distributions only
    python scripts/glacier_geometry_stats.py

    # distributions + score-vs-geometry for one experiment
    python scripts/glacier_geometry_stats.py --experiment paper_M3_seed0

    # re-read the zarr stores instead of using the cache
    python scripts/glacier_geometry_stats.py --refresh
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import zarr
from scipy.stats import pearsonr, spearmanr

REPO      = Path(__file__).resolve().parents[1]          # svalcalvingdb/
SPLIT_DIR = REPO.parent / "glacier_data" / "TrainingDataset_processedV2" / "splits_seasonal"
FIG_DIR   = REPO / "figures"
EXP_DIR   = REPO / "experiments" / "seasonal"
CACHE     = FIG_DIR / "glacier_geometry_per_sample.csv"
ACF_CSV   = REPO.parent / "glacier_data" / "Merged_Seasonality_DecadalRates.csv"

SPLITS  = ("train", "val", "test")
RES_M   = 30.0                                            # metres per pixel
PX_KM2  = RES_M ** 2 / 1e6                                # pixel area in km^2
RGI_PREFIX = "RGI60-07."

# Categorical slots 1-3 of the dataviz reference palette (light surface).
# These three validate all-pairs, which is what violins and scatter need.
COLOR_SPLIT = {"train": "#2a78d6", "val": "#eb6834", "test": "#1baf7a"}
COLOR_SEASONAL     = "#2a78d6"     # blue  — matches plot_skill_per_glacier.py
COLOR_NONSEASONAL  = "#eb6834"     # orange
INK_PRIMARY   = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED     = "#898781"
GRIDLINE      = "#e1e0d9"

# Figures are drawn ~11 in wide and included in the paper at 12 cm; the font
# size below matches the convention already used by plot_skill_per_glacier.py.
FS = 14
plt.rcParams.update({
    "font.size":       FS,
    "axes.titlesize":  FS,
    "axes.labelsize":  FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
})


# ── Geometry scan ─────────────────────────────────────────────────────────────

def _frame_geometry(store: zarr.Group, ti: int) -> tuple[float, float, bool]:
    """(front_length_m, ice_area_km2, trace_valid) for one target frame."""
    sdt  = store["sdt"][ti]
    area = float(np.count_nonzero(sdt > 0) * PX_KM2)      # NaN > 0 is False

    valid = bool(store["valid_trace"][ti])
    length = float(np.count_nonzero(store["trace"][ti]) * RES_M) if valid else np.nan
    return length, area, valid


def scan_splits() -> pd.DataFrame:
    """Per-sample geometry for every split, reading each target frame once."""
    stores: dict[str, zarr.Group] = {}
    frames: dict[tuple[str, int], tuple[float, float, bool]] = {}
    rows = []

    for split in SPLITS:
        samples = json.loads((SPLIT_DIR / f"{split}.json").read_text())
        for s in samples:
            zpath, ti = s["zarr_path"], int(s["target_zarr_index"])
            key = (zpath, ti)
            if key not in frames:
                if zpath not in stores:
                    stores[zpath] = zarr.open_group(str((REPO / zpath).resolve()), mode="r")
                frames[key] = _frame_geometry(stores[zpath], ti)
            length, area, valid = frames[key]
            rows.append({
                "split":             split,
                "glacier_id":        s["glacier_id"],
                "target_zarr_index": ti,
                "front_length_m":    length,
                "ice_area_km2":      area,
                "trace_valid":       valid,
            })
        print(f"[info] {split}: {len(samples)} samples")

    print(f"[info] read {len(frames)} unique target frames from {len(stores)} zarr stores")
    return pd.DataFrame(rows)


def load_geometry(refresh: bool) -> pd.DataFrame:
    if CACHE.exists() and not refresh:
        print(f"[info] reusing cache {CACHE} (--refresh to rescan)")
        return pd.read_csv(CACHE)
    df = scan_splits()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE, index=False)
    print(f"[ok] wrote {CACHE}")
    return df


# ── Part 1: distributions ─────────────────────────────────────────────────────

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """n / min / Q1 / median / Q3 / max / mean per split and variable."""
    rows = []
    for var, label in (("front_length_m", "front length [m]"),
                       ("ice_area_km2",   "domain ice area [km2]")):
        for split in SPLITS:
            v = df.loc[df["split"] == split, var].dropna()
            rows.append({
                "variable":  label,
                "split":     split,
                "n_samples": len(v),
                "n_glaciers": df.loc[(df["split"] == split) & df[var].notna(),
                                     "glacier_id"].nunique(),
                "min":    v.min(),
                "q1":     v.quantile(0.25),
                "median": v.median(),
                "q3":     v.quantile(0.75),
                "max":    v.max(),
                "mean":   v.mean(),
            })
    return pd.DataFrame(rows)


def load_meanacf() -> pd.Series:
    """MeanACF per glacier id. Excel export: ';' separated, ',' decimals, blank tail."""
    pop = pd.read_csv(ACF_CSV, sep=";", decimal=",", index_col=0)
    pop = pop[pop["Glacier"].notna()].drop_duplicates("Glacier")
    return pd.to_numeric(pop.set_index("Glacier")["MeanACF"], errors="coerce")


def test_glacier_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per test glacier: mean/std of front length and domain ice area."""
    t = (df[df["split"] == "test"]
         .groupby("glacier_id")
         .agg(n_samples=("target_zarr_index", "size"),
              front_length_m_mean=("front_length_m", "mean"),
              front_length_m_std=("front_length_m", "std"),
              ice_area_km2_mean=("ice_area_km2", "mean"),
              ice_area_km2_std=("ice_area_km2", "std"))
         .reset_index())
    t["mean_acf"] = t["glacier_id"].map(load_meanacf())
    if t["mean_acf"].isna().any():
        print(f"[warn] no MeanACF for {t.loc[t['mean_acf'].isna(), 'glacier_id'].tolist()}")
    return t


def test_glacier_latex(t: pd.DataFrame, out_tex: Path) -> None:
    """Same table as booktabs LaTeX; full RGI id, '.' decimals, no thousands sep."""
    rows = []
    for r in t.itertuples():
        gid = r.glacier_id.replace("_", "\\_")
        rows.append(f"{gid} & {r.n_samples} & "
                    f"{r.front_length_m_mean:.0f} & {r.front_length_m_std:.0f} & "
                    f"{r.ice_area_km2_mean:.2f} & {r.ice_area_km2_std:.2f} & "
                    f"{r.mean_acf:.2f} \\\\")
    body = "\n".join(rows)
    out_tex.write_text(
        "\\begin{tabular}{llrrrrr}\n\\toprule\n"
        "Glacier (RGI 6.0) & $n$ & \\multicolumn{2}{c}{Front length [m]} & "
        "\\multicolumn{2}{c}{Domain ice area [km$^2$]} & MeanACF \\\\\n"
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}\n"
        " & & mean & std & mean & std & \\\\\n\\midrule\n"
        f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
    print(f"[ok] wrote {out_tex}")


def _decade_ticks(ax, values_log: np.ndarray) -> None:
    """Label a log10-transformed axis with its decades in original units."""
    lo, hi = np.floor(values_log.min()), np.ceil(values_log.max())
    ticks = np.arange(lo, hi + 1)
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{10 ** t:,.0f}" if t >= 0 else f"{10 ** t:g}" for t in ticks])


def _violin_panel(ax, df: pd.DataFrame, var: str, ylabel: str) -> None:
    """Violin + inner box per split, drawn in log10 space with decade labels."""
    data, counts = [], []
    for split in SPLITS:
        v = df.loc[df["split"] == split, var].dropna().to_numpy()
        data.append(np.log10(v[v > 0]))
        counts.append(len(v))

    pos = np.arange(len(SPLITS))
    parts = ax.violinplot(data, positions=pos, widths=0.8,
                          showextrema=False, showmedians=False)
    for body, split in zip(parts["bodies"], SPLITS):
        body.set_facecolor(COLOR_SPLIT[split])
        body.set_edgecolor(COLOR_SPLIT[split])
        body.set_alpha(0.35)
        body.set_linewidth(1.5)

    ax.boxplot(data, positions=pos, widths=0.11, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor=INK_PRIMARY, edgecolor=INK_PRIMARY),
               whiskerprops=dict(color=INK_PRIMARY, linewidth=1.2),
               capprops=dict(color=INK_PRIMARY, linewidth=1.2),
               medianprops=dict(color="white", linewidth=1.6))

    ax.set_xticks(pos)
    ax.set_xticklabels([f"{s}\nn={n:,}" for s, n in zip(SPLITS, counts)])
    ax.set_ylabel(ylabel)
    _decade_ticks(ax, np.concatenate(data))
    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_distributions(df: pd.DataFrame, out_png: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    _violin_panel(axes[0], df, "front_length_m", "Calving front length [m]")
    _violin_panel(axes[1], df, "ice_area_km2",   "Domain ice area [km$^2$]")
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_png}")


# ── Part 2: per-glacier score vs geometry ─────────────────────────────────────

def resolve_experiment(name: str) -> Path:
    """Accept a full path or a bare experiment name under experiments/seasonal/."""
    for cand in (Path(name), REPO / name, EXP_DIR / name):
        if cand.is_dir():
            return cand
    raise FileNotFoundError(f"experiment not found as a path or under {EXP_DIR}: {name}")


def load_scores(exp: Path, eval_subdir: str) -> pd.DataFrame:
    csv = exp / "eval_paper" / eval_subdir / "test_per_glacier.csv"
    if not csv.exists():
        raise FileNotFoundError(f"no per-glacier metrics at {csv}")
    df = pd.read_csv(csv)
    df["mde"]         = df["active_mde_mean"]
    df["skill_persample"] = df["active_skill_score_mean"]
    df["skill_ratio"]     = 1.0 - df["active_mde_mean"] / df["active_persist_mde_mean"]
    print(f"[info] scores from {csv} ({len(df)} glaciers)")
    return df[["glacier_id", "mde", "skill_persample", "skill_ratio"]]


def per_glacier_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Test-split geometry averaged over each glacier's target samples."""
    test = df[df["split"] == "test"]
    return (test.groupby("glacier_id")
                .agg(front_length_m=("front_length_m", "mean"),
                     ice_area_km2=("ice_area_km2", "mean"),
                     n_samples=("target_zarr_index", "size"))
                .reset_index())


def _scatter_panel(ax, d: pd.DataFrame, xvar: str, yvar: str,
                   xlabel: str, ylabel: str) -> None:
    colors = np.where(d["seasonal"], COLOR_SEASONAL, COLOR_NONSEASONAL)
    ax.scatter(d[xvar], d[yvar], s=46, c=colors, zorder=3,
               edgecolors="#fcfcfb", linewidths=0.8)

    for _, r in d.iterrows():
        gid = r["glacier_id"]
        label = gid[len(RGI_PREFIX):] if gid.startswith(RGI_PREFIX) else gid
        ax.annotate(label, xy=(r[xvar], r[yvar]), xytext=(4, 4),
                    textcoords="offset points", fontsize=FS - 5,
                    color=INK_MUTED, zorder=4)

    rho, p_rho = spearmanr(d[xvar], d[yvar])
    ax.annotate(f"Spearman $\\rho$ = {rho:.2f}  (p = {p_rho:.3f})",
                xy=(0.02, 0.97), xycoords="axes fraction",
                va="top", ha="left", fontsize=FS - 2, color=INK_SECONDARY)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(color=GRIDLINE, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def plot_score_vs_geometry(d: pd.DataFrame, out_png: Path) -> None:
    xs = [("front_length_m", "Calving front length [m]"),
          ("ice_area_km2",   "Domain ice area [km$^2$]")]
    ys = [("mde",             "MDE [m]"),
          ("skill_persample", "Skill score\n(per-sample mean)"),
          ("skill_ratio",     "Skill score\n(ratio of means)")]

    fig, axes = plt.subplots(len(ys), len(xs), figsize=(11.0, 12.0))
    for i, (yvar, ylabel) in enumerate(ys):
        for j, (xvar, xlabel) in enumerate(xs):
            _scatter_panel(axes[i, j], d, xvar, yvar, xlabel, ylabel)
            if yvar != "mde":
                axes[i, j].axhline(0.0, color=INK_PRIMARY, linewidth=1.0, zorder=2)

    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          color=COLOR_SEASONAL),
               plt.Line2D([], [], marker="o", linestyle="", markersize=9,
                          color=COLOR_NONSEASONAL)]
    axes[0, 0].legend(handles, ["Seasonal", "Non-seasonal"],
                      loc="upper right", frameon=False, fontsize=FS - 2)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_png}")


def report_correlations(d: pd.DataFrame) -> None:
    print("\nper-glacier correlations (n = %d test glaciers)" % len(d))
    print(f"{'x':16s} {'y':18s} {'spearman':>10s} {'p':>8s} {'pearson':>9s} {'p':>8s}")
    for xvar in ("front_length_m", "ice_area_km2"):
        for yvar in ("mde", "skill_persample", "skill_ratio"):
            rho, p_rho = spearmanr(d[xvar], d[yvar])
            r, p_r     = pearsonr(d[xvar], d[yvar])
            print(f"{xvar:16s} {yvar:18s} {rho:10.3f} {p_rho:8.3f} {r:9.3f} {p_r:8.3f}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", default=None,
                   help="experiment name (under experiments/seasonal/) or path; "
                        "enables the score-vs-geometry figure")
    p.add_argument("--eval-subdir", default="best",
                   help="checkpoint subdirectory inside eval_paper/ (default: best)")
    p.add_argument("--refresh", action="store_true",
                   help="rescan the zarr stores instead of using the cached CSV")
    args = p.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    geom = load_geometry(args.refresh)

    n_bad = int((~geom["trace_valid"]).sum())
    if n_bad:
        by_split = geom.loc[~geom["trace_valid"], "split"].value_counts().to_dict()
        print(f"[warn] {n_bad} sample(s) with valid_trace=False, excluded from "
              f"front length (kept for ice area): {by_split}")

    summary = summary_table(geom)
    print("\n" + summary.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    summary.to_csv(FIG_DIR / "glacier_geometry_summary.csv", index=False)
    print(f"[ok] wrote {FIG_DIR / 'glacier_geometry_summary.csv'}")

    per_glacier = test_glacier_table(geom)
    print("\n" + per_glacier.to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    per_glacier.to_csv(FIG_DIR / "test_glacier_geometry.csv", index=False)
    print(f"[ok] wrote {FIG_DIR / 'test_glacier_geometry.csv'}")
    test_glacier_latex(per_glacier, FIG_DIR / "test_glacier_geometry.tex")

    plot_distributions(geom, FIG_DIR / "glacier_geometry_distributions.png")

    if args.experiment is None:
        return

    exp = resolve_experiment(args.experiment)
    seasonal_map = {s["glacier_id"]: bool(s["seasonal"])
                    for s in json.loads((SPLIT_DIR / "test.json").read_text())}

    d = load_scores(exp, args.eval_subdir).merge(per_glacier_geometry(geom),
                                                 on="glacier_id", how="inner")
    d["seasonal"] = d["glacier_id"].map(seasonal_map)

    n_drop = d[["mde", "skill_persample", "skill_ratio",
                "front_length_m", "ice_area_km2", "seasonal"]].isna().any(axis=1).sum()
    if n_drop:
        print(f"[warn] {n_drop} glacier(s) dropped for missing score/geometry/seasonal flag")
        d = d.dropna(subset=["mde", "skill_persample", "skill_ratio",
                             "front_length_m", "ice_area_km2", "seasonal"])

    report_correlations(d)
    outdir = exp / "eval_paper"
    outdir.mkdir(parents=True, exist_ok=True)
    plot_score_vs_geometry(d, outdir / "score_vs_geometry.png")


if __name__ == "__main__":
    main()
