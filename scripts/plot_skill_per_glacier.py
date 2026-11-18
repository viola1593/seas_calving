"""
Per-glacier active-mask MDE skill score vs the persistence baseline.

Skill score per glacier (decision 1 of the grill plan):

    SS = 1 - active_mde_mean / active_persist_mde_mean

SS > 0 beats persistence, SS = 1 is perfect, SS < 0 is worse than persistence.
Uses the per-glacier ``_mean`` columns of ``test_per_glacier.csv`` directly
(NOT the precomputed per-sample ``active_skill_score_mean`` column).

Test glaciers are split into seasonal / non-seasonal (authoritative flag from
the split json) and drawn as two colour-coded blocks on a single axis:
seasonal left, non-seasonal right, alphabetical within each block. A solid
line marks SS = 0 (persistence baseline) and a dashed line per block marks the
group-mean skill.

Usage:
    python scripts/plot_skill_per_glacier.py \\
        --csv        experiments/.../eval_paper/best/test_per_glacier.csv \\
        --split-json ../glacier_data/TrainingDataset_processedV2/splits_seasonal/test.json \\
        [--outdir    experiments/.../eval_paper/best/]

``--outdir`` defaults to the directory containing ``--csv``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Categorical colours from the dataviz reference palette (light surface):
# slot 1 blue / slot 8 orange — the canonical colourblind-safe pair.
COLOR_SEASONAL = "#2a78d6"      # blue
COLOR_NONSEASONAL = "#eb6834"   # orange
RGI_PREFIX = "RGI60-07."
BLOCK_GAP = 1.6                 # empty x-slots separating the two blocks

# The figures are drawn 11 in wide but included in the paper at 12 cm, so
# LaTeX shrinks them by FIG_W_IN * 2.54 / 12. Draw text pre-scaled by the
# inverse so it reads as PT_ON_PAGE on the printed page.
FIG_W_IN, FIG_H_IN = 11.0, 4.5
_SCALE = FIG_W_IN * 2.54 / 12.0   # ~2.33
PT_ON_PAGE = 11.0
FS = 14# PT_ON_PAGE * _SCALE          # ~25.6 pt in figure units

plt.rcParams.update({
    "font.size": FS,
    "axes.titlesize": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
})


def build_seasonal_map(split_json: Path) -> dict[str, bool]:
    """{glacier_id: seasonal} from the split json (one entry per glacier)."""
    samples = json.loads(split_json.read_text())
    return {s["glacier_id"]: bool(s["seasonal"]) for s in samples}


def load_skill(csv: Path, seasonal_map: dict[str, bool]) -> pd.DataFrame:
    """Per-glacier skill score joined to its seasonal flag; NaN dropped."""
    df = pd.read_csv(csv)
    df["skill"] = 1.0 - df["active_mde_mean"] / df["active_persist_mde_mean"]
    df["skill_mean"] = df["active_skill_score_mean"]
    df["skill_std"] = df["active_skill_score_std"]
    df["seasonal"] = df["glacier_id"].map(seasonal_map)

    missing = df[df["seasonal"].isna()]["glacier_id"].tolist()
    if missing:
        print(
            f"[warn] {len(missing)} glacier(s) absent from split json, dropped: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        df = df[df["seasonal"].notna()].copy()

    nan_ids = df[df["skill"].isna()]["glacier_id"].tolist()
    if nan_ids:
        print(
            f"[warn] {len(nan_ids)} glacier(s) with NaN skill score, dropped: "
            + ", ".join(nan_ids),
            file=sys.stderr,
        )
        df = df[df["skill"].notna()].copy()

    return df

def load_mde(csv: Path, seasonal_map: dict[str, bool]) -> pd.DataFrame:
    """Per-glacier MDE joined to its seasonal flag; NaN dropped."""
    df = pd.read_csv(csv)
    df["mde"] =  df["active_mde_mean"]
    df["mde_std"] = df["active_mde_std"]
    df["mde_persist"] = df["active_persist_mde_mean"]
    df["seasonal"] = df["glacier_id"].map(seasonal_map)

    missing = df[df["seasonal"].isna()]["glacier_id"].tolist()
    if missing:
        print(
            f"[warn] {len(missing)} glacier(s) absent from split json, dropped: "
            + ", ".join(missing),
            file=sys.stderr,
        )
        df = df[df["seasonal"].notna()].copy()

    nan_ids = df[df["mde"].isna()]["glacier_id"].tolist()
    if nan_ids:
        print(
            f"[warn] {len(nan_ids)} glacier(s) with NaN MDE, dropped: "
            + ", ".join(nan_ids),
            file=sys.stderr,
        )
        df = df[df["mde"].notna()].copy()

    return df


def plot_skill(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    seasonal = df[df["seasonal"]].sort_values("glacier_id")
    nonseasonal = df[~df["seasonal"]].sort_values("glacier_id")

    n_s = len(seasonal)
    x_s = np.arange(n_s)
    x_ns = np.arange(len(nonseasonal)) + n_s + BLOCK_GAP
    x_all = np.concatenate([x_s, x_ns])
    # skills = np.concatenate([seasonal["skill"].to_numpy(),
    #                          nonseasonal["skill"].to_numpy()])
    skills = np.concatenate([seasonal["skill"].to_numpy(),
                             nonseasonal["skill"].to_numpy()])
    # stds = np.concatenate([seasonal["skill_std"].to_numpy(),
    #                        nonseasonal["skill_std"].to_numpy()])
    # large stds can draw bar below zero which is possibly happening
    # y_err = np.vstack([stds, stds])
    labels = [g[len(RGI_PREFIX):] if g.startswith(RGI_PREFIX) else g
              for g in list(seasonal["glacier_id"]) + list(nonseasonal["glacier_id"])]
    colors = [COLOR_SEASONAL] * n_s + [COLOR_NONSEASONAL] * len(nonseasonal)


    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.bar(x_all, skills, width=0.8, color=colors, zorder=3, 
    # yerr=y_err, 
    error_kw=dict(ecolor="#52514e", elinewidth=0.9,
                                    capsize=2, zorder=5))

    # SS = 0 persistence baseline.
    ax.axhline(0.0, color="black", linewidth=1.2, zorder=2) ##52514e

    # Per-block dashed mean line + numeric annotation.
    for xs, block, color in ((x_s, seasonal, COLOR_SEASONAL),
                             (x_ns, nonseasonal, COLOR_NONSEASONAL)):
        if len(xs) == 0:
            continue
        m = block["skill_mean"].mean()
        x0, x1 = xs[0] - 0.5, xs[-1] + 0.5
        ax.plot([x0, x1], [m, m], linestyle="--", color=color,
                linewidth=1.6, zorder=4)
        ax.annotate(f"mean {m:.3f}", xy=(x1, m+0.05),
                    xytext=(0, 4), textcoords="offset points",
                    va="center", ha="left", fontsize=FS, color=color,
                    fontweight="bold", 
                    rotation=45
                    )

    ax.set_xticks(x_all)
    ax.set_xticklabels(labels, rotation=90, fontsize=FS)
    ax.set_ylabel("Sample-wise Skill score")
    # ax.set_title("Per-glacier active-mask skill vs persistence", fontsize=FS)
    ax.margins(x=0.01)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Block identity: legend + labels above each block (not colour-alone).
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_SEASONAL),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_NONSEASONAL)]
    ax.legend(handles, ["Seasonal", "Non-seasonal"],
              loc="upper right", frameon=False, fontsize=FS)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    # fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_pdf}")

def plot_mde(df: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    seasonal = df[df["seasonal"]].sort_values("glacier_id")
    nonseasonal = df[~df["seasonal"]].sort_values("glacier_id")

    n_s = len(seasonal)
    x_s = np.arange(n_s)
    x_ns = np.arange(len(nonseasonal)) + n_s + BLOCK_GAP
    x_all = np.concatenate([x_s, x_ns])
    mdes = np.concatenate([seasonal["mde"].to_numpy(),
                             nonseasonal["mde"].to_numpy()])
    mdes_persist = np.concatenate([seasonal["mde_persist"].to_numpy(),
                                    nonseasonal["mde_persist"].to_numpy()])
    stds = np.concatenate([seasonal["mde_std"].to_numpy(),
                           nonseasonal["mde_std"].to_numpy()])
    labels = [g[len(RGI_PREFIX):] if g.startswith(RGI_PREFIX) else g
              for g in list(seasonal["glacier_id"]) + list(nonseasonal["glacier_id"])]
    colors = [COLOR_SEASONAL] * n_s + [COLOR_NONSEASONAL] * len(nonseasonal)

    # MDE >= 0, so clip the lower whisker at the axis floor rather than
    # letting a large std draw the bar below zero. NaN std (n_samples == 1)
    # is drawn as no whisker.
    stds = np.nan_to_num(stds, nan=0.0)
    yerr = np.vstack([np.minimum(stds, mdes), stds])

    fig, ax = plt.subplots(figsize=(FIG_W_IN, FIG_H_IN))
    ax.bar(x_all, mdes_persist, width=0.8, color="#e1e0d9", zorder=1)
    ax.bar(x_all, mdes, width=0.8, color=colors, zorder=3,
           yerr=yerr, error_kw=dict(ecolor="#52514e", elinewidth=0.9,
                                    capsize=2, zorder=5))

    # SS = 0 persistence baseline.
    ax.axhline(0.0, color="#52514e", linewidth=1.2, zorder=2)

    # Per-block dashed mean line + numeric annotation.
    for xs, block, color in ((x_s, seasonal, COLOR_SEASONAL),
                             (x_ns, nonseasonal, COLOR_NONSEASONAL)):
        if len(xs) == 0:
            continue
        m = block["mde"].mean()
        m_persist = block["mde_persist"].mean()
        x0, x1 = xs[0] - 0.5, xs[-1] + 0.5
        ax.plot([x0, x1], [m_persist, m_persist], linestyle="--", color="#e1e0d9",
                linewidth=1.6, zorder=4)
        ax.annotate(f"mean {m_persist:.1f}", xy=(x1, m_persist+30),
                    xytext=(0, 4), textcoords="offset points",
                    va="center", ha="left", fontsize=FS, color="gray",
                    # fontweight="bold",
                    rotation=45
                    )
        ax.plot([x0, x1], [m, m], linestyle="--", color= color,
                linewidth=1.6, zorder=4)
        ax.annotate(f"mean {m:.1f}", xy=(x1, m+20),
                    xytext=(0, 4), textcoords="offset points",
                    va="center", ha="left", fontsize=FS, color=color,
                    fontweight="bold", 
                    rotation=45
                    )

    ax.set_xticks(x_all)
    ax.set_xticklabels(labels, rotation=90, fontsize=FS)
    ax.set_ylabel("MDE [m]")
    # ax.set_title("Per-glacier MDE (±1 std across test samples)", fontsize=FS)
    ax.margins(x=0.01)
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Block identity: legend + labels above each block (not colour-alone).
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR_SEASONAL),
               plt.Rectangle((0, 0), 1, 1, color=COLOR_NONSEASONAL), 
               plt.Rectangle((0, 0), 1, 1, color="#e1e0d9")]
    ax.legend(handles, ["Seasonal", "Non-seasonal", "Persistence"],
              loc="upper right", frameon=False, fontsize=FS)

    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    # fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote {out_png}")
    print(f"[ok] wrote {out_pdf}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", type=Path, required=True,
                   help="test_per_glacier.csv from an eval_paper run")
    p.add_argument("--split-json", type=Path, required=True,
                   help="splits_seasonal/test.json (per-glacier seasonal flag)")
    p.add_argument("--outdir", type=Path, default=None,
                   help="output directory for skill_per_glacier.{png,pdf} "
                        "(default: the directory of --csv)")
    args = p.parse_args()

    outdir = args.outdir if args.outdir is not None else args.csv.parent

    seasonal_map = build_seasonal_map(args.split_json)
    df = load_skill(args.csv, seasonal_map)
    df2 = load_mde(args.csv, seasonal_map)

    n_s = int(df["seasonal"].sum())
    print(f"[info] {len(df)} glaciers: {n_s} seasonal, {len(df) - n_s} non-seasonal")

    outdir.mkdir(parents=True, exist_ok=True)
    plot_skill(df, outdir / "skill_per_glacier.png",
         outdir / "skill_per_glacier.pdf")
    plot_mde(df2, outdir / "mde_per_glacier.png",
         outdir / "mde_per_glacier.pdf")


if __name__ == "__main__":
    main()