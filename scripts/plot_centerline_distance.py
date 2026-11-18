"""
Plot per-glacier calving front distance from the centerline inland anchor.

For each glacier: target_date on x-axis, true_distance_m (blue) and
pred_distance_m (orange, if available) on y-axis.  One PNG per glacier.

Usage:
    python scripts/plot_centerline_distance.py \\
        --data   centerline_distance.parquet \\
        --out-dir centerline_distance_plots/
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd


def plot_glacier(
    gdf: pd.DataFrame,
    out_path: Path,
    glacier_id: str,
) -> None:
    gdf = gdf.sort_values("target_date")
    dates = gdf["target_date"]
    true_m = gdf["true_distance_m"]
    pred_m = gdf["pred_distance_m"]
    has_pred = pred_m.notna().any()

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(dates, true_m, marker="o", markersize=3, color="steelblue",
            linewidth=1.2, label="True")
    if has_pred:
        ax.plot(dates, pred_m, marker="s", markersize=3, color="darkorange",
                linewidth=1.2, label="Predicted")

    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_title(glacier_id, fontsize=11)
    ax.set_xlabel("Target date")
    ax.set_ylabel("Distance from inland anchor (m)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate(rotation=30)
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",    type=Path, required=True,
                        help="Parquet produced by centerline_distance.py")
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="Directory for output PNGs")
    parser.add_argument("--splits",  nargs="*", default=None,
                        help="Restrict to these splits (val, test, …); default: all")
    args = parser.parse_args()

    df = pd.read_parquet(args.data)
    df["target_date"] = pd.to_datetime(df["target_date"])

    if args.splits:
        df = df[df["split"].isin(args.splits)]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    glaciers = sorted(df["glacier_id"].unique())
    print(f"Plotting {len(glaciers)} glaciers → {args.out_dir}")

    for gid in glaciers:
        gdf = df[df["glacier_id"] == gid]
        out_path = args.out_dir / f"{gid}.png"
        plot_glacier(gdf, out_path, gid)

    print("Done.")


if __name__ == "__main__":
    main()
