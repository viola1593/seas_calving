"""
Is per-glacier MDE related to how seasonal the glacier is (MeanACF)?

Joins the per-glacier evaluation table(s) written by evaluate_film.py
(test_per_glacier.csv) against MeanACF from Merged_Seasonality_DecadalRates.csv
and reports Pearson / Spearman correlations.

Several per-glacier CSVs may be passed (e.g. the five paper seeds); the metric
is then averaged per glacier across seeds before correlating, which is the
right unit here — one point per glacier, not one per seed.

The source CSV is an Excel export: ';' separated, ',' as decimal mark, and
~1e6 trailing blank rows. Its glacier IDs are BASE RGI ids while the split uses
basin suffixes (RGI60-07.00030_D), so those fall back to the base id and carry
the WHOLE-glacier MeanACF. The correlation is therefore also reported over the
exactly-matched subset only.

Usage:
    python scripts/mde_vs_meanacf.py \
        experiments/seasonal/paper_M3_seed3/eval_paper/best/test_per_glacier.csv

    # averaged over the five M3 seeds, with scatter plot and merged table
    python scripts/mde_vs_meanacf.py \
        experiments/seasonal/paper_M3_seed*/eval_paper/best/test_per_glacier.csv \
        --plot figures/mde_vs_meanacf.png --out figures/mde_vs_meanacf.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

CSV = Path(__file__).resolve().parents[2] / "glacier_data" / "Merged_Seasonality_DecadalRates.csv"

DEFAULT_METRICS = ["active_mde_mean", "active_mde_median", "active_persist_mde_mean"
                   "active_skill_score_mean", "active_skill_score_median"]

# Metrics that are not in metres — only used for the plot's axis label.
UNITLESS = ("skill_score", "iou", "front_iou")


def base_id(gid: str) -> str:
    """RGI60-07.00030_D -> RGI60-07.00030 (basin suffix stripped)."""
    return gid.split("_")[0]


def load_population(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, sep=";", decimal=",", index_col=0)
    df = df[df["Glacier"].notna()].copy()          # drop the Excel blank tail
    df["base"] = df["Glacier"].map(base_id)
    return df


def load_metrics(paths: list[Path], metrics: list[str]) -> pd.DataFrame:
    """Per-glacier metrics, averaged across the given per-glacier CSVs (seeds)."""
    frames = []
    for p in paths:
        d = pd.read_csv(p)
        missing = [m for m in metrics if m not in d.columns]
        if missing:
            raise SystemExit(f"{p}: missing column(s) {missing}")
        frames.append(d[["glacier_id", "n_samples", *metrics]])

    df = pd.concat(frames, ignore_index=True)
    agg = df.groupby("glacier_id").agg(
        n_seeds=("n_samples", "size"),
        n_samples=("n_samples", "mean"),
        **{m: (m, "mean") for m in metrics},
        **{f"{m}_seed_std": (m, "std") for m in metrics},
    ).reset_index()
    return agg


def attach_meanacf(df: pd.DataFrame, pop: pd.DataFrame) -> pd.DataFrame:
    """Add MeanACF (+ context columns), preferring an exact id hit over the base id."""
    keep = [c for c in ["MeanACF", "seasonal?", "Type", "Sector", "seasonal_range",
                        "Velocity", "Annual_CFL_Rates"] if c in pop.columns]
    exact = pop.set_index("Glacier")
    by_base = pop.drop_duplicates("base").set_index("base")

    rows = []
    for gid in df["glacier_id"]:
        if gid in exact.index:
            src, how = exact.loc[gid], "exact"
        elif base_id(gid) in by_base.index:
            src, how = by_base.loc[base_id(gid)], "base"
        else:
            rows.append({"glacier_id": gid, "matched": "MISSING"})
            continue
        rows.append({"glacier_id": gid, "matched": how, **{k: src[k] for k in keep}})

    out = df.merge(pd.DataFrame(rows), on="glacier_id", how="left")
    out["MeanACF"] = pd.to_numeric(out.get("MeanACF"), errors="coerce")
    return out


def corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float, int]:
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 3:
        return np.nan, np.nan, np.nan, np.nan, len(x)
    pr, pp = pearsonr(x, y)
    sr, sp = spearmanr(x, y)
    return float(pr), float(pp), float(sr), float(sp), len(x)


def report(df: pd.DataFrame, metrics: list[str]) -> None:
    print(f"\n{'='*84}")
    print("  corr(MeanACF, metric)   —   one point per glacier")
    print(f"{'='*84}")
    print(f"  {'metric':<22} {'subset':<10} {'n':>3}  {'Pearson r':>10} {'p':>9}  "
          f"{'Spearman ρ':>11} {'p':>9}")
    print(f"  {'─'*22} {'─'*10} {'─'*3}  {'─'*10} {'─'*9}  {'─'*11} {'─'*9}")

    subsets = [("all", df), ("exact-id", df[df["matched"] == "exact"])]
    for m in metrics:
        for label, sub in subsets:
            pr, pp, sr, sp, n = corr(sub["MeanACF"].to_numpy(dtype=float),
                                     sub[m].to_numpy(dtype=float))
            print(f"  {m:<22} {label:<10} {n:>3}  {pr:>10.3f} {pp:>9.3f}  "
                  f"{sr:>11.3f} {sp:>9.3f}")


def plot(df: pd.DataFrame, metric: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sub = df[["MeanACF", metric, "glacier_id", "matched", "n_samples"]].dropna(
        subset=["MeanACF", metric])
    x = sub["MeanACF"].to_numpy(dtype=float)
    y = sub[metric].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7, 5))
    exact = (sub["matched"] == "exact").to_numpy()
    ax.scatter(x[exact], y[exact], s=sub["n_samples"].to_numpy()[exact],
               alpha=0.75, edgecolor="k", linewidth=0.5, label="exact id match")
    ax.scatter(x[~exact], y[~exact], s=sub["n_samples"].to_numpy()[~exact],
               alpha=0.75, edgecolor="k", linewidth=0.5, marker="s",
               label="base id (whole-glacier MeanACF)")

    if len(x) > 2:
        b, a = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 50)
        ax.plot(xs, a + b * xs, "k--", lw=1)
        pr, pp, sr, sp, n = corr(x, y)
        ax.set_title(f"r={pr:+.2f} (p={pp:.3f}), ρ={sr:+.2f} (p={sp:.3f}), n={n}")

    for _, r in sub.iterrows():
        ax.annotate(r["glacier_id"].replace("RGI60-07.", ""),
                    (r["MeanACF"], r[metric]), fontsize=6,
                    xytext=(3, 3), textcoords="offset points")

    ax.set_xlabel("MeanACF (front-position seasonality)")
    unit = "" if any(u in metric for u in UNITLESS) else "  [m]"
    ax.set_ylabel(f"{metric}{unit}")
    ax.legend(fontsize=8, markerscale=0.4)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200)
    print(f"\nWrote {path}")


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("per_glacier", type=Path, nargs="+",
                   help="One or more test_per_glacier.csv (multiple = seeds, averaged).")
    p.add_argument("--csv", type=Path, default=CSV, help="Seasonality source CSV.")
    p.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS,
                   help=f"Metric columns to correlate (default: {' '.join(DEFAULT_METRICS)}).")
    p.add_argument("--min-samples", type=int, default=0,
                   help="Drop glaciers with fewer than this many test samples.")
    p.add_argument("--plot", type=Path, default=None, help="Write a scatter PNG here.")
    p.add_argument("--out", type=Path, default=None, help="Write the merged table here.")
    args = p.parse_args(argv)

    df = load_metrics(args.per_glacier, args.metrics)
    df = attach_meanacf(df, load_population(args.csv))

    if args.min_samples:
        dropped = df[df["n_samples"] < args.min_samples]["glacier_id"].tolist()
        df = df[df["n_samples"] >= args.min_samples]
        if dropped:
            print(f"Dropped (n_samples < {args.min_samples}): {dropped}")

    df = df.sort_values("MeanACF", ascending=False, na_position="last")

    show = [c for c in ["glacier_id", "matched", "MeanACF", "seasonal?", "Type",
                        "n_seeds", "n_samples", *args.metrics,
                        *[f"{m}_seed_std" for m in args.metrics]] if c in df.columns]
    print(f"Per-glacier metrics from {len(args.per_glacier)} file(s):")
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    missing = df[df["matched"] == "MISSING"]["glacier_id"].tolist()
    if missing:
        print(f"\nNOT FOUND in {args.csv.name}: {missing}")
    based = df[df["matched"] == "base"]["glacier_id"].tolist()
    if based:
        print(f"\nMatched by base id (MeanACF is the whole-glacier value, not this "
              f"basin's): {based}")

    report(df, args.metrics)
    print("\n  NOTE: n ≤ 22 test glaciers, so a single glacier can move r a lot —")
    print("  read the table alongside the coefficient.")

    if args.plot:
        plot(df, args.metrics[0], args.plot)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
