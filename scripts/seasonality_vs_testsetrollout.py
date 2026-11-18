"""
Seasonality (MeanACF) of the 22 test glaciers, and how it relates to rollout skill.

MeanACF is the expert autocorrelation score behind the binary `seasonal?` label
(the same label fed to the model as `glacier_seasonality`). This script pulls it
for the test split, puts it in context against the full Svalbard population, and
— if the rollout CSV is present — joins it against per-glacier rollout behaviour
so we can see whether seasonality separates the glaciers whose rollout tracks
(e.g. 00240) from the ones that freeze (e.g. 00318, 00993, 00061).

The source CSV is an Excel export: ';' separated, ',' as decimal mark, and
~1e6 trailing blank rows. Glacier IDs there are BASE RGI ids, while the training
split uses basin suffixes (RGI60-07.00030_D), so matching falls back to the base
id and flags which test glaciers had to be matched that way.

With --arcs it also reports, per glacier, after how many steps the centerline
stops moving at all (freeze_step / freeze_days) and whether it ever recovers
(escaped) — see freeze_onset.

Usage:
    python scripts/seasonality_vs_testset.py

    EXP=experiments/seasonal/paper_M3_seed3
    python scripts/seasonality_vs_testset.py \
        --rollout $EXP/rollout/test_rollout.csv \
        --arcs    $EXP/rollout/centerlines/test_centerline_arcs.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

CSV = Path(__file__).resolve().parents[2] / "glacier_data" / "Merged_Seasonality_DecadalRates.csv"

RGI_IDS = [
    "RGI60-07.00030_D",
    "RGI60-07.00035_B",
    "RGI60-07.00060",
    "RGI60-07.00061",
    "RGI60-07.00062_C",
    "RGI60-07.00064",
    "RGI60-07.00240",
    "RGI60-07.00250",
    "RGI60-07.00315",
    "RGI60-07.00318",
    "RGI60-07.00424_A",
    "RGI60-07.00470",
    "RGI60-07.00502",
    "RGI60-07.00524",
    "RGI60-07.00681",
    "RGI60-07.00727",
    "RGI60-07.00898",
    "RGI60-07.00993",
    "RGI60-07.01326",
    "RGI60-07.01353",
    "RGI60-07.01354_2",
    "RGI60-07.01493",
]

KEEP = ["Glacier", "MeanACF",   "seasonal_range",
        "Velocity",  "TT_Seasonal", "votemper_seasonal"]


def base_id(gid: str) -> str:
    """RGI60-07.00030_D -> RGI60-07.00030 (basin suffix stripped)."""
    return gid.split("_")[0]


def load_population(csv: Path) -> pd.DataFrame:
    df = pd.read_csv(csv, sep=";", decimal=",", index_col=0)
    df = df[df["Glacier"].notna()].copy()          # drop the Excel blank tail
    df["base"] = df["Glacier"].map(base_id)
    return df


def freeze_onset(dates: pd.Series, arc: pd.Series,
                 tol_m: float = 1.0, min_run: int = 3) -> dict:
    """When does the centerline stop moving, and does it ever start again?

    A step is "still" when the front's centerline arc changes by less than
    tol_m from the previous step. The freeze onset is the start of the first
    run of at least min_run consecutive still steps.

    tol_m defaults to 1 m — far below the 30 m pixel — because a truly frozen
    rollout reproduces a bit-identical field, so the arcs are exactly equal.
    Raising it to ~15 m (half a pixel) instead measures "moves less than the
    grid can represent", which is the redistancing dead-band rather than a
    hard freeze.

    Returns:
        freeze_step   0-based step at which motion ceases (NaN if never)
        freeze_days   days from the first target_date to that step
        frozen_frac   fraction of all steps with no movement
        escaped       True if the front moves again after the onset
        total_range_m peak-to-peak of the whole arc series, as a sanity check
    """
    a = pd.to_numeric(arc, errors="coerce").to_numpy(dtype=float)
    d = pd.to_datetime(dates).to_numpy()
    ok = np.isfinite(a)
    a, d = a[ok], d[ok]

    out = {"freeze_step": np.nan, "freeze_days": np.nan, "frozen_frac": np.nan,
           "escaped": False, "total_range_m": np.nan}
    if len(a) < min_run + 1:
        return out

    still = np.abs(np.diff(a)) < tol_m           # still[i]: arc[i] -> arc[i+1]
    out["frozen_frac"] = float(still.mean())
    out["total_range_m"] = float(np.nanmax(a) - np.nanmin(a))

    # First index where min_run consecutive transitions are all still.
    run = 0
    for i, s in enumerate(still):
        run = run + 1 if s else 0
        if run >= min_run:
            start = i - min_run + 1
            out["freeze_step"] = int(start)
            out["freeze_days"] = float((d[start] - d[0]) / np.timedelta64(1, "D"))
            out["escaped"] = bool((~still[start:]).any())
            break
    return out


def load_arcs(path: Path, tol_m: float, min_run: int) -> pd.DataFrame:
    """Per-glacier freeze diagnostics from the centerline arc series."""
    df = pd.read_csv(path, parse_dates=["target_date"])
    rows = []
    for gid, g in df.groupby("glacier_id"):
        g = g.sort_values("target_date")
        model = freeze_onset(g["target_date"], g["model_arc_m"], tol_m, min_run)
        truth = freeze_onset(g["target_date"], g["true_arc_m"], tol_m, min_run)
        # Depth at the moment the model freezes — does it coincide with depth==K?
        depth_at = np.nan
        if np.isfinite(model["freeze_step"]):
            depth_at = float(g["autoreg_depth"].to_numpy()[int(model["freeze_step"])])
        rows.append({
            "glacier_id":     gid,
            "freeze_step":    model["freeze_step"],
            "freeze_days":    model["freeze_days"],
            "frozen_frac":    model["frozen_frac"],
            "escaped":        model["escaped"],
            "depth_at_freeze": depth_at,
            "model_range_m":  model["total_range_m"],
            "true_range_m":   truth["total_range_m"],
            # <<1 means the rollout under-moves relative to the real front.
            "range_ratio":    (model["total_range_m"] / truth["total_range_m"]
                               if truth["total_range_m"] else np.nan),
        })
    return pd.DataFrame(rows)


def load_rollout(path: Path) -> pd.DataFrame:
    """Per-glacier rollout summary: how deep it got and how much it cost."""
    r = pd.read_csv(path, parse_dates=["target_date"])
    out = []
    for gid, g in r.groupby("glacier_id"):
        depth = g["autoreg_depth"]
        deep = g[depth == depth.max()]
        out.append({
            "glacier_id":      gid,
            "n_steps":         len(g),
            "max_depth":       int(depth.max()),
            "frac_at_max":     float((depth == depth.max()).mean()),
            "model_mde":       float(deep["model_active_mde"].mean()),
            "teacher_mde":     float(deep["teacher_active_mde"].mean()),
            "persist_mde":     float(deep["persist_active_mde"].mean()),
        })
    d = pd.DataFrame(out)
    # >1 means the rollout is worse than its own teacher-forced upper bound.
    d["rollout_penalty"] = d["model_mde"] / d["teacher_mde"]
    # <0 means the rollout is worse than simply freezing the front at t0.
    d["skill_vs_persist"] = 1.0 - d["model_mde"] / d["persist_mde"]
    return d


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument("--csv", type=Path, default=CSV)
    p.add_argument("--rollout", type=Path, default=None,
                   help="Optional test_rollout.csv to join against.")
    p.add_argument("--arcs", type=Path, default=None,
                   help="Optional test_centerline_arcs.csv (written by "
                        "rollout_centerline.py) — adds the freeze-onset columns.")
    p.add_argument("--freeze-tol-m", type=float, default=1.0,
                   help="Arc change below this counts as no movement (default 1 m; "
                        "use ~15 for the half-pixel dead-band instead of a hard freeze).")
    p.add_argument("--freeze-min-run", type=int, default=3,
                   help="Consecutive still steps required to call it frozen.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args(argv)

    pop = load_population(args.csv)

    test = pd.DataFrame({"glacier_id": RGI_IDS})
    test["base"] = test["glacier_id"].map(base_id)
    test["suffixed"] = test["glacier_id"] != test["base"]

    # Prefer an exact hit on the suffixed id; fall back to the base id.
    exact = pop.set_index("Glacier")
    by_base = pop.drop_duplicates("base").set_index("base")

    rows = []
    for _, t in test.iterrows():
        if t["glacier_id"] in exact.index:
            src, how = exact.loc[t["glacier_id"]], "exact"
        elif t["base"] in by_base.index:
            src, how = by_base.loc[t["base"]], "base"
        else:
            rows.append({"glacier_id": t["glacier_id"], "matched": "MISSING"})
            continue
        rows.append({"glacier_id": t["glacier_id"], "matched": how,
                     **{k: src[k] for k in KEEP if k in src.index}})

    df = pd.DataFrame(rows)

    if args.rollout is not None:
        df = df.merge(load_rollout(args.rollout), on="glacier_id", how="left")
    if args.arcs is not None:
        df = df.merge(load_arcs(args.arcs, args.freeze_tol_m, args.freeze_min_run),
                      on="glacier_id", how="left")

    df = df.sort_values("MeanACF", ascending=False, na_position="last")

    show = [c for c in ["glacier_id", "matched", "MeanACF", "seasonal?", "Type",
                        "seasonal_range", "Velocity", "Annual_CFL_Rates",
                        "max_depth", "frac_at_max", "rollout_penalty",
                        "frozen_frac",
                        "freeze_step", "freeze_days", "depth_at_freeze",
                        "escaped", "range_ratio"] if c in df.columns]
    with pd.option_context("display.width", 220, "display.max_columns", 30):
        print(df[show].to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    missing = df[df["matched"] == "MISSING"]["glacier_id"].tolist()
    if missing:
        print(f"\nNOT FOUND in {args.csv.name}: {missing}")
    based = df[df["matched"] == "base"]["glacier_id"].tolist()
    if based:
        print(f"\nMatched by base id (suffix dropped — MeanACF is the WHOLE-glacier "
              f"value, not this basin's): {based}")

    # Test split vs the full population.
    print("\n── MeanACF: test split vs all Svalbard ──")
    for name, s in (("test", df["MeanACF"]), ("population", pop["MeanACF"])):
        s = pd.to_numeric(s, errors="coerce").dropna()
        print(f"  {name:11s} n={len(s):3d}  mean={s.mean():.3f}  median={s.median():.3f}  "
              f"min={s.min():.3f}  max={s.max():.3f}")
    if "seasonal?" in df.columns:
        print(f"  test seasonal?=TRUE : "
              f"{(df['seasonal?'].astype(str).str.upper() == 'TRUE').sum()}/{len(df)}")

    if args.arcs is not None:
        froze = df["freeze_step"].notna()
        print(f"\n── Front freezes (arc still for ≥{args.freeze_min_run} steps, "
              f"tol {args.freeze_tol_m:g} m) ──")
        print(f"  froze at some point : {froze.sum()}/{len(df)}")
        if froze.any():
            f = df[froze]
            print(f"  median freeze_step  : {f['freeze_step'].median():.0f} steps "
                  f"({f['freeze_days'].median():.0f} days into the rollout)")
            if "max_depth" in f.columns:
                at_full = int((f["depth_at_freeze"] == f["max_depth"]).sum())
                print(f"  froze at full depth : {at_full}/{len(f)}")
            print(f"  later escaped       : {int(f['escaped'].sum())}/{len(f)}")
        never = df.loc[~froze, "glacier_id"].tolist()
        if never:
            print(f"  never froze         : {never}")

    if args.rollout is not None or args.arcs is not None:
        print("\n── Does seasonality predict rollout behaviour? ──")
        num = df.copy()
        num["MeanACF"] = pd.to_numeric(num["MeanACF"], errors="coerce")
        for target in ("rollout_penalty", "skill_vs_persist", "frac_at_max",
                       "freeze_step", "frozen_frac", "range_ratio"):
            if target not in num.columns:
                continue
            sub = num[["MeanACF", target]].dropna()
            if len(sub) > 3:
                r = sub["MeanACF"].corr(sub[target])
                print(f"  corr(MeanACF, {target:16s}) = {r:+.3f}   (n={len(sub)})")
        print("\n  NOTE: n is at most 22 and the suffixed glaciers carry whole-glacier")
        print("  MeanACF values, so these correlations are indicative only — read the")
        print("  per-glacier table above rather than the coefficient. corr with")
        print("  freeze_step also drops every glacier that never froze, which is")
        print("  exactly the interesting group — check `never froze` above.")

    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
