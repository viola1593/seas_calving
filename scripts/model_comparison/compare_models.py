#!/usr/bin/env python3
"""
Compare multi-seed model runs on the test set, overall and stratified.

Reads eval_paper/best/test_results.csv of every model x seed, checks that all
runs share identical samples and persistence baselines, then reports:

  1. Per-stratum 5-seed mean +- std of MAE, MDE, Hausdorff, advance/retreat bias and skill,
     where skill = 1 - mean(MDE_model) / mean(MDE_persistence) within the stratum.
     Strata: all, fast/slow mover, glacier seasonality, seasonality x mover,
     target season.
  2. Paired differences (model A - model B) on seed-averaged per-sample errors,
     with a glacier-level bootstrap 95% CI (captures sample/glacier variability,
     NOT training-seed variability).
  3. Seed-level Welch t-test on the stratum means (captures seed variability).

"Fast" = mean_true_displacement_m >= threshold (default 100 m, matches
aggregate_metrics in metrics.py).

Run from: svalcalvingdb/
    python scripts/model_comparison/compare_models.py
    python scripts/model_comparison/compare_models.py \
        --model M3=experiments/seasonal/paper_M3_seed{seed} \
        --model M3nf=experiments/seasonal/noforc/paper_M3nf_seed{seed} \
        --compare M3nf-M3 --out-dir experiments/seasonal/noforc/comparison
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_MODELS = [
    "M1=experiments/seasonal/paper_M1_seed{seed}",
    "M3=experiments/seasonal/paper_M3_seed{seed}",
    "M3nf=experiments/seasonal/noforc/paper_M3nf_seed{seed}",
]
DEFAULT_COMPARE = ["M3nf-M3", "M1-M3", "M1-M3nf"]
REFERENCE = "M3"  # reference model of the Markdown / LaTeX comparison tables
TEX_NAMES = {"M3nf": "M2"}  # model names as shown in the LaTeX table
TEX_METRICS = {"MDE": "MDE", "MAE": "MAE", "HD": "Hausdorff", "Bias": "Bias"}
EVAL_SUBDIR = "eval_paper/best/test_results.csv"
KEY = ["glacier_id", "target_date", "horizon_days"]
METRICS = {"MAE": "mae_front", "MDE": "active_mde", "HD": "active_hausdorff", "Bias": "advance_retreat_bias"}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", action="append", metavar="NAME=PATTERN",
                   help="Run dir pattern with {seed} placeholder; repeatable. Default: M1, M3, M3nf.")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--compare", nargs="+", metavar="A-B",
                   help="Paired comparisons A-B (positive = A worse). Default: all pairs of the defaults.")
    p.add_argument("--threshold", type=float, default=100.0, help="Fast-mover displacement threshold [m].")
    p.add_argument("--out-dir", type=Path, help="If given, write the tables as CSV plus summary_tables.md here.")
    return p.parse_args()


def load_runs(models, seeds, eval_subdir):
    runs = {}
    for name, pattern in models.items():
        for s in seeds:
            path = Path(pattern.format(seed=s)) / eval_subdir
            r = pd.read_csv(path)
            r["model"], r["seed"] = name, s
            runs[(name, s)] = r
    ref_key = next(iter(runs))
    ref = runs[ref_key]
    for k, r in runs.items():
        if len(r) != len(ref) or not (r[KEY].values == ref[KEY].values).all():
            raise ValueError(f"{k}: samples differ from {ref_key}")
        if not np.allclose(r.active_persist_mde, ref.active_persist_mde, equal_nan=True):
            raise ValueError(f"{k}: persistence MDE differs from {ref_key}")
    return pd.concat(runs.values(), ignore_index=True)


def add_strata(df, threshold):
    mover = np.where(df.mean_true_displacement_m >= threshold, "fast", "slow")
    return {
        "all": pd.Series("all", index=df.index),
        "mover": pd.Series(mover, index=df.index),
        "seasonality": df.glacier_seasonality,
        "seasonality_x_mover": df.glacier_seasonality + "/" + mover,
        "target_season": df.season,
    }


def seed_level(df, strata):
    """Per (stratum, model, seed) aggregates."""
    rows = []
    for sname, st in strata.items():
        for (sval, m, s), g in df.groupby([st, df.model, df.seed]):
            row = dict(stratification=sname, stratum=sval, model=m, seed=s, n=len(g),
                       persMDE=g.active_persist_mde.mean())
            for k, c in METRICS.items():
                row[k] = g[c].mean()
            row["skill"] = 1 - g.active_mde.mean() / g.active_persist_mde.mean()
            rows.append(row)
    return pd.DataFrame(rows)


def summarise(per_seed):
    cols = list(METRICS) + ["skill"]
    g = per_seed.groupby(["stratification", "stratum", "model"], sort=False)
    out = g[cols].agg(["mean", "std"])
    out.columns = [f"{a}" if b == "mean" else f"{a}_sd" for a, b in out.columns]
    out.insert(0, "n", g.n.first())
    out.insert(1, "n_seeds", g.seed.nunique())
    out["persMDE"] = g.persMDE.first()
    return out.reset_index()


def welch(per_seed, comparisons):
    rows = []
    for (sname, sval), g in per_seed.groupby(["stratification", "stratum"], sort=False):
        for a, b in comparisons:
            ga, gb = g[g.model == a], g[g.model == b]
            for met in list(METRICS) + ["skill"]:
                t, p = stats.ttest_ind(ga[met], gb[met], equal_var=False)
                rows.append(dict(comparison=f"{a}-{b}", stratification=sname, stratum=sval, metric=met,
                                 diff=ga[met].mean() - gb[met].mean(), t=t, p=p))
    return pd.DataFrame(rows)


def paired_bootstrap(df, strata, comparisons, n_boot, rng):
    """Seed-averaged per-sample diffs, resampling glaciers."""
    tmp = df.assign(**{f"_st_{k}": v for k, v in strata.items()})
    stcols = [f"_st_{k}" for k in strata]
    avg = tmp.groupby(["model"] + KEY + stcols, sort=False)[list(METRICS.values())].mean().reset_index()
    rows = []
    for sname in strata:
        for sval, gs in avg.groupby(f"_st_{sname}", sort=False):
            wide = gs.pivot_table(index=KEY, columns="model", values=list(METRICS.values()))
            for a, b in comparisons:
                for met, col in METRICS.items():
                    d = (wide[(col, a)] - wide[(col, b)]).rename("d").reset_index()
                    per_gl = d.groupby("glacier_id").d.agg(["sum", "count"])
                    D, N = per_gl["sum"].values, per_gl["count"].values
                    idx = rng.integers(0, len(D), (n_boot, len(D)))
                    bs = D[idx].sum(1) / N[idx].sum(1)
                    lo, hi = np.percentile(bs, [2.5, 97.5])
                    rows.append(dict(comparison=f"{a}-{b}", stratification=sname, stratum=sval, metric=met,
                                     diff=D.sum() / N.sum(), ci_lo=lo, ci_hi=hi,
                                     n_glaciers=len(D), n=int(N.sum())))
    return pd.DataFrame(rows)


MD_FMT = {"MAE": "{:.2f}", "MDE": "{:.2f}", "HD": "{:.1f}", "Bias": "{:.1f}", "skill": "{:.3f}"}


def _md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _num(x, fmt="{:.2f}"):
    return fmt.format(x).replace("-", "−")


def _pm(row, met):
    return f"{_num(row[met], MD_FMT[met])} ± {_num(row[met + '_sd'], MD_FMT[met])}"


def write_markdown(path, summary, ref_welch, models, threshold, seeds):
    names = list(models)
    cols = list(METRICS) + ["skill"]
    col_names = [c if c != "skill" else "Skill" for c in cols]
    out = ["# Model comparison\n",
           f"Seeds {seeds}; fast mover = mean true displacement ≥ {threshold:g} m. "
           "Skill = 1 − mean(MDE) / mean(persistence MDE) within the group.\n"]

    s = summary.set_index(["stratification", "stratum", "model"])
    w = ref_welch.set_index(["comparison", "stratification", "stratum", "metric"])
    groups = [("all", "all"), ("seasonality", "seasonal"), ("seasonality", "non-seasonal")]
    diff_cols = ["MDE", "skill", "MAE", "HD", "Bias"]
    rows = []
    for m in names:
        if m == REFERENCE:
            continue
        for i, (st, sv) in enumerate(groups):
            row = [f"{m} − {REFERENCE}" if i == 0 else "", sv, int(s.loc[(st, sv, m), "n"])]
            for met in diff_cols:
                r = w.loc[(f"{m}-{REFERENCE}", st, sv, met)]
                d = _num(r["diff"], "{:+" + MD_FMT[met][2:])
                p = "p < 0.001" if r.p < 0.001 else f"p = {r.p:.3f}"
                row.append(f"**{d}** ({p})" if r.p < 0.05 else f"{d} ({p})")
            rows.append(row)
    out += [f"## Difference to {REFERENCE}\n",
            f"Difference of the seed-mean metric to {REFERENCE} ({len(seeds)} seeds per model). "
            f"Positive ΔMDE / ΔMAE / ΔHD and negative ΔSkill mean worse than {REFERENCE}. "
            "Bias is the signed mean offset of the predicted front from the true front (> 0 = too advanced); "
            f"positive ΔBias means more advanced than {REFERENCE}. "
            "p from Welch's t-test on the per-seed values; **bold** = p < 0.05. "
            "Within a group skill is a rescaled MDE, so both have the same p.\n",
            _md_table(["Comparison", "Group", "n", "ΔMDE [m]", "ΔSkill", "ΔMAE [m]", "ΔHD [m]", "ΔBias [m]"],
                      rows)]

    g = summary[summary.stratification == "all"].set_index("model")
    out += ["## Overall\n", _md_table(["Model"] + col_names, [[m] + [_pm(g.loc[m], c) for c in cols] for m in names])]

    rows = []
    for sval, g in summary[summary.stratification == "mover"].groupby("stratum", sort=False):
        g = g.set_index("model")
        for i, m in enumerate(names):
            rows.append([sval if i == 0 else "", int(g.n.iloc[0]) if i == 0 else "", m]
                        + [_pm(g.loc[m], c) for c in cols])
    out += ["## Fast vs slow movers\n", _md_table(["Group", "n", "Model"] + col_names, rows)]

    rows = []
    seas = summary[summary.stratification.isin(["seasonality", "seasonality_x_mover"])]
    for sval, g in seas.groupby("stratum", sort=False):
        g = g.set_index("model")
        rows.append([sval.replace("/", " / "), int(g.n.iloc[0])]
                    + [_num(g.loc[m, "skill"], MD_FMT["skill"]) for m in names]
                    + [_num(g.loc[m, "MDE"]) for m in names])
    out += ["## Seasonal vs non-seasonal glaciers, overall and split by mover speed\n",
            _md_table(["Group", "n"] + [f"{m} skill" for m in names] + [f"{m} MDE" for m in names], rows)]

    path.write_text("\n".join(out))


def write_latex(path, summary, ref_welch, models, n_seeds):
    """Difference to REFERENCE per model and glacier group, metrics as columns."""
    others = [m for m in models if m != REFERENCE]
    name = lambda m: TEX_NAMES.get(m, m)
    s = summary.set_index(["stratification", "stratum", "model"])
    w = ref_welch.set_index(["comparison", "stratification", "stratum", "metric"])
    groups = [("all", "all", "All"), ("seasonality", "seasonal", "Seasonal"),
              ("seasonality", "non-seasonal", "Non-seasonal")]
    lines = [r"\begin{table}[t]", r"\centering",
             rf"\caption{{Difference of the seed-mean test metrics to {name(REFERENCE)} for all, seasonal and "
             rf"non-seasonal test glaciers ($n$ = number of test samples). For MDE, MAE and Hausdorff, positive "
             rf"values mean worse than {name(REFERENCE)}. Bias is the signed mean offset of the predicted front "
             rf"from the true front (positive = too advanced), so positive $\Delta$ means more advanced than "
             rf"{name(REFERENCE)}. $p$ from a two-sided Welch's $t$-test on the per-seed values "
             rf"({n_seeds} seeds per model); bold: $p < 0.05$.}}",
             r"\label{tab:model_comparison}",
             r"\begin{tabular}{ll" + "rr" * len(TEX_METRICS) + "}",
             r"\hline",
             " & & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{lab} [m]}}" for lab in TEX_METRICS.values())
             + r" \\",
             "Comparison & Glaciers & " + " & ".join([r"$\Delta$ & $p$"] * len(TEX_METRICS)) + r" \\"]
    for m in others:
        lines.append(r"\hline")
        for i, (st, sv, label) in enumerate(groups):
            first = f"{name(m)} $-$ {name(REFERENCE)}" if i == 0 else ""
            cells = [first, rf"{label} ($n = {int(s.loc[(st, sv, m), 'n'])}$)"]
            for met in TEX_METRICS:
                r = w.loc[(f"{m}-{REFERENCE}", st, sv, met)]
                d = ("{:+" + MD_FMT[met][2:]).format(r["diff"])
                cells += [rf"$\mathbf{{{d}}}$" if r.p < 0.05 else f"${d}$",
                          "$<0.001$" if r.p < 0.001 else f"{r.p:.3f}"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}", ""]
    path.write_text("\n".join(lines))


def main():
    args = parse_args()
    models = dict(m.split("=", 1) for m in (args.model or DEFAULT_MODELS))
    if args.compare:
        comparisons = [tuple(c.split("-", 1)) for c in args.compare]
    elif args.model:
        names = list(models)
        comparisons = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    else:
        comparisons = [tuple(c.split("-", 1)) for c in DEFAULT_COMPARE]
    for a, b in comparisons:
        if a not in models or b not in models:
            raise ValueError(f"comparison {a}-{b} refers to unknown model")

    df = load_runs(models, args.seeds, EVAL_SUBDIR)
    strata = add_strata(df, args.threshold)
    per_seed = seed_level(df, strata)
    summary = summarise(per_seed)
    welch_df = welch(per_seed, comparisons)
    boot_df = paired_bootstrap(df, strata, comparisons, 5000, np.random.default_rng(0))

    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(f"Fast mover: mean_true_displacement_m >= {args.threshold} m; seeds {args.seeds}")
    print("Skill = 1 - mean(MDE) / mean(persistence MDE) within stratum\n")
    for sname, g in summary.groupby("stratification", sort=False):
        print(f"### {sname}\n{g.drop(columns='stratification').round(3).to_string(index=False)}\n")
    print("### Paired diff (seed-averaged per sample), glacier-bootstrap 95% CI; positive = first model worse")
    print(boot_df.round(2).to_string(index=False), "\n")
    print("### Seed-level Welch t-test on stratum means")
    print(welch_df.round(3).to_string(index=False))

    if args.out_dir:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        per_seed.to_csv(args.out_dir / "per_seed.csv", index=False)
        summary.to_csv(args.out_dir / "summary.csv", index=False)
        boot_df.to_csv(args.out_dir / "paired_bootstrap.csv", index=False)
        welch_df.to_csv(args.out_dir / "seed_welch.csv", index=False)
        ref_welch = welch(per_seed, [(m, REFERENCE) for m in models if m != REFERENCE])
        write_markdown(args.out_dir / "summary_tables.md", summary, ref_welch, models,
                       args.threshold, args.seeds)
        write_latex(args.out_dir / "comparison_table.tex", summary, ref_welch, models, len(args.seeds))
        print(f"\nWrote CSVs, summary_tables.md and comparison_table.tex to {args.out_dir}")


if __name__ == "__main__":
    main()
