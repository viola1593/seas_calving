#!/usr/bin/env python3
"""
scripts/skill_robustness.py

Skill-robustness aggregator for the seed-replicate paper runs.

Reads the per-seed evaluation summaries for the two seasonal architectures —
M1 (concat baseline, ``CalvingConvLSTMSeasonal``) and M3 (per-variable FiLM,
``CalvingConvLSTMFiLMPerVar``) — and reports forecast skill / MDE / MAE with
mean ± std across the 5 seeds. The overlapping M1/M3 spreads are the evidence
for the "FiLM ≈ concat" claim (no significance test; 5 vs 5 is underpowered).

This is a pure pandas/numpy aggregation — no GPU, no model loading, no torch.
It expects the summaries already written by ``evaluate_film.py`` into a fresh
``eval_paper/`` subdir (one canonical metric version on disk).

Per-seed forecast skill uses the ratio-of-means definition (stable, field
standard):

    skill = 1 − mean(model MDE) / mean(persistence MDE)

NB: the ``active_skill`` block inside the summary is the mean-of-per-sample
definition (unstable, dominated by barely-moving samples) — we do NOT use it;
skill is recomputed here from the two MDE means.

Usage:
    python scripts/skill_robustness.py
    python scripts/skill_robustness.py --root experiments/seasonal
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ── Run enumeration ───────────────────────────────────────────────────────────
MODELS = {"M1": "concat baseline", "M3": "per-variable FiLM"}
SEEDS = range(5)
CHECKPOINT = "best"  # checkpoint-name subdir written by evaluate_film.py


def summary_path(root: Path, model: str, seed: int) -> Path:
    """Path to one per-seed test summary under the eval_paper/ tree."""
    return root / f"paper_{model}_seed{seed}" / "eval_paper" / CHECKPOINT / "test_summary.json"


def load_seed_row(root: Path, model: str, seed: int) -> dict:
    """Read one summary and derive the reported metrics for a single seed."""
    path = summary_path(root, model, seed)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing eval summary for {model} seed{seed}: {path}\n"
            f"Re-run evaluate_film.py (test split, chkpt=best) into eval_paper/ first."
        )
    stats = json.loads(path.read_text())["model"]
    mde = stats["active_mde"]["mean"]
    persist_mde = stats["active_persist_mde"]["mean"]
    return {
        "model": model,
        "seed": seed,
        "skill": 1.0 - mde / persist_mde,
        "mde_m": mde,
        "mae_m": stats["mae_front"]["mean"],
    }


# ── Aggregation ───────────────────────────────────────────────────────────────
METRICS = ["skill", "mde_m", "mae_m"]


def aggregate(root: Path) -> pd.DataFrame:
    """Long-form frame: one row per model×seed, hard-failing on any missing run."""
    rows = [load_seed_row(root, m, s) for m in MODELS for s in SEEDS]
    return pd.DataFrame(rows)


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    """Per-model mean and sample std (ddof=1) across the 5 seeds."""
    return df.groupby("model")[METRICS].agg(["mean", "std"])


# ── Rendering ─────────────────────────────────────────────────────────────────
def _fmt(value: float, metric: str) -> str:
    return f"{value:.3f}" if metric == "skill" else f"{value:.1f}"


def render_markdown(df: pd.DataFrame, stats: pd.DataFrame) -> str:
    """SI table: per-seed values then mean ± std, grouped by model."""
    header = ["Seed"]
    for model in MODELS:
        header += [f"{model} Skill", f"{model} MDE (m)", f"{model} MAE (m)"]

    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(["---"] * len(header)) + "|",
    ]
    for seed in SEEDS:
        cells = [str(seed)]
        for model in MODELS:
            row = df[(df.model == model) & (df.seed == seed)].iloc[0]
            cells += [_fmt(row[m], m) for m in METRICS]
        lines.append("| " + " | ".join(cells) + " |")

    summary_cells = ["**mean ± std**"]
    for model in MODELS:
        for metric in METRICS:
            mean, std = stats.loc[model, (metric, "mean")], stats.loc[model, (metric, "std")]
            summary_cells.append(f"{_fmt(mean, metric)} ± {_fmt(std, metric)}")
    lines.append("| " + " | ".join(summary_cells) + " |")

    caption = (
        f"Seed-robustness of forecast skill on the seasonal test split "
        f"({len(SEEDS)} seeds per model). "
        f"M1 = {MODELS['M1']}, M3 = {MODELS['M3']}. "
        f"Skill = 1 − mean(model MDE) / mean(persistence MDE); "
        f"spread is sample std (ddof=1). Overlapping M1/M3 ranges show the two "
        f"architectures are indistinguishable in skill (FiLM ≈ concat)."
    )
    return f"### Skill robustness across seeds\n\n" + "\n".join(lines) + f"\n\n_{caption}_\n"


def build_tidy_csv(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Per-seed rows plus mean/std rows per model, ready for table or figure."""
    tidy = df.copy()
    tidy["seed"] = tidy["seed"].astype(str)
    stat_rows = []
    for model in MODELS:
        for agg in ("mean", "std"):
            stat_rows.append(
                {"model": model, "seed": agg, **{m: stats.loc[model, (m, agg)] for m in METRICS}}
            )
    return pd.concat([tidy, pd.DataFrame(stat_rows)], ignore_index=True)


# ── Entry point ───────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[3])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("experiments/seasonal"),
        help="Directory holding the paper_M{1,3}_seed{0..4} runs.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output dir (default: <root>/skill_robustness).",
    )
    args = parser.parse_args()

    df = aggregate(args.root)
    stats = summarise(df)

    out_dir = args.out or args.root / "skill_robustness"
    out_dir.mkdir(parents=True, exist_ok=True)

    tidy = build_tidy_csv(df, stats)
    tidy.to_csv(out_dir / "skill_robustness.csv", index=False)

    table = render_markdown(df, stats)
    (out_dir / "skill_robustness.md").write_text(table)

    print(table)
    print(f"\nWrote {out_dir/'skill_robustness.csv'}")
    print(f"Wrote {out_dir/'skill_robustness.md'}")


if __name__ == "__main__":
    main()
