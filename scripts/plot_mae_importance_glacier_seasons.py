"""Per-glacier ΔMAE input importance vs season, one panel per input variable and glacier group.

Reads the <prefix>_by_glacier.csv written by model_analysis/aggregate_input_importance_mae.py
and draws one curve per glacier (DJF → MAM → JJA → SON), skipping glaciers with fewer than
MIN_SAMPLES samples over all seasons. Glaciers are split by the expert `seasonal?` label (the
one fed to the model as `glacier_seasonality`): seasonal glaciers fill the top rows,
non-seasonal glaciers the bottom rows. Each group's legend sits outside the grid beside its
rows so no curve is covered. 22 glaciers exceed any distinguishable colour set, so identity
is colour × line style (8 × 3).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
CSV = (REPO / "experiments/seasonal/paper_M3_seed3/"
       "explain/input_importance_mae/mae_importance_by_glacier.csv")
SEASONALITY_CSV = REPO.parent / "glacier_data" / "Merged_Seasonality_DecadalRates.csv"
OUT = CSV.with_name("mae_importance_by_glacier_seasons.png")

SEASONS = ["DJF", "MAM", "JJA", "SON"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
LINESTYLES = ["-", "--", ":"]
MARKERS = ["o", "s", "^"]
NCOLS = 5
MIN_SAMPLES = 10  # glaciers with fewer samples (all seasons) are not plotted

df = pd.read_csv(CSV)
variables = list(dict.fromkeys(df["variable"]))  # keep CSV order
all_glaciers = sorted(df["glacier_id"].unique())
# styles over all glaciers so a glacier keeps its colour whatever the filter drops
style = {g: dict(color=COLORS[i % len(COLORS)],
                 ls=LINESTYLES[i // len(COLORS)],
                 marker=MARKERS[i // len(COLORS)])
         for i, g in enumerate(all_glaciers)}
# samples per glacier summed over seasons (identical across variables)
n_total = df.groupby(["variable", "glacier_id"])["n"].sum().groupby("glacier_id").min()
glaciers = [g for g in all_glaciers if n_total[g] >= MIN_SAMPLES]
dropped = [f"{g} (n={n_total[g]})" for g in all_glaciers if g not in glaciers]
print(f"Dropped (< {MIN_SAMPLES} samples): {', '.join(dropped) or 'none'}")

# Excel export: ';' separated, ',' decimal mark, trailing blank rows
labels = pd.read_csv(SEASONALITY_CSV, sep=";", decimal=",", index_col=0, low_memory=False)
labels = labels[labels["Glacier"].notna()].set_index("Glacier")["seasonal?"]
is_seasonal = labels.astype(str).str.upper() == "TRUE"
groups = {"seasonal": [g for g in glaciers if is_seasonal[g]],
          "non-seasonal": [g for g in glaciers if not is_seasonal[g]]}
for name, members in groups.items():
    print(f"{name}: {len(members)} glaciers")

rows_per_group = int(np.ceil(len(variables) / NCOLS))
fig, axes = plt.subplots(2 * rows_per_group, NCOLS,
                         figsize=(NCOLS * 3.2, 2 * rows_per_group * 3.0),
                         sharex=True, squeeze=False)
x = np.arange(len(SEASONS))
for b, (name, members) in enumerate(groups.items()):
    block = axes[b * rows_per_group:(b + 1) * rows_per_group]
    for ax, var in zip(block.flat, variables):
        table = (df[df["variable"] == var]
                 .pivot(index="glacier_id", columns="season", values="mean_signed_mae_m")
                 .reindex(columns=SEASONS))
        ax.axhline(0.0, color="gray", lw=0.8, ls="--", zorder=0)
        for g in members:
            if g in table.index:
                ax.plot(x, table.loc[g].to_numpy(), lw=1.2, ms=4, label=g, **style[g])
        ax.set_title(f"{var} · {name}", fontsize=10)
        ax.grid(axis="y", color="0.9", lw=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(SEASONS)
    for ax in block.flat[len(variables):]:
        ax.axis("off")
    for ax in block[:, 0]:
        ax.set_ylabel("ΔMAE (m)")
    handles = [plt.Line2D([], [], lw=1.2, ms=4, **style[g]) for g in members]
    fig.legend(handles, members, title=f"{name} (n ≥ {MIN_SAMPLES})", loc="center left",
               bbox_to_anchor=(1.0, 1 - (b + 0.5) / 2), fontsize=8, frameon=False)

fig.suptitle("Input importance per glacier by season (ΔMAE = MAE_ablated − MAE_full, "
             "+ = driver helps)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight")
print(f"Saved: {OUT}")
