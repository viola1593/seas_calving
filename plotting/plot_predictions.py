# plotting/plot_predictions.py
"""
Visualise sample predictions from CalvingLightningModule.predict_step().

Usage (from train_film.py after trainer.fit):
    from plotting.plot_predictions import plot_predictions
    plot_predictions(samples, out_dir)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import ListedColormap
from matplotlib.font_manager import FontProperties
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar


def _overlay_trace(ax, mask: np.ndarray, color: str, zorder: float = 5) -> None:
    """
    Draw a binary trace mask as its own coloured pixels (the front line itself),
    rather than ``ax.contour``, which would outline the boundary of the thin
    trace blob and produce a doubled line.

    ``zorder`` defaults above contour lines (which default to 2) so the overlay
    stays visible on top of the input-front contours.
    """
    overlay = np.ma.masked_where(mask < 0.5, mask)
    ax.imshow(overlay, cmap=ListedColormap([color]), vmin=0, vmax=1,
              interpolation="nearest", zorder=zorder)


def plot_glacier_predictions(
    samples: list[dict],
    out_path: str | Path,
) -> None:
    """
    One figure per glacier: each row is one prediction sorted by target_date.

    Panels per row:
      1. Last input SDT (m) + all K input front contours (grey, oldest→newest)
      2. Predicted displacement (m)
      3. True (target) displacement (m), shared colour scale with panel 2
      4. Predicted SDT (m) + predicted front (red) + true front (green dashed)
         + anchor front (black dashed)

    samples : list of dicts from _collect_plot_samples(), sorted by target_date.
    Each dict must contain: last_sdt_m, pred_disp_m, true_disp_m, pred_sdt_m,
    input_trace, target_trace, all_input_sdts_m (K,H,W) in metres, original_hw,
    glacier_id, target_date.
    """
    n = len(samples)
    K = samples[0]["all_input_sdts_m"].shape[0]
    grey_colors = plt.cm.Greys(np.linspace(0.3, 1.0, K))

    fig, axes = plt.subplots(n, 4, figsize=(17, 4 * n), squeeze=False)

    def _vlim(arr: np.ndarray) -> float:
        return max(float(np.percentile(np.abs(arr), 98)), 1.0)

    for i, s in enumerate(samples):
        h, w     = s["original_hw"]
        sdt      = s["last_sdt_m"][:h, :w]
        disp     = s["pred_disp_m"][:h, :w]
        tdisp    = s["true_disp_m"][:h, :w]
        psdt     = s["pred_sdt_m"][:h, :w]
        itr      = s["input_trace"][:h, :w]
        ttr      = s["target_trace"][:h, :w]
        all_sdts_m = s["all_input_sdts_m"][:, :h, :w]   # (K, H, W)

        kw = dict(fraction=0.046, pad=0.04, label="m")

        # Panel 1: last input SDT + all K input front contours
        ax = axes[i, 0]
        v  = _vlim(sdt)
        im = ax.imshow(sdt, cmap="RdBu_r", vmin=-v, vmax=v)
        for k in range(K):
            ax.contour(all_sdts_m[k], levels=[0.0], colors=[grey_colors[k]],
                       linewidths=0.8 + k * 0.3, label=f'input front{k+1}')
        fig.colorbar(im, ax=ax, **kw)
        ax.set_title(f"{s['glacier_id']} — {s['target_date']}\nLast input SDF", fontsize=9)
        ax.axis("off")

        # Panels 2 & 3: predicted vs true displacement (shared colour scale)
        v = max(_vlim(disp), _vlim(tdisp))

        ax = axes[i, 1]
        im = ax.imshow(disp, cmap="RdBu_r", vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, **kw)
        ax.set_title("Predicted displacement", fontsize=9)
        ax.axis("off")

        ax = axes[i, 2]
        im = ax.imshow(tdisp, cmap="RdBu_r", vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, **kw)
        ax.set_title("True displacement", fontsize=9)
        ax.axis("off")

        # Panel 4: predicted SDT + fronts
        ax = axes[i, 3]
        v  = _vlim(psdt)
        im = ax.imshow(psdt, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.contour(psdt, levels=[0.0], colors="red", linewidths=1.5)
        
        if itr.any():
            ax.contour(itr, levels=[0.5], colors="k", linewidths=1, linestyles="dashed")
        if ttr.any():
            _overlay_trace(ax, ttr, "green")
        
        fig.colorbar(im, ax=ax, **kw)
        skill = s.get("active_skill_score", float("nan"))
        skill_str = f"active skill = {skill:+.3f}" if np.isfinite(skill) else "active skill = n/a"
        ax.set_title(f"Predicted SDF\n{skill_str}", fontsize=9)
        ax.axis("off")

    axes[-1, -1].legend(
        handles=[
            Line2D([0], [0], color="red",   label="predicted front"),
            Line2D([0], [0], color="green", label="true front", linestyle="dashed"),
            Line2D([0], [0], color="k",     label="input front (anchor)", linestyle="dashed"),
            Line2D([0], [0], color=grey_colors[0],  label="input fronts (oldest→newest)",
                   linewidth=0.8),
            #Line2D([0], [0], color=grey_colors[-1], linewidth=0.8 + (K - 1) * 0.3),
        ],
        loc="lower right", fontsize=8,
    )

    plt.tight_layout()
    path = Path(out_path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved glacier plot → {path}")


def _sdt_vlim(arr: np.ndarray) -> float:
    """Symmetric colour limit for an SDT panel: 98th percentile of |arr|."""
    return max(float(np.percentile(np.abs(arr), 98)), 1.0)


def _metrics_str(sample: dict) -> str:
    """Skill and MDE as a one-line title suffix."""
    skill = sample.get("active_skill_score", float("nan"))
    mde   = sample.get("active_mde", float("nan"))
    return " · ".join((
        f"skill = {skill:+.3f}" if np.isfinite(skill) else "skill = n/a",
        f"MDE = {mde:.1f} m"    if np.isfinite(mde)   else "MDE = n/a",
    ))


def _draw_prediction(
    ax,
    sample:      dict,
    show_legend: bool = True,
    legend_loc:  str = "lower left",
    crop_m:      float = 200.0,
    bbox:        tuple[int, int, int, int] | None = None,
    vmax:        float | None = None,
) -> None:
    """
    Draw a single prediction onto ``ax`` (no title, no fig/save).

      • background : predicted SDT (grayscale)
      • input fronts : K contours in Blues (light = earliest → dark = most recent)
      • predicted front : red contour
      • true front : green contour
      • legend : input front dates + predicted/true (only if show_legend)

    The panel is cropped to the fronts: ``bbox`` (half-open r0, r1, c0, c1) is
    used when given, otherwise one is computed from this sample alone with
    ``crop_m`` metres of margin. Pass a shared ``bbox`` and ``vmax`` to put
    several panels on the same zoom and the same colour scale; leaving both
    ``None`` makes each panel self-contained.

    sample : one dict from _collect_plot_samples(), containing pred_sdt_m,
    target_trace, all_input_sdts_m (K,H,W) in metres, input_dates (len K),
    original_hw, active_skill_score.
    """
    h, w = sample["original_hw"]
    if bbox is None:
        bbox = _front_bbox([sample], crop_m)
    r0, r1, c0, c1 = bbox

    psdt       = sample["pred_sdt_m"][:h, :w][r0:r1, c0:c1]
    ttr        = sample["target_trace"][:h, :w][r0:r1, c0:c1]
    all_sdts_m = sample["all_input_sdts_m"][:, :h, :w][:, r0:r1, c0:c1]   # (K, h, w)
    dates      = sample["input_dates"]
    K          = all_sdts_m.shape[0]
    blues      = plt.cm.Blues(np.linspace(0.25, 1.0, K))  # light → dark = old → new

    if sample['glacier_id'] == 'RGI60-07.00470':
                legend_loc = 'lower right'
    elif sample['glacier_id'] in ['RGI60-07.01493']:
        legend_loc = 'upper left'
    elif sample['glacier_id'] in ['RGI60-07.00061', 'RGI60-07.00250']:
        legend_loc = 'upper right'
    else:
        legend_loc = 'lower left'
    if sample['glacier_id'] in ['RGI60-07.00470', 'RGI60-07.01493']:
        lw_pred = 4.0
        lw_in = 1.5
    else: 
        lw_pred = 2.0
        lw_in = 2.0
    
    # Background: predicted SDT in grayscale
    v = _sdt_vlim(psdt) if vmax is None else vmax
    ax.imshow(psdt, cmap="RdBu_r", vmin=-v, vmax=v)

    # Input fronts, oldest → newest
    for k in range(K):
        linestyle = 'dotted' if k < 4 else 'dashed'
        ax.contour(all_sdts_m[k], levels=[0.0], colors=[blues[k]], 
        linewidths=lw_in,linestyles=linestyle, zorder=k+2)

    # Predicted front (red) and true front (green)
    ax.contour(psdt, levels=[0.0], colors="lightcoral", linewidths=lw_pred, zorder=K)
    if ttr.any():
        _overlay_trace(ax, ttr, "darkmagenta")

    # Skill score, upper-left — now in the panel title instead (see _metrics_str)
    # skill = sample.get("active_skill_score", float("nan"))
    # skill_str = f"skill = {skill:+.3f}" if np.isfinite(skill) else "skill = n/a"
    # mde = sample.get("active_mde", float("nan"))
    # mde_str = f"MDE = {mde:.1f} m" if np.isfinite(mde) else "MDE = n/a"
    #
    # ax.text(0.03, 0.97, skill_str+ '\n' + mde_str, transform=ax.transAxes, va="top", ha="left",
    #         fontsize=11, color="white",
    #         bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", pad=3))

    if show_legend:
        linestyles = ['dotted' if k < 4 else 'dashed' for k in range(K)]
        handles = [Line2D([0], [0], color=blues[k], lw=1.5, label=dates[k][:7], linestyle=linestyles[k])
                   for k in range(K)]
        handles += [
            Line2D([0], [0], color="lightcoral",       lw=lw_pred, label="predicted"),
            Line2D([0], [0], color="darkmagenta", lw=lw_pred, label="true"),
        ]
        
        
        
        ax.legend(handles=handles, loc=legend_loc,  fontsize=14,
                  title="input fronts", title_fontsize=14, framealpha=0.7)

    ax.axis("off")


def plot_single_prediction(
    sample:   dict,
    out_path: str | Path,
) -> None:
    """
    One figure for a single prediction (one target timestamp).

      • background : predicted SDT (grayscale)
      • input fronts : K contours in Blues (light = earliest → dark = most recent)
      • predicted front : red contour
      • true front : green contour
      • skill score : text, upper-left
      • title : glacier RGI id + target date
      • legend : input front dates

    sample : one dict from _collect_plot_samples(), containing last_sdt_m,
    pred_sdt_m, target_trace, all_input_sdts_m (K,H,W) in metres, input_dates
    (len K), original_hw, glacier_id, target_date, active_skill_score.
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    _draw_prediction(ax, sample, show_legend=True)
    ax.set_title(f"{sample['glacier_id']} — {sample['target_date']}\n"
                 f"{_metrics_str(sample)}", fontsize=11)

    plt.tight_layout()
    path = Path(out_path)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved single prediction → {path}")


def _add_scale_bar(ax, width_px: int, px_size_m: float = 30.0, loc: str = "lower left") -> None:
    """
    Draw a horizontal scale bar (lower right). Axes are equal-aspect, so the
    same scale applies to both dimensions.

    width_px : panel width in pixels, used to pick a bar length of roughly a
    quarter of the panel from a set of round distances.
    """
    choices = np.array([500, 1000, 2000])   # metres
    length_m = int(choices[np.argmin(np.abs(choices - width_px * px_size_m / 4))])
    label = f"{length_m // 1000} km" if length_m >= 1000 else f"{length_m} m"
    ax.add_artist(AnchoredSizeBar(
        ax.transData, length_m / px_size_m, label, loc=loc,
        pad=0.4, color="black", frameon=True, size_vertical=width_px * 0.005,
        fontproperties=FontProperties(size=14),
    ))


def _fit_span(lo: int, size: int, limit: int) -> tuple[int, int]:
    """
    Half-open span of ``size`` starting at ``lo``, shifted back inside
    ``[0, limit)`` and truncated only when the tile itself is too small.

    Shifting rather than clipping matters at the tile edge: clipping there would
    silently hand back the sliver the aspect clamp just widened away.
    """
    size = min(size, limit)
    lo   = min(max(lo, 0), limit - size)
    return lo, lo + size


def _front_bbox(
    samples:   list[dict],
    crop_m:    float = 500.0,
    px_size_m: float = 30.0,
) -> tuple[int, int, int, int]:
    """
    Half-open bounding box (r0, r1, c0, c1) around every observed front in these
    samples, i.e. the tile region within ``crop_m`` metres of a front.

    The mask is the union over samples of the input-front contours (|SDT| within
    one pixel of the zero level) and the true trace. The predicted front is
    deliberately excluded: its zero level also runs along the fjord walls, which
    would inflate the box back to the whole tile.

    Padding is isotropic, which is exactly equivalent to "every pixel within
    ``crop_m`` of the mask" — dilating by a disk of radius r grows a bounding box
    by r on each side regardless of how the front is oriented.

    The box is then widened (never shrunk) until its aspect ratio lies within
    [2/3, 3/2], so a line-like front does not yield a sliver panel, and shifted
    back inside the tile so it keeps that size.
    """
    pad = int(round(crop_m / px_size_m))
    rows, cols = [], []
    for s in samples:
        h, w = s["original_hw"]
        near = (s["target_trace"][:h, :w] > 0.5)#((np.abs(s["all_input_sdts_m"][:, :h, :w]) <= px_size_m).any(0)
                #| (s["target_trace"][:h, :w] > 0.5))
        r, c = np.nonzero(near)
        rows.append(r)
        cols.append(c)
    r, c = np.concatenate(rows), np.concatenate(cols)
    h, w = samples[0]["original_hw"]

    r0, r1 = int(r.min()) - pad, int(r.max()) + 1 + pad
    c0, c1 = int(c.min()) - pad, int(c.max()) + 1 + pad

    # Aspect clamp: widen the short side only.
    n_rows, n_cols = r1 - r0, c1 - c0
    if n_cols > 1.5 * n_rows:
        n_rows = int(np.ceil(n_cols / 1.5))
    elif n_cols < n_rows * 2 / 3:
        n_cols = int(np.ceil(n_rows * 2 / 3))

    r0, r1 = _fit_span(r0 - (n_rows - (r1 - r0)) // 2, n_rows, h)
    c0, c1 = _fit_span(c0 - (n_cols - (c1 - c0)) // 2, n_cols, w)
    return r0, r1, c0, c1


def plot_glacier_skill(
    samples: list[dict],
    out_path: str | Path,
    metric: str = "active_skill_score"
) -> None:
    """
    One 1×3 figure per glacier: best / median / worst prediction by
    ``active_skill_score``, side by side.

    Columns are fixed best → median → worst (median = sorted index n//2).
    The legend is drawn only on the worst (rightmost) panel. When fewer than
    three finite-skill samples exist, a panel may reuse a sample already shown
    in another role; the repeat is annotated in its title, e.g.
    ``" (=Best, repeated)"``.

    samples : list of dicts from _collect_plot_samples() for one glacier.
    Caller guarantees at least one sample has a finite active_skill_score.
    """
    valid = [s for s in samples if np.isfinite(s.get(metric, float("nan")))]
    valid.sort(key=lambda s: s[metric])
    if metric == "active_skill_score":
        best, median, worst = valid[-1], valid[len(valid) // 2], valid[0]
    elif metric == "active_mde":
        best, median, worst = valid[0], valid[len(valid) // 2], valid[-1]
    else:
        raise ValueError(f"Unknown metric {metric!r} for glacier skill plot")
    mean = np.mean([s[metric] for s in valid])
    mean_mde = np.mean([s.get("active_mde", float("nan")) for s in valid])
    mean_pers_mde = np.mean([s.get("active_persist_mde", float("nan")) for s in valid])
    mean_skill = 1-mean_mde/mean_pers_mde
    roles = [("Best", best), 
                ("Median", median), 
                ("Worst", worst)]

    # One crop and one colour scale for the whole row, so the three panels are
    # directly comparable: same zoom, same metres-per-colour.
    r0, r1, c0, c1 = bbox = _front_bbox([s for _, s in roles])
    vmax = max(
        _sdt_vlim(s["pred_sdt_m"][:s["original_hw"][0], :s["original_hw"][1]][r0:r1, c0:c1])
        for _, s in roles
    )

    fig, axes = plt.subplots(1, len(roles), figsize=(5*len(roles), 5.5), dpi=300)

    seen: dict[int, str] = {}
    for j, (role, s) in enumerate(roles):
        ax = axes[j]
        if s["glacier_id"] in ["RGI60-07.00240", "RGI60-07.00062_C"]:
            show_legend= True
        else:
            show_legend= True
        _draw_prediction(ax, s, show_legend=show_legend, bbox=bbox, vmax=vmax)
        if j == 0: #len(roles) - 1:   
            if s['glacier_id'] in ["RGI60-07.01493", "RGI60-07.00681",
             "RGI60-07.00062_C", "RGI60-07.00064", "RGI60-07.00240",
              "RGI60-07.00502"]:  
                loc = "lower right"    # geometry is shared: one scale bar suffices
            else:
                loc = "lower left"
            _add_scale_bar(ax, width_px=(c1 - c0), loc=loc)
        title = f"{role} · {s['target_date']} "
        if id(s) in seen:
            title += f" (={seen[id(s)]}, repeated)"
        else:
            seen[id(s)] = role
        ax.set_title(f"{title}\n{_metrics_str(s)}", fontsize=14)

    # fig.suptitle(samples[0]["glacier_id"]+f" (Mean Skill Score: {mean_skill:.2f})", fontsize=12)

    plt.tight_layout()
    path = Path(out_path)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved glacier skill plot → {path}")


def plot_predictions(
    samples: list[dict],
    out_dir: str | Path,
    n_samples: int = 4,
    filename: str = "sample_predictions.png",
) -> None:
    """
    Plot up to n_samples prediction results and save to out_dir.

    Each row shows one sample with three panels:
      1. Last input SDF (m) + input calving front (black contour)
      2. Predicted displacement (m)
      3. Predicted SDF (m) + predicted front (yellow) + true front (cyan)

    samples : list of dicts from CalvingLightningModule.predict_step()
    """
    samples = samples[:n_samples]
    n = len(samples)
    fig, axes = plt.subplots(n, 3, figsize=(13, 4 * n), squeeze=False)

    def _vlim(arr: np.ndarray) -> float:
        return max(float(np.percentile(np.abs(arr), 98)), 1.0)

    for i, s in enumerate(samples):
        h, w = s["original_hw"]
        sdt  = s["last_sdt_m"][:h,  :w]
        disp = s["pred_disp_m"][:h, :w]
        psdt = s["pred_sdt_m"][:h,  :w]
        itr  = s["input_trace"][:h,  :w]
        ttr  = s["target_trace"][:h, :w]

        kw = dict(fraction=0.046, pad=0.04, label="m")

        # ── Panel 1: last input SDF + input front ─────────────────────────
        ax = axes[i, 0]
        v  = _vlim(sdt)
        im = ax.imshow(sdt, cmap="RdBu_r", vmin=-v, vmax=v)
        if itr.any():
            ax.contour(itr, levels=[0.5], colors="k", linewidths=1)
        fig.colorbar(im, ax=ax, **kw)
        ax.set_title(f"{s['glacier_id']}\nLast input SDF", fontsize=9)
        ax.axis("off")

        # ── Panel 2: predicted displacement ───────────────────────────────
        ax = axes[i, 1]
        v  = _vlim(disp)
        im = ax.imshow(disp, cmap="RdBu_r", vmin=-v, vmax=v)
        fig.colorbar(im, ax=ax, **kw)
        ax.set_title("Predicted displacement", fontsize=9)
        ax.axis("off")

        # ── Panel 3: predicted SDF + fronts ───────────────────────────────
        ax = axes[i, 2]
        v  = _vlim(psdt)
        im = ax.imshow(psdt, cmap="RdBu_r", vmin=-v, vmax=v)
        ax.contour(psdt, levels=[0.0], colors="red", linewidths=1.5)
        if ttr.any():
            _overlay_trace(ax, ttr, "green")
        if itr.any():
            ax.contour(itr, levels=[0.5], colors="k", linewidths=1, linestyles="dashed")
        fig.colorbar(im, ax=ax, **kw)
        ax.set_title("Predicted SDF", fontsize=9)
        ax.axis("off")

    axes[-1, -1].legend(
        handles=[
            Line2D([0], [0], color="red", label="predicted front"),
            Line2D([0], [0], color="green",   label="true front"),
            Line2D([0], [0], color="k",      label="input front (persistence)", linestyle="dashed"),
        ],
        loc="lower right", fontsize=8,
    )

    plt.tight_layout()
    path = Path(out_dir) / filename
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved sample predictions → {path}")
