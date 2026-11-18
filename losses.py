# losses.py
"""
Loss function for calving front SDT prediction.

Single-term design: front-proximity-weighted L1 loss on the displacement field.

Why sign loss was removed
--------------------------
Analysis of training runs showed loss_sign dominated gradient by 14–24× over
loss_l1 even at lambda_sign=0.2, because raw normalised displacement values
fed as BCE logits are outside the well-behaved sigmoid range.  The sign BCE
collapsed to a near-constant value early in training, providing no further
gradient while masking the L1 signal entirely.  Pure L1 on the displacement
field is sufficient: predicting the wrong sign for a nonzero displacement
incurs the same L1 penalty as predicting the correct sign with double the
magnitude.

Why decay_pixels=10 (not 5)
-----------------------------
With decay_pixels=5 (150 m half-power), the gradient weight at 300–500 m from
the trace is exp(-10)–exp(-17) ≈ 0–5%.  In early training the model's
predicted front is typically 300–500 m from the true front; those error pixels
receive near-zero gradient and the model cannot self-correct.  decay_pixels=10
(300 m half-power) keeps ~37% of the peak weight at 300 m and ~14% at 500 m,
allowing the model to learn from its own early errors.

Weighting strategy
------------------
trace_proximity_weight uses zarr["trace"] (Trace_MASK.tif) which contains
ONLY the calving front — not the static fjord boundary.  Distance from the
trace is computed via iterative BFS dilation using F.max_pool2d, fully
GPU-compatible with no scipy/numpy in the forward pass.

Dataset alignment
-----------------
- Spatial resolution : 30 m (EPSG:3995)
- decay_pixels = 10  : half-power at 10 px × 30 m = 300 m from the front
- min_weight   = 0.0 : all gradient concentrated near the calving front
- input_trace        : batch["input_trace"] from CalvingFrontDataset
                       (1, H, W) float32, 1 = on front, 0 = elsewhere
- spatial_valid_mask : batch["spatial_valid_mask"] from calving_collate_fn
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Utility: trace-based proximity weight ─────────────────────────────────────

def trace_proximity_weight(
    trace:        torch.Tensor,                   # (B, 1, H, W) float32
    decay_pixels: float = 10.0,
    min_weight:   float = 0.0,
    mask:         Optional[torch.Tensor] = None,  # (B, 1, H, W) bool
) -> torch.Tensor:                                # (B, 1, H, W) float32
    """
    Compute a spatial weight map that is highest at the calving front and
    decays exponentially with pixel distance from the trace.

    w(x) = (1 - min_weight) × exp(-dist(x) / decay_pixels) + min_weight

    Distance is computed via iterative BFS dilation (one pixel per step)
    using a 3×3 F.max_pool2d kernel — fully GPU-compatible and
    differentiability-graph-safe (dist_map is not part of the pred/target
    computation graph; it is a fixed weighting scaffold).

    The weight map is normalised so the mean over valid pixels equals 1.0,
    keeping total loss magnitude stable across different glacier sizes.

    Parameters
    ----------
    trace        : (B, 1, H, W) float32 — 1.0 at calving front pixels,
                   0.0 elsewhere.  From batch["input_trace"].
    decay_pixels : exponential decay length in pixels.
                   10 px × 30 m/px = 300 m half-power distance.
    min_weight   : weight floor for far-field pixels (0–1).
    mask         : (B, 1, H, W) bool — if provided, normalisation is
                   restricted to valid (non-padded) pixels.

    Returns
    -------
    (B, 1, H, W) float32 weight map, mean ≈ 1.0 over valid pixels.
    """
    trace = trace.float()
    H, W  = trace.shape[-2], trace.shape[-1]

    # Maximum distance (pixels) for BFS expansion.
    # Capped at 4 × decay_pixels: weight at that distance = exp(-4) ≈ 2%,
    # negligible regardless of min_weight.
    max_dist_px = min(
        math.ceil(-math.log(min_weight + 1e-8) * decay_pixels),
        round(4 * decay_pixels),
        max(H, W),
    )

    # Initialise distance map: 0 at trace pixels, max_dist_px elsewhere.
    dist_map = torch.where(
        trace > 0.5,
        torch.zeros_like(trace),
        torch.full_like(trace, float(max_dist_px)),
    )

    # BFS dilation: expand the front outward one pixel per step.
    for step in range(1, max_dist_px + 1):
        reachable     = (dist_map < step).float()
        dilated       = F.max_pool2d(reachable, kernel_size=3, stride=1, padding=1)
        newly_reached = (dilated > 0) & (dist_map >= step)
        dist_map      = torch.where(
            newly_reached,
            torch.full_like(dist_map, float(step)),
            dist_map,
        )
        # if mask is not None:  ## this costs a lot of GPU time
        #     remaining = (dist_map >= max_dist_px) & mask
        # else:
        #     remaining = dist_map >= max_dist_px
        # if not remaining.any():
        #     break

    # Convert distance to exponential weight.
    w = (1.0 - min_weight) * torch.exp(-dist_map / (decay_pixels + 1e-8)) + min_weight

    # Normalise so mean over valid pixels ≈ 1.0.
    if mask is not None:
        m     = mask.float()
        denom = (w * m).sum(dim=(-2, -1), keepdim=True) + 1e-8
        count = m.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
        w     = w * count / denom
    else:
        w = w / (w.mean(dim=(-2, -1), keepdim=True) + 1e-8)

    return w


def _error_term(
    pred:   torch.Tensor,
    target: torch.Tensor,
    norm:   str,
) -> torch.Tensor:
    """Per-pixel error: |e| for norm='l1', e² for norm='l2'."""
    err = pred - target
    if norm == "l1":
        return err.abs()
    if norm == "l2":
        return err.pow(2)
    raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")


def _masked_mean(
    tensor: torch.Tensor,
    mask:   Optional[torch.Tensor],
) -> torch.Tensor:
    """Mean over valid pixels only; falls back to .mean() when mask is None."""
    if mask is None:
        return tensor.mean()
    m = mask.float()
    return (tensor * m).sum() / (m.sum() + 1e-8)


def _per_sample_spatial_mean(
    tensor: torch.Tensor,
    mask:   Optional[torch.Tensor],
) -> torch.Tensor:                          # (B,)
    if mask is None:
        return tensor.mean(dim=(-3, -2, -1))
    m = mask.float()
    return (tensor * m).sum(dim=(-3, -2, -1)) / (m.sum(dim=(-3, -2, -1)).clamp(min=1.0) + 1e-8)


def displacement_sample_scale(
    target_displacement: torch.Tensor,       # (B, 1, H, W) normalised
    disp_std:            float,
    mask:                Optional[torch.Tensor] = None,  # (B, 1, H, W) bool
) -> torch.Tensor:                           # (B,) float32
    abs_m = target_displacement.float().abs() * disp_std
    mean_abs_m = _per_sample_spatial_mean(abs_m, mask)   # (B,)
    return 1.0 + (mean_abs_m / 150.0).clamp(0.0, 3.0)


# ── Master loss ────────────────────────────────────────────────────────────────

class CalvingLoss(nn.Module):
    """
    Front-proximity-weighted L1 loss on the SDT displacement field.

    Total loss
    ----------
    L = mean_valid[ w(x) × |pred_disp(x) − target_disp(x)| ]      (norm='l1')
    L = mean_valid[ w(x) × (pred_disp(x) − target_disp(x))² ]     (norm='l2')

    w(x) is from trace_proximity_weight with min_weight=0, concentrating all
    gradient near the calving front.  When input_trace is unavailable, uniform
    weights are used with a warning.

    Displacement denormalisation (for reference, not used here):
      displacement_metres = displacement_norm × disp_std   (no + sdt_mean)
      disp_std ≈ 182 m is stored in stats["displacement"]["std"].

    Parameters
    ----------
    decay_pixels : exponential decay in pixels; 10 px × 30 m = 300 m half-power.
    min_weight   : weight floor. 0.0 = all gradient at/near front only.
    norm         : 'l1' (absolute error, default — robust to cloud/registration
                   outliers) or 'l2' (squared error, penalises large errors more).
    """

    def __init__(
        self,
        decay_pixels: float = 10.0,
        min_weight:   float = 0.0,
        norm:         str   = "l1",
    ):
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        self.decay_pixels = decay_pixels
        self.min_weight   = min_weight
        self.norm         = norm

    def forward(
        self,
        pred_displacement:   torch.Tensor,                   # (B, 1, H, W) normalised
        target_displacement: torch.Tensor,                   # (B, 1, H, W) normalised
        input_trace:         Optional[torch.Tensor] = None,  # (B, 1, H, W) float32
        mask:                Optional[torch.Tensor] = None,  # (B, 1, H, W) bool
        per_sample:          bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Compute front-proximity-weighted L1 loss on displacement.

        Parameters
        ----------
        pred_displacement   : (B, 1, H, W) normalised predicted displacement
        target_displacement : (B, 1, H, W) normalised true displacement
        input_trace         : (B, 1, H, W) float32 — calving front trace at
                              t-30; 1.0 on front pixels, 0.0 elsewhere.
        mask                : (B, 1, H, W) bool — valid pixel mask.

        Returns
        -------
        dict with key ``loss`` (front-proximity-weighted L1 or L2).
        """
        if input_trace is not None and input_trace.any():
            w = trace_proximity_weight(
                input_trace,
                decay_pixels = self.decay_pixels,
                min_weight   = self.min_weight,
                mask         = mask,
            )
        else:
            warnings.warn(
                "input_trace is None or empty — using uniform weights. "
                "Loss will not be focused on the calving front.",
                stacklevel=2,
            )
            w = torch.ones_like(pred_displacement)
            if mask is not None:
                w = w * mask.float()

        weighted_err = w * _error_term(pred_displacement, target_displacement, self.norm)
        if per_sample:
            loss = _per_sample_spatial_mean(weighted_err, mask)  # (B,)
        else:
            loss = _masked_mean(weighted_err, mask)               # scalar

        return {"loss": loss}


# ── Movement-weighted loss ─────────────────────────────────────────────────────

def movement_weight(target_disp_norm: torch.Tensor, disp_std: float) -> torch.Tensor:
    """(B,1,H,W) normalised displacement → per-pixel weight in [1, 4]."""
    return 1.0 + (target_disp_norm.float().abs() * disp_std).clamp(0.0, 450.0) / 150.0


class CalvingMovementLoss(nn.Module):
    """
    Front-proximity × displacement-magnitude weighted L1 loss.

    Combines trace_proximity_weight with movement_weight so that pixels near
    fast-moving fronts receive up to 4× more gradient than slow/static pixels.
    The combined weight is re-normalised to mean=1 over valid pixels, keeping
    total loss magnitude comparable to CalvingLoss.

    When target_displacement is identically zero, movement_weight = 1 everywhere
    and this loss is numerically equivalent to CalvingLoss.

    Parameters
    ----------
    disp_std     : displacement std in metres (from stats["displacement"]["std"]).
                   Required — no default. Converts normalised displacement to metres
                   for the movement weight thresholds (150 m denominator, 450 m clip).
    decay_pixels : exponential decay in pixels (10 px × 30 m = 300 m half-power).
    min_weight   : proximity weight floor.
    norm         : 'l1' (absolute error, default) or 'l2' (squared error).
    """

    def __init__(self, disp_std: float, decay_pixels: float = 10.0, min_weight: float = 0.0,
                 norm: str = "l1"):
        super().__init__()
        if norm not in ("l1", "l2"):
            raise ValueError(f"norm must be 'l1' or 'l2', got {norm!r}")
        self.disp_std     = disp_std
        self.decay_pixels = decay_pixels
        self.min_weight   = min_weight
        self.norm         = norm

    def forward(
        self,
        pred_displacement:   torch.Tensor,                   # (B, 1, H, W) normalised
        target_displacement: torch.Tensor,                   # (B, 1, H, W) normalised
        input_trace:         Optional[torch.Tensor] = None,  # (B, 1, H, W) float32
        mask:                Optional[torch.Tensor] = None,  # (B, 1, H, W) bool
    ) -> dict[str, torch.Tensor]:
        if input_trace is not None and input_trace.any():
            w_prox = trace_proximity_weight(
                input_trace,
                decay_pixels = self.decay_pixels,
                min_weight   = self.min_weight,
                mask         = mask,
            )
        else:
            warnings.warn(
                "input_trace is None or empty — using uniform weights. "
                "Loss will not be focused on the calving front.",
                stacklevel=2,
            )
            w_prox = torch.ones_like(pred_displacement)

        w_move  = movement_weight(target_displacement, self.disp_std)
        w_total = w_prox * w_move

        # Re-normalise combined weight so mean over valid pixels ≈ 1.0.
        if mask is not None:
            m       = mask.float()
            denom   = (w_total * m).sum(dim=(-2, -1), keepdim=True) + 1e-8
            count   = m.sum(dim=(-2, -1), keepdim=True).clamp(min=1.0)
            w_total = w_total * count / denom
        else:
            w_total = w_total / (w_total.mean(dim=(-2, -1), keepdim=True) + 1e-8)

        loss = _masked_mean(
            w_total * _error_term(pred_displacement, target_displacement, self.norm), mask
        )
        return {"loss": loss}


# ── Convenience: build loss from config dict ──────────────────────────────────

def build_loss(cfg: dict) -> nn.Module:
    """
    Construct a loss module from a config dict (from configs/*.yaml).

    Pass ``movement_weight: true`` and ``disp_std: <float>`` to get
    CalvingMovementLoss; otherwise returns CalvingLoss.
    Pass ``norm: l1`` (default) or ``norm: l2`` to select the error term.

    All legacy keys (lambda_sign, lambda_recon, etc.) are ignored silently.
    """
    if cfg.get("movement_weight", False):
        if "disp_std" not in cfg:
            raise ValueError("build_loss: 'disp_std' required when movement_weight=True")
        return CalvingMovementLoss(
            disp_std     = cfg["disp_std"],
            decay_pixels = cfg.get("decay_pixels", 10.0),
            min_weight   = cfg.get("min_weight",   0.0),
            norm         = cfg.get("norm",         "l1"),
        )
    return CalvingLoss(
        decay_pixels = cfg.get("decay_pixels", 10.0),
        min_weight   = cfg.get("min_weight",   0.0),
        norm         = cfg.get("norm",         "l1"),
    )


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import warnings as _warnings

    print("Running losses self-test (displacement formulation)...\n")
    torch.manual_seed(0)

    B, H, W = 2, 64, 64

    true_disp = torch.zeros(B, 1, H, W)
    true_disp[:, :, H // 2 :, :] = -0.25

    trace = torch.zeros(B, 1, H, W)
    trace[:, :, H // 2, :] = 1.0

    mask = torch.ones(B, 1, H, W, dtype=torch.bool)
    mask[:, :, :, -4:] = False

    criterion = CalvingLoss(decay_pixels=4.0)

    def check(name: str, cond: bool, val: float | None = None) -> None:
        tag = "PASS" if cond else "FAIL"
        msg = f"[{tag}]  {name}"
        if val is not None:
            msg += f"  (value={val:.6f})"
        print(msg)

    # ── 1. Perfect displacement prediction → loss near 0 ──────────────────
    out = criterion(true_disp.clone(), true_disp, input_trace=trace, mask=mask)
    check("1. Perfect pred → loss near 0",
          out["loss"].item() < 1e-6, out["loss"].item())

    # ── 2. Persistence (predict zero) with real displacement → loss > 0 ───
    persist_pred = torch.zeros_like(true_disp)
    out_persist  = criterion(persist_pred, true_disp, input_trace=trace, mask=mask)
    check("2. Persistence (zero pred) with nonzero disp → loss > 0",
          out_persist["loss"].item() > 0, out_persist["loss"].item())

    # ── 3. Good model beats persistence ───────────────────────────────────
    near_correct = true_disp + 0.02 * torch.randn_like(true_disp)
    out_model    = criterion(near_correct, true_disp, input_trace=trace, mask=mask)
    check("3. Near-correct model loss < persistence loss",
          out_model["loss"].item() < out_persist["loss"].item(),
          out_model["loss"].item())

    # ── 4. With trace vs None fallback (should warn) ───────────────────────
    out_with = criterion(near_correct, true_disp, input_trace=trace)
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        out_none = criterion(near_correct, true_disp, input_trace=None)
    check("4. With trace → valid positive loss",
          not torch.isnan(out_with["loss"]) and out_with["loss"].item() > 0,
          out_with["loss"].item())
    check("4. None trace → valid loss + warning containing 'uniform'",
          len(caught) >= 1
          and "uniform" in str(caught[0].message).lower()
          and not torch.isnan(out_none["loss"])
          and out_none["loss"].item() > 0,
          out_none["loss"].item())

    # ── 5. Spatial mask excludes padded pixels ─────────────────────────────
    out_nomask = criterion(persist_pred, true_disp, input_trace=trace)
    out_masked = criterion(persist_pred, true_disp, input_trace=trace, mask=mask)
    check("5. No mask → positive loss",  out_nomask["loss"].item() > 0, out_nomask["loss"].item())
    check("5. With mask → positive loss", out_masked["loss"].item() > 0, out_masked["loss"].item())

    # ── 6. Backward pass produces valid gradients ─────────────────────────
    pred_grad = true_disp + 0.3 * torch.randn_like(true_disp)
    pred_grad.requires_grad_(True)
    out = criterion(pred_grad, true_disp, input_trace=trace, mask=mask)
    out["loss"].backward()
    grad_ok = (
        pred_grad.grad is not None
        and not torch.isnan(pred_grad.grad).any()
        and not torch.isinf(pred_grad.grad).any()
    )
    check("6. Backward → non-None, non-NaN, non-Inf grads", grad_ok,
          pred_grad.grad.norm().item() if pred_grad.grad is not None else float("nan"))

    # ── 7. build_loss from dict → only 'loss' key returned ────────────────
    cfg_dict = {"decay_pixels": 8.0, "min_weight": 0.1}
    loss2    = build_loss(cfg_dict)
    out2     = loss2(pred_grad.detach(), true_disp, input_trace=trace)
    check("7. build_loss from dict → valid output",
          "loss" in out2 and not torch.isnan(out2["loss"]))
    check("7. build_loss → only 'loss' key in return dict",
          set(out2.keys()) == {"loss"})

    # build_loss must ignore legacy keys without crashing
    legacy_cfg = {
        "decay_pixels": 6.0, "min_weight": 0.05,
        "lambda_recon": 1.0, "lambda_focal": 0.3,
        "lambda_sign": 0.2, "decay_factor": 0.2,
    }
    loss_legacy = build_loss(legacy_cfg)
    out_legacy  = loss_legacy(pred_grad.detach(), true_disp, input_trace=trace)
    check("   build_loss ignores legacy config keys",
          "loss" in out_legacy and not torch.isnan(out_legacy["loss"]))

    print("\nSelf-test complete.")
