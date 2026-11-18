# tests/test_losses.py

import torch
import pytest
from losses import (
    CalvingLoss,
    build_loss,
    trace_proximity_weight,
)


def make_batch(B=2, H=64, W=64):
    """Flat glacier: positive upper half, negative lower half. Normalised units."""
    target = torch.zeros(B, 1, H, W)
    for r in range(H):
        target[:, :, r, :] = float(r - H // 2) * 0.05
    return target


def make_trace(B=2, H=64, W=64):
    """Single row of trace pixels at the zero-crossing (row H//2)."""
    trace = torch.zeros(B, 1, H, W)
    trace[:, :, H // 2, :] = 1.0
    return trace


# ── Test 1: perfect prediction gives near-zero loss ──────────────────────────

def test_zero_loss_on_perfect_prediction():
    # L1 component must be 0 on perfect prediction.
    # The sign auxiliary loss (BCE) is nonzero even at pred==target when
    # displacement at trace pixels is 0 (ambiguous sign), so we check
    # loss_l1 specifically.
    target = make_batch()
    trace  = make_trace()
    criterion = CalvingLoss()
    losses = criterion(target.clone(), target, input_trace=trace)
    assert losses["loss"].item() == pytest.approx(0.0, abs=1e-4)


# ── Test 2: loss is larger for worse predictions ──────────────────────────────

def test_loss_increases_with_error():
    target    = make_batch()
    trace     = make_trace()
    small_err = target + 0.1 * torch.randn_like(target)
    large_err = target + 1.0 * torch.randn_like(target)

    criterion  = CalvingLoss()
    loss_small = criterion(small_err, target, input_trace=trace)["loss"].item()
    loss_large = criterion(large_err, target, input_trace=trace)["loss"].item()

    assert loss_large > loss_small


# ── Test 3: trace proximity weight is highest at the front ───────────────────

def test_trace_weight_highest_at_front():
    trace = make_trace(B=1)
    w     = trace_proximity_weight(trace, decay_pixels=10.0, min_weight=0.05)

    # Front pixels: row 32 ± 0
    front_w    = w[0, 0, 32, :].mean().item()
    farfield_w = w[0, 0, 0:5, :].mean().item()

    assert front_w > farfield_w, \
        f"Front weight {front_w:.4f} should exceed far-field weight {farfield_w:.4f}"


# ── Test 4: fallback when trace is None emits warning ────────────────────────

def test_none_trace_fallback_warns():
    target = make_batch()
    criterion = CalvingLoss()
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        losses = criterion(target + 0.1, target, input_trace=None)
    assert len(caught) >= 1
    assert losses["loss"].item() > 0


# ── Test 5: API compatibility keys are zero tensors ──────────────────────────

def test_return_dict_has_expected_keys():
    target = make_batch()
    trace  = make_trace()
    losses = CalvingLoss()(target + 0.1, target, input_trace=trace)

    assert set(losses.keys()) == {"loss"}, \
        f"Expected keys {{'loss'}}, got {set(losses.keys())}"


# ── Test 6: backward pass propagates gradients ───────────────────────────────

def test_backward_pass():
    target = make_batch()
    trace  = make_trace()
    pred   = target + 0.3 * torch.randn_like(target)
    pred.requires_grad_(True)

    losses = CalvingLoss()(pred, target, input_trace=trace)
    losses["loss"].backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any(), "NaN in gradients"
    assert not torch.isinf(pred.grad).any(), "Inf in gradients"


# ── Test 7: spatial mask restricts loss to valid pixels ──────────────────────

def test_spatial_mask_excludes_padded_pixels():
    """Masked-out pixels must not contribute to the loss.

    Strategy: set pred == target everywhere EXCEPT in the right 32 columns
    (which are then masked out).  With the mask, loss should be near zero
    because all errors are in excluded pixels.  Without the mask, loss > 0.
    """
    B, C, H, W = 2, 1, 64, 64
    target = torch.zeros(B, C, H, W)
    trace  = torch.zeros(B, C, H, W)
    trace[:, :, H // 2, :] = 1.0           # calving front at mid-row

    # Large error only in the rightmost 32 columns
    pred = target.clone()
    pred[:, :, :, -32:] += 1.0

    mask_full    = torch.ones(B, C, H, W, dtype=torch.bool)
    mask_excluded = mask_full.clone()
    mask_excluded[:, :, :, -32:] = False    # exclude the erroneous columns

    criterion = CalvingLoss()
    loss_full     = criterion(pred, target, input_trace=trace, mask=mask_full)["loss"].item()
    loss_excluded = criterion(pred, target, input_trace=trace, mask=mask_excluded)["loss"].item()

    assert loss_full > 0.1,        f"Unmasked L1 loss should be large, got {loss_full:.4f}"
    assert loss_excluded < 1e-5,   f"Excluded-region L1 loss should be ~0, got {loss_excluded:.6f}"


# ── Test 8: build_loss from config dict ──────────────────────────────────────

def test_build_loss_from_config():
    cfg  = {"decay_pixels": 5.0, "min_weight": 0.0}
    loss = build_loss(cfg)
    assert isinstance(loss, CalvingLoss)
    assert loss.decay_pixels == pytest.approx(5.0)
    assert loss.min_weight   == pytest.approx(0.0)


def test_build_loss_ignores_legacy_keys():
    legacy_cfg = {
        "decay_pixels": 5.0, "min_weight": 0.0, "lambda_sign": 0.5,
        "lambda_recon": 1.0, "lambda_focal": 0.3,
        "decay_factor": 0.2,
    }
    loss = build_loss(legacy_cfg)
    target = make_batch()
    trace  = make_trace()
    out    = loss(target + 0.1, target, input_trace=trace)
    assert not torch.isnan(out["loss"])


# ── Test 9: output shapes are scalar ─────────────────────────────────────────

def test_output_is_scalar():
    target = make_batch()
    trace  = make_trace()
    losses = CalvingLoss()(target + 0.1, target, input_trace=trace)
    for k, v in losses.items():
        assert v.shape == torch.Size([]), \
            f"Loss component '{k}' has shape {v.shape}, expected scalar"


# ── Tests 11–20: CalvingMovementLoss and movement_weight ─────────────────────
# disp_std=1.0 throughout: tensor values are treated as displacement in metres
# directly, so test thresholds (150 m, 450 m, etc.) can be passed as-is.


def test_movement_weight_at_zero():
    from losses import movement_weight
    w = movement_weight(torch.zeros(1, 1, 8, 8), disp_std=1.0)
    assert torch.allclose(w, torch.ones_like(w))


def test_movement_weight_at_150m():
    from losses import movement_weight
    w = movement_weight(torch.full((1, 1, 8, 8), 150.0), disp_std=1.0)
    assert torch.allclose(w, torch.full_like(w, 2.0))


def test_movement_weight_clipped_at_450m():
    from losses import movement_weight
    w_450 = movement_weight(torch.full((1, 1, 4, 4),  450.0), disp_std=1.0)
    w_900 = movement_weight(torch.full((1, 1, 4, 4),  900.0), disp_std=1.0)
    expected = torch.full_like(w_450, 4.0)
    assert torch.allclose(w_450, expected)
    assert torch.allclose(w_900, expected)


def test_movement_weight_symmetry():
    from losses import movement_weight
    w_pos = movement_weight(torch.full((1, 1, 4, 4),  300.0), disp_std=1.0)
    w_neg = movement_weight(torch.full((1, 1, 4, 4), -300.0), disp_std=1.0)
    assert torch.allclose(w_pos, w_neg)


def test_movement_loss_zero_disp_matches_calving_loss():
    from losses import CalvingMovementLoss
    B, H, W = 2, 64, 64
    trace  = make_trace(B, H, W)
    target = torch.zeros(B, 1, H, W)
    pred   = target + 0.3

    loss_calving  = CalvingLoss()(pred, target, input_trace=trace)["loss"].item()
    loss_movement = CalvingMovementLoss(disp_std=1.0)(pred, target, input_trace=trace)["loss"].item()
    assert loss_calving == pytest.approx(loss_movement, rel=1e-5)


def test_movement_loss_upweights_fast_over_slow():
    from losses import CalvingMovementLoss
    B, H, W = 1, 64, 64
    trace_row = H // 2
    trace = torch.zeros(B, 1, H, W)
    trace[:, :, trace_row, :] = 1.0

    # Slow: zero displacement everywhere; prediction error only at the trace row
    slow_target = torch.zeros(B, 1, H, W)
    slow_pred   = slow_target.clone()
    slow_pred[:, :, trace_row, :] = 0.5

    # Fast: displacement clips to maximum movement weight at the trace row;
    # same prediction error magnitude there
    fast_target = torch.zeros(B, 1, H, W)
    fast_target[:, :, trace_row, :] = 450.0   # metres; w_move = 4.0
    fast_pred = fast_target.clone()
    fast_pred[:, :, trace_row, :] += 0.5

    criterion = CalvingMovementLoss(disp_std=1.0)
    loss_slow = criterion(slow_pred, slow_target, input_trace=trace)["loss"].item()
    loss_fast = criterion(fast_pred, fast_target, input_trace=trace)["loss"].item()
    assert loss_fast > loss_slow


def test_movement_loss_backward():
    from losses import CalvingMovementLoss
    trace  = make_trace()
    target = make_batch()
    pred   = (target + 0.2 * torch.randn_like(target)).requires_grad_(True)

    loss = CalvingMovementLoss(disp_std=1.0)(pred, target.detach(), input_trace=trace)["loss"]
    loss.backward()

    assert pred.grad is not None
    assert not torch.isnan(pred.grad).any()
    assert not torch.isinf(pred.grad).any()


def test_build_loss_movement_flag_returns_movement_loss():
    from losses import CalvingMovementLoss
    loss = build_loss({"movement_weight": True, "disp_std": 1.0})
    assert isinstance(loss, CalvingMovementLoss)


def test_build_loss_default_still_returns_calving_loss():
    loss = build_loss({})
    assert isinstance(loss, CalvingLoss)


# ── Tests: displacement_sample_scale and per-sample loss mode ────────────────
# disp_std=182.0 throughout; tensor values are normalised displacements.
# Tests 1a–1e check displacement_sample_scale; 1f–1g check per_sample=True.
# All will fail until Step 2 (losses.py) is implemented.


def test_sample_scale_at_zero_displacement():
    from losses import displacement_sample_scale
    target = torch.zeros(2, 1, 64, 64)
    scale  = displacement_sample_scale(target, disp_std=182.0)
    assert scale.shape == (2,)
    assert torch.allclose(scale, torch.ones(2))


def test_sample_scale_at_150m():
    from losses import displacement_sample_scale
    target = torch.full((2, 1, 64, 64), 150.0 / 182.0)
    scale  = displacement_sample_scale(target, disp_std=182.0)
    assert torch.allclose(scale, torch.full((2,), 2.0))


def test_sample_scale_clips_at_450m():
    from losses import displacement_sample_scale
    target = torch.full((2, 1, 64, 64), 900.0 / 182.0)
    scale  = displacement_sample_scale(target, disp_std=182.0)
    assert torch.allclose(scale, torch.full((2,), 4.0))


def test_sample_scale_is_per_sample():
    from losses import displacement_sample_scale
    target = torch.zeros(2, 1, 64, 64)
    target[1] = 450.0 / 182.0
    scale  = displacement_sample_scale(target, disp_std=182.0)
    assert scale[0].item() == pytest.approx(1.0, abs=1e-5)
    assert scale[1].item() == pytest.approx(4.0, abs=1e-5)


def test_sample_scale_respects_mask():
    from losses import displacement_sample_scale
    # Large displacement only in the right 32 columns, which are masked out.
    target = torch.zeros(2, 1, 64, 64)
    target[:, :, :, 32:] = 450.0 / 182.0
    mask = torch.ones(2, 1, 64, 64, dtype=torch.bool)
    mask[:, :, :, 32:] = False
    scale = displacement_sample_scale(target, disp_std=182.0, mask=mask)
    assert torch.allclose(scale, torch.ones(2), atol=1e-5)


def test_per_sample_loss_shape():
    B = 2
    target = make_batch(B)
    trace  = make_trace(B)
    pred   = target + 0.3
    loss_dict = CalvingLoss()(pred, target, input_trace=trace, per_sample=True)
    assert loss_dict["loss"].shape == torch.Size([B])
    assert (loss_dict["loss"] > 0).all()


def test_scaled_loss_upweights_fast_mover():
    from losses import displacement_sample_scale
    B, H, W = 2, 64, 64
    trace  = make_trace(B, H, W)
    target = torch.zeros(B, 1, H, W)
    target[1] = 450.0 / 182.0   # sample 1 is fast-moving
    pred   = target + 0.3        # equal prediction error for both samples

    per_sample_losses = CalvingLoss()(pred, target, input_trace=trace, per_sample=True)["loss"]
    scales     = displacement_sample_scale(target, disp_std=182.0)
    final_loss   = (per_sample_losses * scales).mean()
    unscaled_loss = per_sample_losses.mean()
    assert final_loss > unscaled_loss


def test_movement_loss_output_keys():
    from losses import CalvingMovementLoss
    trace  = make_trace()
    target = make_batch()
    out = CalvingMovementLoss(disp_std=1.0)(target + 0.1, target, input_trace=trace)
    assert set(out.keys()) == {"loss"}
