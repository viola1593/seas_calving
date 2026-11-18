# tests/test_metrics.py

import numpy as np
import pytest
from metrics import (
    compute_metrics,
    compute_skill_score,
    build_evaluation_zone,
    extract_predicted_front,
    trace_to_points,
    denormalise_sdt,
)

# ── Fixtures: synthetic glacier geometry ──────────────────────────────────────

def make_flat_glacier(H=128, W=128, front_row=64, resolution_m=30.0):
    """
    Glacier occupies rows 0..front_row (positive SDT),
    ocean occupies rows front_row+1..H (negative SDT).
    Front is a straight horizontal line at row front_row.
    Values in metres.
    """
    sdt = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        dist = (front_row - r) * resolution_m   # + = glacier, - = ocean
        sdt[r, :] = np.clip(dist, -2000, 2000)

    trace = np.zeros((H, W), dtype=bool)
    trace[front_row, :] = True

    return sdt, trace


# ── Test 1: perfect prediction gives zero distance errors ─────────────────────

def test_perfect_prediction_zero_mde():
    sdt, trace = make_flat_glacier()
    m = compute_metrics(sdt, sdt, trace, resolution_m=30.0)

    assert m["mde"]       == pytest.approx(0.0, abs=1.0)
    assert m["hausdorff"] == pytest.approx(0.0, abs=1.0)
    assert m["mae_front"] == pytest.approx(0.0, abs=1.0)
    assert m["advance_retreat_bias"] == pytest.approx(0.0, abs=1.0)


# ── Test 2: known shift gives known MDE ───────────────────────────────────────

def test_known_shift_gives_known_mde():
    res        = 30.0
    shift_px   = 4          # 4 pixels = 120m
    sdt, trace = make_flat_glacier(front_row=64, resolution_m=res)

    pred_sdt = np.roll(sdt, shift_px, axis=0).astype(np.float32)

    m = compute_metrics(pred_sdt, sdt, trace, resolution_m=res)

    expected_mde = shift_px * res   # 120m
    assert m["mde"] == pytest.approx(expected_mde, rel=0.05)
    assert m["advance_retreat_bias"] > 0


# ── Test 3: retreat gives negative bias ───────────────────────────────────────

def test_retreat_gives_negative_bias():
    res = 30.0
    sdt, trace = make_flat_glacier(front_row=64, resolution_m=res)
    pred_sdt = np.roll(sdt, -4, axis=0).astype(np.float32)

    m = compute_metrics(pred_sdt, sdt, trace, resolution_m=res)
    assert m["advance_retreat_bias"] < 0


# ── Test 4: area error is correct for a known shifted polygon ─────────────────

def test_area_error_known_value():
    res      = 50.0
    H, W     = 128, 128
    shift_px = 2   # 2 pixels = 100m advance
    sdt, trace = make_flat_glacier(H=H, W=W, front_row=64, resolution_m=res)

    # Adding shift_px * res moves the zero-crossing shift_px pixels seaward.
    pred_sdt = np.clip(sdt + shift_px * res, -2000, 2000).astype(np.float32)

    m = compute_metrics(pred_sdt, sdt, trace, resolution_m=res)

    # The symmetric difference is a strip shift_px pixels wide × W pixels wide,
    # entirely within the front band (default 500m >> 100m shift).
    expected_area_km2 = shift_px * W * (res / 1000.0) ** 2
    assert m["area_error_km2"] == pytest.approx(expected_area_km2, rel=0.05)


# ── Test 5: skill score properties ────────────────────────────────────────────

def test_skill_score_perfect():
    assert compute_skill_score(0.0, 100.0) == pytest.approx(1.0)

def test_skill_score_equal_persistence():
    assert compute_skill_score(100.0, 100.0) == pytest.approx(0.0)

def test_skill_score_worse_than_persistence():
    assert compute_skill_score(200.0, 100.0) == pytest.approx(-1.0)

def test_skill_score_nan_propagation():
    assert np.isnan(compute_skill_score(float("nan"), 100.0))
    assert np.isnan(compute_skill_score(50.0, float("nan")))
    assert np.isnan(compute_skill_score(50.0, 0.0))


# ── Test 6: None trace returns all NaN ───────────────────────────────────────

def test_none_trace_returns_all_nan():
    sdt, _ = make_flat_glacier()
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        m = compute_metrics(sdt, sdt, target_trace=None)
    for k, v in m.items():
        assert np.isnan(v), f"Expected NaN for {k} when trace is None, got {v}"


# ── Test 7: evaluation zone is a band around the front only ──────────────────

def test_evaluation_zone_is_front_band():
    H, W   = 64, 64
    res    = 30.0
    front_band_m = 300.0
    front_band_px = int(round(front_band_m / res))   # 10 pixels

    # Signed distance (metres) to a horizontal front at row 32.
    target_sdt = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        target_sdt[r, :] = (r - 32) * res

    zone = build_evaluation_zone(target_sdt, front_band_m=front_band_m)

    # Zone must include the front row
    assert zone[32, :].all(), "Front row must be inside the evaluation zone"

    # Pixels beyond front_band_px from the front must be outside the zone
    far_row = 32 + front_band_px + 2
    if far_row < H:
        assert not zone[far_row, :].any(), \
            f"Row {far_row} is beyond front band and should be excluded"


# ── Test 8: denormalise is the inverse of normalise ───────────────────────────

def test_denormalise_roundtrip():
    stats = {"sdt": {"mean": -50.0, "std": 600.0}}
    original  = np.random.randn(64, 64).astype(np.float32) * 600 - 50
    normalised = (original - (-50.0)) / 600.0
    recovered  = denormalise_sdt(normalised, stats)
    np.testing.assert_allclose(recovered, original, rtol=1e-5)
