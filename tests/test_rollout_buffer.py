# tests/test_rollout_buffer.py
"""
Unit tests for the buffer & date-matching functions of rollout_film.py (section 2a).
"""

from datetime import date, timedelta

import numpy as np
import pytest

from rollout_film import (
    frame_dates,
    recompute_sdt_diff,
    PredictionBuffer,
    substitute_sdt,
    FrameProvenance,
    autoreg_depth,
)


# ── frame_dates ───────────────────────────────────────────────────────────

def test_frame_dates_matches_t_ref_plus_input_days():
    input_days = [-360.0, -270.0, -180.0, -90.0, 0.0]
    sample = {
        "target_date": "2020-06-01",
        "horizon_days": 90,
        "input_days": input_days,
    }
    t_ref = date(2020, 6, 1) - timedelta(days=90)   # 2020-03-03

    fds = frame_dates(sample)

    assert len(fds) == len(input_days)
    assert all(isinstance(d, date) for d in fds)
    for k, off in enumerate(input_days):
        assert fds[k] == t_ref + timedelta(days=int(off))
    # Last (anchor) frame sits exactly at t_ref for the seasonal task.
    assert fds[-1] == t_ref


# ── recompute_sdt_diff ────────────────────────────────────────────────────

def test_recompute_sdt_diff_is_np_diff_prepend():
    rng = np.random.default_rng(0)
    sdt = rng.standard_normal((5, 8, 8)).astype(np.float32)

    got = recompute_sdt_diff(sdt)
    expected = np.diff(sdt, axis=0, prepend=sdt[[0]])

    assert got.shape == sdt.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6)
    # Frame 0's diff is defined as zero (prepend of itself).
    np.testing.assert_allclose(got[0], np.zeros_like(sdt[0]), atol=1e-6)


# ── PredictionBuffer.match ────────────────────────────────────────────────

def _arr(val, shape=(4, 4)):
    return np.full(shape, float(val), dtype=np.float32)


def test_match_empty_buffer_returns_none():
    buf = PredictionBuffer()
    assert buf.match(date(2020, 6, 15), delta_days=30) is None


def test_match_picks_nearest_within_delta():
    buf = PredictionBuffer()
    buf.add(date(2020, 6, 10), _arr(10))   # |Δ| = 5
    buf.add(date(2020, 6, 25), _arr(25))   # |Δ| = 10

    res = buf.match(date(2020, 6, 15), delta_days=20)
    assert res is not None
    key, arr = res
    assert key == date(2020, 6, 10)
    np.testing.assert_array_equal(arr, _arr(10))


def test_match_returns_none_outside_tolerance():
    buf = PredictionBuffer()
    buf.add(date(2020, 7, 30), _arr(1))    # |Δ| = 45 days
    assert buf.match(date(2020, 6, 15), delta_days=10) is None


def test_match_tie_break_prefers_later_key_date():
    buf = PredictionBuffer()
    buf.add(date(2020, 6, 10), _arr(10))   # |Δ| = 5 (earlier)
    buf.add(date(2020, 6, 20), _arr(20))   # |Δ| = 5 (later)  <-- winner

    res = buf.match(date(2020, 6, 15), delta_days=10)
    assert res is not None
    key, arr = res
    assert key == date(2020, 6, 20)
    np.testing.assert_array_equal(arr, _arr(20))


def test_match_entries_are_reused_not_consumed():
    buf = PredictionBuffer()
    buf.add(date(2020, 6, 10), _arr(10))

    first = buf.match(date(2020, 6, 12), delta_days=30)
    second = buf.match(date(2020, 6, 12), delta_days=30)
    assert first is not None and second is not None
    assert first[0] == second[0] == date(2020, 6, 10)


# ── substitute_sdt ────────────────────────────────────────────────────────

def test_substitute_overwrites_only_matched_frames_and_copies():
    K, H, W = 3, 4, 4
    real = np.stack([_arr(k, (H, W)) for k in range(K)])   # frame k == constant k
    fds = [date(2020, 3, 3), date(2020, 4, 2), date(2020, 5, 2)]

    buf = PredictionBuffer()
    buf.add(date(2020, 4, 2), _arr(99, (H, W)))   # matches frame 1 exactly

    out, prov = substitute_sdt(real, fds, buf, delta_days=30)

    # returns a copy; original untouched
    assert out is not real
    np.testing.assert_array_equal(real[1], _arr(1, (H, W)))
    # only frame 1 overwritten
    np.testing.assert_array_equal(out[0], _arr(0, (H, W)))
    np.testing.assert_array_equal(out[1], _arr(99, (H, W)))
    np.testing.assert_array_equal(out[2], _arr(2, (H, W)))

    # provenance
    assert len(prov) == K
    assert prov[0].synthetic is False
    assert prov[0].borrowed_date is None
    assert prov[0].day_offset is None
    assert prov[1].synthetic is True
    assert prov[1].borrowed_date == date(2020, 4, 2)
    assert prov[1].day_offset == 0
    assert prov[2].synthetic is False


def test_substitute_day_offset_is_signed_frame_minus_borrowed():
    K, H, W = 2, 3, 3
    real = np.stack([_arr(0, (H, W)), _arr(1, (H, W))])
    frame = date(2020, 6, 15)
    borrowed = date(2020, 6, 10)           # frame is 5 days AFTER borrowed key
    fds = [date(2019, 1, 1), frame]        # frame 0 far from any key

    buf = PredictionBuffer()
    buf.add(borrowed, _arr(7, (H, W)))

    _, prov = substitute_sdt(real, fds, buf, delta_days=30)

    assert prov[0].synthetic is False
    assert prov[1].synthetic is True
    assert prov[1].borrowed_date == borrowed
    assert prov[1].day_offset == (frame - borrowed).days == 5


def test_substitute_one_prediction_reused_across_multiple_frames():
    K, H, W = 3, 3, 3
    real = np.stack([_arr(k, (H, W)) for k in range(K)])
    # two distinct frames both within tol of the single buffered key
    fds = [date(2020, 6, 8), date(2020, 6, 12), date(2019, 1, 1)]

    buf = PredictionBuffer()
    buf.add(date(2020, 6, 10), _arr(42, (H, W)))

    out, prov = substitute_sdt(real, fds, buf, delta_days=30)

    np.testing.assert_array_equal(out[0], _arr(42, (H, W)))
    np.testing.assert_array_equal(out[1], _arr(42, (H, W)))
    np.testing.assert_array_equal(out[2], _arr(2, (H, W)))   # frame 2 stays real
    assert [p.synthetic for p in prov] == [True, True, False]


# ── autoreg_depth ─────────────────────────────────────────────────────────

def test_autoreg_depth_counts_only_synthetic_frames():
    prov = [
        FrameProvenance(synthetic=False, borrowed_date=None, day_offset=None),
        FrameProvenance(synthetic=True, borrowed_date=date(2020, 1, 1), day_offset=2),
        FrameProvenance(synthetic=True, borrowed_date=date(2020, 2, 1), day_offset=-3),
        FrameProvenance(synthetic=False, borrowed_date=None, day_offset=None),
    ]
    assert autoreg_depth(prov) == 2


def test_autoreg_depth_zero_when_all_real():
    prov = [FrameProvenance(False, None, None) for _ in range(5)]
    assert autoreg_depth(prov) == 0


def test_autoreg_depth_full_when_all_synthetic():
    prov = [FrameProvenance(True, date(2020, 1, 1), 0) for _ in range(5)]
    assert autoreg_depth(prov) == 5
