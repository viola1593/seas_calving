# tests/test_rollout_zone.py
"""
Unit tests for the eval-zone functions of rollout_film.py (section 2c).
"""

import numpy as np
import pytest

from rollout_film import compute_active_mask, persistence_sdt_m, reaches_full_depth


# ── compute_active_mask ───────────────────────────────────────────────────

def test_active_mask_excludes_static_keeps_moving():
    K, H, W = 5, 2, 2
    stack = np.zeros((K, H, W), dtype=np.float32)

    # pixel (0,0): perfectly static rock -> var 0 -> excluded
    stack[:, 0, 0] = 5.0
    # pixel (0,1): tiny wobble, var well below 100 m^2 -> excluded
    stack[:, 0, 1] = [0.0, 1.0, 0.0, 1.0, 0.0]
    # pixel (1,0): large front migration -> var >> 100 m^2 -> kept
    stack[:, 1, 0] = [0.0, 100.0, 200.0, 300.0, 400.0]
    # pixel (1,1): large migration -> kept
    stack[:, 1, 1] = [-200.0, -100.0, 0.0, 100.0, 200.0]

    mask = compute_active_mask(stack, threshold_m2=100.0)

    assert mask.shape == (H, W)
    assert mask.dtype == bool
    assert mask[0, 0] == False
    assert mask[0, 1] == False
    assert mask[1, 0] == True
    assert mask[1, 1] == True


def test_active_mask_matches_variance_threshold_formula():
    rng = np.random.default_rng(5)
    stack = rng.standard_normal((5, 8, 8)).astype(np.float32) * 50.0
    thr = 100.0
    got = compute_active_mask(stack, threshold_m2=thr)
    expected = np.var(stack, axis=0) > thr
    np.testing.assert_array_equal(got, expected)


def test_active_mask_default_threshold_is_100():
    stack = np.zeros((3, 1, 1), dtype=np.float32)
    stack[:, 0, 0] = [0.0, 15.0, 30.0]      # var = 150 > 100
    assert compute_active_mask(stack)[0, 0] == True


# ── persistence_sdt_m ─────────────────────────────────────────────────────

def test_persistence_is_identity():
    rng = np.random.default_rng(6)
    t0 = rng.standard_normal((7, 7)).astype(np.float32) * 300.0
    got = persistence_sdt_m(t0)
    np.testing.assert_array_equal(got, t0)


# ── reaches_full_depth ────────────────────────────────────────────────────

def test_reaches_full_depth_true_only_when_equal_K():
    assert reaches_full_depth(5, 5) is True
    assert reaches_full_depth(4, 5) is False
    assert reaches_full_depth(0, 5) is False
    assert reaches_full_depth(6, 5) is False