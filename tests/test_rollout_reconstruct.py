# tests/test_rollout_reconstruct.py
"""
Unit tests for the reconstruction math of rollout_film.py (section 2b).
"""

import numpy as np
import pytest

from rollout_film import reconstruct_pred_sdt_norm, denorm_sdt, norm_sdt


# ── reconstruct_pred_sdt_norm ─────────────────────────────────────────────

def test_reconstruct_matches_formula():
    rng = np.random.default_rng(1)
    disp = rng.standard_normal((8, 8)).astype(np.float32)
    last = rng.standard_normal((8, 8)).astype(np.float32)
    disp_scale = 6.8

    got = reconstruct_pred_sdt_norm(disp, last, disp_scale)
    expected = disp / disp_scale + last     # evaluate_film.py:209

    np.testing.assert_allclose(got, expected, rtol=1e-6)


def test_reconstruct_zero_displacement_is_persistence():
    last = np.linspace(-1, 1, 16, dtype=np.float32).reshape(4, 4)
    got = reconstruct_pred_sdt_norm(np.zeros_like(last), last, disp_scale=6.8)
    np.testing.assert_allclose(got, last, rtol=1e-6)


def test_reconstruct_round_trip_known_displacement():
    # If displacement is defined as (target - last) * disp_scale, reconstruction
    # must recover the target exactly.
    rng = np.random.default_rng(2)
    last = rng.standard_normal((5, 5)).astype(np.float32)
    target = rng.standard_normal((5, 5)).astype(np.float32)
    disp_scale = 6.8

    disp = (target - last) * disp_scale
    recovered = reconstruct_pred_sdt_norm(disp, last, disp_scale)
    np.testing.assert_allclose(recovered, target, rtol=1e-5)


def test_reconstruct_linearity_in_displacement():
    rng = np.random.default_rng(3)
    last = rng.standard_normal((4, 4)).astype(np.float32)
    d1 = rng.standard_normal((4, 4)).astype(np.float32)
    d2 = rng.standard_normal((4, 4)).astype(np.float32)
    s = 6.8

    # reconstruct(a) + reconstruct(b) - last == reconstruct(a + b)
    lhs = reconstruct_pred_sdt_norm(d1, last, s) + reconstruct_pred_sdt_norm(d2, last, s) - last
    rhs = reconstruct_pred_sdt_norm(d1 + d2, last, s)
    np.testing.assert_allclose(lhs, rhs, rtol=1e-5)


# ── denorm / norm inverse pair ────────────────────────────────────────────

def test_denorm_sdt_formula():
    x = np.array([-1.0, 0.0, 2.0], dtype=np.float32)
    got = denorm_sdt(x, sdt_mean=-50.0, sdt_std=600.0)
    np.testing.assert_allclose(got, x * 600.0 - 50.0, rtol=1e-6)


def test_norm_sdt_formula():
    x = np.array([-650.0, -50.0, 1150.0], dtype=np.float32)
    got = norm_sdt(x, sdt_mean=-50.0, sdt_std=600.0)
    np.testing.assert_allclose(got, (x + 50.0) / 600.0, rtol=1e-6)


def test_norm_denorm_are_inverses():
    rng = np.random.default_rng(4)
    mean, std = -50.0, 600.0
    x_norm = rng.standard_normal((6, 6)).astype(np.float32)

    np.testing.assert_allclose(
        norm_sdt(denorm_sdt(x_norm, mean, std), mean, std), x_norm, rtol=1e-5
    )

    x_m = rng.standard_normal((6, 6)).astype(np.float32) * std + mean
    np.testing.assert_allclose(
        denorm_sdt(norm_sdt(x_m, mean, std), mean, std), x_m, rtol=1e-5
    )
