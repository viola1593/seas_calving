# tests/test_dataset_trend.py
"""
TDD tests for the SDT OLS trend channel in CalvingFrontDatasetV2.
Tests 1–10 match the spec in bundle_a_implementation_plan.md § Step 1.
All tests use K=5. All tests in this file should FAIL until Step 3 is
implemented in data/dataset_v2.py and data/datamodule.py.
"""

import json
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from data.dataset_v2 import CalvingFrontDatasetV2, collate_fn, GEOPHYS_IDX
from data.datamodule import CalvingFrontDataModule

# ── Constants ──────────────────────────────────────────────────────────────────

K_TEST           = 5
H_TEST, W_TEST   = 32, 32
N_OBS            = 12
TREND_CH_IDX     = 2                        # sdt=0, sdt_diff=1, sdt_trend=2
BASELINE_CHANNELS = 2 + len(GEOPHYS_IDX)   # sdt + sdt_diff + geophys = 7

CFG_NO_TREND = {"inputs": {"spatial": ["sdt", "geophys"]}}
CFG_TREND    = {"inputs": {"spatial": ["sdt", "sdt_trend", "geophys"]}}


# ── Synthetic-data helpers ─────────────────────────────────────────────────────

def _make_zarr(
    path: Path,
    sdt_fn=None,          # callable(i: int) -> np.ndarray (H, W), or None → zeros
    *,
    n_obs: int = N_OBS,
    h:     int = H_TEST,
    w:     int = W_TEST,
) -> None:
    import zarr
    from datetime import date, timedelta

    store = zarr.open_group(str(path), mode="w")
    store.create_dataset("sdt",     shape=(n_obs, h, w),    dtype="f4", fill_value=0.0)
    store.create_dataset("geophys", shape=(n_obs, 6, h, w), dtype="f4", fill_value=0.5)
    store.create_dataset("climate", shape=(n_obs, 2),        dtype="f4", fill_value=0.0)
    store.create_dataset("trace",   shape=(n_obs, h, w),     dtype=bool, fill_value=False)
    store.create_dataset("dates",   shape=(n_obs,),          dtype=str,  fill_value="")
    store.create_dataset("sensors", shape=(n_obs,),          dtype="u1", fill_value=0)
    for key in ("valid_sdt", "valid_img", "valid_clm", "valid_trace"):
        store.create_dataset(key, shape=(n_obs,), dtype=bool, fill_value=True)

    start = date(2018, 1, 1)
    for i in range(n_obs):
        store["sdt"][i] = (
            sdt_fn(i) if sdt_fn is not None
            else np.zeros((h, w), dtype=np.float32)
        )
        store["dates"][i] = (start + timedelta(days=i * 30)).isoformat()
        tr = np.zeros((h, w), dtype=bool)
        tr[h // 2, :] = True
        store["trace"][i] = tr

    store.attrs.update({
        "glacier_id":   "test", "T": n_obs, "H": h, "W": w,
        "resolution_m": 30.0,   "crs": "EPSG:3995", "sdt_clip_m": 2000.0,
    })


def _make_index(zarr_path: Path, index_path: Path) -> None:
    """Single sample: inputs = zarr[0 .. K_TEST-1], target = zarr[K_TEST]."""
    anchor_offsets = list(range(-(K_TEST - 1) * 30, 1, 30))   # [-120,-90,-60,-30,0]
    sample = {
        "glacier_id":         "test",
        "zarr_path":          str(zarr_path),
        "reference_date":     "2018-06-01",
        "horizon_days":       365,
        "input_zarr_indices": list(range(K_TEST)),
        "target_zarr_index":  K_TEST,
        "input_days":         [float(a) for a in anchor_offsets],
        "target_day":         365.0,
        "day_gaps":           [30.0] * (K_TEST - 1) + [0.0],
        "anchor_offsets":     anchor_offsets,
    }
    with open(index_path, "w") as f:
        json.dump([sample], f)


def _make_stats(path: Path, sdt_std: float = 600.0) -> None:
    stats = {
        "sdt":     {"mean": 0.0, "std": sdt_std},
        "geophys": {"mean": [0.0] * 6, "std": [1.0] * 6},
        "climate": {"mean": [0.0, 0.0], "std": [1.0, 1.0]},
    }
    with open(path, "w") as f:
        json.dump(stats, f)


# ── Module-scoped fixtures (standard retreating glacier) ──────────────────────

@pytest.fixture(scope="module")
def _tmp():
    d = tempfile.mkdtemp(prefix="svalcalving_trend_")
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture(scope="module")
def base_ds_files(_tmp):
    """Retreating-glacier zarr + index + stats, shared by tests that don't need custom SDT."""
    zarr_path  = _tmp / "base.zarr"
    index_path = _tmp / "base.json"
    stats_path = _tmp / "base_stats.json"

    def sdt_fn(i: int) -> np.ndarray:
        sdt = np.zeros((H_TEST, W_TEST), dtype=np.float32)
        front_row = min(H_TEST // 2 + i // 3, H_TEST - 5)
        for r in range(H_TEST):
            sdt[r, :] = np.clip((front_row - r) * 30.0, -2000.0, 2000.0)
        return sdt

    _make_zarr(zarr_path, sdt_fn=sdt_fn)
    _make_index(zarr_path, index_path)
    _make_stats(stats_path)
    return zarr_path, index_path, stats_path


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_default_no_trend_channel(base_ds_files):
    """1. Without sdt_trend in inputs.spatial, channel count equals baseline (7).
    Backward compatibility check."""
    _, index_path, stats_path = base_ds_files
    ds     = CalvingFrontDatasetV2(index_path, CFG_NO_TREND, stats_path)
    sample = ds[0]
    assert sample["inputs"].shape[1] == BASELINE_CHANNELS, (
        f"Expected {BASELINE_CHANNELS} channels without sdt_trend, "
        f"got {sample['inputs'].shape[1]}"
    )


def test_trend_channel_adds_one_channel(base_ds_files):
    """2. With sdt_trend in inputs.spatial, channel count = baseline + 1 (8)."""
    _, index_path, stats_path = base_ds_files
    ds     = CalvingFrontDatasetV2(index_path, CFG_TREND, stats_path)
    sample = ds[0]
    assert sample["inputs"].shape[1] == BASELINE_CHANNELS + 1, (
        f"Expected {BASELINE_CHANNELS + 1} channels with sdt_trend, "
        f"got {sample['inputs'].shape[1]}"
    )


def test_trend_channel_zero_for_static_sdt(tmp_path):
    """3. All K SDT maps identical → trend channel = 0 everywhere."""
    zarr_p = tmp_path / "static.zarr"
    idx_p  = tmp_path / "static.json"
    stat_p = tmp_path / "stats.json"

    # Non-trivial but constant-over-time SDT (so normalised values are also constant)
    base = np.linspace(-300.0, 300.0, H_TEST, dtype=np.float32)[:, None] * np.ones(
        (1, W_TEST), dtype=np.float32
    )
    _make_zarr(zarr_p, sdt_fn=lambda i: base.copy())
    _make_index(zarr_p, idx_p)
    _make_stats(stat_p)

    ds    = CalvingFrontDatasetV2(idx_p, CFG_TREND, stat_p)
    trend = ds[0]["inputs"][:, TREND_CH_IDX, :, :]   # (K, H, W)
    assert torch.allclose(trend, torch.zeros_like(trend), atol=1e-5), (
        "Trend channel should be exactly zero when SDT does not change across K slots"
    )


def test_trend_channel_sign_retreating(tmp_path):
    """4. SDT strictly decreases across K slots at all pixels → trend < 0 everywhere."""
    zarr_p = tmp_path / "retreat.zarr"
    idx_p  = tmp_path / "retreat.json"
    stat_p = tmp_path / "stats.json"

    delta = 50.0   # metres per time slot
    _make_zarr(zarr_p, sdt_fn=lambda i: np.full((H_TEST, W_TEST), -i * delta, dtype=np.float32))
    _make_index(zarr_p, idx_p)
    _make_stats(stat_p)

    trend = CalvingFrontDatasetV2(idx_p, CFG_TREND, stat_p)[0]["inputs"][:, TREND_CH_IDX]
    assert (trend < 0).all(), "Trend should be strictly negative when SDT decreases every slot"


def test_trend_channel_sign_advancing(tmp_path):
    """5. SDT strictly increases across K slots at all pixels → trend > 0 everywhere."""
    zarr_p = tmp_path / "advance.zarr"
    idx_p  = tmp_path / "advance.json"
    stat_p = tmp_path / "stats.json"

    delta = 50.0
    _make_zarr(zarr_p, sdt_fn=lambda i: np.full((H_TEST, W_TEST), i * delta, dtype=np.float32))
    _make_index(zarr_p, idx_p)
    _make_stats(stat_p)

    trend = CalvingFrontDatasetV2(idx_p, CFG_TREND, stat_p)[0]["inputs"][:, TREND_CH_IDX]
    assert (trend > 0).all(), "Trend should be strictly positive when SDT increases every slot"


def test_trend_channel_identical_across_slots(base_ds_files):
    """6. Trend is a static spatial map broadcast over time — same value at every K slot."""
    _, index_path, stats_path = base_ds_files
    ds    = CalvingFrontDatasetV2(index_path, CFG_TREND, stats_path)
    trend = ds[0]["inputs"][:, TREND_CH_IDX, :, :]   # (K, H, W)
    for k in range(1, K_TEST):
        assert torch.allclose(trend[k], trend[0], atol=1e-5), (
            f"Trend at slot {k} differs from slot 0; it should be the same static map"
        )


def test_trend_channel_no_nan(base_ds_files):
    """7. No NaN values in the trend channel across all pixels."""
    _, index_path, stats_path = base_ds_files
    ds    = CalvingFrontDatasetV2(index_path, CFG_TREND, stats_path)
    trend = ds[0]["inputs"][:, TREND_CH_IDX, :, :]
    assert not torch.isnan(trend).any(), "Trend channel must not contain NaN"


def test_trend_channel_flip_consistency(tmp_path):
    """8. trend(W-flipped SDT) equals W-flip of trend(SDT): OLS commutes with spatial flip.
    This verifies the existing _flip augmentation helper handles the trend channel correctly."""
    stat_p = tmp_path / "stats.json"
    _make_stats(stat_p)

    delta = 20.0   # left-to-right spatial gradient in metres

    def _sdt(i: int) -> np.ndarray:
        # Non-symmetric in W, varies over time (so trend != 0)
        sdt = (
            np.arange(W_TEST, dtype=np.float32)[None, :]
            * delta
            * np.ones((H_TEST, 1), dtype=np.float32)
        )
        sdt += i * 5.0
        return np.clip(sdt, -2000.0, 2000.0)

    zarr_n = tmp_path / "normal.zarr"
    zarr_f = tmp_path / "flipped.zarr"
    idx_n  = tmp_path / "idx_n.json"
    idx_f  = tmp_path / "idx_f.json"

    _make_zarr(zarr_n, sdt_fn=_sdt)
    _make_zarr(zarr_f, sdt_fn=lambda i: np.flip(_sdt(i), axis=-1).copy())
    _make_index(zarr_n, idx_n)
    _make_index(zarr_f, idx_f)

    ds_n = CalvingFrontDatasetV2(idx_n, CFG_TREND, stat_p, augment=False)
    ds_f = CalvingFrontDatasetV2(idx_f, CFG_TREND, stat_p, augment=False)

    trend_n = ds_n[0]["inputs"][:, TREND_CH_IDX].numpy()   # (K, H, W)
    trend_f = ds_f[0]["inputs"][:, TREND_CH_IDX].numpy()

    np.testing.assert_allclose(
        trend_f,
        np.flip(trend_n, axis=-1),
        atol=1e-5,
        err_msg="trend(W-flipped SDT) must equal W-flip of trend(SDT)",
    )


def test_batched_trend_channel_shape(tmp_path):
    """9. collate_fn pads inputs correctly for two glaciers with different spatial sizes."""
    stat_p = tmp_path / "stats.json"
    _make_stats(stat_p)

    H2, W2 = 24, 40   # second glacier is taller-and-wider than first

    zarr1 = tmp_path / "g1.zarr"; idx1 = tmp_path / "g1.json"
    zarr2 = tmp_path / "g2.zarr"; idx2 = tmp_path / "g2.json"

    _make_zarr(zarr1, h=H_TEST, w=W_TEST)
    _make_zarr(zarr2, h=H2,     w=W2)
    _make_index(zarr1, idx1)
    _make_index(zarr2, idx2)

    ds1 = CalvingFrontDatasetV2(idx1, CFG_TREND, stat_p, augment=False)
    ds2 = CalvingFrontDatasetV2(idx2, CFG_TREND, stat_p, augment=False)

    batch = collate_fn([ds1[0], ds2[0]])

    expected = (2, K_TEST, BASELINE_CHANNELS + 1, max(H_TEST, H2), max(W_TEST, W2))
    assert tuple(batch["inputs"].shape) == expected, (
        f"Expected batched inputs shape {expected}, got {tuple(batch['inputs'].shape)}"
    )


def test_datamodule_channel_count_with_trend(tmp_path):
    """10. CalvingFrontDataModule with sdt_trend in config reports n_input_channels = baseline + 1."""
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    stats_path = splits_dir / "normalisation_stats.json"
    _make_stats(stats_path)

    # sampling_params.json sets K without needing a full index file
    with open(splits_dir / "sampling_params.json", "w") as f:
        json.dump({"anchor_offsets": list(range(-(K_TEST - 1) * 30, 1, 30))}, f)

    dm = CalvingFrontDataModule(
        splits_dir  = splits_dir,
        config      = CFG_TREND,
        stats_path  = stats_path,
        num_workers = 0,
    )
    assert dm.n_input_channels == BASELINE_CHANNELS + 1, (
        f"Expected n_input_channels={BASELINE_CHANNELS + 1} with sdt_trend, "
        f"got {dm.n_input_channels}"
    )
