# tests/test_integration_v2.py

import json
import shutil
import tempfile
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pytest
import torch
import zarr
from torch.utils.data import DataLoader

from data.dataset_v2 import CalvingFrontDatasetV2, collate_fn
from losses import CalvingLoss
from metrics import compute_metrics, denormalise_sdt


# ── Dataset config used throughout ───────────────────────────────────────────

DATASET_CONFIG = {
    "inputs":   {"spatial": ["sdt", "geophys"], "scalar": ["climate"]},
    "forcings": {"spatial": [],                 "scalar": []},
}

# K=4, C_in = 1(sdt) + 1(sdt_diff) + 5(geophys) = 7
K       = 4
C_TOTAL = 7


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_dir():
    d = tempfile.mkdtemp(prefix="svalcalving_v2_test_")
    yield Path(d)
    shutil.rmtree(d)


def make_synthetic_zarr(
    zarr_path:    Path,
    glacier_id:   str,
    n_obs:        int   = 20,
    H:            int   = 64,
    W:            int   = 64,
    resolution_m: float = 30.0,
) -> list[str]:
    store = zarr.open_group(str(zarr_path), mode="w")

    store.create_dataset("sdt",         shape=(n_obs, H, W),    dtype="f4",  fill_value=0.0)
    store.create_dataset("imagery",     shape=(n_obs, 5, H, W), dtype="f4",  fill_value=0.0)
    store.create_dataset("geophys",     shape=(n_obs, 6, H, W), dtype="f4",  fill_value=0.0)
    store.create_dataset("climate",     shape=(n_obs, 2),        dtype="f4",  fill_value=0.0)
    store.create_dataset("trace",       shape=(n_obs, H, W),    dtype=bool,   fill_value=False)
    store.create_dataset("dates",       shape=(n_obs,),          dtype=str,    fill_value="")
    store.create_dataset("sensors",     shape=(n_obs,),          dtype="u1",   fill_value=0)
    store.create_dataset("valid_sdt",   shape=(n_obs,),          dtype=bool,   fill_value=False)
    store.create_dataset("valid_img",   shape=(n_obs,),          dtype=bool,   fill_value=False)
    store.create_dataset("valid_clm",   shape=(n_obs,),          dtype=bool,   fill_value=False)
    store.create_dataset("valid_trace", shape=(n_obs,),          dtype=bool,   fill_value=False)

    start_date = date(2018, 1, 1)
    dates_written = []

    for i in range(n_obs):
        front_row = H // 2 + (i // 5)
        front_row = min(front_row, H - 5)

        sdt = np.zeros((H, W), dtype=np.float32)
        for r in range(H):
            sdt[r, :] = np.clip((front_row - r) * resolution_m, -2000, 2000)
        store["sdt"][i]       = sdt
        store["valid_sdt"][i] = True

        store["imagery"][i]   = np.random.rand(5, H, W).astype(np.float32)
        store["valid_img"][i] = True

        store["geophys"][i]   = np.random.rand(6, H, W).astype(np.float32)

        store["climate"][i]   = np.array([1.5, -1.2], dtype=np.float32)
        store["valid_clm"][i] = True

        trace = np.zeros((H, W), dtype=bool)
        trace[front_row, :] = True
        store["trace"][i]       = trace
        store["valid_trace"][i] = True

        obs_date = start_date + timedelta(days=i * 10)
        store["dates"][i]   = obs_date.isoformat()
        store["sensors"][i] = 0

        dates_written.append(obs_date.isoformat())

    store.attrs.update({
        "glacier_id":   glacier_id,
        "T":            n_obs,
        "H":            H,
        "W":            W,
        "resolution_m": resolution_m,
        "crs":          "EPSG:3995",
        "sdt_clip_m":   2000.0,
    })
    return dates_written


def make_synthetic_index(zarr_path: Path, index_path: Path, n_obs: int = 20) -> None:
    anchors = [-40, -30, -20, -10]   # K=4
    samples = []

    for ref in range(K, n_obs - 1):
        input_indices = list(range(ref - K, ref))
        target_index  = ref + 1
        if target_index >= n_obs:
            continue

        ref_day    = ref * 10
        input_days = [(i * 10 - ref_day) for i in input_indices]
        target_day = float(target_index * 10 - ref_day)

        samples.append({
            "glacier_id":         "test_glacier",
            "zarr_path":          str(zarr_path),
            "reference_date":     f"2018-{(ref+1):02d}-01",
            "horizon_days":       10,
            "input_zarr_indices": input_indices,
            "target_zarr_index":  target_index,
            "input_days":         input_days,
            "target_day":         target_day,
            "day_gaps":           [d - a for d, a in zip(input_days, anchors)],
            "anchor_offsets":     anchors,
        })

    with open(index_path, "w") as f:
        json.dump(samples, f)


def make_synthetic_stats(stats_path: Path) -> None:
    stats = {
        "sdt":     {"mean": 0.0,   "std": 600.0},
        "imagery": {"mean": [0.1, 0.15, 0.12, 0.3, 0.08],
                    "std":  [0.05, 0.06, 0.06, 0.12, 0.04]},
        "geophys": {"mean": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "std":  [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]},
        "climate": {"mean": [1.5, -1.2], "std": [0.5, 0.3]},
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f)


# ── Integration tests ─────────────────────────────────────────────────────────

class TestIntegrationV2:

    @pytest.fixture(autouse=True)
    def setup(self, tmp_dir):
        self.zarr_path  = tmp_dir / "test_glacier.zarr"
        self.index_path = tmp_dir / "train.json"
        self.stats_path = tmp_dir / "normalisation_stats.json"

        make_synthetic_zarr(self.zarr_path, "test_glacier", n_obs=20)
        make_synthetic_index(self.zarr_path, self.index_path)
        make_synthetic_stats(self.stats_path)

    # ── Dataset shape test ────────────────────────────────────────────────

    def test_dataset_item_shapes(self):
        ds     = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        sample = ds[0]

        # inputs: (K, C_total, H, W)  K=4, C_total=1+1+5=7 (sdt, sdt_diff, geophys)
        assert sample["inputs"].shape             == (K, C_TOTAL, 64, 64)
        assert sample["target_displacement"].shape == (1, 64, 64)
        assert sample["climate"].shape             == (K, 2)
        assert sample["tau"].shape                 == (K, 4)
        assert sample["valid_clm"].shape           == (K,)
        assert sample["input_trace"].shape         == (1, 64, 64)
        assert sample["target_trace"].shape        == (1, 64, 64)
        assert sample["forcing_spatial"].shape[0]  == 0   # no spatial forcings configured

    # ── Dataset dtype test ────────────────────────────────────────────────

    def test_dataset_item_dtypes(self):
        ds     = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        sample = ds[0]

        assert sample["inputs"].dtype              == torch.float32
        assert sample["target_displacement"].dtype == torch.float32
        assert sample["climate"].dtype             == torch.float32
        assert sample["tau"].dtype                 == torch.float32

    # ── DataLoader batching test ──────────────────────────────────────────

    def test_dataloader_batching(self):
        ds     = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        loader = DataLoader(ds, batch_size=2, num_workers=0, shuffle=False,
                            collate_fn=collate_fn)
        batch  = next(iter(loader))

        assert batch["inputs"].shape              == (2, K, C_TOTAL, 64, 64)
        assert batch["target_displacement"].shape  == (2, 1, 64, 64)
        assert batch["climate"].shape              == (2, K, 2)
        assert batch["tau"].shape                  == (2, K, 4)

    # ── No NaN in normalised outputs ──────────────────────────────────────

    def test_no_nan_in_dataset_output(self):
        ds = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        for i in range(min(5, len(ds))):
            sample = ds[i]
            for k, v in sample.items():
                if isinstance(v, torch.Tensor):
                    assert not torch.isnan(v).any(), \
                        f"NaN in sample[{i}]['{k}']"

    # ── Loss accepts dataset output ───────────────────────────────────────

    def test_loss_on_dataset_output(self):
        ds        = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        loader    = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_fn)
        batch     = next(iter(loader))
        criterion = CalvingLoss()

        pred   = batch["target_displacement"] + 0.1 * torch.randn_like(batch["target_displacement"])
        losses = criterion(pred, batch["target_displacement"],
                           input_trace=batch["input_trace"])

        assert not torch.isnan(losses["loss"]), "Loss is NaN"
        assert losses["loss"].item() >= 0.0,    "Loss is negative"

    # ── Loss backward on dataset output ──────────────────────────────────

    def test_loss_backward_on_dataset_output(self):
        ds     = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_fn)
        batch  = next(iter(loader))

        pred = batch["target_displacement"] + 0.1 * torch.randn_like(batch["target_displacement"])
        pred.requires_grad_(True)

        losses = CalvingLoss()(pred, batch["target_displacement"],
                               input_trace=batch["input_trace"])
        losses["loss"].backward()

        assert pred.grad is not None
        assert not torch.isnan(pred.grad).any()

    # ── Metrics on denormalised dataset output ────────────────────────────

    def test_metrics_on_dataset_output(self):
        ds     = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        sample = ds[0]

        with open(self.stats_path) as f:
            stats = json.load(f)

        # Reconstruct absolute target SDT: last_input_sdt + displacement
        # disp_scale = 1.0 when displacement key is absent from stats
        last_sdt_norm  = sample["inputs"][-1, 0].numpy()          # (H, W) normalised
        disp_norm      = sample["target_displacement"][0].numpy()  # (H, W) normalised
        tgt_sdt_norm   = last_sdt_norm + disp_norm
        target_m       = denormalise_sdt(tgt_sdt_norm, stats)

        ti    = sample["target_zarr_index"]
        store = zarr.open_group(str(self.zarr_path), mode="r")
        trace = store["trace"][ti].astype(bool)

        m = compute_metrics(target_m, target_m, trace, resolution_m=30.0)

        assert not np.isnan(m["mae_front"]), "mae_front is NaN"
        assert not np.isnan(m["mde"]),       "mde is NaN"
        assert m["mae_front"] == pytest.approx(0.0, abs=1.0)
        assert m["mde"]       == pytest.approx(0.0, abs=1.0)

    # ── Full pipeline: dataset → dummy model → loss → metrics ─────────────

    def test_full_pipeline_dummy_model(self):
        ds     = CalvingFrontDatasetV2(self.index_path, DATASET_CONFIG, self.stats_path)
        loader = DataLoader(ds, batch_size=2, num_workers=0, collate_fn=collate_fn)
        batch  = next(iter(loader))

        class PerfectModel(torch.nn.Module):
            def forward(self, batch):
                return batch["target_displacement"]

        class ZeroModel(torch.nn.Module):
            def forward(self, batch):
                return torch.zeros_like(batch["target_displacement"])

        with open(self.stats_path) as f:
            stats = json.load(f)

        criterion = CalvingLoss()

        for model_name, model in [("perfect", PerfectModel()),
                                   ("zero",    ZeroModel())]:
            pred   = model(batch)
            losses = criterion(pred, batch["target_displacement"],
                               input_trace=batch["input_trace"])

            assert not torch.isnan(losses["loss"]), \
                f"{model_name} model produced NaN loss"

            # Reconstruct absolute SDT for the first batch item
            last_sdt_norm = batch["inputs"][0, -1, 0].detach().numpy()   # (H, W)
            pred_disp     = pred[0, 0].detach().numpy()
            tgt_disp      = batch["target_displacement"][0, 0].numpy()

            pred_m   = denormalise_sdt(last_sdt_norm + pred_disp, stats)
            target_m = denormalise_sdt(last_sdt_norm + tgt_disp,  stats)

            store = zarr.open_group(str(self.zarr_path), mode="r")
            ti    = batch["target_zarr_index"][0]   # list in V2 collate_fn
            trace = store["trace"][ti].astype(bool)

            m = compute_metrics(pred_m, target_m, trace, resolution_m=30.0)

            print(f"\n{model_name} model metrics:")
            for k, v in m.items():
                print(f"  {k:30s} = {v:.3f}")

            # active_mde / active_hausdorff require active_mask=...; NaN here is expected
            optional_nan = {"active_mde", "active_hausdorff"}
            assert all(not np.isnan(v) for k, v in m.items() if k not in optional_nan), \
                f"{model_name} model produced NaN metric: {m}"