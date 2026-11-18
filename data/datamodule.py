# data/datamodule.py
"""
CalvingFrontDataModule — Lightning DataModule for calving front prediction.

Wraps CalvingFrontDatasetV2 for train / val / test splits and exposes
the data-side hyperparameters the Lightning module needs to configure
the model, loss, and metrics:

    dm = CalvingFrontDataModule(
        splits_dir = "path/to/splits_v2",
        config     = "data/configs/dataset_v2.yaml",
        stats_path = "path/to/splits_v2/normalisation_stats.json",
    )
    model = CalvingLightningModule(**dm.model_hparams, **arch_hparams)
    trainer.fit(model, dm)
"""

import json
import yaml
from pathlib import Path

import lightning as L
from torch.utils.data import DataLoader

from data.bucket_sampler import SizeBucketedBatchSampler
from data.dataset_v2 import (
    CalvingFrontDatasetV2,
    collate_fn,
    GEOPHYS_IDX,
    N_CLIMATE,
    resolve_geophys_idx,
)


class CalvingFrontDataModule(L.LightningDataModule):
    """
    Responsibilities:
      - Creates train / val / test datasets in setup().
      - Exposes channel counts, K, disp_scale, and disp_std so the Lightning
        module never needs to inspect the dataset or stats file directly.
      - Computes disp_scale = sdt_std / disp_std at init time (before setup),
        so it is available when the model is constructed.
    """

    def __init__(
        self,
        splits_dir:    str | Path,
        config:        str | Path | dict,   # YAML path or pre-parsed dict
        stats_path:    str | Path,          # normalisation_stats.json from 04_compute_stats.py
        batch_size:    int  = 8,
        num_workers:   int  = 4,
        pin_memory:    bool = True,
        augment_train: bool = True,
        bucket_batches:        bool = False,   # batch samples of similar spatial size
        bucket_window_batches: int  = 16,
    ):
        super().__init__()
        self.splits_dir = Path(splits_dir)
        self.stats_path = Path(stats_path)
        # Parse once here; passed as dict to each Dataset so the file is not re-opened.
        self.config = yaml.safe_load(open(config)) if not isinstance(config, dict) else config
        self.batch_size    = batch_size
        self.num_workers   = num_workers
        self.pin_memory    = pin_memory
        self.augment_train = augment_train
        self.bucket_batches        = bucket_batches
        self.bucket_window_batches = bucket_window_batches

        if not self.stats_path.exists():
            raise FileNotFoundError(
                f"Normalisation stats not found at {self.stats_path}. "
                "Run 04_compute_stats.py before training."
            )

        # ── Load normalisation stats ──────────────────────────────────────
        # Done at init so disp_scale is available before setup() / model init.
        with open(self.stats_path) as f:
            self.stats = json.load(f)

        sdt_std       = float(self.stats["sdt"]["std"])
        disp_std      = float(self.stats.get("displacement", {}).get("std", sdt_std))
        self.disp_scale = sdt_std / max(disp_std, 1e-6)  # pred_metres = pred_norm * disp_std
        self.disp_std   = disp_std                        # for metric logging in real units

        # ── Derive channel counts from config ─────────────────────────────
        # Available before setup() so the model can be built before fit().
        in_sp = self.config.get("inputs",   {}).get("spatial", [])
        in_sc = self.config.get("inputs",   {}).get("scalar",  [])
        fo_sp = self.config.get("forcings", {}).get("spatial", [])

        # 2 = SDT + sdt_diff (always present)
        self.n_input_channels = (
            2
            + (1 if "sdt_trend" in in_sp else 0)
            + len(resolve_geophys_idx(in_sp))
            + (1 if "bed_slope" in in_sp else 0)
        )
        self.n_forcing_spatial_channels = len(GEOPHYS_IDX) if "geophys" in fo_sp else 0
        self.n_forcing_channels         = N_CLIMATE if "climate" in in_sc else 0
        self.n_scalar_input_channels    = self.n_forcing_channels + (1 if "sdt_diff" in in_sc else 0)

        # ── K: number of input temporal slots ────────────────────────────
        # Read from sampling_params.json to avoid loading the full index file.
        params_path = self.splits_dir / "sampling_params.json"
        if params_path.exists():
            with open(params_path) as f:
                self.K = len(json.load(f)["anchor_offsets"])
        else:
            with open(self.splits_dir / "train.json") as f:
                self.K = len(json.load(f)[0]["input_zarr_indices"])

        self._train_ds: CalvingFrontDatasetV2 | None = None
        self._val_ds:   CalvingFrontDatasetV2 | None = None
        self._test_ds:  CalvingFrontDatasetV2 | None = None

    # ── Dataset setup ─────────────────────────────────────────────────────────

    def setup(self, stage: str | None = None) -> None:
        """Called on every rank before the dataloaders are created."""
        shared = dict(config=self.config, 
                      stats_path=self.stats_path)

        if stage in ("fit", None) and self._train_ds is None:
            self._train_ds = CalvingFrontDatasetV2(
                self.splits_dir / "train.json", augment=self.augment_train, **shared
            )
            self._val_ds = CalvingFrontDatasetV2(
                self.splits_dir / "val.json", augment=False, **shared
            )

        if stage in ("test", "predict", None) and self._test_ds is None:
            self._test_ds = CalvingFrontDatasetV2(
                self.splits_dir / "test.json", augment=False, **shared
            )

    # ── DataLoaders ───────────────────────────────────────────────────────────

    def _loader(self, ds: CalvingFrontDatasetV2, *, shuffle: bool) -> DataLoader:
        kw = dict(
            num_workers        = self.num_workers,
            pin_memory         = self.pin_memory,
            collate_fn         = collate_fn,
            persistent_workers = self.num_workers > 0,
        )
        # batch_sampler is mutually exclusive with batch_size / shuffle / drop_last.
        if self.bucket_batches:
            sampler = SizeBucketedBatchSampler(
                ds.sample_sizes(),
                self.batch_size,
                window_batches = self.bucket_window_batches,
                shuffle        = shuffle,
            )
            return DataLoader(ds, batch_sampler=sampler, **kw)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle, **kw)

    def train_dataloader(self) -> DataLoader:
        return self._loader(self._train_ds, shuffle=True)

    def val_dataloader(self, shuffle=False) -> DataLoader:
        return self._loader(self._val_ds, shuffle=shuffle)

    def test_dataloader(self, shuffle=False) -> DataLoader:
        return self._loader(self._test_ds, shuffle=shuffle)

    # ── Interface for the Lightning module ────────────────────────────────────

    @property
    def model_hparams(self) -> dict:
        """
        Data-side hyperparameters for the Lightning module constructor.
        Pass alongside architecture-specific kwargs:

            model = CalvingLightningModule(**dm.model_hparams, hidden_dim=256, ...)

        Keys
        ----
        n_input_channels           C_in in inputs (K, C_in, H, W)
        n_forcing_spatial_channels C_f  in forcing_spatial (C_f, H, W); 0 if disabled
        n_climate_channels         D_c  in forcing_climate tensors (raw climate channels only)
        n_scalar_input_channels    D_c  in batch["climate"]; = n_climate_channels + 1 when sdt_diff enabled
        K                          number of input temporal slots
        disp_scale                 sdt_std / disp_std; used to reconstruct absolute SDT:
                                     target_sdt = target_disp / disp_scale + inputs[:,-1,0:1]
        disp_std                   displacement std in metres; multiply norm prediction
                                     by disp_std to get metres
        """
        return {
            "n_input_channels":           self.n_input_channels,
            "n_forcing_spatial_channels": self.n_forcing_spatial_channels,
            "n_climate_channels":         self.n_forcing_channels,
            "n_scalar_input_channels":    self.n_scalar_input_channels,
            "K":                          self.K,
            "disp_scale":                 self.disp_scale,
            "disp_std":                   self.disp_std,
            "sdt_mean":                   float(self.stats["sdt"]["mean"]),
        }
