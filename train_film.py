# train_film.py
"""
Train CalvingConvLSTMFiLM with PyTorch Lightning.

Usage
-----
  python train_film.py --config configs/config.yaml

"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import yaml
import lightning as L
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    Callback
)
from lightning.pytorch.loggers import TensorBoardLogger

import torch

from data.datamodule import CalvingFrontDataModule
from models.lightning_module import CalvingLightningModule
from plotting.plot_predictions import plot_predictions

try:
    import wandb
    from lightning.pytorch.loggers import WandbLogger
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

class OverfitPlotCallback(Callback):
    def __init__(self, out_dir, every_n_epochs=10):
        self.out_dir = Path(out_dir)
        self.every_n = every_n_epochs
        self._batch = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self._batch is None:
            # capture the actual batch Lightning is overfitting on
            self._batch = {k: v.detach() if isinstance(v, torch.Tensor) else v
                           for k, v in batch.items()}
        if (trainer.current_epoch + 1) % self.every_n == 0 and batch_idx == 0:
            samples = pl_module.predict_step(self._batch, 0)
            plot_predictions(
                samples, self.out_dir,
                filename=f"overfit_epoch{trainer.current_epoch+1:04d}.png"
            )

def main(cfg: dict, ckpt: str | None = None) -> None:
    # resolve seed (CLI override already applied to cfg) and record it in the
    # saved config.yaml below for provenance
    seed = cfg["experiment"].get("seed", 42)
    cfg["experiment"]["seed"] = seed
    L.seed_everything(seed, workers=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    dm = CalvingFrontDataModule(**cfg["data"])

    # ── Model ─────────────────────────────────────────────────────────────────
    module = CalvingLightningModule(
        **dm.model_hparams,
        **cfg["model"],
        **cfg["training"],
    )
    n_params = sum(p.numel() for p in module.parameters())
    print(f"Parameters: {n_params:,}")

    # ── Output directory ──────────────────────────────────────────────────────
    # With --run_name: deterministic, seed-addressable dir (no timestamp) so a
    # driver can predict the path. Otherwise: {name}_{timestamp} to avoid
    # overwriting previous runs with the same name.
    run_name = cfg["experiment"].get("run_name")
    if run_name:
        out_dir = Path(cfg["experiment"]["output_dir"]) / run_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(cfg["experiment"]["output_dir"]) / f"{cfg['experiment']['name']}_{timestamp}"

    # save config file for reproducibility
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f)

    # ── Loggers ───────────────────────────────────────────────────────────────
    loggers = [TensorBoardLogger(save_dir=str(out_dir), name="tb")]

    wb_cfg = cfg.get("wandb", {})
    if wb_cfg.get("enabled") and _WANDB_AVAILABLE:
        wb_logger = WandbLogger(
            project  = wb_cfg.get("project", "svalcalvingdb"),
            entity   = wb_cfg.get("entity") or None,
            name     = wb_cfg.get("run_name") or None,
            tags     = wb_cfg.get("tags", []),
            save_dir = str(out_dir),
        )
        loggers.append(wb_logger)

    # ── Callbacks ─────────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            dirpath   = str(out_dir / "checkpoints"),
            filename  = "best-{epoch:03d}",
            monitor   = "val/active_mde_mean",
            mode      = "min",
            save_top_k = 1,
            save_last  = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        OverfitPlotCallback(out_dir, every_n_epochs=100),
    ]

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = L.Trainer(
        default_root_dir = str(out_dir),
        max_epochs       = cfg["training"]["max_epochs"],
        callbacks        = callbacks,
        logger           = loggers,
        **cfg.get("trainer", {}),
    )

    # ── Fit ───────────────────────────────────────────────────────────────────
    if _WANDB_AVAILABLE and any(isinstance(l, WandbLogger) for l in loggers):
        wb_logger.watch(module, log="all", log_freq=100)
    trainer.fit(module, dm, ckpt_path=ckpt)

    # ── Sample predictions plot ───────────────────────────────────────────────
    dm.setup("fit")
    device = next(module.parameters()).device
    batch  = next(iter(dm.val_dataloader())) if cfg["trainer"]["overfit_batches"] == 0 else next(iter(dm.train_dataloader()))
    batch  = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    plot_predictions(module.predict_step(batch, 0), out_dir)

    # ── Test ──────────────────────────────────────────────────────────────────
    if cfg["experiment"].get("run_test_at_end", True):
        trainer.test(module, dm)
    if _WANDB_AVAILABLE and any(isinstance(l, WandbLogger) for l in loggers):
        wb_logger.experiment.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--ckpt", default=None, help="Checkpoint to resume from")
    parser.add_argument("--device", default=None, help="If specified, overrides device in config", type=int)
    parser.add_argument("--seed", default=None, type=int,
                        help="Random seed; overrides experiment.seed in config (default 42)")
    parser.add_argument("--run_name", default=None,
                        help="If specified, output dir is {output_dir}/{run_name} with no "
                             "timestamp suffix (deterministic, seed-addressable path)")
    parser.add_argument("--max_epochs", help="If specified, overrides max_epochs in config", type=int)
    parser.add_argument("--fixed_lr", default=None, type=float,
                        help="Use a constant LR instead of cosine schedule (e.g. when resuming)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    if args.device is not None:
        cfg["trainer"]["devices"] = [args.device]
    if args.seed is not None:
        cfg.setdefault("experiment", {})["seed"] = args.seed
    if args.run_name is not None:
        cfg.setdefault("experiment", {})["run_name"] = args.run_name
    if args.max_epochs is not None:
        cfg["training"]["max_epochs"] = args.max_epochs
    if args.fixed_lr is not None:
        cfg["training"]["lr"] = args.fixed_lr
        cfg["training"]["lr_schedule"] = "constant"
    main(cfg, ckpt=args.ckpt)
