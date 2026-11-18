# models/lightning_module.py
"""
Lightning module wrapping CalvingConvLSTMFiLM for calving front prediction.

Usage
-----
    dm     = CalvingFrontDataModule(splits_dir=..., config=..., stats_path=...)
    module = CalvingLightningModule(**dm.model_hparams, base_channels=64, num_layers=3)
    trainer = L.Trainer(max_epochs=100, ...)
    trainer.fit(module, dm)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
import lightning as L

from losses import CalvingLoss, CalvingMovementLoss, displacement_sample_scale
from metrics import (
    DEFAULT_SDT_VAR_THRESHOLD_M2,
    aggregate_metrics,
    denormalise_sdt,
    evaluate_batch,
)

from models.convlstm_film_pervar import CalvingConvLSTMFiLMPerVar
from models.convlstm_seasonal import CalvingConvLSTMSeasonal

_MODEL_REGISTRY = {
    "convlstm_film_pervar":    CalvingConvLSTMFiLMPerVar,
    "convlstm_seasonal":       CalvingConvLSTMSeasonal,
}


class CalvingLightningModule(L.LightningModule):
    """
    ConvLSTM+FiLM calving front predictor wrapped in a Lightning module.

    Parameters
    ----------
    n_input_channels, n_forcing_spatial_channels, n_climate_channels, K,
    disp_scale, disp_std, sdt_mean
        Data-side hyperparameters — pass ``**dm.model_hparams`` directly.
    base_channels, hidden_channels, num_layers
        Architecture hyperparameters for CalvingConvLSTMFiLM.
    lr, weight_decay, max_epochs
        AdamW + CosineAnnealingLR.
    loss_decay_pixels, loss_min_weight, loss_norm
        CalvingLoss hyperparameters (see losses.py).  loss_norm is 'l1'
        (default) or 'l2'.
    """

    def __init__(
        self,
        # ── data-side (from dm.model_hparams) ────────────────────────────
        n_input_channels:           int,
        n_forcing_spatial_channels: int,
        n_climate_channels:         int,
        K:                          int,
        disp_scale:                 float,
        disp_std:                   float,
        sdt_mean:                   float       = 0.0,
        n_scalar_input_channels:    int | None  = None,
        # ── architecture ─────────────────────────────────────────────────
        architecture:               str         = "convlstm_film_pervar",
        base_channels:              int         = 32,
        hidden_channels:            int | None  = None,
        num_layers:                 int         = 2,
        film_hidden_dim:            int | None  = None,
        trunk_hidden_dim:           int | None  = None,
        use_seasonality_film:       bool        = False,
        use_ocean_temp_forcing_film: bool       = True,
        use_mar_runoff_forcing_film: bool       = True,
        film_compose:               str         = "parallel_sum",
        norm:                       str         = "batch",
        # ── optimiser ────────────────────────────────────────────────────
        lr:                         float       = 1e-3,
        weight_decay:               float       = 1e-4,
        max_epochs:                 int         = 100,
        lr_schedule:                str         = "cosine",
        # ── loss ─────────────────────────────────────────────────────────
        loss_decay_pixels:          float       = 10.0,
        loss_min_weight:            float       = 0.0,
        loss_norm:                  str         = "l1",
        loss_movement_weight:       bool        = False,
        loss_disp_reweight:         bool        = False,
        cls_loss_weight:            float       = 1.0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        model_cls = _MODEL_REGISTRY[architecture]
        if architecture == "convlstm_film":
            _arch_kwargs = {"film_hidden_dim": film_hidden_dim}
        elif architecture == "convlstm_film_pervar":
            _arch_kwargs = {"trunk_hidden_dim": trunk_hidden_dim,
                            "use_seasonality_film": use_seasonality_film,
                            "use_ocean_temp_forcing_film": use_ocean_temp_forcing_film,
                            "use_mar_runoff_forcing_film": use_mar_runoff_forcing_film,
                            "film_compose": film_compose,
                            "norm": norm}
        elif architecture in ("convlstm_film_seasonal", "convlstm_forecast_cls"):
            _arch_kwargs = {"trunk_hidden_dim": trunk_hidden_dim}
        else:  # convlstm_seasonal
            _arch_kwargs = {}
        self.model = model_cls(
            n_input_channels            = n_input_channels,
            n_forcing_spatial_channels  = n_forcing_spatial_channels,
            n_climate_channels          = n_climate_channels,
            n_scalar_input_channels     = n_scalar_input_channels,
            K                           = K,
            base_channels               = base_channels,
            hidden_channels             = hidden_channels,
            num_layers                  = num_layers,
            disp_scale                  = disp_scale,
            disp_std                    = disp_std,
            **_arch_kwargs,
        )
        if loss_movement_weight:
            self.criterion = CalvingMovementLoss(
                disp_std     = disp_std,
                decay_pixels = loss_decay_pixels,
                min_weight   = loss_min_weight,
                norm         = loss_norm,
            )
        else:
            self.criterion = CalvingLoss(
                decay_pixels = loss_decay_pixels,
                min_weight   = loss_min_weight,
                norm         = loss_norm,
            )

        self._val_records:  list[dict] = []
        self._test_records: list[dict] = []

    # ── SDT stats for denormalisation ─────────────────────────────────────────

    @property
    def _sdt_stats(self) -> dict:
        sdt_std = self.hparams.disp_scale * self.hparams.disp_std
        return {"sdt": {"mean": self.hparams.sdt_mean, "std": float(sdt_std)}}

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        return self.model(batch)

    # ── Prediction ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def predict_step(self, batch: dict, _batch_idx: int) -> list[dict]:
        """
        Returns a list of per-sample dicts with denormalised arrays (metres)
        ready for plotting. Called by trainer.predict() or directly.
        """
        is_cls = self.hparams.architecture == "convlstm_forecast_cls"
        if is_cls:
            pred, cls_logit = self.model(batch)
        else:
            pred = self.model(batch)                         # (B, 1, H, W)
        stats = self._sdt_stats
        sdt_std  = float(stats["sdt"]["std"])
        sdt_mean = float(stats["sdt"]["mean"])
        disp_scale = self.model.disp_scale.item()

        last_norm    = batch["inputs"][:, -1, 0:1]         # (B, 1, H, W)
        pred_sdt_n   = pred / disp_scale + last_norm       # (B, 1, H, W)

        last_sdt_m   = (last_norm[:, 0]  * sdt_std + sdt_mean).cpu().numpy()
        pred_disp_m  = (pred[:, 0] * float(self.hparams.disp_std)).cpu().numpy()
        pred_sdt_m   = (pred_sdt_n[:, 0] * sdt_std + sdt_mean).cpu().numpy()
        input_trace  = batch["input_trace"][:, 0].cpu().numpy()
        target_trace = batch["target_trace"][:, 0].cpu().numpy()

        original_hw = batch.get("original_hw")
        glacier_ids = batch.get("glacier_id", [""] * pred.shape[0])

        return [
            {
                "last_sdt_m":   last_sdt_m[b],
                "pred_disp_m":  pred_disp_m[b],
                "pred_sdt_m":   pred_sdt_m[b],
                "input_trace":  input_trace[b],
                "target_trace": target_trace[b],
                "original_hw":  original_hw[b] if original_hw else last_sdt_m[b].shape,
                "glacier_id":   glacier_ids[b],
            }
            for b in range(pred.shape[0])
        ]

    # ── SDT reconstruction + metrics ──────────────────────────────────────────

    @torch.no_grad()
    def _batch_to_records(self, pred: torch.Tensor, batch: dict) -> list[dict]:
        """
        Reconstruct absolute SDT (metres) from normalised displacement and run
        evaluate_batch to get per-sample metric dicts.

        pred                : (B, 1, H, W)   normalised displacement output
        batch["inputs"][:,−1, 0:1] = last input SDT (normalised)
        batch["target_displacement"] = normalised displacement target
        """
        last_norm  = batch["inputs"][:, -1, 0:1]         # (B, 1, H, W)
        disp_scale = self.model.disp_scale                # buffer, same device

        # pred / disp_scale converts norm-displacement → norm-SDT increment
        pred_sdt_norm   = pred                           / disp_scale + last_norm
        target_sdt_norm = batch["target_displacement"]   / disp_scale + last_norm
        persist_sdt_norm = last_norm                      # zero displacement

        stats     = self._sdt_stats
        pred_m    = denormalise_sdt(pred_sdt_norm.cpu().numpy(),    stats)
        target_m  = denormalise_sdt(target_sdt_norm.cpu().numpy(),  stats)
        persist_m = denormalise_sdt(persist_sdt_norm.cpu().numpy(), stats)

        # Active mask: pixels where input SDT variance > threshold (m²)
        sdt_stack  = batch["inputs"][:, :, 0, :, :].cpu().numpy()  # (B, K, H, W)
        sdt_std    = float(stats["sdt"]["std"])
        sdt_var_m2 = np.var(sdt_stack, axis=1) * sdt_std ** 2      # (B, H, W)
        active_masks = sdt_var_m2 > DEFAULT_SDT_VAR_THRESHOLD_M2

        target_traces = batch.get("target_trace")
        if target_traces is not None:
            tt = target_traces.cpu().numpy()
            target_traces = (tt[:, 0] if tt.ndim == 4 else tt).astype(bool)

        svm = batch.get("spatial_valid_mask")
        if svm is not None:
            svm = svm.cpu().numpy()

        return evaluate_batch(
            pred_m, target_m, target_traces,
            persist_sdts        = persist_m,
            active_masks        = active_masks,
            spatial_valid_masks = svm,
            original_hw         = batch.get("original_hw"),
        )

    # ── Training ──────────────────────────────────────────────────────────────

    def on_train_epoch_end(self) -> None:
        """Log per-variable FiLM parameter norms (weight norm, mean |bias|)."""
        for name, m in self.model.named_modules():
            if type(m).__name__ != "SingleVarFiLM":
                continue
            out = m.net[-1] if isinstance(m.net, torch.nn.Sequential) else m.net
            self.log(f"film/{name}/w_norm",     out.weight.norm().item())
            self.log(f"film/{name}/b_mean_abs",  out.bias.abs().mean().item())

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        is_cls = self.hparams.architecture == "convlstm_forecast_cls"
        if is_cls:
            pred, cls_logit = self.model(batch)
        else:
            pred = self.model(batch)
        mask = batch.get("spatial_valid_mask")
        if self.hparams.loss_disp_reweight:
            loss_dict = self.criterion(
                pred, batch["target_displacement"],
                input_trace=batch.get("target_trace"), # used to be conditioned on input_trace but target trace makes more sense
                mask=mask, 
                per_sample=True,
            )
            scale = displacement_sample_scale(
                batch["target_displacement"],
                disp_std=self.hparams.disp_std,
                mask=mask,
            )
            loss = (loss_dict["loss"] * scale).mean()
        else:
            loss_dict = self.criterion(
                pred, batch["target_displacement"],
                input_trace=batch.get("target_trace"), mask=mask,
            )
            loss = loss_dict["loss"]
        if is_cls:
            L_cls = F.binary_cross_entropy_with_logits(
                cls_logit.squeeze(1), batch["glacier_seasonality"]
            )
            loss = loss + self.hparams.cls_loss_weight * L_cls
            self.log("train/cls_loss", L_cls, on_step=True, on_epoch=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ── Validation ────────────────────────────────────────────────────────────

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        is_cls = self.hparams.architecture == "convlstm_forecast_cls"
        if is_cls:
            pred, cls_logit = self.model(batch)
            acc = ((cls_logit.squeeze(1) > 0) == batch["glacier_seasonality"].bool()).float().mean()
            self.log("val/cls_acc", acc, on_step=False, on_epoch=True)
        else:
            pred = self.model(batch)
        loss_dict = self.criterion(
            pred,
            batch["target_displacement"],
            input_trace = batch.get("target_trace"),
            mask        = batch.get("spatial_valid_mask"),
        )
        self.log("val/loss", loss_dict["loss"], on_step=False, on_epoch=True, prog_bar=True)
        
        self._val_records.extend(self._batch_to_records(pred, batch))

    def on_validation_epoch_end(self) -> None:
        self._log_records(self._val_records, prefix="val")
        self._val_records.clear()

    # ── Test ──────────────────────────────────────────────────────────────────

    def test_step(self, batch: dict, batch_idx: int) -> None:
        is_cls = self.hparams.architecture == "convlstm_forecast_cls"
        if is_cls:
            pred, cls_logit = self.model(batch)
            acc = ((cls_logit.squeeze(1) > 0) == batch["glacier_seasonality"].bool()).float().mean()
            self.log("test/cls_acc", acc, on_step=False, on_epoch=True)
        else:
            pred = self.model(batch)
        self._test_records.extend(self._batch_to_records(pred, batch))

    def on_test_epoch_end(self) -> None:
        self._log_records(self._test_records, prefix="test")
        self._test_records.clear()

    # ── Shared logging helper ─────────────────────────────────────────────────

    def _log_records(self, records: list[dict], prefix: str) -> None:
        if not records:
            return
        is_test = prefix == "test"
        agg = aggregate_metrics(records, quantiles=[0.5, 0.9, 0.95] if is_test else [])

        for k, v in agg.items():
            # during val, skip std and n_valid — only _mean values are useful
            if not is_test and (k.startswith("n_valid_") or k.endswith("_std")):
                continue
            if np.isfinite(v):
                self.log(f"{prefix}/{k}", float(v), sync_dist=True)

    # ── Optimiser ─────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.parameters(),
            lr           = self.hparams.lr,
            weight_decay = self.hparams.weight_decay,
        )
        if self.hparams.lr_schedule == "constant":
            return opt
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max   = self.hparams.max_epochs,
            eta_min = self.hparams.lr * 1e-2,
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": sch, "interval": "epoch"},
        }
