# models/convlstm_seasonal.py
"""
Model 1 (seasonal ablation): ConvLSTM baseline with broadcast-concat future forcing.

Architecture
------------
inputs  (B, K, C_in, H, W)
tau     (B, K, 4)        ── broadcast → (B*K, 4, H, W)   ──┐
climate (B, K, D_sc)     ── broadcast → (B*K, D_sc, H, W) ──┤ concat
                                                              ↓
                                            frame_encoder  (B*K, F, H, W)
                                                              ↓ reshape
                                            ConvLSTM       (B, F_h, H, W)
                                                              ↓
forcing_climate (B, D_c) ── broadcast → (B, D_c, H, W)    ──┤ concat
                                                              ↓
                                            head           (B, 1, H, W)

Compared with Model 2 (FiLMSeasonal): no FiLM trunk, no pathway heads,
no forcing_spatial — future climate reaches the head via a plain concat+conv.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.convlstm import ConvLSTM


class CalvingConvLSTMSeasonal(nn.Module):
    """
    Parameters
    ----------
    n_input_channels            C_in  spatial channels per input timestep
    n_forcing_spatial_channels  must be 0 (API consistency check only)
    n_climate_channels          D_c   dim of batch["forcing_climate"]
    n_scalar_input_channels     D_sc  dim of batch["climate"] per timestep
    K                           number of historical input timesteps
    base_channels               frame encoder + head feature width F
    hidden_channels             ConvLSTM hidden width (default 2×F)
    num_layers                  ConvLSTM depth
    disp_scale, disp_std        registered as buffers for predict_step
    """

    def __init__(
        self,
        n_input_channels:           int,
        n_forcing_spatial_channels: int,
        n_climate_channels:         int,
        n_scalar_input_channels:    int,
        K:                          int,
        base_channels:              int        = 32,
        hidden_channels:            int | None = None,
        num_layers:                 int        = 2,
        disp_scale:                 float      = 1.0,
        disp_std:                   float      = 1.0,
    ) -> None:
        super().__init__()
        assert n_forcing_spatial_channels == 0, \
            "Model 1 ignores forcing_spatial; pass n_forcing_spatial_channels=0"

        F   = base_channels
        F_h = hidden_channels if hidden_channels is not None else F * 2
        self.K = K

        self.register_buffer("disp_scale", torch.tensor(disp_scale))
        self.register_buffer("disp_std",   torch.tensor(disp_std))

        # Per-frame encoder: spatial + tau (4) + historical climate scalars
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels + 4 + n_scalar_input_channels, F, 3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
            nn.Conv2d(F, F, 3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
        )

        self.convlstm = ConvLSTM(in_channels=F, hidden_channels=F_h, num_layers=num_layers)

        # Head input = hidden state concat with broadcast future climate
        self.head = nn.Sequential(
            nn.Conv2d(F_h + n_climate_channels, F, 3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
            nn.Conv2d(F, 1, 1),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init final conv → zero displacement at init
        nn.init.normal_(self.head[-1].weight, mean=0.0, std=0.01) # small init for stability
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, batch: dict) -> torch.Tensor:
        x               = batch["inputs"]           # (B, K, C_in, H, W)
        tau             = batch["tau"]              # (B, K, 4)
        climate         = batch["climate"]          # (B, K, D_sc)
        forcing_climate = batch["forcing_climate"]  # (B, D_c)

        B, K, C, H, W = x.shape

        # Broadcast tau + historical climate per timestep, concat with spatial
        tau_sp  = tau.reshape(B * K, 4, 1, 1).expand(-1, -1, H, W)
        clim_sp = climate.reshape(B * K, -1, 1, 1).expand(-1, -1, H, W)
        x_enc   = torch.cat([x.reshape(B * K, C, H, W), tau_sp, clim_sp], dim=1)

        enc    = self.frame_encoder(x_enc)          # (B*K, F, H, W)
        enc    = enc.reshape(B, K, -1, H, W)        # (B, K, F, H, W)

        hidden = self.convlstm(enc)                 # (B, F_h, H, W)

        # Broadcast future forcing climate and concat with hidden state
        fc_sp  = forcing_climate[:, :, None, None].expand(-1, -1, H, W)
        fused  = torch.cat([hidden, fc_sp], dim=1)  # (B, F_h + D_c, H, W)

        return self.head(fused)                     # (B, 1, H, W)
