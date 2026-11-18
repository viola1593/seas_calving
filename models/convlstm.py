# models/convlstm.py
"""
Calving front SDT displacement predictor using a ConvLSTM spatiotemporal encoder,
inspired by the OpenSTL framework (Tan et al., NeurIPS 2023).

Output: normalised SDT displacement SDT_norm(t+365) − SDT_norm(t−30).
Reconstruction: pred_abs_m = pred_disp_norm × sdt_std + last_input_sdt_m.

Architecture overview
---------------------
inputs  (B, K, C_total, H, W)   — C_total includes sdt_diff channel
  │
  ├─ tau     (B, K, 4)   ─── broadcast to (B*K, 4, H, W) ──┐
  ├─ climate (B, K, D_c) ─── broadcast to (B*K, D_c, H, W) ─┤
  │                                                           │ concat
  └─► (B*K, C_total+4+D_c, H, W) ──► frame_encoder ──► (B*K, F, H, W)
                                           reshape ──► (B, K, F, H, W)
                                                │
                                           ConvLSTM   multi-layer, K steps
                                                │
                                          hidden (B, F_h, H, W)
                                                │
                                            head  ──► pred (B, 1, H, W)

Key design choices
------------------
- tau and climate are broadcast spatially and concatenated to each frame
  before the encoder, so the ConvLSTM gates see time-gap and climate
  forcing at every recurrent step — not only at the output.
- Stride-1 convolutions throughout — safe for variable (H, W) sizes.
- ConvLSTM hidden dim = 2 × base_channels.
- Zero-init on the final prediction conv for stable early training.
"""

from __future__ import annotations

import torch
import torch.nn as nn


# ── ConvLSTM building blocks (OpenSTL-style) ──────────────────────────────────

class ConvLSTMCell(nn.Module):
    """
    Single ConvLSTM cell (Shi et al. 2015).

    All four gates (i, f, g, o) are computed with one fused Conv2d so that
    the input + hidden tensors are concatenated only once per step.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        kernel_size:     int = 3,
    ) -> None:
        super().__init__()
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=kernel_size // 2,
            bias=True,
        )

    def forward(
        self,
        x:     torch.Tensor,                               # (B, C_in, H, W)
        state: tuple[torch.Tensor, torch.Tensor],          # (h, c)  (B, H_ch, H, W)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h, c = state
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        c_new = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
        h_new = torch.sigmoid(o) * torch.tanh(c_new)
        return h_new, c_new

    def init_state(
        self,
        B:      int,
        H:      int,
        W:      int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone()


class ConvLSTM(nn.Module):
    """
    Multi-layer ConvLSTM translator (OpenSTL convention).

    Processes a sequence of K encoded frames and returns the hidden state
    of the deepest layer at the last time step.  The hidden states are NOT
    returned for intermediate steps because our task outputs a single SDT
    map rather than a full prediction sequence.
    """

    def __init__(
        self,
        in_channels:     int,
        hidden_channels: int,
        num_layers:      int = 3,
        kernel_size:     int = 3,
    ) -> None:
        super().__init__()
        self.cells = nn.ModuleList([
            ConvLSTMCell(
                in_channels     if i == 0 else hidden_channels,
                hidden_channels,
                kernel_size,
            )
            for i in range(num_layers)
        ])

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x_seq : (B, K, C, H, W)

        Returns
        -------
        (B, hidden_channels, H, W)  — hidden state of last layer at t = K
        """
        B, K, _, H, W = x_seq.shape
        device = x_seq.device
        states = [cell.init_state(B, H, W, device) for cell in self.cells]

        for t in range(K):
            x_t = x_seq[:, t]                              # (B, C, H, W)
            for layer_idx, cell in enumerate(self.cells):
                h_new, c_new = cell(x_t, states[layer_idx])
                states[layer_idx] = (h_new, c_new)
                x_t = h_new                                # feed into next layer

        return states[-1][0]                               # (B, hidden_ch, H, W)


# ── Main model ────────────────────────────────────────────────────────────────

class CalvingUNet(nn.Module):
    """
    Calving front SDT predictor built on a ConvLSTM core.

    Compatible with the existing train.py / evaluate.py infrastructure:
      - takes a batch dict (not a single tensor)
      - returns (B, 1, H, W) normalised SDT displacement

    Parameters
    ----------
    in_channels : int
        K × C_total  (e.g. 5 × 13 = 65).
        Internally the model infers K = tau_dim // 4 and
        C_total = in_channels // K.
    base_channels : int
        Feature width F of the frame encoder.
        ConvLSTM hidden width = 2 × base_channels.
    climate_dim : int
        Flat climate vector length = K × D_c.  Used to infer D_c = climate_dim // K.
    tau_dim : int
        Flat tau vector length = K × 4.  Used to infer K = tau_dim // 4.
    num_layers : int
        Depth of the stacked ConvLSTM (default 3).
    """

    def __init__(
        self,
        in_channels:   int,
        base_channels: int = 32,
        climate_dim:   int = 8,
        tau_dim:       int = 16,
        num_layers:    int = 3,
    ) -> None:
        super().__init__()

        K         = tau_dim // 4           # number of input time steps
        C_total   = in_channels // K       # spatial channels per frame
        D_c       = climate_dim // K       # climate scalars per frame
        hidden_ch = base_channels * 2      # ConvLSTM hidden width

        self.K       = K
        self.C_total = C_total

        # ── Per-frame 2-D spatial encoder (weights shared across K) ───────
        # Input = spatial channels + tau (4) + climate (D_c) broadcast spatially.
        # Stride-1 throughout: safe for variable glacier spatial sizes.
        frame_enc_in = C_total + 4 + D_c
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(frame_enc_in, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        # ── ConvLSTM translator ───────────────────────────────────────────
        self.convlstm = ConvLSTM(
            in_channels     = base_channels,
            hidden_channels = hidden_ch,
            num_layers      = num_layers,
        )

        # ── Displacement prediction head ────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(hidden_ch, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 1, 1),
        )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────────────

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init final conv → zero displacement at init,
        # preventing large early loss spikes.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Parameters
        ----------
        batch : dict with keys
            inputs     (B, K, C_total, H, W)  normalised spatial stack
            climate    (B, K, D_c)             normalised climate scalars
            tau        (B, K, 4)               temporal feature vectors
            valid_img  (B, K) bool             already applied in dataset
            valid_clm  (B, K) bool             already applied in dataset
            ...other keys are ignored

        Returns
        -------
        pred : (B, 1, H, W)  normalised SDT displacement
        """
        x       = batch["inputs"]    # (B, K, C_total, H, W)
        climate = batch["climate"]   # (B, K, D_c)
        tau     = batch["tau"]       # (B, K, 4)

        B, K, C, H, W = x.shape

        # ── Per-timestep tau + climate injection ──────────────────────────
        # Broadcast scalars to (B*K, D, H, W) so each frame enters the
        # encoder tagged with its own time and climate context.  The
        # ConvLSTM forget/input gates can then use ocean temp and time gap
        # when deciding what to retain from the previous hidden state.
        tau_spatial  = tau.reshape(B * K, 4, 1, 1).expand(-1, -1, H, W)          # (B*K, 4, H, W)
        clim_spatial = climate.reshape(B * K, -1, 1, 1).expand(-1, -1, H, W)     # (B*K, D_c, H, W)
        x_aug        = torch.cat([x.reshape(B * K, C, H, W),
                                   tau_spatial, clim_spatial], dim=1)             # (B*K, C+4+D_c, H, W)

        # ── Per-frame spatial encoding ────────────────────────────────────
        enc = self.frame_encoder(x_aug)                        # (B*K, F, H, W)
        enc = enc.reshape(B, K, -1, H, W)                     # (B, K, F, H, W)

        # ── ConvLSTM: process K frames sequentially ───────────────────────
        hidden = self.convlstm(enc)                            # (B, 2F, H, W)

        # ── Predict displacement ──────────────────────────────────────────
        return self.head(hidden)                               # (B, 1, H, W)
