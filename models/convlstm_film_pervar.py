# models/convlstm_film_pervar.py
"""
Calving front SDT displacement predictor: ConvLSTM backbone with per-variable
FiLM conditioning (Model 3 in the seasonal ablation).

Replaces TwoPathwayFiLM with separate SingleVarFiLM modules — one per climate
scalar — applied sequentially to the frame-encoded features.

Injection point 1 (after frame encoder, per timestep):
    ocean_temp  → mar_runoff  → sdt_diff (only if sdt_diff in scalar inputs)
    dims:  [1]      [0]           [2]   of batch["climate"]

Injection point 2 (after forcing encoder, per sample):
    ocean_temp  → mar_runoff
    dims:  [1]      [0]          of batch["forcing_climate"]

film_compose selects how the per-variable FiLMs combine at each injection point:
    sequential       out = FiLMₙ(…FiLM₂(FiLM₁(x))…)  — the default.
                     Cross-terms are intentional: each variable scales what the
                     previous one already did. ocean_temp (structural, slow) goes
                     first; surface runoff (seasonal) scales that result; sdt_diff
                     (recent dynamics) adjusts last.
    parallel_sum     out = x + Σ(x·gᵢ + bᵢ)          — variables act independently.
    parallel_concat  out = x + projₖ(cat[deltaᵢ])    — a 1×1 conv mixes per-variable
                     residuals. All three modes are identity-at-init: the FiLM g, b
                     start at zero (so the deltas are zero), and the proj bias is zero.
                     The proj *weight* is kaiming-init (not zero) — zeroing it would
                     deadlock gradient flow to/from the FiLM nets, pinning both at zero.

trunk_hidden_dim controls per-variable FiLM expressiveness:
    0 or None  →  direct linear  (1 → 2*C)
    > 0        →  MLP with ReLU  (1 → trunk_hidden_dim → 2*C)

Architecture
------------
inputs  (B, K, C_in, H, W)
tau     (B, K, 4)     ─── broadcast → (B*K, 4, H, W)
              ↓ concat
        (B*K, C_in+4, H, W)
              ↓ frame_encoder
        (B*K, F, H, W)
              ↓ SingleVarFiLM(ocean_temp) ← climate[:,1] (B*K, 1)
              ↓ SingleVarFiLM(mar_runoff) ← climate[:,0] (B*K, 1)
             [↓ SingleVarFiLM(sdt_diff)  ← climate[:,2] (B*K, 1)]
        (B*K, F, H, W) reshape → (B, K, F, H, W)
              ↓ ConvLSTM
        (B, F_h, H, W)
              ↓ cat ← forcing_spatial (B, C_f, H, W)
        (B, F_h+C_f, H, W)
              ↓ forcing_encoder
        (B, F_h, H, W)
              ↓ SingleVarFiLM(ocean_temp) ← forcing_climate[:,1] (B, 1)
              ↓ SingleVarFiLM(mar_runoff) ← forcing_climate[:,0] (B, 1)
             [↓ SingleVarFiLM(seasonality) ← glacier_seasonality (B, 1)]
        (B, F_h, H, W)
              ↓ head
        (B, 1, H, W)   — normalised SDT displacement
"""

from __future__ import annotations

import torch
import torch.nn as nn

from models.convlstm import ConvLSTM


# ── SingleVarFiLM ─────────────────────────────────────────────────────────────

class SingleVarFiLM(nn.Module):
    """
    FiLM conditioned on a single scalar variable.

    Parameters
    ----------
    n_channels       : number of feature-map channels to modulate
    trunk_hidden_dim : 0 or None → direct linear; >0 → MLP with one ReLU hidden layer
    """

    def __init__(self, n_channels: int, trunk_hidden_dim: int | None = 0) -> None:
        super().__init__()
        if trunk_hidden_dim:
            self.net = nn.Sequential(
                nn.Linear(1, trunk_hidden_dim),
                nn.ReLU(),
                nn.Linear(trunk_hidden_dim, 2 * n_channels),
            )
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        else:
            self.net = nn.Linear(1, 2 * n_channels)
            nn.init.zeros_(self.net.weight)
            nn.init.zeros_(self.net.bias)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        x : (B, C, H, W)
        c : (B, 1)
        """
        g, b = self.net(c).chunk(2, dim=-1)          # (B, C) each
        return x * (1.0 + g[:, :, None, None]) + b[:, :, None, None]

    def delta(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Residual-only FiLM: x·g + b (no +1). Used by parallel compose modes."""
        g, b = self.net(c).chunk(2, dim=-1)
        return x * g[:, :, None, None] + b[:, :, None, None]


# ── Main model ────────────────────────────────────────────────────────────────

class CalvingConvLSTMFiLMPerVar(nn.Module):
    """
    ConvLSTM calving front predictor with per-variable FiLM conditioning.

    Parameters
    ----------
    n_input_channels            C_in  spatial channels per input timestep
    n_forcing_spatial_channels  C_f   spatial channels of future geophys (0 = disabled)
    n_climate_channels          D_c   number of pure climate scalars (= 2: mar_runoff, ocean_temp)
    K                           number of historical input timesteps
    base_channels               feature width F of the frame encoder
    hidden_channels             ConvLSTM hidden width (defaults to 2 × base_channels)
    num_layers                  depth of the stacked ConvLSTM
    trunk_hidden_dim            hidden size for per-variable FiLM MLP; 0 = direct linear
    n_scalar_input_channels     total scalar dim including optional sdt_diff
    disp_scale, disp_std        data-side scalars from DataModule.model_hparams
    norm                        'batch' (default) | 'group' — normalisation at the two
                                per-sample sites (forcing_encoder, head), which reduce
                                over B = batch_size only. 'group' makes them independent
                                of batch composition, as required when batches are
                                size-bucketed. frame_encoder always keeps BatchNorm
                                (it reduces over B*K). Checkpoints are not
                                interchangeable between the two settings.
    """

    def __init__(
        self,
        n_input_channels:           int,
        n_forcing_spatial_channels: int,
        n_climate_channels:         int,
        K:                          int,
        base_channels:              int         = 64,
        hidden_channels:            int | None  = None,
        num_layers:                 int         = 3,
        trunk_hidden_dim:           int | None  = 0,
        n_scalar_input_channels:    int | None  = None,
        disp_scale:                 float       = 1.0,
        disp_std:                   float       = 1.0,
        use_seasonality_film:       bool        = False,
        use_ocean_temp_forcing_film: bool       = True,
        use_mar_runoff_forcing_film: bool       = True,
        film_compose:               str         = "sequential",
        norm:                       str         = "batch",
    ) -> None:
        super().__init__()

        if film_compose not in ("sequential", "parallel_sum", "parallel_concat"):
            raise ValueError(
                f"film_compose must be one of "
                f"'sequential'|'parallel_sum'|'parallel_concat', got {film_compose!r}"
            )
        self.film_compose = film_compose

        if norm not in ("batch", "group"):
            raise ValueError(f"norm must be 'batch' or 'group', got {norm!r}")

        def _norm(c: int) -> nn.Module:
            """Norm for the two B-sized sites: GroupNorm removes batch-composition dependence."""
            if norm == "group":
                assert c % 8 == 0, f"norm='group' needs channels divisible by 8, got {c}"
                return nn.GroupNorm(8, c)
            return nn.BatchNorm2d(c)

        F   = base_channels
        F_h = hidden_channels if hidden_channels is not None else F * 2

        self.register_buffer("disp_scale", torch.tensor(disp_scale))
        self.register_buffer("disp_std",   torch.tensor(disp_std))
        self.K = K
        self.use_seasonality_film = use_seasonality_film

        _n_scalar = n_scalar_input_channels if n_scalar_input_channels is not None else n_climate_channels
        self.has_sdt_diff_film = _n_scalar > n_climate_channels

        # ── Frame encoder (weights shared across K timesteps) ─────────────
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(n_input_channels + 4, F, 3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
            nn.Conv2d(F, F, 3, padding=1, bias=False),
            nn.BatchNorm2d(F),
            nn.ReLU(inplace=True),
        )

        # Injection point 1: per-variable FiLMs on frame encoder output
        # Order: ocean_temp (dim 1) → mar_runoff (dim 0) → sdt_diff (dim 2, optional)
        self.film_ocean_temp_input  = SingleVarFiLM(F, trunk_hidden_dim)
        self.film_mar_runoff_input  = SingleVarFiLM(F, trunk_hidden_dim)
        if self.has_sdt_diff_film:
            self.film_sdt_diff_input = SingleVarFiLM(F, trunk_hidden_dim)

        # ── ConvLSTM backbone ─────────────────────────────────────────────
        self.convlstm = ConvLSTM(
            in_channels     = F,
            hidden_channels = F_h,
            num_layers      = num_layers,
        )

        # ── Future-forcing integration ────────────────────────────────────
        self.forcing_encoder = nn.Sequential(
            nn.Conv2d(F_h + n_forcing_spatial_channels, F_h, 3, padding=1, bias=False),
            _norm(F_h),
            nn.ReLU(inplace=True),
        )

        self.use_ocean_temp_forcing_film = use_ocean_temp_forcing_film
        self.use_mar_runoff_forcing_film = use_mar_runoff_forcing_film
        # Injection point 2: per-variable FiLMs on forcing encoder output
        # Order: ocean_temp (dim 1) → mar_runoff (dim 0)
        if self.use_ocean_temp_forcing_film:
            self.film_ocean_temp_forcing = SingleVarFiLM(F_h, trunk_hidden_dim)
        if self.use_mar_runoff_forcing_film:
            self.film_mar_runoff_forcing = SingleVarFiLM(F_h, trunk_hidden_dim)
        if self.use_seasonality_film:
            self.film_seasonality_forcing = SingleVarFiLM(F_h, trunk_hidden_dim)

        # ── Prediction head ───────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Conv2d(F_h, F, 3, padding=1, bias=False),
            _norm(F),
            nn.ReLU(inplace=True),
            nn.Conv2d(F, 1, 1),
        )

        # Concat-mode projections: N·C → C. Only the *bias* is zero-init: at init the
        # per-variable deltas are zero (FiLM nets are zero-init), so proj(0·…)+0 = 0 and
        # x + proj(…) still starts as x. The weight must stay nonzero, otherwise it would
        # block gradient flow to the FiLM nets — and the FiLM nets (deltas=0) would block
        # gradient to the proj weight — a deadlock that pins both at zero forever.
        if film_compose == "parallel_concat":
            n_input   = 2 + int(self.has_sdt_diff_film)     # ocean_temp, mar_runoff, [sdt_diff]
            n_forcing = (int(self.use_ocean_temp_forcing_film)
                         + int(self.use_mar_runoff_forcing_film)
                         + int(self.use_seasonality_film))
            self.proj_input   = nn.Conv2d(n_input   * F,   F,   1)
            self.proj_forcing = nn.Conv2d(n_forcing * F_h, F_h, 1)
            nn.init.kaiming_normal_(self.proj_input.weight,   nonlinearity="relu")
            nn.init.kaiming_normal_(self.proj_forcing.weight, nonlinearity="relu")
            nn.init.zeros_(self.proj_input.bias)
            nn.init.zeros_(self.proj_forcing.bias)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is getattr(self, "proj_input", None) or m is getattr(self, "proj_forcing", None):
                    continue  # parallel_concat projections initialised above (weight kaiming, bias zero)
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        nn.init.normal_(self.head[-1].weight, std=0.01)
        nn.init.zeros_(self.head[-1].bias)
        # SingleVarFiLM output layers are already zero-initialised in __init__

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Reads from the batch dict produced by CalvingFrontDatasetV2 / collate_fn.

        Returns
        -------
        (B, 1, H, W)  normalised SDT displacement
        """
        x               = batch["inputs"]           # (B, K, C_in, H, W)
        tau             = batch["tau"]              # (B, K, 4)
        climate         = batch["climate"]          # (B, K, D_c)
        forcing_spatial = batch["forcing_spatial"]  # (B, C_f, H, W)
        forcing_climate = batch["forcing_climate"]  # (B, D_c)

        B, K, C, H, W = x.shape

        # ── Per-frame encoding with τ ─────────────────────────────────────
        tau_sp = tau.reshape(B * K, 4, 1, 1).expand(-1, -1, H, W)
        x_enc  = torch.cat([x.reshape(B * K, C, H, W), tau_sp], dim=1)
        enc    = self.frame_encoder(x_enc)          # (B*K, F, H, W)

        # ── Per-variable FiLM: injection point 1 ─────────────────────────
        clm = climate.reshape(B * K, -1)            # (B*K, D_c)
        oc, mr = clm[:, 1:2], clm[:, 0:1]
        if self.film_compose == "sequential":
            enc = self.film_ocean_temp_input(enc, oc)   # ocean_temp
            enc = self.film_mar_runoff_input(enc, mr)   # mar_runoff
            if self.has_sdt_diff_film:
                enc = self.film_sdt_diff_input(enc, clm[:, 2:3])  # sdt_diff
        else:
            deltas = [self.film_ocean_temp_input.delta(enc, oc),
                      self.film_mar_runoff_input.delta(enc, mr)]
            if self.has_sdt_diff_film:
                deltas.append(self.film_sdt_diff_input.delta(enc, clm[:, 2:3]))
            if self.film_compose == "parallel_sum":
                enc = enc + sum(deltas)
            else:  # parallel_concat
                enc = enc + self.proj_input(torch.cat(deltas, dim=1))

        enc = enc.reshape(B, K, -1, H, W)          # (B, K, F, H, W)

        # ── ConvLSTM ──────────────────────────────────────────────────────
        hidden = self.convlstm(enc)                 # (B, F_h, H, W)

        # ── Future-forcing integration ────────────────────────────────────
        fused = torch.cat([hidden, forcing_spatial], dim=1)   # (B, F_h+C_f, H, W)
        fused = self.forcing_encoder(fused)                   # (B, F_h, H, W)

        # ── Per-variable FiLM: injection point 2 ─────────────────────────
        oc_f, mr_f = forcing_climate[:, 1:2], forcing_climate[:, 0:1]
        if self.film_compose == "sequential":
            if self.use_ocean_temp_forcing_film:
                fused = self.film_ocean_temp_forcing(fused, oc_f)  # ocean_temp
            if self.use_mar_runoff_forcing_film:
                fused = self.film_mar_runoff_forcing(fused, mr_f)  # mar_runoff
            if self.use_seasonality_film:
                seas = batch["glacier_seasonality"].unsqueeze(1).float()  # (B,) int → (B,1) float
                fused = self.film_seasonality_forcing(fused, seas)
        else:
            deltas = []
            if self.use_ocean_temp_forcing_film:
                deltas.append(self.film_ocean_temp_forcing.delta(fused, oc_f))
            if self.use_mar_runoff_forcing_film:
                deltas.append(self.film_mar_runoff_forcing.delta(fused, mr_f))
            if self.use_seasonality_film:
                seas = batch["glacier_seasonality"].unsqueeze(1).float()  # (B,) int → (B,1) float
                deltas.append(self.film_seasonality_forcing.delta(fused, seas))
            if self.film_compose == "parallel_sum":
                fused = fused + sum(deltas)        # sum([]) == 0 → fused unchanged ("none" works free)
            else:  # parallel_concat
                fused = fused + self.proj_forcing(torch.cat(deltas, dim=1))

        return self.head(fused)                     # (B, 1, H, W)
