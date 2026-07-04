"""Time-domain branch: series decomposition + a sequence encoder.

The branch decomposes the input into a slow-moving trend and a seasonal
residual (a la Autoformer/DLinear), embeds the two components per timestep,
then runs a **Mamba** (selective state-space) encoder over the time axis to
model long-range temporal dynamics. Operating directly on the raw samples lets
this branch capture local, non-periodic and trend structure that the frequency
branch (which sees global spectral content) tends to smooth out.

The encoder is selectable via ``encoder={mamba, mlp}``; Mamba is the default.
Processing is channel-independent: every variable is encoded with shared
weights and summarized to a per-channel feature of width ``d_model``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mamba_block import MambaEncoder


class MovingAvg(nn.Module):
    """Moving-average smoother used to extract the trend component."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> pad the ends by replication so length is preserved.
        pad = self.kernel_size - 1
        front = pad // 2
        back = pad - front
        x_pad = torch.cat(
            [x[:, :1].repeat(1, front, 1), x, x[:, -1:].repeat(1, back, 1)], dim=1
        )
        out = self.avg(x_pad.transpose(1, 2)).transpose(1, 2)
        return out


class SeriesDecomp(nn.Module):
    """Split a series into (seasonal, trend)."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.moving_avg = MovingAvg(kernel_size)

    def forward(self, x: torch.Tensor):
        trend = self.moving_avg(x)
        seasonal = x - trend
        return seasonal, trend


class TimeBranch(nn.Module):
    """Encode a look-back window into a per-channel feature of size d_model.

    Input:  (B, L, C)
    Output: (B, C, d_model)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        kernel_size: int = 25,
        dropout: float = 0.1,
        encoder: str = "mamba",
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_official_mamba: bool = True,
    ):
        super().__init__()
        self.encoder_kind = encoder
        self.decomp = SeriesDecomp(kernel_size)
        # Per-timestep embedding of the [seasonal, trend] pair -> d_model.
        self.embed = nn.Linear(2, d_model)
        self.dropout = nn.Dropout(dropout)

        if encoder == "mamba":
            self.encoder = MambaEncoder(
                d_model=d_model,
                n_layers=mamba_layers,
                d_state=mamba_d_state,
                d_conv=mamba_d_conv,
                expand=mamba_expand,
                use_official=use_official_mamba,
            )
        elif encoder == "mlp":
            # Length-mixing MLP fallback over the flattened sequence.
            self.encoder = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, d_model),
            )
        else:
            raise ValueError(f"Unknown time encoder: {encoder!r}")

    @property
    def using_official_mamba(self) -> bool:
        return getattr(self.encoder, "using_official_kernels", False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, l, c = x.shape
        seasonal, trend = self.decomp(x)                 # each (B, L, C)
        # Stack components as 2 features per (timestep, channel).
        feats = torch.stack([seasonal, trend], dim=-1)   # (B, L, C, 2)
        # Channel-independent: fold channels into the batch dimension.
        feats = feats.permute(0, 2, 1, 3).reshape(b * c, l, 2)  # (B*C, L, 2)
        h = self.dropout(self.embed(feats))              # (B*C, L, d_model)

        if self.encoder_kind == "mamba":
            h = self.encoder(h)                          # (B*C, L, d_model)
            summary = h[:, -1, :]                        # last state (causal) -> (B*C, D)
        else:
            h = h + self.encoder(h)                      # residual refine
            summary = h.mean(dim=1)                      # (B*C, D)

        return summary.reshape(b, c, -1)                 # (B, C, d_model)
