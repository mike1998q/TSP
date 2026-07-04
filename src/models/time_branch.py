"""Time-domain branch: series decomposition + temporal MLP encoder.

The branch decomposes the input into a slow-moving trend and a seasonal
residual (a la Autoformer/DLinear), then models each with per-channel linear
maps plus a shared temporal MLP. Operating directly on the raw samples lets
this branch capture local, non-periodic and trend structure that the
frequency branch (which sees global spectral content) tends to smooth out.
"""
from __future__ import annotations

import torch
import torch.nn as nn


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
    ):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        # Linear maps along the time axis for each component.
        self.seasonal_proj = nn.Linear(seq_len, d_model)
        self.trend_proj = nn.Linear(seq_len, d_model)
        # Shared temporal MLP refines the combined representation.
        self.mlp = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seasonal, trend = self.decomp(x)            # each (B, L, C)
        seasonal = seasonal.transpose(1, 2)         # (B, C, L)
        trend = trend.transpose(1, 2)               # (B, C, L)
        feat = self.seasonal_proj(seasonal) + self.trend_proj(trend)  # (B, C, d_model)
        feat = feat + self.mlp(feat)                # residual refinement
        return feat
