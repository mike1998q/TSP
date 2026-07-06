"""Forecast-level fusion of the time- and frequency-domain branches.

Each branch produces its own forecast; fusion combines the two predictions
with a gate conditioned on both branches' features. The final output is
always a convex combination of the branch forecasts — there is no path that
bypasses the branches.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ForecastFusion(nn.Module):
    """Combine two branch forecasts (B, C, H) into one (B, C, H).

    Modes
    -----
    gated  : per-channel scalar gate g = sigmoid(MLP([feat_t, feat_f])),
             y = g * y_time + (1 - g) * y_freq. Zero-init keeps g = 0.5 at
             the start so neither branch dominates before training.
    concat : like gated, but the gate is per-horizon-step (finer control).
    sum    : plain average of the two forecasts (no parameters).
    """

    def __init__(self, d_model: int, pred_len: int, mode: str = "gated"):
        super().__init__()
        self.mode = mode
        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.gate[2].weight)
            nn.init.zeros_(self.gate[2].bias)
        elif mode == "concat":
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, pred_len),
                nn.Sigmoid(),
            )
            nn.init.zeros_(self.gate[0].weight)
            nn.init.zeros_(self.gate[0].bias)
        elif mode == "sum":
            pass
        else:
            raise ValueError(f"Unknown fusion mode: {mode!r}")

    def forward(
        self,
        time_feat: torch.Tensor,
        freq_feat: torch.Tensor,
        time_forecast: torch.Tensor,
        freq_forecast: torch.Tensor,
    ) -> torch.Tensor:
        if self.mode == "sum":
            return 0.5 * (time_forecast + freq_forecast)
        g = self.gate(torch.cat([time_feat, freq_feat], dim=-1))  # (B,C,1) or (B,C,H)
        return g * time_forecast + (1.0 - g) * freq_forecast
