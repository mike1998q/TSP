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
    gated     : per-channel scalar gate g = sigmoid(MLP([feat_t, feat_f])),
                y = g * y_time + (1 - g) * y_freq.
    concat    : like gated, but the gate is per-horizon-step (finer control).
    sum       : plain average of the two forecasts (no parameters).
    time_only : return the time-branch forecast (ablation: no freq branch).
    freq_only : return the freq-branch forecast (ablation: no time branch).

    Gate initialization: weights are zeroed and the bias set to +2.2, so the
    initial gate is a constant g ~ 0.9 favoring the time branch. At init the
    time branch is an exact linear forecaster while the frequency head is
    random — an even 0.5/0.5 mix would make half the initial forecast noise,
    wasting the highest-LR epochs on compensating for it (visible as early
    val-loss fluctuation on small datasets). The gate remains fully learnable.
    """

    GATE_BIAS_INIT = 2.2  # sigmoid(2.2) ~ 0.90

    def __init__(self, d_model: int, pred_len: int, mode: str = "gated",
                 zero_init: bool = True):
        super().__init__()
        self.mode = mode
        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
                nn.Sigmoid(),
            )
            if zero_init:
                nn.init.zeros_(self.gate[2].weight)
                nn.init.constant_(self.gate[2].bias, self.GATE_BIAS_INIT)
        elif mode == "concat":
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, pred_len),
                nn.Sigmoid(),
            )
            if zero_init:
                nn.init.zeros_(self.gate[0].weight)
                nn.init.constant_(self.gate[0].bias, self.GATE_BIAS_INIT)
        elif mode in ("sum", "time_only", "freq_only"):
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
        if self.mode == "time_only":
            return time_forecast
        if self.mode == "freq_only":
            return freq_forecast
        if self.mode == "sum":
            return 0.5 * (time_forecast + freq_forecast)
        g = self.gate(torch.cat([time_feat, freq_feat], dim=-1))  # (B,C,1) or (B,C,H)
        return g * time_forecast + (1.0 - g) * freq_forecast
