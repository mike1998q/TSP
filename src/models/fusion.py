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
    residual  : y = y_time + alpha * y_freq, with a single learned scalar
                alpha (zero-initialized).
    affine    : y = g_t * y_time + g_f * y_freq with two *independent* gates
                (no sum-to-one constraint).
    doubly_residual : y = y_time + phi(feat) * y_freq, where the per-channel
                weight phi is produced by a zero-initialized MLP, so the
                frequency branch fits the time branch's residual (N-BEATS
                style, but with a learned data-dependent weight).
    time_only : return the time-branch forecast (ablation: no freq branch).
    freq_only : return the freq-branch forecast (ablation: no time branch).

    Convex vs. additive fusion
    --------------------------
    ``gated``/``concat``/``sum`` are *convex*: the output is constrained to the
    segment between the two branch forecasts, so the branches act as
    substitutes. If the time branch is right about the trend and the frequency
    branch is right about a spectral peak, a convex rule cannot emit
    ``trend + peak`` -- only a weighted average. ``residual``, ``affine``, and
    ``doubly_residual`` lift that constraint, letting the branches *superpose*
    (cf. N-BEATS doubly-residual stacking). They strictly generalize the convex
    modes: with alpha = (1-g)/g the residual form reproduces any gated output up
    to the overall scale g.

    Initialization: every mode starts at (or very near) the pure time-branch
    forecast. For the convex modes the gate weights are zeroed and the bias set
    to +2.2, giving a constant g ~ 0.9; for the additive modes the frequency
    weight is exactly zero. At init the time branch is an exact linear
    forecaster while the frequency head may be random (it is zero-initialized
    only when the spectral backbone is enabled), so an even 0.5/0.5 mix would
    make half the initial forecast noise, wasting the highest-LR epochs on
    compensating for it. All weights remain fully learnable.
    """

    GATE_BIAS_INIT = 2.2  # sigmoid(2.2) ~ 0.90

    #: modes whose output is a convex combination of the two branch forecasts
    CONVEX_MODES = ("gated", "concat", "sum")
    #: modes that allow the branches to superpose
    ADDITIVE_MODES = ("residual", "affine", "doubly_residual")

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
        elif mode == "residual":
            # One global scalar: the smallest possible departure from the
            # convex rule. zero_init=False starts it at 1.0 (equal weight).
            self.alpha = nn.Parameter(
                torch.zeros(1) if zero_init else torch.ones(1)
            )
        elif mode == "affine":
            # Two independent gates; both may be large at once.
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 2),
            )
            if zero_init:
                nn.init.zeros_(self.gate[2].weight)
                # bias -> (time=1, freq=0): exactly the time-branch forecast.
                with torch.no_grad():
                    self.gate[2].bias.copy_(torch.tensor([1.0, 0.0]))
        elif mode == "doubly_residual":
            # Data-dependent per-channel weight on the frequency correction.
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.GELU(),
                nn.Linear(d_model, 1),
            )
            if zero_init:
                nn.init.zeros_(self.gate[2].weight)
                nn.init.zeros_(self.gate[2].bias)
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
        if self.mode == "residual":
            return time_forecast + self.alpha * freq_forecast
        feats = torch.cat([time_feat, freq_feat], dim=-1)
        if self.mode == "affine":
            w = self.gate(feats)                                  # (B, C, 2)
            g_t, g_f = w[..., :1], w[..., 1:]                     # (B, C, 1)
            return g_t * time_forecast + g_f * freq_forecast
        if self.mode == "doubly_residual":
            phi = self.gate(feats)                                # (B, C, 1)
            return time_forecast + phi * freq_forecast
        g = self.gate(feats)  # (B,C,1) or (B,C,H)
        return g * time_forecast + (1.0 - g) * freq_forecast
