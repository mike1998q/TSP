"""Fusion strategies for combining the time- and frequency-domain features."""
from __future__ import annotations

import torch
import torch.nn as nn


class FeatureFusion(nn.Module):
    """Combine two (B, C, d_model) feature tensors into one (B, C, d_model).

    Modes
    -----
    gated  : learn a per-element gate g so out = g * time + (1 - g) * freq.
    sum    : simple element-wise addition.
    concat : concatenate then project back to d_model.
    """

    def __init__(self, d_model: int, mode: str = "gated"):
        super().__init__()
        self.mode = mode
        if mode == "gated":
            self.gate = nn.Sequential(
                nn.Linear(2 * d_model, d_model),
                nn.Sigmoid(),
            )
        elif mode == "concat":
            self.proj = nn.Linear(2 * d_model, d_model)
        elif mode == "sum":
            pass
        else:
            raise ValueError(f"Unknown fusion mode: {mode!r}")

    def forward(self, time_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        if self.mode == "sum":
            return time_feat + freq_feat
        pair = torch.cat([time_feat, freq_feat], dim=-1)
        if self.mode == "concat":
            return self.proj(pair)
        # gated
        g = self.gate(pair)
        return g * time_feat + (1.0 - g) * freq_feat
