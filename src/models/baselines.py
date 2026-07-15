"""Reference baselines implemented inside this framework.

Running a baseline through the exact same data pipeline, splits, training
loop, schedule, and evaluation script as the main model removes every
implementation confound from the comparison — the fair-comparison
requirement reviewers rightly insist on. Select via ``model.arch`` in the
config (or ``--arch`` on scripts/run_main_results.py).
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .time_branch import SeriesDecomp


class DLinearBaseline(nn.Module):
    """Faithful DLinear (Zeng et al., AAAI 2023), shared-weights variant.

    Moving-average decomposition into seasonal and trend components, each
    forecast by a single linear map from the look-back window to the
    horizon. No normalization, no deep components — the canonical linear
    baseline for long-term forecasting.
    """

    def __init__(self, seq_len: int, pred_len: int, kernel_size: int = 25):
        super().__init__()
        self.decomp = SeriesDecomp(kernel_size)
        self.lin_seasonal = nn.Linear(seq_len, pred_len)
        self.lin_trend = nn.Linear(seq_len, pred_len)

    def forward(self, x: torch.Tensor, stats: torch.Tensor = None,
                return_components: bool = False):
        seasonal, trend = self.decomp(x)                       # (B, L, C)
        y = self.lin_seasonal(seasonal.transpose(1, 2)) + self.lin_trend(
            trend.transpose(1, 2)
        )                                                      # (B, C, H)
        out = y.transpose(1, 2)                                # (B, H, C)
        if return_components:
            return out, {"time": out, "freq": out}
        return out
