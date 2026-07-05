"""Dual-domain forecaster: parallel time- and frequency-domain branches.

Architecture
------------
             input window (B, L, C)
                 /            \\
        TimeBranch          FreqBranch
    (decomp + Mamba SSM)  (rFFT + spectral filter)
         (B,C,D)             (B,C,D)
                 \\            /
                FeatureFusion (gated)
                     (B,C,D)
                        |
                 forecast head
                     (B,C,H)  ->  (B,H,C)

The model is channel-independent: every variable/channel shares the same
weights and is processed independently, which is a strong, robust baseline for
multivariate long-horizon forecasting.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .freq_branch import FreqBranch
from .fusion import FeatureFusion
from .time_branch import SeriesDecomp, TimeBranch


class DualDomainForecaster(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        d_model: int = 128,
        time_kernel_size: int = 25,
        time_dropout: float = 0.1,
        time_encoder: str = "mamba",
        mamba_layers: int = 2,
        mamba_d_state: int = 16,
        mamba_d_conv: int = 4,
        mamba_expand: int = 2,
        use_official_mamba: bool = True,
        freq_hidden: int = 128,
        freq_dropout: float = 0.1,
        freq_sparsity: float = 0.0,
        freq_encoder: str = "linear",
        fusion: str = "gated",
        head_dropout: float = 0.1,
        direct_skip: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels

        self.time_branch = TimeBranch(
            seq_len=seq_len,
            d_model=d_model,
            kernel_size=time_kernel_size,
            dropout=time_dropout,
            encoder=time_encoder,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            use_official_mamba=use_official_mamba,
        )
        self.freq_branch = FreqBranch(
            seq_len=seq_len,
            d_model=d_model,
            freq_hidden=freq_hidden,
            dropout=freq_dropout,
            sparsity=freq_sparsity,
            encoder=freq_encoder,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            use_official_mamba=use_official_mamba,
        )
        self.fusion = FeatureFusion(d_model=d_model, mode=fusion)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(head_dropout),
            nn.Linear(d_model, pred_len),
        )

        # DLinear-style direct linear skip: per-component linear maps from the
        # look-back window straight to the horizon. On benchmarks like ETT a
        # plain linear history->future map is a near-SOTA baseline; without
        # it, forecasting from a pooled d_model summary alone collapses toward
        # mean-reverting predictions. The deep dual-domain path then learns a
        # *correction*: its head is zero-initialized so training starts from
        # exactly the linear forecast and adds nonlinearity only as needed.
        self.direct_skip = direct_skip
        if direct_skip:
            self.direct_decomp = SeriesDecomp(time_kernel_size)
            self.direct_seasonal = nn.Linear(seq_len, pred_len)
            self.direct_trend = nn.Linear(seq_len, pred_len)
            nn.init.zeros_(self.head[-1].weight)
            nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, C) -> forecast (B, H, C).

        We standardize each instance by its own last-window mean/std
        (reversible instance normalization, RevIN-style) to handle
        distribution shift, then undo it on the output.
        """
        # Instance normalization (per sample, per channel).
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_norm = (x - mean) / std

        time_feat = self.time_branch(x_norm)   # (B, C, D)
        freq_feat = self.freq_branch(x_norm)    # (B, C, D)
        fused = self.fusion(time_feat, freq_feat)  # (B, C, D)

        out = self.head(fused)                  # (B, C, H)
        out = out.transpose(1, 2)               # (B, H, C)

        if self.direct_skip:
            seasonal, trend = self.direct_decomp(x_norm)          # (B, L, C)
            direct = self.direct_seasonal(seasonal.transpose(1, 2)) + self.direct_trend(
                trend.transpose(1, 2)
            )                                                     # (B, C, H)
            out = out + direct.transpose(1, 2)                    # (B, H, C)

        # De-normalize.
        out = out * std + mean
        return out


def build_model(cfg: dict, n_channels: int) -> DualDomainForecaster:
    """Construct the model from a parsed config dict."""
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    return DualDomainForecaster(
        seq_len=dcfg["seq_len"],
        pred_len=dcfg["pred_len"],
        n_channels=n_channels,
        d_model=mcfg["d_model"],
        time_kernel_size=mcfg["time_kernel_size"],
        time_dropout=mcfg["time_dropout"],
        time_encoder=mcfg.get("time_encoder", "mamba"),
        mamba_layers=mcfg.get("mamba_layers", 2),
        mamba_d_state=mcfg.get("mamba_d_state", 16),
        mamba_d_conv=mcfg.get("mamba_d_conv", 4),
        mamba_expand=mcfg.get("mamba_expand", 2),
        use_official_mamba=mcfg.get("use_official_mamba", True),
        freq_hidden=mcfg["freq_hidden"],
        freq_dropout=mcfg["freq_dropout"],
        freq_sparsity=mcfg.get("freq_sparsity", 0.0),
        freq_encoder=mcfg.get("freq_encoder", "linear"),
        fusion=mcfg["fusion"],
        head_dropout=mcfg["head_dropout"],
        direct_skip=mcfg.get("direct_skip", True),
    )
