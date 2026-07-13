"""Dual-domain forecaster: parallel time- and frequency-domain branches.

Architecture
------------
             input window (B, L, C)  --RevIN-->  x_norm
                 /                       \\
        TimeBranch                     FreqBranch
   decomp -> linear backbone       rFFT -> spectral encoder
   + Mamba correction (0-init)     (linear filter | BiMamba)
   -> y_time, feat_t               -> y_freq, feat_f
                 \\                       /
              ForecastFusion (gate from [feat_t, feat_f])
              y = g * y_time + (1 - g) * y_freq
                        |
                 de-normalize  ->  forecast (B, H, C)

Every prediction is produced *inside* a branch and fused as a convex
combination — there is no path that bypasses the dual-branch architecture.
The time branch carries a DLinear-style linear backbone internally (critical
on benchmarks like ETT), with its deep head zero-initialized so training
starts from an exact linear forecaster and learns corrections.

Per-channel processing shares weights across variables. With
``channel_mixer_layers > 0`` (default 1), each branch additionally runs a
bidirectional Mamba across the *variate* dimension (S-Mamba style) before its
head, so forecasts can exploit cross-channel dependencies — the main edge of
first-class multivariate models on benchmarks like electricity and weather.
Set it to 0 for a strictly channel-independent model.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .freq_branch import FreqBranch
from .fusion import ForecastFusion
from .time_branch import TimeBranch


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
        freq_backbone: str = "none",
        fusion: str = "gated",
        head_dropout: float = 0.1,
        channel_mixer_layers: int = 1,
        use_revin: bool = True,
        time_linear_backbone: bool = True,
        zero_init: bool = True,
    ):
        super().__init__()
        self.use_revin = use_revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels

        self.time_branch = TimeBranch(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            kernel_size=time_kernel_size,
            dropout=time_dropout,
            head_dropout=head_dropout,
            encoder=time_encoder,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            use_official_mamba=use_official_mamba,
            channel_mixer_layers=channel_mixer_layers,
            use_linear_backbone=time_linear_backbone,
            zero_init=zero_init,
        )
        self.freq_branch = FreqBranch(
            seq_len=seq_len,
            pred_len=pred_len,
            d_model=d_model,
            freq_hidden=freq_hidden,
            dropout=freq_dropout,
            head_dropout=head_dropout,
            sparsity=freq_sparsity,
            encoder=freq_encoder,
            backbone=freq_backbone,
            mamba_layers=mamba_layers,
            mamba_d_state=mamba_d_state,
            mamba_d_conv=mamba_d_conv,
            mamba_expand=mamba_expand,
            use_official_mamba=use_official_mamba,
            channel_mixer_layers=channel_mixer_layers,
            zero_init=zero_init,
        )
        self.fusion = ForecastFusion(d_model=d_model, pred_len=pred_len,
                                     mode=fusion, zero_init=zero_init)

    def forward(
        self, x: torch.Tensor, return_components: bool = False
    ):
        """x: (B, L, C) -> forecast (B, H, C).

        We standardize each instance by its own last-window mean/std
        (reversible instance normalization, RevIN-style) to handle
        distribution shift, then undo it on the output.

        With ``return_components=True`` also returns the de-normalized
        per-branch forecasts and the fusion gate for analysis/ablation.
        """
        # Instance normalization (per sample, per channel).
        if self.use_revin:
            mean = x.mean(dim=1, keepdim=True)
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_norm = (x - mean) / std
        else:
            mean = torch.zeros_like(x[:, :1])
            std = torch.ones_like(x[:, :1])
            x_norm = x

        time_feat, y_time = self.time_branch(x_norm)   # (B,C,D), (B,C,H)
        freq_feat, y_freq = self.freq_branch(x_norm)   # (B,C,D), (B,C,H)
        y = self.fusion(time_feat, freq_feat, y_time, y_freq)  # (B,C,H)

        out = y.transpose(1, 2) * std + mean           # (B, H, C)
        if not return_components:
            return out
        components = {
            "time": y_time.transpose(1, 2) * std + mean,
            "freq": y_freq.transpose(1, 2) * std + mean,
        }
        return out, components


def build_model(cfg: dict, n_channels: int):
    """Construct the model from a parsed config dict.

    ``model.arch`` selects the architecture: ``dual_domain`` (default) or an
    in-framework baseline (``dlinear``) trained/evaluated through the
    identical pipeline for fair comparison.
    """
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    arch = mcfg.get("arch", "dual_domain")
    if arch == "dlinear":
        from .baselines import DLinearBaseline

        return DLinearBaseline(
            seq_len=dcfg["seq_len"],
            pred_len=dcfg["pred_len"],
            kernel_size=mcfg.get("time_kernel_size", 25),
        )
    if arch != "dual_domain":
        raise ValueError(f"Unknown model.arch: {arch!r}")
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
        freq_backbone=mcfg.get("freq_backbone", "none"),
        fusion=mcfg["fusion"],
        head_dropout=mcfg["head_dropout"],
        channel_mixer_layers=mcfg.get("channel_mixer_layers", 1),
        use_revin=mcfg.get("use_revin", True),
        time_linear_backbone=mcfg.get("time_linear_backbone", True),
        zero_init=mcfg.get("zero_init", True),
    )
