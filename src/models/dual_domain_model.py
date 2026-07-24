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

from .dispersion import DispersionHead
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
        mixer_placement: str = "both",
        use_revin: bool = True,
        time_linear_backbone: bool = True,
        zero_init: bool = True,
        dispersion: str = "none",
        dispersion_resolutions: tuple = (),
        dispersion_hidden: int = 64,
    ):
        super().__init__()
        self.use_revin = use_revin
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.dispersion = dispersion

        # Cross-variate mixer placement. The mixer is the dominant parameter
        # cost on high-channel datasets (it scales with d_model^2 and is
        # otherwise instantiated in *both* branches), so this switch controls
        # where it lives:
        #   'both'   -- independent mixer in each branch (default; unchanged).
        #   'time'   -- mixer only in the time branch (freq branch has none).
        #   'freq'   -- mixer only in the frequency branch.
        #   'shared' -- one mixer, weight-tied across both branches (both
        #               branches still mix, at ~half the mixer parameters).
        if mixer_placement not in ("both", "time", "freq", "shared"):
            raise ValueError(
                f"Unknown mixer_placement: {mixer_placement!r} "
                "(use both, time, freq, shared)"
            )
        self.mixer_placement = mixer_placement
        time_mix = channel_mixer_layers if mixer_placement != "freq" else 0
        freq_mix = channel_mixer_layers if mixer_placement != "time" else 0

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
            channel_mixer_layers=time_mix,
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
            channel_mixer_layers=freq_mix,
            zero_init=zero_init,
        )
        # Weight-tie the two mixers when sharing (both branches were built with
        # a mixer; point the frequency branch at the time branch's instance).
        if mixer_placement == "shared" and channel_mixer_layers > 0:
            self.freq_branch.channel_mixer = self.time_branch.channel_mixer
        self.fusion = ForecastFusion(d_model=d_model, pred_len=pred_len,
                                     mode=fusion, zero_init=zero_init)

        # Optional learned dispersion head: predicts a positive per-horizon
        # scale from multi-resolution historical statistics (2 features -- mean,
        # std -- per resolution). At init it reduces to RevIN de-normalization.
        self.disp_head = None
        if dispersion == "learned":
            stats_dim = 2 * len(dispersion_resolutions)
            self.disp_head = DispersionHead(
                stats_dim=stats_dim, pred_len=pred_len, hidden=dispersion_hidden,
                dropout=head_dropout,
            )

    def forward(
        self, x: torch.Tensor, stats: torch.Tensor = None,
        return_components: bool = False,
    ):
        """x: (B, L, C) -> forecast (B, H, C).

        Each instance is standardized by its own window mean/std (RevIN-style).
        The forecast is reconstructed as ``Y = T_hat + D_hat (.) S_hat`` where
        ``S_hat`` is the normalized-space fused forecast. With
        ``dispersion='none'`` the reconstruction is exactly RevIN
        (``T_hat=mean``, ``D_hat=std``); with ``'learned'`` the scale/location
        are predicted per horizon by the dispersion head from ``stats``; with
        ``'fixed'`` the scale is the longest-resolution historical std.

        ``stats``: (B, S, C) multi-resolution history stats, or None.
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
        y = self.fusion(time_feat, freq_feat, y_time, y_freq)  # (B,C,H) = S_hat

        mu = mean.transpose(1, 2)                      # (B, C, 1)
        sigma = std.transpose(1, 2)                    # (B, C, 1)
        loc, scale = self._reconstruct(mu, sigma, stats)   # (B,C,H) or (B,C,1)
        out = (loc + scale * y).transpose(1, 2)        # (B, H, C)
        if not return_components:
            return out
        components = {
            "time": (loc + scale * y_time).transpose(1, 2),
            "freq": (loc + scale * y_freq).transpose(1, 2),
            "scale": scale, "loc": loc,
        }
        return out, components

    def _reconstruct(self, mu, sigma, stats):
        """Return (location T_hat, scale D_hat) for de-normalization."""
        if self.dispersion == "learned" and self.disp_head is not None:
            r_hat, t_hat = self.disp_head(stats.transpose(1, 2))  # (B,C,H)
            return mu + t_hat, sigma * r_hat
        if self.dispersion == "fixed" and stats is not None and stats.shape[1] > 0:
            # Longest-resolution historical std (last row) as a fixed scale.
            d_fixed = stats[:, -1, :].unsqueeze(-1).clamp_min(1e-4)  # (B,C,1)
            return mu, d_fixed
        return mu, sigma


def build_model(cfg: dict, n_channels: int):
    """Construct the model from a parsed config dict.

    ``model.arch`` selects the architecture: ``dual_domain`` (default) or an
    in-framework baseline (``dlinear``) trained/evaluated through the
    identical pipeline for fair comparison.
    """
    mcfg = cfg["model"]
    dcfg = cfg["data"]
    arch = mcfg.get("arch", "dual_domain")
    if arch != "dual_domain":
        from .baselines import BASELINE_ARCHS, build_baseline

        model = build_baseline(arch, cfg, n_channels)
        if model is not None:
            return model
        raise ValueError(
            f"Unknown model.arch: {arch!r} "
            f"(expected 'dual_domain' or one of {BASELINE_ARCHS})"
        )
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
        mixer_placement=mcfg.get("mixer_placement", "both"),
        use_revin=mcfg.get("use_revin", True),
        time_linear_backbone=mcfg.get("time_linear_backbone", True),
        zero_init=mcfg.get("zero_init", True),
        dispersion=mcfg.get("dispersion", "none"),
        dispersion_resolutions=tuple(
            mcfg.get("dispersion_resolutions", [dcfg["seq_len"], 144, 288, 336])
        ),
        dispersion_hidden=mcfg.get("dispersion_hidden", 64),
    )
