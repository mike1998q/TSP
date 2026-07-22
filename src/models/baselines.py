"""Reference baselines implemented inside this framework.

Running a baseline through the exact same data pipeline, splits, training
loop, schedule, and evaluation script as the main model removes every
implementation confound from the comparison --- the fair-comparison
requirement reviewers rightly insist on. Select via ``model.arch`` in the
config (or ``--arch`` on scripts/run_main_results.py).

Faithfulness note
-----------------
The linear and Transformer baselines here are faithful reimplementations of
well-specified architectures (DLinear, NLinear, RLinear, PatchTST,
iTransformer). The Mamba baselines (S-Mamba, ms-Mamba) reuse this repo's
selective-scan encoder (:mod:`src.models.mamba_block`), which falls back to a
pure-PyTorch scan when the official ``mamba-ssm`` kernels are unavailable;
they follow the published designs (variate-token BiMamba for S-Mamba;
multi-scale parallel Mamba for ms-Mamba) but are not the authors' code.
``tf4tf`` has no faithful in-repo reimplementation and is exposed only as an
adapter over an externally supplied module (see :func:`build_baseline`). For a
camera-ready comparison, prefer dropping in the authors' official code through
that adapter; the in-repo versions are for a controlled, same-pipeline sanity
comparison.
"""
from __future__ import annotations

import importlib
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mamba_block import BiMambaEncoder, MambaEncoder
from .time_branch import SeriesDecomp


def _wrap(out: torch.Tensor, return_components: bool):
    if return_components:
        return out, {"time": out, "freq": out}
    return out


class _InstanceNorm(nn.Module):
    """RevIN-style reversible per-window, per-variate normalization.

    ``normalize`` standardizes a look-back window by its own per-variate mean
    and std (optionally with learned affine parameters); ``denormalize``
    inverts the transform on the forecast. Statistics are cached per forward
    call, the standard RevIN usage pattern.
    """

    def __init__(self, n_channels: int | None = None, affine: bool = False,
                 eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            assert n_channels is not None, "affine RevIN needs n_channels"
            self.weight = nn.Parameter(torch.ones(n_channels))
            self.bias = nn.Parameter(torch.zeros(n_channels))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:  # x: (B, L, C)
        self._mean = x.mean(1, keepdim=True).detach()
        self._std = torch.sqrt(
            x.var(1, keepdim=True, unbiased=False) + self.eps
        ).detach()
        x = (x - self._mean) / self._std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, y: torch.Tensor) -> torch.Tensor:  # y: (B, H, C)
        if self.affine:
            y = (y - self.bias) / (self.weight + self.eps)
        return y * self._std + self._mean


class DLinearBaseline(nn.Module):
    """Faithful DLinear (Zeng et al., AAAI 2023), shared-weights variant.

    Moving-average decomposition into seasonal and trend components, each
    forecast by a single linear map from the look-back window to the
    horizon. No normalization, no deep components --- the canonical linear
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
        return _wrap(y.transpose(1, 2), return_components)     # (B, H, C)


class NLinearBaseline(nn.Module):
    """Faithful NLinear (Zeng et al., AAAI 2023): subtract-last, linear, add.

    Subtracting the last look-back value before a single linear map is a
    minimal, effective handling of distribution shift on the standard
    benchmarks.
    """

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x, stats=None, return_components=False):
        last = x[:, -1:, :]                                    # (B, 1, C)
        z = self.linear((x - last).transpose(1, 2)).transpose(1, 2)
        return _wrap(z + last, return_components)


class RLinearBaseline(nn.Module):
    """RLinear (Li et al., 2023): RevIN + a single linear map.

    A reversible instance-normalized linear projection --- the strong linear
    map the review asks us to compare against directly.
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int):
        super().__init__()
        self.norm = _InstanceNorm(n_channels, affine=True)
        self.linear = nn.Linear(seq_len, pred_len)

    def forward(self, x, stats=None, return_components=False):
        z = self.norm.normalize(x)
        z = self.linear(z.transpose(1, 2)).transpose(1, 2)
        return _wrap(self.norm.denormalize(z), return_components)


class PatchTSTBaseline(nn.Module):
    """PatchTST (Nie et al., ICLR 2023): channel-independent patched Transformer.

    Each variate is normalized, split into overlapping patches, embedded, and
    processed by a shared Transformer encoder; a flatten-and-linear head maps
    the patch tokens to the horizon. Channel-independent (weights shared over
    variates).
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 d_model: int = 128, n_heads: int = 8, n_layers: int = 3,
                 patch_len: int = 16, stride: int = 8, dropout: float = 0.1,
                 d_ff: int | None = None):
        super().__init__()
        self.patch_len, self.stride = patch_len, stride
        self.norm = _InstanceNorm(n_channels, affine=True)
        n_patches = (seq_len + stride - patch_len) // stride + 1
        self.n_patches = n_patches
        self.embed = nn.Linear(patch_len, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_patches, d_model) * 0.02)
        self.dropout = nn.Dropout(dropout)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff or 2 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.head = nn.Linear(n_patches * d_model, pred_len)

    def forward(self, x, stats=None, return_components=False):
        B, L, C = x.shape
        z = self.norm.normalize(x)                             # (B, L, C)
        z = z.permute(0, 2, 1).reshape(B * C, L)               # (B*C, L)
        z = F.pad(z, (0, self.stride), mode="replicate")       # PatchTST end-pad
        patches = z.unfold(-1, self.patch_len, self.stride)    # (B*C, n_p, p_len)
        h = self.embed(patches) + self.pos                     # (B*C, n_p, d)
        h = self.encoder(self.dropout(h))                      # (B*C, n_p, d)
        y = self.head(h.reshape(B * C, -1))                    # (B*C, H)
        y = y.reshape(B, C, -1).permute(0, 2, 1)               # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


class _VariateTokenModel(nn.Module):
    """Shared skeleton for inverted (variate-token) models.

    Each variate's whole look-back series is embedded into one token; a
    sequence model mixes the ``C`` tokens; a linear head projects each token
    to the horizon. iTransformer, S-Mamba, and ms-Mamba differ only in the
    token mixer (:meth:`build_mixer`).
    """

    def __init__(self, seq_len: int, pred_len: int, n_channels: int,
                 d_model: int, dropout: float):
        super().__init__()
        self.norm = _InstanceNorm(n_channels, affine=False)
        self.embed = nn.Linear(seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(d_model, pred_len)
        self.mixer = self.build_mixer(d_model)

    def build_mixer(self, d_model: int) -> nn.Module:  # pragma: no cover
        raise NotImplementedError

    def forward(self, x, stats=None, return_components=False):
        z = self.norm.normalize(x)                             # (B, L, C)
        tok = self.embed(z.permute(0, 2, 1))                  # (B, C, d)
        tok = self.mixer(self.dropout(tok))                    # (B, C, d)
        y = self.head(tok).permute(0, 2, 1)                    # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


class ITransformerBaseline(_VariateTokenModel):
    """iTransformer (Liu et al., ICLR 2024): attention over variate tokens."""

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_heads=8,
                 n_layers=2, dropout=0.1, d_ff=None):
        self._n_heads, self._n_layers, self._d_ff = n_heads, n_layers, d_ff
        super().__init__(seq_len, pred_len, n_channels, d_model, dropout)

    def build_mixer(self, d_model):
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=self._n_heads,
            dim_feedforward=self._d_ff or 2 * d_model, dropout=0.1,
            activation="gelu", batch_first=True,
        )
        return nn.TransformerEncoder(enc_layer, num_layers=self._n_layers)


class SMambaBaseline(_VariateTokenModel):
    """S-Mamba (Wang et al., Neurocomputing 2025): BiMamba over variate tokens.

    The inverted-embedding recipe of iTransformer with the attention block
    replaced by a bidirectional Mamba mixer, matching the S-Mamba design.
    Reuses this repo's :class:`BiMambaEncoder` (official kernels when present,
    pure-PyTorch scan otherwise).
    """

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_layers=2,
                 d_state=16, dropout=0.1):
        self._n_layers, self._d_state = n_layers, d_state
        super().__init__(seq_len, pred_len, n_channels, d_model, dropout)

    def build_mixer(self, d_model):
        return BiMambaEncoder(
            d_model=d_model, n_layers=self._n_layers, d_state=self._d_state,
            use_official=True, ffn_dropout=0.1,
        )


class MsMambaBaseline(nn.Module):
    """ms-Mamba (Karadag et al., 2025): parallel Mamba blocks at multiple scales.

    The look-back is subsampled at several rates; each scale embeds its
    variate series into a token and runs a Mamba encoder, and the per-scale
    tokens are summed before a horizon head. Captures the multi-scale
    processing of ms-Mamba with this repo's selective-scan encoder.
    """

    def __init__(self, seq_len, pred_len, n_channels, d_model=128, n_layers=2,
                 d_state=16, scales=(1, 2, 4), dropout=0.1):
        super().__init__()
        self.scales = tuple(scales)
        self.norm = _InstanceNorm(n_channels, affine=False)
        self.dropout = nn.Dropout(dropout)
        self.embeds = nn.ModuleList([
            nn.Linear(math.ceil(seq_len / s), d_model) for s in self.scales
        ])
        self.encoders = nn.ModuleList([
            MambaEncoder(d_model=d_model, n_layers=n_layers, d_state=d_state,
                         use_official=True)
            for _ in self.scales
        ])
        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x, stats=None, return_components=False):
        z = self.norm.normalize(x)                             # (B, L, C)
        zc = z.permute(0, 2, 1)                                # (B, C, L)
        tok = 0.0
        for s, embed, enc in zip(self.scales, self.embeds, self.encoders):
            sub = zc[:, :, ::s]                                # (B, C, ceil(L/s))
            tok = tok + enc(self.dropout(embed(sub)))          # (B, C, d)
        y = self.head(tok).permute(0, 2, 1)                    # (B, H, C)
        return _wrap(self.norm.denormalize(y), return_components)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

#: Baselines with faithful / repo-native implementations, keyed by ``model.arch``.
BASELINE_ARCHS = (
    "dlinear", "nlinear", "rlinear", "patchtst", "itransformer",
    "smamba", "msmamba", "tf4tf",
)


def build_baseline(arch: str, cfg: dict, n_channels: int):
    """Return a baseline module for ``arch`` or ``None`` if it is not a baseline.

    Reads shared shape from ``cfg['data']`` and optional per-baseline
    hyperparameters from ``cfg['model']`` (all have sensible defaults). ``tf4tf``
    is an adapter: it requires ``model.external_impl='package.module:ClassName'``
    pointing at the authors' implementation (constructed with the same
    ``seq_len``/``pred_len``/``n_channels`` kwargs); there is intentionally no
    fabricated in-repo TF4TF.
    """
    if arch not in BASELINE_ARCHS:
        return None
    d = cfg["data"]
    m = cfg.get("model", {})
    seq_len, pred_len = d["seq_len"], d["pred_len"]
    dm = m.get("baseline_d_model", 128)
    if arch == "dlinear":
        return DLinearBaseline(seq_len, pred_len, m.get("time_kernel_size", 25))
    if arch == "nlinear":
        return NLinearBaseline(seq_len, pred_len)
    if arch == "rlinear":
        return RLinearBaseline(seq_len, pred_len, n_channels)
    if arch == "patchtst":
        return PatchTSTBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 3),
            patch_len=m.get("patch_len", 16), stride=m.get("patch_stride", 8),
        )
    if arch == "itransformer":
        return ITransformerBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 2),
        )
    if arch == "smamba":
        return SMambaBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 2),
            d_state=m.get("baseline_d_state", 16),
        )
    if arch == "msmamba":
        return MsMambaBaseline(
            seq_len, pred_len, n_channels, d_model=dm,
            n_layers=m.get("baseline_layers", 2),
            d_state=m.get("baseline_d_state", 16),
            scales=tuple(m.get("ms_scales", (1, 2, 4))),
        )
    if arch == "tf4tf":
        spec = m.get("external_impl")
        if not spec:
            raise NotImplementedError(
                "TF4TF has no faithful in-repo reimplementation. Set "
                "model.external_impl='pkg.module:ClassName' to the authors' "
                "implementation (constructed with seq_len, pred_len, "
                "n_channels), or omit tf4tf from the baseline sweep."
            )
        mod_name, _, cls_name = spec.partition(":")
        cls = getattr(importlib.import_module(mod_name), cls_name)
        return cls(seq_len=seq_len, pred_len=pred_len, n_channels=n_channels)
    return None  # pragma: no cover
