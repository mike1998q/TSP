"""A self-contained Mamba (selective state-space) block.

This is a pure-PyTorch implementation of the Mamba-1 selective SSM
(Gu & Dao, 2023) that runs on any backend (CPU or GPU) with no custom CUDA
kernels. If the official ``mamba-ssm`` package *is* installed, ``MambaLayer``
transparently uses its fused, hardware-aware kernels instead for speed.

Why a pure fallback matters here: the official ``mamba-ssm`` /
``causal-conv1d`` kernels are frequently hard to build against brand-new
stacks (e.g. RTX 5090 / Blackwell sm_120 + CUDA 13.0). The pure path lets the
model train today; drop in the fast kernels later with zero code changes.

Shapes throughout: (B, L, D) where D == d_model.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fast path
    from mamba_ssm import Mamba as _OfficialMamba  # type: ignore

    _HAS_MAMBA_SSM = True
except Exception:  # pragma: no cover - depends on environment
    _OfficialMamba = None
    _HAS_MAMBA_SSM = False


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute the statistics in fp32 (autocast treats nn.LayerNorm this
        # way automatically, but not custom norms like this one).
        xf = x.float()
        norm = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (norm * self.weight.float()).to(x.dtype)


class MambaSSM(nn.Module):
    """Pure-PyTorch selective SSM core (the Mamba mixer)."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        conv_bias: bool = True,
        bias: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = expand * d_model
        self.dt_rank = dt_rank or math.ceil(d_model / 16)

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=bias)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            bias=conv_bias,
        )
        # Projects x -> (delta, B, C) parameters of the selective SSM.
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # A is kept in log space and negated to guarantee stability (Re(A) < 0).
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)

    def _selective_scan(
        self,
        u: torch.Tensor,       # (B, L, d_inner)
        delta: torch.Tensor,   # (B, L, d_inner)
        A: torch.Tensor,       # (d_inner, d_state)
        B: torch.Tensor,       # (B, L, d_state)
        C: torch.Tensor,       # (B, L, d_state)
        D: torch.Tensor,       # (d_inner,)
    ) -> torch.Tensor:
        b, l, d_in = u.shape
        n = A.shape[1]
        # Zero-order-hold discretization.
        deltaA = torch.exp(delta.unsqueeze(-1) * A)               # (B, L, d_in, n)
        deltaB_u = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (B,L,d_in,n)

        h = torch.zeros(b, d_in, n, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(l):
            h = deltaA[:, t] * h + deltaB_u[:, t]                 # (B, d_in, n)
            y = torch.einsum("bdn,bn->bd", h, C[:, t])           # (B, d_in)
            ys.append(y)
        y = torch.stack(ys, dim=1)                                # (B, L, d_in)
        return y + u * D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The recurrent scan (exp(delta*A) + L-step state accumulation) is
        # numerically fragile in fp16/bf16: under AMP it under/overflows and
        # poisons training with NaNs. Run the whole mixer in fp32 and cast
        # back, mirroring what the official fused kernels do internally.
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            return self._forward_fp32(x.float()).to(in_dtype)

    def _forward_fp32(self, x: torch.Tensor) -> torch.Tensor:
        b, l, _ = x.shape
        x_res = self.in_proj(x)                                   # (B, L, 2*d_inner)
        x_in, res = x_res.chunk(2, dim=-1)

        # Causal depthwise conv over the sequence.
        x_in = x_in.transpose(1, 2)                               # (B, d_inner, L)
        x_in = self.conv1d(x_in)[..., :l]                         # trim right pad
        x_in = x_in.transpose(1, 2)                               # (B, L, d_inner)
        x_in = F.silu(x_in)

        A = -torch.exp(self.A_log.float())                        # (d_inner, d_state)
        x_dbl = self.x_proj(x_in)                                 # (B, L, dt_rank+2n)
        delta, B_mat, C_mat = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )
        delta = F.softplus(self.dt_proj(delta))                   # (B, L, d_inner)

        y = self._selective_scan(x_in, delta, A, B_mat, C_mat, self.D)
        y = y * F.silu(res)
        return self.out_proj(y)


class MambaLayer(nn.Module):
    """Pre-norm residual Mamba layer: x + Mamba(RMSNorm(x)).

    Uses the official ``mamba_ssm.Mamba`` mixer when available (and running on
    CUDA), otherwise the pure-PyTorch :class:`MambaSSM`.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_official: bool = True,
    ):
        super().__init__()
        self.norm = RMSNorm(d_model)
        self._official = None
        if use_official and _HAS_MAMBA_SSM:
            self._official = _OfficialMamba(
                d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand
            )
            self.mixer = None
        else:
            self.mixer = MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dt_rank=dt_rank,
            )

    @property
    def using_official_kernels(self) -> bool:
        return self._official is not None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm(x)
        mix = self._official(normed) if self._official is not None else self.mixer(normed)
        return x + mix


class BiMambaEncoder(nn.Module):
    """Bidirectional Mamba: forward scan + backward scan, fused per layer.

    Mamba's scan is inherently directional. Over the *time* axis that is the
    right bias (causality), but over axes with no arrow of time — e.g. the
    frequency bins of a spectrum — a single direction is arbitrary. Each layer
    here runs two mixers, one on the sequence and one on its reverse, and sums
    them inside the residual so every position sees both sides.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_official: bool = True,
    ):
        super().__init__()
        self.fwd_layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    use_official=use_official,
                )
                for _ in range(n_layers)
            ]
        )
        self.bwd_layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    use_official=use_official,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    @property
    def using_official_kernels(self) -> bool:
        return any(l.using_official_kernels for l in self.fwd_layers) or any(
            l.using_official_kernels for l in self.bwd_layers
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for fwd, bwd in zip(self.fwd_layers, self.bwd_layers):
            # MambaLayer returns x + mix(norm(x)); combining both directions
            # and subtracting one x keeps a single residual stream.
            x = fwd(x) + bwd(x.flip(1)).flip(1) - x
        return self.norm(x)


class MambaEncoder(nn.Module):
    """A stack of :class:`MambaLayer` blocks with a final norm."""

    def __init__(
        self,
        d_model: int,
        n_layers: int = 2,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: Optional[int] = None,
        use_official: bool = True,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                MambaLayer(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    dt_rank=dt_rank,
                    use_official=use_official,
                )
                for _ in range(n_layers)
            ]
        )
        self.norm = RMSNorm(d_model)

    @property
    def using_official_kernels(self) -> bool:
        return any(layer.using_official_kernels for layer in self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)
