"""Frequency-domain branch: learnable spectral filtering via rFFT.

The window is transformed to the frequency domain with a real FFT, then a
complex-valued linear filter (implemented as two real matrices acting on the
real and imaginary parts) reweights and mixes frequency bins. This lets the
model learn global, periodic structure with a receptive field spanning the
whole look-back window at O(L log L) cost -- complementary to the local view
of the time-domain branch. The idea follows FreTS/FEDformer-style frequency
MLPs.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ComplexLinear(nn.Module):
    """Apply a complex linear map y = W x over the last (frequency) dim.

    W = W_r + i W_i acts on x = x_r + i x_i giving:
        y_r = W_r x_r - W_i x_i
        y_i = W_r x_i + W_i x_r
    Implemented with real weights so it runs on any backend.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.wr = nn.Linear(in_features, out_features)
        self.wi = nn.Linear(in_features, out_features)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        yr = self.wr(xr) - self.wi(xi)
        yi = self.wr(xi) + self.wi(xr)
        return yr, yi


class FreqBranch(nn.Module):
    """Encode a look-back window into a per-channel feature of size d_model.

    Input:  (B, L, C)
    Output: (B, C, d_model)
    """

    def __init__(
        self,
        seq_len: int,
        d_model: int,
        freq_hidden: int = 128,
        dropout: float = 0.1,
        sparsity: float = 0.0,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.n_freq = seq_len // 2 + 1  # rFFT output length
        self.sparsity = float(sparsity)

        # Complex spectral filter that mixes frequency bins.
        self.filter = ComplexLinear(self.n_freq, self.n_freq)
        # Map the (real|imag) spectrum to the shared feature width.
        self.proj = nn.Sequential(
            nn.Linear(2 * self.n_freq, freq_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(freq_hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def _low_pass_mask(self, device) -> torch.Tensor:
        """Optionally zero out the highest `sparsity` fraction of frequencies."""
        if self.sparsity <= 0.0:
            return torch.ones(self.n_freq, device=device)
        keep = int(round(self.n_freq * (1.0 - self.sparsity)))
        keep = max(1, keep)
        mask = torch.zeros(self.n_freq, device=device)
        mask[:keep] = 1.0
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C) -> operate along time (dim=1).
        xc = x.transpose(1, 2)                       # (B, C, L)
        spec = torch.fft.rfft(xc, dim=-1, norm="ortho")  # (B, C, n_freq) complex
        xr, xi = spec.real, spec.imag

        mask = self._low_pass_mask(x.device)
        xr = xr * mask
        xi = xi * mask

        yr, yi = self.filter(xr, xi)                 # learned spectral mixing
        feat = torch.cat([yr, yi], dim=-1)           # (B, C, 2*n_freq)
        feat = self.proj(feat)                       # (B, C, d_model)
        return self.norm(feat)
