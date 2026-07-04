"""Smoke tests for shapes and a tiny end-to-end training step (CPU-only)."""
from __future__ import annotations

import numpy as np
import torch

from src.data.dataset import (
    SlidingWindowDataset,
    build_splits,
    generate_synthetic,
)
from src.models import DualDomainForecaster
from src.models.freq_branch import FreqBranch
from src.models.time_branch import TimeBranch


def test_time_branch_shape():
    b, l, c, d = 4, 96, 7, 32
    x = torch.randn(b, l, c)
    branch = TimeBranch(seq_len=l, d_model=d)
    out = branch(x)
    assert out.shape == (b, c, d)


def test_freq_branch_shape():
    b, l, c, d = 4, 96, 7, 32
    x = torch.randn(b, l, c)
    branch = FreqBranch(seq_len=l, d_model=d)
    out = branch(x)
    assert out.shape == (b, c, d)


def test_model_forward_shape():
    b, l, h, c = 4, 96, 24, 7
    x = torch.randn(b, l, c)
    model = DualDomainForecaster(seq_len=l, pred_len=h, n_channels=c, d_model=32)
    out = model(x)
    assert out.shape == (b, h, c)


def test_fusion_modes():
    b, l, h, c = 2, 48, 12, 3
    x = torch.randn(b, l, c)
    for mode in ("gated", "sum", "concat"):
        model = DualDomainForecaster(
            seq_len=l, pred_len=h, n_channels=c, d_model=16, fusion=mode
        )
        assert model(x).shape == (b, h, c)


def test_dataset_windowing():
    data = generate_synthetic(length=500, channels=3, seed=1)
    ds = SlidingWindowDataset(data, seq_len=96, pred_len=24)
    assert len(ds) == 500 - 96 - 24 + 1
    x, y = ds[0]
    assert x.shape == (96, 3)
    assert y.shape == (24, 3)


def test_build_splits_no_leakage_shapes():
    data = generate_synthetic(length=2000, channels=4, seed=2)
    train, val, test, scaler = build_splits(
        data, seq_len=96, pred_len=24, train_ratio=0.7, val_ratio=0.1, scale=True
    )
    assert len(train) > 0 and len(val) > 0 and len(test) > 0
    # Scaler should roughly standardize the training slice.
    scaled = scaler.transform(data[: int(len(data) * 0.7)])
    assert abs(scaled.mean()) < 0.1


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    b, l, h, c = 16, 48, 12, 2
    x = torch.randn(b, l, c)
    # Target correlated with input so the model can actually learn.
    y = x[:, -h:, :] * 0.5
    model = DualDomainForecaster(seq_len=l, pred_len=h, n_channels=c, d_model=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    first = None
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
        if first is None:
            first = loss.item()
    assert loss.item() < first
