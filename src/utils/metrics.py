"""Forecasting metrics computed on NumPy arrays of shape (N, H, C)."""
from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - true)))


def mse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.mean((pred - true) ** 2))


def rmse(pred: np.ndarray, true: np.ndarray) -> float:
    return float(np.sqrt(mse(pred, true)))


def mape(pred: np.ndarray, true: np.ndarray, eps: float = 1e-7) -> float:
    return float(np.mean(np.abs((pred - true) / (np.abs(true) + eps))))


def all_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    return {
        "mse": mse(pred, true),
        "mae": mae(pred, true),
        "rmse": rmse(pred, true),
        "mape": mape(pred, true),
    }
