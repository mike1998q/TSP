"""Factory helpers that turn a config dict into ready-to-use DataLoaders."""
from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader

from .dataset import ETT_BORDERS, Scaler, build_splits, load_raw_series


def get_dataloaders(
    cfg: dict,
) -> Tuple[DataLoader, DataLoader, DataLoader, Scaler, int]:
    """Build train/val/test DataLoaders and the fitted Scaler.

    Returns
    -------
    train_loader, val_loader, test_loader, scaler, n_channels
    """
    dcfg = cfg["data"]
    tcfg = cfg["train"]

    data = load_raw_series(
        source=dcfg["source"],
        csv_path=dcfg.get("csv_path"),
        target_columns=dcfg.get("target_columns"),
        synthetic_length=dcfg.get("synthetic_length", 8000),
        synthetic_channels=dcfg.get("synthetic_channels", 7),
        seed=cfg["experiment"]["seed"],
    )
    n_channels = data.shape[1]

    protocol = dcfg.get("split_protocol", "ratio")
    if protocol == "ratio":
        borders = None
    elif protocol in ETT_BORDERS:
        borders = ETT_BORDERS[protocol]
    else:
        raise ValueError(
            f"Unknown data.split_protocol: {protocol!r} (use ratio, ETTh, ETTm)"
        )

    train_ds, val_ds, test_ds, scaler = build_splits(
        data=data,
        seq_len=dcfg["seq_len"],
        pred_len=dcfg["pred_len"],
        train_ratio=dcfg["train_ratio"],
        val_ratio=dcfg["val_ratio"],
        scale=dcfg.get("scale", True),
        borders=borders,
    )

    common = dict(
        batch_size=tcfg["batch_size"],
        num_workers=tcfg.get("num_workers", 0),
        pin_memory=True,
        drop_last=False,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)
    return train_loader, val_loader, test_loader, scaler, n_channels
