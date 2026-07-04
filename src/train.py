"""Training entry point for the dual-domain forecaster.

Usage
-----
    python -m src.train --config configs/default.yaml
    python -m src.train --epochs 5 --seq_len 192 --pred_len 96 --device cuda
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .data import get_dataloaders
from .models import build_model
from .utils import (
    add_config_args,
    all_metrics,
    apply_overrides,
    describe_device,
    get_device,
    load_config,
    parse_overrides,
    set_seed,
)


def get_loss_fn(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss(delta=1.0)
    raise ValueError(f"Unknown loss: {name!r}")


def build_scheduler(optimizer, cfg, steps_per_epoch):
    kind = cfg["train"].get("lr_scheduler", "none")
    epochs = cfg["train"]["epochs"]
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs * steps_per_epoch)
        )
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, epochs // 3), gamma=0.5
        )
    return None


@torch.no_grad()
def evaluate(model, loader, loss_fn, device) -> dict:
    model.eval()
    preds, trues, losses = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        losses.append(loss_fn(out, y).item())
        preds.append(out.cpu().numpy())
        trues.append(y.cpu().numpy())
    preds = np.concatenate(preds, axis=0)
    trues = np.concatenate(trues, axis=0)
    metrics = all_metrics(preds, trues)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def train(cfg: dict) -> dict:
    set_seed(cfg["experiment"]["seed"])
    device = get_device(cfg["train"].get("device", "auto"))
    print(f"[device] {describe_device(device)}")

    train_loader, val_loader, test_loader, scaler, n_channels = get_dataloaders(cfg)
    print(
        f"[data] channels={n_channels} | "
        f"train={len(train_loader.dataset)} "
        f"val={len(val_loader.dataset)} "
        f"test={len(test_loader.dataset)} windows"
    )

    model = build_model(cfg, n_channels).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] DualDomainForecaster | trainable params={n_params:,}")

    if cfg["train"].get("compile", False):
        try:
            model = torch.compile(model)
            print("[model] torch.compile enabled")
        except Exception as e:  # pragma: no cover
            print(f"[model] torch.compile failed ({e}); continuing eager.")

    loss_fn = get_loss_fn(cfg["train"]["loss"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))

    use_amp = bool(cfg["train"].get("amp", False)) and device.type == "cuda"
    scaler_amp = torch.amp.GradScaler("cuda", enabled=use_amp)
    grad_clip = cfg["train"].get("grad_clip", 0.0)

    ckpt_dir = cfg["experiment"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    best_path = os.path.join(ckpt_dir, f"{cfg['experiment']['name']}_best.pt")

    best_val = float("inf")
    patience = cfg["train"].get("patience", 10)
    bad_epochs = 0

    for epoch in range(1, cfg["train"]["epochs"] + 1):
        model.train()
        running = 0.0
        t0 = time.time()
        pbar = tqdm(train_loader, desc=f"epoch {epoch:02d}", leave=False)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                out = model(x)
                loss = loss_fn(out, y)
            scaler_amp.scale(loss).backward()
            if grad_clip and grad_clip > 0:
                scaler_amp.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler_amp.step(optimizer)
            scaler_amp.update()
            if scheduler is not None:
                scheduler.step()
            running += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        train_loss = running / max(1, len(train_loader))
        val_metrics = evaluate(model, val_loader, loss_fn, device)
        dt = time.time() - t0
        print(
            f"epoch {epoch:02d} | train_loss={train_loss:.4f} | "
            f"val_mse={val_metrics['mse']:.4f} val_mae={val_metrics['mae']:.4f} "
            f"| {dt:.1f}s"
        )

        if val_metrics["loss"] < best_val - 1e-6:
            best_val = val_metrics["loss"]
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": cfg,
                    "n_channels": n_channels,
                    "val_metrics": val_metrics,
                    "epoch": epoch,
                },
                best_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"[early-stop] no val improvement for {patience} epochs.")
                break

    # Final test evaluation on the best checkpoint.
    if os.path.exists(best_path):
        state = torch.load(best_path, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
    test_metrics = evaluate(model, test_loader, loss_fn, device)
    print(
        f"[test] mse={test_metrics['mse']:.4f} mae={test_metrics['mae']:.4f} "
        f"rmse={test_metrics['rmse']:.4f} mape={test_metrics['mape']:.4f}"
    )

    results = {
        "best_val_loss": best_val,
        "test_metrics": test_metrics,
        "checkpoint": best_path,
    }
    with open(os.path.join(ckpt_dir, f"{cfg['experiment']['name']}_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results


def main():
    parser = argparse.ArgumentParser(description="Train the dual-domain forecaster.")
    add_config_args(parser)
    args, unknown = parser.parse_known_args()
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, parse_overrides(unknown))
    train(cfg)


if __name__ == "__main__":
    main()
