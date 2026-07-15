#!/usr/bin/env python
"""Measure the engineering profile of DD-Mamba per dataset config.

Reports hardware-independent metrics only --- model size, per-forecast
compute, and forward activation footprint --- so the numbers are properties
of the model and input shapes, not of the accelerator. Wall-clock throughput
on the target GPU is intentionally out of scope here.

Columns
-------
  params      trainable parameters (millions)
  MACs        multiply-accumulates for one forecast, batch=1 (millions):
              nn.Linear + nn.Conv1d + the selective-scan recurrence
  act. mem    sum of forward activation tensor bytes at batch=1 (MB, fp32);
              an upper-bound proxy for peak activation memory

Usage
-----
    python scripts/profile_efficiency.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.models.dual_domain_model import build_model  # noqa: E402
from src.models.mamba_block import MambaSSM  # noqa: E402
from src.utils import load_config  # noqa: E402

# (label, config, representative channel count)
CONFIGS = [
    ("ETTh1", "configs/ETTh1.yaml", 7),
    ("Exchange", "configs/exchange_rate.yaml", 8),
    ("Weather", "configs/weather.yaml", 21),
    ("Solar", "configs/solar.yaml", 137),
    ("Electricity", "configs/electricity.yaml", 321),
    ("Traffic", "configs/traffic.yaml", 862),
]


def profile(cfg, n_channels):
    model = build_model(cfg, n_channels).eval()
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    macs = {"v": 0}
    act_bytes = {"v": 0}
    handles = []

    def lin_hook(m, inp, out):
        macs["v"] += out.numel() * m.in_features
        act_bytes["v"] += out.numel() * out.element_size()

    def conv_hook(m, inp, out):
        k = m.kernel_size[0]
        macs["v"] += out.numel() * (m.in_channels // m.groups) * k
        act_bytes["v"] += out.numel() * out.element_size()

    def scan_hook(m, inp, out):
        # Recurrence: deltaA*h + deltaB_u and the C-einsum, per step.
        x = inp[0]
        b, l, _ = x.shape
        macs["v"] += 2 * b * l * m.d_inner * m.d_state
        act_bytes["v"] += out.numel() * out.element_size()

    for mod in model.modules():
        if isinstance(mod, nn.Linear):
            handles.append(mod.register_forward_hook(lin_hook))
        elif isinstance(mod, nn.Conv1d):
            handles.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, MambaSSM):
            handles.append(mod.register_forward_hook(scan_hook))

    seq_len = cfg["data"]["seq_len"]
    x = torch.randn(1, seq_len, n_channels)
    # Feed dummy statistics when a dispersion head is active.
    stats = None
    disp = cfg["model"].get("dispersion", "none")
    if disp in ("fixed", "learned"):
        res = cfg["model"].get("dispersion_resolutions", [seq_len, 144, 288, 336])
        stats = torch.randn(1, 2 * len(res), n_channels).abs()
    with torch.no_grad():
        model(x, stats=stats)
    for h in handles:
        h.remove()

    return {
        "params": params,
        "params_m": params / 1e6,
        "macs_m": macs["v"] / 1e6,
        "act_mb": act_bytes["v"] / 1e6,
    }


def profile_dispersion_overhead(path, n_channels):
    """Return (base, learned) profiles so the caller can report the head's
    extra parameters / compute / memory --- the efficiency test for the
    dispersion experiment."""
    base_cfg = load_config(path)
    base_cfg["data"]["pred_len"] = 96
    base_cfg.setdefault("model", {})["dispersion"] = "none"
    disp_cfg = load_config(path)
    disp_cfg["data"]["pred_len"] = 96
    disp_cfg.setdefault("model", {})["dispersion"] = "learned"
    disp_cfg["model"]["dispersion_resolutions"] = [96, 144, 288, 336]
    return profile(base_cfg, n_channels), profile(disp_cfg, n_channels)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--dispersion", action="store_true",
                    help="report the dispersion head's efficiency overhead")
    args = ap.parse_args()

    if args.dispersion:
        print(f"{'dataset':12s} {'C':>4s}  {'base params':>12s} "
              f"{'+head params':>13s} {'overhead':>9s}  {'base GMACs':>10s} "
              f"{'+head GMACs':>11s}  {'+head act(MB)':>13s}")
        for label, path, ch in CONFIGS:
            base, disp = profile_dispersion_overhead(path, ch)
            dp = disp["params"] - base["params"]
            frac = dp / base["params"]
            print(f"{label:12s} {ch:4d}  {base['params']:12,d} "
                  f"{dp:13,d} {frac:8.2%}  {base['macs_m']/1e3:10.3f} "
                  f"{disp['macs_m']/1e3:11.3f}  {disp['act_mb']-base['act_mb']:13.2f}")
        print("\nThe dispersion head adds a fixed number of parameters "
              "(independent of channel count C): it is shared across variates, "
              "so its relative cost shrinks as C grows.")
        return

    rows = []
    for label, path, ch in CONFIGS:
        cfg = load_config(path)
        cfg["data"]["pred_len"] = 96
        r = profile(cfg, ch)
        rows.append((label, ch, r))
        print(f"{label:12s} C={ch:4d}  params={r['params_m']:7.3f}M  "
              f"MACs/forecast={r['macs_m']:8.1f}M  act={r['act_mb']:7.1f}MB")

    print("\n% LaTeX rows (label & C & params(M) & MACs(M) & act(MB)):")
    for label, ch, r in rows:
        print(f"{label} & {ch} & {r['params_m']:.2f} & "
              f"{r['macs_m']:.0f} & {r['act_mb']:.1f} \\\\")


if __name__ == "__main__":
    main()
