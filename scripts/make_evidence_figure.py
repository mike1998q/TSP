#!/usr/bin/env python
"""Generate the dual-domain evidence figure (paper Fig. 2).

Two panels:
  (a) Learned fusion gate on a trained model: the per-variate convex weight
      the gate assigns to the *time* branch, averaged over the test set,
      alongside the initialization value g0 and the equal-mix line. Shows the
      gate moves off its time-heavy init toward the branch the data favors.
  (b) Per-branch contribution across datasets: the paired increase in MSE
      when each branch is removed (freq contribution = time-only minus full;
      time contribution = freq-only minus full), with t-based 95% CIs.
      A bar whose CI clears zero is a branch the ablation certifies as
      load-bearing on that dataset.

Panel (a) needs a trained checkpoint (checkpoints/<name>_best.pt); if absent
the panel is drawn empty with a note, so the figure can be built from the
ablation JSONs alone.

Usage
-----
    python scripts/make_evidence_figure.py \
        --gate-ckpt checkpoints/ETTh1_best.pt --gate-name ETTh1 \
        --out paper/fig_evidence.pdf
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# t critical value, two-sided 95%, df = n-1.
T_CRIT = {2: 4.302653, 3: 3.182446, 4: 2.776445, 5: 2.570582}

# Colorblind-safe (Okabe-Ito): time = blue, freq = vermillion.
C_TIME = "#0072B2"
C_FREQ = "#D55E00"


def paired_delta(full_runs, var_runs):
    """Paired per-seed (variant - full) mean and t-based 95% half-width."""
    d = np.array([v - f for f, v in zip(full_runs, var_runs)], dtype=float)
    n = len(d)
    mean = float(d.mean())
    if n < 2:
        return mean, float("nan")
    sd = float(d.std(ddof=1))
    half = T_CRIT.get(n, 4.303) * sd / math.sqrt(n)
    return mean, half


def load_contributions(paths):
    """For each dataset JSON, return time- and freq-branch contributions."""
    out = []
    for name, path in paths:
        res = json.load(open(path))["results"]
        full = [r["mse"] for r in res["full"]["runs"]]
        # freq contribution: removing freq = time_only variant.
        fmean, fhalf = paired_delta(full, [r["mse"] for r in res["time_only"]["runs"]])
        # time contribution: removing time = freq_only variant.
        tmean, thalf = paired_delta(full, [r["mse"] for r in res["freq_only"]["runs"]])
        out.append({"name": name, "time": (tmean, thalf), "freq": (fmean, fhalf)})
    return out


def extract_gate(ckpt_path):
    """Load a trained model and return per-variate mean time-branch gate."""
    import torch

    from src.data.data_loader import get_dataloaders
    from src.models.dual_domain_model import build_model

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = state["config"]
    cfg["train"]["num_workers"] = 0
    n_channels = state["n_channels"]
    _, _, test_loader, _, _ = get_dataloaders(cfg)
    model = build_model(cfg, n_channels)
    model.load_state_dict(state["model_state"])
    model.eval()

    captured = {}

    def hook(_module, _inp, out):
        captured["g"] = out.detach()  # (B, C, 1)

    handle = model.fusion.gate.register_forward_hook(hook)
    sums = torch.zeros(n_channels)
    count = 0
    with torch.no_grad():
        for x, _ in test_loader:
            model(x)
            g = captured["g"].squeeze(-1)  # (B, C)
            sums += g.sum(dim=0)
            count += g.shape[0]
    handle.remove()
    return (sums / count).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/fig_evidence.pdf")
    ap.add_argument("--gate-ckpt", default="checkpoints/ETTh1_best.pt")
    ap.add_argument("--gate-name", default="ETTh1")
    ap.add_argument("--gate-init", type=float, default=0.90)
    args = ap.parse_args()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 8,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#444444",
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "pdf.fonttype": 42,
    })

    contrib = load_contributions([
        ("ETTh1", "results/Ablation_ETTh1.json"),
        ("Solar", "results/Ablation_solar_final.json"),
        ("Weather", "results/Ablation_weather.json"),
    ])

    gate = None
    if Path(args.gate_ckpt).exists():
        try:
            gate = extract_gate(args.gate_ckpt)
        except Exception as e:  # pragma: no cover
            print(f"[warn] gate extraction failed: {e}")

    fig, (axa, axb) = plt.subplots(1, 2, figsize=(6.6, 2.5))

    # ---- Panel (a): learned gate ----
    if gate is not None:
        ch = np.arange(len(gate))
        axa.bar(ch, gate, color=C_TIME, width=0.68, label="time-branch weight $\\bar g_c$")
        axa.axhline(args.gate_init, ls="--", lw=0.8, color="#888888")
        axa.axhline(0.5, ls=":", lw=0.8, color="#bbbbbb")
        axa.text(len(gate) - 0.4, args.gate_init + 0.01, "init $g_0$",
                 ha="right", va="bottom", fontsize=6.5, color="#666666")
        axa.text(len(gate) - 0.4, 0.5 + 0.01, "equal mix",
                 ha="right", va="bottom", fontsize=6.5, color="#999999")
        axa.set_ylim(0, 1.0)
        axa.set_xticks(ch)
        axa.set_xlabel(f"{args.gate_name} variate index")
        axa.set_ylabel("gate $\\bar g_c$ (time weight)")
        axa.set_title("(a) Learned fusion gate", fontsize=8.5, loc="left")
    else:
        axa.text(0.5, 0.5, "gate panel: no checkpoint", ha="center", va="center",
                 transform=axa.transAxes, fontsize=7, color="#999999")
        axa.set_title("(a) Learned fusion gate", fontsize=8.5, loc="left")
        axa.set_xticks([])
        axa.set_yticks([])

    # ---- Panel (b): per-branch contribution ----
    names = [c["name"] for c in contrib]
    x = np.arange(len(names))
    w = 0.36
    tvals = [c["time"][0] for c in contrib]
    terr = [c["time"][1] for c in contrib]
    fvals = [c["freq"][0] for c in contrib]
    ferr = [c["freq"][1] for c in contrib]

    axb.axhline(0, lw=0.7, color="#444444")
    b1 = axb.bar(x - w / 2, tvals, w, yerr=terr, capsize=2.5, color=C_TIME,
                 error_kw={"elinewidth": 0.8, "capthick": 0.8},
                 label="time branch")
    b2 = axb.bar(x + w / 2, fvals, w, yerr=ferr, capsize=2.5, color=C_FREQ,
                 error_kw={"elinewidth": 0.8, "capthick": 0.8},
                 label="freq branch")

    # significance stars: CI excludes zero (lower bound > 0).
    for xi, (m, h) in [(x[i] - w / 2, contrib[i]["time"]) for i in range(len(x))]:
        if m - h > 0:
            axb.text(xi, m + h + 0.001, "*", ha="center", va="bottom", fontsize=10)
    for xi, (m, h) in [(x[i] + w / 2, contrib[i]["freq"]) for i in range(len(x))]:
        if m - h > 0:
            axb.text(xi, m + h + 0.001, "*", ha="center", va="bottom", fontsize=10)

    axb.set_xticks(x)
    axb.set_xticklabels(names)
    axb.set_ylabel("$\\Delta$MSE when branch removed")
    axb.set_title("(b) Per-branch contribution (paired, 95% CI)",
                  fontsize=8.5, loc="left")
    axb.legend(frameon=False, fontsize=7, loc="upper left")
    axb.margins(y=0.18)

    for ax in (axa, axb):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout(w_pad=1.5)
    out = Path(args.out)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=200, bbox_inches="tight")
    print(f"[saved] {out} and {out.with_suffix('.png')}")

    # Echo the numbers that go in the caption/text.
    for c in contrib:
        print(f"{c['name']:8s} time Δ={c['time'][0]:+.4f}±{c['time'][1]:.4f}  "
              f"freq Δ={c['freq'][0]:+.4f}±{c['freq'][1]:.4f}")
    if gate is not None:
        print(f"{args.gate_name} gate per variate:",
              np.array2string(gate, precision=3, floatmode="fixed"))


if __name__ == "__main__":
    main()
