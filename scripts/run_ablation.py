#!/usr/bin/env python
"""Component-ablation suite: train controlled variants and report deltas.

Each variant changes exactly ONE component relative to the dataset's base
config, so the metric delta isolates that component's contribution. All
variants share the split, schedule, seed(s), and every other hyperparameter.

Usage
-----
    # Full suite on a dataset (auto-selects the variants that apply):
    python scripts/run_ablation.py --config configs/ETTh1.yaml

    # Subset, multiple seeds, extra overrides forwarded to every variant:
    python scripts/run_ablation.py --config configs/weather.yaml \
        --variants full time_only freq_only no_revin --seeds 3 --epochs 5

Results are written to checkpoints/ablation_<name>.json and printed as a
markdown table (mean +/- std over seeds; delta vs the full model).
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import apply_overrides, load_config, parse_overrides  # noqa: E402

# variant name -> (description, {(section, key): value})
VARIANTS = {
    "full": ("complete model (reference)", {}),
    # --- dual-branch structure ---
    "time_only": ("time branch alone (no frequency branch)",
                  {("model", "fusion"): "time_only"}),
    "freq_only": ("frequency branch alone (no time branch)",
                  {("model", "fusion"): "freq_only"}),
    "fusion_sum": ("plain average instead of the learned gate",
                   {("model", "fusion"): "sum"}),
    # --- convex vs. additive fusion ---
    # The gated/concat/sum rules are convex, so the output is confined to the
    # segment between the two branch forecasts and the branches act as
    # substitutes. These variants lift that constraint and let them superpose.
    "fusion_residual": ("additive fusion y = y_time + alpha * y_freq (scalar alpha)",
                        {("model", "fusion"): "residual"}),
    "fusion_affine": ("two independent gates (no sum-to-one constraint)",
                      {("model", "fusion"): "affine"}),
    "fusion_doubly_residual": ("freq branch fits the time branch's residual",
                               {("model", "fusion"): "doubly_residual"}),
    # --- normalization & linear anchors ---
    "no_revin": ("no instance normalization (RevIN off)",
                 {("model", "use_revin"): False}),
    # alpha-RevIN: normalization strength learned from training data instead of
    # chosen per dataset as an on/off switch (removes a test-informed choice).
    "revin_alpha_learned": ("alpha-RevIN: one learned global normalization strength",
                            {("model", "revin_alpha"): "learned"}),
    "revin_alpha_channel": ("alpha-RevIN: per-channel learned normalization strength",
                            {("model", "revin_alpha"): "channel"}),
    "no_linear_backbone": ("time branch without its DLinear backbone",
                           {("model", "time_linear_backbone"): False}),
    "rand_init": ("standard random init instead of zero-init heads/gate",
                  {("model", "zero_init"): False}),
    "no_fits": ("frequency branch without the FITS spectral backbone",
                {("model", "freq_backbone"): "none"}),
    "with_fits": ("add the FITS spectral backbone",
                  {("model", "freq_backbone"): "fits"}),
    # Separates the two things 'with_fits' conflates: silencing the frequency
    # head at init, and adding the spectral map. On backbone='none' datasets the
    # head is randomly initialized and injects noise; this variant silences it
    # at zero parameter cost and without the map's n_freq -> n_out_freq
    # extrapolation, whose ratio grows with the horizon.
    "freq_head_zero_init": ("silence the frequency head at init (no FITS map)",
                            {("model", "freq_zero_init_head"): "always"}),
    # --- Mamba components ---
    "time_mlp": ("replace the time-axis Mamba encoder with an MLP",
                 {("model", "time_encoder"): "mlp"}),
    "time_mamba": ("replace the time-axis MLP encoder with Mamba",
                   {("model", "time_encoder"): "mamba"}),
    "no_channel_mixer": ("no cross-channel (variate) Mamba mixing",
                         {("model", "channel_mixer_layers"): 0}),
    "with_channel_mixer": ("add 1 layer of cross-channel Mamba mixing",
                           {("model", "channel_mixer_layers"): 1}),
    "shared_mixer": ("variate mixer weight-tied across branches (~half mixer params)",
                     {("model", "mixer_placement"): "shared"}),
    "time_mixer_only": ("variate mixer in the time branch only (freq unmixed)",
                        {("model", "mixer_placement"): "time"}),
    "freq_mamba": ("bidirectional Mamba over frequency bins instead of the linear filter",
                   {("model", "freq_encoder"): "mamba"}),
    # --- dispersion chain (STD-style scale forecasting) ---
    "disp_base": ("base: no RevIN, no dispersion (raw de-norm)",
                  {("model", "use_revin"): False, ("model", "dispersion"): "none"}),
    "disp_revin": ("RevIN only (window mean/std de-norm)",
                   {("model", "use_revin"): True, ("model", "dispersion"): "none"}),
    "disp_fixed": ("fixed historical dispersion (longest-resolution std)",
                   {("model", "use_revin"): True, ("model", "dispersion"): "fixed"}),
    "disp_learned": ("learned dispersion head (per-horizon predicted scale)",
                     {("model", "use_revin"): True, ("model", "dispersion"): "learned"}),
}

# The dispersion ablation chain (base -> RevIN -> fixed -> learned), run as an
# explicit ordered subset: python scripts/run_ablation.py --config ... \
#     --variants disp_base disp_revin disp_fixed disp_learned --seeds 3
DISPERSION_CHAIN = ["disp_base", "disp_revin", "disp_fixed", "disp_learned"]

# Convex-vs-additive fusion study: does the frequency branch look inert because
# spectral modeling does not help here, or because the convex gate forces the
# branches to compete instead of superpose? Run as an explicit subset:
#   python scripts/run_ablation.py --config ... --variants full fusion_residual \
#       fusion_affine fusion_doubly_residual --seeds 5
FUSION_RULE_CHAIN = ["full", "fusion_sum", "fusion_residual",
                     "fusion_affine", "fusion_doubly_residual"]

# alpha-RevIN study: replaces the per-dataset on/off RevIN switch (a
# test-informed choice on Solar) with a strength learned on training data.
#   python scripts/run_ablation.py --config ... --variants full no_revin \
#       revin_alpha_learned revin_alpha_channel --seeds 5
REVIN_ALPHA_CHAIN = ["full", "no_revin", "revin_alpha_learned",
                     "revin_alpha_channel"]


def default_variants(cfg: dict) -> list:
    """Pick the variants that actually toggle something in this config."""
    m = cfg["model"]
    names = ["full", "time_only", "freq_only", "fusion_sum",
             "no_revin", "no_linear_backbone", "rand_init"]
    names.append("no_fits" if m.get("freq_backbone", "none") == "fits" else "with_fits")
    names.append("time_mlp" if m.get("time_encoder", "mamba") == "mamba" else "time_mamba")
    if m.get("channel_mixer_layers", 1) > 0:
        names.append("no_channel_mixer")
    else:
        names.append("with_channel_mixer")
    if m.get("freq_encoder", "linear") == "linear":
        names.append("freq_mamba")
    return names


def apply_variant(cfg: dict, changes: dict) -> dict:
    out = copy.deepcopy(cfg)
    for (section, key), value in changes.items():
        out[section][key] = value
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--variants", nargs="*", default=None,
                        help="Subset of variants (default: auto-select).")
    parser.add_argument("--seeds", type=int, default=1,
                        help="Seeds per variant (base seed, +1, +2, ...).")
    args, unknown = parser.parse_known_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, parse_overrides(unknown))
    base_name = base_cfg["experiment"]["name"]
    base_seed = base_cfg["experiment"]["seed"]

    names = args.variants or default_variants(base_cfg)
    unknown_names = [n for n in names if n not in VARIANTS]
    if unknown_names:
        raise SystemExit(f"Unknown variants: {unknown_names}. "
                         f"Available: {list(VARIANTS)}")
    if "full" not in names:
        names = ["full"] + names

    results = {}
    for name in names:
        desc, changes = VARIANTS[name]
        runs = []
        for s in range(args.seeds):
            cfg = apply_variant(base_cfg, changes)
            cfg["experiment"]["seed"] = base_seed + s
            cfg["experiment"]["name"] = f"{base_name}_abl_{name}_s{s}"
            print(f"\n===== variant: {name} (seed {base_seed + s}) — {desc} =====")
            out = train(cfg)
            runs.append(out["test_metrics"])
        mean = {k: sum(r[k] for r in runs) / len(runs) for k in ("mse", "mae")}
        std = {
            k: (sum((r[k] - mean[k]) ** 2 for r in runs) / len(runs)) ** 0.5
            for k in ("mse", "mae")
        }
        results[name] = {"desc": desc, "mean": mean, "std": std, "runs": runs}

    # Report.
    ref = results["full"]["mean"]
    lines = [
        f"\n## Ablation results — {base_name} "
        f"(pred_len={base_cfg['data']['pred_len']}, seeds={args.seeds})\n",
        "| variant | mse | mae | Δmse vs full | component isolated |",
        "|---|---|---|---|---|",
    ]
    for name in names:
        r = results[name]
        m, s = r["mean"], r["std"]
        dm = m["mse"] - ref["mse"]
        delta = "—" if name == "full" else f"{dm:+.4f}"
        mse_str = f"{m['mse']:.4f}" + (f" ±{s['mse']:.4f}" if args.seeds > 1 else "")
        mae_str = f"{m['mae']:.4f}" + (f" ±{s['mae']:.4f}" if args.seeds > 1 else "")
        lines.append(f"| {name} | {mse_str} | {mae_str} | {delta} | {r['desc']} |")
    report = "\n".join(lines)
    print(report)

    ckpt_dir = base_cfg["experiment"]["checkpoint_dir"]
    os.makedirs(ckpt_dir, exist_ok=True)
    out_path = os.path.join(ckpt_dir, f"ablation_{base_name}.json")
    with open(out_path, "w") as f:
        json.dump({"config": args.config, "seeds": args.seeds,
                   "results": results}, f, indent=2)
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
