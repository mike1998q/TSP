#!/usr/bin/env python
"""Unified, same-pipeline baseline comparison.

Runs DD-Mamba and every baseline through the IDENTICAL data pipeline, splits,
training loop, schedule, and evaluation for one dataset --- the fair
comparison the review requires (baselines are otherwise quoted from S-Mamba,
which mixes implementations). Each architecture is trained over ``--seeds``
seeds and ``--horizons`` horizons; results are collected into one JSON and a
markdown table.

Usage
-----
    # DD-Mamba vs the standard baselines on ETTh1, 5 seeds:
    python scripts/run_unified_baselines.py --config configs/ETTh1.yaml \
        --seeds 5 --archs dual_domain dlinear nlinear rlinear patchtst \
        itransformer smamba msmamba

    # Include TF4TF by pointing at the authors' implementation:
    python scripts/run_unified_baselines.py --config configs/ETTh1.yaml \
        --archs tf4tf --external_impl tf4tf_official.model:TF4TF

Writes checkpoints/unified_<name>.json.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import apply_overrides, load_config, parse_overrides  # noqa: E402

DEFAULT_ARCHS = ["dual_domain", "dlinear", "nlinear", "rlinear", "patchtst",
                 "itransformer", "smamba", "msmamba"]


def mean_std(xs):
    m = sum(xs) / len(xs)
    s = (sum((v - m) ** 2 for v in xs) / len(xs)) ** 0.5
    return m, s


def run_arch(base_cfg, arch, horizons, seeds, base_seed, external_impl):
    cfg0 = copy.deepcopy(base_cfg)
    cfg0["model"]["arch"] = arch
    if external_impl:
        cfg0["model"]["external_impl"] = external_impl
    per_h = {}
    for h in horizons:
        mses, maes, runs = [], [], []
        for s in range(seeds):
            cfg = copy.deepcopy(cfg0)
            cfg["data"]["pred_len"] = h
            cfg["experiment"]["seed"] = base_seed + s
            cfg["experiment"]["name"] = f"{base_cfg['experiment']['name']}_{arch}_h{h}_s{s}"
            print(f"\n===== {arch} | H={h} | seed {base_seed + s} =====")
            out = train(cfg)
            runs.append(out["test_metrics"])
            mses.append(out["test_metrics"]["mse"])
            maes.append(out["test_metrics"]["mae"])
        per_h[str(h)] = {"runs": runs, "mse": mean_std(mses), "mae": mean_std(maes)}
    return per_h


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--archs", nargs="*", default=DEFAULT_ARCHS)
    ap.add_argument("--horizons", nargs="*", type=int, default=[96, 192, 336, 720])
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--external_impl", default=None,
                    help="module:Class for the tf4tf adapter (authors' code).")
    ap.add_argument("--out", default=None)
    args, unknown = ap.parse_known_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, parse_overrides(unknown))
    name = base_cfg["experiment"]["name"]
    base_seed = base_cfg["experiment"]["seed"]

    results, failures = {}, {}
    for arch in args.archs:
        try:
            results[arch] = run_arch(base_cfg, arch, args.horizons, args.seeds,
                                     base_seed, args.external_impl)
        except NotImplementedError as e:
            print(f"[skip] {arch}: {e}")
            failures[arch] = str(e)
        except Exception as e:  # keep the sweep going; record the failure
            print(f"[error] {arch}: {e}")
            traceback.print_exc()
            failures[arch] = str(e)

    out_path = args.out or os.path.join(
        base_cfg["experiment"]["checkpoint_dir"], f"unified_{name}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"config": args.config, "seeds": args.seeds,
                   "horizons": args.horizons, "results": results,
                   "failures": failures}, f, indent=2)

    # Markdown table: avg MSE/MAE over horizons per arch.
    print(f"\n## Unified same-pipeline comparison — {name} ({args.seeds} seeds)\n")
    print("| Arch | " + " | ".join(f"H{h} MSE" for h in args.horizons) + " | Avg MSE | Avg MAE |")
    print("|---" * (len(args.horizons) + 3) + "|")
    for arch, per_h in results.items():
        cells = [f"{per_h[str(h)]['mse'][0]:.3f}" for h in args.horizons]
        avg_mse = sum(per_h[str(h)]["mse"][0] for h in args.horizons) / len(args.horizons)
        avg_mae = sum(per_h[str(h)]["mae"][0] for h in args.horizons) / len(args.horizons)
        print(f"| {arch} | " + " | ".join(cells) + f" | {avg_mse:.3f} | {avg_mae:.3f} |")
    if failures:
        print("\nSkipped/failed:", ", ".join(failures))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
