#!/usr/bin/env python
"""Branch-ablation matrix: full / time-only / freq-only on every dataset.

Motivation
----------
The paper's frequency-branch (dual-domain) evidence currently rests on a
single cell: Solar, H=96, time-only, +0.0114 MSE with a 95% CI barely
excluding zero. One marginal result on one dataset at one horizon cannot
carry the conclusion. This script replaces "Solar as the single test" with
the full matrix: for each dataset and horizon it trains the full model and
both single-branch variants under shared seeds, then reports paired per-seed
deltas with t-based 95% CIs for every cell.

Run on the training GPU (hours of compute; not a CPU job):

    python scripts/run_branch_matrix.py                    # all 9 datasets, H=96, 3 seeds
    python scripts/run_branch_matrix.py --horizons 96 192 336 720
    python scripts/run_branch_matrix.py --datasets ETTh1 solar weather --seeds 3

Each (dataset, horizon) cell reuses the ablation harness with variants
full / time_only / freq_only, so results are directly comparable to the
existing tables. Output: checkpoints/branch_matrix.json plus a rendered
markdown/LaTeX summary with one row per cell and a star when the paired CI
excludes zero.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import load_config  # noqa: E402

DATASETS = {
    "ETTh1": "configs/ETTh1.yaml",
    "ETTh2": "configs/ETTh2.yaml",
    "ETTm1": "configs/ETTm1.yaml",
    "ETTm2": "configs/ETTm2.yaml",
    "weather": "configs/weather.yaml",
    "solar": "configs/solar.yaml",
    "electricity": "configs/electricity.yaml",
    "traffic": "configs/traffic.yaml",
    "exchange_rate": "configs/exchange_rate.yaml",
}
VARIANTS = {"full": None, "time_only": "time_only", "freq_only": "freq_only"}
T_CRIT3 = 4.302653  # two-sided 95%, df=2


def paired_ci(full_runs, var_runs):
    d = [v - f for f, v in zip(full_runs, var_runs)]
    n = len(d)
    mean = sum(d) / n
    if n < 2:
        return mean, float("nan"), float("nan")
    sd = math.sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    half = T_CRIT3 * sd / math.sqrt(n)
    return mean, mean - half, mean + half


def run_cell(cfg_path, horizon, seeds):
    base = load_config(cfg_path)
    base["data"]["pred_len"] = horizon
    name = base["experiment"]["name"]
    seed0 = base["experiment"]["seed"]
    cell = {}
    for variant, fusion in VARIANTS.items():
        runs = []
        for s in range(seeds):
            cfg = copy.deepcopy(base)
            if fusion is not None:
                cfg["model"]["fusion"] = fusion
            cfg["experiment"]["seed"] = seed0 + s
            cfg["experiment"]["name"] = f"{name}_bm_{variant}_H{horizon}_s{s}"
            print(f"\n===== {name} H={horizon} {variant} seed={seed0 + s} =====",
                  flush=True)
            out = train(cfg)
            runs.append(out["test_metrics"]["mse"])
        cell[variant] = runs
    return cell


def summarize(results):
    md = ["| dataset | H | freq removed Δ [95% CI] | time removed Δ [95% CI] |",
          "|---|---|---|---|"]
    tex = []
    for key, cell in sorted(results.items()):
        ds, h = key.rsplit("@", 1)
        rows = {}
        for variant in ("time_only", "freq_only"):
            m, lo, hi = paired_ci(cell["full"], cell[variant])
            star = "*" if (lo > 0 or hi < 0) else ""
            rows[variant] = f"{m:+.4f} [{lo:+.4f},{hi:+.4f}]{star}"
        md.append(f"| {ds} | {h} | {rows['time_only']} | {rows['freq_only']} |")
        tex.append(f"{ds} & {h} & {rows['time_only']} & {rows['freq_only']} \\\\")
    return "\n".join(md), "\n".join(tex)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets", nargs="*", default=list(DATASETS),
                    choices=list(DATASETS))
    ap.add_argument("--horizons", nargs="*", type=int, default=[96])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="checkpoints/branch_matrix.json")
    args = ap.parse_args()

    results = {}
    if os.path.exists(args.out):  # resume support
        results = json.load(open(args.out)).get("results", {})
        print(f"[resume] loaded {len(results)} finished cells from {args.out}")

    for ds in args.datasets:
        for h in args.horizons:
            key = f"{ds}@{h}"
            if key in results:
                print(f"[skip] {key} already done")
                continue
            results[key] = run_cell(DATASETS[ds], h, args.seeds)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump({"seeds": args.seeds, "results": results}, f, indent=2)
            print(f"[saved] {args.out} ({len(results)} cells)")

    md, tex = summarize(results)
    print("\n## Branch-ablation matrix (paired ΔMSE vs full; * = CI excludes 0)\n")
    print(md)
    print("\n% LaTeX rows:\n" + tex)


if __name__ == "__main__":
    main()
