#!/usr/bin/env python
"""Paired DD-Mamba-vs-baseline analysis for the unified same-pipeline runs.

Consumes ``results/unified_<dataset>.json`` (from run_unified_baselines.py) and,
because every architecture shares seeds through the identical pipeline, computes
*paired* per-seed differences between DD-Mamba and each baseline on the
average-over-horizons MSE. Reports the effect, a t-based two-sided p-value, a
95% CI, and a Benjamini-Hochberg q-value over the whole DD-vs-baseline family,
plus a per-dataset headline (DD's margin over the best baseline and whether it
survives correction). Writes results/unified_analysis.json.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.compute_stats_correction import benjamini_hochberg, paired_test  # noqa

RESULTS = Path(__file__).resolve().parent.parent / "results"

NICE = {"configs/ETTh1.yaml": "ETTh1", "configs/ETTh2.yaml": "ETTh2",
        "configs/ETTm1.yaml": "ETTm1", "configs/ETTm2.yaml": "ETTm2",
        "configs/weather.yaml": "Weather", "configs/electricity.yaml": "Electricity",
        "configs/solar.yaml": "Solar", "configs/traffic.yaml": "Traffic",
        "configs/exchange_rate.yaml": "Exchange"}


def per_seed_avg_mse(cell_by_h, horizons):
    """Average the per-horizon MSE within each seed -> one value per seed."""
    n = len(cell_by_h[str(horizons[0])]["runs"])
    out = []
    for s in range(n):
        out.append(sum(cell_by_h[str(h)]["runs"][s]["mse"] for h in horizons) / len(horizons))
    return out


def load_unified():
    """Find unified files under results/ (or the repo root as uploaded)."""
    paths = sorted(glob.glob(str(RESULTS / "unified_*.json")))
    if not paths:
        paths = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "JSON*.json")))
    datasets = {}
    for p in paths:
        d = json.load(open(p))
        datasets[NICE.get(d["config"], d["config"])] = d
    return datasets


def main():
    datasets = load_unified()
    if not datasets:
        raise SystemExit("No unified_*.json found in results/ (or JSON*.json at root).")

    rows = []
    per_dataset = {}
    for ds, d in datasets.items():
        res, hor = d["results"], d["horizons"]
        archs = [a for a in res if a != "dual_domain"]
        dd = per_seed_avg_mse(res["dual_domain"], hor)
        dd_avg = sum(dd) / len(dd)
        base_avgs = {}
        for a in archs:
            b = per_seed_avg_mse(res[a], hor)
            base_avgs[a] = sum(b) / len(b)
            # paired diff (baseline - DD): >0 means DD is better
            r = paired_test(dd, b)
            rows.append(dict(dataset=ds, baseline=a, dd_avg_mse=dd_avg,
                             baseline_avg_mse=base_avgs[a], **r))
        best = min(archs, key=lambda a: base_avgs[a])
        per_dataset[ds] = dict(dd_avg_mse=dd_avg, best_baseline=best,
                               best_baseline_mse=base_avgs[best],
                               margin=base_avgs[best] - dd_avg, seeds=len(dd),
                               horizons=hor)

    qs = benjamini_hochberg([r["p"] for r in rows])
    for r, q in zip(rows, qs):
        r["q_bh"] = q

    out = {"n_comparisons": len(rows), "per_dataset": per_dataset, "rows": rows}
    (RESULTS / "unified_analysis.json").write_text(json.dumps(out, indent=2))

    print("=" * 82)
    print(f"{'dataset':12s} {'DD avg':>7s} {'best base':>16s} {'margin':>8s} "
          f"{'vs-best p':>9s} {'q_bh':>7s}  verdict")
    for ds, pd in per_dataset.items():
        # find the DD-vs-best-baseline row
        br = next(r for r in rows if r["dataset"] == ds and r["baseline"] == pd["best_baseline"])
        verdict = ("DD better*" if (pd["margin"] > 0 and br["q_bh"] < 0.05)
                   else "DD better (ns)" if pd["margin"] > 0 else "baseline better")
        print(f"{ds:12s} {pd['dd_avg_mse']:7.3f} "
              f"{pd['best_baseline']+' '+format(pd['best_baseline_mse'],'.3f'):>16s} "
              f"{pd['margin']:+8.4f} {br['p']:9.4f} {br['q_bh']:7.4f}  {verdict}")

    n_win = sum(1 for pd in per_dataset.values() if pd["margin"] > 0)
    n_sig = sum(1 for ds, pd in per_dataset.items()
                if pd["margin"] > 0 and next(
                    r for r in rows if r["dataset"] == ds
                    and r["baseline"] == pd["best_baseline"])["q_bh"] < 0.05)
    print(f"\nDD-Mamba has lowest avg MSE on {n_win}/{len(per_dataset)} datasets; "
          f"beats the best baseline at BH q<0.05 on {n_sig}.")
    print(f"[saved] {RESULTS / 'unified_analysis.json'}")


if __name__ == "__main__":
    main()
