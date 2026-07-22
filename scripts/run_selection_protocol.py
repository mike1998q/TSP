#!/usr/bin/env python
"""Validation-only switch selection with a labeled test-oracle bound.

Addresses the model-selection-bias threat: instead of choosing switches (e.g.
RevIN on/off, mixer depth, spectral map) after inspecting *test* ablations,
this script fixes a candidate switch space up front, selects on *validation*
loss only, freezes the choice, and then reports three clearly labeled models:

  * FIXED           -- one configuration applied as-is (no per-dataset tuning).
  * VALIDATION-SELECTED -- the candidate with the lowest mean validation loss
                       (the honest, deployable choice).
  * TEST-ORACLE     -- the candidate with the lowest mean test MSE. This is an
                       UPPER BOUND that peeks at the test set; it is reported
                       only to quantify how much headroom selection bias could
                       have added, and must never be presented as a result.

Each candidate is run over ``--seeds`` seeds and ``--horizons`` horizons; the
selection uses validation loss aggregated over both.

Usage
-----
    python scripts/run_selection_protocol.py --config configs/solar.yaml \
        --horizons 96 --seeds 3 \
        --switch model.use_revin=true,false \
        --switch model.channel_mixer_layers=0,1 \
        --fixed model.use_revin=true,model.channel_mixer_layers=0

Writes checkpoints/selection_<name>.json.
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.train import train  # noqa: E402
from src.utils import apply_overrides, load_config, parse_overrides  # noqa: E402


def parse_value(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def parse_assignments(text: str) -> dict:
    """'model.use_revin=true,model.channel_mixer_layers=0' -> dict of path->value."""
    out = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, _, val = item.partition("=")
        out[key.strip()] = parse_value(val.strip())
    return out


def set_by_path(cfg: dict, dotted: str, value) -> None:
    section, _, key = dotted.partition(".")
    if not key:  # bare key defaults to the model section
        section, key = "model", section
    cfg.setdefault(section, {})[key] = value


def apply_candidate(cfg: dict, candidate: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    for path, value in candidate.items():
        set_by_path(cfg, path, value)
    return cfg


def candidate_label(candidate: dict) -> str:
    return ", ".join(f"{k.split('.')[-1]}={v}" for k, v in candidate.items()) or "default"


def eval_candidate(base_cfg, candidate, horizons, seeds, base_seed, tag):
    """Run one candidate over horizons x seeds; return aggregated metrics."""
    per_h = {}
    val_losses, test_mses, test_maes = [], [], []
    for h in horizons:
        vloss, tmse, tmae = [], [], []
        for s in range(seeds):
            cfg = apply_candidate(base_cfg, candidate)
            cfg["data"]["pred_len"] = h
            cfg["experiment"]["seed"] = base_seed + s
            cfg["experiment"]["name"] = f"{tag}_h{h}_s{s}"
            out = train(cfg)
            vloss.append(out["best_val_loss"])
            tmse.append(out["test_metrics"]["mse"])
            tmae.append(out["test_metrics"]["mae"])
        mean = lambda xs: sum(xs) / len(xs)
        per_h[h] = {"val_loss": mean(vloss), "test_mse": mean(tmse),
                    "test_mae": mean(tmae)}
        val_losses += vloss
        test_mses += tmse
        test_maes += tmae
    mean = lambda xs: sum(xs) / len(xs)
    return {
        "candidate": candidate,
        "label": candidate_label(candidate),
        "per_horizon": per_h,
        "val_loss": mean(val_losses),   # selection criterion
        "test_mse": mean(test_mses),
        "test_mae": mean(test_maes),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--horizons", nargs="*", type=int, default=[96])
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--switch", action="append", default=[],
                    help="dotted.path=v1,v2[,v3]; repeatable. Candidate space is "
                         "the Cartesian product of all --switch options.")
    ap.add_argument("--fixed", default=None,
                    help="assignments for the FIXED model (defaults to the "
                         "config's own values).")
    ap.add_argument("--out", default=None)
    args, unknown = ap.parse_known_args()

    base_cfg = load_config(args.config)
    base_cfg = apply_overrides(base_cfg, parse_overrides(unknown))
    base_name = base_cfg["experiment"]["name"]
    base_seed = base_cfg["experiment"]["seed"]

    # Build candidate grid from --switch options.
    switch_axes = {}  # path -> [values]
    for spec in args.switch:
        key, _, vals = spec.partition("=")
        switch_axes[key.strip()] = [parse_value(v.strip()) for v in vals.split(",")]
    if not switch_axes:
        raise SystemExit("Provide at least one --switch path=v1,v2")

    paths = list(switch_axes)
    grid = [dict(zip(paths, combo))
            for combo in itertools.product(*(switch_axes[p] for p in paths))]
    n_runs = len(grid) * len(args.horizons) * args.seeds
    print(f"[selection] {len(grid)} candidates x {len(args.horizons)} horizons "
          f"x {args.seeds} seeds = {n_runs} training runs")

    results = []
    for i, cand in enumerate(grid):
        print(f"\n===== candidate {i+1}/{len(grid)}: {candidate_label(cand)} =====")
        tag = f"{base_name}_sel{i}"
        results.append(eval_candidate(base_cfg, cand, args.horizons,
                                      args.seeds, base_seed, tag))

    # FIXED model.
    fixed_assign = parse_assignments(args.fixed) if args.fixed else {}
    fixed = eval_candidate(base_cfg, fixed_assign, args.horizons, args.seeds,
                           base_seed, f"{base_name}_fixed")

    val_selected = min(results, key=lambda r: r["val_loss"])
    test_oracle = min(results, key=lambda r: r["test_mse"])

    report = {
        "config": args.config,
        "horizons": args.horizons,
        "seeds": args.seeds,
        "switch_space": {k: v for k, v in switch_axes.items()},
        "candidates": results,
        "models": {
            "fixed": {k: fixed[k] for k in ("label", "candidate", "val_loss",
                                            "test_mse", "test_mae", "per_horizon")},
            "validation_selected": {k: val_selected[k] for k in
                                    ("label", "candidate", "val_loss", "test_mse",
                                     "test_mae", "per_horizon")},
            "test_oracle_upper_bound": {k: test_oracle[k] for k in
                                        ("label", "candidate", "val_loss", "test_mse",
                                         "test_mae", "per_horizon")},
        },
    }

    out_path = args.out or os.path.join(
        base_cfg["experiment"]["checkpoint_dir"], f"selection_{base_name}.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 70)
    print(f"{'model':22s} {'switches':28s} {'val':>8s} {'test MSE':>9s}")
    for name, m in report["models"].items():
        print(f"{name:22s} {m['label'][:28]:28s} {m['val_loss']:8.4f} "
              f"{m['test_mse']:9.4f}")
    gap = report["models"]["validation_selected"]["test_mse"] - \
        report["models"]["test_oracle_upper_bound"]["test_mse"]
    print(f"\nselection gap (val-selected - oracle) = {gap:+.4f} MSE "
          "(how much test-peeking could have added)")
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
