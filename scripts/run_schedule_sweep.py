#!/usr/bin/env python3
"""Training-budget sweep: is the model undertrained, or is it converged?

Motivation
----------
The shipped recipe uses ``lr_scheduler: halve`` (lr = base * 0.5^(epoch-1)).
That is a geometric series, so the *total* learning budget is

    sum_{e>=1} 0.5^(e-1) = 2 * base

**bounded regardless of the epoch count** -- ``epochs: 10`` and ``epochs: 100``
deliver an identical budget, and only ~4 epochs ever run above 10% of the base
rate. The Electricity training logs show the consequence directly: training
loss is still falling at the final epoch on every horizon, and at H=96 the
validation loss is pinned to the same value for four consecutive epochs
because the weights have effectively stopped moving.

This script settles whether that costs accuracy, by running the same model
under several schedules and reporting test MSE per horizon. It is the
"training-budget ablation" named in DESIGN_LIMITATIONS_LITERATURE.md as the
experiment that determines whether the component-ablation nulls are real or
artifacts of the recipe.

Usage
-----
    # the headline comparison on Electricity
    python scripts/run_schedule_sweep.py --config configs/electricity.yaml \
        --horizons 96 192 336 720 --seeds 1

    # add seeds once a winner is apparent
    python scripts/run_schedule_sweep.py --config configs/electricity.yaml \
        --horizons 96 720 --seeds 3 --arms halve cosine_warmup_30

Results are written to results/schedule_sweep_<name>.json.

Interpretation
--------------
* If ``cosine_warmup_30`` beats ``halve`` at the short horizons, the reported
  numbers were budget-limited and the recipe should change (for *every*
  dataset, since they all share it -- so re-run the full suite before touching
  the manuscript).
* If it does not, ``halve`` is vindicated and the accuracy ceiling is
  architectural, which is a materially different paper.
* H=720 is expected to behave differently: its logs show genuine overfitting
  (best validation at epoch 2, then monotone degradation), so a larger budget
  should *not* help there and more regularization is the lever instead.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.train import train  # noqa: E402
from src.utils import load_config  # noqa: E402

# arm name -> train-section overrides
ARMS = {
    # current recipe: budget capped at 2x base lr
    "halve": {"lr_scheduler": "halve", "epochs": 10},
    # same epoch count, ~2.8x the budget
    "cosine_10": {"lr_scheduler": "cosine", "epochs": 10},
    # warmup + cosine at the current epoch count
    "cosine_warmup_10": {"lr_scheduler": "cosine_warmup", "epochs": 10,
                         "warmup_epochs": 1.0},
    # the main candidate: ~5x the budget, still early-stopped
    "cosine_warmup_30": {"lr_scheduler": "cosine_warmup", "epochs": 30,
                         "warmup_epochs": 2.0, "patience": 6},
    # for the horizons that overfit (H=720): more regularization, not more lr
    "halve_reg": {"lr_scheduler": "halve", "epochs": 10,
                  "weight_decay": 1e-3},
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--horizons", type=int, nargs="+", default=[96, 192, 336, 720])
    ap.add_argument("--arms", nargs="+", default=["halve", "cosine_warmup_30"],
                    choices=sorted(ARMS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    base = load_config(args.config)
    name = base["experiment"]["name"]
    rows = []

    for arm in args.arms:
        for h in args.horizons:
            for s in range(args.seeds):
                cfg = copy.deepcopy(base)
                cfg["data"]["pred_len"] = h
                cfg["train"].update(ARMS[arm])
                cfg["experiment"]["seed"] = base["experiment"]["seed"] + s
                cfg["experiment"]["name"] = f"{name}_{arm}_h{h}_s{s}"
                print(f"\n=== {name} | arm={arm} | H={h} | seed offset {s} ===")
                res = train(cfg)
                rows.append({
                    "arm": arm, "horizon": h, "seed_offset": s,
                    "test_mse": res["test_metrics"]["mse"],
                    "test_mae": res["test_metrics"]["mae"],
                    "best_val": res["best_val_loss"],
                })

    out = Path(args.out or ROOT / "results" / f"schedule_sweep_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"config": args.config, "rows": rows}, open(out, "w"), indent=1)

    print(f"\n[saved] {out}\n")
    print(f"{'arm':20s} {'H':>5s} {'test MSE':>10s} {'test MAE':>10s}")
    for arm in args.arms:
        for h in args.horizons:
            sel = [r for r in rows if r["arm"] == arm and r["horizon"] == h]
            if not sel:
                continue
            mse = sum(r["test_mse"] for r in sel) / len(sel)
            mae = sum(r["test_mae"] for r in sel) / len(sel)
            print(f"{arm:20s} {h:5d} {mse:10.4f} {mae:10.4f}")
    # headline: average over horizons per arm
    print()
    for arm in args.arms:
        sel = [r for r in rows if r["arm"] == arm]
        if sel:
            print(f"  {arm:20s} avg MSE over horizons = "
                  f"{sum(r['test_mse'] for r in sel)/len(sel):.4f}")


if __name__ == "__main__":
    main()
