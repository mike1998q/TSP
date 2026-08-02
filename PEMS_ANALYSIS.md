# PEMS: Why Accuracy Is Poor, and What Changed

Analysis of the four-dataset × four-horizon training logs.

## Results as run

| dataset | H=12 | H=24 | H=48 | H=96 |
|---|---|---|---|---|
| PEMS03 | 0.0674 | 0.0951 | 0.1536 | 0.2452 |
| PEMS04 | 0.0846 | 0.1137 | 0.1848 | 0.2832 |
| PEMS07 | 0.0658 | 0.0956 | 0.1519 | 0.2287 |
| PEMS08 | 0.0784 | 0.1111 | 0.1923 | 0.3173 |

Error grows 3.6–4.0× from H=12 to H=96, much faster than on the hourly
benchmarks.

## Root cause: the model is UNDERFITTING, and it is my fault

This is the opposite of the Electricity regime, so the earlier schedule-sweep
conclusion does **not** transfer. Comparing final training loss to best
validation loss:

| dataset | H=12 | H=24 | H=48 | H=96 | regime |
|---|---|---|---|---|---|
| PEMS03 | **1.023** | **0.997** | 0.957 | 0.940 | **underfit** at short H |
| PEMS07 | **1.032** | **1.005** | 0.959 | 0.912 | **underfit** at short H |
| PEMS04 | 0.910 | 0.902 | 0.896 | 0.896 | mild |
| PEMS08 | 0.898 | 0.800 | 0.724 | **0.616** | overfits with horizon |
| *Electricity* | *0.864* | — | — | *0.755 (H=720)* | *overfits* |

On PEMS03 and PEMS07 the **training loss is at or above the validation loss**.
The model is not even fitting the training split. That is underfitting, and
more optimization is exactly what it needs — unlike Electricity, where the
sweep showed a 5× budget bought nothing because it already overfits.

### The mechanism

Two changes I made earlier to stop the NaN divergence over-corrected:

1. **`lr` 5e-4 → 2e-4.** I lowered it alongside disabling AMP. But the actual
   cause of the NaN was fp16 overflow in the variate mixer, which `amp: false`
   already fixes on its own. The learning-rate cut was redundant.
2. **`d_model` 512 → 256.** Halves the variate-mixer capacity on datasets with
   170–883 channels.

Combined with `halve`'s bounded budget (2 × base regardless of epoch count),
the total learning budget became **2 × 2e-4 = 4.0e-4** — only 40% of what
Electricity gets, at half the width.

### The logs show the waste directly

`halve` at base 2e-4 drives the learning rate to absurd values:

| epoch | 1 | 5 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| lr | 2.0e-4 | 1.25e-5 | 3.9e-7 | 3.8e-10 | **3.7e-13** |

**PEMS03 H=24 ran 30 epochs with the training loss frozen at 0.0859 from epoch
9 onward — 21 epochs of literally zero progress.** PEMS04 H=12 is the same
story (frozen at 0.0798 from epoch 11). This is the bounded-budget property
of `halve` doing visible damage rather than merely being theoretically
suboptimal.

(Note: the repo config said `epochs: 10` while the logs ran to 30, so the run
used an override. Either way the extra epochs achieved nothing, which is itself
the proof.)

## Changes made

Applied uniformly to all four PEMS configs — they are one benchmark family and
should share a protocol; differentiating per dataset on test behaviour would be
the selection bias the paper criticizes.

| setting | before | after |
|---|---|---|
| `lr` | 2e-4 | **5e-4** (restored; AMP-off already handles the NaN) |
| `lr_scheduler` | `halve` | **`cosine_warmup`** |
| `warmup_epochs` | — | **2** |
| `epochs` | 10 | **30** |
| `patience` | 5 | **6** |
| `amp` | false | false (unchanged — this is the real NaN fix) |
| `grad_clip` | 5.0 | 5.0 (unchanged) |

Effective learning rate per epoch now:

| epoch | 1 | 3 | 10 | 20 | 30 |
|---|---|---|---|---|---|
| lr | 5.0e-6 | 5.0e-4 | 4.3e-4 | 1.7e-4 | 1.5e-6 |

**Total budget: 4.0e-4 → 7.5e-3, a 19× increase.**

### Why this is lower-risk than my earlier failed changes

- **Early stopping still guards the overfitting cases.** PEMS08 overfits with
  horizon; with `patience: 6` and best-checkpoint restore, a larger budget
  costs it compute, not accuracy.
- **Warmup plus `amp: false` plus `grad_clip: 5.0`** address the stability that
  motivated the lr cut. The non-finite-batch guard added earlier catches any
  stragglers without poisoning the weights.
- The change is motivated by **training dynamics** (frozen loss, train ≥ val),
  not by test-set performance.

**Still unvalidated.** All four datasets must be re-run before any number is
reported. If PEMS03/07 do not improve, the underfitting reading is wrong and
`lr` should go back to 2e-4.

## What was deliberately not changed

- **`d_model` stays 256.** Restoring 512 is the other plausible
  capacity lever, but changing width and schedule together would confound the
  result. Test the schedule first; if underfitting persists, width is next.
- **PEMS08 keeps the same recipe** despite overfitting, for protocol
  consistency. Early stopping handles it.

## Validation command

```bash
# schedule A/B on the two datasets that underfit
python scripts/run_schedule_sweep.py --config configs/PEMS03.yaml \
    --horizons 12 96 --arms halve cosine_warmup_30 --seeds 1

# then the full suite
./scripts/run_benchmarks.sh PEMS03 PEMS04 PEMS07 PEMS08
```

The end-of-run diagnostic added earlier will now print UNDERTRAINED or OVERFIT
for each run, so the regime is visible without re-reading curves.
