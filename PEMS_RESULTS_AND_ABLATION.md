# PEMS + Electricity Results, Headroom, and the S-Mamba-Style Ablation

## 1. Extracted accuracy

The PEMS recipe fix (cosine_warmup, lr 5e-4, 30 epochs) **worked**.

### PEMS, MSE / MAE, look-back 96

| dataset | H=12 | H=24 | H=48 | H=96 | avg |
|---|---|---|---|---|---|
| PEMS03 | 0.0649 / 0.1682 | 0.0846 / 0.1928 | 0.1282 / 0.2380 | 0.1872 / 0.2916 | **0.1162 / 0.2227** |
| PEMS04 | 0.0753 / 0.1786 | 0.0929 / 0.2003 | 0.1229 / 0.2330 | 0.1713 / 0.2758 | **0.1156 / 0.2219** |
| PEMS07 | 0.0587 / 0.1554 | 0.0785 / 0.1751 | 0.1048 / 0.2085 | 0.1392 / 0.2435 | **0.0953 / 0.1956** |
| PEMS08 | 0.0740 / 0.1760 | 0.0992 / 0.2014 | 0.1457 / 0.2514 | 0.2340 / 0.2836 | **0.1382 / 0.2281** |

### Improvement over the previous recipe

| dataset | H=12 | H=24 | H=48 | H=96 |
|---|---|---|---|---|
| PEMS03 | −3.7% | −11.0% | −16.5% | −23.7% |
| PEMS04 | −11.0% | −18.3% | −33.5% | **−39.5%** |
| PEMS07 | −10.8% | −17.9% | −31.0% | **−39.1%** |
| PEMS08 | −5.6% | −10.7% | −24.2% | −26.3% |

**Overall PEMS mean MSE 0.1543 → 0.1163, a 24.6% reduction.** The gain grows
monotonically with horizon on every dataset, which is the signature the
underfitting diagnosis predicted: a starved optimizer hurts most where the task
is hardest.

### Electricity (config unchanged, as intended)

| H=96 | H=192 | H=336 | H=720 | avg |
|---|---|---|---|---|
| 0.1404 / 0.2361 | 0.1583 / 0.2528 | 0.1740 / 0.2699 | 0.2018 / 0.2977 | **0.1686 / 0.2641** |

Reproduces the known-good 0.169 exactly. The run diagnostic correctly fired
`OVERFIT` on H=720 only.

## 2. Is further improvement possible? Yes — and the lever has changed

The fit regime **flipped**. Final train loss over best validation loss:

| dataset | H=12 | H=24 | H=48 | H=96 |
|---|---|---|---|---|
| PEMS03 | 0.643 | 0.612 | 0.536 | 0.468 |
| PEMS04 | 0.614 | 0.577 | 0.498 | 0.428 |
| PEMS07 | 0.712 | 0.545 | 0.476 | 0.427 |
| PEMS08 | 0.567 | 0.447 | 0.397 | **0.243** |

Mean **0.512**, against 0.86 / 0.76 on Electricity. Before the fix, PEMS03 and
PEMS07 sat at 1.02–1.03 (underfitting). They now overfit roughly **twice as
hard as Electricity ever did**. PEMS08 H=96 is extreme: training loss 0.0555
against validation 0.2287, a 4.1× gap.

**So there is real headroom, and it is now on the regularization axis, not the
optimization axis.** This is the mirror image of the Electricity finding, where
a 5× budget bought nothing precisely because that model already overfit.

Sweep arms are implemented (`scripts/run_schedule_sweep.py`), holding the
validated schedule fixed and varying only regularization:

```bash
# weight decay
python scripts/run_schedule_sweep.py --config configs/PEMS08.yaml \
    --horizons 12 96 --arms pems_base pems_wd pems_wd_hi --seeds 3

# dropout 0.1 -> 0.2 / 0.3, on top of the base schedule
python scripts/run_schedule_sweep.py --config configs/PEMS08.yaml \
    --horizons 12 96 --arms pems_base --model-arm drop2 --seeds 3
```

PEMS08 first: it has the largest gap and the fewest channels, so it should show
the effect most clearly.

**The current PEMS config is deliberately left alone.** It has just been
validated by a 24.6% improvement; stacking another unvalidated change on top is
the mistake that cost several rounds earlier in this work. Regularization is a
sweep, not a config edit, until it has evidence.

## 3. S-Mamba-style component ablation

S-Mamba (Wang et al., *Neurocomputing* 2025) validates its design by ablating
two blocks — the **VC** (Variate Correlation) bidirectional Mamba over the
variate axis, and the **TD** (Temporal Dependency) block over time — replacing
or removing each and attributing the accuracy delta to it. DD-Mamba has
structurally matching components plus two S-Mamba does not have (the parallel
frequency branch, and the fusion rule), so the analogous ablation is:

| block | variant | what it tests |
|---|---|---|
| — | `full` | reference |
| **VC** | `no_channel_mixer` | remove cross-variate mixing entirely |
| **VC** | `both_mixer` / `shared_mixer` | untie vs. weight-tie the two mixers |
| **VC** | `time_mixer_only` | mix in the time branch only |
| **TD** | `time_mamba` / `time_mlp` | Mamba ↔ MLP over time |
| freq | `freq_mamba` | linear spectral filter → BiMamba over bins |
| decomp | `time_only` | drop the frequency branch |
| decomp | `freq_only` | drop the time branch |
| norm | `no_revin` | instance normalization off |

Run it with:

```bash
python scripts/run_ablation.py --config configs/PEMS04.yaml --chain smamba --seeds 3
python scripts/run_ablation.py --config configs/PEMS08.yaml --chain smamba --seeds 3
```

### Two implementation points that matter

**The chain is resolved against each config, so no variant is a no-op.**
PEMS ships `time_encoder: mlp` and `mixer_placement: shared`, while ETT ships
`mamba` / `both`. A fixed variant list would have silently included cells
identical to the reference — inflating the correction family with duplicates of
`full` and diluting the FDR correction. `resolve_chain()` picks the direction
that actually differs, and drops mixer variants on configs with no mixer.
Verified: PEMS04 resolves to 9 distinct variants, ETTh1 to 6, zero no-ops in
either.

**Our method is stricter than S-Mamba's.** S-Mamba reports single-run deltas.
We run multiple seeds, report paired confidence intervals, and apply
Benjamini–Hochberg across the family (`scripts/compute_stats_correction.py`).
With 9 variants, roughly one would clear an uncorrected 0.05 threshold by
chance alone, so uncorrected deltas would not support component claims. This
difference should be stated when the comparison is written up — it is a
methodological strength, not a formatting detail.

### What to expect

PEMS is the first dataset family where cross-variate mixing has a strong prior
reason to matter (170–883 spatially correlated sensors). If the VC-block
variants show an effect anywhere, this is where. On the nine benchmark datasets
the component family produced **0 of 29** BH-significant effects, and the
Electricity schedule sweep has since ruled out undertraining as the explanation
— so a positive result on PEMS would be genuinely new evidence rather than a
confirmation.
