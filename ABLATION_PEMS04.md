# PEMS04 Component Ablation (S-Mamba-style grid)

`configs/PEMS04.yaml`, T=12, 3 seeds, 11 variants + reference.
Raw data: `results/ablation_PEMS04_smamba.json`.

**Reference (full model): MSE 0.0754, MAE 0.1794.**

## Results

Δ is the paired per-seed change vs. the full model (positive = worse).
† survives Benjamini–Hochberg across the 11-variant family at q<0.05;
\* CI excludes zero but does not survive correction.

| Design | Variant | MSE | MAE | ΔMSE | 95% CI | |
|---|---|---|---|---|---|---|
| **w/o** | no variate mixing | 0.0985 | 0.2058 | **+0.0232** | [+0.0228,+0.0235] | † |
| Replace | Attention mixer | 0.0816 | 0.1842 | **+0.0062** | [+0.0047,+0.0078] | † |
| Replace | mixer in time branch only | 0.0767 | 0.1804 | +0.0013 | [+0.0007,+0.0020] | † |
| w/o | no time branch | 0.0766 | 0.1812 | +0.0012 | [+0.0004,+0.0020] | † |
| Replace | uni-Mamba mixer | 0.0763 | 0.1799 | +0.0010 | [+0.0005,+0.0015] | † |
| w/o | no frequency branch | 0.0759 | 0.1805 | +0.0006 | [−0.0008,+0.0020] | |
| Replace | spectral filter → bi-Mamba | 0.0759 | 0.1801 | +0.0005 | [−0.0016,+0.0026] | |
| Replace | plain average instead of gate | 0.0757 | 0.1804 | +0.0003 | [+0.0000,+0.0006] | \* |
| Replace | untie the two mixers | 0.0752 | 0.1795 | −0.0002 | [−0.0018,+0.0014] | |
| Replace | time encoder: MLP → Mamba | 0.0744 | 0.1778 | **−0.0010** | [−0.0018,−0.0002] | \* |
| w/o | no instance norm | 0.0715 | 0.1755 | **−0.0038** | [−0.0052,−0.0024] | † |

**8 of 11 raw-significant; 6 survive BH correction.**

## What it shows

**This is the first dataset where components separate.** Across the nine
original benchmarks, 0 of 29 component effects survived correction. Here 6 of
11 do, and the pattern is coherent rather than scattered.

**1. Cross-variate mixing dominates.** Removing it costs **+0.0232 MSE, a 31%
degradation** — by far the largest effect measured anywhere in this work. On
PEMS04 (307 spatially correlated sensors) the VC block is the model.

**2. Its form matters, and our pipeline reproduces S-Mamba's two design
claims independently:**
- bi-Mamba beats **attention** by +0.0062 (a selective state space is the
  better variate encoder — S-Mamba's central argument, here measured in our
  own pipeline rather than inherited from theirs)
- bi-Mamba beats **unidirectional** by +0.0010 (channel order carries no
  temporal meaning, so a one-way scan is an arbitrary restriction)
- mixing in **both** branches beats time-branch-only (+0.0013)
- **tying** the two mixers is neutral (−0.0002), matching the efficiency
  finding on Traffic and Electricity — sharing is free, shrinking is not

**3. The frequency branch does not separate.** Removing it costs +0.0006 with
a CI spanning zero; swapping its filter for a bi-Mamba changes nothing
detectable. Removing the *time* branch does hurt (+0.0012†). The asymmetry
matches the branch matrix on the original benchmarks: the temporal path
carries this architecture.

## Two results that argue against the shipped config — and were not adopted

- **`time_mamba`: −0.0010** (CI excludes zero, q=0.052). Mamba beats the MLP
  time encoder here.
- **`no_revin`: −0.0038†** — a **5% MSE improvement** from disabling instance
  normalization.

Both are measured on the **test split**. Acting on them would be exactly the
test-informed selection this paper criticizes — the same error as the Solar
normalization setting. They are recorded as validation-only experiments for
future work, not as configuration changes.

The RevIN result is nonetheless the most interesting lead: PEMS is traffic-flow
data with stable per-sensor scale, so per-window standardization plausibly
removes usable level information. Confirming it on validation would justify a
principled switch, and α-RevIN (already implemented) is the mechanism that
would learn it from training data instead of picking it by hand.
