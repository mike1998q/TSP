# Why Weather and Electricity Got Worse

Analysis of the uploaded run (`TEST` columns) against the previously reported
results, and the changes made in response.

## The observation

Seven of nine datasets improved or stayed flat. Exactly two regressed:

| Dataset | Previous avg MSE | New avg MSE | Δ |
|---|---|---|---|
| ETTh1 | 0.4394 | 0.4385 | −0.0009 |
| ETTh2 | 0.3761 | 0.3745 | −0.0016 |
| ETTm1 | 0.3871 | 0.3860 | −0.0011 |
| ETTm2 | 0.2801 | 0.2795 | −0.0006 |
| Traffic | 0.438 | 0.437 | −0.001 |
| Exchange | 0.363 | 0.363 | 0.000 |
| Solar | 0.199 | 0.196 | −0.003 |
| **Weather** | **0.2485** | **0.2497** | **+0.0013** |
| **Electricity** | **0.1678** | **0.1722** | **+0.0045** |

## The cause

**The regression was caused by the parameter reductions made earlier in this
work, not by anything in the data or the training pipeline.** Git history
confirms the two regressed datasets are precisely the two whose capacity was
cut, and that no other dataset's capacity was touched in the same period:

| Dataset | Commit | Change | Params |
|---|---|---|---|
| Electricity | `7718c26` | `d_model` 512→256, `freq_hidden` 512→256 | 10.62M → 2.83M (−73%) |
| Weather | `8fe736c` | `mixer_placement` both→shared | 5.77M → 3.49M (−40%) |

ETT, Traffic, Exchange and Solar configs were last touched for capacity in a
much earlier commit — and all of them improved.

The magnitudes line up with the size of the cut: Electricity took a 3.75×
parameter reduction and lost 3.5× more accuracy than Weather, which took a
1.65× reduction.

## Why each one hurt — two different mechanisms

These are *not* the same failure, and separating them is the useful finding.

**Electricity — width, not sharing.** Tying the mixer was already measured
accuracy-neutral here (ΔMSE −0.0001, CI includes 0,
`results/Ablation_electricity_mixer.json`). What cost accuracy was halving
`d_model`. The variate mixer's capacity scales as `d_model²`, and Electricity
is the dataset that most needs it: 321 channels with the strongest
cross-channel coupling in the suite (mean |PCC| = 0.50, `results/pcc.json`).
Cutting the width removed capacity the data actually uses.

The consequence is not cosmetic: at 0.1722 the model is now **worse than the
published S-Mamba average of 0.170**, having been better at 0.1678. A headline
comparison flipped.

**Weather — sharing, not width.** The width was unchanged; only the tie was
introduced. The cost (+0.0013) is concentrated at the longest horizon
(H=720: 0.352 → 0.356). With 21 channels the mixer is cheap either way, so
this is not a capacity story — it says the time and frequency branches want
**different** cross-channel mixing on this dataset, and forcing one shared
mixer degrades the long-horizon forecast. The margin over PatchTST (0.2489)
was already a statistical tie; +0.0013 erases it.

## The underlying mistake

The reduction was justified by a Traffic-only result — tied mixer at H=96,
three seeds, ΔMSE +0.0007 with a CI including zero — and generalized to other
datasets without testing. It failed on the first two datasets it was applied
to. **Sharing a mixer and shrinking a mixer are also not interchangeable**:
sharing was free on Traffic and Electricity, shrinking was not.

This is exactly the transfer assumption the manuscript's own Threats section
warns about, and the manuscript already said Electricity, Weather and Solar
"require separate validation." That validation has now happened and it came
back negative.

## Changes made

1. **`configs/electricity.yaml`** — reverted to `d_model: 512`,
   `freq_hidden: 512` (10.62M). `mixer_placement: shared` is kept, since
   tying was measured neutral here; only the width cut is reverted.
2. **`configs/weather.yaml`** — reverted to `mixer_placement: both` (5.77M).
3. **`paper/neurocomputing/main.tex`**
   - `tab:horizons` updated with the new per-horizon numbers for the seven
     datasets whose configuration is unchanged. Weather and Electricity keep
     their full-capacity numbers and carry a marker pointing at the new table.
   - New **`tab:capacity`** reports the reduced variants and their measured
     cost, so the trade-off is published rather than hidden.
   - The Complexity-accounting paragraph now states that the Traffic reduction
     did **not** transfer, and distinguishes the two mechanisms above.
   - The `tab:hparams` caption no longer claims the released configs are the
     reduced ones (they are not, as of this change).

## Notes on the uploaded table

- The `S-mamba` column (e.g. Electricity 0.170) contains **published** numbers,
  not the in-pipeline reproduction (0.183) used in `tab:unified`. These were
  deliberately **not** merged into the unified table: the manuscript's stated
  protocol is that published results are not mixed with same-pipeline reruns
  because preprocessing, optimization and seeds differ.
- The `FLD-mamba` column is likewise published, and covers only six of the nine
  datasets. It is not added as a baseline for the same reason. Adding it
  properly means running it through `scripts/run_unified_baselines.py`.
- The upload has **no per-seed values**, so the `±std` annotations were removed
  from `tab:horizons` rather than carried over from the earlier three-seed
  study. If seed-level outputs are available, the dispersion can be restored.
