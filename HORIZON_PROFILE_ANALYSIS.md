# Why H=96/192 Still Trail While H=336/720 Are Strong

Analysis of the re-run Electricity and Weather results (after the capacity
revert), and the change made in response.

## The re-run confirms the revert worked

| Dataset | H=96 | H=192 | H=336 | H=720 | Avg | vs. previous |
|---|---|---|---|---|---|---|
| Weather | 0.162 | 0.212 | **0.267** | **0.349** | **0.2475** | better than 0.2485 |
| Electricity | 0.141 | 0.159 | 0.174 | 0.200 | 0.1685 | back from 0.1722 |

Weather is now *better* than its previously reported value at the two long
horizons (0.269→0.267, 0.352→0.349). Electricity recovered from the 0.1722
regression. Both are back above the published S-Mamba average.

## The remaining pattern is real and it is monotone

Relative to the published S-Mamba per-horizon numbers:

| H | Electricity | Weather |
|---|---|---|
| 96 | **+1.44%** (worse) | −1.82% (better) |
| 192 | 0.00% | −0.93% |
| 336 | −1.14% (better) | −2.55% |
| 720 | −1.96% (better) | −0.29% |

So the short-horizon deficit is **specific to Electricity**. Weather leads at
every horizon. Against FLD-Mamba the Electricity profile is the same shape:
+2.92% at H=96, +0.63% at H=192, −4.40% at H=336, 0.00% at H=720.

## Leading explanation: a random-initialized frequency head

Weather and Electricity differ in exactly the way that predicts this.

`src/models/freq_branch.py:147` zero-initializes the frequency forecast head
**only when a spectral backbone is present**:

```python
if self.spec_backbone is not None and zero_init:
    nn.init.zeros_(self.head[-1].weight); nn.init.zeros_(self.head[-1].bias)
```

- **Weather** sets `freq_backbone: fits` → head zero-init → branch silent at
  init → **no short-horizon deficit.**
- **Electricity** inherited `freq_backbone: none` → head **randomly
  initialized** → the branch injects noise from step 0 → **short-horizon
  deficit.**

Measured directly (electricity recipe, normalized space):

| `freq_backbone` | H=96 | H=192 | H=336 | H=720 |
|---|---|---|---|---|
| `none` — RMS of injected perturbation | 0.0580 | 0.0572 | 0.0580 | 0.0571 |
| `fits` — RMS of injected perturbation | 0.0000 | 0.0000 | 0.0000 | 0.0000 |

The perturbation is **constant across horizons**. But the achievable error is
not: 0.141 at H=96 rising to 0.200 at H=720. A fixed perturbation therefore
consumes a *larger share* of a smaller error budget — 2.39% of the H=96 budget
versus 1.68% at H=720 — which is the same direction and rough proportion as the
observed gap.

**Honest limit on this evidence.** The horizon trend has four points, and any
quantity monotone in H would correlate with it, so the reported r=0.99 is
suggestive, not confirmatory. The load-bearing evidence is (a) the direct
measurement of a horizon-independent perturbation, and (b) the
Weather/Electricity contrast, which is a natural experiment on exactly this
switch.

## Two further mechanisms, same direction, not separated by these data

**Linear dominance interacts with horizon.** The architecture starts as a
linear forecaster (zero-init corrections, g₀≈0.9) and the schedule halves the
LR every epoch over 10 epochs, so roughly 2–3 epochs carry the real learning. A
near-linear predictor is *relatively* stronger at long horizons, where every
model converges toward seasonal climatology, and *relatively* weaker at short
horizons, where recoverable local dynamics reward nonlinear capacity. This is
the same undertraining confound documented in `DESIGN_LIMITATIONS_LITERATURE.md`.

**The decomposition kernel is tuned to the daily scale.** `time_kernel_size:
25` at hourly sampling is ≈ one day. It smooths away precisely the sub-daily
detail an H=96 forecast could still exploit, while being harmless at H=720
where that detail is unpredictable anyway.

## Change made

`configs/electricity.yaml`: **`freq_backbone: fits`** (was `none`, inherited
from the default). Verified: the frequency branch is now silent at
initialization, at a cost of +0.01M parameters (10.62M → 10.63M).

This is the single highest-confidence intervention because it is the one
mechanism that is *measured* rather than inferred, and because the dataset that
already has this setting (Weather) does not exhibit the symptom.

**The change is marked UNVALIDATED in the config.** The reported Electricity
numbers were produced with `none`. All four horizons must be re-run before the
manuscript numbers are touched, and if H=96/192 do not improve the line should
be reverted. This is the discipline that was missing when the capacity cuts
were made and had to be reverted.

## Paper changes

- `tab:horizons`: Weather and Electricity updated to the re-run values
  (Weather avg MSE 0.248 → 0.247).
- New Discussion paragraph *"The error profile is horizon-dependent, and
  initialization explains part of it"* — states the leading explanation, the
  direct measurement, the honest limit on the correlation evidence, and the two
  competing mechanisms.
- `tab:hparams`: Electricity's spectral-map cell carries a footnote recording
  that its reported numbers came from the disabled-pathway configuration.

---

# UPDATE: the FITS remedy was tested and failed

The `freq_backbone: fits` change proposed above was re-run. It made Electricity
**worse**, and the failure is diagnostic.

| H | before (`none`) | after (`fits`) | Δ | FITS fan-out |
|---|---|---|---|---|
| 96 | 0.141 | 0.143 | +0.002 | 49→97 (2.0×) |
| 192 | 0.159 | 0.160 | +0.001 | 49→145 (3.0×) |
| 336 | 0.174 | 0.174 | 0.000 | 49→217 (4.4×) |
| 720 | **0.200** | **0.224** | **+0.024** | 49→409 (**8.3×**) |
| avg | **0.1685** | **0.1753** | +0.0068 | |

**Cause.** The spectral map interpolates the length-*L* spectrum onto a
length-(*L*+*H*) grid, so its fan-out grows with the horizon. At H=720 it
extrapolates 49 measured coefficients to 409 output bins, and the damage tracks
that ratio monotonically. Compounding it, this config sets `freq_sparsity: 0.0`
— no low-pass — whereas ETTh1/ETTh2 pair `fits` with ρ ∈ {0.3, 0.4}. The cutoff
is the regularizer that makes linear-in-frequency forecasting stable; without
it the map extrapolates the noisiest high-frequency bins the furthest.

**`num_workers` is not implicated.** It was changed 4→2 in the same run, but it
cannot affect accuracy: `SlidingWindowDataset.__getitem__` is pure indexing with
no RNG, and shuffling happens in the sampler in the main process. Verified
directly — the batch stream is bit-identical for `num_workers` ∈ {0, 2, 4}.
It affects loading throughput only, and 2 is the right setting for memory.

**What was actually wrong with the remedy.** The measurement implicated the
*random initialization of the frequency head*, not the absence of a spectral
map. Enabling FITS conflates the two: it silences the head **and** adds an
expensive, horizon-sensitive extrapolation. Only the first was wanted.

## Correction applied

1. **`configs/electricity.yaml`: `freq_backbone` reverted to `none`.**
2. **New `freq_zero_init_head: {auto, always, never}` option**
   (`src/models/freq_branch.py`). `auto` is the historical behaviour and the
   default, so every existing config is unchanged. `always` zero-initializes
   the head regardless of backbone — silencing it at init **at zero parameter
   cost and with no extrapolation**.

   Verified: with `backbone=none`, `zero_init_head=always` gives RMS(y_freq)=0
   at init versus 0.580 under `auto`, at an identical 10.62M parameters (`fits`
   was 10.63M *and* carried the extrapolation). The frequency encoder receives
   no gradient at step 0 and self-heals from step 1, exactly as the time
   branch's long-standing zero-init head does.
3. **`configs/electricity.yaml` now sets `freq_zero_init_head: always`**,
   marked UNVALIDATED — the reported numbers predate it. Re-run all four
   horizons; revert to `auto` if H=96/192 do not improve.
4. **New ablation variant `freq_head_zero_init`** so this can be A/B tested
   properly rather than by flipping a default.

The marking protocol worked as intended here: the change was flagged
unvalidated, the re-run falsified it, and it has been reverted rather than
written into the manuscript.

## Experiments to separate the three mechanisms

1. **Spectral-pathway ablation on Electricity** (one line, already staged):
   `freq_backbone` ∈ {none, fits} × H ∈ {96,192,336,720}, 5 seeds.
2. **Training budget**: 10 epochs/halve vs 50 epochs/linear decay. Tests whether
   the short-horizon deficit is undertraining.
3. **Kernel width**: `time_kernel_size` ∈ {5, 13, 25} at H=96.
4. Optionally `fusion: residual` (already implemented) — lets the frequency
   branch add rather than compete, which should matter most where its
   contribution is small relative to the time branch.
