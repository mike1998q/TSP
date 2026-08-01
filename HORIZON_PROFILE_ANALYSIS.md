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

## Experiments to separate the three mechanisms

1. **Spectral-pathway ablation on Electricity** (one line, already staged):
   `freq_backbone` ∈ {none, fits} × H ∈ {96,192,336,720}, 5 seeds.
2. **Training budget**: 10 epochs/halve vs 50 epochs/linear decay. Tests whether
   the short-horizon deficit is undertraining.
3. **Kernel width**: `time_kernel_size` ∈ {5, 13, 25} at H=96.
4. Optionally `fusion: residual` (already implemented) — lets the frequency
   branch add rather than compete, which should matter most where its
   contribution is small relative to the time branch.
