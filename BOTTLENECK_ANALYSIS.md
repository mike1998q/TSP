# DD-Mamba: Bottleneck Analysis and Literature for Next Steps

This memo answers three questions: (1) what we did to shrink the Weather
config, (2) where the current model's real bottlenecks are — grounded in the
repository's own result files, not intuition — and (3) which published methods
address each bottleneck.

---

## 1. Weather parameter reduction (done)

Weather ran at `d256 + mixer_placement=both`, 2 mixer layers = **5.77M**
parameters. A module breakdown shows the two variate mixers (time + freq
branch) are **79%** of that cost:

| Module group | Params |
|---|---|
| time-branch variate mixer (fwd+bwd+ffn) | 2.28M |
| freq-branch variate mixer (fwd+bwd+ffn) | 2.28M |
| time-branch temporal Mamba encoder | 0.88M |
| fusion gate + heads + embeds | ~0.32M |

Weather has only **21 channels**, and its config notes (and the component
ablation) say accuracy is carried by the **temporal** encoder, not
cross-channel mixing. So the mixer is over-provisioned. The fix weight-ties
one mixer across both branches (`mixer_placement=shared`), keeping the time
encoder at full width:

**`d256 + shared` = 3.49M (−40%)**, verified to build and train (mixer tie
confirmed: `freq_branch.channel_mixer is time_branch.channel_mixer`).

More aggressive options if further shrinkage is wanted (numbers computed from
the actual model): `d256+shared, 1 layer` = 2.35M; `d128+both` = 1.56M;
`d128+shared` = 0.97M. These also shrink the temporal encoder, which is
Weather's accuracy driver, so they carry more risk and should be validated.

---

## 2. The model's bottlenecks (grounded in `results/`)

### Bottleneck A — the variate mixer costs the most and earns the least

- **Cost.** The bidirectional variate mixer scales with `d_model²` (not
  channel count) and is 79% of Weather params, up to ~92% on the high-channel
  datasets. It is the single dominant parameter and activation-memory term.
- **Payoff.** In the component-ablation family, **0 of 29** placement/switch
  effects survive Benjamini–Hochberg correction
  (`results/stats_correction.json`). Removing the mixer on Weather costs only
  `+0.0104` MSE (`p=0.022`, `q=0.169` — not significant); *adding* it to
  ETTh1 actually *hurts* (`+0.0077`). So the biggest cost buys no
  statistically supported accuracy.

**Reading:** the model is, in effect, a strong linear/RLinear-class temporal
forecaster wrapped in an expensive channel-mixing apparatus whose value is not
confirmed. This is an *efficiency* bottleneck: quadratic cost, unproven return.

### Bottleneck B — the short fixed look-back caps accuracy where we lose

Against the seven in-pipeline baselines (`results/unified_analysis.json`):

| Dataset | DD-Mamba | Best baseline | Margin |
|---|---|---|---|
| Electricity | 0.1678 | S-Mamba 0.1830 | **+0.0153** win |
| ETTh2 | 0.3761 | iTransformer 0.3835 | +0.0074 win |
| ETTh1 | 0.4394 | iTransformer 0.4438 | +0.0044 win |
| ETTm2 | 0.2801 | PatchTST 0.2842 | +0.0041 win |
| **Weather** | 0.2485 | PatchTST 0.2489 | **+0.0004 tie** |
| **ETTm1** | 0.3871 | PatchTST 0.3813 | **−0.0058 loss** |

The only loss and the only tie are both **against PatchTST**, and both are on
sub-hourly / high-resolution series. The common cause is the **fixed 96-step
look-back**:

- Weather is 10-min sampled, so its **daily cycle is 144 samples > L=96**.
  The look-back cannot contain one full day, so the frequency branch
  structurally *cannot* represent the dominant seasonality (stated in
  `configs/weather.yaml`). PatchTST's patching + longer effective context can.
- The frequency branch under-delivers everywhere: only **Solar@96** survives
  BH in the branch matrix (and that one is exploratory, from a test-informed
  RevIN choice). The "dual-domain" half contributes almost no confirmed value,
  which is consistent with the look-back being too short for the spectral view
  to help.

**Reading:** the *accuracy ceiling* is set by look-back length and the
resulting inability of the frequency branch to see long periods — not by
model capacity. Adding parameters (Bottleneck A) does not move this ceiling.

---

## 3. Literature addressing each bottleneck

### For Bottleneck A — cheaper cross-channel interaction

- **SOFTS — series-core fusion (Han et al., NeurIPS 2024).** Replaces
  O(C²) channel attention with a centralized *core* that aggregates all
  channels once and redistributes, giving **O(C)** channel interaction.
  Directly substitutable for our quadratic mixer. *Already in `refs.bib`* and
  a natural drop-in ablation.
  <https://arxiv.org/html/2404.14197v2>
- **TSCG — grouped channel interaction (Expert Syst. Appl., 2025).** Groups
  channels so interaction cost is sub-quadratic; useful for the 300–883
  channel datasets. <https://www.sciencedirect.com/science/article/abs/pii/S0957417425039144>
- **xLSTM-Mixer (2024/25).** Channel mixing via scalar memories instead of
  dense attention — a lightweight alternative mixer.
  <https://arxiv.org/html/2410.16928v4>
- **TimePro (Ma et al., ICML 2025).** Variable- and time-aware hyper-states
  for cross-variable effects at linear complexity. *Already added to
  `refs.bib`.* <https://arxiv.org/abs/2505.20774>
- **TimeMachine (ECAI 2024, already cited)** and **TSMamba** use
  channel-compressed / multi-scale Mamba to get cross-channel context at lower
  cost — relevant if we keep an SSM mixer but want it cheaper.

### For Bottleneck B — extending effective context / seasonality beyond L

- **"Overcoming Lookback Window Limitations" (PIH, ICLR/OpenReview).**
  Model-agnostic patch module that pushes usable look-back to **1024**,
  where naive longer windows normally overfit. Most directly targets our
  Weather daily-cycle problem. <https://openreview.net/forum?id=hVpAjJPfgZ>
- **"Enhancing the Maximum Effective Window" (OpenReview).** Analyses why
  longer windows stop helping (distribution shift, noise) and how to extend
  the *effective* window. <https://openreview.net/forum?id=Gmwsy7TlFI>
- **Multi-scale patching / S2TX (2025).** Coarse patches over the full window
  for global/seasonal context + fine patches over a short window for local
  dynamics — a concrete way to see the daily cycle without a full quadratic
  cost. <https://arxiv.org/pdf/2502.11340>
- **PatchTST (ICLR 2023, already cited).** The baseline that beats us on
  ETTm1/Weather; patching is the mechanism to adopt or match.
- **PENGUIN — periodic-nested group attention (2025).** Explicitly models
  periodicity longer than the raw window. <https://arxiv.org/pdf/2508.13773>
- **"Position: There are no Champions in LTSF" (2025).** Cautions that
  look-back and protocol choices dominate rankings — supports our
  honest-measurement framing and the recommendation to sweep L rather than
  add capacity. <https://arxiv.org/html/2502.14045v1>

### Frequency-branch specific (why the spectral view is inert)

- **FITS (ICLR 2024, already cited)** and **FreTS (NeurIPS 2023, cited)**:
  frequency-interpolation forecasting works, but is bounded by look-back
  resolution — reinforcing that the fix is a longer/multi-scale window, not a
  bigger spectral head.
- **FLDmamba (2025).** Fourier + Laplace decomposition with Mamba — a richer
  spectral pathway if we want the frequency branch to earn its place.
  <https://arxiv.org/pdf/2507.12803>

---

## Recommended next experiments (cheap → expensive)

1. **Swap the quadratic mixer for a SOFTS-style O(C) core** and re-run the
   component ablation — tests whether we can keep accuracy at a fraction of
   the params (addresses A directly; SOFTS already cited).
2. **Look-back sweep** L ∈ {96, 192, 336, 512} on Weather/ETTm1, so the daily
   cycle (144) fits — the cleanest test of Bottleneck B, no new architecture.
3. **Add multi-scale patching** (coarse-full + fine-short) to the time branch
   if (2) confirms look-back is the ceiling.

Items 1–3 need a GPU; they are logged here rather than run in this
environment. See `REVISION_RESPONSE.md` Part B for the existing run tooling.
