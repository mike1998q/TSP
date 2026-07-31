# Addressing Three Design Limitations: Literature and Concrete Plans

Follow-up to `BOTTLENECK_ANALYSIS.md`. That memo covered the two *bottlenecks*
(quadratic mixer cost, short look-back). This one addresses three **design**
limitations that are not yet written into the manuscript:

1. **Convex fusion can only interpolate, never superpose.**
2. **Linear-dominated by init *and* schedule** — a capacity/undertraining confound.
3. **RevIN discards level and cross-series scale** (and its on/off switch is a
   selection-bias surface).

A cross-cutting observation first, because it changes how to test them:
**these three interact.** The frequency branch may look inert not because
spectral modeling is useless here, but because (1) it is forced to *compete*
with the time branch on a simplex, (3) RevIN has already removed the trend
while leaving the seasonality the branch would need, and (2) the 10-epoch
`0.5^(epoch-1)` schedule gives its zero-initialized head almost no time to
grow. Any one of these alone would depress the measured effect. They should
therefore be disentangled by one factorial experiment, not three separate ones.

Citation status is marked: ✓ = author list and venue verified; ◇ = arXiv ID
verified but author list not confirmed from this environment (do not cite
without checking).

---

## 1. Convex fusion cannot superpose

### The defect, precisely

`src/models/fusion.py:77` computes `y = g·y_time + (1−g)·y_freq` with
`g = sigmoid(...) ∈ [0,1]`. The output is constrained to the **segment between
the two branch forecasts**. If the time branch is right about the trend and the
frequency branch is right about a spectral peak, the model cannot output
*trend + peak* — the best it can do is a weighted average, whose error is
bounded below by the better of the two on any component where they disagree.
The convex constraint makes the branches **substitutes**, when the whole
premise of a dual-domain model is that they are **complements**.

This is very likely a partial explanation for "only Solar@96 survives BH on the
frequency side": we may be measuring a fusion-rule artifact, not the value of
spectral modeling.

### Literature

- **CoCoAFusE — "Beyond Mixtures of Experts via Model Fusion"** ✓
  (Aurelio Raffa Ugolini, Mara Tanelli, Valentina Breschi; arXiv:2505.01105, 2025).
  This is the cleanest statement of exactly our problem in the general MoE
  setting: classical MoE *mixes* (superimposes) expert predictions as a convex
  combination, and the paper argues this is expressively limiting. Its remedy is
  to contemplate **fusion of the experts' distributions in addition to** the
  usual convex mixing. The framing — "mixing is not enough" — transfers directly
  to a two-expert time/frequency model.
  <https://arxiv.org/abs/2505.01105>

- **N-BEATS doubly-residual stacking** ✓ (Oreshkin et al., ICLR 2020 —
  *already in `refs.bib` as `nbeats`*). The canonical **additive** alternative:
  each block emits a backcast subtracted from the running input and a forecast
  **added** to the running total. Pure superposition, no simplex, and it is the
  architecture our zero-init design is already halfway toward. Strong precedent
  that additive composition of specialized blocks is both stable and effective.

- **Dualformer — "Time-Frequency Dual Domain Learning for LTSF"** ◇
  (arXiv:2601.15669, 2026). The closest architectural analogue to DD-Mamba:
  a dual-branch time+frequency model. Notably it *keeps* a weighting mechanism
  but derives it from the **signal** rather than from learned features — a
  "periodicity-aware weighting … based on the harmonic energy ratio of inputs."
  It also allocates distinct frequency bands to different layers. Two lessons:
  (a) a data-derived gate is a defensible middle ground between our learned gate
  and plain addition; (b) our single-shot rFFT with no band structure is
  coarse. <https://arxiv.org/abs/2601.15669>

- **T3Time** ◇ (arXiv:2508.04251, 2025) — uses channel-wise **residual fusion**
  between temporal-spectral embeddings and aligned cross-modal representations,
  i.e. residual rather than convex combination in a multi-branch forecaster.

- **Dual-domain MoE routing** ◇ ("Learning to Route in Time and Frequency
  Domains", *Scientific Reports*, 2026) — routes across time/frequency experts;
  relevant if we want to keep gating but make it sparse/conditional.

### Concrete plan (cheap; no GPU to implement, one run to test)

Add fusion modes to `ForecastFusion` — the class already dispatches on `mode`,
so this is additive and does not disturb existing configs:

| New mode | Formula | Rationale |
|---|---|---|
| `residual` | `y = y_time + α·y_freq`, α scalar, **zero-init** | Preserves the exact current initialization story (`α=0` ⇒ starts as pure time branch, same as `g=0.9`… but *unconstrained* thereafter). Minimal, one new parameter. |
| `affine` | `y = g_t·y_time + g_f·y_freq`, two independent gates, no sum-to-1 | Strictly generalizes the current gate; lets both branches be up-weighted simultaneously. |
| `doubly_residual` | freq branch fits the residual `y − y_time` | N-BEATS-style; the most principled given zero-init corrections. |

**Why this is the highest-value change of the three:** it is a ~20-line change,
it strictly generalizes the current model (the existing behavior remains
reachable), and it directly tests the alternative explanation for the paper's
weakest empirical claim. If the frequency branch becomes significant under
`residual` fusion, that is a *finding*, not just a fix.

---

## 2. Linear-dominated by init **and** schedule (undertraining confound)

### The defect, precisely

Two choices compound:

- **Init:** deep corrections are zero-initialized and the gate is biased to
  `g₀≈0.9` (`fusion.py:34`), so training *starts* as a linear forecaster.
- **Schedule:** `adjust_lr` sets `lr = base·0.5^(epoch−1)` (`train.py:68`) over
  **10 epochs**. By epoch 4 the LR is 1/8 of base; by epoch 6, 1/32. Effectively
  **2–3 epochs carry all the learning.**

A zero-initialized branch must *grow from zero* — and it is given the smallest
learning-rate budget in which to do so. So "0 of 29 component effects survive
BH" has (at least) two readings: *the components do not help*, or *the recipe
never trained them*. **The paper currently cannot distinguish these**, which
matters a lot because component measurement is the paper's central claim.

### Literature

- **"Deep Double Descent for Time Series Forecasting: Avoiding Undertrained
  Models"** ◇ (arXiv:2311.01442). The most directly damaging citation for our
  current recipe: it demonstrates **epoch-wise deep double descent** in
  Transformer-based forecasters and shows that apparent overfitting *reverts*
  with more epochs, reaching state-of-the-art on long-sequence forecasting. If
  the phenomenon holds for our deep branches, a 10-epoch budget lands squarely
  in the undertrained regime and the ablation conclusions are recipe artifacts.
  <https://arxiv.org/abs/2311.01442>

- **"Optimal Linear Decay Learning Rate Schedules and Further Refinements"** ◇
  (arXiv:2310.07831). Reports that **cosine underperforms linear decay when
  training fewer than ~30 epochs**. Our `halve` schedule decays far more
  aggressively than cosine, over 10 epochs — i.e. we are outside the regime
  where the schedule is known to be well-behaved.
  <https://arxiv.org/abs/2310.07831>

- **STAIR — "Three-Stage Learning Unlocks Strong Performance in Simple Models
  for LTSF"** ◇ (arXiv:2605.13678, 2026). The key positive result: it shows a
  **training curriculum**, not new architecture, unlocks capacity. Its three
  stages are (i) learn shared temporal dynamics across variables via a shared
  temporal mapping, (ii) **shared-to-individual fine-tuning** per variable,
  (iii) complement the backbone with **cross-variable information through
  residual learning**. DD-Mamba already has exactly this decomposition (shared
  temporal backbone + per-variate structure + cross-variate mixer) but trains
  all of it simultaneously under a collapsing LR. STAIR is close to a drop-in
  curriculum for our architecture. <https://arxiv.org/abs/2605.13678>

- **ReZero** ✓ (Bachlechner et al., UAI 2021 — *already cited*). Justifies
  zero-init residuals, but note it pairs zero-init with *long* training; it is
  not evidence that zero-init works under a 10-epoch collapsing schedule.

### Concrete plan

1. **Training-budget ablation (the integrity experiment).** Same architecture,
   three recipes: `{10 epochs, halve}` (current) vs `{50 epochs, linear decay}`
   vs `{50 epochs, cosine}` with early stopping. Re-run the component family
   under the best recipe. Two possible outcomes, both publishable:
   - Components stay non-significant ⇒ the "capacity does not help" claim is
     **strengthened**, now robust to the recipe.
   - Components become significant ⇒ the current negative result is a recipe
     artifact and must be re-stated. Better to find this ourselves.
2. **Per-group learning rates.** Give zero-initialized modules (deep heads,
   gate, spectral map) a higher LR than the linear backbone — a two-line
   `AdamW` param-group change. Removes the "must grow from zero on the smallest
   LR" pathology without changing architecture.
3. **STAIR-style curriculum** as the more ambitious version of (2).

**Manuscript consequence:** until (1) is run, the Threats section should say
plainly that the component-ablation nulls are conditional on the training
recipe, and that zero-init + aggressive decay is a plausible alternative
explanation. That paragraph should be added regardless of whether the
experiment gets run.

---

## 3. RevIN discards level and cross-series scale

### The defect, precisely

`dual_domain_model.py:168` removes per-window, per-channel mean and std. This
discards (a) absolute level, (b) the **relative scale between channels** — which
is exactly the cross-series information a multivariate model should exploit, and
(c) any predictive signal carried by *changes* in local mean/variance. And our
only control is a **binary per-dataset switch** (`use_revin`), one of whose
settings (Solar = off) was chosen with test feedback — so this limitation is
also the source of the paper's most-cited threat to validity.

### Literature (this is the richest of the three)

- **FAN — "Frequency Adaptive Normalization for Non-stationary Time Series
  Forecasting"** ✓ (Weiwei Ye, Songgaojun Deng, Qiaosha Zou, Ning Gui;
  **NeurIPS 2024**). The central finding: reversible instance normalization with
  statistical measures is **limited to expressing basic trends and incapable of
  handling seasonal patterns**. FAN instead removes the instance-wise **top-K
  dominant Fourier components**, and explicitly models the input↔output
  discrepancy of those components as a prediction task with a simple MLP
  (normalize → frequency residual learning → denormalize). Model-agnostic,
  reported **7.76%–37.90% average MSE improvement**.
  **This is the single most relevant paper to DD-Mamba as a whole**, because it
  targets our Weather failure (daily cycle 144 > L=96 seasonality that RevIN
  cannot touch) *and* it implies our inert frequency branch may be solving the
  right problem in the wrong place — a normalization layer rather than a
  parallel forecasting branch. Being honest: adopting FAN could make the
  frequency branch redundant. That would itself be a clean, reportable result.
  <https://arxiv.org/abs/2409.20371>

- **SAN — "Adaptive Normalization for Non-stationary Time Series Forecasting:
  A Temporal Slice Perspective"** ✓ (Zhiding Liu, Mingyue Cheng, Zhi Li, Zhenya
  Huang, Qi Liu, Yanhu Xie, Enhong Chen; **NeurIPS 2023**). Argues
  whole-instance statistics are **too coarse**; normalizes at the temporal-slice
  level and explicitly *predicts future slice statistics*. Directly addresses
  our (c): statistics that evolve within the window carry signal.
  <https://proceedings.neurips.cc/paper_files/paper/2023/hash/2e19dab94882bc95ed094c4399cfda02-Abstract-Conference.html>

- **Dish-TS** ✓ (Wei Fan, Pengyang Wang, Dongkun Wang, Dongjie Wang, Yuanchun
  Zhou, Yanjie Fu; **AAAI 2023, 37:7522–7529**). Points out that the **input and
  horizon windows follow different distributions**, so normalizing and
  denormalizing with *input-window* statistics alone — exactly what we do —
  cannot characterize the output space. Learns separate coefficient networks
  for input and output distributions.

- **Non-stationary Transformers** ✓ (Liu et al., NeurIPS 2022 — *already cited
  as `nonstationary`*). The **over-stationarization** argument: aggressive
  normalization weakens the model's ability to distinguish genuine
  non-stationary events. We cite this paper already but do not engage with its
  critique of our own normalization.

- **α-RevIN**, introduced within **STAIR** ◇ (arXiv:2605.13678). Explicitly
  designed to mitigate "the overly strong normalization prior induced by
  standard RevIN" by softening rather than fully applying it.
  **This is the most immediately useful item for us** — see plan below.

- **NoRIN — "Backbone-Adaptive Reversible Normalization"** ◇
  (arXiv:2605.10823, 2026) and **"On the Role of Reversible Instance
  Normalization"** ◇ (arXiv:2603.11869, 2026) — recent analyses of when RevIN
  helps or hurts. The general structural critique they share: RevIN and its
  successors apply a strictly **affine** map `x ↦ ax + b`, so they cannot
  reshape the underlying distribution at all.

### Concrete plan

1. **Replace the binary `use_revin` switch with a learnable α-blend
   (α-RevIN).** This is the highest-leverage change in this entire memo, because
   it fixes two problems at once:
   - *Modeling:* normalization strength becomes continuous and data-driven
     rather than all-or-nothing.
   - *Methodology:* it **eliminates the paper's most consequential
     threat-to-validity**. The Solar RevIN=off choice — made by inspecting a
     test-set ablation, and the reason the one BH-significant frequency result
     is labelled exploratory — stops being a discrete decision selected on test
     data and becomes a parameter learned from training data. An entire class of
     model-selection bias disappears by construction.
   Implementation is small: keep `x_norm = (x−μ)/σ`, output
   `α·x_norm + (1−α)·x` with α a learned scalar (per-channel optional), and
   invert consistently.
2. **Add FAN as an optional normalization backend** and evaluate whether it
   subsumes the frequency branch. Run it on Weather first — the dataset where
   the seasonality RevIN cannot express is provably outside the look-back.
3. **Report level/scale ablation:** RevIN on/off/α on the high-channel datasets,
   to quantify how much cross-channel scale information we are discarding.

---

## Implementation status

Items 1 and 2 of the plan below are **implemented and unit-tested** (CPU only;
evaluation still needs a GPU).

### `revin_alpha` — α-RevIN (`src/models/dual_domain_model.py`)

New model options `revin_alpha: {fixed, learned, channel}` and
`revin_alpha_init`. The window is blended rather than switched:

```
x_norm = a*(x - mu)/sigma + (1 - a)*x  =  s_inv * x - b
       with s_inv = a/sigma + (1-a),  b = a*mu/sigma
```

Because this is affine in `x`, de-normalization stays exact by feeding
`scale = 1/s_inv`, `loc = b*scale` into the existing reconstruction path — no
other part of the model changes. `a` is sigmoid-parameterized, so it is
unconstrained in optimization but always in `[0,1]`.

Verified:
- `a = 1` reproduces the original RevIN path **exactly** (max |diff| = 0.0e+00).
- `a = 0` reproduces `use_revin: false` **exactly** (max |diff| = 0.0e+00).
- normalize → de-normalize round-trip is exact to float precision (1.9e-06)
  for `a ∈ {0, .25, .5, .75, 1}`.
- `learned`/`channel` receive gradient and `a` moves during training.

The point is not that α is better in one direction — it is that the strength
becomes a *training-fitted parameter* instead of a per-dataset switch chosen by
inspecting a test-set ablation. That removes the selection-bias threat behind
the Solar `IN = no` setting by construction.

### `fusion` — additive modes (`src/models/fusion.py`)

Three new modes alongside the existing convex ones:

| mode | formula | params added |
|---|---|---|
| `residual` | `y = y_time + α·y_freq`, α a scalar, zero-init | 1 |
| `affine` | `y = g_t·y_time + g_f·y_freq`, two independent gates | ~2·d² |
| `doubly_residual` | `y = y_time + φ(feat)·y_freq`, φ zero-init MLP | ~d² |

All three initialize to **exactly** the time-branch forecast, so the
initialization story in the manuscript is unchanged (verified: identical loss
to `time_only` at step 0).

The expressiveness gap is directly demonstrable. Fitting the target
`y_time + y_freq` (pure superposition) with each rule, 600 Adam steps:

| mode | best MSE |
|---|---|
| `gated` (convex) | **0.478** ← cannot represent it |
| `residual` | 0.000000 |
| `affine` | 0.000000 |
| `doubly_residual` | 0.000000 |

This is the concrete form of the limitation: a convex rule confines the output
to the segment between the two branch forecasts, so it cannot emit their sum.

### Running the studies

`scripts/run_ablation.py` gains two named chains:

```bash
# Is the frequency branch inert, or just out-competed by the convex rule?
python scripts/run_ablation.py --config configs/ETTm1.yaml --seeds 5 \
    --variants full fusion_sum fusion_residual fusion_affine fusion_doubly_residual

# Does a learned normalization strength match the hand-picked switch?
python scripts/run_ablation.py --config configs/solar.yaml --seeds 5 \
    --variants full no_revin revin_alpha_learned revin_alpha_channel
```

Solar is the dataset to run the α-RevIN chain on first: if
`revin_alpha_learned` matches or beats the hand-picked `IN = no`, the paper can
drop a test-informed choice and promote the frequency result out of
"exploratory."

---

## Suggested order of work

| # | Action | Cost | Why first |
|---|---|---|---|
| 1 | α-RevIN replacing the binary switch | small code + 1 run/dataset | Removes the paper's biggest validity threat *and* fixes limitation 3. |
| 2 | `fusion: residual` mode | ~20 lines + 1 run | Strictly generalizes current model; tests the alternative explanation for the inert frequency branch. |
| 3 | Training-budget ablation (10-epoch halve vs 50-epoch linear) | GPU-heavy | Determines whether the component nulls are real or recipe artifacts. Publishable either way. |
| 4 | FAN as normalization backend | medium | Highest potential accuracy gain; may reveal the frequency branch is redundant. |

Items 1 and 2 are implementable and smoke-testable **without a GPU**; only
their evaluation needs one. Item 3 is the one that most affects what the paper
is allowed to claim.

**Regardless of what gets run**, three paragraphs should be added to §Threats to
validity: convex-fusion expressiveness (1), the recipe/undertraining confound
(2), and RevIN information loss (3). They are honest limitations that a
Neurocomputing reviewer familiar with FAN/SAN/Dish-TS is likely to raise.
