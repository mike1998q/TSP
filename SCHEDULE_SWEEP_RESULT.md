# Schedule Sweep: the Undertraining Hypothesis Is Falsified

The training-budget ablation was run on Electricity, all four horizons, one
seed. **The hypothesis I proposed from the training logs was wrong.**

## Result

| arm | H=96 | H=192 | H=336 | H=720 | **avg** |
|---|---|---|---|---|---|
| `halve` (shipped, budget = 2× base) | 0.1401 | 0.1588 | **0.1740** | **0.2033** | **0.1691** |
| `cosine_warmup_30` (~5× the budget) | 0.1402 | 0.1612 | 0.1790 | 0.2045 | 0.1712 |

`halve` wins or ties at every horizon. More optimization budget does not help.

## Why the hypothesis was wrong

The larger budget *did* do what it was supposed to do — it just didn't buy
generalization:

| H | final train loss | | test MSE | |
|---|---|---|---|---|
| | `halve` | `cosine_wu_30` | `halve` | `cosine_wu_30` |
| 96 | 0.1023 | **0.0858** (−16%) | 0.1401 | 0.1402 (+0.1%) |
| 192 | 0.1187 | **0.0822** (−31%) | 0.1588 | 0.1612 (+1.5%) |
| 336 | 0.1337 | **0.0952** (−29%) | 0.1740 | 0.1790 (+2.9%) |
| 720 | 0.1514 | **0.1168** (−23%) | 0.2033 | 0.2045 (+0.6%) |

Training loss falls 16–31% further; test error is flat or worse. The
train→validation gap widens from 14–32% under `halve` to **38–69%** under
cosine.

**The model was never optimization-limited.** It can already fit the training
distribution considerably harder than it can generalize. The aggressive `halve`
decay was functioning as an *implicit regularizer* — annealing the learning
rate to near zero is a form of early stopping — not starving the fit. It is a
reasonable recipe, not a bug.

My reasoning error: I treated "training loss still falling at the last epoch"
as evidence of undertraining. It is not. It only shows the model has capacity
left to memorize; whether that capacity helps is an empirical question, and
here the answer is no.

## What the sweep actually reveals

The limit is the **generalization gap**, in two stages (under `halve`):

| H | train→val | val→test |
|---|---|---|
| 96 | +15.7% | +18.3% |
| 192 | +14.2% | +17.1% |
| 336 | +16.5% | +11.7% |
| 720 | +32.4% | +1.4% |

The val→test gap of 12–18% at the shorter horizons is **distribution shift**
between the validation and test periods, not capacity or optimization. This is
the same phenomenon that FAN (NeurIPS 2024), SAN (NeurIPS 2023) and Dish-TS
(AAAI 2023) target, as surveyed in `DESIGN_LIMITATIONS_LITERATURE.md`.

Combined with the earlier capacity result — reducing width *hurt*
(`CAPACITY_REGRESSION_ANALYSIS.md`) — the model sits near the useful capacity
for this data and recipe. Three levers have now been tested and none is the
answer:

| lever | result |
|---|---|
| more parameters (d512 vs d256) | reducing hurt; already at the good setting |
| more optimization budget | no gain, slightly worse |
| spectral pathway (FITS) | worse, badly so at H=720 |

## A positive consequence for the paper

This *strengthens* the component-ablation nulls. A standing objection was that
zero-initialized branches and mixers must grow their contribution and were
never given the budget to do so. Under a 5× larger budget they still produce no
test gain, so "no component effect survives correction" is **not** an artifact
of the learning-rate schedule. That objection is now closed with evidence.

## Where the remaining gap actually is

On Electricity we are close to, not far from, the strongest published results:

| model | avg MSE |
|---|---|
| **DD-Mamba** | **0.169** |
| FLD-Mamba (published) | 0.169 |
| S-Mamba (published) | 0.170 |

The dataset average is at parity. The gap is specifically at **H=96** (0.140 vs
FLD-Mamba's 0.137, ≈2%). Framing this as "not top-tier" overstates it — it is a
short-horizon gap of a couple of percent on one dataset.

## Evidence-backed next steps

Optimization and capacity are excluded, so the remaining levers are
regularization and non-stationarity handling:

1. **Regularization sweep** — the `halve_reg` arm (`weight_decay: 1e-3`) is
   already implemented; add dropout variants. This directly targets the
   14–32% train→val gap and is the cheapest remaining test.
   ```bash
   python scripts/run_schedule_sweep.py --config configs/electricity.yaml \
       --horizons 96 720 --arms halve halve_reg --seeds 3
   ```
2. **α-RevIN** (`revin_alpha: learned`, already implemented and tested) —
   targets the val→test distribution shift, which is the larger of the two gaps
   at short horizons.
   ```bash
   python scripts/run_ablation.py --config configs/electricity.yaml --seeds 3 \
       --variants full revin_alpha_learned revin_alpha_channel
   ```
3. **FAN-style frequency-domain normalization** — the literature's strongest
   reported answer to this exact gap (7.76–37.90% MSE improvement,
   model-agnostic). Not yet implemented.

Note that H=720 overfits under *both* schedules (the run diagnostic fired
correctly for it in both arms), so regularization is the indicated lever there
regardless.

## Corrections made

The manuscript claim added in the previous round — that the models are
undertrained and the component nulls may reflect the recipe — was falsified by
this experiment and has been rewritten. The Threats paragraph now reports the
sweep, states that the ceiling is generalization rather than optimization, and
notes that the result strengthens rather than qualifies the component nulls.
The `halve` warnings in `configs/default.yaml` and `build_scheduler` were
likewise corrected to say the budget bound is real but does not cost accuracy.
