# Electricity: Which Configuration to Use

Two 3-seed experiments were run. **Neither candidate is adopted.** The shipped
configuration stands.

## Experiment 1 — regularization (`halve` vs `halve_reg`, weight_decay 1e-3)

| H | `halve` | `halve_reg` | paired Δ | 95% CI | significant |
|---|---|---|---|---|---|
| 96 | 0.14025 | 0.14016 | −0.00010 | [−0.00051, +0.00031] | no |
| 720 | 0.20715 | 0.20693 | −0.00021 | [−0.00435, +0.00392] | no |

The effect is essentially zero (|t| = 1.02 and 0.22). Raising weight decay 10×
changes nothing measurable. This closes the regularization hypothesis raised
after the schedule sweep: the train→val gap is real, but *this* regularizer
does not shrink it.

**Verdict: do not adopt.** It would change a hyperparameter for no benefit.

## Experiment 2 — α-RevIN (`full` vs `revin_alpha_learned` / `_channel`)

| variant | MSE | Δ vs full | 95% CI | MAE | Δ vs full |
|---|---|---|---|---|---|
| `full` | 0.13982 | — | — | 0.23546 | — |
| `revin_alpha_learned` | **0.13886** | −0.00096 | [−0.00324, +0.00131] | 0.23624 | +0.00078 |
| `revin_alpha_channel` | 0.13899 | −0.00083 | [−0.00419, +0.00253] | 0.23643 | +0.00097 |

Neither reaches significance, and the direction is **not consistent across
metrics**:

- MSE: better in **2 of 3** seeds (−0.00155, −0.00142, +0.00009)
- MAE: worse in **3 of 3** seeds (+0.00051, +0.00007, +0.00175)

So α-RevIN trades MSE for MAE — plausible mechanically, since softening the
normalization retains some level/scale information (helping the large errors
MSE punishes) at the cost of slightly worse typical error. A 0.7% MSE gain that
comes with a 0.3% MAE loss, neither significant, is not a basis for changing
the reported configuration.

**Verdict: do not adopt on this evidence.** Adopting a non-significant,
metric-dependent improvement selected on the test set is precisely the
model-selection bias this paper criticizes elsewhere (§Threats). Doing it here
would be inconsistent with our own stated standard.

### What would settle it

The α-RevIN MSE effect has Cohen's *d* ≈ 1.05 against a seed-to-seed difference
std of 0.00092, so **n = 10 seeds** are needed for 80% power at α = 0.05. If
that run is done and the MSE gain holds under BH correction, the MAE regression
still has to be reported alongside it — the honest framing is a trade, not a
win.

## Decision

Keep `configs/electricity.yaml` as shipped: `d_model: 512`,
`freq_hidden: 512`, `mixer_placement: shared`, `freq_backbone: none`,
`revin_alpha: fixed` (α = 1), `lr_scheduler: halve`, `epochs: 10`.

This is now the configuration that has survived **five** falsification
attempts:

| tested | result |
|---|---|
| `d_model` 512→256 | +0.0045 MSE — worse |
| `freq_backbone` none→fits | +0.0068 MSE — worse |
| `freq_zero_init_head` always | structurally unsound |
| `cosine_warmup_30` (≈5× budget) | +0.0021 MSE — worse |
| `weight_decay` 1e-4→1e-3 | −0.0001 — no effect |
| α-RevIN learned / channel | MSE −0.7%, MAE +0.3%, neither significant |

That is a meaningful result in itself: the configuration is at a local optimum
that six perturbations across capacity, initialization, optimization and
normalization failed to improve.
