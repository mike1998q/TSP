# TSP — Dual-Domain Time Series Forecasting

A PyTorch project for multivariate time series forecasting that processes the
look-back window through **two parallel branches** and fuses them:

- **Time-domain branch** — series decomposition (trend/seasonal) with an
  internal **linear backbone** (DLinear-style window→horizon maps) plus a
  **Mamba** (selective state-space) encoder whose zero-initialized head adds
  nonlinear corrections. Captures local, trend, and long-range temporal
  dynamics.
- **Frequency-domain branch** — real FFT + a spectral encoder: either a
  learnable complex linear filter (default) or a **bidirectional Mamba**
  scanned over the frequency bins, followed by its own forecast head.
  Optionally anchored by a **FITS-style spectral linear backbone**
  (`freq_backbone: fits`): a zero-initialized complex linear map that
  interpolates the low-passed spectrum to length L+H and reads off the
  horizon — linear-in-frequency forecasting, near-SOTA on ETTh/weather.
  Captures global, periodic structure with a full-window receptive field.

**Each branch produces its own forecast**; the output is a gated convex
combination of the two, with the gate conditioned on both branches' features.
Nothing bypasses the dual-branch architecture — every prediction flows
through a branch. Instance normalization (RevIN-style) makes the model robust
to distribution shift.

With `model.channel_mixer_layers > 0` (default 1), each branch also runs a
**bidirectional Mamba across the variate dimension** (S-Mamba style) before
its head, so every channel's forecast can exploit the other channels' state —
the key ingredient separating first-class multivariate models from
channel-independent ones on electricity/traffic/weather. Set 0 for a strictly
channel-independent model.

```
             input window (B, L, C)  --RevIN-->
                 /                    \
        TimeBranch                  FreqBranch
   decomp -> linear backbone     rFFT -> spectral encoder
   + Mamba correction (0-init)   -> forecast head
   -> y_time, feat_t             -> y_freq, feat_f
                 \                    /
          g = σ(gate(feat_t, feat_f))
          y = g·y_time + (1−g)·y_freq   ->  forecast (B, H, C)
```

## Environment

Targeted local setup:

- Python **3.10**
- GPU: **NVIDIA RTX 5090** (Blackwell, compute capability **sm_120**)
- CUDA **13.0**
- PyTorch **2.11.0+cu130** (verified)

> **Important — Blackwell needs a matching PyTorch build.** RTX 5090 is
> `sm_120`. The default PyPI torch wheels do **not** contain Blackwell
> kernels and will fail at runtime with
> *"no kernel image is available for execution on the device."* Install the
> `+cu130` build from the PyTorch cu130 wheel index (see below).

## Installation

```bash
# 1. Create the environment (Python 3.10)
conda create -n tsp python=3.10 -y
conda activate tsp        # or: python3.10 -m venv .venv && source .venv/bin/activate

# 2. Install the Blackwell-capable PyTorch FIRST (torch 2.11.0+cu130).
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu130

# 3. Install the rest.
pip install -r requirements.txt
#   (or: pip install -e ".[dev]" to also get pytest)
```

Verify the GPU is actually usable:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
# expect: 2.11.0+cu130 True  NVIDIA GeForce RTX 5090  (12, 0)
```

## Quick start

```bash
# Smoke test on synthetic data (no dataset required):
python scripts/run_synthetic.py

# Full training with the default config:
python -m src.train --config configs/default.yaml

# Override any config field from the CLI (dotted or bare keys):
python -m src.train --seq_len 192 --pred_len 96 --epochs 50 --batch_size 64 --device cuda

# Evaluate a checkpoint and save a forecast plot:
python -m src.evaluate --checkpoint checkpoints/dual_domain_default_best.pt --plot forecast.png
```

## Benchmark datasets

Per-dataset configs with tuned hyperparameters ship in `configs/` for the
standard long-term forecasting benchmarks:

| Dataset | Config | Channels | Freq | Rows | Split | Batch | LR | Notes |
|---|---|---|---|---|---|---|---|---|
| ETTh1 | `configs/ETTh1.yaml` | 7 | 1 h | 17,420 | canonical 12/4/4 mo | 32 | 1e-4 | halve LR, 1 Mamba layer, dropout 0.2, mixer off |
| ETTh2 | `configs/ETTh2.yaml` | 7 | 1 h | 17,420 | canonical 12/4/4 mo | 32 | 1e-4 | halve LR, 1 Mamba layer, dropout 0.3, low-pass 0.3, mixer off |
| ETTm1 | `configs/ETTm1.yaml` | 7 | 15 min | 69,680 | canonical 12/4/4 mo | 32 | 1e-4 | halve LR, mixer off |
| ETTm2 | `configs/ETTm2.yaml` | 7 | 15 min | 69,680 | canonical 12/4/4 mo | 32 | 1e-4 | halve LR, dropout 0.2, low-pass 0.2, mixer off |
| Weather | `configs/weather.yaml` | 21 | 10 min | 52,696 | 0.7/0.1/0.2 | 32 | 1e-4 | halve LR, d_model 256, MLP time + 2 variate-Mamba layers, FITS (~5M params) |
| Electricity | `configs/electricity.yaml` | 321 | 1 h | 26,304 | 0.7/0.1/0.2 | 16 | 5e-4 | halve LR, d_model 512, MLP time + 2 variate-Mamba layers |
| Solar-Energy | `configs/solar.yaml` | 137 | 10 min | 52,560 | 0.7/0.1/0.2 | 16 | 5e-4 | halve LR, reads `solar_AL.txt` directly |
| Exchange-Rate | `configs/exchange_rate.yaml` | 8 | 1 day | 7,588 | 0.7/0.1/0.2 | 32 | 1e-4 | halve LR, d_model 64, dropout 0.3, low-pass 0.5, mixer off |
| Traffic | `configs/traffic.yaml` | 862 | 1 h | 17,544 | 0.7/0.1/0.2 | 16 | 1e-3 | halve LR, d_model 512, MLP time + 2 variate-Mamba layers |

All configs train 10 epochs with the halve LR schedule. The cross-channel
mixer is sized to the dataset: **off** on ETT and Exchange-Rate (few channels,
small data — it only adds overfitting capacity), 1 layer on weather/solar,
2 layers at d_model 512 on electricity/traffic where it is the core of the
model (the S-Mamba recipe: MLP over time, Mamba over variates).

All datasets use `seq_len: 96` — the standard fixed look-back of the
Autoformer/TimesNet/iTransformer evaluation protocol, kept identical across
datasets for fair comparison — and default to `pred_len: 96`; the standard horizons
{96, 192, 336, 720} are a CLI override away. Batch sizes are sized for a
32 GB RTX 5090 given the channel-independent folding (effective sequences per
step = `batch_size × channels`).

**Getting the data:** all nine files are in the standard benchmark bundle used
by Autoformer / TimesNet / iTransformer — see the "datasets" download link in
the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) README
(Google Drive). Place the files in `data/`:

```
data/ETTh1.csv  data/ETTh2.csv  data/ETTm1.csv  data/ETTm2.csv
data/weather.csv  data/electricity.csv  data/traffic.csv
data/exchange_rate.csv  data/solar_AL.txt
```

Run a single benchmark, or sweep everything:

```bash
python -m src.train --config configs/ETTh1.yaml                # H=96
python -m src.train --config configs/ETTh1.yaml --pred_len 336 # other horizon

./scripts/run_benchmarks.sh                    # all datasets x {96,192,336,720}
./scripts/run_benchmarks.sh ETTh1 weather      # subset
./scripts/run_benchmarks.sh solar --freq_encoder mamba  # forward extra flags
```

Results land in `checkpoints/<name>_results.json` (MSE/MAE per run).

## Using your own data

Set the data source to `csv` and point at a wide CSV whose first column is a
timestamp/date and remaining columns are numeric channels:

```bash
python -m src.train \
  --source csv \
  --csv_path data/my_series.csv \
  --seq_len 96 --pred_len 96
```

Or edit `configs/default.yaml`:

```yaml
data:
  source: csv
  csv_path: data/my_series.csv
  target_columns: null   # null = all numeric columns
  seq_len: 96
  pred_len: 96
```

## Project layout

```
TSP/
├── configs/default.yaml        # all hyperparameters
├── src/
│   ├── data/                   # sliding-window dataset + loaders + scaler
│   ├── models/
│   │   ├── mamba_block.py      # pure-torch Mamba SSM (+ optional fast kernels)
│   │   ├── time_branch.py      # decomp + linear backbone + Mamba correction
│   │   ├── freq_branch.py      # rFFT + spectral encoder + forecast head
│   │   ├── fusion.py           # forecast-level gated fusion
│   │   └── dual_domain_model.py# full model + RevIN normalization
│   ├── utils/                  # metrics, config, seeding, device, plotting
│   ├── train.py                # training loop (AMP, early stop, cosine LR)
│   └── evaluate.py             # checkpoint evaluation + plotting
├── scripts/                    # run_synthetic.py, train.sh
├── tests/                      # pytest shape + learning smoke tests
└── requirements.txt / pyproject.toml
```

## Key config options

| Field | Meaning |
|-------|---------|
| `data.seq_len` / `data.pred_len` | look-back / horizon lengths |
| `model.d_model` | shared hidden width of both branches |
| `model.time_encoder` | `mamba` (default) or `mlp` for the time branch |
| `model.mamba_layers` | number of stacked Mamba blocks |
| `model.mamba_d_state` | SSM state dimension `N` |
| `model.mamba_expand` | inner expansion factor (`d_inner = expand * d_model`) |
| `model.time_kernel_size` | moving-average window for trend extraction |
| `model.freq_encoder` | `linear` (default) or `mamba` for the frequency branch |
| `model.freq_sparsity` | fraction of high frequencies to drop (low-pass) |
| `model.fusion` | forecast fusion: `gated` (per-channel gate), `concat` (per-step gate), `sum` (average) |
| `model.channel_mixer_layers` | BiMamba layers across the variate dim (0 = channel-independent) |
| `model.freq_backbone` | `none` or `fits` (spectral linear forecast anchor, zero-init) |
| `model.use_revin` / `model.time_linear_backbone` | ablation switches (default on) |
| `train.patience` / `train.min_delta` | early-stopping knobs |
| `train.amp` | mixed precision (recommended on RTX 5090) |
| `train.compile` | `torch.compile` (supported on torch 2.11+cu130; opt-in) |

## Mamba encoders

Both branches can use **Mamba** selective state-space encoders
(`src/models/mamba_block.py`), shipped as a **self-contained pure-PyTorch
implementation** (selective scan + causal depthwise conv), so they train on
any CPU or GPU with no custom kernels.

- **Time branch** (`time_encoder: mamba`, default): a *causal* `MambaEncoder`
  over the timesteps — the natural arrow-of-time bias.
- **Frequency branch** (`freq_encoder: mamba`, off by default): a
  *bidirectional* `BiMambaEncoder` over the frequency bins. Each bin's
  [real, imag] pair is embedded and scanned DC→Nyquist **and** Nyquist→DC —
  frequency has no arrow of time, so a one-way scan would be an arbitrary
  bias. Compared to the dense `linear` complex filter, the mixing is
  input-dependent (selective) and scales linearly in the number of bins.

```bash
# Fully-SSM dual domain: Mamba over time AND over frequency bins
python -m src.train --freq_encoder mamba

# A/B against the dense complex filter baseline
python -m src.train --freq_encoder linear
```

If the official `mamba-ssm` package is installed, `MambaLayer` **automatically**
uses its fused CUDA kernels instead — no code change needed:

```bash
# Optional speed-up (only if your toolchain can build them):
pip install causal-conv1d>=1.4.0
pip install mamba-ssm>=2.2.0
```

> On brand-new stacks like RTX 5090 / Blackwell (sm_120) + CUDA 13.0, the
> official kernels can be hard to build. The pure-PyTorch path lets you train
> immediately; add the fast kernels later and the model picks them up on its
> own. The pure selective scan is a sequential `O(L)` loop — fine on GPU, but
> slow on CPU for long windows, so prefer CUDA (or shorter `seq_len`) there.

Switch back to the plain MLP encoder any time:

```bash
python -m src.train --time_encoder mlp
```

## Metrics

`src/utils/metrics.py` reports **MSE** and **MAE** on the held-out
chronological test split (also used for validation/early stopping).

## Ablation studies

`scripts/run_ablation.py` verifies each component's contribution: every
variant flips exactly **one** switch relative to the dataset's base config
(same split, schedule, seeds, and all other hyperparameters), so the metric
delta isolates that component.

| variant | component isolated | question it answers |
|---|---|---|
| `full` | — | reference |
| `time_only` | frequency branch | does the spectral view add accuracy? |
| `freq_only` | time branch | does the temporal view add accuracy? |
| `fusion_sum` | learned gate | is the feature-conditioned gate better than averaging? |
| `no_revin` | instance normalization | does RevIN handle distribution shift? |
| `no_linear_backbone` | DLinear backbone (time) | is the linear anchor load-bearing? |
| `no_fits` / `with_fits` | FITS spectral backbone | is linear-in-frequency load-bearing? |
| `time_mlp` / `time_mamba` | time-axis Mamba | does the selective SSM beat an MLP over time? |
| `no_channel_mixer` / `with_channel_mixer` | variate Mamba mixing | do cross-channel dependencies matter here? |
| `freq_mamba` | spectral encoder type | selective vs dense-linear bin mixing |

```bash
# Full suite on one dataset (auto-selects the variants that apply):
python scripts/run_ablation.py --config configs/ETTh1.yaml

# 3 seeds for mean +/- std, or a subset of variants:
python scripts/run_ablation.py --config configs/weather.yaml --seeds 3
python scripts/run_ablation.py --config configs/electricity.yaml \
    --variants full no_channel_mixer time_only freq_only

# Any config field can be overridden for all variants (e.g. quick pass):
python scripts/run_ablation.py --config configs/ETTh1.yaml --epochs 5
```

Results print as a markdown table (Δmse vs `full`) and are saved to
`checkpoints/ablation_<name>.json`. Recommended reading of the table:
`time_only`/`freq_only` quantify the dual-branch claim itself; the paired
on/off variants quantify each Mamba and each linear anchor. Expected
signatures: the channel mixer matters on electricity/traffic/weather but not
ETT; FITS and the linear backbone matter most on ETTh; RevIN matters
everywhere there is distribution shift (ETT especially). Use `--seeds 3`
before drawing conclusions — single-seed deltas below ~0.005 MSE are noise.

For a per-sample view of branch contributions, the model also exposes
`model(x, return_components=True)`, returning the de-normalized per-branch
forecasts alongside the fused output.

## Troubleshooting

**Training stops after only a few epochs ("early shutdown").**
Two mechanisms used to cause this, both fixed:

1. *AMP fp16 instability in the Mamba scan* — the selective scan
   (`exp(Δ·A)` + an L-step recurrent state accumulation) under/overflows in
   half precision. When the validation loss went NaN, `NaN < best` is always
   false, so patience silently ran out and training stopped with a garbage
   model. The scan (and RMSNorm) now always executes in fp32 internally, even
   under `train.amp: true`, matching what the official fused kernels do. If
   you still see the `[warn] validation loss is not finite` message, lower
   the LR or set `train.amp: false`.
2. *Over-aggressive early stopping* — patience was too small relative to the
   cosine LR schedule, killing runs while the LR was still high. Patience is
   raised (ETT: 10/6) and `train.min_delta` controls the improvement
   threshold explicitly.

**Poor accuracy on ETT.**
ETT is the benchmark where a plain linear history→future map (DLinear) is
near-SOTA. The model originally forecast from a pooled `d_model` summary
alone — an information bottleneck that collapses toward mean-reverting
predictions on ETT. The time-domain branch now carries a **linear backbone
inside the branch**: per-component (seasonal/trend) linear maps from the
look-back window straight to the horizon, with the branch's deep head
zero-initialized so it starts as an exact DLinear and learns nonlinear
*corrections* on top. Fusion happens at the **forecast level** (a gated
convex combination of the two branch forecasts), so the linear capability
strengthens the time branch rather than bypassing the dual-branch
architecture. All configs keep the protocol-standard `seq_len: 96` for fair
comparison; if you don't need protocol comparability, linear-backbone models
often gain further on ETT from a longer look-back (`--seq_len 336`).

Old checkpoints from before this change are incompatible with the new
`state_dict` — retrain them.

**ETT numbers not comparable to published results / val fluctuates while
train falls.** Three causes, all fixed:

1. *Wrong split protocol* — published ETT numbers use fixed month borders
   (train 0–12, val 12–16, test 16–20; the file's tail is discarded), not
   ratio splits over all rows. `data.split_protocol: ETTh|ETTm` now
   reproduces the canonical borders exactly (e.g. ETTh1 H=96 →
   8449/2785/2785 windows); the ETT configs set it by default. A ratio split
   evaluates on a different — and harder — test window than the literature.
2. *Schedule mismatch* — cosine over 30 epochs holds the LR high while the
   deep branches memorize the small training set (train loss falls, val
   rises from ~epoch 6). ETT configs now use the standard recipe:
   10 epochs, `lr_scheduler: halve` (lr × 0.5 per epoch), patience 3.
3. *Noisy fusion start* — the gate initialized at 0.5 while the frequency
   head was random, so half the initial forecast was noise. The gate now
   initializes at g ≈ 0.9 toward the time branch (an exact linear model at
   init) and remains fully learnable.

**Early stop on weather/electricity/traffic; accuracy below first-class
Mamba models.** The big-dataset configs previously kept the cosine schedule
with patience 3 — the same mismatch as ETT (val plateaus around epoch 5 while
cosine still holds the LR high, so patience fires mid-schedule). All configs
now use the halve recipe. The remaining accuracy gap to S-Mamba-class models
was architectural: they are **channel-mixing** — each variate's forecast uses
the other variates' state — while this model was strictly
channel-independent. `model.channel_mixer_layers` adds a bidirectional Mamba
(with per-layer FFN, the S-Mamba block) over the variate dimension inside
each branch before its head, plus a whole-window series embedding in the time
branch.

**Where the mixer belongs — capacity must match the dataset.** Two opposite
failure modes, both observed:

- *ETTh1/ETTh2 early-stopped again* after the mixer was enabled there: it
  tripled the model to ~824K params against only 8,449 train windows on
  7 near-independent channels — pure overfitting capacity, val degraded from
  epoch ~2 and patience fired. ETT configs now run **mixer off**, a shallow
  time encoder (1 Mamba layer), and patience 10 so all 10 (cheap, halve-LR)
  epochs always run with the best checkpoint kept — early stop cannot fire.
- *Electricity/traffic were poor* for the opposite reason: underfitting.
  Top models run these at d_model 512 with multiple variate-mixing layers;
  we were at d_model 128 because the time-axis Mamba scan (folded over
  batch x channels) was the memory hog. The high-channel configs now follow
  the actual S-Mamba recipe: **MLP/linear over the time axis, Mamba capacity
  in the variate dimension** — `time_encoder: mlp`, `d_model: 512`,
  `channel_mixer_layers: 2` (~19M params), traffic batch 16. The Mamba scan
  over 321/862 variate tokens is cheap; the model scales to top-model width.

**Closing the remaining gap to top-level models (per dataset group):**

- *ETTh1/ETTh2* — two fixes: (1) the linear backbones were **undertrained**:
  lr 1e-4 under the halve schedule integrates to ~2 effective epochs, while
  DLinear-class linear maps train at 50x that; ETTh lr is now 5e-4. (2) The
  freq branch gained the **FITS backbone** (`freq_backbone: fits` +
  `freq_sparsity` as its low-pass cutoff) — on ETTh, linear-in-frequency is
  what actually works, and the branch previously had no linear anchor. Both
  branches now start as exact linear forecasters in their own domain.
- *Weather* — moved to the high-channel recipe *shape* (MLP time encoder,
  2 variate-mixer layers, FITS backbone) but sized to the dataset:
  `d_model: 256` (~5M params). The full d512/19M width is justified for
  321/862 variates, not 21 — capacity must scale with the channel count.
- *Electricity/traffic* — `mamba_d_state: 32` (richer per-variate SSM state)
  and dropout wired into the mixer FFNs; capacity was already right.
- *ETTm1/ETTm2* — untouched: they already perform well, and `freq_backbone`
  defaults to `none`, so their model is bit-identical to before.

## Tests

```bash
pytest -q
```

Tests cover branch/model output shapes, dataset windowing, split construction,
and a tiny end-to-end run that asserts the loss decreases (CPU-only, no GPU
required).
