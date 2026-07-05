# TSP — Dual-Domain Time Series Forecasting

A PyTorch project for multivariate time series forecasting that processes the
look-back window through **two parallel branches** and fuses them:

- **Time-domain branch** — series decomposition (trend/seasonal) + a **Mamba**
  (selective state-space) encoder over the time axis. Captures local, trend,
  and long-range temporal dynamics with linear-time sequence modeling.
- **Frequency-domain branch** — real FFT + a spectral encoder: either a
  learnable complex linear filter (default) or a **bidirectional Mamba**
  scanned over the frequency bins. Captures global, periodic structure with a
  full-window receptive field.

The two per-channel feature streams are combined with a **gated fusion** and
mapped to the forecast horizon by a linear head. Instance normalization
(RevIN-style) makes the model robust to distribution shift.

```
             input window (B, L, C)
                 /            \
        TimeBranch          FreqBranch
   (decomp + Mamba SSM)  (rFFT + spectral filter)
         (B,C,D)             (B,C,D)
                 \            /
                FeatureFusion (gated)
                        |
                 linear head  ->  forecast (B, H, C)
```

## Environment

Targeted local setup:

- Python **3.10**
- GPU: **NVIDIA RTX 5090** (Blackwell, compute capability **sm_120**)
- CUDA **13.0**

> **Important — Blackwell needs a matching PyTorch build.** RTX 5090 is
> `sm_120`. The default PyPI torch wheels (CPU / cu121) do **not** contain
> Blackwell kernels and will fail at runtime with
> *"no kernel image is available for execution on the device."* Install a
> torch build compiled for CUDA 12.8+/13.x (see below).

## Installation

```bash
# 1. Create the environment (Python 3.10)
conda create -n tsp python=3.10 -y
conda activate tsp        # or: python3.10 -m venv .venv && source .venv/bin/activate

# 2. Install a Blackwell-capable PyTorch FIRST.
#    Prefer the CUDA 13.0 channel; fall back to cu128 if cu130 wheels aren't
#    published for your date.
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130
#   fallback:
#   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128

# 3. Install the rest.
pip install -r requirements.txt
#   (or: pip install -e ".[dev]" to also get pytest)
```

Verify the GPU is actually usable:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available()); \
print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_capability(0))"
# expect: ... True  NVIDIA GeForce RTX 5090  (12, 0)
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

## Using your own data

Set the data source to `csv` and point at a wide CSV whose first column is a
timestamp/date and remaining columns are numeric channels:

```bash
python -m src.train \
  --source csv \
  --csv_path data/my_series.csv \
  --seq_len 336 --pred_len 96
```

Or edit `configs/default.yaml`:

```yaml
data:
  source: csv
  csv_path: data/my_series.csv
  target_columns: null   # null = all numeric columns
  seq_len: 336
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
│   │   ├── time_branch.py      # decomposition + Mamba encoder (or MLP)
│   │   ├── freq_branch.py      # rFFT + complex spectral filter
│   │   ├── fusion.py           # gated / sum / concat fusion
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
| `model.fusion` | `gated`, `sum`, or `concat` |
| `train.amp` | mixed precision (recommended on RTX 5090) |
| `train.compile` | `torch.compile` (enable once your build supports sm_120) |

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

`src/utils/metrics.py` reports **MSE**, **MAE**, **RMSE**, and **MAPE** on the
held-out chronological test split.

## Tests

```bash
pytest -q
```

Tests cover branch/model output shapes, dataset windowing, split construction,
and a tiny end-to-end run that asserts the loss decreases (CPU-only, no GPU
required).
