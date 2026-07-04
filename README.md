# TSP — Dual-Domain Time Series Forecasting

A PyTorch project for multivariate time series forecasting that processes the
look-back window through **two parallel branches** and fuses them:

- **Time-domain branch** — series decomposition (trend/seasonal) + a temporal
  MLP. Captures local, trend, and non-periodic structure.
- **Frequency-domain branch** — real FFT + a learnable complex spectral filter.
  Captures global, periodic structure with a full-window receptive field at
  `O(L log L)` cost.

The two per-channel feature streams are combined with a **gated fusion** and
mapped to the forecast horizon by a linear head. Instance normalization
(RevIN-style) makes the model robust to distribution shift.

```
             input window (B, L, C)
                 /            \
        TimeBranch          FreqBranch
      (decomp + MLP)     (rFFT + spectral filter)
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
│   │   ├── time_branch.py      # decomposition + temporal MLP
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
| `model.time_kernel_size` | moving-average window for trend extraction |
| `model.freq_sparsity` | fraction of high frequencies to drop (low-pass) |
| `model.fusion` | `gated`, `sum`, or `concat` |
| `train.amp` | mixed precision (recommended on RTX 5090) |
| `train.compile` | `torch.compile` (enable once your build supports sm_120) |

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
