# Revision experiment runbook (review items 2–6)

Turnkey commands for the experiments the *Neurocomputing* review asks for.
All scaffolding is in place and smoke-tested on CPU; the runs themselves need
a CUDA GPU (an RTX 5090 was used for the paper). Everything routes through the
**same** data pipeline, splits, training loop, schedule, and evaluation as
DD-Mamba, so comparisons are implementation-confound-free.

Nothing here is run for you (no GPU / dataset files in the authoring
environment). Each command writes JSON under `checkpoints/`; the analysis
scripts turn those into the paper's tables and statistics.

> **Not covered (item 1, deliberately excluded per request):** raising the
> seed count is orthogonal — just pass `--seeds 10` to any runner below. At
> `n=3` all component-placement effects are exploratory; `--seeds 10` is what
> makes them confirmatory.

---

## Item 2 — Unified, same-pipeline baselines

> **Status: done for 6/9 datasets.** ETTh1/2, ETTm1/2, Weather, Electricity
> are complete (`results/unified_*.json`, analyzed in
> `results/unified_analysis.json`, folded into the paper's Table 4). Remaining:
> Solar, Traffic, Exchange (same command, other configs).

Replaces the quoted-from-S-Mamba numbers with reruns in one pipeline. Faithful
in-repo baselines: `dlinear`, `nlinear`, `rlinear`, `patchtst`, `itransformer`,
`smamba`, `msmamba`, and `crossformer` (DSW embedding + two-stage attention;
`src/models/baselines.py`). `crossformer` is now in the default sweep but has
**not been run yet** — its numbers are pending and not in the paper's Table 4.
`tf4tf` has **no** fabricated in-repo model — supply the authors' code through
the adapter.

```bash
# One dataset, DD-Mamba + all standard baselines, 5 seeds, 4 horizons:
python scripts/run_unified_baselines.py --config configs/ETTh1.yaml --seeds 5

# Repeat per dataset (ETTh1/2, ETTm1/2, weather, solar, electricity, traffic,
# exchange_rate, and the PEMS configs from item 6).

# TF4TF via the authors' implementation (installed separately):
python scripts/run_unified_baselines.py --config configs/ETTh1.yaml \
    --archs tf4tf --external_impl tf4tf_official.model:TF4TF --seeds 5
```

Per-baseline capacity knobs live in the config's `model:` block
(`baseline_d_model`, `baseline_layers`, `baseline_d_state`, `patch_len`,
`patch_stride`, `ms_scales`); tune them under a **shared budget** for a fair
comparison. Output: `checkpoints/unified_<name>.json` + a markdown table.

## Item 3 — Validation-only model-selection protocol

Removes the model-selection bias (e.g. Solar RevIN-off chosen on test
ablations). Fixes a candidate switch space, selects on **validation only**,
and reports three labeled models: FIXED, VALIDATION-SELECTED, and a
TEST-ORACLE upper bound (never a result — just the headroom test-peeking buys).

```bash
python scripts/run_selection_protocol.py --config configs/solar.yaml \
    --horizons 96 192 336 720 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,1 \
    --switch model.freq_backbone=fits,none \
    --fixed model.use_revin=true,model.channel_mixer_layers=0
```

Run per dataset; the FIXED assignment should be the SAME everywhere (that is
the point of a fixed model). Output: `checkpoints/selection_<name>.json`
reporting each model's val/test and the selection gap.

## Item 4 — Complete the branch-ablation matrix

The runner is resumable and already parametrized; the paper's gap is
Electricity/Traffic/Exchange at H∈{192,336,720}.

```bash
python scripts/run_branch_matrix.py \
    --datasets electricity traffic exchange_rate \
    --horizons 192 336 720 --seeds 3 \
    --out results/Branch_matrix.json      # merges into the existing file
```

Then recompute significance **with multiplicity control** (feeds Table 6 and
the exploratory/confirmatory split in the paper):

```bash
python scripts/compute_stats_correction.py   # -> results/stats_correction.json
```

## Item 5 — Matched efficiency (wall-clock / latency / throughput / memory)

Profiles DD-Mamba and the baselines on the **same** device and batch, so the
complexity table has matched runtime instead of params/MACs only. Peak GPU
memory needs CUDA.

```bash
python scripts/profile_efficiency.py --device cuda --batch 32 \
    --config configs/traffic.yaml \
    --archs dual_domain smamba itransformer patchtst dlinear rlinear msmamba
```

Emits a per-arch table and LaTeX rows (params, MACs, peak mem, latency,
throughput).

## Item 6 — Confirmatory datasets (PEMS03/04/07/08)

Untouched datasets that were **not** used to design DD-Mamba, to test whether
the validation-only selection logic (item 3) generalizes. Configs:
`configs/PEMS0{3,4,7,8}.yaml`; loader reads the standard `.npz`
(`source: npz`, key `data`, feature 0 = flow).

```bash
# Place the standard files at data/PEMS0{3,4,7,8}.npz, then:
python scripts/run_unified_baselines.py --config configs/PEMS04.yaml --seeds 5
python scripts/run_selection_protocol.py --config configs/PEMS04.yaml \
    --horizons 96 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,2 \
    --fixed model.use_revin=true,model.channel_mixer_layers=0
python scripts/run_branch_matrix.py --datasets PEMS04 --horizons 96 192 336 720
```

(PEMS node counts: 03→358, 04→307, 07→883, 08→170; 5-min sampling;
0.6/0.2/0.2 split, matching the iTransformer/S-Mamba protocol.)

---

## After the runs: fold results back into the paper

1. `compute_stats_correction.py` → refresh the raw-vs-BH counts in
   §Experiments and the branch-matrix caption.
2. Unified table (item 2) → replace the quoted-baseline Table with rerun
   numbers and drop the "indicative, not confirmatory" caveat.
3. Selection report (item 3) → add the fixed/val-selected/oracle table and
   report the selection gap in Threats to Validity.
4. Efficiency table (item 5) → fill the matched-runtime columns.
5. PEMS (item 6) → add as confirmatory rows.
