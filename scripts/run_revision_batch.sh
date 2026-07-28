#!/usr/bin/env bash
# =============================================================================
# DD-Mamba revision experiments — run together, in one batch.
#
# Requires: a CUDA GPU and the dataset files under data/ (ETT*.csv, weather,
# solar_AL.txt.gz, electricity/traffic/exchange_rate, and PEMS0{3,4,7,8}.npz
# for the confirmatory step). No arguments; run from the repo root:
#     bash scripts/run_revision_batch.sh
#
# Safe to re-run: the unified and branch-matrix runners resume/overwrite
# per-config JSON, so an interrupted batch can simply be started again.
# Every command is teed to logs/revision_batch.log. Ordered by the three
# reviewer concerns; earlier sections are higher priority.
# =============================================================================
set -u
cd "$(dirname "$0")/.."
mkdir -p logs results
LOG="logs/revision_batch.log"
run () { echo -e "\n\033[1m=== $* ===\033[0m" | tee -a "$LOG"; "$@" 2>&1 | tee -a "$LOG"; }

ALL_DATASETS="ETTh1 ETTh2 ETTm1 ETTm2 weather electricity solar traffic exchange_rate"

# -----------------------------------------------------------------------------
# CONCERN 1 (most serious): kill the Solar RevIN test-set leakage.
# Re-select every tuned switch on VALIDATION only; report fixed / val-selected /
# test-oracle. After this, set each config to its *validation-selected* switch
# (printed in checkpoints/selection_<name>.json) BEFORE trusting Section-4 numbers
# — most importantly Solar's RevIN flag.
# -----------------------------------------------------------------------------
run python scripts/run_selection_protocol.py --config configs/solar.yaml \
    --horizons 96 192 336 720 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,1 \
    --fixed model.use_revin=true,model.channel_mixer_layers=1

run python scripts/run_selection_protocol.py --config configs/electricity.yaml \
    --horizons 96 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.mixer_placement=both,shared \
    --fixed model.use_revin=true,model.mixer_placement=both

run python scripts/run_selection_protocol.py --config configs/weather.yaml \
    --horizons 96 --seeds 5 \
    --switch model.use_revin=true,false \
    --switch model.channel_mixer_layers=0,2 \
    --fixed model.use_revin=true,model.channel_mixer_layers=2

# Hands-off: write each VALIDATION-SELECTED switch back into its config
# (comments preserved), so the runs below use validation-chosen settings only
# — no test-informed choice survives. --dry-run first to log the diff.
run python scripts/apply_selection.py --dry-run \
    checkpoints/selection_solar.json \
    checkpoints/selection_electricity.json \
    checkpoints/selection_weather.json
run python scripts/apply_selection.py \
    checkpoints/selection_solar.json \
    checkpoints/selection_electricity.json \
    checkpoints/selection_weather.json

# -----------------------------------------------------------------------------
# CONCERN 3: unified same-pipeline baselines (now INCLUDING Crossformer, which
# is in the default arch sweep) across ALL nine datasets — completes the 6/9
# coverage and the Crossformer rerun. Electricity here uses the shipped
# d512+shared config, so this run also confirms the accuracy/params balance.
# -----------------------------------------------------------------------------
for ds in $ALL_DATASETS; do
    run python scripts/run_unified_baselines.py --config "configs/${ds}.yaml" --seeds 5
done
run python scripts/analyze_unified_baselines.py   # -> results/unified_analysis.json

# -----------------------------------------------------------------------------
# CONCERN 2 (support the narrowed claims): raise ablation power to 10 seeds and
# complete the branch matrix, then recompute BH-corrected significance.
# -----------------------------------------------------------------------------
for ds in ETTh1 solar weather; do
    run python scripts/run_ablation.py --config "configs/${ds}.yaml" --seeds 10
done
run python scripts/run_branch_matrix.py \
    --datasets electricity traffic exchange_rate --horizons 192 336 720 --seeds 3 \
    --out results/Branch_matrix.json
run python scripts/compute_stats_correction.py    # -> results/stats_correction.json

# -----------------------------------------------------------------------------
# Confirmatory datasets (does validation-only selection generalize?) — PEMS.
# Needs data/PEMS0{3,4,7,8}.npz. dual_domain only keeps it cheap; add more
# --archs to widen the comparison.
# -----------------------------------------------------------------------------
for ds in PEMS03 PEMS04 PEMS07 PEMS08; do
    run python scripts/run_unified_baselines.py --config "configs/${ds}.yaml" \
        --archs dual_domain --seeds 5
done

# -----------------------------------------------------------------------------
# Matched efficiency (wall-clock / latency / throughput / peak memory) on GPU.
# -----------------------------------------------------------------------------
run python scripts/profile_efficiency.py --device cuda --batch 32 \
    --config configs/traffic.yaml \
    --archs dual_domain smamba itransformer patchtst crossformer dlinear rlinear msmamba

echo -e "\n\033[1mBATCH COMPLETE.\033[0m Outputs: checkpoints/unified_*.json, "\
"checkpoints/selection_*.json, results/{unified_analysis,stats_correction}.json. "\
"Fold them back into the paper per EXPERIMENTS.md." | tee -a "$LOG"
