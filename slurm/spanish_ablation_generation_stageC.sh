#!/bin/bash
#SBATCH --job-name=gen-C-es-abl
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/spanish_ablation_generation_stageC_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/spanish_ablation_generation_stageC_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Spanish ablation grid, stage C: rows 9-10, which restrict retrieval to a
# single source corpus (GuiaSalud only, CasiMedicos-Exp only), for both
# Qwen3.5-9B variants. Every config has self-feedback enabled, so each
# row's initial and revised answers both come from one generation run (2
# rows x 2 model variants = 4 configurations x 3 seeds = 12 tasks). The dev
# set itself (`input`) stays the full mixed Spanish corpus; only
# `retrieval_index` is restricted to a single-source index. Must run after
# spanish_ablation_evaluation_stageB.sh, which scores stages A+B and
# rewrites these rows' retrieval settings to match whichever earlier row
# actually scored best (scripts/rewire_spanish_ablation_stage.py --stage
# C). Requires the single-source retrieval indices
# (guiasalud_only_multilingual_e5_large, casimedicos_only_multilingual_e5_
# large) to already be built.

set -euo pipefail

CONFIGS=(
  configs/experiments/12009_qwen35_9b_rag_domain_guiasalud_e5_rerank5_no_think_extractive_guiasalud_dev.json
  configs/experiments/12010_qwen35_9b_rag_domain_casimedicos_e5_rerank5_no_think_extractive_guiasalud_dev.json
  configs/experiments/12020_qwen35_9b_rag_domain_guiasalud_e5_rerank5_think_extractive_guiasalud_dev.json
  configs/experiments/12021_qwen35_9b_rag_domain_casimedicos_e5_rerank5_think_extractive_guiasalud_dev.json
)
SEEDS=(42 43 44)

CFG_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
CONFIG="${CONFIGS[$CFG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
RUN_NAME="$(basename "$CONFIG" .json)"
OUTPUT="experiments/runs/${RUN_NAME}_seed${SEED}/predictions.jsonl"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Spanish ablation generation (stage C) started on $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID=${SLURM_JOB_ID:-}"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID:-}"
echo "CONFIG=${CONFIG}"
echo "SEED=${SEED}"
echo "OUTPUT=${OUTPUT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
nvidia-smi || true

scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
  --config "$CONFIG" --seed "$SEED" --output "$OUTPUT"

echo "Spanish ablation generation (stage C) finished at $(date)"
