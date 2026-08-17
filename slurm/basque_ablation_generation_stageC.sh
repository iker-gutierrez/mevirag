#!/bin/bash
#SBATCH --job-name=gen-C-eu-abl
#SBATCH --array=0-23%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_generation_stageC_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_generation_stageC_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Basque ablation grid, stage C: rows 9-10, which restrict retrieval to a
# single source corpus (GuiaSalud only, CasiMedicos-Exp only), for both
# models, each in both a plain and a self-feedback generation variant
# (2 rows x 2 models x 2 generation variants = 8 configurations x 3 seeds =
# 24 tasks). The dev set itself (`input`) stays the full mixed Basque
# corpus; only `retrieval_index` is restricted to a single-source index, the
# same mechanism used by the Spanish track's own stage C. Must run after
# basque_ablation_evaluation_stageB.sh, which scores stages A+B and rewrites
# these rows' retrieval settings (in both their plain and self-feedback
# configuration) to match whichever earlier row actually scored best
# (scripts/rewire_basque_ablation_stage.py --stage C).

set -euo pipefail

CONFIGS=(
  configs/experiments/5018_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/5019_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/5020_latxa_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/5021_latxa_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/9009_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_sf_dev.json
  configs/experiments/9010_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_sf_dev.json
  configs/experiments/9020_latxa_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_sf_dev.json
  configs/experiments/9021_latxa_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_sf_dev.json
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

echo "Basque ablation generation (stage C) started on $(hostname)"
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

echo "Basque ablation generation (stage C) finished at $(date)"
