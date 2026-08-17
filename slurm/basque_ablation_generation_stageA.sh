#!/bin/bash
#SBATCH --job-name=gen-A-eu-abl
#SBATCH --array=0-41%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_generation_stageA_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_generation_stageA_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Basque ablation grid, stage A: rows 0-6 (no retrieval, dense-retrieval
# depth 1/3/5, reranking depth 1/3/5) for both Llama-3.1-8B-Instruct and
# Latxa-Llama-3.1-8B-Instruct. Every config has self-feedback enabled: a
# single generation run produces both an initial answer and a self-feedback
# revision of it, so each row's two readings (initial and revised) are both
# available for the staged selection to compare without a second generation
# pass (7 rows x 2 models = 14 configurations x 3 seeds = 42 tasks). Output
# goes into a dedicated configuration id range (11000-11013) reserved for
# this rerun of the grid, so no earlier round's configuration or generated
# predictions are ever overwritten.

set -euo pipefail

CONFIGS=(
  configs/experiments/11000_llama31_8b_no_rag_extractive_guiasalud_dev.json
  configs/experiments/11001_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev.json
  configs/experiments/11002_llama31_8b_rag_e5_topk3_extractive_guiasalud_dev.json
  configs/experiments/11003_llama31_8b_rag_e5_topk5_extractive_guiasalud_dev.json
  configs/experiments/11004_llama31_8b_rag_e5_rerank1_extractive_guiasalud_dev.json
  configs/experiments/11005_llama31_8b_rag_e5_rerank3_extractive_guiasalud_dev.json
  configs/experiments/11006_llama31_8b_rag_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/11011_latxa_llama31_8b_no_rag_extractive_guiasalud_dev.json
  configs/experiments/11012_latxa_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev.json
  configs/experiments/11013_latxa_llama31_8b_rag_e5_topk3_extractive_guiasalud_dev.json
  configs/experiments/11014_latxa_llama31_8b_rag_e5_topk5_extractive_guiasalud_dev.json
  configs/experiments/11015_latxa_llama31_8b_rag_e5_rerank1_extractive_guiasalud_dev.json
  configs/experiments/11016_latxa_llama31_8b_rag_e5_rerank3_extractive_guiasalud_dev.json
  configs/experiments/11017_latxa_llama31_8b_rag_e5_rerank5_extractive_guiasalud_dev.json
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

echo "Basque ablation generation (stage A) started on $(hostname)"
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

echo "Basque ablation generation (stage A) finished at $(date)"
