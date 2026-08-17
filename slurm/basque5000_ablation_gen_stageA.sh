#!/bin/bash
#SBATCH --job-name=gen-A-eu-abl-5k
#SBATCH --array=0-41%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque5000_ablation_gen_stageA_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque5000_ablation_gen_stageA_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Basque ablation grid, STAGE A = rows 0-6 (no RAG, e5 top-1/3/5, rerank
# top-1/3/5) for both Llama-3.1-8B-Instruct and Latxa-Llama-3.1-8B-Instruct:
# 7 rows x 2 models = 14 configs x 3 seeds = 42 tasks. Reruns the original
# Basque grid (3299-3314) at fresh ids 5000-5013, cloned by a one-off script
# (not tracked) after fixing format_question()'s language=='eu' guard, which
# was silently sending every eu mixed-corpus prompt with NO question text at
# all (topic/subtopic/question/focus don't exist on mixed-corpus records;
# `query` does, but the guard skipped it for eu). The original 3299-3314
# configs/predictions are left on disk untouched (known-bad, not overwritten,
# per standing no-overwrite convention) -- fresh ids only.

set -euo pipefail

CONFIGS=(
  configs/experiments/5000_llama31_8b_no_rag_extractive_guiasalud_dev.json
  configs/experiments/5001_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev.json
  configs/experiments/5002_llama31_8b_rag_e5_topk3_extractive_guiasalud_dev.json
  configs/experiments/5003_llama31_8b_rag_e5_topk5_extractive_guiasalud_dev.json
  configs/experiments/5004_llama31_8b_rag_e5_rerank1_extractive_guiasalud_dev.json
  configs/experiments/5005_llama31_8b_rag_e5_rerank3_extractive_guiasalud_dev.json
  configs/experiments/5006_llama31_8b_rag_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/5007_latxa_llama31_8b_no_rag_extractive_guiasalud_dev.json
  configs/experiments/5008_latxa_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev.json
  configs/experiments/5009_latxa_llama31_8b_rag_e5_topk3_extractive_guiasalud_dev.json
  configs/experiments/5010_latxa_llama31_8b_rag_e5_topk5_extractive_guiasalud_dev.json
  configs/experiments/5011_latxa_llama31_8b_rag_e5_rerank1_extractive_guiasalud_dev.json
  configs/experiments/5012_latxa_llama31_8b_rag_e5_rerank3_extractive_guiasalud_dev.json
  configs/experiments/5013_latxa_llama31_8b_rag_e5_rerank5_extractive_guiasalud_dev.json
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
