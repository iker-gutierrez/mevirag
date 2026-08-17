#!/bin/bash
#SBATCH --job-name=gen-B-eu-abl
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_generation_stageB_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_generation_stageB_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Basque ablation grid, stage B: row 7 (few-shot prompting, no retrieval)
# and row 8 (few-shot prompting combined with the best retrieval setting
# found in stage A), for both models. Every config has self-feedback
# enabled, so each row's initial and revised answers both come from one
# generation run (2 rows x 2 models = 4 configurations x 3 seeds = 12
# tasks). Must run after basque_ablation_evaluation_stageA.sh, which scores
# stage A and rewrites row 8's retrieval settings to match whichever
# stage-A row actually scored best (scripts/rewire_basque_ablation_stage.py
# --stage B).

set -euo pipefail

CONFIGS=(
  configs/experiments/11007_llama31_8b_3shot_no_rag_extractive_guiasalud_dev.json
  configs/experiments/11008_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/11018_latxa_llama31_8b_3shot_no_rag_extractive_guiasalud_dev.json
  configs/experiments/11019_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_dev.json
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

echo "Basque ablation generation (stage B) started on $(hostname)"
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

echo "Basque ablation generation (stage B) finished at $(date)"
