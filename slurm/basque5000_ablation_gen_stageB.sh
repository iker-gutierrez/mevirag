#!/bin/bash
#SBATCH --job-name=gen-B-eu-abl-5k
#SBATCH --array=0-23%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque5000_ablation_gen_stageB_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque5000_ablation_gen_stageB_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Basque ablation grid, STAGE B = rows 7-8 (3-shot no RAG, 3-shot + best RAG)
# for both models: 2 rows x 2 models x 2 (noSF+SF) = 8 configs x 3 seeds =
# 24 tasks. MUST run after basque5000_ablation_eval_stageA.sh, which
# rewires row 8's retrieval fields (both noSF and SF variants) to each
# model's own stage-A MeanQ winner (scripts/rewire_basque5000_stage.py
# --stage B).

set -euo pipefail

CONFIGS=(
  configs/experiments/5014_llama31_8b_3shot_no_rag_extractive_guiasalud_dev.json
  configs/experiments/5015_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/5016_latxa_llama31_8b_3shot_no_rag_extractive_guiasalud_dev.json
  configs/experiments/5017_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_dev.json
  configs/experiments/9007_llama31_8b_3shot_no_rag_extractive_guiasalud_sf_dev.json
  configs/experiments/9008_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_sf_dev.json
  configs/experiments/9018_latxa_llama31_8b_3shot_no_rag_extractive_guiasalud_sf_dev.json
  configs/experiments/9019_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_sf_dev.json
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
