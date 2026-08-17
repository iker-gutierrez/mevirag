#!/bin/bash
#SBATCH --job-name=gen-A-es-abl-7k
#SBATCH --array=0-83%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_gen_stageA_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_gen_stageA_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Spanish/Qwen ablation grid, STAGE A = rows 0-6 (no RAG, e5 top-1/3/5,
# rerank top-1/3/5) for both Qwen3.5-9B no-think and think: 7 rows x 2
# variants x 2 (noSF+SF) = 28 configs x 3 seeds = 84 tasks. Reruns the
# original Qwen grid (3281-3296) at fresh ids 7000-7013, cloned by a one-off
# script (not tracked) after adding dash bullets to format_question()'s
# query composite and prepare_sns1064.py's build_justification() composite,
# and fixing the guiasalud_272 "levopremazina" -> "levomepromazina"
# data-quality bug. SF clones (self_feedback=true) at 10000-10006/10011-
# 10017, written by scripts/create_5000_7000_sf_configs.py, included here
# so both noSF and SF predictions exist for the staged MeanQ selection to
# compare (see scripts/rewire_qwen7000_stage.py's own docstring). The
# original 3281-3296 configs/predictions are left on disk untouched
# (not overwritten, per standing no-overwrite convention) -- fresh ids only.

set -euo pipefail

CONFIGS=(
  configs/experiments/7000_qwen35_9b_no_rag_no_think_extractive_guiasalud_dev.json
  configs/experiments/7001_qwen35_9b_rag_e5_topk1_no_think_extractive_guiasalud_dev.json
  configs/experiments/7002_qwen35_9b_rag_e5_topk3_no_think_extractive_guiasalud_dev.json
  configs/experiments/7003_qwen35_9b_rag_e5_topk5_no_think_extractive_guiasalud_dev.json
  configs/experiments/7004_qwen35_9b_rag_e5_rerank1_no_think_extractive_guiasalud_dev.json
  configs/experiments/7005_qwen35_9b_rag_e5_rerank3_no_think_extractive_guiasalud_dev.json
  configs/experiments/7006_qwen35_9b_rag_e5_rerank5_no_think_extractive_guiasalud_dev.json
  configs/experiments/7007_qwen35_9b_no_rag_think_extractive_guiasalud_dev.json
  configs/experiments/7008_qwen35_9b_rag_e5_topk1_think_extractive_guiasalud_dev.json
  configs/experiments/7009_qwen35_9b_rag_e5_topk3_think_extractive_guiasalud_dev.json
  configs/experiments/7010_qwen35_9b_rag_e5_topk5_think_extractive_guiasalud_dev.json
  configs/experiments/7011_qwen35_9b_rag_e5_rerank1_think_extractive_guiasalud_dev.json
  configs/experiments/7012_qwen35_9b_rag_e5_rerank3_think_extractive_guiasalud_dev.json
  configs/experiments/7013_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_dev.json
  configs/experiments/10000_qwen35_9b_no_rag_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10001_qwen35_9b_rag_e5_topk1_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10002_qwen35_9b_rag_e5_topk3_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10003_qwen35_9b_rag_e5_topk5_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10004_qwen35_9b_rag_e5_rerank1_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10005_qwen35_9b_rag_e5_rerank3_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10006_qwen35_9b_rag_e5_rerank5_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10011_qwen35_9b_no_rag_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10012_qwen35_9b_rag_e5_topk1_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10013_qwen35_9b_rag_e5_topk3_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10014_qwen35_9b_rag_e5_topk5_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10015_qwen35_9b_rag_e5_rerank1_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10016_qwen35_9b_rag_e5_rerank3_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10017_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_sf_dev.json
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

echo "Spanish/Qwen ablation generation (stage A, 7000-series) started on $(hostname)"
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

echo "Spanish/Qwen ablation generation (stage A, 7000-series) finished at $(date)"
