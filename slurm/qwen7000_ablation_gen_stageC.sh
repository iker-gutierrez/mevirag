#!/bin/bash
#SBATCH --job-name=gen-C-es-abl-7k
#SBATCH --array=0-23%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_gen_stageC_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_gen_stageC_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Spanish/Qwen ablation grid, STAGE C = rows 9-10 (domain restriction:
# GuiaSalud-only retrieval, CasiMedicos-only retrieval) for both variants:
# 2 rows x 2 variants x 2 (noSF+SF) = 8 configs x 3 seeds = 24 tasks.
# `input` stays the full mixed Spanish dev set
# (data/processed/guiasalud_casimedicos/dev.jsonl), only `retrieval_index`
# is restricted. MUST run after qwen7000_ablation_eval_stageB.sh, which
# rewires these configs' retrieval_top_k/reranker_model/reranker_top_k
# (both noSF and SF variants) to each variant's own stage-A+B MeanQ winner
# (scripts/rewire_qwen7000_stage.py --stage C). Needs the freshly-rebuilt
# guiasalud_only_multilingual_e5_large / casimedicos_only_multilingual_e5_
# large indices (jobs 13987/13988) to reflect today's bullet/levomepromazina
# fixes.

set -euo pipefail

CONFIGS=(
  configs/experiments/7018_qwen35_9b_rag_domain_guiasalud_e5_rerank5_no_think_extractive_guiasalud_dev.json
  configs/experiments/7019_qwen35_9b_rag_domain_casimedicos_e5_rerank5_no_think_extractive_guiasalud_dev.json
  configs/experiments/7020_qwen35_9b_rag_domain_guiasalud_e5_rerank5_think_extractive_guiasalud_dev.json
  configs/experiments/7021_qwen35_9b_rag_domain_casimedicos_e5_rerank5_think_extractive_guiasalud_dev.json
  configs/experiments/10009_qwen35_9b_rag_domain_guiasalud_e5_rerank5_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10010_qwen35_9b_rag_domain_casimedicos_e5_rerank5_no_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10020_qwen35_9b_rag_domain_guiasalud_e5_rerank5_think_extractive_guiasalud_sf_dev.json
  configs/experiments/10021_qwen35_9b_rag_domain_casimedicos_e5_rerank5_think_extractive_guiasalud_sf_dev.json
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

echo "Spanish/Qwen ablation generation (stage C, 7000-series) started on $(hostname)"
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

echo "Spanish/Qwen ablation generation (stage C, 7000-series) finished at $(date)"
