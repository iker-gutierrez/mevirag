#!/bin/bash
#SBATCH --job-name=llm-only-test
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/slurm_logs/llm_only_test_%A_%a.log
#SBATCH --error=experiments/slurm_logs/llm_only_test_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Four LLM-only baselines x three seeds. %2 permits exactly two one-GPU vLLM
# tasks at a time.
set -euo pipefail
STEMS=(
  17100_qwen35_9b_no_rag_no_think_extractive_guiasalud_llm_only_final_test
  17101_qwen35_9b_no_rag_think_extractive_guiasalud_llm_only_final_test
  17102_llama31_8b_no_rag_extractive_guiasalud_llm_only_final_test
  17103_latxa_llama31_8b_no_rag_extractive_guiasalud_llm_only_final_test
)
SEEDS=(42 43 44)
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
STEM="${STEMS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
  --config "reproducibility/final_test/configs/${STEM}.json" --seed "$SEED" \
  --output "experiments/runs/${STEM}_seed${SEED}/predictions.jsonl"
