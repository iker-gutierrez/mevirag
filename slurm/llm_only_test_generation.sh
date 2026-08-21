#!/bin/bash
#SBATCH --job-name=llm-only-test
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llm_only_test_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/llm_only_test_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Four LLM-only baselines x three seeds. %2 permits exactly two one-GPU vLLM
# tasks at a time.
set -euo pipefail
mapfile -t TASKS < experiments/llm_only_test_seeded_tasks.txt
read -r STEM SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
  --config "configs/experiments/${STEM}.json" --seed "$SEED" \
  --output "experiments/runs/${STEM}_seed${SEED}/predictions.jsonl"
