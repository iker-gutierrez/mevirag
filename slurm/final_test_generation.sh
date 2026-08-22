#!/bin/bash
#SBATCH --job-name=final-test-rag
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/slurm_logs/final_test_%A_%a.log
#SBATCH --error=experiments/slurm_logs/final_test_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Four RP-stage winners x three seeds. Each array task uses exactly one vLLM
# GPU; Slurm's %2 cap therefore permits exactly two concurrent GPUs.
set -euo pipefail
mapfile -t TASKS < experiments/final_test_seeded_tasks.txt
read -r STEM DRIVER SEED <<< "${TASKS[$SLURM_ARRAY_TASK_ID]}"
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
if [[ "$DRIVER" == "reasoning" ]]; then
  # The RP runner appends _seedN to config.output itself.
  scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
    --config "configs/experiments/${STEM}.json" --seed "$SEED"
elif [[ "$DRIVER" == "generation" ]]; then
  scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
    --config "configs/experiments/${STEM}.json" --seed "$SEED" \
    --output "experiments/runs/${STEM}_seed${SEED}/predictions.jsonl"
else
  echo "Unknown driver '$DRIVER' for $STEM" >&2
  exit 2
fi
