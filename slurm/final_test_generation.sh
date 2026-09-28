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
STEMS=(
  17000_qwen35_9b_no_think_marag_e5_topk5_extractive_guiasalud_final_test_costaware
  17001_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_final_test
  17002_llama31_8b_rag_e5_rerank3_extractive_guiasalud_final_test
  17003_latxa_llama31_8b_rag_e5_topk1_extractive_guiasalud_final_test
)
DRIVERS=(reasoning generation generation generation)
SEEDS=(42 43 44)
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
STEM="${STEMS[$MODEL_IDX]}"
DRIVER="${DRIVERS[$MODEL_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
if [[ "$DRIVER" == "reasoning" ]]; then
  # The RP runner appends _seedN to config.output itself.
  scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
    --config "reproducibility/final_test/configs/${STEM}.json" --seed "$SEED"
elif [[ "$DRIVER" == "generation" ]]; then
  scripts/pick_free_gpu.sh 40000 python scripts/run_generation_from_config.py \
    --config "reproducibility/final_test/configs/${STEM}.json" --seed "$SEED" \
    --output "experiments/runs/${STEM}_seed${SEED}/predictions.jsonl"
else
  echo "Unknown driver '$DRIVER' for $STEM" >&2
  exit 2
fi
