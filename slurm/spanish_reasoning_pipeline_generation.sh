#!/bin/bash
#SBATCH --job-name=gen-rp-es
#SBATCH --array=0-29%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/spanish_reasoning_pipeline_generation_%A_%a.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/spanish_reasoning_pipeline_generation_%A_%a.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Runs the reasoning-pipeline generation configs for both Qwen3.5-9B
# variants: 5 pipeline variants (structured_cot, thought_rag,
# thought_rag_iter, marag, and a causal-scoring variant of structured_cot)
# x 2 variants (no-think/think) = 10 configurations x 3 seeds = 30 tasks.
# The configs themselves are written by
# scripts/finalize_spanish_ablation_and_write_reasoning_configs.py, run
# automatically at the end of spanish_ablation_evaluation_stageC.sh, with
# each configuration's retrieval settings frozen to that variant's own true
# best-scoring ablation-grid row. Must run after
# spanish_ablation_evaluation_stageC.sh has written the configuration
# manifest this script reads from.

set -euo pipefail

mapfile -t CONFIGS < <(python3 -c "
from pathlib import Path
manifest = Path('reports/metrics/guiasalud_reasoning_configs_manifest_12000.txt')
print(manifest.read_text().strip())
" | sed '/^$/d')

SEEDS=(42 43 44)

N_CFG=${#CONFIGS[@]}
CFG_IDX=$(( SLURM_ARRAY_TASK_ID / 3 ))
SEED_IDX=$(( SLURM_ARRAY_TASK_ID % 3 ))
if [ "$CFG_IDX" -ge "$N_CFG" ]; then
  echo "Task index $SLURM_ARRAY_TASK_ID has no corresponding config (only $N_CFG configs found); exiting cleanly."
  exit 0
fi
CONFIG="${CONFIGS[$CFG_IDX]}"
SEED="${SEEDS[$SEED_IDX]}"

source /home/igutierrez134/envs/med_rag_thesis/bin/activate

export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false

echo "Spanish reasoning-pipeline generation started on $(hostname) at $(date)"
echo "SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}  (found $N_CFG configs)"
echo "CONFIG=${CONFIG}  SEED=${SEED}"

scripts/pick_free_gpu.sh 40000 python scripts/run_reasoning_pipeline.py \
  --config "$CONFIG" --seed "$SEED"

echo "Spanish reasoning-pipeline generation finished at $(date)"
