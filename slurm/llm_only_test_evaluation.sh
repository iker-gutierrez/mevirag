#!/bin/bash
#SBATCH --job-name=eval-llm-only-test
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/slurm_logs/eval_llm_only_test_%j.log
#SBATCH --error=experiments/slurm_logs/eval_llm_only_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
STEMS=(
  17100_qwen35_9b_no_rag_no_think_extractive_guiasalud_llm_only_final_test
  17101_qwen35_9b_no_rag_think_extractive_guiasalud_llm_only_final_test
  17102_llama31_8b_no_rag_extractive_guiasalud_llm_only_final_test
  17103_latxa_llama31_8b_no_rag_extractive_guiasalud_llm_only_final_test
)
for STEM in "${STEMS[@]}"; do
 for SEED in 42 43 44; do
  RUN="${STEM}_seed${SEED}"
  LANG=es
  [[ "$STEM" == 17102_* || "$STEM" == 17103_* ]] && LANG=eu
  if [[ ! -f "experiments/runs/${RUN}/predictions.jsonl" ]]; then
    echo "Skipping ${RUN}: no predictions (generation task failed or was not run)"
    continue
  fi
  python scripts/evaluate_predictions_by_source.py \
    --predictions "experiments/runs/${RUN}/predictions.jsonl" \
    --output "reports/metrics/${RUN}.json" \
    --semantic-model '' --bertscore-model bert-base-multilingual-cased --bertscore-lang "$LANG"
 done
done
python3 scripts/patch_mc_accuracy.py
