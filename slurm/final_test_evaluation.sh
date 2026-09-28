#!/bin/bash
#SBATCH --job-name=eval-final-test
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --output=experiments/slurm_logs/eval_final_test_%j.log
#SBATCH --error=experiments/slurm_logs/eval_final_test_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
STEMS=(
  17000_qwen35_9b_no_think_marag_e5_topk5_extractive_guiasalud_final_test_costaware
  17001_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_final_test
  17002_llama31_8b_rag_e5_rerank3_extractive_guiasalud_final_test
  17003_latxa_llama31_8b_rag_e5_topk1_extractive_guiasalud_final_test
)
for STEM in "${STEMS[@]}"; do
 for SEED in 42 43 44; do
  RUN="${STEM}_seed${SEED}"
  LANG=es
  [[ "$STEM" == 17002_* || "$STEM" == 17003_* ]] && LANG=eu
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
