#!/bin/bash
#SBATCH --job-name=reeval-interaccionaban-es
#SBATCH --array=0-11%2
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/slurm_logs/reeval_interaccionaban_es_%A_%a.log
#SBATCH --error=experiments/slurm_logs/reeval_interaccionaban_es_%A_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Reevaluate existing Spanish final-test predictions against the corrected
# reference JSONL. This script never invokes generation or index building.
set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false

TASKS=(
  17000_qwen35_9b_no_think_marag_e5_topk5_extractive_guiasalud_final_test_costaware_seed42
  17000_qwen35_9b_no_think_marag_e5_topk5_extractive_guiasalud_final_test_costaware_seed43
  17000_qwen35_9b_no_think_marag_e5_topk5_extractive_guiasalud_final_test_costaware_seed44
  17001_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_final_test_seed42
  17001_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_final_test_seed43
  17001_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_final_test_seed44
  17100_qwen35_9b_no_rag_no_think_extractive_guiasalud_llm_only_final_test_seed42
  17100_qwen35_9b_no_rag_no_think_extractive_guiasalud_llm_only_final_test_seed43
  17100_qwen35_9b_no_rag_no_think_extractive_guiasalud_llm_only_final_test_seed44
  17101_qwen35_9b_no_rag_think_extractive_guiasalud_llm_only_final_test_seed42
  17101_qwen35_9b_no_rag_think_extractive_guiasalud_llm_only_final_test_seed43
  17101_qwen35_9b_no_rag_think_extractive_guiasalud_llm_only_final_test_seed44
)

RUN="${TASKS[$SLURM_ARRAY_TASK_ID]}"
mkdir -p reports/metrics/qwen35_9b_final_test_es
python scripts/evaluate_predictions_by_source.py \
  --predictions "experiments/runs/${RUN}/predictions.jsonl" \
  --output "reports/metrics/qwen35_9b_final_test_es/${RUN}.json" \
  --semantic-model '' \
  --bertscore-model bert-base-multilingual-cased \
  --bertscore-lang es \
  --bertscore-device cuda:0
