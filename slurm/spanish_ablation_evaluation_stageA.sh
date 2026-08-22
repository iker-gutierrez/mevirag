#!/bin/bash
#SBATCH --job-name=eval-A-es-abl
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/slurm_logs/spanish_ablation_evaluation_stageA_%j.log
#SBATCH --error=experiments/slurm_logs/spanish_ablation_evaluation_stageA_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Scores every stage-A prediction (rows 0-6, for both Qwen3.5-9B variants).
# Every config has self-feedback enabled, so scoring one run yields both a
# plain (initial-answer) and a self-feedback (revised-answer) metric block;
# see scripts/evaluate_predictions.py's before_feedback/after_feedback
# split. Then rewires row 8's retrieval settings to whichever reading of
# whichever row actually scored best.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

SEEDS=(42 43 44)
RUN_IDS=(
  12000_qwen35_9b_no_rag_no_think_extractive_guiasalud_dev
  12001_qwen35_9b_rag_e5_topk1_no_think_extractive_guiasalud_dev
  12002_qwen35_9b_rag_e5_topk3_no_think_extractive_guiasalud_dev
  12003_qwen35_9b_rag_e5_topk5_no_think_extractive_guiasalud_dev
  12004_qwen35_9b_rag_e5_rerank1_no_think_extractive_guiasalud_dev
  12005_qwen35_9b_rag_e5_rerank3_no_think_extractive_guiasalud_dev
  12006_qwen35_9b_rag_e5_rerank5_no_think_extractive_guiasalud_dev
  12011_qwen35_9b_no_rag_think_extractive_guiasalud_dev
  12012_qwen35_9b_rag_e5_topk1_think_extractive_guiasalud_dev
  12013_qwen35_9b_rag_e5_topk3_think_extractive_guiasalud_dev
  12014_qwen35_9b_rag_e5_topk5_think_extractive_guiasalud_dev
  12015_qwen35_9b_rag_e5_rerank1_think_extractive_guiasalud_dev
  12016_qwen35_9b_rag_e5_rerank3_think_extractive_guiasalud_dev
  12017_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_dev
)

echo "Spanish ablation evaluation (stage A) started on $(hostname) at $(date)"

# Pre-flight: confirm reference-enrichment resolves real gold text before
# spending time on the full eval sweep (see scripts/check_reference_
# enrichment.py's own docstring for the incident that motivated this check).
FIRST_PRED="experiments/runs/${RUN_IDS[0]}_seed42/predictions.jsonl"
if [ -f "$FIRST_PRED" ]; then
  echo "Pre-flight: checking reference enrichment on ${FIRST_PRED}"
  python scripts/check_reference_enrichment.py "$FIRST_PRED"
fi

for run_id in "${RUN_IDS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run="${run_id}_seed${seed}"
    predictions="experiments/runs/${run}/predictions.jsonl"
    if [ ! -f "$predictions" ]; then
      echo "Skipping ${run}: no predictions at ${predictions}"
      continue
    fi
    echo "Evaluating ${run}"
    python scripts/evaluate_predictions_by_source.py \
      --predictions "$predictions" \
      --output "reports/metrics/${run}.json" \
      --semantic-model intfloat/multilingual-e5-large \
      --bertscore-model bert-base-multilingual-cased \
      --bertscore-lang es
  done
done

# Hit-rate@k summary (informational only, see scripts/aggregate_hit_rate.py's
# own docstring for why it never feeds into the selection rule below).
python scripts/aggregate_hit_rate.py \
  --run-ids "${RUN_IDS[@]}" \
  --seeds "${SEEDS[@]}" \
  --output reports/metrics/hit_rate_summary_12000_stageA.json

# Stage A -> B: rewire row 8 to whichever reading of whichever stage-A row
# (plain or self-feedback) actually scored best, for each model variant.
python scripts/rewire_spanish_ablation_stage.py --stage B

echo "Spanish ablation evaluation (stage A) finished at $(date)"
