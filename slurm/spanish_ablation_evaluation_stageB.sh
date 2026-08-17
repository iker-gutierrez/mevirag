#!/bin/bash
#SBATCH --job-name=eval-B-es-abl
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/spanish_ablation_evaluation_stageB_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/spanish_ablation_evaluation_stageB_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Scores every stage-B prediction (rows 7-8, both plain and self-feedback
# generation variants, for both Qwen3.5-9B variants), then rewires rows
# 9-10's retrieval settings to whichever configuration seen so far actually
# scored best.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

SEEDS=(42 43 44)
RUN_IDS=(
  7014_qwen35_9b_3shot_no_rag_no_think_extractive_guiasalud_dev
  7015_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_guiasalud_dev
  7016_qwen35_9b_3shot_no_rag_think_extractive_guiasalud_dev
  7017_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_guiasalud_dev
  10007_qwen35_9b_3shot_no_rag_no_think_extractive_guiasalud_sf_dev
  10008_qwen35_9b_rag_3shot_e5_rerank5_no_think_extractive_guiasalud_sf_dev
  10018_qwen35_9b_3shot_no_rag_think_extractive_guiasalud_sf_dev
  10019_qwen35_9b_rag_3shot_e5_rerank5_think_extractive_guiasalud_sf_dev
)

echo "Spanish ablation evaluation (stage B) started on $(hostname) at $(date)"

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

# Stage B -> C: rewire rows 9-10 to whichever configuration seen in stages
# A+B (plain or self-feedback) scored best, for each model variant.
python scripts/rewire_spanish_ablation_stage.py --stage C

echo "Spanish ablation evaluation (stage B) finished at $(date)"
