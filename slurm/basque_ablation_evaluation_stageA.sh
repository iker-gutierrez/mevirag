#!/bin/bash
#SBATCH --job-name=eval-A-eu-abl
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_evaluation_stageA_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_evaluation_stageA_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Scores every stage-A prediction (rows 0-6, for both Basque models). Every
# config has self-feedback enabled, so scoring one run yields both a plain
# (initial-answer) and a self-feedback (revised-answer) metric block; see
# scripts/evaluate_predictions.py's before_feedback/after_feedback split.
# Then rewires row 8's retrieval settings to whichever reading of whichever
# row actually scored best.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

SEEDS=(42 43 44)
RUN_IDS=(
  11000_llama31_8b_no_rag_extractive_guiasalud_dev
  11001_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev
  11002_llama31_8b_rag_e5_topk3_extractive_guiasalud_dev
  11003_llama31_8b_rag_e5_topk5_extractive_guiasalud_dev
  11004_llama31_8b_rag_e5_rerank1_extractive_guiasalud_dev
  11005_llama31_8b_rag_e5_rerank3_extractive_guiasalud_dev
  11006_llama31_8b_rag_e5_rerank5_extractive_guiasalud_dev
  11011_latxa_llama31_8b_no_rag_extractive_guiasalud_dev
  11012_latxa_llama31_8b_rag_e5_topk1_extractive_guiasalud_dev
  11013_latxa_llama31_8b_rag_e5_topk3_extractive_guiasalud_dev
  11014_latxa_llama31_8b_rag_e5_topk5_extractive_guiasalud_dev
  11015_latxa_llama31_8b_rag_e5_rerank1_extractive_guiasalud_dev
  11016_latxa_llama31_8b_rag_e5_rerank3_extractive_guiasalud_dev
  11017_latxa_llama31_8b_rag_e5_rerank5_extractive_guiasalud_dev
)

echo "Basque ablation evaluation (stage A) started on $(hostname) at $(date)"

# Pre-flight: confirm reference-enrichment resolves real gold text before
# spending time on the full eval sweep. Catches a broken/stale dataset path
# (e.g. a mid-flight dataset rebuild race) loudly instead of silently
# deflating every MeanQ number in this stage -- see
# scripts/check_reference_enrichment.py's own docstring for the incident
# that motivated this check.
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
      --bertscore-lang eu
  done
done

# Stage A -> B: rewire row 8 to whichever reading of whichever stage-A row
# (plain or self-feedback) actually scored best, for each model.
python scripts/rewire_basque_ablation_stage.py --stage B

echo "Basque ablation evaluation (stage A) finished at $(date)"
