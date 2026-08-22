#!/bin/bash
#SBATCH --job-name=eval-B-eu-abl
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/slurm_logs/basque_ablation_evaluation_stageB_%j.log
#SBATCH --error=experiments/slurm_logs/basque_ablation_evaluation_stageB_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Scores every stage-B prediction (rows 7-8, for both Basque models). Every
# config has self-feedback enabled, so scoring one run yields both a plain
# and a self-feedback metric block. Then rewires rows 9-10's retrieval
# settings to whichever reading seen so far actually scored best.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

SEEDS=(42 43 44)
RUN_IDS=(
  11007_llama31_8b_3shot_no_rag_extractive_guiasalud_dev
  11008_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_dev
  11018_latxa_llama31_8b_3shot_no_rag_extractive_guiasalud_dev
  11019_latxa_llama31_8b_rag_3shot_e5_rerank5_extractive_guiasalud_dev
)

echo "Basque ablation evaluation (stage B) started on $(hostname) at $(date)"

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
      --bertscore-lang eu
  done
done

# Hit-rate@k summary (informational only, see scripts/aggregate_hit_rate.py's
# own docstring for why it never feeds into the selection rule below).
python scripts/aggregate_hit_rate.py \
  --run-ids "${RUN_IDS[@]}" \
  --seeds "${SEEDS[@]}" \
  --output reports/metrics/hit_rate_summary_11000_stageB.json

# Stage B -> C: rewire rows 9-10 to whichever reading seen in stages A+B
# (plain or self-feedback) actually scored best, for each model.
python scripts/rewire_basque_ablation_stage.py --stage C

echo "Basque ablation evaluation (stage B) finished at $(date)"
