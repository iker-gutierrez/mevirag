#!/bin/bash
#SBATCH --job-name=eval-C-eu-abl
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_evaluation_stageC_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque_ablation_evaluation_stageC_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# Scores every stage-C prediction (rows 9-10, both plain and self-feedback
# generation variants, for both Basque models), closing out the Basque
# ablation grid: all 11 rows are now scored for both models, so this script
# also runs the final selection across the complete grid and writes the
# reasoning-pipeline configuration files, frozen to whichever configuration
# (across all 11 rows, plain or self-feedback) actually scored best for each
# model.

SEEDS=(42 43 44)
RUN_IDS=(
  5018_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_dev
  5019_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_dev
  5020_latxa_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_dev
  5021_latxa_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_dev
  9009_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_sf_dev
  9010_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_sf_dev
  9020_latxa_llama31_8b_rag_domain_guiasalud_e5_rerank5_extractive_guiasalud_sf_dev
  9021_latxa_llama31_8b_rag_domain_casimedicos_e5_rerank5_extractive_guiasalud_sf_dev
)

echo "Basque ablation evaluation (stage C) started on $(hostname) at $(date)"

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

# Final selection across all 11 rows + reasoning-pipeline config write, one
# set of results per Basque model, kept in this run's own output files
# (mixed_meanq_selection_5000.json, guiasalud_reasoning_configs_manifest_
# 5000.txt) rather than the shared files other experiment rounds use.
python scripts/finalize_basque_ablation_and_write_reasoning_configs.py --base-id 6000

echo "Basque ablation evaluation (stage C) finished at $(date)"
