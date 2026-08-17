#!/bin/bash
#SBATCH --job-name=eval-C-es-abl-7k
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=08:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_eval_stageC_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_eval_stageC_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

# STAGE C eval = ablation-grid rows 9-10 (domain restriction) for
# qwen35_9b_no_think and qwen35_9b_think (2 rows x 2 variants = 4 runs x 3
# seeds), 7000-series rerun. Closes the fresh Qwen ablation grid: rows 0-10
# are now scored for both variants, so the final ALL_ROWS selection + RP
# config write can run here, via the STANDALONE scripts/finalize_qwen7000_
# and_write_rp_configs.py (NOT the real scripts/guiasalud_meanq.py /
# create_guiasalud_reasoning_configs.py, which are hardcoded to the ORIGINAL
# 3281/3290 start_ids and would silently overwrite the real, shared
# guiasalud_meanq_selection.json / reasoning-configs manifest with data
# derived from a completely different id range if pointed at the 7000-series
# by mistake). Structural correctness of stage-to-stage winner propagation
# (including the row-7/non-retrieving-row exclusion) was verified with
# synthetic metrics via --dry-run before this rerun was ever submitted for
# real -- this reproducibility property (later readers must be able to
# regenerate the SAME winner-selection decisions, not have them silently
# hardcoded) is the entire point of the staged-rewiring machinery.

SEEDS=(42 43 44)
RUN_IDS=(
  7018_qwen35_9b_rag_domain_guiasalud_e5_rerank5_no_think_extractive_guiasalud_dev
  7019_qwen35_9b_rag_domain_casimedicos_e5_rerank5_no_think_extractive_guiasalud_dev
  7020_qwen35_9b_rag_domain_guiasalud_e5_rerank5_think_extractive_guiasalud_dev
  7021_qwen35_9b_rag_domain_casimedicos_e5_rerank5_think_extractive_guiasalud_dev
)

echo "Spanish/Qwen ablation evaluation (stage C, 7000-series) started on $(hostname) at $(date)"

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

# Whole-grid post-processing, 7000-series only: final ALL_ROWS (0-10) MeanQ
# selection + reasoning-pipeline config write for qwen35_9b_no_think/think,
# to their OWN files (guiasalud_meanq_selection_7000.json,
# guiasalud_reasoning_configs_manifest_7000.txt), never touching the real
# shared ones.
python scripts/finalize_qwen7000_and_write_rp_configs.py --base-id 8000

echo "Spanish/Qwen ablation evaluation (stage C, 7000-series) finished at $(date)"
