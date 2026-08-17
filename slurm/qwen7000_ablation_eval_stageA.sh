#!/bin/bash
#SBATCH --job-name=eval-A-es-abl-7k
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=32GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_eval_stageA_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/qwen7000_ablation_eval_stageA_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

SEEDS=(42 43 44)
RUN_IDS=(
  7000_qwen35_9b_no_rag_no_think_extractive_guiasalud_dev
  7001_qwen35_9b_rag_e5_topk1_no_think_extractive_guiasalud_dev
  7002_qwen35_9b_rag_e5_topk3_no_think_extractive_guiasalud_dev
  7003_qwen35_9b_rag_e5_topk5_no_think_extractive_guiasalud_dev
  7004_qwen35_9b_rag_e5_rerank1_no_think_extractive_guiasalud_dev
  7005_qwen35_9b_rag_e5_rerank3_no_think_extractive_guiasalud_dev
  7006_qwen35_9b_rag_e5_rerank5_no_think_extractive_guiasalud_dev
  7007_qwen35_9b_no_rag_think_extractive_guiasalud_dev
  7008_qwen35_9b_rag_e5_topk1_think_extractive_guiasalud_dev
  7009_qwen35_9b_rag_e5_topk3_think_extractive_guiasalud_dev
  7010_qwen35_9b_rag_e5_topk5_think_extractive_guiasalud_dev
  7011_qwen35_9b_rag_e5_rerank1_think_extractive_guiasalud_dev
  7012_qwen35_9b_rag_e5_rerank3_think_extractive_guiasalud_dev
  7013_qwen35_9b_rag_e5_rerank5_think_extractive_guiasalud_dev
)

echo "Spanish/Qwen ablation evaluation (stage A, 7000-series) started on $(hostname) at $(date)"

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

# Stage A -> B: rewire row 8 to stage A's per-variant MeanQ winner.
python scripts/rewire_qwen7000_stage.py --stage B

echo "Spanish/Qwen ablation evaluation (stage A, 7000-series) finished at $(date)"
