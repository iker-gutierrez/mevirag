#!/bin/bash
#SBATCH --job-name=eval-rp-eu-5k
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=03:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque5000_reasoning_eval_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/basque5000_reasoning_eval_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

# Uses evaluate_predictions_by_source.py (subprocess-per-call, has a real
# --bertscore-lang flag), NOT evaluate_predictions_by_source_cached.py,
# which hardcodes BERTSCORE_LANG = "es" -- would silently score Basque
# predictions with the wrong-language BERTScore model otherwise.

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

mapfile -t CONFIGS < <(python3 -c "
from pathlib import Path
manifest = Path('reports/metrics/guiasalud_reasoning_configs_manifest_5000.txt')
print(manifest.read_text().strip())
" | sed '/^$/d')

SEEDS=(42 43 44)

FIRST_BASE=$(basename "${CONFIGS[0]}" .json)
FIRST_PRED="experiments/runs/${FIRST_BASE}_seed42/predictions.jsonl"
if [ -f "$FIRST_PRED" ]; then
  echo "Pre-flight: checking reference enrichment on ${FIRST_PRED}"
  python scripts/check_reference_enrichment.py "$FIRST_PRED"
fi

echo "Basque 5000-series reasoning-pipeline evaluation started on $(hostname) at $(date)"

for cfg_path in "${CONFIGS[@]}"; do
  base=$(basename "$cfg_path" .json)
  for seed in "${SEEDS[@]}"; do
    run="${base}_seed${seed}"
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

echo "Basque 5000-series reasoning-pipeline evaluation finished at $(date)"
