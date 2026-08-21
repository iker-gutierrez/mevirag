#!/bin/bash
#SBATCH --job-name=eval-final-test
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=04:00:00
#SBATCH --mem=48GB
#SBATCH --output=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_final_test_%j.log
#SBATCH --error=/home/igutierrez134/med_rag_thesis/experiments/slurm_logs/eval_final_test_%j.err
#SBATCH --chdir=/home/igutierrez134/med_rag_thesis
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME=/home/igutierrez134/.cache/huggingface
export HF_HUB_CACHE=/home/igutierrez134/.cache/huggingface
export TOKENIZERS_PARALLELISM=false
while read -r STEM DRIVER SEED; do
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
done < experiments/final_test_seeded_tasks.txt
python3 scripts/patch_mc_accuracy.py
