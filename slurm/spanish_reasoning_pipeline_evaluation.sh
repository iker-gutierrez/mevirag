#!/bin/bash
#SBATCH --job-name=eval-rp-es
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=02:00:00
#SBATCH --mem=48GB
#SBATCH --gres=gpu:1
#SBATCH --output=experiments/slurm_logs/spanish_reasoning_pipeline_evaluation_%j.log
#SBATCH --error=experiments/slurm_logs/spanish_reasoning_pipeline_evaluation_%j.err
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --mail-user=igutierrez134@ikasle.ehu.eus

set -euo pipefail
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
export HF_HOME="/home/igutierrez134/.cache/huggingface"
export TRANSFORMERS_CACHE="/home/igutierrez134/.cache/huggingface"
export HF_HUB_CACHE="/home/igutierrez134/.cache/huggingface"
export TOKENIZERS_PARALLELISM=false

RUN_LIST="experiments/tmp_lists/spanish_reasoning_pipeline_runs.txt"
mkdir -p experiments/tmp_lists
python3 -c "
from pathlib import Path
manifest = Path('reports/metrics/guiasalud_reasoning_configs_manifest_12000.txt')
paths = [l.strip() for l in manifest.read_text().splitlines() if l.strip()]
bases = [Path(p).stem for p in paths]
seeds = [42, 43, 44]
lines = [f'{b}_seed{s}' for b in bases for s in seeds]
Path('$RUN_LIST').write_text(chr(10).join(lines))
print(len(lines), 'run-seeds')
"

FIRST_PRED="experiments/runs/$(head -1 "$RUN_LIST")/predictions.jsonl"
if [ -f "$FIRST_PRED" ]; then
  echo "Pre-flight: checking reference enrichment on ${FIRST_PRED}"
  python scripts/check_reference_enrichment.py "$FIRST_PRED"
fi

echo "Spanish reasoning-pipeline evaluation started on $(hostname) at $(date)"
python scripts/evaluate_predictions_by_source_cached.py "$RUN_LIST"
echo "Spanish reasoning-pipeline evaluation finished at $(date)"
