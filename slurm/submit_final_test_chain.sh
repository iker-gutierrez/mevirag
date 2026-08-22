#!/bin/bash
# Build test configs from the already-finalized RP-stage selection, then submit
# generation and evaluation. This helper deliberately does not reselect a
# winner: ``rp_stage_selection.json`` is the frozen post-RP-evaluation record.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
python scripts/prepare_final_test_configs.py

GEN_JOB="$(sbatch --parsable slurm/final_test_generation.sh)"
EVAL_JOB="$(sbatch --parsable --dependency="afterany:${GEN_JOB}" slurm/final_test_evaluation.sh)"
printf 'Generation array: %s\nEvaluation: %s (afterany:%s)\n' "$GEN_JOB" "$EVAL_JOB" "$GEN_JOB"
