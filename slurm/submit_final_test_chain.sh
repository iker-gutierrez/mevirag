#!/bin/bash
# Submit generation and evaluation with the frozen configurations bundled under
# reproducibility/final_test/configs/. This helper does not reselect a winner.
set -euo pipefail

cd "$(dirname "$0")/.."
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
GEN_JOB="$(sbatch --parsable slurm/final_test_generation.sh)"
EVAL_JOB="$(sbatch --parsable --dependency="afterany:${GEN_JOB}" slurm/final_test_evaluation.sh)"
printf 'Generation array: %s\nEvaluation: %s (afterany:%s)\n' "$GEN_JOB" "$EVAL_JOB" "$GEN_JOB"
