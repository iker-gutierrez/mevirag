#!/bin/bash
set -euo pipefail
cd /home/igutierrez134/med_rag_thesis
source /home/igutierrez134/envs/med_rag_thesis/bin/activate
python scripts/prepare_llm_only_test_configs.py
GEN_JOB="$(sbatch --parsable slurm/llm_only_test_generation.sh)"
EVAL_JOB="$(sbatch --parsable --dependency="afterany:${GEN_JOB}" slurm/llm_only_test_evaluation.sh)"
printf 'Generation array: %s\nEvaluation: %s (afterany:%s)\n' "$GEN_JOB" "$EVAL_JOB" "$GEN_JOB"
