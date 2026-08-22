#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

generation_job_id="$(sbatch --parsable slurm/qwen35_9b_spanish_generation.sh)"
echo "Submitted Qwen3.5-9B Spanish generation array: ${generation_job_id}"

evaluation_job_id="$(sbatch --parsable --dependency=afterok:${generation_job_id} slurm/qwen35_9b_spanish_evaluation.sh)"
echo "Submitted dependent Qwen3.5-9B Spanish evaluation: ${evaluation_job_id}"
