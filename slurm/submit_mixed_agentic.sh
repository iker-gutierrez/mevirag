#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

dependency_args=()
if [[ $# -ge 1 && -n "${1:-}" ]]; then
  dependency_args=(--dependency="$1")
fi

agentic_job_id="$(sbatch --parsable "${dependency_args[@]}" slurm/mixed_agentic_dev.sh)"
echo "Submitted mixed agentic dev array: ${agentic_job_id}"

summary_job_id="$(sbatch --parsable --dependency=afterok:"${agentic_job_id}" slurm/mixed_agentic_summary.sh)"
echo "Submitted dependent mixed agentic summary: ${summary_job_id}"
