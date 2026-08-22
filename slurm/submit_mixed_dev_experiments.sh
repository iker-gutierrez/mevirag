#!/bin/bash
# Submits the full mixed-dataset dev-experiment chain, for both language tracks,
# end to end: retrieval indexing -> staged ablation grid (stage A -> B -> C,
# 11 configuration rows) -> final selection across the complete grid ->
# reasoning-pipeline generation and evaluation.
#
# Every stage-to-stage decision (which configuration a later, dependent row
# should be built on) and the final overall winner are chosen by the same
# rule: scripts/meanq.py's best_by_meanq_robust, which compares candidates
# on mean quality, stability across seeds, and computational cost, imported
# unmodified by every rewire/finalize script this chain calls, never
# reimplemented. Every ablation-grid row has self-feedback enabled: one
# generation run per row produces both an initial answer and a
# self-feedback revision of it, each scored as its own, independently
# evaluated candidate, so the winning configuration for a stage may turn
# out to be either reading.
#
# Spanish (Qwen3.5-9B, no-think and think variants) and Basque
# (Llama-3.1-8B-Instruct and Latxa-Llama-3.1-8B-Instruct) run as two
# independent, parallel chains, both sharing the compute account's 2-GPU cap
# via each Slurm array's own %2 throttle. Each chain:
#
#   indices (parallel, no dependency between them)
#     -> stage A generation (14 configs x 3 seeds = 42 tasks) -> stage A evaluation + rewire stage B
#     -> stage B generation (4 configs x 3 seeds = 12 tasks)  -> stage B evaluation + rewire stage C
#     -> stage C generation (4 configs x 3 seeds = 12 tasks)  -> stage C evaluation + final selection
#        across all 11 rows + reasoning-pipeline configuration write
#     -> reasoning-pipeline generation (10 configs x 3 seeds = 30 tasks)
#     -> reasoning-pipeline evaluation
#
# Every configuration and prediction file this chain writes lives in a
# dedicated id range, kept clear of any earlier experiment round's ids, so
# nothing here ever overwrites an existing configuration or its generated
# predictions — it is safe to run alongside, or after, earlier rounds.
#
# This script only submits jobs; it runs no GPU work itself.
#
# Usage: bash slurm/submit_mixed_dev_experiments.sh
#
# To check progress: squeue -u $USER
# To see a chain's final result once finished: reports/metrics/
#   mixed_meanq_selection_11000.json (Basque) / _12000.json (Spanish) hold
#   each model's winning configuration, and
#   reports/metrics/guiasalud_reasoning_configs_manifest_11000.txt / _12000.txt
#   list the reasoning-pipeline configs frozen to those winners.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "Submitting the full mixed-dataset dev-experiment chain (Spanish + Basque, staged ablation + reasoning pipelines)."
echo

# ============================================================================
# Retrieval indices (both languages, parallel, no deps on each other)
# ============================================================================
IDX_ES_MIXED=$(sbatch --parsable slurm/guiasalud_casimedicos_e5_indexing.sh)
echo "  index ES mixed (GuiaSalud+CasiMedicos)      : ${IDX_ES_MIXED}"
IDX_ES_GS=$(sbatch --parsable slurm/guiasalud_only_e5_indexing.sh)
echo "  index ES GuiaSalud-only (row 9)              : ${IDX_ES_GS}"
IDX_ES_CM=$(sbatch --parsable slurm/casimedicos_only_e5_indexing.sh)
echo "  index ES CasiMedicos-only (row 10)           : ${IDX_ES_CM}"

IDX_EU_MIXED=$(sbatch --parsable slurm/guiasalud_casimedicos_eu_e5_indexing.sh)
echo "  index EU mixed (GuiaSalud+CasiMedicos)       : ${IDX_EU_MIXED}"
IDX_EU_GS=$(sbatch --parsable slurm/guiasalud_eu_indexing.sh)
echo "  index EU GuiaSalud-only (row 9)               : ${IDX_EU_GS}"
IDX_EU_CM=$(sbatch --parsable slurm/casimedicos_only_eu_e5_indexing.sh)
echo "  index EU CasiMedicos-only (row 10)            : ${IDX_EU_CM}"
echo

# ============================================================================
# Spanish chain (Qwen3.5-9B, no-think and think variants)
# ============================================================================
ES_GEN_A=$(sbatch --parsable --dependency=afterok:"${IDX_ES_MIXED}" \
  slurm/spanish_ablation_generation_stageA.sh)
echo "  [ES] stage A generation (rows 0-6, 42 tasks) : ${ES_GEN_A}  [after ${IDX_ES_MIXED}]"

ES_EVAL_A=$(sbatch --parsable --dependency=afterany:"${ES_GEN_A}" \
  slurm/spanish_ablation_evaluation_stageA.sh)
echo "  [ES] stage A eval + rewire to B              : ${ES_EVAL_A}  [after ${ES_GEN_A}]"

ES_GEN_B=$(sbatch --parsable --dependency=afterok:"${ES_EVAL_A}" \
  slurm/spanish_ablation_generation_stageB.sh)
echo "  [ES] stage B generation (rows 7-8, 12 tasks) : ${ES_GEN_B}  [after ${ES_EVAL_A}]"

ES_EVAL_B=$(sbatch --parsable --dependency=afterany:"${ES_GEN_B}" \
  slurm/spanish_ablation_evaluation_stageB.sh)
echo "  [ES] stage B eval + rewire to C               : ${ES_EVAL_B}  [after ${ES_GEN_B}]"

ES_GEN_C=$(sbatch --parsable \
  --dependency=afterok:"${ES_EVAL_B}":"${IDX_ES_GS}":"${IDX_ES_CM}" \
  slurm/spanish_ablation_generation_stageC.sh)
echo "  [ES] stage C generation (rows 9-10, 12 tasks): ${ES_GEN_C}  [after ${ES_EVAL_B}, ${IDX_ES_GS}, ${IDX_ES_CM}]"

ES_EVAL_C=$(sbatch --parsable --dependency=afterany:"${ES_GEN_C}" \
  slurm/spanish_ablation_evaluation_stageC.sh)
echo "  [ES] stage C eval + final selection          : ${ES_EVAL_C}  [after ${ES_GEN_C}]"

ES_RP_GEN=$(sbatch --parsable --dependency=afterok:"${ES_EVAL_C}" \
  slurm/spanish_reasoning_pipeline_generation.sh)
echo "  [ES] reasoning-pipeline generation (30 tasks): ${ES_RP_GEN}  [after ${ES_EVAL_C}]"

ES_RP_EVAL=$(sbatch --parsable --dependency=afterany:"${ES_RP_GEN}" \
  slurm/spanish_reasoning_pipeline_evaluation.sh)
echo "  [ES] reasoning-pipeline evaluation            : ${ES_RP_EVAL}  [after ${ES_RP_GEN}]"
echo

# ============================================================================
# Basque chain (Llama-3.1-8B-Instruct and Latxa-Llama-3.1-8B-Instruct)
# ============================================================================
EU_GEN_A=$(sbatch --parsable --dependency=afterok:"${IDX_EU_MIXED}" \
  slurm/basque_ablation_generation_stageA.sh)
echo "  [EU] stage A generation (rows 0-6, 42 tasks) : ${EU_GEN_A}  [after ${IDX_EU_MIXED}]"

EU_EVAL_A=$(sbatch --parsable --dependency=afterany:"${EU_GEN_A}" \
  slurm/basque_ablation_evaluation_stageA.sh)
echo "  [EU] stage A eval + rewire to B              : ${EU_EVAL_A}  [after ${EU_GEN_A}]"

EU_GEN_B=$(sbatch --parsable --dependency=afterok:"${EU_EVAL_A}" \
  slurm/basque_ablation_generation_stageB.sh)
echo "  [EU] stage B generation (rows 7-8, 12 tasks) : ${EU_GEN_B}  [after ${EU_EVAL_A}]"

EU_EVAL_B=$(sbatch --parsable --dependency=afterany:"${EU_GEN_B}" \
  slurm/basque_ablation_evaluation_stageB.sh)
echo "  [EU] stage B eval + rewire to C               : ${EU_EVAL_B}  [after ${EU_GEN_B}]"

EU_GEN_C=$(sbatch --parsable \
  --dependency=afterok:"${EU_EVAL_B}":"${IDX_EU_GS}":"${IDX_EU_CM}" \
  slurm/basque_ablation_generation_stageC.sh)
echo "  [EU] stage C generation (rows 9-10, 12 tasks): ${EU_GEN_C}  [after ${EU_EVAL_B}, ${IDX_EU_GS}, ${IDX_EU_CM}]"

EU_EVAL_C=$(sbatch --parsable --dependency=afterany:"${EU_GEN_C}" \
  slurm/basque_ablation_evaluation_stageC.sh)
echo "  [EU] stage C eval + final selection          : ${EU_EVAL_C}  [after ${EU_GEN_C}]"

EU_RP_GEN=$(sbatch --parsable --dependency=afterok:"${EU_EVAL_C}" \
  slurm/basque_reasoning_pipeline_generation.sh)
echo "  [EU] reasoning-pipeline generation (30 tasks): ${EU_RP_GEN}  [after ${EU_EVAL_C}]"

EU_RP_EVAL=$(sbatch --parsable --dependency=afterany:"${EU_RP_GEN}" \
  slurm/basque_reasoning_pipeline_evaluation.sh)
echo "  [EU] reasoning-pipeline evaluation            : ${EU_RP_EVAL}  [after ${EU_RP_GEN}]"
echo

echo "All jobs submitted (both chains run in parallel, sharing the account's 2-GPU cap)."
echo "Watch with: squeue -u \$USER"
