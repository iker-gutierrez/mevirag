#!/bin/bash
# Submit the FULL GuiaSalud dev-experiment chain, both languages, end to end:
# retrieval indexing -> staged ablation (A -> B -> C, rows 0-10) -> final
# ALL_ROWS MeanQ selection -> reasoning-pipeline generation+evaluation, with
# every stage-to-stage decision made by the manuscript's own selection rule
# (sec:selection-rule / subsection 4.4.3: decisive-margin MeanQ, else
# stability+cost point-scoring, else mean tie-break -- scripts/meanq.py's
# best_by_meanq_robust, imported unmodified by every rewire/finalize script
# below, never reimplemented).
#
# Spanish (Qwen3.5-9B no-think/think) and Basque (Llama-3.1-8B-Instruct /
# Latxa-Llama-3.1-8B-Instruct) run as two independent, parallel chains, both
# sharing the account's 2-GPU cap via each Slurm array's own %2 throttle.
# Each chain is fully self-contained. Every stage's generation step includes
# BOTH the noSF config and its self_feedback=true clone for each row, so the
# MeanQ selection at the end of each stage is choosing from a pool where
# every row's noSF and SF variant both have real predictions to compare
# (the manuscript's own rule: the best config per stage is whichever variant
# -- noSF or SF -- of any row actually wins the decision algorithm, not
# necessarily the noSF one; see scripts/rewire_basque5000_stage.py /
# rewire_qwen7000_stage.py / finalize_*_and_write_rp_configs.py docstrings):
#
#   indices (parallel, no deps)
#     -> stage A gen (14 configs x 2 [noSF+SF] x 3 seeds = 84) -> stage A eval + rewire B
#     -> stage B gen (4 configs x 2 [noSF+SF] x 3 seeds = 24)  -> stage B eval + rewire C
#     -> stage C gen (4 configs x 2 [noSF+SF] x 3 seeds = 24)  -> stage C eval + final
#        ALL_ROWS (0-10, noSF+SF) MeanQ selection + reasoning-pipeline config write
#     -> reasoning-pipeline gen (10 configs x 3 seeds = 30)
#     -> reasoning-pipeline eval
#
# Every config/prediction is written at a dedicated id block (5000-5021 +
# 9000-9021 SF clones + 6000-6204 Basque, 7000-7021 + 10000-10021 SF clones
# + 8000-8104 Qwen), distinct from any prior round's ids, so nothing here
# ever overwrites an existing config or prediction file -- safe to run
# alongside, or after, earlier experiment rounds.
#
# The Basque/Qwen "5000-series"/"7000-series" naming (and the standalone
# rewire_basque5000_stage.py / rewire_qwen7000_stage.py / finalize_
# basque5000_and_write_rp_configs.py / finalize_qwen7000_and_write_rp_
# configs.py scripts this chain calls) exists because the ORIGINAL staged
# ablation (scripts/guiasalud_meanq.py, scripts/create_guiasalud_reasoning_
# configs.py, both hardcoded to start_id 3281/3290/3299/3308) is shared
# infrastructure also used by earlier rounds; a fresh, freestanding id block
# with its own rewire/finalize scripts avoids touching that shared state
# while reusing its exact same selection algorithm.
#
# This script only SUBMITS. It runs no GPU work itself.
#
# Usage: bash slurm/submit_guiasalud_dev_experiments.sh
#
# To check progress:  squeue -u $USER
# To watch a chain's outcome once finished: see reports/metrics/
#   guiasalud_meanq_selection_5000.json / _7000.json (final per-model/
#   per-variant winners) and reports/metrics/guiasalud_reasoning_configs_
#   manifest_5000.txt / _7000.txt (the reasoning-pipeline configs frozen to
#   those winners).

set -euo pipefail
cd /home/igutierrez134/med_rag_thesis

echo "Submitting the full GuiaSalud dev-experiment chain (Spanish + Basque, staged ablation + reasoning pipelines)."
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
# Spanish / Qwen3.5-9B chain (7000-series)
# ============================================================================
ES_GEN_A=$(sbatch --parsable --dependency=afterok:"${IDX_ES_MIXED}" \
  slurm/qwen7000_ablation_gen_stageA.sh)
echo "  [ES] stage A generation (rows 0-6, 42 tasks) : ${ES_GEN_A}  [after ${IDX_ES_MIXED}]"

ES_EVAL_A=$(sbatch --parsable --dependency=afterany:"${ES_GEN_A}" \
  slurm/qwen7000_ablation_eval_stageA.sh)
echo "  [ES] stage A eval + rewire to B              : ${ES_EVAL_A}  [after ${ES_GEN_A}]"

ES_GEN_B=$(sbatch --parsable --dependency=afterok:"${ES_EVAL_A}" \
  slurm/qwen7000_ablation_gen_stageB.sh)
echo "  [ES] stage B generation (rows 7-8, 12 tasks) : ${ES_GEN_B}  [after ${ES_EVAL_A}]"

ES_EVAL_B=$(sbatch --parsable --dependency=afterany:"${ES_GEN_B}" \
  slurm/qwen7000_ablation_eval_stageB.sh)
echo "  [ES] stage B eval + rewire to C               : ${ES_EVAL_B}  [after ${ES_GEN_B}]"

ES_GEN_C=$(sbatch --parsable \
  --dependency=afterok:"${ES_EVAL_B}":"${IDX_ES_GS}":"${IDX_ES_CM}" \
  slurm/qwen7000_ablation_gen_stageC.sh)
echo "  [ES] stage C generation (rows 9-10, 12 tasks): ${ES_GEN_C}  [after ${ES_EVAL_B}, ${IDX_ES_GS}, ${IDX_ES_CM}]"

ES_EVAL_C=$(sbatch --parsable --dependency=afterany:"${ES_GEN_C}" \
  slurm/qwen7000_ablation_eval_stageC.sh)
echo "  [ES] stage C eval + final ALL_ROWS selection : ${ES_EVAL_C}  [after ${ES_GEN_C}]"

ES_RP_GEN=$(sbatch --parsable --dependency=afterok:"${ES_EVAL_C}" \
  slurm/qwen7000_reasoning_gen.sh)
echo "  [ES] reasoning-pipeline generation (30 tasks): ${ES_RP_GEN}  [after ${ES_EVAL_C}]"

ES_RP_EVAL=$(sbatch --parsable --dependency=afterany:"${ES_RP_GEN}" \
  slurm/qwen7000_reasoning_eval.sh)
echo "  [ES] reasoning-pipeline evaluation            : ${ES_RP_EVAL}  [after ${ES_RP_GEN}]"
echo

# ============================================================================
# Basque / Llama-3.1-8B-Instruct + Latxa-Llama-3.1-8B-Instruct chain
# (5000-series)
# ============================================================================
EU_GEN_A=$(sbatch --parsable --dependency=afterok:"${IDX_EU_MIXED}" \
  slurm/basque5000_ablation_gen_stageA.sh)
echo "  [EU] stage A generation (rows 0-6, 42 tasks) : ${EU_GEN_A}  [after ${IDX_EU_MIXED}]"

EU_EVAL_A=$(sbatch --parsable --dependency=afterany:"${EU_GEN_A}" \
  slurm/basque5000_ablation_eval_stageA.sh)
echo "  [EU] stage A eval + rewire to B              : ${EU_EVAL_A}  [after ${EU_GEN_A}]"

EU_GEN_B=$(sbatch --parsable --dependency=afterok:"${EU_EVAL_A}" \
  slurm/basque5000_ablation_gen_stageB.sh)
echo "  [EU] stage B generation (rows 7-8, 12 tasks) : ${EU_GEN_B}  [after ${EU_EVAL_A}]"

EU_EVAL_B=$(sbatch --parsable --dependency=afterany:"${EU_GEN_B}" \
  slurm/basque5000_ablation_eval_stageB.sh)
echo "  [EU] stage B eval + rewire to C               : ${EU_EVAL_B}  [after ${EU_GEN_B}]"

EU_GEN_C=$(sbatch --parsable \
  --dependency=afterok:"${EU_EVAL_B}":"${IDX_EU_GS}":"${IDX_EU_CM}" \
  slurm/basque5000_ablation_gen_stageC.sh)
echo "  [EU] stage C generation (rows 9-10, 12 tasks): ${EU_GEN_C}  [after ${EU_EVAL_B}, ${IDX_EU_GS}, ${IDX_EU_CM}]"

EU_EVAL_C=$(sbatch --parsable --dependency=afterany:"${EU_GEN_C}" \
  slurm/basque5000_ablation_eval_stageC.sh)
echo "  [EU] stage C eval + final ALL_ROWS selection : ${EU_EVAL_C}  [after ${EU_GEN_C}]"

EU_RP_GEN=$(sbatch --parsable --dependency=afterok:"${EU_EVAL_C}" \
  slurm/basque5000_reasoning_gen.sh)
echo "  [EU] reasoning-pipeline generation (30 tasks): ${EU_RP_GEN}  [after ${EU_EVAL_C}]"

EU_RP_EVAL=$(sbatch --parsable --dependency=afterany:"${EU_RP_GEN}" \
  slurm/basque5000_reasoning_eval.sh)
echo "  [EU] reasoning-pipeline evaluation            : ${EU_RP_EVAL}  [after ${EU_RP_GEN}]"
echo

echo "All jobs submitted (both chains run in parallel, sharing the account's 2-GPU cap)."
echo "Watch with: squeue -u \$USER"
