#!/usr/bin/env python
"""Pick each Spanish model variant's overall best ablation-grid
configuration (across all 11 rows, 0-10) and freeze it as the retrieval base
for that variant's reasoning-pipeline experiments.

The ablation grid (see scripts/rewire_spanish_ablation_stage.py for how each
stage's dependent rows get wired) produces, by the time all three stages
have run, a fully scored set of 11 rows per model variant (Qwen3.5-9B
no-think and think). Every row is generated with self-feedback enabled, so
each row yields two independently-evaluated readings from the same run: a
plain (initial-answer) reading and a self-feedback (revised-answer)
reading. This script re-runs the same MeanQ selection rule
(scripts/meanq.py's best_by_meanq_robust) one final time across the
complete set of 22 candidates (11 rows x 2 readings) to pick each variant's
single best configuration, then writes reasoning-pipeline configs
(structured_cot, thought_rag, thought_rag_iter, marag, and a causal-scoring
variant of structured_cot) whose retrieval settings are frozen to that
winner. The reasoning pipelines are more expensive, multi-turn generation
strategies layered on top of a fixed retrieval configuration, so freezing
that configuration to the ablation grid's actual best result (rather than a
guess or a fixed default) is the point of this step.

Writes:
  - reports/metrics/mixed_meanq_selection_12000.json: the final winning row
    per variant (own file, kept separate from the shared
    mixed_meanq_selection.json used by earlier experiment rounds, so this
    run's selection never overwrites theirs).
  - Reasoning-pipeline config files at --base-id (default 14000, a range
    kept clear of the ablation grid's own 12000-12021 ids and the Basque
    track's 11000-11021/13000+ ids), 5 per variant.
  - reports/metrics/guiasalud_reasoning_configs_manifest_12000.txt: the list
    of reasoning-pipeline config paths just written, for the generation/
    evaluation scripts to read.

This script imports its config-writing logic (base_config_for,
retrieval_tag_for, PIPELINES) from scripts/create_mixed_reasoning_configs.py
rather than duplicating it, and reads model-family metadata from
scripts/mixed_meanq.py, but writes only to its own output files above, never
touching the shared selection/manifest files those modules' own callers use.

Usage:
    python scripts/finalize_spanish_ablation_and_write_reasoning_configs.py --dry-run
    python scripts/finalize_spanish_ablation_and_write_reasoning_configs.py --base-id 14000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from mixed_meanq import CELLS  # noqa: E402
from meanq import best_by_meanq_robust  # noqa: E402
from create_mixed_reasoning_configs import (  # noqa: E402
    PIPELINES, QWEN_ENGINE_FIELDS, base_config_for, retrieval_tag_for,
)

CONFIG_DIR = ROOT / "configs" / "experiments"
METRICS = ROOT / "reports" / "metrics"
SELECTION_PATH = METRICS / "mixed_meanq_selection_12000.json"
MANIFEST_PATH = METRICS / "guiasalud_reasoning_configs_manifest_12000.txt"

NON_RETRIEVING_ROWS = {0, 7}
DEFAULT_BASE_ID = 14000

# (family key, model tag, name template, start id for rows 0-6, explicit ids
# for rows 7/8/9/10, id offset used when writing reasoning-pipeline configs,
# engine-specific sampling fields)
FAMILIES = [
    ("qwen35_9b_no_think", "qwen35_9b_no_think",
     "qwen35_9b_{cell}_no_think_extractive_guiasalud_dev", 12000,
     {7: 12007, 8: 12008, 9: 12009, 10: 12010}, 0, QWEN_ENGINE_FIELDS),
    ("qwen35_9b_think", "qwen35_9b_think",
     "qwen35_9b_{cell}_think_extractive_guiasalud_dev", 12011,
     {7: 12018, 8: 12019, 9: 12020, 10: 12021}, 100, QWEN_ENGINE_FIELDS),
]


def candidates_all_rows(family_key: str, name_template: str, start_id: int, explicit_ids: dict) -> dict:
    """Build the MeanQ candidate pool covering all 11 rows: each row
    contributes both a plain candidate (its initial answer) and a
    "(SF)"-suffixed self-feedback candidate (its revised answer), read from
    the SAME run's before_feedback/after_feedback metric blocks, so the
    final winner can legitimately be either reading of any row."""
    candidates = {}
    for row in range(0, len(CELLS)):
        cell_slug, cell_label, _ = CELLS[row]
        run_id = start_id + row if row <= 6 else explicit_ids[row]
        base = name_template.format(cell=cell_slug)
        candidates[cell_label] = (str(run_id), base)
        candidates[f"{cell_label} (SF)"] = (str(run_id), base, True)
    return candidates


def retrieving_pool(candidates: dict) -> dict:
    """Drop rows that never retrieve anything (the no-retrieval baseline and
    the few-shot-no-retrieval row): the reasoning pipelines all use
    retrieval, so a configuration with no retrieval settings cannot serve as
    their base. The self-feedback suffix is stripped before checking
    exclusion, so a row's SF reading is excluded exactly when its plain
    reading would be."""
    excluded = {CELLS[row][1] for row in range(0, len(CELLS)) if row in NON_RETRIEVING_ROWS}

    def base_label(label: str) -> str:
        return label[: -len(" (SF)")] if label.endswith(" (SF)") else label

    return {k: v for k, v in candidates.items() if base_label(k) not in excluded}


def config_path(prefix: str, base: str) -> Path:
    return CONFIG_DIR / f"{prefix}_{base}.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-id", type=int, default=DEFAULT_BASE_ID)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    selection = {}
    for family_key, model_tag, name_template, start_id, explicit_ids, id_offset, engine_fields in FAMILIES:
        print(f"=== {family_key} (final selection across all 11 rows) ===")
        candidates = candidates_all_rows(family_key, name_template, start_id, explicit_ids)
        pool = retrieving_pool(candidates)

        winner, stats = best_by_meanq_robust(pool)
        if winner is None:
            print(f"  SKIP {family_key}: no metrics for any retrieving candidate yet")
            continue

        winner_prefix, winner_base = pool[winner][0], pool[winner][1]
        print(f"  winner: {winner} (id {winner_prefix}, MeanQ {stats[winner]['mean']:.2f})")

        winning_config_path = config_path(winner_prefix, winner_base)
        winning_config = json.loads(winning_config_path.read_text(encoding="utf-8"))
        selection[family_key] = {
            "model_label": model_tag,
            "winning_cell": winner,
            "run_id": f"{winner_prefix}_{winner_base}",
            "config_path": f"configs/experiments/{winner_prefix}_{winner_base}.json",
            "meanq_mean": stats[winner]["mean"],
            "meanq_std": stats[winner]["std"],
            "n_seeds": stats[winner]["n"],
            "retrieval_index": winning_config.get("retrieval_index", ""),
            "retrieval_top_k": winning_config.get("retrieval_top_k", 0),
            "reranker_model": winning_config.get("reranker_model", ""),
            "reranker_top_k": winning_config.get("reranker_top_k", 0),
            "few_shot_k": winning_config.get("few_shot_k", 0),
        }
        print()

    if args.dry_run:
        print("Dry run: selection computed, no files written.")
        print(json.dumps(selection, indent=2, ensure_ascii=False))
        return

    if not selection:
        print("No family resolved a winner -- nothing written.")
        return

    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    SELECTION_PATH.write_text(json.dumps(selection, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Written: {SELECTION_PATH}")

    written = []
    base_id = args.base_id
    for family_key, model_tag, name_template, start_id, explicit_ids, id_offset, engine_fields in FAMILIES:
        winner = selection.get(family_key)
        if not winner:
            continue

        base = base_config_for(winner, engine_fields)
        retrieval_tag = retrieval_tag_for(winner)
        baseline_run = winner["run_id"]

        for offset, pipeline, suffix, overrides, drop_reranker in PIPELINES:
            config_id = base_id + id_offset + offset
            name_retrieval_tag = "causal" if drop_reranker else retrieval_tag
            name = f"{model_tag}_{suffix}_{name_retrieval_tag}_extractive_guiasalud_dev"
            run_dir = f"{config_id}_{name}"
            variant_base = dict(base)
            if drop_reranker:
                variant_base.pop("reranker_model", None)
                variant_base.pop("reranker_top_k", None)
            payload = {
                "experiment_name": name,
                "pipeline": pipeline,
                "baseline_run": baseline_run,
                "output": f"experiments/runs/{run_dir}/predictions.jsonl",
                **variant_base,
                **overrides,
            }
            path = CONFIG_DIR / f"{run_dir}.json"
            path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            written.append(path)
            print(f"Written: {path.name}  (baseline={baseline_run}, retrieval={retrieval_tag})")

    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(
        "\n".join(f"configs/experiments/{p.name}" for p in written) + ("\n" if written else ""),
        encoding="utf-8",
    )
    print(f"\n{len(written)} reasoning-pipeline config(s) written. Manifest: {MANIFEST_PATH}")


if __name__ == "__main__":
    main()
