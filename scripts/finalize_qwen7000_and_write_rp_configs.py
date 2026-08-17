#!/usr/bin/env python
"""Final ALL_ROWS (0-10) MeanQ selection + reasoning-pipeline config write for
the fresh 7000-series Qwen3.5-9B GuiaSalud rerun, standalone rather than
touching scripts/guiasalud_meanq.py / scripts/create_guiasalud_reasoning_
configs.py's own FAMILIES (hardcoded to start_id 3281/3290, shared with the
Basque/Llama/Latxa path, which must not be disturbed).

Mirrors create_guiasalud_reasoning_configs.py's own logic (base_config_for,
retrieval_tag_for, PIPELINES) via direct import, so a future change to the
common RP-config shape (sampling fields, pipeline overrides) automatically
applies here too, rather than drifting from a second hardcoded copy.

Writes:
  - reports/metrics/guiasalud_meanq_selection_7000.json (final rows-0-10
    winner per variant, own file so it never collides with/overwrites the
    real guiasalud_meanq_selection.json)
  - 10 RP configs at --base-id (default 8000, clear of the 5000-5021/6000-
    6204/7000-7021 ranges already in use), 5 per variant (5 pipelines):
    structured_cot/thought_rag/thought_rag_iter/marag frozen to the true
    rows-0-10 winner (configs 11/13/14/15 per the staged-ablation
    convention), structured_cot-causal (config "12") left independent of
    the winner (drop_reranker=True, matches create_guiasalud_reasoning_
    configs.py's own PIPELINES entry unmodified).
  - reports/metrics/guiasalud_reasoning_configs_manifest_7000.txt (own
    manifest, does not touch the real
    guiasalud_reasoning_configs_manifest.txt)

Usage:
    python scripts/finalize_qwen7000_and_write_rp_configs.py --dry-run
    python scripts/finalize_qwen7000_and_write_rp_configs.py --base-id 8000
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from guiasalud_meanq import CELLS  # noqa: E402
from meanq import best_by_meanq_robust  # noqa: E402
from create_guiasalud_reasoning_configs import (  # noqa: E402
    PIPELINES, QWEN_ENGINE_FIELDS, base_config_for, retrieval_tag_for,
)

CONFIG_DIR = ROOT / "configs" / "experiments"
METRICS = ROOT / "reports" / "metrics"
SELECTION_PATH = METRICS / "guiasalud_meanq_selection_7000.json"
MANIFEST_PATH = METRICS / "guiasalud_reasoning_configs_manifest_7000.txt"

NON_RETRIEVING_ROWS = {0, 7}
DEFAULT_BASE_ID = 8000

# (family key, model tag, name template, start id, explicit ids for rows
# 7/8/9/10, id offset within base_id, engine fields)
FAMILIES_7K = [
    ("qwen35_9b_no_think", "qwen35_9b_no_think",
     "qwen35_9b_{cell}_no_think_extractive_guiasalud_dev", 7000,
     {7: 7014, 8: 7015, 9: 7018, 10: 7019}, 0, QWEN_ENGINE_FIELDS),
    ("qwen35_9b_think", "qwen35_9b_think",
     "qwen35_9b_{cell}_think_extractive_guiasalud_dev", 7007,
     {7: 7016, 8: 7017, 9: 7020, 10: 7021}, 100, QWEN_ENGINE_FIELDS),
]

# SF-clone ids, row-ordered (row 0 first, row 10 last) -- see
# scripts/create_5000_7000_sf_configs.py, which wrote these in this exact
# order from the noSF ids above.
SF_ROW_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SF_START_ID = {
    "qwen35_9b_no_think": 10000,
    "qwen35_9b_think": 10011,
}


def sf_run_id(family_key: str, row: int) -> int:
    return SF_START_ID[family_key] + SF_ROW_ORDER.index(row)


def candidates_all_rows(family_key: str, name_template: str, start_id: int, explicit_ids: dict) -> dict:
    candidates = {}
    for row in range(0, len(CELLS)):
        cell_slug, cell_label, _ = CELLS[row]
        run_id = start_id + row if row <= 6 else explicit_ids[row]
        base = name_template.format(cell=cell_slug)
        candidates[cell_label] = (str(run_id), base)

        sf_id = sf_run_id(family_key, row)
        assert base.endswith("_dev"), f"expected base to end in _dev, got {base!r}"
        sf_base = base[: -len("_dev")] + "_sf_dev"
        candidates[f"{cell_label} (SF)"] = (str(sf_id), sf_base, True)
    return candidates


def retrieving_pool(candidates: dict) -> dict:
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
    for family_key, model_tag, name_template, start_id, explicit_ids, id_offset, engine_fields in FAMILIES_7K:
        print(f"=== {family_key} (ALL_ROWS 0-10 final selection) ===")
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
    for family_key, model_tag, name_template, start_id, explicit_ids, id_offset, engine_fields in FAMILIES_7K:
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
