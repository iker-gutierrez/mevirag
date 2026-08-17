#!/usr/bin/env python
"""Final ALL_ROWS (0-10) MeanQ selection + reasoning-pipeline config write for
the fresh 5000-series Llama/Latxa GuiaSalud rerun, standalone rather than
touching scripts/mixed_meanq.py / scripts/create_mixed_reasoning_
configs.py's own FAMILIES (hardcoded to start_id 3299/3308, shared with the
Spanish/Qwen path, which must not be disturbed).

Mirrors create_mixed_reasoning_configs.py's own logic (base_config_for,
retrieval_tag_for, PIPELINES) via direct import, so a future change to the
common RP-config shape (sampling fields, pipeline overrides) automatically
applies here too, rather than drifting from a second hardcoded copy.

Writes:
  - reports/metrics/mixed_meanq_selection_5000.json (final rows-0-10
    winner per family, own file so it never collides with/overwrites the
    real mixed_meanq_selection.json)
  - 20 RP configs at --base-id (default 6000, clear of the 5000-5021 range
    used by the ablation grid rerun itself), 10 per model (5 pipelines):
    structured_cot/thought_rag/thought_rag_iter/marag frozen to the true
    rows-0-10 winner (configs 11/13/14/15 per the staged-ablation
    convention), structured_cot-causal (config "12") left independent of
    the winner (drop_reranker=True, matches create_mixed_reasoning_
    configs.py's own PIPELINES entry unmodified).
  - reports/metrics/guiasalud_reasoning_configs_manifest_5000.json (own
    manifest, does not touch the real
    guiasalud_reasoning_configs_manifest.txt)

Usage:
    python scripts/finalize_basque5000_and_write_rp_configs.py --dry-run
    python scripts/finalize_basque5000_and_write_rp_configs.py --base-id 6000
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
    PIPELINES, LLAMA_ENGINE_FIELDS, base_config_for, retrieval_tag_for,
)

CONFIG_DIR = ROOT / "configs" / "experiments"
METRICS = ROOT / "reports" / "metrics"
SELECTION_PATH = METRICS / "mixed_meanq_selection_5000.json"
MANIFEST_PATH = METRICS / "guiasalud_reasoning_configs_manifest_5000.txt"

NON_RETRIEVING_ROWS = {0, 7}
DEFAULT_BASE_ID = 6000

# (family key, model tag, id blocks: rows 0-6, rows 7-8, rows 9-10)
FAMILIES_5K = [
    ("llama31_8b", "llama31_8b",
     "llama31_8b_{cell}_extractive_guiasalud_dev", 5000,
     {7: 5014, 8: 5015, 9: 5018, 10: 5019}, 0, LLAMA_ENGINE_FIELDS),
    ("latxa_llama31_8b", "latxa_llama31_8b",
     "latxa_llama31_8b_{cell}_extractive_guiasalud_dev", 5007,
     {7: 5016, 8: 5017, 9: 5020, 10: 5021}, 200, LLAMA_ENGINE_FIELDS),
]

# SF-clone ids, row-ordered (row 0 first, row 10 last) -- see
# scripts/create_5000_7000_sf_configs.py, which wrote these in this exact
# order from the noSF ids above.
SF_ROW_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SF_START_ID = {
    "llama31_8b": 9000,
    "latxa_llama31_8b": 9011,
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
    for family_key, model_tag, name_template, start_id, explicit_ids, id_offset, engine_fields in FAMILIES_5K:
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
    for family_key, model_tag, name_template, start_id, explicit_ids, id_offset, engine_fields in FAMILIES_5K:
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
