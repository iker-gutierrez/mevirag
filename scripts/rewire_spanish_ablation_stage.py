#!/usr/bin/env python
"""Propagate the winning retrieval configuration between stages of the
Spanish ablation grid (Qwen3.5-9B, no-think and think variants, on the mixed
GuiaSalud+CasiMedicos-Exp dev set).

The ablation grid tests 11 configuration "rows" per model variant, numbered
0-10: row 0 is a no-retrieval baseline, rows 1-6 sweep dense-retrieval depth
and cross-encoder reranking depth, row 7 is a few-shot no-retrieval
baseline, row 8 combines few-shot prompting with the best retrieval setting
found in rows 1-6, and rows 9-10 restrict retrieval to a single source
corpus (GuiaSalud only / CasiMedicos-Exp only) using the best retrieval
setting found so far. Rows 8-10 are therefore *dependent* on an earlier
stage's result: their retrieval_top_k/reranker_model/reranker_top_k fields
must be copied from whichever earlier-row configuration scored highest, not
hardcoded, or the "staged" comparison stops being meaningful. This script
computes that winner and rewrites the dependent rows' config files to match
it, run between stages so each stage always starts from a config that
reflects the previous stage's actual result.

Each row is also generated twice: once as a plain, single-pass generation,
and once with an added self-feedback (SF) refinement step, where the model
critiques and revises its own first answer. Both variants are scored and
both are eligible to win a stage: the "best" configuration for a stage may
turn out to be a row's self-feedback variant rather than its plain one, and
the code here always compares both. See scripts/meanq.py's
best_by_meanq_robust for the actual "which configuration wins" rule
(a decisive-mean-difference check, falling back to a stability/cost
point-scored tie-break, falling back to a final mean comparison).

This script operates on a specific block of ablation-grid config files
(configs/experiments/7000-7021 for the plain runs, 10000-10021 for their
self-feedback clones) reserved for this rerun of the Spanish grid, kept
separate from any earlier round's ablation configs so that no earlier
round's config file or generated predictions are ever overwritten. It
therefore does not use the older shared selection module
(scripts/mixed_meanq.py), which is hardcoded to a different, older id block
and is left untouched.

Usage:
    python scripts/rewire_spanish_ablation_stage.py --stage B
    python scripts/rewire_spanish_ablation_stage.py --stage C
    python scripts/rewire_spanish_ablation_stage.py --stage C --dry-run
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

CONFIG_DIR = ROOT / "configs" / "experiments"

BASE_FIELDS = ("retrieval_top_k", "reranker_model", "reranker_top_k")
NON_RETRIEVING_ROWS = {0, 7}

# (family key, display label, name template, start id for rows 0-6).
# Rows 7-8 and 9-10 use separate explicit id blocks below, since ids
# 7000-7013 are already fully used by rows 0-6 for both model variants.
FAMILIES = [
    ("qwen35_9b_no_think", "Qwen3.5-9B (no-think)",
     "qwen35_9b_{cell}_no_think_extractive_guiasalud_dev", 7000),
    ("qwen35_9b_think", "Qwen3.5-9B (think)",
     "qwen35_9b_{cell}_think_extractive_guiasalud_dev", 7007),
]

# Explicit config ids for row 7 (few-shot baseline) and row 8 (few-shot +
# best retrieval), per model variant.
ROW_7_8_IDS = {
    "qwen35_9b_no_think": {7: 7014, 8: 7015},
    "qwen35_9b_think": {7: 7016, 8: 7017},
}

# Explicit config ids for rows 9-10 (single-corpus domain restriction).
ROW_9_10_IDS = {
    "qwen35_9b_no_think": {9: 7018, 10: 7019},
    "qwen35_9b_think": {9: 7020, 10: 7021},
}

# Config ids for the self-feedback clone of each row, in row order (row 0
# first, row 10 last). Each clone is identical to its plain counterpart
# except for enabling the self-feedback refinement step and its own output
# path; see scripts/create_self_feedback_ablation_configs.py, which
# generated these clone files from the plain ids above.
SF_ROW_ORDER = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
SF_START_ID = {
    "qwen35_9b_no_think": 10000,
    "qwen35_9b_think": 10011,
}

STAGES = {
    "B": {
        "pool_rows": range(0, 7),
        "pool_label": "stage A (rows 0-6)",
        "target_rows": [8],
        "target_label": "row 8 (few-shot + best retrieval)",
    },
    "C": {
        "pool_rows": range(0, 9),
        "pool_label": "stages A+B (rows 0-8)",
        "target_rows": [9, 10],
        "target_label": "rows 9-10 (single-corpus domain restriction)",
    },
}


def config_path(prefix: str, base: str) -> Path:
    return CONFIG_DIR / f"{prefix}_{base}.json"


def sf_run_id(family_key: str, row: int) -> int:
    return SF_START_ID[family_key] + SF_ROW_ORDER.index(row)


def candidates_for_pool(family_key: str, name_template: str, start_id: int, rows) -> dict:
    """Build the MeanQ candidate pool for a set of rows: each row contributes
    both its plain configuration and a "(SF)"-suffixed self-feedback
    variant, so the comparison can pick whichever actually scores higher."""
    candidates = {}
    for row in rows:
        cell_slug, cell_label, _ = CELLS[row]
        if row <= 6:
            run_id = start_id + row
        elif row in (7, 8):
            run_id = ROW_7_8_IDS[family_key][row]
        elif row in (9, 10):
            run_id = ROW_9_10_IDS[family_key][row]
        else:
            raise ValueError(f"unexpected row {row}")
        base = name_template.format(cell=cell_slug)
        candidates[cell_label] = (str(run_id), base)

        sf_id = sf_run_id(family_key, row)
        assert base.endswith("_dev"), f"expected base to end in _dev, got {base!r}"
        sf_base = base[: -len("_dev")] + "_sf_dev"
        candidates[f"{cell_label} (SF)"] = (str(sf_id), sf_base, True)
    return candidates


def retrieving_pool(candidates: dict, pool_rows) -> dict:
    """Drop rows that never retrieve anything (the no-retrieval baseline and
    the few-shot-no-retrieval row) from the candidate pool: a dependent row
    that restricts *which corpus* is retrieved from, or combines few-shot
    prompting with retrieval, cannot sensibly inherit settings from a
    configuration that has no retrieval settings to give it. The
    self-feedback suffix is stripped before checking exclusion, so a row's
    SF variant is excluded exactly when its plain variant would be."""
    excluded = {CELLS[row][1] for row in pool_rows if row in NON_RETRIEVING_ROWS}

    def base_label(label: str) -> str:
        return label[: -len(" (SF)")] if label.endswith(" (SF)") else label

    return {k: v for k, v in candidates.items() if base_label(k) not in excluded}


def base_fields_of(prefix: str, base: str) -> dict:
    cfg = json.loads(config_path(prefix, base).read_text(encoding="utf-8"))
    return {field: cfg.get(field) for field in BASE_FIELDS}


def apply_base(prefix: str, base: str, fields: dict, *, dry_run: bool) -> bool:
    path = config_path(prefix, base)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    changed = any(cfg.get(k) != v for k, v in fields.items())
    if not changed:
        print(f"    {path.name}: already matches, no change")
        return False
    print(f"    {path.name}: {({k: cfg.get(k) for k in fields})} -> {fields}")
    if not dry_run:
        cfg.update(fields)
        path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(STAGES))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    stage = STAGES[args.stage]
    any_changed = False

    for family_key, family_label, name_template, start_id in FAMILIES:
        print(f"=== {family_label} ({stage['pool_label']} -> {stage['target_label']}) ===")
        candidates = candidates_for_pool(family_key, name_template, start_id, stage["pool_rows"])
        pool = retrieving_pool(candidates, stage["pool_rows"])

        winner, stats = best_by_meanq_robust(pool)
        if winner is None:
            print(f"  SKIP {family_label}: no metrics for any retrieving candidate yet")
            continue

        winner_spec = pool[winner]
        winner_prefix, winner_base = winner_spec[0], winner_spec[1]
        print(f"  winner: {winner} (id {winner_prefix}, MeanQ {stats[winner]['mean']:.2f})")
        fields = base_fields_of(winner_prefix, winner_base)

        # Both the plain and self-feedback config of each dependent row are
        # rewritten to the winner's retrieval settings, so the next stage's
        # own candidate pool (which also compares both variants of every
        # row) has a correctly-based self-feedback candidate to consider.
        for target_row in stage["target_rows"]:
            cell_slug, cell_label, _ = CELLS[target_row]
            if target_row in (7, 8):
                target_id = ROW_7_8_IDS[family_key][target_row]
            else:
                target_id = ROW_9_10_IDS[family_key][target_row]
            target_base = name_template.format(cell=cell_slug)
            changed = apply_base(str(target_id), target_base, fields, dry_run=args.dry_run)
            any_changed = any_changed or changed

            target_sf_id = sf_run_id(family_key, target_row)
            target_sf_base = target_base[: -len("_dev")] + "_sf_dev"
            changed_sf = apply_base(str(target_sf_id), target_sf_base, fields, dry_run=args.dry_run)
            any_changed = any_changed or changed_sf
        print()

    if args.dry_run:
        print("Dry run: no files written.")
    elif not any_changed:
        print("No configs needed rewiring.")


if __name__ == "__main__":
    main()
